# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Segmented row reduction for the EC dedup-expand backward.

Replaces ``scatter_add_`` in ``_CastExpandFp32Bwd``. Each program owns one
unique row and one column block, gathers that row's segment in tiles, and
stores once. Segments longer than ``split_above`` finish with an atomic add.

The gradient is loaded as bf16 and upcast in registers. TorchRec's dedup does
not return a permutation, so this kernel argsorts ``reverse_indices`` itself.
The fp32 add order differs from ``scatter_add_``.
"""

import logging
import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

logger: logging.Logger = logging.getLogger(__name__)

# --- launch geometry --------------------------------------------------------
# Set by `configure_segment_reduce()` below, which `ec_bf16_output.
# enable_segment_reduce()` calls with whatever the gin bound there.  These are
# module state rather than import-time `os.environ` reads on purpose: an
# import-time constant is invisible to gin's operative config, so a run cannot
# report what it was tuned with.  See that function for what each one means.
_SPLIT_ABOVE: int = 1024
_BLOCK_S: int = 32
_NUM_WARPS: int = 2
_BLOCK_W_CAP: int = 512
_GPU_PLANNER: bool = False
_FIXED_SIZE_OFFSETS: bool = False
# DLRM_HSTU_FP8_EC_STRIDED_GRAD (default on; absent = off): read a row-strided
# gradient in place instead of materialising grad_out.contiguous() first.
# The kernels load the same elements in the same order. Cold segments match;
# hot pieces differ only by atomic-add order.
_STRIDED_GRAD: bool = os.environ.get("DLRM_HSTU_FP8_EC_STRIDED_GRAD", "0") != "0"


def configure_segment_reduce(
    split_above: int = 1024,
    block_s: int = 32,
    num_warps: int = 2,
    block_w_cap: int = 512,
    gpu_planner: bool = False,
    fixed_size_offsets: bool = False,
) -> None:
    """Set the segment-reduce kernel's launch geometry.

    Plain setter, not a gin configurable: this module is imported lazily, so
    it is not registered when gin parses. The gin surface is
    `ec_bf16_output.enable_segment_reduce`.

    Args:
      split_above: rows longer than this are cut into pieces that finish with
        an atomic add. 0 disables splitting.
      block_s: rows gathered per inner iteration. The tile is
        [block_s, block_w] fp32 in registers.
      num_warps: warps per program. The default is 2; a wider tile spills.
      block_w_cap: upper bound on the width tile. The actual block_w is
        min(next_pow2(W), block_w_cap).
      gpu_planner: construct hot-piece metadata on the current GPU stream.
        Default off; reduction geometry and fp32 arithmetic stay unchanged.
      fixed_size_offsets: use exact GPU lower bounds in the incumbent sorted
        IDs instead of bincount/cumsum. Requires valid IDs in [0, num_rows).
        Default off and independent of gpu_planner.
    """
    if type(gpu_planner) is not bool or type(fixed_size_offsets) is not bool:
        raise ValueError("gpu_planner and fixed_size_offsets must be booleans")
    global _SPLIT_ABOVE, _BLOCK_S, _NUM_WARPS, _BLOCK_W_CAP
    global _GPU_PLANNER, _FIXED_SIZE_OFFSETS
    _SPLIT_ABOVE, _BLOCK_S = split_above, block_s
    _NUM_WARPS, _BLOCK_W_CAP = num_warps, block_w_cap
    _GPU_PLANNER, _FIXED_SIZE_OFFSETS = gpu_planner, fixed_size_offsets
    logger.info(
        "DLRMV4 ec_segment_reduce geometry: SPLIT_ABOVE=%d BLOCK_S=%d "
        "num_warps=%d BLOCK_W_CAP=%d GPU_PLANNER=%s FIXED_SIZE_OFFSETS=%s",
        _SPLIT_ABOVE,
        _BLOCK_S,
        _NUM_WARPS,
        _BLOCK_W_CAP,
        _GPU_PLANNER,
        _FIXED_SIZE_OFFSETS,
    )


@triton.jit
def _seg_reduce_kernel(
    grad_ptr,          # [T, W]  incoming gradient, any float dtype
    order_ptr,         # [T]     row ids sorted by destination
    row_ptr,           # [P]     destination row of each piece
    lo_ptr,            # [P]     half-open range into order
    hi_ptr,            # [P]
    out_ptr,           # [U, W]  fp32 accumulator
    T,
    W,
    stride_g,
    BLOCK_S: tl.constexpr,
    BLOCK_W: tl.constexpr,
    ATOMIC: tl.constexpr,
):
    """Reduce ``order[lo:hi]`` rows of ``grad`` into ``out[row]``.

    One launch covers whole segments with ``ATOMIC=False`` (each output element
    is owned by exactly one program, so the write is a plain store); a second
    launch covers the pieces of split segments with ``ATOMIC=True``.
    """
    p = tl.program_id(0)
    cb = tl.program_id(1)
    cols = cb * BLOCK_W + tl.arange(0, BLOCK_W)
    cmask = cols < W

    row = tl.load(row_ptr + p)
    start = tl.load(lo_ptr + p)
    end = tl.load(hi_ptr + p)

    # fp32 accumulator -- the accumulator dtype the shipped backward uses,
    # unchanged.  Only the ORDER of the adds differs.
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)
    for s0 in range(start, end, BLOCK_S):
        srows = s0 + tl.arange(0, BLOCK_S)
        smask = srows < end
        # BLOCK_S independent index loads, then BLOCK_S independent gathers --
        # one dependent chain per chunk instead of one per row.
        src = tl.load(order_ptr + srows, mask=smask, other=0)
        tile = tl.load(
            grad_ptr + src[:, None] * stride_g + cols[None, :],
            mask=smask[:, None] & cmask[None, :],
            other=0.0,
        ).to(tl.float32)
        # Wave64 shuffle reduce + LDS combine, emitted by tl.sum.
        acc += tl.sum(tile, axis=0)

    if ATOMIC:
        tl.atomic_add(out_ptr + row * W + cols, acc, mask=cmask)
    else:
        tl.store(out_ptr + row * W + cols, acc, mask=cmask)


def build_segments(
    reverse_indices: torch.Tensor, num_rows: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sort row ids by destination and return ``(order, offsets)``.

    ``offsets`` has ``num_rows + 1`` entries, so segment ``u`` is
    ``order[offsets[u]:offsets[u + 1]]``.
    """
    if _FIXED_SIZE_OFFSETS:
        from .ec_offsets import build_segments_valid_ids

        return build_segments_valid_ids(reverse_indices, num_rows)

    idx = reverse_indices.reshape(-1)
    order = torch.argsort(idx)
    counts = torch.bincount(idx, minlength=num_rows)
    offsets = torch.zeros(num_rows + 1, dtype=torch.int64, device=idx.device)
    torch.cumsum(counts, 0, out=offsets[1:])
    return order.to(torch.int64), offsets


def _plan(
    offsets: torch.Tensor, num_rows: int, limit: int
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    """Split ``offsets`` into a whole-segment plan and a hot-piece plan.

    Returns ``(cold, hot)``, each ``(row, lo, hi)``.  Every output row appears
    in exactly one of the two, so the cold launch may store rather than
    accumulate and the hot launch's atomics land in rows nothing else writes.
    ``hot`` is None when no segment exceeds ``limit``.
    """
    dev = offsets.device
    rows = torch.arange(num_rows, dtype=torch.int64, device=dev)
    lo, hi = offsets[:-1], offsets[1:]
    if limit <= 0:
        return (rows, lo, hi), None

    is_hot = (hi - lo) > limit
    if not bool(is_hot.any()):
        return (rows, lo, hi), None

    cold = (rows[~is_hot], lo[~is_hot], hi[~is_hot])

    # Hot rows are few by construction, so building their pieces on the host is
    # cheap; doing it on device would cost a sync to read the lengths anyway.
    hot_rows, hot_lo, hot_hi = [], [], []
    for u, a, b in zip(
        rows[is_hot].tolist(), lo[is_hot].tolist(), hi[is_hot].tolist()
    ):
        for c in range(a, b, limit):
            hot_rows.append(u)
            hot_lo.append(c)
            hot_hi.append(min(c + limit, b))
    mk = lambda v: torch.tensor(v, dtype=torch.int64, device=dev)  # noqa: E731
    return cold, (mk(hot_rows), mk(hot_lo), mk(hot_hi))


def segment_reduce_rows(
    grad_out: torch.Tensor,
    reverse_indices: torch.Tensor,
    num_rows: int,
    out_dtype: torch.dtype = torch.float32,
    order: Optional[torch.Tensor] = None,
    offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``zeros(num_rows, W).index_add_(0, reverse_indices, grad_out)`` equivalent.

    Accumulates in fp32 regardless of ``grad_out``'s dtype, matching the shipped
    backward.  ``order``/``offsets`` may be supplied to reuse a prior sort.
    """
    # Both kernels address row t as grad_out + t * W, i.e. they assume a
    # row-contiguous gradient. Since ec_merge_uih_candidate the
    # artist/album expands feed torch.cat(dim=1) in ContextualPreprocessor, so
    # their incoming grads are column-slice VIEWS (row stride 2W): the kernel
    # read the wrong elements and those two tables' gradients were effectively
    # uncorrelated with the true ones (rel-L2 err 1.41). No-op for contiguous grads.
    if _STRIDED_GRAD and grad_out.dim() == 2 and grad_out.stride(1) == 1:
        pass  # kernels take the row stride; no [T, W] copy
    elif not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()
    if _GPU_PLANNER:
        from .ec_gpu_plan import segment_reduce_rows as gpu_segment_reduce_rows

        return gpu_segment_reduce_rows(
            grad_out, reverse_indices, num_rows, out_dtype, order, offsets
        )

    assert grad_out.dim() == 2, grad_out.shape
    T, W = grad_out.shape
    if order is None or offsets is None:
        order, offsets = build_segments(reverse_indices, num_rows)

    out = torch.zeros((num_rows, W), dtype=torch.float32, device=grad_out.device)
    BLOCK_W = min(triton.next_power_of_2(W), _BLOCK_W_CAP)
    ncb = triton.cdiv(W, BLOCK_W)
    cold, hot = _plan(offsets, num_rows, _SPLIT_ABOVE)

    def launch(plan, atomic):
        row, lo, hi = plan
        if row.numel() == 0:
            return
        _seg_reduce_kernel[(row.numel(), ncb)](
            grad_out, order, row, lo, hi, out, T, W, grad_out.stride(0),
            BLOCK_S=_BLOCK_S, BLOCK_W=BLOCK_W, ATOMIC=atomic,
            num_warps=_NUM_WARPS, num_stages=1,
        )

    launch(cold, False)
    if hot is not None:
        launch(hot, True)

    return out if out_dtype == torch.float32 else out.to(out_dtype)
