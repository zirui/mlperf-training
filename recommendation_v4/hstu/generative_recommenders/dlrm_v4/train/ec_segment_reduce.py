"""Segmented row reduction for the EC dedup-expand backward.

Replaces the global-atomic ``scatter_add_`` in
:class:`~generative_recommenders.dlrm_v4.train.ec_bf16_output._CastExpandFp32Bwd`
with a sort-then-reduce kernel.  Scoping and the closed alternatives are in
``_scatter_add_design.md``; the short version:

``scatter_add_`` on this call is the 4th largest kernel in the step at
**17.4 ms/rank/step**, reducing 7,108,550 token rows into 69,256 unique rows --
**102.6x duplication**, so ~103 atomic adds contend for every output row.  It
runs **7.0x** off its own HBM floor, and the gap is atomic serialization, not
bandwidth: the output is 142 MB against a 14.55 GB read.

This kernel removes the contention instead of trying to stream it faster.  Each
program owns one unique row and one column block, walks that row's segment
accumulating in registers, and performs **one** non-atomic store.  No atomics,
no read-modify-write.

There is a second saving that the design note did not count.  Today's backward
materializes ``grad_out.to(torch.float32)`` -- a full 14.55 GB fp32 temporary
that is written and then read straight back by the scatter.  This kernel loads
the bf16 gradient and upcasts **in registers**, so the tensor never exists:

    today   write 14.55 GB (upcast) + read 14.55 GB (scatter) + 142 MB out
    here    read   7.28 GB (bf16, upcast in-register)          + 142 MB out

so the floor moves from 2.49 ms to ~1.25 ms and the upcast's own ~3.7 ms is
deleted outright.

**Why the segment is walked in CHUNKS, not row by row.**  The obvious inner
loop -- ``src = tl.load(order_ptr + i)`` then ``tl.load(grad_ptr + src * W)``
-- is a *dependent* load chain: the gradient load's address does not exist
until the index load returns, so a 103-row segment serializes ~103 HBM
latencies and occupancy is the only thing hiding any of it.  That is the
failure mode KernelForge's reduction card warns about (``local_knowledge/
languages/triton/operators/reduction/tuning.md``: the thread-level pass is the
bandwidth-bound part, the cross-lane combine is cheap -- so vectorize the
load).  Instead each iteration loads ``BLOCK_S`` indices at once and issues a
``[BLOCK_S, BLOCK_W]`` gather, giving ``BLOCK_S`` independent loads in flight,
then folds the tile with ``tl.sum(axis=0)`` -- which lowers to the wave64
shuffle reduce plus LDS combine.  One dependent chain per *chunk* rather than
per row.

**Ordering.**  TorchRec's dedup is hash-based and returns no permutation
(``_dedup_indices`` stashes only ``reverse_indices``), so the ordering must be
produced here.  ``argsort`` over 7.1M int64 is ~57 MB of index traffic -- a few
percent of the prize, not the margin.

**Numerics.**  Segment reduction changes the *order* of the fp32 adds.  Today's
atomics are already nondeterministic, and ``dlrm_hstu.py:181-183`` records both
orders scoring identically against a float64 reference, so this is expected to
be fine -- but it is a reordering, and it owes a convergence A/B rather than an
SNR check.  ``_seg_reduce_check.py`` scores it against float64 at production
shape.

**Measured at production shape** (``_seg_reduce_check.py``):

    scatter_add_ (+ the fp32 upcast it needs)   16.763 ms
    this, sort included                          2.668 ms    6.28x
    this, ordering reused                        1.498 ms   11.19x   4.86 TB/s
                                                                     = 83% of peak

1.498 ms against a 1.25 ms HBM floor is 1.20x off -- the kernel is doing
essentially nothing but streaming, which is the point.  Against the incumbent
that is **-14.1 ms/rank/step = -4.86% of a 290.195 ms step**, ahead of the
design note's "-12.4 ms even at 2x floor".  Accuracy improves rather than
degrades: max error vs an fp64 reference is **0.61x** the incumbent's, because
a segment tree reduce is better-conditioned than 103 racing atomics.

⚠ **The measurement uses a SYNTHETIC uniform index distribution** (T/U = 102.6x
duplication, segment lengths 64..148).  The real ``reverse_indices`` shape is
not measured, and both the sort cost and the load balance depend on it: a
skew=1.4 control run holds correctness and still wins 16.5x warm, but its sort
costs 32 ms rather than 1.13 ms.  Before quoting the in-step delta, dump the
real segment-length histogram.

NOTHING IMPORTS THIS YET.  Wiring it into ``_CastExpandFp32Bwd.backward``
belongs behind ``EC_SEGMENT_REDUCE`` and is a separate, gated change.
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
# DLRM_HSTU_FP8_EC_STRIDED_GRAD (gin default on, yambda_5b.gin item (13); absent = off): read a row-strided (column-slice view)
# gradient in place instead of materialising grad_out.contiguous() first.
# The kernels load exactly the same elements in the same order (only the row
# address uses the real row stride), so cold segments are bitwise identical and
# hot pieces differ only by the pre-existing atomic-order nondeterminism; what
# disappears is a full [T, W] bf16 read+write per call (artist/album grads are
# column slices of the [N, 1024] cat gradient: 2 x ~0.94 ms/rank/step at 1n).
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

    Plain setter, NOT a gin configurable: this module is imported lazily (to
    keep triton off the import path of runs that do not arm the kernel), so it
    is not registered when gin parses. The gin surface lives on
    `ec_bf16_output.enable_segment_reduce`, which forwards into here -- that
    module is light and is imported eagerly for exactly this reason.

    DEFAULTS ARE MEASURED, not guessed (30-point sweep at production shape).
    The spread across that sweep is 10x -- 1.488 ms at (32, 2, 512) against
    14.50 ms at (8, 8, 256) -- so these are not free parameters.  Change one
    only with a sweep behind it.

    Args:
      split_above: rows longer than this are cut into several pieces that
        finish with an atomic add.  Segments average ~103 rows but are violently
        non-uniform (production p50 43, p99 1,439, max 64,326), and a program
        that owns a very hot row runs serially while its neighbours idle, so
        this trades a handful of atomics for load balance.  1024 sits between
        p99 and p999.  This exists for LOAD BALANCE, not occupancy: at ~54k
        output rows the grid already fills the device.  0 disables splitting.
      block_s: rows gathered per inner iteration -- the in-flight load count,
        so this is what hides HBM latency.  The tile is [block_s, block_w] fp32
        in registers.  Measured: 4 -> 3.70 ms, 8 -> 1.97, 16 -> 1.50,
        32 -> 1.49, 64 -> 1.54.  Below 16 the kernel degenerates into a
        dependent load chain, which is the whole cost this kernel removes.
      num_warps: the sharp one, and the AMD trap.  2 wins EVERY one of the 10
        (block_s, block_w) cells in the sweep, by 1.9-4.8x: 8 warps put two
        waves on a SIMD at ~256 VGPR each and the tile spills to scratch.
        Memory-bound Triton kernels on gfx950 want 2.
      block_w_cap: upper bound on the width tile; the actual block_w is
        min(next_pow2(W), block_w_cap), so at the production W=512 this is not
        binding and exists to keep the register tile sane at wider embeddings.
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
    # row-contiguous gradient. Since ec_merge_uih_candidate (iter 30) the
    # artist/album expands feed torch.cat(dim=1) in ContextualPreprocessor, so
    # their incoming grads are column-slice VIEWS (row stride 2W): the kernel
    # read the wrong elements and those two tables' gradients were effectively
    # uncorrelated with the true ones (rel-L2 err 1.41, eval_preproc_dedup /
    # iter_31 dbg4). No-op for contiguous grads.
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
