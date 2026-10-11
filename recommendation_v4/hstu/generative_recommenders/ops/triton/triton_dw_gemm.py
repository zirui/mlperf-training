# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Weight-gradient GEMM ``dW = X^T @ dY`` for a token-sized reduction.

Every linear in the HSTU stack computes its weight gradient as ``X^T @ dY``, where X is
``[tokens, in_features]`` and dY is ``[tokens, out_features]``. Because the sequences are
jagged and flattened, ``tokens`` is in the millions while the output is only weight-sized
(256x512 up to 512x2048). That is a reduction, not a square GEMM: it streams two large
operands into a small accumulator, so what matters is HBM traffic and how the reduction
over K is split.

``dw_gemm`` selects an implementation with ``HSTU_DW_GEMM``:

* ``aten`` (default): ``torch.mm``.
* ``bmm``: ``chunked_bmm`` reshapes the reduction into a batch of medium-K GEMMs and
  reduces the partials in fp32.
* ``triton``: a split-K Triton kernel with an fp32 atomic combine.

All three paths are numerically equivalent to the bf16 precision of the output.

The training call sites (``triton_addmm.py`` and ``triton_hstu_linear.py``) use plain
``torch.mm``; this module is reachable only by calling ``dw_gemm`` directly.
"""

import os
from typing import List

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl

from generative_recommenders.common import backend_allow_tf32, triton_autotune

# Below this reduction length neither replacement is worth it: a short K means the output
# tiles alone already fill the machine and the vendor kernel is fine.
MIN_REDUCTION: int = 65536

# On narrow outputs ([512,K]x[K,256] and the 24-wide action MLP) the batched GEMM's
# per-chunk output becomes too small to amortise its own launch, so chunked_bmm is used
# only for N >= MIN_BMM_N.
MIN_BMM_N: int = 512

# Enough chunks to expose parallelism without shrinking each GEMM's own K too far.
BMM_CHUNKS: int = 16


def dw_gemm_mode() -> str:
    """Which dW implementation to use: ``aten`` (default), ``bmm`` or ``triton``."""
    return os.environ.get("HSTU_DW_GEMM", "aten").lower()


def chunked_bmm(
    x: torch.Tensor, dy: torch.Tensor, chunks: int = BMM_CHUNKS
) -> torch.Tensor:
    """``x.t() @ dy`` as a batch of medium-K GEMMs with an fp32 reduction.

    Rather than asking the library for one GEMM with K in the millions, this asks for
    ``chunks`` GEMMs with K a few hundred thousand and reduces the partials afterwards.
    """
    k, m = x.shape
    n = dy.shape[1]
    # Keep each chunk's row count a multiple of 8 so every batch element starts on a
    # well-aligned boundary; the remainder is swept up by a single small mm below.
    per = (k // chunks) // 8 * 8
    used = per * chunks
    partials = torch.bmm(
        x[:used].view(chunks, per, m).transpose(1, 2),
        dy[:used].view(chunks, per, n),
    )
    # sum(dtype=fp32) accumulates in fp32 without materialising an fp32 copy of partials.
    out = partials.sum(0, dtype=torch.float32)
    if used < k:
        out += torch.mm(x[used:].t(), dy[used:]).float()
    return out.to(torch.promote_types(x.dtype, dy.dtype))


def _get_dw_configs() -> List[triton.Config]:
    """Tile candidates for the split-K kernel, biased towards large tiles.

    Operand re-reads dominate: each output tile re-reads its M-slice of X once per N-tile
    and its N-slice of dY once per M-tile, so traffic amplification over a single
    streaming pass is ``M*N*(1/BLOCK_M + 1/BLOCK_N) / (M+N)``. At the hot 512x2048 shape
    that is 6.4x for 128x128 tiles but 3.2x for 256x256, which is why the space leans on
    big tiles even though they spend most of the register file on the fp32 accumulator.

    SPLIT_K then has to be large enough to fill 256 CUs: 512x2048 in 256x256 tiles is only
    16 output tiles, so it takes SPLIT_K=16 to reach one workgroup per CU.
    """
    # (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages)
    space = [
        (256, 256, 32, 8, 8, 2),
        (256, 256, 32, 16, 8, 2),
        (256, 256, 64, 8, 8, 2),
        (256, 256, 64, 16, 8, 2),
        (128, 256, 64, 16, 8, 2),
        (256, 128, 64, 16, 8, 2),
        (128, 256, 64, 32, 8, 2),
        (256, 128, 64, 32, 8, 2),
        (128, 128, 64, 32, 8, 2),
        (128, 128, 128, 16, 8, 2),
        (128, 128, 64, 16, 4, 2),
        (256, 256, 64, 8, 8, 3),
        (128, 256, 64, 16, 8, 3),
        (64, 256, 64, 32, 4, 2),
        (256, 64, 64, 32, 4, 2),
        (64, 64, 128, 32, 4, 2),
    ]
    configs = []
    for block_m, block_n, block_k, split_k, num_warps, num_stages in space:
        kwargs = {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "SPLIT_K": split_k,
        }
        if torch.version.hip:
            # kpack is deprecated on gfx950 and silently forced to 1, so it is left out.
            kwargs.update({"matrix_instr_nonkdim": 16, "waves_per_eu": 0})
        configs.append(
            triton.Config(
                kwargs,
                num_stages=num_stages if triton.__version__ >= "3.2.0" else 0,
                num_warps=num_warps,
            )
        )
    return configs


@triton_autotune(
    configs=_get_dw_configs(),
    key=["M", "N"],
    # The output is accumulated into, so every trial has to start from zero or the
    # autotuner's own repetitions would corrupt the result it leaves behind.
    reset_to_zero=["dw_ptr"],
    # K is in the millions, so one trial is milliseconds; the default 25/100 warmup and
    # rep would spend a minute per shape for no extra signal.
    warmup=3,
    rep=10,
)
@triton.jit
def _dw_gemm_split_k(
    x_ptr,
    dy_ptr,
    dw_ptr,
    M,
    N,
    K,
    stride_xk,
    stride_xm,
    stride_dyk,
    stride_dyn,
    stride_dwm,
    stride_dwn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
) -> None:
    """One ``[BLOCK_M, BLOCK_N]`` output tile per program, over a strided slice of K.

    ``SPLIT_K`` programs cooperate per tile so the machine stays full even though the
    output is tiny, and their partials are combined with ``tl.atomic_add`` into an fp32
    buffer. That makes the result nondeterministic in the same way ``scatter_add_``,
    ``index_add_`` and rocBLAS's own split-K reduction already are on this path, and it
    keeps ``SPLIT_K`` free for the autotuner -- a workspace-plus-reduction scheme would
    have to size the workspace before autotuning has chosen it.

    The autotune key above is deliberately ``["M", "N"]`` and *not* K: K is the token
    count and differs on every step, so keying on it would re-tune continuously and
    never amortise. M and N select the tile; K only sets the trip count.
    """
    pid = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # K * stride overflows int32 at these token counts (2.4M rows x 2048 cols), so the
    # reduction offsets have to be 64-bit from the start.
    offs_k = (pid_k * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.int64)

    mask_m = offs_m < M
    mask_n = offs_n < N

    x_ptrs = x_ptr + offs_k[:, None] * stride_xk + offs_m[None, :] * stride_xm
    dy_ptrs = dy_ptr + offs_k[:, None] * stride_dyk + offs_n[None, :] * stride_dyn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_step = BLOCK_K * SPLIT_K
    for k0 in range(0, tl.cdiv(K, k_step)):
        mask_k = offs_k + k0 * k_step < K
        # X is row-major [K, M], so loading [BLOCK_K, BLOCK_M] is coalesced and the
        # transpose happens in registers. Gathering [BLOCK_M, BLOCK_K] instead would
        # stride by M on the inner axis and destroy coalescing.
        x = tl.load(x_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
        dy = tl.load(dy_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(tl.trans(x), dy, allow_tf32=ALLOW_TF32)
        x_ptrs += k_step * stride_xk
        dy_ptrs += k_step * stride_dyk

    dw_ptrs = (
        dw_ptr + offs_m[:, None].to(tl.int64) * stride_dwm + offs_n[None, :] * stride_dwn
    )
    tl.atomic_add(dw_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_dw_gemm(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """Compute ``x.t() @ dy`` with the split-K kernel.

    Args:
        x: ``[K, M]`` activations, row-major (the forward input, *not* pre-transposed).
        dy: ``[K, N]`` output gradients, row-major.

    Returns:
        ``[M, N]``, in the dtype ``torch.mm`` would have produced.
    """
    assert x.dim() == 2 and dy.dim() == 2, f"expected 2D, got {x.shape} and {dy.shape}"
    assert x.shape[0] == dy.shape[0], f"K mismatch: {x.shape} vs {dy.shape}"
    k, m = x.shape
    n = dy.shape[1]
    out_dtype = torch.promote_types(x.dtype, dy.dtype)

    # fp32 because the atomic combine needs a type that can hold a 2M-term sum; the cast
    # back matches torch.mm's output dtype.
    dw = torch.zeros((m, n), device=x.device, dtype=torch.float32)

    def grid(meta):  # pyre-ignore[3]
        return (
            triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
            meta["SPLIT_K"],
        )

    _dw_gemm_split_k[grid](
        x,
        dy,
        dw,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        dy.stride(0),
        dy.stride(1),
        dw.stride(0),
        dw.stride(1),
        ALLOW_TF32=backend_allow_tf32(),
    )
    return dw.to(out_dtype)


def dw_gemm(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """``x.t() @ dy``, routed to whichever implementation is the better tool.

    Falls back to ``torch.mm`` whenever the replacements' assumptions do not hold, so call
    sites can use this unconditionally.
    """
    mode = dw_gemm_mode()
    if (
        mode == "aten"
        or x.device.type != "cuda"
        or x.shape[0] < MIN_REDUCTION
        or x.dtype != dy.dtype
        # Non-contiguous operands would turn the coalesced loads into gathers, and the
        # bmm path needs to reshape the token axis.
        or not x.is_contiguous()
        or not dy.is_contiguous()
    ):
        return torch.mm(x.t(), dy)
    if mode == "bmm":
        if dy.shape[1] < MIN_BMM_N:
            return torch.mm(x.t(), dy)
        return chunked_bmm(x, dy)
    if mode == "triton":
        return triton_dw_gemm(x, dy)
    raise ValueError(
        f"HSTU_DW_GEMM={mode!r} is not one of 'aten', 'bmm', 'triton'"
    )
