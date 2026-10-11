"""FP8 machinery for the HSTU backward projection GEMMs.

Kept separate from the forward module `triton_fp8.py`.

**The cast.** `quantize_with_scale` in the forward module is
`(v / scale).clamp(...).to(fp8)`, an elementwise chain that materializes
full-size intermediates. This module uses a one-pass, bit-exact Triton cast
instead.

**The scale.** The gradient amax grows over training by orders of magnitude,
so the forward's calibrate-then-freeze policy does not fit here. This one
refreshes on a cadence at a margin; the margin covers single-step spikes that
no cadence tracks.

**The format.** E5M2 for gradients, E4M3 for weights. See the format note
below.

Refreshing needs an amax. The cast already reads every element, so the
reduction is computed in the same pass, and the controller never reads a
device value back to the host in the steady-state step.

Enable with the single switch:

    DLRM_HSTU_FP8_BWD=1                (gin: apply_env_bootstrap.DLRM_HSTU_FP8_BWD)

That switch enables the whole path. The knobs below are tunables:

    DLRM_HSTU_FP8_BWD_MARGIN=8         headroom over the observed amax
    DLRM_HSTU_FP8_BWD_REFRESH_CALLS    steps between scale refreshes
    DLRM_HSTU_FP8_BWD_WARMUP_CALLS     backward calls run in BF16 first
    DLRM_HSTU_FP8_BWD_GRAD_FORMAT      e5m2 (default) or e4m3
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Dict, Optional, Tuple

import torch
from generative_recommenders.ops.triton import fp8_tm_alloc, wgrad_ck

logger: logging.Logger = logging.getLogger(__name__)

# Weights are E4M3 and gradients are E5M2, the standard FP8 training recipe.
#
# In a float format the margin costs nothing in mantissa, so its price is paid
# at the bottom of the range: E4M3 flushes a large fraction of the gradient to
# zero, and Adam normalizes per parameter, so a flushed coordinate receives no
# update at all. E5M2 spends one mantissa bit for four decades of range.
# Weights keep E4M3: their range is narrow, so the mantissa bit matters more.
_FP8_E4M3: Optional[torch.dtype] = getattr(torch, "float8_e4m3fn", None)
_FP8_E5M2: Optional[torch.dtype] = getattr(torch, "float8_e5m2", None)
_E4M3_MAX: float = 448.0
_E5M2_MAX: float = 57344.0

_GRAD_FORMAT: str = os.environ.get("DLRM_HSTU_FP8_BWD_GRAD_FORMAT", "e5m2").lower()
_GRAD_DTYPE: Optional[torch.dtype] = (
    _FP8_E4M3 if _GRAD_FORMAT == "e4m3" else _FP8_E5M2
)
_GRAD_MAX: float = _E4M3_MAX if _GRAD_FORMAT == "e4m3" else _E5M2_MAX

# FP8 output-projection wgrad (DLRM_HSTU_FP8_BWD_OUT_WGRAD=1, default off; the
# call site is HSTUComputeOutputFunction in triton_hstu_linear.py). Read here so
# the one switch can choose the numerics of the wgrad's gradient operand below.
# Unset (the default path): nothing in this block is ever read by a step.
_OUT_WGRAD_ARMED: bool = os.environ.get("DLRM_HSTU_FP8_BWD_OUT_WGRAD", "0") == "1"

# Format of the token-major gradient copy that only the FP8 output-projection
# wgrad reads (requires snapshot_decode_scale=True at the call site). "e5m2"
# writes the same bytes as the row-major dgrad operand. "e4m3" writes them at
# the same delayed scale times 2^SHIFT (2^7 = 57344 / 448 keeps clipping
# identical to the e5m2 operand), so the decode scale stays an exact power of
# two; the dgrad keeps e5m2. e4m3 gives the wgrad operand three more mantissa
# bits at the cost of range the delayed margin already provides. Only valid
# with e5m2 grads (with e4m3 grads there is no separate copy).
# Armed default: with DLRM_HSTU_FP8_BWD_OUT_WGRAD=1 and this knob unset the arm
# uses e4m3; unarmed (or explicitly set) it is e5m2 / the given value.
_OUT_WGRAD_DFMT: str = os.environ.get(
    "DLRM_HSTU_FP8_BWD_OUT_WGRAD_DFMT", "e4m3" if _OUT_WGRAD_ARMED else "e5m2"
).lower()
if _OUT_WGRAD_DFMT not in ("e5m2", "e4m3"):
    raise ValueError(f"DLRM_HSTU_FP8_BWD_OUT_WGRAD_DFMT={_OUT_WGRAD_DFMT!r}")
_OUT_WGRAD_T_E4M3: bool = _OUT_WGRAD_DFMT == "e4m3" and _GRAD_FORMAT != "e4m3"
# Headroom of that e4m3 copy (only read when DFMT=e4m3). The copy is cast at
# inv_scale * 2^-SHIFT. SHIFT=7 is 448 / 57344: the e4m3 copy clips exactly
# where the e5m2 dgrad operand does, so it carries the full
# DLRM_HSTU_FP8_BWD_MARGIN headroom. Each step down (6, 5, 4) gives one binade
# of that headroom back to resolution, but the copy clips once a window's amax
# grows more than MARGIN * 2^(SHIFT-7) over the published one. Such windows are
# counted per site in `t_clipped_windows` (site_diagnostics, and the scale log
# line under DLRM_HSTU_FP8_BWD_SCALE_LOG=1). The decode scale stays an exact
# power of two for every SHIFT.
#
# SHIFT=auto adapts it per site on the host, at the scale refresh (one
# t_clipped_windows host read per site per refresh window, i.e. every
# DLRM_HSTU_FP8_BWD_REFRESH_CALLS steps, armed path only): it starts at
# DLRM_HSTU_FP8_BWD_OUT_WGRAD_DFMT_SHIFT_FLOOR (default 5), steps up one binade
# (max 7, the clip-equivalent of the e5m2 operand) after any refresh window in
# which the copy clipped, and steps back down one after
# DLRM_HSTU_FP8_BWD_OUT_WGRAD_DFMT_SHIFT_RELAX (default 4) consecutive clean
# windows. A late-run amax jump therefore costs at most one window of clipping.
# Each call decodes with the shift it was cast at.
# Armed default: with DLRM_HSTU_FP8_BWD_OUT_WGRAD=1, an e4m3 copy in use and
# this knob unset, SHIFT=auto; otherwise 7.
_OUT_WGRAD_DFMT_SHIFT_RAW: str = os.environ.get(
    "DLRM_HSTU_FP8_BWD_OUT_WGRAD_DFMT_SHIFT",
    "auto" if _OUT_WGRAD_ARMED and _OUT_WGRAD_T_E4M3 else "7",
).lower()
_OUT_WGRAD_DFMT_SHIFT_AUTO: bool = _OUT_WGRAD_DFMT_SHIFT_RAW == "auto"
_OUT_WGRAD_DFMT_SHIFT_FLOOR: int = int(
    os.environ.get("DLRM_HSTU_FP8_BWD_OUT_WGRAD_DFMT_SHIFT_FLOOR", "5")
)
_OUT_WGRAD_DFMT_SHIFT_RELAX: int = max(
    1, int(os.environ.get("DLRM_HSTU_FP8_BWD_OUT_WGRAD_DFMT_SHIFT_RELAX", "4"))
)
if _OUT_WGRAD_DFMT_SHIFT_FLOOR not in (4, 5, 6, 7):
    raise ValueError(
        "DLRM_HSTU_FP8_BWD_OUT_WGRAD_DFMT_SHIFT_FLOOR="
        f"{_OUT_WGRAD_DFMT_SHIFT_FLOOR} (4..7)"
    )
# Initial (and, unless auto, fixed) shift.
_OUT_WGRAD_DFMT_SHIFT: int = (
    _OUT_WGRAD_DFMT_SHIFT_FLOOR
    if _OUT_WGRAD_DFMT_SHIFT_AUTO
    else int(_OUT_WGRAD_DFMT_SHIFT_RAW)
)
if _OUT_WGRAD_DFMT_SHIFT not in (4, 5, 6, 7):
    raise ValueError(
        f"DLRM_HSTU_FP8_BWD_OUT_WGRAD_DFMT_SHIFT={_OUT_WGRAD_DFMT_SHIFT} (4..7)"
    )


def next_t_shift(
    shift: int, new_clips: int, clean_windows: int, floor: int, relax: int
) -> Tuple[int, int]:
    """One auto-shift update per refresh window -> (shift, clean_windows).

    Pure host arithmetic so it can be unit-tested without a GPU. Clipping in the
    window steps the shift up (more headroom) and resets the clean count; a run
    of `relax` clean windows steps it down (more resolution), never below
    `floor` and never above 7.
    """
    if new_clips > 0:
        return min(7, shift + 1), 0
    clean_windows += 1
    if clean_windows >= relax and shift > floor:
        return shift - 1, 0
    return shift, clean_windows


# The whole path, from one switch: the site is meaningless without the scale
# controller, so both are enabled together.
_BWD_ENABLED: bool = os.environ.get("DLRM_HSTU_FP8_BWD", "0") == "1"

# Tunables, not gates.
_BWD_MARGIN: float = float(os.environ.get("DLRM_HSTU_FP8_BWD_MARGIN", "8"))
# Submission logging profile: the per-refresh scale line (two blocking
# .item() reads per site every DLRM_HSTU_FP8_BWD_REFRESH_CALLS steps, mid-backward)
# is a diagnostic, not an MLPerf requirement. Off by default; DLRM_HSTU_FP8_BWD_SCALE_LOG=1
# restores it. The scale and clipped_windows state is unchanged either way.
_BWD_SCALE_LOG: bool = os.environ.get("DLRM_HSTU_FP8_BWD_SCALE_LOG", "0") == "1"
_BWD_REFRESH_CALLS: int = max(
    1, int(os.environ.get("DLRM_HSTU_FP8_BWD_REFRESH_CALLS", "200"))
)
# Long enough to see a representative amax, short relative to a training run.
_BWD_WARMUP_CALLS: int = max(
    1, int(os.environ.get("DLRM_HSTU_FP8_BWD_WARMUP_CALLS", "4"))
)

_CAST_BLOCK: int = 8192

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - ROCm images always carry Triton
    _TRITON_AVAILABLE = False


def scale_controller_enabled() -> bool:
    return (
        _BWD_ENABLED
        and _TRITON_AVAILABLE
        and _FP8_E4M3 is not None
        and _GRAD_DTYPE is not None
    )


if _TRITON_AVAILABLE:

    @triton.jit
    def _quantize_amax_kernel(
        SRC,
        DST,
        PARTIAL,
        INV_SCALE,
        n_elements,
        BLOCK: tl.constexpr,
        LIMIT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        # These gradients can exceed 2^31 elements, where a 32-bit element
        # offset wraps, so the offset is built in 64-bit from the start.
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        mask = offs < n_elements
        x = tl.load(SRC + offs, mask=mask, other=0.0)
        # One partial per program rather than an atomic. 600k programs
        # contending on a single address would serialize the pass that is
        # supposed to be free; the partials are a few MB and reduce separately.
        tl.store(PARTIAL + pid, tl.max(tl.abs(x)))
        scaled = x.to(tl.float32) * tl.load(INV_SCALE)
        scaled = tl.minimum(tl.maximum(scaled, -LIMIT), LIMIT)
        tl.store(DST + offs, scaled.to(DST.dtype.element_ty), mask=mask)


if _TRITON_AVAILABLE:

    @triton.jit
    def _quantize_amax_dual_kernel(
        SRC,
        DST,
        DST_T,
        PARTIAL,
        INV_SCALE,
        M,
        M_PAD,
        N: tl.constexpr,
        stride_src,
        stride_dst_t,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        LIMIT: tl.constexpr,
        SEPARATE_T: tl.constexpr = False,
        T_INV_MUL: tl.constexpr = 1.0,
        LIMIT_T: tl.constexpr = 1.0,
        T_TILED: tl.constexpr = False,
    ):
        # Same arithmetic as _quantize_amax_kernel, on a [BLOCK_M, BLOCK_N] tile,
        # also storing the tile transposed (tokens contiguous) for the FP8 wgrad.
        # SEPARATE_T: the transposed copy gets its own format, at the row-major
        # scale times a power of two (T_INV_MUL), clamped to LIMIT_T.
        # T_TILED: the transposed copy is step-tiled [M_PAD/128][N][128]
        # (token rm of column rn at (rm // 128) * N * 128 + rn * 128 + rm % 128;
        # M_PAD = roundup(M, 128)) instead of [N, M_PAD] at stride_dst_t. Same
        # values; rows >= M are zeros in both layouts.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rm < M
        x = tl.load(
            SRC + rm[:, None] * stride_src + rn[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        tl.store(
            PARTIAL + pid_m * tl.num_programs(1) + pid_n,
            tl.max(tl.max(tl.abs(x), axis=1), axis=0),
        )
        scaled0 = x.to(tl.float32) * tl.load(INV_SCALE)
        scaled = tl.minimum(tl.maximum(scaled0, -LIMIT), LIMIT)
        q = scaled.to(DST.dtype.element_ty)
        tl.store(DST + rm[:, None] * N + rn[None, :], q, mask=row_mask[:, None])
        if SEPARATE_T:
            st = scaled0 * T_INV_MUL
            st = tl.minimum(tl.maximum(st, -LIMIT_T), LIMIT_T)
            qt = st.to(DST_T.dtype.element_ty)
        else:
            qt = q
        if T_TILED:
            tl.store(
                DST_T
                + (rm[None, :] // 128) * (N * 128)
                + rn[:, None].to(tl.int64) * 128
                + rm[None, :] % 128,
                tl.trans(qt),
                mask=(rm < M_PAD)[None, :],
            )
        else:
            tl.store(
                DST_T + rn[:, None].to(tl.int64) * stride_dst_t + rm[None, :],
                tl.trans(qt),
                mask=(rm < M_PAD)[None, :],
            )


# (BLOCK_M tokens, BLOCK_N columns, num_warps) of the dual-layout cast.
_DUAL_CAST_CONFIG: Tuple[int, int, int] = (256, 128, 8)


def quantize_row_major_dual(
    value: torch.Tensor,
    inv_scale: torch.Tensor,
    pad_to: int = 16,
    dtype: Optional[torch.dtype] = None,
    limit: Optional[float] = None,
    config: Optional[Tuple[int, int, int]] = None,
    t_dtype: Optional[torch.dtype] = None,
    t_inv_mul: float = 1.0,
    t_limit: Optional[float] = None,
    return_partial: bool = False,
    t_tiled: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`quantize_row_major` for a 2-D [M, N] value that also writes the same FP8
    bytes token-major, [N, M_pad] with zero padding to a multiple of `pad_to`
    tokens (the K-contiguous operand a library FP8 TN wgrad needs). One read of
    the source serves both layouts. Returns (row_major, token_major, amax); the
    row-major bytes and the amax equal `quantize_row_major`'s exactly.

    t_dtype (default: same as the row-major copy) casts the token-major copy to
    its own format at `inv_scale * t_inv_mul` (pass a power of two so the decode
    scale stays exact), clamped to t_limit. The row-major bytes are unchanged.

    t_tiled (for the custom out-proj wgrad wgrad_ck_w3_t; default off): the
    token-major copy is STEP-TILED, a 3-D [ceil(M/128), N, 128] tensor (step k =
    tokens [128k, 128k+128) of all N columns, zero tail tokens [M, ceil(M/128)*128);
    `pad_to` is ignored), so a 2-D consumer fails loudly instead of misreading it.
    Same bytes per (column, token) as the 2-D copy; the row-major copy, the amax
    and the partials are unchanged (same grid). wgrad_ck_w3.untile_to_row_major
    gives back the 2-D pad_to=16 copy bit for bit.
    """
    dtype = _GRAD_DTYPE if dtype is None else dtype
    limit = _GRAD_MAX if limit is None else limit
    if not _TRITON_AVAILABLE or dtype is None:
        raise RuntimeError("Triton FP8 cast requested without Triton")
    source = value.detach()
    assert source.dim() == 2
    if not source.is_contiguous():
        source = source.contiguous()
    m, n = source.shape
    bm, bn, num_warps = config or _DUAL_CAST_CONFIG
    bn = min(bn, n)
    assert n % bn == 0, (n, bn)
    if t_tiled:
        pad_to = 128  # whole 128-token steps
    m_pad = (m + pad_to - 1) // pad_to * pad_to
    out = torch.empty((m, n), dtype=dtype, device=source.device)
    separate_t = t_dtype is not None and (t_dtype != dtype or t_inv_mul != 1.0)
    # DLRM_HSTU_FP8_TM_NONEXP: token-major wgrad operand from a non-expandable pool.
    out_t = fp8_tm_alloc.empty_tm(
        (m_pad // 128, n, 128) if t_tiled else (n, m_pad),
        t_dtype if separate_t else dtype,
        source.device,
    )
    grid = ((m_pad + bm - 1) // bm, n // bn)
    partial = torch.empty(grid[0] * grid[1], dtype=torch.float32, device=source.device)
    _quantize_amax_dual_kernel[grid](
        source,
        out,
        out_t,
        partial,
        inv_scale,
        m,
        m_pad,
        n,
        source.stride(0),
        m_pad,
        BLOCK_M=bm,
        BLOCK_N=bn,
        LIMIT=limit,
        SEPARATE_T=separate_t,
        T_INV_MUL=float(t_inv_mul) if separate_t else 1.0,
        LIMIT_T=float(t_limit if t_limit is not None else limit) if separate_t else 1.0,
        T_TILED=bool(t_tiled),
        num_warps=num_warps,
    )
    if return_partial:
        # DLRM_HSTU_FP8_BWD_HELPERS_FUSED: the site kernel reduces these.
        return out, out_t, partial
    return out, out_t, partial.amax()


if _TRITON_AVAILABLE:

    @triton.jit
    def _quantize_amax_colsum_kernel(
        SRC,
        DST,
        PARTIAL_AMAX,
        PARTIAL_COL,
        INV_SCALE,
        n_rows,
        n_rows_out,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        ROW_ITERS: tl.constexpr,
        LIMIT: tl.constexpr,
    ):
        """Gradient cast + input amax + fp32 column partial sums in ONE read.

        Program (pid_m, pid_n) owns ROW_ITERS*BLOCK_M rows x BLOCK_N columns.
        The column sums are of the UNQUANTIZED input: they become the bias
        gradient, which stays full precision. Partials reduce separately.
        Rows in [n_rows, n_rows_out) of DST are written as zero (padding for a
        K-padded wgrad); programs that start at or past n_rows store no
        partials, so amax/colsum are those of the unpadded input bit for bit.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        inv = tl.load(INV_SCALE)
        # Accumulate element-wise in registers and reduce ONCE at the end: a
        # per-iteration tl.sum(axis=0) is a cross-warp LDS reduction per
        # row-block and is far from bandwidth.
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        amax = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        row0 = pid_m.to(tl.int64) * (ROW_ITERS * BLOCK_M)
        rows = row0 + tl.arange(0, BLOCK_M).to(tl.int64)
        offs = rows[:, None] * N + cols[None, :]
        for _ in range(ROW_ITERS):
            mask = rows[:, None] < n_rows
            x = tl.load(SRC + offs, mask=mask, other=0.0).to(tl.float32)
            acc += x
            amax = tl.maximum(amax, tl.abs(x))
            scaled = tl.minimum(tl.maximum(x * inv, -LIMIT), LIMIT)
            tl.store(
                DST + offs,
                scaled.to(DST.dtype.element_ty),
                mask=rows[:, None] < n_rows_out,
            )
            rows += BLOCK_M
            offs += BLOCK_M * N
        if row0 < n_rows:
            tl.store(
                PARTIAL_COL + pid_m.to(tl.int64) * N + cols, tl.sum(acc, axis=0)
            )
            tl.store(
                PARTIAL_AMAX + pid_m * tl.num_programs(1) + pid_n,
                tl.max(tl.max(amax, axis=1), axis=0),
            )


if _TRITON_AVAILABLE:

    @triton.jit
    def _quantize_amax_colsum_v2_tile(
        SRC,
        DST,
        PARTIAL_AMAX,
        PARTIAL_COL,
        inv,
        pid_m,
        pid_n,
        n_rows,
        n_rows_out,
        n_cb,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        ROW_ITERS: tl.constexpr,
        LIMIT: tl.constexpr,
    ):
        """One (pid_m, pid_n) tile of `_quantize_amax_colsum_kernel`, bitwise.

        Same tile -> partial mapping, per-element arithmetic and fp32
        accumulate order as that kernel. Four row-blocks of loads are
        issued before they are consumed in row order (memory-level
        parallelism only), and the E5M2 stores are streaming (.cs): the
        payload is read back only by the wgrad/dgrad GEMMs.
        """
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        amax = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        row0 = pid_m.to(tl.int64) * (ROW_ITERS * BLOCK_M)
        rows = row0 + tl.arange(0, BLOCK_M).to(tl.int64)
        offs = rows[:, None] * N + cols[None, :]
        for _ in range(ROW_ITERS // 4):
            x0 = tl.load(SRC + offs, mask=rows[:, None] < n_rows, other=0.0)
            x1 = tl.load(
                SRC + offs + BLOCK_M * N,
                mask=(rows + BLOCK_M)[:, None] < n_rows,
                other=0.0,
            )
            x2 = tl.load(
                SRC + offs + 2 * BLOCK_M * N,
                mask=(rows + 2 * BLOCK_M)[:, None] < n_rows,
                other=0.0,
            )
            x3 = tl.load(
                SRC + offs + 3 * BLOCK_M * N,
                mask=(rows + 3 * BLOCK_M)[:, None] < n_rows,
                other=0.0,
            )
            xf = x0.to(tl.float32)
            acc += xf
            amax = tl.maximum(amax, tl.abs(xf))
            tl.store(
                DST + offs,
                tl.minimum(tl.maximum(xf * inv, -LIMIT), LIMIT).to(
                    DST.dtype.element_ty
                ),
                mask=rows[:, None] < n_rows_out,
                cache_modifier=".cs",
            )
            xf = x1.to(tl.float32)
            acc += xf
            amax = tl.maximum(amax, tl.abs(xf))
            tl.store(
                DST + offs + BLOCK_M * N,
                tl.minimum(tl.maximum(xf * inv, -LIMIT), LIMIT).to(
                    DST.dtype.element_ty
                ),
                mask=(rows + BLOCK_M)[:, None] < n_rows_out,
                cache_modifier=".cs",
            )
            xf = x2.to(tl.float32)
            acc += xf
            amax = tl.maximum(amax, tl.abs(xf))
            tl.store(
                DST + offs + 2 * BLOCK_M * N,
                tl.minimum(tl.maximum(xf * inv, -LIMIT), LIMIT).to(
                    DST.dtype.element_ty
                ),
                mask=(rows + 2 * BLOCK_M)[:, None] < n_rows_out,
                cache_modifier=".cs",
            )
            xf = x3.to(tl.float32)
            acc += xf
            amax = tl.maximum(amax, tl.abs(xf))
            tl.store(
                DST + offs + 3 * BLOCK_M * N,
                tl.minimum(tl.maximum(xf * inv, -LIMIT), LIMIT).to(
                    DST.dtype.element_ty
                ),
                mask=(rows + 3 * BLOCK_M)[:, None] < n_rows_out,
                cache_modifier=".cs",
            )
            rows += 4 * BLOCK_M
            offs += 4 * BLOCK_M * N
        if row0 < n_rows:
            tl.store(
                PARTIAL_COL + pid_m.to(tl.int64) * N + cols, tl.sum(acc, axis=0)
            )
            tl.store(
                PARTIAL_AMAX + pid_m * n_cb + pid_n,
                tl.max(tl.max(amax, axis=1), axis=0),
            )

    @triton.jit
    def _quantize_amax_colsum_v2_kernel(
        SRC,
        DST,
        PARTIAL_AMAX,
        PARTIAL_COL,
        INV_SCALE,
        n_rows,
        n_rows_out,
        n_tiles_m,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        ROW_ITERS: tl.constexpr,
        LIMIT: tl.constexpr,
        XCD_GROUP: tl.constexpr,
    ):
        """`_quantize_amax_colsum_kernel` on a 1-D, XCD-grouped launch (bitwise).

        DLRM_HSTU_FP8_BWD_COLSUM_V2=1 (default off). Workgroups are dispatched
        round-robin over the 8 XCDs, so programs k, k+8, ..., k+8*(XCD_GROUP-1)
        (one XCD) take XCD_GROUP consecutive (row-stripe, column-block) tiles,
        column block fastest. Programs past the last tile do nothing. The
        tile -> partial mapping and the per-tile arithmetic are those of the
        2-D-grid kernel, so the E5M2 payload, amax and colsum partials are bit
        for bit the same.
        """
        tl.static_assert(ROW_ITERS % 4 == 0)
        k = tl.program_id(0)
        xcd = k % 8
        r = k // 8
        t = ((r // XCD_GROUP) * 8 + xcd) * XCD_GROUP + (r % XCD_GROUP)
        n_cb = N // BLOCK_N
        if t < n_tiles_m * n_cb:
            inv = tl.load(INV_SCALE)
            _quantize_amax_colsum_v2_tile(
                SRC,
                DST,
                PARTIAL_AMAX,
                PARTIAL_COL,
                inv,
                t // n_cb,
                t % n_cb,
                n_rows,
                n_rows_out,
                n_cb,
                N,
                BLOCK_M,
                BLOCK_N,
                ROW_ITERS,
                LIMIT,
            )


_COLSUM_BLOCK_M: int = 32
_COLSUM_BLOCK_N: int = 256
_COLSUM_ROW_ITERS: int = 16
# DLRM_HSTU_FP8_BWD_COLSUM_V2=1 (default off): the cast
# above launches `_quantize_amax_colsum_v2_kernel` instead -- same tiles, same
# BLOCK_M/BLOCK_N/ROW_ITERS and num_warps, bitwise payload/amax/colsum. Read
# here at import, checked by `quantize_amax_colsum` at call time.
_COLSUM_V2: bool = os.environ.get("DLRM_HSTU_FP8_BWD_COLSUM_V2", "0") == "1"
# Programs per XCD that take consecutive tiles (v2 only).
_COLSUM_V2_XCD_GROUP: int = 2


def quantize_amax_colsum(
    value: torch.Tensor,
    inv_scale: torch.Tensor,
    pad_to: int = 1,
    out: Optional[torch.Tensor] = None,
    return_partial_amax: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One read of a row-major [M, N] gradient -> (E5M2 copy, amax, colsum).

    `colsum` is the fp32 sum over rows of the unquantized input: same math as
    `torch.sum(value, 0)`, different fp32 reduction order.

    The E5M2 copy is [Mp, N] row-major with Mp = roundup(M, pad_to) (default
    pad_to=1: Mp = M); rows >= M are zero, rows < M and amax/colsum are
    bitwise identical to the unpadded call. `out`, if given, must be a
    contiguous [Mp', N] tensor of the gradient FP8 dtype with Mp' >= M; it is
    filled the same way (rows >= M zeroed) and returned. Slice `[:M]` for the
    dgrad.
    """
    if not _TRITON_AVAILABLE or _GRAD_DTYPE is None:
        raise RuntimeError("Triton FP8 cast requested without Triton")
    assert value.dim() == 2 and value.is_contiguous()
    m, n = value.shape
    assert n % _COLSUM_BLOCK_N == 0
    if out is None:
        assert pad_to >= 1
        m_out = (m + pad_to - 1) // pad_to * pad_to
        out = torch.empty((m_out, n), dtype=_GRAD_DTYPE, device=value.device)
    else:
        assert out.dim() == 2 and out.is_contiguous()
        assert out.dtype == _GRAD_DTYPE and out.shape[1] == n
        assert out.shape[0] >= m and out.device == value.device
        m_out = out.shape[0]
    rows_per_prog = _COLSUM_BLOCK_M * _COLSUM_ROW_ITERS
    # Partials are sized by the INPUT rows so the reductions see exactly the
    # same operands as the unpadded call; extra programs only write zeros.
    grid_m = max(1, (m + rows_per_prog - 1) // rows_per_prog)
    grid_m_launch = max(grid_m, (m_out + rows_per_prog - 1) // rows_per_prog)
    grid_n = n // _COLSUM_BLOCK_N
    partial_amax = torch.empty(
        grid_m * grid_n, dtype=torch.float32, device=value.device
    )
    partial_col = torch.empty((grid_m, n), dtype=torch.float32, device=value.device)
    if m == 0:
        partial_amax.zero_()
        partial_col.zero_()
    if _COLSUM_V2:
        # Same tiles on a 1-D grid rounded up to whole XCD groups.
        group = 8 * _COLSUM_V2_XCD_GROUP
        n_programs = (grid_m_launch * grid_n + group - 1) // group * group
        _quantize_amax_colsum_v2_kernel[(n_programs,)](
            value,
            out,
            partial_amax,
            partial_col,
            inv_scale,
            m,
            m_out,
            grid_m_launch,
            N=n,
            BLOCK_M=_COLSUM_BLOCK_M,
            BLOCK_N=_COLSUM_BLOCK_N,
            ROW_ITERS=_COLSUM_ROW_ITERS,
            LIMIT=_GRAD_MAX,
            XCD_GROUP=_COLSUM_V2_XCD_GROUP,
            num_warps=8,
        )
        log_armed_v2(
            "colsum_v2",
            value.device,
            lambda: (
                f"ARMED colsum_v2 on {value.device}: "
                f"_quantize_amax_colsum_v2_kernel (1-D XCD-grouped launch, "
                f"XCD_GROUP={_COLSUM_V2_XCD_GROUP}, 4 row-blocks in flight, "
                f".cs e5m2 stores; payload/amax/colsum bitwise vs "
                f"_quantize_amax_colsum_kernel), M={m} N={n} rows_out={m_out}"
            ),
        )
    else:
        _quantize_amax_colsum_kernel[(grid_m_launch, grid_n)](
            value,
            out,
            partial_amax,
            partial_col,
            inv_scale,
            m,
            m_out,
            N=n,
            BLOCK_M=_COLSUM_BLOCK_M,
            BLOCK_N=_COLSUM_BLOCK_N,
            ROW_ITERS=_COLSUM_ROW_ITERS,
            LIMIT=_GRAD_MAX,
            num_warps=8,
        )
    if return_partial_amax:
        # DLRM_HSTU_FP8_BWD_HELPERS_FUSED: the site kernel reduces the amax
        # partials; the colsum reduction is untouched.
        return out, partial_amax, partial_col.sum(dim=0)
    return out, partial_amax.amax(), partial_col.sum(dim=0)


if _TRITON_AVAILABLE:

    @triton.jit
    def _quantize_amax_colsum_sel_kernel(
        SRC,
        DST,
        PARTIAL_AMAX,
        PARTIAL_COL,
        INV_SCALE,
        n_rows,
        n_rows_out,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        ROW_ITERS: tl.constexpr,
        LIMIT: tl.constexpr,
        GW: tl.constexpr,
        GS: tl.constexpr,
    ):
        """`_quantize_amax_colsum_kernel` verbatim over a column keep-list.

        Program (pid_m, pid_n) owns column block (pid_n // GW) * GS + pid_n % GW
        (u|q of u|v|q|k: GW = H*D/BLOCK_N, GS = 2*GW, grid_n = 2*GW).
        Per element and per (program, block) partial the arithmetic is that
        of the full cast, so the kept columns' bytes, amax and column partials
        are identical to `quantize_amax_colsum`.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pn = (pid_n // GW) * GS + pid_n % GW
        cols = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        inv = tl.load(INV_SCALE)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        amax = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        row0 = pid_m.to(tl.int64) * (ROW_ITERS * BLOCK_M)
        rows = row0 + tl.arange(0, BLOCK_M).to(tl.int64)
        offs = rows[:, None] * N + cols[None, :]
        for _ in range(ROW_ITERS):
            mask = rows[:, None] < n_rows
            x = tl.load(SRC + offs, mask=mask, other=0.0).to(tl.float32)
            acc += x
            amax = tl.maximum(amax, tl.abs(x))
            scaled = tl.minimum(tl.maximum(x * inv, -LIMIT), LIMIT)
            tl.store(
                DST + offs,
                scaled.to(DST.dtype.element_ty),
                mask=rows[:, None] < n_rows_out,
            )
            rows += BLOCK_M
            offs += BLOCK_M * N
        if row0 < n_rows:
            tl.store(
                PARTIAL_COL + pid_m.to(tl.int64) * N + cols, tl.sum(acc, axis=0)
            )
            tl.store(
                PARTIAL_AMAX + pid_m * tl.num_programs(1) + pid_n,
                tl.max(tl.max(amax, axis=1), axis=0),
            )


def quantize_amax_colsum_select(
    value: torch.Tensor,
    inv_scale: torch.Tensor,
    out: torch.Tensor,
    group_cols: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """`quantize_amax_colsum(value, inv_scale, out=out)` restricted
    to the u and q column groups of a u|v|q|k dz ([0, group_cols) and
    [2*group_cols, 3*group_cols), group_cols = H*D); the producer (the
    bwd_dkdv e5m2 epilogue) owns the v and k columns. (With the dq sink also
    engaged the caller casts u only with hstu_dq_e5m2.quantize_amax_colsum_skip.)

    Writes only those columns of `out` (rows >= M zeroed there, like the full
    cast). Returns (amax over the kept columns, colsum [N] fp32 whose kept
    columns equal the full cast's -- the other columns are UNDEFINED and must
    be replaced).
    """
    if not _TRITON_AVAILABLE or _GRAD_DTYPE is None:
        raise RuntimeError("Triton FP8 cast requested without Triton")
    assert value.dim() == 2 and value.is_contiguous()
    m, n = value.shape
    assert group_cols % _COLSUM_BLOCK_N == 0 and n == 4 * group_cols
    assert out.dim() == 2 and out.is_contiguous() and out.dtype == _GRAD_DTYPE
    assert out.shape[1] == n and out.shape[0] >= m and out.device == value.device
    gw = group_cols // _COLSUM_BLOCK_N
    rows_per_prog = _COLSUM_BLOCK_M * _COLSUM_ROW_ITERS
    grid_m = max(1, (m + rows_per_prog - 1) // rows_per_prog)
    grid_m_launch = max(grid_m, (out.shape[0] + rows_per_prog - 1) // rows_per_prog)
    grid_n = 2 * gw
    partial_amax = torch.empty(
        grid_m * grid_n, dtype=torch.float32, device=value.device
    )
    # Same [grid_m, N] partial layout (and so the same column reduction) as
    # the full cast; the columns not cast here stay uninitialised.
    partial_col = torch.empty((grid_m, n), dtype=torch.float32, device=value.device)
    if m == 0:
        partial_amax.zero_()
        partial_col.zero_()
    _quantize_amax_colsum_sel_kernel[(grid_m_launch, grid_n)](
        value,
        out,
        partial_amax,
        partial_col,
        inv_scale,
        m,
        out.shape[0],
        N=n,
        BLOCK_M=_COLSUM_BLOCK_M,
        BLOCK_N=_COLSUM_BLOCK_N,
        ROW_ITERS=_COLSUM_ROW_ITERS,
        LIMIT=_GRAD_MAX,
        GW=gw,
        GS=2 * gw,
        num_warps=8,
    )
    return partial_amax.amax(), partial_col.sum(dim=0)


def quantize_row_major(
    value: torch.Tensor,
    inv_scale: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    limit: Optional[float] = None,
    return_partial: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cast to a row-major FP8 format in one pass, returning it and its amax.

    Defaults to the gradient format; `quantize_weight` asks for E4M3 instead.
    The amax is of the *input*, computed before scaling, so it is the statistic
    the next refresh needs and is unaffected by the scale currently in use.
    """
    dtype = _GRAD_DTYPE if dtype is None else dtype
    limit = _GRAD_MAX if limit is None else limit
    if not _TRITON_AVAILABLE or dtype is None:
        raise RuntimeError("Triton FP8 cast requested without Triton")
    source = value.detach()
    if not source.is_contiguous():
        source = source.contiguous()
    flat = source.reshape(-1)
    n = flat.numel()
    out = torch.empty(n, dtype=dtype, device=source.device)
    grid = (n + _CAST_BLOCK - 1) // _CAST_BLOCK
    partial = torch.empty(grid, dtype=torch.float32, device=source.device)
    _quantize_amax_kernel[(grid,)](
        flat,
        out,
        partial,
        inv_scale,
        n,
        BLOCK=_CAST_BLOCK,
        LIMIT=limit,
        num_warps=8,
    )
    if return_partial:
        return out.view_as(source), partial
    return out.view_as(source), partial.amax()


class _SiteScale:
    """One backward tensor's scale, held and published entirely on device.

    `scale` is the dequantization scale in the forward arm's convention:
    multiply an E4M3 value by it to recover the original. `inv_scale` is its
    reciprocal, which is what the cast kernel multiplies by, kept alongside so
    the steady-state path never divides.
    """

    def __init__(self, site: str, device: torch.device) -> None:
        self.site = site
        self.calls = 0
        self.scale = torch.zeros(1, dtype=torch.float32, device=device)
        self.inv_scale = torch.zeros(1, dtype=torch.float32, device=device)
        self.pending = torch.zeros((), dtype=torch.float32, device=device)
        # Incremented on device whenever a window's observed amax exceeded what
        # the scale in use could represent. Free, because that amax is already
        # computed; read only when something asks, never in the step.
        self.clipped_windows = torch.zeros((), dtype=torch.int64, device=device)
        # Same, for a token-major e4m3 copy cast with less headroom than the
        # row-major operand (DLRM_HSTU_FP8_BWD_OUT_WGRAD_DFMT_SHIFT < 7 or auto).
        # Allocated only when the FP8 output wgrad is armed, so the default
        # path launches nothing extra at site creation.
        self.t_clipped_windows: Optional[torch.Tensor] = (
            torch.zeros((), dtype=torch.int64, device=device)
            if _OUT_WGRAD_ARMED
            else None
        )
        # Shift of the token-major e4m3 copy; moves only under SHIFT=auto.
        self.t_shift = _OUT_WGRAD_DFMT_SHIFT
        self.t_calls = 0
        self._t_clipped_seen = 0
        self._t_clean_windows = 0
        self.published = False

    def accumulate(self, amax: torch.Tensor) -> None:
        torch.maximum(self.pending, amax, out=self.pending)

    def publish(self) -> None:
        """Turn the window's amax into the scale the next window will use.

        Clamped away from zero because a window whose gradient was entirely zero
        would otherwise publish a scale of zero and make every later value
        infinite, which is unrecoverable.
        """
        self.scale.copy_(
            (self.pending.clamp_min(1.0e-30) * (_BWD_MARGIN / _GRAD_MAX)).reshape(1)
        )
        self.inv_scale.copy_(1.0 / self.scale)
        self.pending.zero_()
        first = not self.published
        self.published = True
        # The DFMT_SHIFT=auto controller runs before the logging gate
        # below, so SHIFT=auto still adapts when SCALE_LOG is off. Its host
        # read happens only under DLRM_HSTU_FP8_BWD_OUT_WGRAD_DFMT_SHIFT=auto
        # with the token-major e4m3 copy in use (t_calls > 0), once every
        # _BWD_REFRESH_CALLS steps; the default path adds no device read.
        t_shift_used = self.t_shift
        if (
            _OUT_WGRAD_DFMT_SHIFT_AUTO
            and self.t_calls > 0
            and self.t_clipped_windows is not None
        ):
            t_clipped = int(self.t_clipped_windows.item())
            self.t_shift, self._t_clean_windows = next_t_shift(
                self.t_shift,
                t_clipped - self._t_clipped_seen,
                self._t_clean_windows,
                _OUT_WGRAD_DFMT_SHIFT_FLOOR,
                _OUT_WGRAD_DFMT_SHIFT_RELAX,
            )
            self._t_clipped_seen = t_clipped
        # One host read every _BWD_REFRESH_CALLS steps, not in the step. Off by
        # default (submission logging profile): the gradient amax growth it shows
        # is still available on demand with DLRM_HSTU_FP8_BWD_SCALE_LOG=1 or
        # through site_diagnostics().
        if not _BWD_SCALE_LOG:
            return
        if self.t_clipped_windows is None:
            # FP8 output wgrad not armed: the default log line.
            logger.info(
                "[hstu-fp8-bwd] scale %s: site=%s call=%d scale=%.6e clipped_windows=%d",
                "published" if first else "refreshed",
                self.site,
                self.calls,
                float(self.scale.item()),
                int(self.clipped_windows.item()),
            )
            return
        t_clipped = int(self.t_clipped_windows.item())
        logger.info(
            "[hstu-fp8-bwd] scale %s: site=%s call=%d scale=%.6e clipped_windows=%d"
            " t_clipped_windows=%d t_shift=%d->%d",
            "published" if first else "refreshed",
            self.site,
            self.calls,
            float(self.scale.item()),
            int(self.clipped_windows.item()),
            t_clipped,
            t_shift_used,
            self.t_shift,
        )

    def note_clipping(self, amax: torch.Tensor) -> None:
        # Anything above scale * FP8_MAX was clamped by the cast. The comparison
        # is exact and costs nothing: both sides are already on device and the
        # amax was computed by the pass that did the clamping.
        self.clipped_windows += (amax > self.scale.reshape(()) * _GRAD_MAX).to(
            torch.int64
        )

    def note_t_clipping(self, amax: torch.Tensor, t_limit: float) -> None:
        # The token-major copy is cast at inv_scale * 2^-shift and clamped to
        # t_limit, so it clipped iff amax * inv_scale * 2^-shift > t_limit.
        if self.t_clipped_windows is None:
            return
        self.t_clipped_windows += (
            amax > self.scale.reshape(()) * (t_limit * 2.0**self.t_shift)
        ).to(torch.int64)


_sites: Dict[str, _SiteScale] = {}
# A site name alone is the wrong key: the model stacks several STU layers and
# every one of them calls the same site, so keying on the name puts different
# layers' gradients behind one scale. That is not a correctness bug -- each
# tensor is dequantized with whatever quantized it -- but it forces the margin
# to cover the spread *between* layers on top of the spread within one.
#
# The discriminator is the weight's storage, which is only a stable identity
# when the weight is the parameter itself. `STU.forward_targets_only` and
# `STU.cached_forward` hand the backward a per-call `.to(dtype)` copy, whose
# address is whatever the caching allocator returns, so keying on it would
# mint a new scale state per address. `_has_stable_identity` restricts this to
# the path where the key means something; see the leaf-parameter note there.
_layer_index: Dict[Tuple[str, int], int] = {}
_engaged_sites: set = set()
_logged_resolved: bool = False


def _log_resolved_once() -> None:
    """Say what the backward arm resolved to, once, in the run's own log.

    The loss curve alone does not show whether the FP8 path executed, so the
    resolved configuration is logged explicitly.
    """
    global _logged_resolved
    if _logged_resolved:
        return
    _logged_resolved = True
    logger.info(
        "[hstu-fp8-bwd] resolved: enabled=%d grad_format=%s margin=%g "
        "refresh=%d warmup=%d triton=%d",
        int(_BWD_ENABLED),
        _GRAD_FORMAT,
        _BWD_MARGIN,
        _BWD_REFRESH_CALLS,
        _BWD_WARMUP_CALLS,
        int(_TRITON_AVAILABLE),
    )


def record_engaged(site: str) -> None:
    if site in _engaged_sites:
        return
    _engaged_sites.add(site)
    _log_resolved_once()
    logger.info("[hstu-fp8-bwd] engaged: %s", site)


def _site_name(site: str, discriminator: Optional[int]) -> str:
    """Name one layer's instance of a site, stably and readably.

    The index is assigned in order of first appearance, which is deterministic
    but is a label rather than the model's own layer numbering -- backward runs
    the stack in reverse. It exists so a log line names something a human can
    follow, not so anything can be inferred from the number.
    """
    if discriminator is None:
        return site
    key = (site, discriminator)
    index = _layer_index.get(key)
    if index is None:
        index = sum(1 for existing in _layer_index if existing[0] == site)
        _layer_index[key] = index
    return f"{site}.L{index}"


def _has_stable_identity(weight: torch.Tensor) -> bool:
    """Whether this weight's storage address is a durable name for one layer.

    True for a parameter, which is a leaf that requires grad and lives for the
    run. False for the `.to(dtype)` copy two of the three call sites build per
    call: it carries a `grad_fn`, so it is not a leaf, and its address belongs
    to the allocator rather than to the layer.
    """
    return weight.is_leaf and weight.requires_grad


def _site_state(site: str, device: torch.device) -> _SiteScale:
    state = _sites.get(site)
    if state is None:
        state = _SiteScale(site, device)
        _sites[site] = state
    return state


def quantize_gradient(
    site: str,
    gradient: torch.Tensor,
    discriminator: Optional[int] = None,
    snapshot_decode_scale: bool = False,
    token_major: bool = False,
    token_major_e4m3: bool = False,
    token_major_tiled: bool = False,
) -> Optional[Tuple[torch.Tensor, ...]]:
    """Quantize a backward gradient under the delayed-scale policy.

    token_major=True (2-D gradients only) also returns the same FP8 bytes laid
    out token-major, `(fp8, decode_scale, fp8_token_major)`, from one fused cast.
    token_major_tiled=True writes that copy step-tiled [ceil(M/128), N, 128]
    (quantize_row_major_dual t_tiled; the custom out-proj wgrad only).

    snapshot_decode_scale=True returns a copy of the scale the payload was
    actually quantized with, taken before a refresh publishes the next one in
    place. The default (False) hands back, on a refresh call, the
    already-refreshed scale.

    Returns `(fp8_gradient, dequantization_scale)`, or `None` while the site is
    still warming up, which is the caller's signal to run BF16. Returning None
    rather than a guessed scale is the fail-closed path: a scale that has never
    seen this tensor is worse than no FP8 at all.

    One cast serves both consumers at a site. The dgrad contracts over N and a
    row-major-reading wgrad tiles the same layout, so callers must not cast
    twice.
    """
    if not scale_controller_enabled():
        return None
    if gradient.numel() == 0 or gradient.dtype not in (
        torch.bfloat16,
        torch.float32,
        torch.float16,
    ):
        return None

    state = _site_state(_site_name(site, discriminator), gradient.device)
    state.calls += 1

    if not state.published:
        # Warmup runs BF16 and pays for a standalone amax, which is bounded to a
        # handful of calls and never recurs.
        state.accumulate(gradient.detach().abs().amax().float())
        if state.calls >= _BWD_WARMUP_CALLS:
            state.publish()
        return None

    t_e4m3 = token_major and token_major_e4m3 and _GRAD_MAX == _E5M2_MAX
    # SHIFT=7 gives 2^-7 == 448 / 57344 exactly (the default e4m3-copy scale).
    t_shift = state.t_shift
    t_inv_mul = 2.0**-t_shift
    if token_major:
        quantized, quantized_t, amax = quantize_row_major_dual(
            gradient,
            state.inv_scale,
            t_dtype=_FP8_E4M3 if t_e4m3 else None,
            t_inv_mul=t_inv_mul if t_e4m3 else 1.0,
            t_limit=_E4M3_MAX if t_e4m3 else None,
            t_tiled=token_major_tiled,
        )
    else:
        quantized, amax = quantize_row_major(gradient, state.inv_scale)
    state.note_clipping(amax)
    if t_e4m3 and (t_shift != 7 or _OUT_WGRAD_DFMT_SHIFT_AUTO):
        state.t_calls += 1
        state.note_t_clipping(amax, _E4M3_MAX)
    state.accumulate(amax)
    decode_scale = state.scale.clone() if snapshot_decode_scale else state.scale
    if state.calls % _BWD_REFRESH_CALLS == 0:
        state.publish()
    if token_major:
        if t_e4m3:
            # Cast at inv_scale * 2^-shift, so decode at scale * 2^shift (exact;
            # shift=7 is 57344 / 448).
            return quantized, decode_scale, quantized_t, decode_scale * (
                2.0**t_shift
            )
        return quantized, decode_scale, quantized_t
    return quantized, decode_scale


def quantize_weight(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cast a weight to E4M3 at unit margin, deriving its scale on device.

    No margin, unlike the gradient: a weight is quantized from the amax of the
    very tensor being quantized, so there is nothing to leave headroom for, and
    every factor of headroom is a bit of mantissa given away. The forward arm
    packs its weights at margin 1.0 for the same reason.

    Small enough that the cast is noise -- the largest here is 1536x512 -- but
    it still goes through the device-side path, because `float(amax)` in a
    backward would be a synchronization per step for a number nothing reads.
    """
    amax = weight.detach().abs().amax().float().clamp_min(1.0e-30)
    scale = (amax / _E4M3_MAX).reshape(1)
    quantized, _ = quantize_row_major(
        weight, 1.0 / scale, dtype=_FP8_E4M3, limit=_E4M3_MAX
    )
    return quantized, scale


# ---------------------------------------------------------------------------
# DEFAULT OFF:
#   DLRM_HSTU_FP8_BWD_HELPERS_FUSED=1
# The per-call scale bookkeeping of a published site (partial-amax reduce,
# note_clipping, note_t_clipping, accumulate, decode-scale snapshot and its
# 2^shift product) runs as program 0 of the weight-amax kernel, and
# quantize_weight becomes that kernel + one cast kernel
# (fp8_helpers_fused). Byte-identical state tensors, payloads and scales; the
# refresh (publish), warm-up and fallback paths are the unfused code, and the
# refresh decision stays the host-side call counter (no extra device reads).
_BWD_HELPERS_FUSED: bool = (
    os.environ.get("DLRM_HSTU_FP8_BWD_HELPERS_FUSED", "0") == "1"
)


def _helpers_fused_ok(weight: torch.Tensor) -> bool:
    return (
        _BWD_HELPERS_FUSED
        and _TRITON_AVAILABLE
        and _FP8_E4M3 is not None
        and weight.is_cuda
        and weight.dim() == 2
        and weight.is_contiguous()
        and weight.dtype in (torch.float32, torch.bfloat16)
    )


def _fused_site_launch(
    state: "_SiteScale",
    partial: torch.Tensor,
    weight: torch.Tensor,
    snapshot: bool,
    t_note: bool,
    t_shift: int,
    snapshot_t: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """One launch: site bookkeeping (program 0) + |weight| tile maxima.

    Returns (weight partials, scale snapshot or None, snapshot * 2^t_shift or
    None). Mirrors note_clipping, [note_t_clipping], accumulate and
    `state.scale.clone()` in that order; the caller publishes afterwards."""
    from generative_recommenders.ops.triton import fp8_helpers_fused

    device = weight.device
    snap = torch.empty(1, dtype=torch.float32, device=device) if snapshot else None
    snap_t = (
        torch.empty(1, dtype=torch.float32, device=device) if snapshot_t else None
    )
    site = {
        "partial": partial.reshape(-1),
        "scale": state.scale,
        "clipped": state.clipped_windows,
        "t_clipped": state.t_clipped_windows if t_note else None,
        "pending": state.pending,
        "snap": snap,
        "snap_t": snap_t,
        "grad_max": _GRAD_MAX,
        "t_factor": _E4M3_MAX * 2.0**t_shift,
        "t_mul": 2.0**t_shift,
    }
    w_partial = fp8_helpers_fused.weight_amax_with_site(weight, site)
    _log_armed(
        state.site.split(".L")[0],
        device,
        "helpers_fused",
        f"ARMED helpers_fused {state.site} on {device}: site bookkeeping + "
        f"weight amax in 1 launch, weight cast in 1 launch "
        f"(partials={partial.numel()}, w={tuple(weight.shape)}:{weight.dtype}); "
        "bitwise vs the shipped helper chain",
    )
    return w_partial, snap, snap_t


def _fused_weight_quant(
    weight: torch.Tensor, w_partial: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    from generative_recommenders.ops.triton import fp8_helpers_fused

    assert _FP8_E4M3 is not None
    return fp8_helpers_fused.weight_quant_from_partials(weight, w_partial, _FP8_E4M3)


def _quantize_gradient_and_weight_fused(
    site: str,
    gradient: torch.Tensor,
    weight: torch.Tensor,
    discriminator: Optional[int],
    snapshot_decode_scale: bool,
    token_major: bool,
    token_major_e4m3: bool,
    token_major_tiled: bool = False,
) -> Tuple[Optional[Tuple[torch.Tensor, ...]], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """`quantize_gradient(...)` then `quantize_weight(weight)` with the helper
    chain fused. Returns (quantize_gradient's result, (weight_fp8, scale) or
    None when the caller must run `quantize_weight` itself)."""
    if not scale_controller_enabled():
        return None, None
    if gradient.numel() == 0 or gradient.dtype not in (
        torch.bfloat16,
        torch.float32,
        torch.float16,
    ):
        return None, None
    state = _site_state(_site_name(site, discriminator), gradient.device)
    if not state.published:
        # Warm-up: the unfused path (it advances calls itself).
        return (
            quantize_gradient(
                site,
                gradient,
                discriminator=discriminator,
                snapshot_decode_scale=snapshot_decode_scale,
                token_major=token_major,
                token_major_e4m3=token_major_e4m3,
                token_major_tiled=token_major_tiled,
            ),
            None,
        )
    state.calls += 1
    t_e4m3 = token_major and token_major_e4m3 and _GRAD_MAX == _E5M2_MAX
    t_shift = state.t_shift
    t_inv_mul = 2.0**-t_shift
    if token_major:
        quantized, quantized_t, partial = quantize_row_major_dual(
            gradient,
            state.inv_scale,
            t_dtype=_FP8_E4M3 if t_e4m3 else None,
            t_inv_mul=t_inv_mul if t_e4m3 else 1.0,
            t_limit=_E4M3_MAX if t_e4m3 else None,
            return_partial=True,
            t_tiled=token_major_tiled,
        )
    else:
        quantized, partial = quantize_row_major(
            gradient, state.inv_scale, return_partial=True
        )
    t_note = False
    if t_e4m3 and (t_shift != 7 or _OUT_WGRAD_DFMT_SHIFT_AUTO):
        state.t_calls += 1
        t_note = state.t_clipped_windows is not None
    w_partial, snap, snap_t = _fused_site_launch(
        state,
        partial,
        weight,
        snapshot=snapshot_decode_scale,
        t_note=t_note,
        t_shift=t_shift,
        snapshot_t=token_major and t_e4m3 and snapshot_decode_scale,
    )
    decode_scale = snap if snap is not None else state.scale
    if state.calls % _BWD_REFRESH_CALLS == 0:
        state.publish()
    if token_major:
        if t_e4m3:
            decode_t = (
                snap_t if snap_t is not None else decode_scale * (2.0**t_shift)
            )
            result: Tuple[torch.Tensor, ...] = (
                quantized, decode_scale, quantized_t, decode_t
            )
        else:
            result = (quantized, decode_scale, quantized_t)
    else:
        result = (quantized, decode_scale)
    if gradient.shape[-1] % 16 != 0:
        # The caller declines this shape before quantize_weight; launch nothing.
        return result, None
    return result, _fused_weight_quant(weight, w_partial)


def output_projection_dgrad(
    gradient: torch.Tensor, weight: torch.Tensor
) -> Optional[torch.Tensor]:
    """`dy = dout @ weight^T` in FP8, or None to fall back to BF16.

    The only site in the arm. It needs no epilogue work: the gradient is
    already row-major, which is the layout `_scaled_mm` wants. The wgrads
    would need both operands re-laid-out along K.
    """
    if not _BWD_ENABLED:
        return None
    if not _has_stable_identity(weight):
        return None
    fused_weight = None
    if _helpers_fused_ok(weight):
        quantized, fused_weight = _quantize_gradient_and_weight_fused(
            "output_projection",
            gradient,
            weight,
            discriminator=weight.data_ptr(),
            snapshot_decode_scale=False,
            token_major=False,
            token_major_e4m3=False,
        )
    else:
        quantized = quantize_gradient(
            "output_projection", gradient, discriminator=weight.data_ptr()
        )
    if quantized is None:
        return None
    gradient_fp8, gradient_scale = quantized
    # `_scaled_mm` rejects a mat1 whose trailing dimension is not a multiple of
    # 16. Here that is the projection's output dim, which is 512, but a shape
    # that silently fell back would be worse than one that never engaged.
    if gradient_fp8.shape[-1] % 16 != 0:
        return None
    if fused_weight is not None:
        weight_fp8, weight_scale = fused_weight
    else:
        weight_fp8, weight_scale = quantize_weight(weight)
    record_engaged("output_projection_dgrad")
    return torch._scaled_mm(
        gradient_fp8,
        weight_fp8.t(),
        scale_a=gradient_scale,
        scale_b=weight_scale,
        out_dtype=gradient.dtype,
    )


def output_projection_dgrad_with_operand(
    gradient: torch.Tensor,
    weight: torch.Tensor,
    token_major: bool = False,
    t_tiled: bool = False,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """`output_projection_dgrad`, also returning the wgrad's gradient operand and
    its own decode scale (the e5m2 bytes and scale, or with
    DLRM_HSTU_FP8_BWD_OUT_WGRAD_DFMT=e4m3 the token-major e4m3 copy at scale*2^7)
    so the output-projection wgrad can reuse the same cast (one cast serves
    both consumers). Only the FP8 output-wgrad path calls this
    (DLRM_HSTU_FP8_BWD_OUT_WGRAD, default off). Unlike `output_projection_dgrad`
    it decodes with the scale snapshot taken before a refresh, so on refresh
    calls its dgrad differs from `output_projection_dgrad` (correctly).

    t_tiled (the caller passes wgrad_ck_w3.tiled_active()): the
    token-major operand is written step-tiled [ceil(M/128), 512, 128] for the
    custom W3 kernel -- only when it is an e4m3 copy (the kernel's format);
    otherwise the 2-D layout, as without the flag. Same bytes, same scales.
    """
    operands = output_projection_dgrad_operands(
        gradient, weight, token_major, t_tiled=t_tiled
    )
    if operands is None:
        return None
    gradient_fp8, gradient_scale, operand, operand_scale, weight_fp8, weight_scale = (
        operands
    )
    dgrad = output_projection_dgrad_gemm(
        gradient_fp8, weight_fp8, gradient_scale, weight_scale, gradient.dtype
    )
    return dgrad, operand, operand_scale


def output_projection_dgrad_operands(
    gradient: torch.Tensor,
    weight: torch.Tensor,
    token_major: bool = False,
    t_tiled: bool = False,
) -> Optional[
    Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]
]:
    """The cast half of `output_projection_dgrad_with_operand`: the same site,
    discriminator, scale snapshot and scale-state update (including the
    DLRM_HSTU_FP8_BWD_HELPERS_FUSED fused gradient + weight cast), in the same
    order, without the GEMM. Returns `(gradient_fp8 e5m2 [M, K],
    gradient_scale [1], operand, operand_scale, weight_fp8 e4m3 [N, K],
    weight_scale [1])`, or None (warmup, or a contract the dgrad does not take).
    Split out for the fused out-proj dgrad + LN*U*dropout backward path, which
    consumes `(gradient_fp8, weight_fp8, gradient_scale, weight_scale)` instead
    of running `output_projection_dgrad_gemm`;
    `output_projection_dgrad_with_operand` is this plus that GEMM.
    t_tiled: as in `output_projection_dgrad_with_operand` (the token-major
    operand step-tiled for the custom W3 kernel); every caller passes the same
    `_out_wgrad_dout_tiled` decision, so the fused path and the unfused path
    hand W3 the same layout."""
    if not _BWD_ENABLED:
        return None
    if not _has_stable_identity(weight):
        return None
    # The token-major copy is e4m3 iff DFMT=e4m3 with e5m2 grads, or e4m3 grads.
    t_tiled = (
        t_tiled
        and token_major
        and (_OUT_WGRAD_T_E4M3 or _GRAD_DTYPE == _FP8_E4M3)
    )
    fused_weight = None
    if _helpers_fused_ok(weight):
        quantized, fused_weight = _quantize_gradient_and_weight_fused(
            "output_projection",
            gradient,
            weight,
            discriminator=weight.data_ptr(),
            snapshot_decode_scale=True,
            token_major=token_major,
            token_major_e4m3=_OUT_WGRAD_T_E4M3,
            token_major_tiled=t_tiled,
        )
    else:
        quantized = quantize_gradient(
            "output_projection",
            gradient,
            discriminator=weight.data_ptr(),
            snapshot_decode_scale=True,
            token_major=token_major,
            token_major_e4m3=_OUT_WGRAD_T_E4M3,
            token_major_tiled=t_tiled,
        )
    if quantized is None:
        return None
    gradient_fp8, gradient_scale = quantized[0], quantized[1]
    operand = quantized[2] if token_major else gradient_fp8
    # The wgrad operand's own decode scale (differs only for an e4m3 copy).
    operand_scale = quantized[3] if len(quantized) > 3 else gradient_scale
    if gradient_fp8.shape[-1] % 16 != 0:
        return None
    if fused_weight is not None:
        weight_fp8, weight_scale = fused_weight
    else:
        weight_fp8, weight_scale = quantize_weight(weight)
    record_engaged("output_projection_dgrad")
    return (
        gradient_fp8,
        gradient_scale,
        operand,
        operand_scale,
        weight_fp8,
        weight_scale,
    )


def output_projection_dgrad_gemm(
    gradient_fp8: torch.Tensor,
    weight_fp8: torch.Tensor,
    gradient_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """The GEMM half: `dy = (gradient_fp8 * gradient_scale) @ (weight_fp8 *
    weight_scale)^T` in `out_dtype` (hipBLASLt via torch._scaled_mm)."""
    return torch._scaled_mm(
        gradient_fp8,
        weight_fp8.t(),
        scale_a=gradient_scale,
        scale_b=weight_scale,
        out_dtype=out_dtype,
    )


# ---------------------------------------------------------------------------
# FP8 uvqk / vk projection backward. DEFAULT OFF.
#
# A precision change, independent of DLRM_HSTU_FP8_BWD (the output-projection
# dgrad above): every knob below is absent = off, and with them off nothing in
# this section runs, so the default triton_addmm_bwd path is untouched.
#
#   DLRM_HSTU_FP8_BWD_UVQK=1             E5M2 dgrad of the uvqk projection of
#                                        every ordinary STU layer and of the
#                                        targets-only layer's VK projection.
#   DLRM_HSTU_FP8_BWD_UVQK_WGRAD=fp8t    (needs UVQK=1) the weight gradient of
#                                        those projections in FP8 too.
#   DLRM_HSTU_FP8_BWD_UVQK_STORE_T=1     (needs WGRAD=fp8t) the forward LN also
#                                        stores the token-major E4M3 normed_x.
#   DLRM_HSTU_FP8_BWD_VK_PRODUCER=1      (needs WGRAD=fp8t) the targets-only
#                                        delta attention backward emits the
#                                        E5M2 dV|dK payload directly.
#
# Operand delivery, with no transposed copy of the gradient anywhere:
#   * d(uvqk) [M, N] bf16 is read ONCE by `quantize_amax_colsum`, which writes
#     the row-major E5M2 copy, the amax for the delayed-scale controller, and
#     the fp32 bias-gradient column sum (replacing the separate colsum pass).
#   * dgrad  d_normed_x = dz @ W^T : torch._scaled_mm, TN, as output_projection.
#   * wgrad  dW = x^T @ dz          : DLRM_HSTU_FP8_BWD_UVQK_WGRAD selects
#       bf16 (default) BF16 torch.mm on the BF16 normed_x (only the dgrad is FP8).
#       fp8t  e5m2 x e4m3 with a TOKEN-MAJOR activation. When this call's FP8
#             path will run (`uvqk_fp8t_will_run`: armed, scale published, not
#             broken -- no side effects), the caller provides ONLY
#             e4m3(normed_x / input_scale) token-major [K, Mp], Mp =
#             roundup(M, 128), zero tail (recomputed by
#             `triton_weighted_layer_norm_recompute_fp8t`, or stored by the
#             forward LN under STORE_T), in place of the BF16 normed_x. The
#             gradient stays the row-major E5M2 payload above, cast with
#             pad_to=128 (zero tail rows; the dgrad reads exactly its first M
#             rows). The hipBLASLt NN problem
#             C'[N,K] = dz[N,M](op N) @ xT[M,K](op N), ld Mp, k_tokens=M
#             returns dW [K, N] fp32 directly (extension
#             hipblaslt_fp8_wgrad_tm.cpp). The first call per (device, K, N)
#             resolves a concrete solution index (DLRM_HSTU_FP8_BWD_UVQK_WGRAD_
#             SOLUTION if set, else the heuristic top-1) and checks that exact
#             kernel at the full production K against an exact-product,
#             fp64-summed matmul of the same FP8 operands; later calls run that
#             index strictly (no per-call heuristic query). A pin that fails
#             falls back to the heuristic, and a heuristic that fails disables
#             fp8t (BF16 wgrad from then on), as does an unsupported problem at
#             a later M. Warmup calls, a declined call and any failure run the
#             BF16 wgrad; a call that declines AFTER the caller skipped the BF16
#             normed_x recomputes it through the caller's `recompute_normed_x`
#             hook.
#
# Numerics (declared): E5M2 d(uvqk) under the delayed-scale controller above
# feeds the dgrad (and, with fp8t, the wgrad together with the e4m3 forward
# activation, fp32 accumulate); the bias gradient is an fp32 column sum of the
# unquantized gradient in a different reduction order. The VK producer is
# byte-identical in the FP8 payload to casting the BF16 dV|dK it replaces.
_UVQK_ENABLED: bool = os.environ.get("DLRM_HSTU_FP8_BWD_UVQK", "0") == "1"
_UVQK_WGRAD_RAW: str = (
    os.environ.get("DLRM_HSTU_FP8_BWD_UVQK_WGRAD", "bf16").strip().lower() or "bf16"
)
# Only "bf16" and "fp8t" are supported; anything else runs the BF16 wgrad and
# says so once below.
_UVQK_WGRAD: str = _UVQK_WGRAD_RAW if _UVQK_WGRAD_RAW in ("bf16", "fp8t") else "bf16"
# Optional pinned hipBLASLt solution index for the fp8t wgrad (-1: heuristic
# top-1). Guarded like the heuristic choice.
_UVQK_WGRAD_SOLUTION: int = int(
    os.environ.get("DLRM_HSTU_FP8_BWD_UVQK_WGRAD_SOLUTION", "-1")
)
# Token padding of the token-major activation and of the E5M2 gradient rows.
_UVQK_TM_PAD: int = 128
# Relative-L2 tolerance of the first-call fp8t guard against the exact-product
# reference: only the fp32 accumulation order may differ.
_UVQK_TM_GUARD_TOL: float = 1.0e-4
_UVQK_STORE_T: bool = os.environ.get("DLRM_HSTU_FP8_BWD_UVQK_STORE_T", "0") == "1"
_VK_PRODUCER: bool = os.environ.get("DLRM_HSTU_FP8_BWD_VK_PRODUCER", "0") == "1"
_uvqk_scale_tensors: Dict[Tuple[int, float], torch.Tensor] = {}
_uvqk_broken: Optional[str] = None
_uvqk_tm_broken: Optional[str] = None
# (device index, K, N) -> (armed solution index, human label)
_uvqk_tm_armed: Dict[Tuple[int, int, int], Tuple[int, str]] = {}
_uvqk_armed_logged: set = set()
_uvqk_once_logged: set = set()


def _uvqk_print(message: str) -> None:
    print(f"[hstu-fp8-bwd-uvqk] {message}", flush=True)


if _UVQK_ENABLED or _UVQK_STORE_T or _VK_PRODUCER or _UVQK_WGRAD_RAW != "bf16":
    # Startup record of what this process resolved, so a run can prove the
    # knobs reached it (and an inert combination is visible, not silent).
    _uvqk_print(
        f"requested: uvqk={int(_UVQK_ENABLED)} wgrad={_UVQK_WGRAD}"
        f"{'' if _UVQK_WGRAD == _UVQK_WGRAD_RAW else f' (unsupported {_UVQK_WGRAD_RAW!r} -> bf16)'}"
        f" store_t={int(_UVQK_STORE_T)} vk_producer={int(_VK_PRODUCER)}"
        f" wgrad_solution={_UVQK_WGRAD_SOLUTION}"
        + (
            ""
            if _UVQK_ENABLED
            else " -> INERT (DLRM_HSTU_FP8_BWD_UVQK=1 is the switch)"
        )
        + (
            " -> store_t/vk_producer INERT without WGRAD=fp8t"
            if (_UVQK_STORE_T or _VK_PRODUCER) and _UVQK_WGRAD != "fp8t"
            else ""
        )
    )

# v2 kernels, each DEFAULT OFF and owned by the module that
# launches it (read there at import, checked there at call time):
#   DLRM_HSTU_FP8_BWD_COLSUM_V2=1   quantize_amax_colsum (this module), bitwise
#   DLRM_HSTU_FP8_BWD_STORE_T_V2=1  triton_layer_norm store_t LN, bitwise
#   DLRM_HSTU_FP8_BWD_VKP_V2=1      hstu_delta_bwd_e5m2 VK producer; payload,
#                                   amax and colsum bitwise, dq fp32-order
#                                   (DECLARED)
# This line only records what the process requested, like the one above.
_STORE_T_V2_REQUESTED: bool = (
    os.environ.get("DLRM_HSTU_FP8_BWD_STORE_T_V2", "0") == "1"
)
_VKP_V2_REQUESTED: bool = os.environ.get("DLRM_HSTU_FP8_BWD_VKP_V2", "0") == "1"
if _COLSUM_V2 or _STORE_T_V2_REQUESTED or _VKP_V2_REQUESTED:
    _uvqk_print(
        f"requested v2 kernels: colsum_v2={int(_COLSUM_V2)} "
        f"store_t_v2={int(_STORE_T_V2_REQUESTED)} vkp_v2={int(_VKP_V2_REQUESTED)}"
        + (
            " -> colsum_v2 INERT without DLRM_HSTU_FP8_BWD_UVQK=1"
            if _COLSUM_V2 and not _UVQK_ENABLED
            else ""
        )
        + (
            " -> store_t_v2 INERT without UVQK=1 WGRAD=fp8t UVQK_STORE_T=1"
            if _STORE_T_V2_REQUESTED
            and not (_UVQK_ENABLED and _UVQK_WGRAD == "fp8t" and _UVQK_STORE_T)
            else ""
        )
        + (
            " -> vkp_v2 INERT without UVQK=1 WGRAD=fp8t VK_PRODUCER=1"
            if _VKP_V2_REQUESTED
            and not (_UVQK_ENABLED and _UVQK_WGRAD == "fp8t" and _VK_PRODUCER)
            else ""
        )
    )

# DEFAULT OFF:
#   DLRM_HSTU_FP8_BWD_DQ_E5M2=1  the ordinary layer's dq-split emits the e5m2
#                                q columns of dz (+ amax / colsum partials) and
#                                the cast pass covers u|v|k only
#                                (hstu_dq_e5m2.py); payload and amax bitwise,
#                                q-column bias colsum fp32-order (DECLARED).
# Checked at call time by the ordinary backward (dq_producer_enabled).
_DQ_E5M2: bool = os.environ.get("DLRM_HSTU_FP8_BWD_DQ_E5M2", "0") == "1"
if _DQ_E5M2:
    _uvqk_print(
        "requested dq e5m2 producer: dq_e5m2=1"
        + (
            " -> dq_e5m2 INERT without UVQK=1 WGRAD=fp8t"
            if not (_UVQK_ENABLED and _UVQK_WGRAD == "fp8t")
            else ""
        )
        + (
            " -> dq_e5m2 INERT without the C8 dq-split "
            "(DLRM_HSTU_FP8_BWD_DQ_SPLIT=1 + a DQ_SPLIT_SCHED)"
            if os.environ.get("DLRM_HSTU_FP8_BWD_DQ_SPLIT", "").strip().lower()
            not in ("1", "true", "on", "yes")
            else ""
        )
    )

# DEFAULT OFF:
#   DLRM_HSTU_FP8_BWD_DU_E5M2=1  the ordinary layer's LN*U*dropout backward
#                                (triton_hstu_linear) emits the e5m2 u columns
#                                of dz (+ amax / u colsum) straight into the
#                                layer's dz8 and writes no BF16 du; the cast
#                                then skips the u block (and the q block when
#                                the dq sink filled it; hstu_du_e5m2.py).
#                                Payload and amax bitwise, u-column bias colsum
#                                fp32-order (DECLARED). Independent of DQ_E5M2.
# Checked at call time by du_producer_enabled().
_DU_E5M2: bool = os.environ.get("DLRM_HSTU_FP8_BWD_DU_E5M2", "0") == "1"
if _DU_E5M2:
    _uvqk_print(
        "requested du e5m2 producer: du_e5m2=1"
        + (
            " -> du_e5m2 INERT without UVQK=1 WGRAD=fp8t"
            if not (_UVQK_ENABLED and _UVQK_WGRAD == "fp8t")
            else ""
        )
        + (
            " -> du_e5m2 INERT without DLRM_HSTU_FP8_LNMD_BWD_V2=1 and "
            "DLRM_HSTU_FP8_MASK16=1"
            if os.environ.get("DLRM_HSTU_FP8_LNMD_BWD_V2", "0") == "0"
            or os.environ.get("DLRM_HSTU_FP8_MASK16", "0") != "1"
            else ""
        )
    )

# DEFAULT OFF:
#   DLRM_HSTU_FP8_BWD_WGRAD_CK=1  the fp8t uvqk (N=2048) and vk (N=1024) wgrad
#                                 run the custom K=tokens kernel wgrad_tm_ck.cu
#                                 (wgrad_ck.py) instead of hipBLASLt; the first
#                                 call per (device, N) is checked against the
#                                 hipBLASLt result and any failure latches back
#                                 to hipBLASLt. dW fp32-order (DECLARED),
#                                 deterministic.
# wgrad_ck prints its own "requested: wgrad_ck=1" line at import.
if wgrad_ck.requested() and not (_UVQK_ENABLED and _UVQK_WGRAD == "fp8t"):
    _uvqk_print("wgrad_ck INERT without UVQK=1 WGRAD=fp8t")


def log_uvqk_once(key: str, message: str) -> None:
    """One line per (key) per process: arming / decline evidence."""
    if key in _uvqk_once_logged:
        return
    _uvqk_once_logged.add(key)
    _uvqk_print(message)


def uvqk_fp8_backward_enabled() -> bool:
    return (
        _UVQK_ENABLED
        and _uvqk_broken is None
        and _TRITON_AVAILABLE
        and _FP8_E4M3 is not None
        and _FP8_E5M2 is not None
        and _GRAD_DTYPE is _FP8_E5M2
    )


def uvqk_wgrad_fp8t() -> bool:
    """The token-major FP8 wgrad mode is selected (it may still decline)."""
    return _UVQK_WGRAD == "fp8t"


def uvqk_fp8t_requested() -> bool:
    """Cheap, side-effect-free first gate for callers (no tensor access)."""
    return _UVQK_ENABLED and _UVQK_WGRAD == "fp8t" and _uvqk_tm_broken is None


def uvqk_tm_pad() -> int:
    return _UVQK_TM_PAD


def uvqk_store_fp8t() -> bool:
    """STORE_T: the forward LN also writes the token-major E4M3 normed_x."""
    return _UVQK_STORE_T and uvqk_fp8t_requested()


def log_uvqk_store_t_once(site: str, used: bool) -> None:
    """Arming line: first stored-tensor use (or fallback) per site per process."""
    what = (
        "used forward-stored token-major E4M3 normed_x (recompute skipped)"
        if used
        else "fell back to the backward recompute"
    )
    log_uvqk_once(f"store_t:{site}:{int(used)}", f"STORE_T site={site} {what}")


def vk_producer_enabled() -> bool:
    return _VK_PRODUCER and _GRAD_DTYPE is _FP8_E5M2 and uvqk_fp8t_requested()


def dq_producer_enabled() -> bool:
    """DLRM_HSTU_FP8_BWD_DQ_E5M2: the ordinary layer's dq-split may emit
    the e5m2 q columns (the caller still needs the fp8t wgrad to run and a
    published per-layer scale)."""
    return _DQ_E5M2 and _GRAD_DTYPE is _FP8_E5M2 and uvqk_fp8t_requested()


def du_producer_requested() -> bool:
    """DLRM_HSTU_FP8_BWD_DU_E5M2 was requested at import. Cheap gate for the
    forward plumbing; False keeps every du-producer code path dormant."""
    return _DU_E5M2


def du_producer_enabled() -> bool:
    """The LN*U*dropout backward may emit the e5m2 u columns of the
    ordinary layer's dz: e5m2 grads and the FP8 uvqk backward with the fp8t
    wgrad requested and not disabled. Does NOT require DQ_E5M2; the
    caller still needs a published per-layer scale (producer_inv_scale)."""
    return (
        _DU_E5M2
        and _GRAD_DTYPE is _FP8_E5M2
        and uvqk_fp8t_requested()
        and uvqk_fp8_backward_enabled()
    )


def grad_dtype_and_limit() -> Tuple[Optional[torch.dtype], float]:
    return _GRAD_DTYPE, _GRAD_MAX


def _peek_site_state(site: str, discriminator: Optional[int]) -> Optional[_SiteScale]:
    """`_site_state(_site_name(...))` without creating anything."""
    if discriminator is None:
        return _sites.get(site)
    index = _layer_index.get((site, discriminator))
    if index is None:
        return None
    return _sites.get(f"{site}.L{index}")


def producer_inv_scale(
    site: str, discriminator: Optional[int] = None
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """(inv_scale, scale snapshot) a producer must cast with.

    `discriminator=None` is the per_layer=False site key (VK producer); the
    dq producer passes the weight's data_ptr() (per_layer=True key).
    No state change. None when the site has not published yet; the caller then
    takes the BF16 path and `uvqk_projection_backward` runs the warm-up. The
    scale snapshot (one tiny device copy) is what `dequantize_prequant` decodes
    with: a refresh inside the projection backward rewrites state.scale in place.
    """
    state = _peek_site_state(site, discriminator)
    if state is None or not state.published:
        return None
    return state.inv_scale, state.scale.clone()


def producer_generation(
    site: str, discriminator: Optional[int] = None
) -> Optional[Tuple[object, int]]:
    """Read-only scale generation of a published site, or None.

    (state object, state.calls). Every `publish()` (warm-up end, refresh)
    runs right after a `calls += 1` of the same state, and the in-place
    rewrite of `state.inv_scale` only happens in `publish()`, so an
    unchanged token proves the device inv_scale a producer cast with is the
    one a later consumer of the same site casts with. The state object (not
    its id) guards against `reset_for_testing` / site re-creation. No state
    change, no device access.
    """
    state = _peek_site_state(site, discriminator)
    if state is None or not state.published:
        return None
    return state, state.calls


def producer_generation_matches(
    token: Optional[Tuple[object, int]],
    site: str,
    discriminator: Optional[int] = None,
) -> bool:
    """`token` (from producer_generation) is still current."""
    now = producer_generation(site, discriminator)
    return (
        token is not None
        and now is not None
        and now[0] is token[0]
        and now[1] == token[1]
    )


def producer_fp8t_inv_scale(
    site: str,
    weight: torch.Tensor,
    rows: int,
    normed_x_fp8_t: Optional[torch.Tensor],
    input_scale_value: Optional[float],
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """ONE scale epoch for a per-layer producer call, or None.

    Returns (inv_scale, scale) as device SNAPSHOTS of this layer's published
    state (keyed exactly as `uvqk_projection_backward(per_layer=True)` keys
    it; two tiny device copies, no host sync, no state change). Every
    producer of the call (bwd_dkdv e5m2 epilogue, the dq sink)
    and the cast of the remaining columns use the inv snapshot; the
    projection backward decodes with the scale snapshot (`prequant_scale`),
    so an in-place refresh of the live state cannot split one call's cast
    and decode across two epochs. Equal in value to the live state at every
    producer launch (publish runs only inside that layer's projection
    backward, after all producers).

    None unless the prequant projection backward will run EXACTLY like the
    BF16-input path: FP8 enabled, published per-layer scale and the
    token-major fp8t wgrad. Mirrors every contract check of
    `_uvqk_projection_backward_prequant` (no side effects), so an armed
    producer is not declined there for a contract reason.
    """
    if not (uvqk_fp8_backward_enabled() and uvqk_wgrad_fp8t()):
        return None
    if _uvqk_tm_broken is not None:
        return None
    if input_scale_value is None or not (input_scale_value > 0.0):
        return None
    if (
        rows <= 0
        or weight.dim() != 2
        or weight.shape[1] % _COLSUM_BLOCK_N != 0
        or not _has_stable_identity(weight)
        or not _tm_operand_ok(normed_x_fp8_t, weight, rows)
    ):
        return None
    state = _peek_site_state(site, weight.data_ptr())
    if state is None or not state.published:
        return None
    return state.inv_scale.clone(), state.scale.clone()


def dequantize_prequant(
    dz_fp8_full: torch.Tensor, rows: int, cast_scale: torch.Tensor
) -> torch.Tensor:
    """BF16 view of a producer-cast gradient, for the fallback paths only.

    Lossy (E5M2): reached only when the FP8 dgrad/wgrad of a call that was
    already cast by the producer fails or declines (first-call failures),
    after which the producer is no longer selected. `cast_scale` is the scale
    snapshot taken with the inv_scale the producer cast with.
    """
    return (dz_fp8_full[:rows].float() * cast_scale).to(torch.bfloat16)


def uvqk_fp8t_will_run(
    site: str,
    weight: torch.Tensor,
    rows: int,
    dtype: torch.dtype,
    input_scale_value: Optional[float],
    per_layer: bool = True,
) -> bool:
    """Whether `uvqk_projection_backward` will take the fp8t wgrad this call.

    Mirrors its arm / broken / scale / identity / published checks with NO side
    effects: no site state is created and the call counter is not advanced
    (the backward call itself advances it exactly once). The caller uses it to
    skip the BF16 normed_x (recompute or forward save); the shape checks that
    need dz can still decline later, which the `recompute_normed_x` hook
    covers.
    """
    if not (uvqk_fp8t_requested() and uvqk_fp8_backward_enabled()):
        return False
    if input_scale_value is None or not (input_scale_value > 0.0):
        return False
    if dtype != torch.bfloat16 or rows <= 0:
        return False
    if weight.dim() != 2 or weight.shape[1] % _COLSUM_BLOCK_N != 0:
        return False
    if per_layer and not _has_stable_identity(weight):
        return False
    state = _peek_site_state(site, weight.data_ptr() if per_layer else None)
    return state is not None and state.published


def _uvqk_scale_tensor(device: torch.device, value: float) -> torch.Tensor:
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, float(value))
    tensor = _uvqk_scale_tensors.get(key)
    if tensor is None:
        tensor = torch.full((1,), float(value), dtype=torch.float32, device=device)
        _uvqk_scale_tensors[key] = tensor
    return tensor


_uvqk_tm_ext = None


def _tm_extension():  # pyre-ignore[3]
    """The fp8t wgrad hipBLASLt extension (its own module and source file, so
    the residual-fused extension is not rebuilt or changed)."""
    global _uvqk_tm_ext
    if _uvqk_tm_ext is None:
        from pathlib import Path

        from torch.utils.cpp_extension import load

        source = Path(__file__).with_name("hipblaslt_fp8_wgrad_tm.cpp")
        build_directory = Path("/tmp/dlrm_hstu_fp8_wgrad_tm")
        build_directory.mkdir(parents=True, exist_ok=True)
        _uvqk_tm_ext = load(
            name="dlrm_hstu_fp8_wgrad_tm",
            sources=[str(source)],
            extra_include_paths=["/opt/rocm/include"],
            extra_cflags=["-O3"],
            extra_ldflags=["-L/opt/rocm/lib", "-lhipblaslt", "-lamdhip64"],
            build_directory=str(build_directory),
            verbose=False,
        )
    return _uvqk_tm_ext


def prebuild_uvqk_extension() -> bool:
    """Build the fp8t wgrad extension before RUN_START when fp8t is requested.

    Called from triton_fp8.prebuild_extensions (DLRM_HSTU_FP8_PREBUILD). Same
    module and flags as the lazy path. No-op (False) unless UVQK=1 and
    WGRAD=fp8t.
    """
    if not uvqk_fp8t_requested():
        return False
    try:
        _tm_extension()
    except Exception as error:  # the first fp8t call then disables fp8t
        _uvqk_print(f"prebuild of the fp8t wgrad extension FAILED: {error}")
        return False
    _uvqk_print("prebuild: fp8t wgrad extension (dlrm_hstu_fp8_wgrad_tm) built")
    if wgrad_ck.requested():
        # DLRM_HSTU_FP8_BWD_WGRAD_CK: hipcc build + load of the custom wgrad
        # (a failure is latched and logged inside; hipBLASLt then runs).
        import time

        t0 = time.perf_counter()
        if wgrad_ck.prebuild():
            _uvqk_print(f"prebuild: wgrad_ck built ({time.perf_counter() - t0:.1f}s)")
    return True


def _tm_workspace(device: torch.device) -> torch.Tensor:
    # The residual-fused GEMM's per-device workspace: same stream, never
    # concurrent with it.
    from generative_recommenders.ops.triton.triton_fp8 import _residual_workspace

    return _residual_workspace(device)


def _tm_arm(
    ext,  # pyre-ignore[2]
    dz_fp8: torch.Tensor,
    act_t: torch.Tensor,
    grad_scale: torch.Tensor,
    act_scale: torch.Tensor,
    workspace: torch.Tensor,
    k_tokens: int,
    site: str,
) -> Tuple[int, str]:
    """First-call guard for the token-major wgrad on this (device, K, N).

    Resolves a CONCRETE hipBLASLt solution index (the pin, else the heuristic
    top-1 at this production problem, k_tokens = M) and checks that exact
    kernel, at the full production K, against a matmul of the same FP8
    operands (fp32 GEMM per token chunk, chunks summed in fp64). e4m3 x e5m2
    products are exact in fp32, so the only legitimate difference is the fp32
    accumulation order; a layout, tail or split-K bug is O(1). Every later
    call runs the armed index with strict_solution=True (no per-call heuristic
    query, and no silent switch to an unchecked kernel: an unsupported problem
    raises and fp8t is disabled). A pinned solution that is unsupported or
    fails falls back to the heuristic top-1; if that fails too this raises.
    """
    device = dz_fp8.device
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, int(act_t.shape[0]), int(dz_fp8.shape[1]))
    armed = _uvqk_tm_armed.get(key)
    if armed is not None:
        return armed
    k = int(k_tokens)
    candidates = []
    if _UVQK_WGRAD_SOLUTION >= 0:
        candidates.append((_UVQK_WGRAD_SOLUTION, f"{_UVQK_WGRAD_SOLUTION} [pinned]"))
    top = ext.fp8_wgrad_tm_algos(
        dz_fp8, act_t, grad_scale, act_scale, workspace, True, k, 0, False, 1
    )
    if top and bool(top[0][2]):
        candidates.append(
            (int(top[0][0]), f"{top[0][0]} [heuristic top-1 {top[0][1][:48]}]")
        )
    ref = torch.zeros(
        act_t.shape[0], dz_fp8.shape[1], dtype=torch.float64, device=device
    )
    # fp32 GEMM per token chunk (e4m3 x e5m2 products are exact in fp32; TF32
    # off), chunks summed in fp64 instead of a full-K fp64 GEMM.
    step = 131072
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for i in range(0, k, step):
            j = min(k, i + step)
            ref += (act_t[:, i:j].float() @ dz_fp8[i:j].float()).double()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    ref *= act_scale.double() * grad_scale.double()
    ref_norm = float(ref.norm())
    report = []
    for solution, label in candidates:
        try:
            got = ext.fp8_wgrad_tm(
                dz_fp8, act_t, grad_scale, act_scale, workspace,
                True, 0, solution, k, 0, True,
            )
        except Exception as error:
            line = (
                f"fp8t guard {site} on {device}: solution {label}: "
                f"{type(error).__name__}: {str(error)[:160]}"
            )
            report.append(line)
            _uvqk_print(line + " -> rejected")
            continue
        diff = got.double() - ref
        rel = float(diff.norm()) / max(ref_norm, 1.0e-300)
        max_abs = float(diff.abs().max())
        finite = bool(torch.isfinite(got).all())
        del got, diff
        line = (
            f"fp8t guard {site} on {device}: solution {label} K={key[1]} "
            f"N={key[2]} tokens={k}: rel={rel:.3e} max_abs={max_abs:.3e} "
            f"ref_norm={ref_norm:.3e} finite={int(finite)}"
        )
        report.append(line)
        if finite and rel <= _UVQK_TM_GUARD_TOL:
            _uvqk_print(line + " -> armed")
            _uvqk_tm_armed[key] = (solution, label)
            return solution, label
        _uvqk_print(line + " -> rejected")
    raise RuntimeError("fp8t wgrad guard failed: " + " | ".join(report or ["no candidate"]))


def _tm_label_desc(label: str) -> str:
    """ARMED-line text for the token-major wgrad that ran (`_run_tm_wgrad` label)."""
    if label.startswith("wgrad_ck"):
        return f"custom {label} (wgrad_tm_ck.cu)"
    return f"hipBLASLt NN, solution {label}"


def _tm_arm_kind(kind: str, tm_ran: bool, label: str) -> str:
    """`_log_armed` key for an ARMED line: the custom wgrad gets its own key, so a
    later fallback to hipBLASLt (wgrad_ck latched off) prints a new ARMED line.
    Unchanged (`kind`) for hipBLASLt / BF16, i.e. with the knob off."""
    return f"{kind} wgrad_ck" if tm_ran and label.startswith("wgrad_ck") else kind


def _run_tm_wgrad(
    dz_fp8_full: torch.Tensor,
    act_t: torch.Tensor,
    grad_scale: torch.Tensor,
    act_scale: torch.Tensor,
    k_tokens: int,
    site: str,
) -> Optional[Tuple[torch.Tensor, str]]:
    """(dW fp32 [K, N], solution label) or None after disabling fp8t.

    A failure here disables only fp8t (BF16 wgrad from then on); the FP8 dgrad
    of this call is unaffected.

    DLRM_HSTU_FP8_BWD_WGRAD_CK=1: the custom kernel (wgrad_ck) runs first and
    returns (dW, 'wgrad_ck_w1' | 'wgrad_ck_w2'); a call outside its contract,
    or once it is latched off, runs the hipBLASLt body below unchanged. Its
    first call per (device, N) is checked against this call's hipBLASLt result,
    which is computed at most once and returned if the custom kernel is
    rejected or fails.

    DLRM_HSTU_FP8_BWD_WGRAD_CK_XT_TILED=1: `act_t` may be the step-tiled 3-D
    xT [KT, 512, 128] (wgrad_ck.tiled_active producers); wgrad_ck runs it
    directly, and every hipBLASLt use (the guard's reference, a decline or a
    latch) reads `wgrad_ck.untile_to_row_major(act_t)`, the row-major bytes of
    the knob-off producers (bitwise the knob-off hipBLASLt result). That copy
    comes from the fp8_tm_alloc pool when it is active (fp8_tm_alloc.empty_copy); when it
    is out of memory this returns None for THIS call only (the caller's BF16
    wgrad; nothing latched, logged once per site) and a first-call guard that
    needed it is deferred (wgrad_ck.GuardDeferred: no verdict, not armed).
    """
    untiled: list = []

    def row_major() -> Optional[torch.Tensor]:
        # 2-D: as is. 3-D (step-tiled): one untiled copy per call, on demand;
        # None (also for a second ask in this call) when it is out of memory.
        if act_t.dim() != 3:
            return act_t
        if not untiled:
            untiled.append(_untile_xt(act_t, site))
        return untiled[0]

    if wgrad_ck.enabled():
        hip: list = []

        def reference() -> torch.Tensor:
            if not hip:
                operand = row_major()
                if operand is None:
                    raise wgrad_ck.GuardDeferred("untiled xT copy out of memory")
                hip.append(
                    _run_tm_wgrad_hipblaslt(
                        dz_fp8_full, operand, grad_scale, act_scale, k_tokens, site
                    )
                )
            if hip[0] is None:
                raise RuntimeError(
                    f"hipBLASLt reference unavailable (fp8t wgrad: {_uvqk_tm_broken})"
                )
            return hip[0][0]

        got = wgrad_ck.run(
            dz_fp8_full, act_t, grad_scale, act_scale, k_tokens, site, reference
        )
        if got is not None:
            return got
        if hip:
            return hip[0]
    operand = row_major()
    if operand is None:
        # The untiled copy is out of memory: BF16 wgrad for this call only (the
        # caller's `d_w is None` path); fp8t and wgrad_ck stay armed.
        return None
    return _run_tm_wgrad_hipblaslt(
        dz_fp8_full, operand, grad_scale, act_scale, k_tokens, site
    )


def _untile_xt(act_t: torch.Tensor, site: str) -> Optional[torch.Tensor]:
    """`wgrad_ck.untile_to_row_major(act_t)` with the copy from the fp8_tm_alloc pool when it
    is active (fp8_tm_alloc.empty_copy, normally its prewarmed spare block), or None
    when it is out of memory (logged once per site; no latch, no empty_cache)."""
    try:
        return wgrad_ck.untile_to_row_major(act_t, empty=fp8_tm_alloc.empty_copy)
    except Exception as error:
        if not fp8_tm_alloc.is_oom(error):
            raise
        log_uvqk_once(
            f"untile_oom:{site}",
            f"{site}: untiled copy of the step-tiled xT {tuple(act_t.shape)} out of "
            f"memory on {act_t.device} ({type(error).__name__}) -> BF16 wgrad for "
            "such calls only (nothing latched)",
        )
        return None


def _run_tm_wgrad_hipblaslt(
    dz_fp8_full: torch.Tensor,
    act_t: torch.Tensor,
    grad_scale: torch.Tensor,
    act_scale: torch.Tensor,
    k_tokens: int,
    site: str,
) -> Optional[Tuple[torch.Tensor, str]]:
    """The hipBLASLt fp8t wgrad (`_run_tm_wgrad` without the custom kernel)."""
    global _uvqk_tm_broken
    try:
        ext = _tm_extension()
        workspace = _tm_workspace(dz_fp8_full.device)
        solution, label = _tm_arm(
            ext, dz_fp8_full, act_t, grad_scale, act_scale, workspace,
            k_tokens, site,
        )
        d_w = ext.fp8_wgrad_tm(
            dz_fp8_full, act_t, grad_scale, act_scale, workspace,
            True, 0, solution, k_tokens, 0, True,
        )
        return d_w, label
    except Exception as error:
        _uvqk_tm_broken = f"{type(error).__name__}: {error}"
        logger.warning("[hstu-fp8-bwd] uvqk fp8t wgrad disabled: %s", _uvqk_tm_broken)
        _uvqk_print(f"fp8t wgrad DISABLED: {_uvqk_tm_broken}")
        return None


def _tm_operand_ok(
    normed_x_fp8_t: Optional[torch.Tensor], weight: torch.Tensor, m: int
) -> bool:
    if (
        normed_x_fp8_t is not None
        and normed_x_fp8_t.dim() == 3
        and normed_x_fp8_t.dtype == _FP8_E4M3
    ):
        # DLRM_HSTU_FP8_BWD_WGRAD_CK_XT_TILED: step-tiled [KT, K, 128],
        # contiguous (`_run_tm_wgrad` untiles it for hipBLASLt).
        return (
            normed_x_fp8_t.shape[1] == weight.shape[0]
            and normed_x_fp8_t.shape[2] == _UVQK_TM_PAD
            and normed_x_fp8_t.shape[0] * _UVQK_TM_PAD >= m
            and normed_x_fp8_t.is_contiguous()
        )
    return (
        normed_x_fp8_t is not None
        and normed_x_fp8_t.dtype == _FP8_E4M3
        and normed_x_fp8_t.dim() == 2
        and normed_x_fp8_t.shape[0] == weight.shape[0]
        and normed_x_fp8_t.shape[1] >= m
        and normed_x_fp8_t.stride(1) == 1
    )


def _log_armed(site: str, device: torch.device, kind: str, line: str) -> None:
    arm_key = (site, str(device), kind)
    if arm_key in _uvqk_armed_logged:
        return
    _uvqk_armed_logged.add(arm_key)
    _uvqk_print(line)


def log_armed_v2(site: str, device: torch.device, line: Callable[[], str]) -> None:
    """ARMED line for the first call of a v2 kernel per (site, device).

    Goes through `_log_armed` (same key set, same print); the line is only
    formatted on that first call.
    """
    if (site, str(device), "v2") not in _uvqk_armed_logged:
        _log_armed(site, device, "v2", line())


def uvqk_projection_backward(
    dz: Optional[torch.Tensor],
    normed_x: Optional[torch.Tensor],
    input_scale_value: Optional[float],
    weight: torch.Tensor,
    site: str = "uvqk_projection",
    per_layer: bool = True,
    normed_x_fp8_t: Optional[torch.Tensor] = None,
    recompute_normed_x: Optional[Callable[[], torch.Tensor]] = None,
    prequant: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = None,
    producer_desc: Optional[str] = None,
    prequant_scale: Optional[torch.Tensor] = None,
    prequant_bf16_dz: Optional[Callable[[], torch.Tensor]] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """(d_normed_x, d_weight, d_bias fp32) with an FP8 dgrad, or None for BF16.

    `normed_x_fp8_t` is the token-major E4M3 copy [K, >=M] for the fp8t wgrad.
    `normed_x` may be None only when `recompute_normed_x` is given: it is then
    recomputed here if the BF16 wgrad has to run after all, and the caller
    must do the same when this returns None.
    `per_layer=True` keys the delayed scale on the weight parameter's storage
    (one scale per layer) and requires it to be the leaf parameter;
    `per_layer=False` keys on `site` alone, for a call site that runs once per
    step (the targets-only layer, whose weight is a per-call packed copy).
    `prequant` = (dz_fp8_full [Mp, N], amax, colsum, M): the producer already
    cast dz with this site's published inv_scale (`producer_inv_scale`); `dz`
    is then None. On a decline/failure the caller must rebuild BF16 dz with
    `dequantize_prequant` (lossy, first-call failures only).
    `producer_desc` (dq producer) only changes the ARMED line's producer
    text; per_layer=True prequant keys the scale on the weight like the
    BF16-input path.
    Per-layer producer (both optional; absent = the behaviour
    above): `prequant_scale` is the decode-scale snapshot of the scale epoch
    the producer cast with (`producer_fp8t_inv_scale`), used for every GEMM
    instead of the live state.scale; `prequant_bf16_dz` returns the EXACT
    BF16 dz for a BF16 wgrad inside this call (token-major wgrad failed or
    off) instead of decoding the e5m2 payload. On a None return such a
    caller re-runs its producers without the sinks and calls this again with
    the BF16 dz (no lossy decode).
    """
    global _uvqk_broken
    if not uvqk_fp8_backward_enabled():
        return None
    if prequant is not None:
        return _uvqk_projection_backward_prequant(
            prequant, normed_x, input_scale_value, weight, site, per_layer,
            normed_x_fp8_t, recompute_normed_x,
            **({} if producer_desc is None else {"producer_desc": producer_desc}),
            **(
                {}
                if prequant_scale is None and prequant_bf16_dz is None
                else {
                    "decode_scale": prequant_scale,
                    "bf16_dz_provider": prequant_bf16_dz,
                }
            ),
        )
    if per_layer and not _has_stable_identity(weight):
        return None
    if (
        dz is None
        or dz.dim() != 2
        or not dz.is_contiguous()
        or dz.dtype != torch.bfloat16
        or dz.shape[1] % _COLSUM_BLOCK_N != 0
        or dz.shape[0] == 0
        or weight.dim() != 2
        or weight.shape[1] != dz.shape[1]
        or (normed_x is None and recompute_normed_x is None)
        or (
            normed_x is not None
            and normed_x.shape != (dz.shape[0], weight.shape[0])
        )
    ):
        log_uvqk_once(
            f"declined:{site}",
            f"{site}: FP8 projection backward DECLINED for this contract "
            f"(dz={None if dz is None else (tuple(dz.shape), dz.dtype, dz.is_contiguous())}, "
            f"weight={tuple(weight.shape)}) -> shipped BF16 path",
        )
        return None
    m = dz.shape[0]
    use_tm = (
        uvqk_wgrad_fp8t()
        and _uvqk_tm_broken is None
        and input_scale_value is not None
        and input_scale_value > 0.0
        and _tm_operand_ok(normed_x_fp8_t, weight, m)
    )
    state = _site_state(
        _site_name(site, weight.data_ptr() if per_layer else None), dz.device
    )
    state.calls += 1
    if not state.published:
        state.accumulate(dz.detach().abs().amax().float())
        if state.calls >= _BWD_WARMUP_CALLS:
            state.publish()
        return None

    w_partial: Optional[torch.Tensor] = None
    if _helpers_fused_ok(weight):
        # HELPERS_FUSED: amax partials reduced by the site program of the weight-amax
        # launch (note_clipping + accumulate + refresh snapshot in it).
        dz_fp8_full, partial_amax, colsum = quantize_amax_colsum(
            dz,
            state.inv_scale,
            pad_to=_UVQK_TM_PAD if use_tm else 1,
            return_partial_amax=True,
        )
        dz_fp8 = dz_fp8_full[:m] if dz_fp8_full.shape[0] != m else dz_fp8_full
        refresh = state.calls % _BWD_REFRESH_CALLS == 0
        w_partial, snap, _ = _fused_site_launch(
            state, partial_amax, weight, snapshot=refresh, t_note=False,
            t_shift=0, snapshot_t=False,
        )
        if refresh:
            assert snap is not None
            grad_scale = snap
            state.publish()
        else:
            grad_scale = state.scale
    else:
        dz_fp8_full, amax, colsum = quantize_amax_colsum(
            dz, state.inv_scale, pad_to=_UVQK_TM_PAD if use_tm else 1
        )
        # Exactly M rows for the dgrad (a contiguous row slice; rows >= M are the
        # zero padding the token-major wgrad never reads with k_tokens = M).
        dz_fp8 = dz_fp8_full[:m] if dz_fp8_full.shape[0] != m else dz_fp8_full
        state.note_clipping(amax)
        state.accumulate(amax)
        # The payload above was cast with the scale in use NOW. A refresh rewrites
        # state.scale in place, so decode with a snapshot taken before publishing.
        if state.calls % _BWD_REFRESH_CALLS == 0:
            grad_scale = state.scale.clone()
            state.publish()
        else:
            grad_scale = state.scale

    def bf16_normed_x() -> torch.Tensor:
        if normed_x is not None:
            return normed_x
        assert recompute_normed_x is not None
        return recompute_normed_x()

    if w_partial is not None:
        weight_fp8, weight_scale = _fused_weight_quant(weight, w_partial)
    else:
        weight_fp8, weight_scale = quantize_weight(weight)
    tm_label = ""
    try:
        # Issue order matches triton_addmm_bwd (colsum -> wgrad -> dgrad): the
        # wgrad is issued before the FP8 dgrad, so the BF16 launch order is
        # kept. Math does not depend on the order.
        d_w: Optional[torch.Tensor] = None
        if use_tm:
            assert normed_x_fp8_t is not None and input_scale_value is not None
            got = _run_tm_wgrad(
                dz_fp8_full,
                normed_x_fp8_t,
                grad_scale,
                _uvqk_scale_tensor(dz.device, input_scale_value),
                m,
                site,
            )
            if got is not None:
                d_w, tm_label = got
        else:
            d_w = torch.mm(bf16_normed_x().t(), dz)
        d_x = torch._scaled_mm(
            dz_fp8,
            weight_fp8.t(),
            scale_a=grad_scale,
            scale_b=weight_scale,
            out_dtype=dz.dtype,
        )
        tm_ran = use_tm and d_w is not None
        if d_w is None:
            # fp8t declined inside this call: BF16 wgrad, recomputing the BF16
            # normed_x if the caller skipped it.
            d_w = torch.mm(bf16_normed_x().t(), dz)
    except Exception as error:  # first-call failure: never take training down
        _uvqk_broken = f"{type(error).__name__}: {error}"
        logger.warning("[hstu-fp8-bwd] uvqk FP8 backward disabled: %s", _uvqk_broken)
        _uvqk_print(f"DISABLED: {_uvqk_broken}")
        return None
    if tm_ran:
        assert normed_x_fp8_t is not None
        wgrad_desc = (
            f"e5m2xe4m3 token-major ({_tm_label_desc(tm_label)}, "
            f"ld={normed_x_fp8_t.stride(0)} k_tokens={m})"
        )
    else:
        wgrad_desc = "bf16 (torch.mm)"
    _log_armed(
        site,
        dz.device,
        _tm_arm_kind(wgrad_desc.split(" (")[0], tm_ran, tm_label),
        f"ARMED {site} on {dz.device}: dgrad e5m2xe4m3 TN (_scaled_mm) + wgrad "
        f"{wgrad_desc} + fused bias colsum, M={m} N={dz.shape[1]} "
        f"K={weight.shape[0]}",
    )
    record_engaged(f"{site}_backward")
    if tm_ran:
        record_engaged(f"{site}_wgrad_fp8t")
        if tm_label.startswith("wgrad_ck"):
            record_engaged(f"{site}_wgrad_ck")
    return d_x, d_w, colsum


def _uvqk_projection_backward_prequant(
    prequant: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
    normed_x: Optional[torch.Tensor],
    input_scale_value: Optional[float],
    weight: torch.Tensor,
    site: str,
    per_layer: bool,
    normed_x_fp8_t: Optional[torch.Tensor],
    recompute_normed_x: Optional[Callable[[], torch.Tensor]],
    producer_desc: str = "delta attention bwd, no BF16 dvk, no cast pass",
    decode_scale: Optional[torch.Tensor] = None,
    bf16_dz_provider: Optional[Callable[[], torch.Tensor]] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """`uvqk_projection_backward` for a gradient the producer already cast.

    Same scale bookkeeping (calls, note_clipping, accumulate, refresh with a
    decode snapshot) and the same GEMMs in the same order as the BF16-input
    path after its `quantize_amax_colsum`; only the cast pass is gone.
    per_layer=False keys the scale on the site (VK producer); per_layer=True
    on the weight parameter (dq producer, ordinary layers).
    `decode_scale`: the scale snapshot of the producers' scale
    epoch, the decode scale of every GEMM (the bookkeeping is unchanged).
    `bf16_dz_provider`: the exact BF16 dz for the BF16 wgrad
    fallback instead of the e5m2 decode.
    """
    global _uvqk_broken
    dz_fp8_full, amax, colsum, m = prequant
    device = dz_fp8_full.device
    n = dz_fp8_full.shape[1]
    if (
        input_scale_value is None
        or not (input_scale_value > 0.0)
        or weight.dim() != 2
        or weight.shape[1] != n
        or m <= 0
        or dz_fp8_full.shape[0] < m
        or (normed_x is None and recompute_normed_x is None)
        or (per_layer and not _has_stable_identity(weight))
    ):
        return None
    state = _site_state(
        _site_name(site, weight.data_ptr() if per_layer else None), device
    )
    if not state.published:
        return None
    use_tm = (
        uvqk_wgrad_fp8t()
        and _uvqk_tm_broken is None
        and _tm_operand_ok(normed_x_fp8_t, weight, m)
    )
    state.calls += 1
    dz_fp8 = dz_fp8_full[:m] if dz_fp8_full.shape[0] != m else dz_fp8_full
    w_partial: Optional[torch.Tensor] = None
    if (
        _helpers_fused_ok(weight)
        and amax.numel() == 1
        and amax.dtype == torch.float32
    ):
        # HELPERS_FUSED: note_clipping + accumulate (+ refresh snapshot) as program 0 of
        # the weight-amax launch.
        refresh = state.calls % _BWD_REFRESH_CALLS == 0
        w_partial, snap, _ = _fused_site_launch(
            state, amax, weight, snapshot=refresh, t_note=False,
            t_shift=0, snapshot_t=False,
        )
        if refresh:
            assert snap is not None
            grad_scale = snap
            state.publish()
        else:
            grad_scale = state.scale
    else:
        state.note_clipping(amax)
        state.accumulate(amax)
        if state.calls % _BWD_REFRESH_CALLS == 0:
            grad_scale = state.scale.clone()
            state.publish()
        else:
            grad_scale = state.scale
    if decode_scale is not None:
        # The producers' scale epoch. Their casts used the matching
        # inv snapshot, so every GEMM decodes with this snapshot whatever the
        # refresh above wrote into the live state (same value as grad_scale:
        # nothing publishes between the epoch and this call's refresh).
        grad_scale = decode_scale
    bf16_dz_cache: list = []

    def bf16_dz() -> torch.Tensor:
        if bf16_dz_provider is not None:
            # Exact BF16 dz from the producers' caller (no decode).
            return bf16_dz_provider()
        # Decode with the scale the payload was cast with (grad_scale).
        if not bf16_dz_cache:
            bf16_dz_cache.append((dz_fp8.float() * grad_scale).to(torch.bfloat16))
        return bf16_dz_cache[0]

    def bf16_normed_x() -> torch.Tensor:
        if normed_x is not None:
            return normed_x
        assert recompute_normed_x is not None
        return recompute_normed_x()

    if w_partial is not None:
        weight_fp8, weight_scale = _fused_weight_quant(weight, w_partial)
    else:
        weight_fp8, weight_scale = quantize_weight(weight)
    tm_label = ""
    try:
        # Issue order identical to the BF16-input path (wgrad, then dgrad).
        d_w: Optional[torch.Tensor] = None
        if use_tm:
            assert normed_x_fp8_t is not None
            got = _run_tm_wgrad(
                dz_fp8_full,
                normed_x_fp8_t,
                grad_scale,
                _uvqk_scale_tensor(device, input_scale_value),
                m,
                site,
            )
            if got is not None:
                d_w, tm_label = got
        d_x = torch._scaled_mm(
            dz_fp8,
            weight_fp8.t(),
            scale_a=grad_scale,
            scale_b=weight_scale,
            out_dtype=torch.bfloat16,
        )
        tm_ran = d_w is not None
        if d_w is None:
            d_w = torch.mm(bf16_normed_x().t(), bf16_dz())
    except Exception as error:  # first-call failure: never take training down
        _uvqk_broken = f"{type(error).__name__}: {error}"
        logger.warning("[hstu-fp8-bwd] uvqk FP8 backward disabled: %s", _uvqk_broken)
        _uvqk_print(f"DISABLED: {_uvqk_broken}")
        return None
    wgrad_desc = (
        f"e5m2xe4m3 token-major ({_tm_label_desc(tm_label)}, k_tokens={m})"
        if tm_ran
        else (
            "bf16 (torch.mm, dequantized dz)"
            if bf16_dz_provider is None
            else "bf16 (torch.mm, exact BF16 dz from the producer's caller)"
        )
    )
    _log_armed(
        site,
        device,
        _tm_arm_kind("producer " + wgrad_desc.split(" (")[0], tm_ran, tm_label),
        f"ARMED {site} on {device}: PRODUCER-cast e5m2 dz ({producer_desc}) "
        f"+ dgrad e5m2xe4m3 TN (_scaled_mm) + "
        f"wgrad {wgrad_desc}, M={m} N={n} K={weight.shape[0]}",
    )
    record_engaged(f"{site}_backward_producer")
    if tm_ran and tm_label.startswith("wgrad_ck"):
        record_engaged(f"{site}_wgrad_ck")
    return d_x, d_w, colsum


def site_diagnostics() -> Dict[str, Dict[str, float]]:
    """Per-site counters, for logging. Reads device values, so not in the step."""
    report: Dict[str, Dict[str, float]] = {}
    for site, state in _sites.items():
        report[site] = {
            "calls": float(state.calls),
            "scale": float(state.scale.item()) if state.published else 0.0,
            "clipped_windows": float(state.clipped_windows.item()),
        }
        if state.t_clipped_windows is not None:
            report[site]["t_clipped_windows"] = float(
                state.t_clipped_windows.item()
            )
    return report


def reset_for_testing() -> None:
    global _uvqk_tm_broken, _uvqk_broken
    _sites.clear()
    _layer_index.clear()
    _engaged_sites.clear()
    _uvqk_armed_logged.clear()
    _uvqk_once_logged.clear()
    _uvqk_tm_armed.clear()
    _uvqk_tm_broken = None
    _uvqk_broken = None
    wgrad_ck.reset_for_testing()
