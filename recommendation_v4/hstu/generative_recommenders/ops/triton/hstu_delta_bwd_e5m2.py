# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

"""Targets-only delta attention backward that emits the E5M2 dV|dK pack.

DLRM_HSTU_FP8_BWD_VK_PRODUCER (default off). With the FP8 VK projection backward armed (DLRM_HSTU_FP8_BWD_UVQK=1,
DLRM_HSTU_FP8_BWD_UVQK_WGRAD=fp8t), the shipped path is

    _hstu_attn_delta_bwd     -> dvk  bf16 [M, H*(D_V+D_Q)]   (write 2B/elem)
    _quantize_amax_colsum    -> e5m2 [Mp, ...] + amax + colsum
                                (read 2B/elem, write 1B/elem)

The producer variant below runs the SAME delta backward math and, instead of
storing the BF16 dV/dK, stores the E5M2 cast of exactly the BF16 value the
stock kernel would have stored (round to BF16, back to fp32, x inv_scale,
clamp +-LIMIT, cast). The FP8 payload is therefore byte-identical to
`bwd_fp8.quantize_amax_colsum(dvk)`; amax is exact (max is order-free); the
fp32 bias column sum is the same math in a different fp32 reduction order
(per-sequence partials, then a sum over sequences). The BF16 dvk is never
materialized and the separate cast pass is gone.

This is a copy of `triton_hstu_attention._hstu_attn_delta_bwd` (left
untouched); keep the math in sync with it.

DLRM_HSTU_FP8_BWD_VKP_V2 (default off) launches
`_hstu_attn_delta_bwd_e5m2_v2` instead: the amax comes from per-row maxima of
|silu| and |dqk| (exact, see the kernel comment), dq accumulates element-wise
over the tiles with one cross-row reduction after the loop instead of one
cross-warp (LDS + barrier) reduction per 32-row tile, and programs run
longest-sequence-first. Payload, amax and colsum are bitwise unchanged; dq is
the same math in a different fp32 summation order (DECLARED).
"""

import os
from typing import Tuple

import torch
import triton
import triton.language as tl

from generative_recommenders.common import switch_to_contiguous_if_needed
from generative_recommenders.ops.triton.triton_hstu_attention import (
    fast_dividef,
    fast_expf,
)


# Launch config (fixed, like the stock kernel's BLOCK_N=32 / 4 warps).
_BLOCK_N: int = 32
_NUM_WARPS: int = 4
_REDUCE_EACH: bool = False
# DLRM_HSTU_FP8_BWD_VKP_V2=1 (default off): read at import, checked by
# `hstu_attention_delta_bwd_e5m2` at call time.
_VKP_V2: bool = os.environ.get("DLRM_HSTU_FP8_BWD_VKP_V2", "0") == "1"
# The v2 configuration (module constants, not knobs): exact row-max amax,
# element-wise dq accumulator (DECLARED fp32-order change of dq only), and
# longest-first launch order (2: one-program Triton sort for Z <= 4096 else
# torch.argsort; 1: torch.argsort; 0: natural order). Order is result-free.
# Any change of this tuple makes the v2 diet kernel decline.
_VKP_V2_AMAX_ROWMAX: bool = True
_VKP_V2_DQ_ELEM: bool = True
_VKP_V2_LPT: int = 2
_VKP_V2_NUM_STAGES: int = 1


@triton.jit
def _hstu_attn_delta_bwd_e5m2(
    Q,
    K,
    V,
    seq_offsets,
    DOut,
    DQ,
    DVK8,
    PARTIAL_COL,
    PARTIAL_AMAX,
    INV_SCALE,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_dom,
    stride_doh,
    stride_dqm,
    stride_dqh,
    stride_dvk,
    alpha,
    K_SCALE,
    V_SCALE,
    H,
    MAX_SEQ_LEN,
    COL_K0,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LIMIT: tl.constexpr,
    REDUCE_EACH: tl.constexpr,
):
    """Same math as `_hstu_attn_delta_bwd`; dV|dK leave as E5M2 + partials.

    dV of head h occupies DVK8 columns [h*D_V, (h+1)*D_V); dK of head h
    occupies [COL_K0 + h*D_Q, ...). PARTIAL_COL is [Z, n_cols] fp32 (one row per
    sequence, each program writes its own head's columns), PARTIAL_AMAX is
    [Z*H] fp32. Rows of DVK8 not covered by any sequence are not touched.
    """
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = (off_hz % H).to(tl.int64)
    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)
    inv = tl.load(INV_SCALE)

    offs_q = tl.arange(0, BLOCK_D_Q)
    offs_v = tl.arange(0, BLOCK_D_V)
    q = tl.load(
        Q + off_z * stride_qm + off_h * stride_qh + offs_q,
        mask=offs_q < BLOCK_D_Q,
        other=0.0,
    ).to(tl.float32)
    dout = tl.load(
        DOut + off_z * stride_dom + off_h * stride_doh + offs_v,
        mask=offs_v < BLOCK_D_V,
        other=0.0,
    ).to(tl.float32)
    dq = tl.zeros([BLOCK_D_Q], dtype=tl.float32)
    # REDUCE_EACH=False: element-wise register accumulators reduced ONCE after
    # the loop; True: [D] vectors reduced every block (fewer registers, one
    # cross-warp reduction per block). Same values either way up to fp32 order.
    if REDUCE_EACH:
        col_k = tl.zeros([BLOCK_D_Q], dtype=tl.float32)
        col_v = tl.zeros([BLOCK_D_V], dtype=tl.float32)
        amax_k = tl.zeros([BLOCK_D_Q], dtype=tl.float32)
        amax_v = tl.zeros([BLOCK_D_V], dtype=tl.float32)
    else:
        col_k = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
        col_v = tl.zeros([BLOCK_N, BLOCK_D_V], dtype=tl.float32)
        amax_k = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
        amax_v = tl.zeros([BLOCK_N, BLOCK_D_V], dtype=tl.float32)

    for start_n in tl.range(0, seq_len, BLOCK_N, num_stages=1):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len
        k_ptrs = (
            K
            + (seq_start + offs_n[:, None]) * stride_kn
            + off_h * stride_kh
            + offs_q[None, :]
        )
        v_ptrs = (
            V
            + (seq_start + offs_n[:, None]) * stride_vn
            + off_h * stride_vh
            + offs_v[None, :]
        )
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        k_real = (k.to(tl.float32) * K_SCALE).to(tl.bfloat16).to(tl.float32)
        v_real = (v.to(tl.float32) * V_SCALE).to(tl.bfloat16).to(tl.float32)

        qk = tl.sum(k_real * q[None, :], axis=1) * alpha
        sig = fast_dividef(1.0, 1.0 + tl.exp(-qk))
        silu = fast_dividef(qk, 1.0 + fast_expf(-qk)) * (1.0 / MAX_SEQ_LEN)
        silu = tl.where(mask_n, silu, 0.0).to(DOut.dtype.element_ty)

        dv = silu[:, None].to(tl.float32) * dout[None, :]
        dqk = tl.sum(v_real * dout[None, :], axis=1)
        dqk *= sig * (1.0 + qk * (1.0 - sig)) * (1.0 / MAX_SEQ_LEN)
        dqk = tl.where(mask_n, dqk, 0.0).to(tl.bfloat16)
        dk = dqk[:, None].to(tl.float32) * q[None, :] * alpha
        dq += tl.sum(dqk[:, None].to(tl.float32) * k_real, axis=0)

        # The value the stock kernel stores (BF16), back in fp32: what the
        # stock cast kernel would have loaded.
        dk_b = dk.to(tl.bfloat16).to(tl.float32)
        dv_b = dv.to(tl.bfloat16).to(tl.float32)
        # Masked rows are exactly 0 (silu/dqk zeroed above), so they add
        # nothing to the sums or the max.
        if REDUCE_EACH:
            col_k += tl.sum(dk_b, axis=0)
            col_v += tl.sum(dv_b, axis=0)
            amax_k = tl.maximum(amax_k, tl.max(tl.abs(dk_b), axis=0))
            amax_v = tl.maximum(amax_v, tl.max(tl.abs(dv_b), axis=0))
        else:
            col_k += dk_b
            col_v += dv_b
            amax_k = tl.maximum(amax_k, tl.abs(dk_b))
            amax_v = tl.maximum(amax_v, tl.abs(dv_b))
        dk8 = tl.minimum(tl.maximum(dk_b * inv, -LIMIT), LIMIT)
        dv8 = tl.minimum(tl.maximum(dv_b * inv, -LIMIT), LIMIT)

        row_ptr = DVK8 + (seq_start + offs_n[:, None]) * stride_dvk
        tl.store(
            row_ptr + off_h * BLOCK_D_V + offs_v[None, :],
            dv8.to(DVK8.dtype.element_ty),
            mask=mask_n[:, None],
        )
        tl.store(
            row_ptr + COL_K0 + off_h * BLOCK_D_Q + offs_q[None, :],
            dk8.to(DVK8.dtype.element_ty),
            mask=mask_n[:, None],
        )

    tl.store(
        DQ + off_z * stride_dqm + off_h * stride_dqh + offs_q,
        (dq * alpha).to(DQ.dtype.element_ty),
        mask=offs_q < BLOCK_D_Q,
    )
    col_row = PARTIAL_COL + off_z.to(tl.int64) * stride_dvk
    if REDUCE_EACH:
        tl.store(col_row + off_h * BLOCK_D_V + offs_v, col_v)
        tl.store(col_row + COL_K0 + off_h * BLOCK_D_Q + offs_q, col_k)
        tl.store(
            PARTIAL_AMAX + off_hz,
            tl.maximum(tl.max(amax_k, axis=0), tl.max(amax_v, axis=0)),
        )
    else:
        tl.store(col_row + off_h * BLOCK_D_V + offs_v, tl.sum(col_v, axis=0))
        tl.store(col_row + COL_K0 + off_h * BLOCK_D_Q + offs_q, tl.sum(col_k, axis=0))
        tl.store(
            PARTIAL_AMAX + off_hz,
            tl.maximum(
                tl.max(tl.max(amax_k, axis=1), axis=0),
                tl.max(tl.max(amax_v, axis=1), axis=0),
            ),
        )


@triton.jit
def _hstu_attn_delta_bwd_e5m2_lpt_perm(
    seq_offsets, PERM, Z, BLOCK_Z: tl.constexpr, ZBITS: tl.constexpr
):
    # One program: PERM = sequence ids by decreasing length (ties: lower id
    # first). Key = len << ZBITS | (BLOCK_Z - 1 - z); padded lanes sort last.
    # The length is clamped below 2^(31 - ZBITS) so a key never reaches the
    # sign bit: PERM is a permutation of [0, Z) for any offsets.
    offs = tl.arange(0, BLOCK_Z)
    m = offs < Z
    s0 = tl.load(seq_offsets + offs, mask=m, other=0)
    s1 = tl.load(seq_offsets + offs + 1, mask=m, other=0)
    ln = tl.minimum((s1 - s0).to(tl.int32), (1 << (31 - ZBITS)) - 1)
    key = tl.where(m, (ln << ZBITS) | ((BLOCK_Z - 1) - offs), -1)
    key = tl.sort(key, descending=True)
    tl.store(PERM + offs, (BLOCK_Z - 1) - (key & (BLOCK_Z - 1)), mask=m)


def _log_vkp_v2_armed(
    device: torch.device, z: int, h: int, rows: int, lpt: int
) -> None:
    # Imported here: only the v2 path needs bwd_fp8 (its ARMED-line helper).
    from generative_recommenders.ops.triton import bwd_fp8

    bwd_fp8.log_armed_v2(
        "vkp_v2",
        device,
        lambda: (
            f"ARMED vkp_v2 on {device}: _hstu_attn_delta_bwd_e5m2_v2 "
            f"(amax_rowmax={int(_VKP_V2_AMAX_ROWMAX)} "
            f"dq_elem={int(_VKP_V2_DQ_ELEM)} lpt={lpt} "
            f"num_stages={_VKP_V2_NUM_STAGES}; payload/amax/colsum bitwise vs "
            f"_hstu_attn_delta_bwd_e5m2, dq "
            f"{'fp32-order (DECLARED)' if _VKP_V2_DQ_ELEM else 'bitwise'}), "
            f"Z={z} H={h} rows={rows}"
        ),
    )


# DLRM_HSTU_FP8_DELTA_BWD_V2_DIET (read per call, armed
# only by "1"; code default OFF, the gin binds it on): the VALU-diet form
# of the v2 kernel (hstu_delta_bwd_e5m2_v2_diet.py), byte-identical outputs. It
# replaces the v2 launch ONLY in the configuration it was verified in (the v2
# constants above as shipped, BLOCK_N=32, 4 warps, REDUCE_EACH off, D_Q = D_V
# = 128, bf16 q/dout/dQ, e4m3 K/V, e5m2 payload with its 57344 limit, gfx950,
# Triton 3.6.0); any other call keeps the v2 kernel and logs one
# "declined (<reason>)" line per reason. Inert without BWD_VKP_V2.
_V2_DIET_TRITON = "3.6.0"
_V2_DIET_LOGGED = [False]
_V2_DIET_DECLINE_LOGGED = set()
_V2_DIET_ARCH = {}


def delta_bwd_v2_diet_requested() -> bool:
    return os.environ.get("DLRM_HSTU_FP8_DELTA_BWD_V2_DIET", "0") == "1"


def _v2_diet_decline_reason(
    q, k, v, dout, dq, dvk_fp8, limit, block_n, num_warps, reduce_each
):
    consts = (_VKP_V2_AMAX_ROWMAX, _VKP_V2_DQ_ELEM, _VKP_V2_LPT, _VKP_V2_NUM_STAGES)
    if consts != (True, True, 2, 1):
        return f"v2 constants (amax_rowmax, dq_elem, lpt, num_stages)={consts}"
    if block_n != 32 or num_warps != 4 or reduce_each:
        return f"block_n={block_n} num_warps={num_warps} reduce_each={reduce_each}"
    if q.shape[-1] != 128 or k.shape[-1] != 128 or v.shape[-1] != 128:
        return f"D_Q={q.shape[-1]} D_K={k.shape[-1]} D_V={v.shape[-1]}"
    if (
        q.dtype != torch.bfloat16
        or dout.dtype != torch.bfloat16
        or dq.dtype != torch.bfloat16
    ):
        return f"q/dout/dq dtype {q.dtype}/{dout.dtype}/{dq.dtype}"
    if k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
        return f"k/v dtype {k.dtype}/{v.dtype}"
    if dvk_fp8.dtype != torch.float8_e5m2 or float(limit) != 57344.0:
        return f"payload dtype {dvk_fp8.dtype} limit={limit}"
    if triton.__version__ != _V2_DIET_TRITON:
        return f"triton {triton.__version__} (verified on {_V2_DIET_TRITON})"
    arch = _V2_DIET_ARCH.get(q.device)
    if arch is None:
        props = torch.cuda.get_device_properties(q.device)
        arch = _V2_DIET_ARCH[q.device] = str(getattr(props, "gcnArchName", ""))
    if not arch.startswith("gfx950"):
        return f"arch {arch!r}"
    return None


@triton.jit
def _hstu_attn_delta_bwd_e5m2_v2(
    Q,
    K,
    V,
    seq_offsets,
    DOut,
    DQ,
    DVK8,
    PARTIAL_COL,
    PARTIAL_AMAX,
    INV_SCALE,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_dom,
    stride_doh,
    stride_dqm,
    stride_dqh,
    stride_dvk,
    alpha,
    K_SCALE,
    V_SCALE,
    H,
    MAX_SEQ_LEN,
    COL_K0,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LIMIT: tl.constexpr,
    REDUCE_EACH: tl.constexpr,
    PERM,
    USE_PERM: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    AMAX_ROWMAX: tl.constexpr,
    DQ_ELEM: tl.constexpr,
):
    """`_hstu_attn_delta_bwd_e5m2` v2 (DLRM_HSTU_FP8_BWD_VKP_V2, default off).

    Same outputs and layout as the shipped kernel. AMAX_ROWMAX: the amax from
    running row-maxima of |silu| and |dqk| (exact; payload, amax and colsum
    bitwise). DQ_ELEM: element-wise dq accumulator, one cross-row reduction
    after the loop (DECLARED fp32-order change of dq). USE_PERM: program ->
    sequence through PERM (longest first); outputs are per sequence, so the
    order does not change any value.
    """
    off_hz = tl.program_id(0)
    if USE_PERM:
        # longest-first (LPT) sequence order; outputs are per-sequence, order-free
        off_z = tl.load(PERM + off_hz // H)
    else:
        off_z = off_hz // H
    off_h = (off_hz % H).to(tl.int64)
    off_hz = off_z * H + off_h
    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)
    inv = tl.load(INV_SCALE)

    offs_q = tl.arange(0, BLOCK_D_Q)
    offs_v = tl.arange(0, BLOCK_D_V)
    q = tl.load(
        Q + off_z * stride_qm + off_h * stride_qh + offs_q,
        mask=offs_q < BLOCK_D_Q,
        other=0.0,
    ).to(tl.float32)
    dout = tl.load(
        DOut + off_z * stride_dom + off_h * stride_doh + offs_v,
        mask=offs_v < BLOCK_D_V,
        other=0.0,
    ).to(tl.float32)
    if DQ_ELEM:
        # DECLARED fp32-order change: element-wise [BLOCK_N, D] accumulator over
        # the tiles, one cross-row reduction after the loop (instead of one
        # cross-warp reduction per tile, summed in tile order).
        dq_acc = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
    else:
        dq = tl.zeros([BLOCK_D_Q], dtype=tl.float32)
    if AMAX_ROWMAX:
        # Exact: dv = s_n*dout_d and dk = (dqk_n*q_d)*alpha are products of bf16
        # values (exact in fp32) followed by monotone roundings, hence
        # max|dv_b| = bf16(max|s| * max|dout|), max|dk_b| = bf16((max|dqk| * max|q|) * |alpha|) (RNE is sign-symmetric).
        m_s = tl.zeros([BLOCK_N], dtype=tl.float32)
        m_d = tl.zeros([BLOCK_N], dtype=tl.float32)
    # REDUCE_EACH=False: element-wise register accumulators reduced ONCE after
    # the loop; True: [D] vectors reduced every block (fewer registers, one
    # cross-warp reduction per block). Same values either way up to fp32 order.
    if REDUCE_EACH:
        col_k = tl.zeros([BLOCK_D_Q], dtype=tl.float32)
        col_v = tl.zeros([BLOCK_D_V], dtype=tl.float32)
        amax_k = tl.zeros([BLOCK_D_Q], dtype=tl.float32)
        amax_v = tl.zeros([BLOCK_D_V], dtype=tl.float32)
    else:
        col_k = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
        col_v = tl.zeros([BLOCK_N, BLOCK_D_V], dtype=tl.float32)
        if not AMAX_ROWMAX:
            amax_k = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
            amax_v = tl.zeros([BLOCK_N, BLOCK_D_V], dtype=tl.float32)

    for start_n in tl.range(0, seq_len, BLOCK_N, num_stages=NUM_STAGES):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len
        k_ptrs = (
            K
            + (seq_start + offs_n[:, None]) * stride_kn
            + off_h * stride_kh
            + offs_q[None, :]
        )
        v_ptrs = (
            V
            + (seq_start + offs_n[:, None]) * stride_vn
            + off_h * stride_vh
            + offs_v[None, :]
        )
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        k_real = (k.to(tl.float32) * K_SCALE).to(tl.bfloat16).to(tl.float32)
        v_real = (v.to(tl.float32) * V_SCALE).to(tl.bfloat16).to(tl.float32)

        qk = tl.sum(k_real * q[None, :], axis=1) * alpha
        sig = fast_dividef(1.0, 1.0 + tl.exp(-qk))
        silu = fast_dividef(qk, 1.0 + fast_expf(-qk)) * (1.0 / MAX_SEQ_LEN)
        silu = tl.where(mask_n, silu, 0.0).to(DOut.dtype.element_ty)

        dv = silu[:, None].to(tl.float32) * dout[None, :]
        dqk = tl.sum(v_real * dout[None, :], axis=1)
        dqk *= sig * (1.0 + qk * (1.0 - sig)) * (1.0 / MAX_SEQ_LEN)
        dqk = tl.where(mask_n, dqk, 0.0).to(tl.bfloat16)
        dk = dqk[:, None].to(tl.float32) * q[None, :] * alpha
        if DQ_ELEM:
            dq_acc += dqk[:, None].to(tl.float32) * k_real
        else:
            dq += tl.sum(dqk[:, None].to(tl.float32) * k_real, axis=0)

        # The value the stock kernel stores (BF16), back in fp32: what the
        # stock cast kernel would have loaded.
        dk_b = dk.to(tl.bfloat16).to(tl.float32)
        dv_b = dv.to(tl.bfloat16).to(tl.float32)
        # Masked rows are exactly 0 (silu/dqk zeroed above), so they add
        # nothing to the sums or the max.
        if REDUCE_EACH:
            col_k += tl.sum(dk_b, axis=0)
            col_v += tl.sum(dv_b, axis=0)
            amax_k = tl.maximum(amax_k, tl.max(tl.abs(dk_b), axis=0))
            amax_v = tl.maximum(amax_v, tl.max(tl.abs(dv_b), axis=0))
        else:
            col_k += dk_b
            col_v += dv_b
            if AMAX_ROWMAX:
                m_s = tl.maximum(m_s, tl.abs(silu.to(tl.float32)))
                m_d = tl.maximum(m_d, tl.abs(dqk.to(tl.float32)))
            else:
                amax_k = tl.maximum(amax_k, tl.abs(dk_b))
                amax_v = tl.maximum(amax_v, tl.abs(dv_b))
        dk8 = tl.minimum(tl.maximum(dk_b * inv, -LIMIT), LIMIT)
        dv8 = tl.minimum(tl.maximum(dv_b * inv, -LIMIT), LIMIT)

        row_ptr = DVK8 + (seq_start + offs_n[:, None]) * stride_dvk
        tl.store(
            row_ptr + off_h * BLOCK_D_V + offs_v[None, :],
            dv8.to(DVK8.dtype.element_ty),
            mask=mask_n[:, None],
        )
        tl.store(
            row_ptr + COL_K0 + off_h * BLOCK_D_Q + offs_q[None, :],
            dk8.to(DVK8.dtype.element_ty),
            mask=mask_n[:, None],
        )

    if DQ_ELEM:
        dq = tl.sum(dq_acc, axis=0)
    tl.store(
        DQ + off_z * stride_dqm + off_h * stride_dqh + offs_q,
        (dq * alpha).to(DQ.dtype.element_ty),
        mask=offs_q < BLOCK_D_Q,
    )
    col_row = PARTIAL_COL + off_z.to(tl.int64) * stride_dvk
    if REDUCE_EACH:
        tl.store(col_row + off_h * BLOCK_D_V + offs_v, col_v)
        tl.store(col_row + COL_K0 + off_h * BLOCK_D_Q + offs_q, col_k)
        tl.store(
            PARTIAL_AMAX + off_hz,
            tl.maximum(tl.max(amax_k, axis=0), tl.max(amax_v, axis=0)),
        )
    else:
        tl.store(col_row + off_h * BLOCK_D_V + offs_v, tl.sum(col_v, axis=0))
        tl.store(col_row + COL_K0 + off_h * BLOCK_D_Q + offs_q, tl.sum(col_k, axis=0))
        if AMAX_ROWMAX:
            av = (tl.max(m_s, axis=0) * tl.max(tl.abs(dout), axis=0)).to(tl.bfloat16).to(tl.float32)
            ak = ((tl.max(m_d, axis=0) * tl.max(tl.abs(q), axis=0)) * tl.abs(alpha)).to(tl.bfloat16).to(tl.float32)
            tl.store(PARTIAL_AMAX + off_hz, tl.maximum(ak, av))
        else:
            tl.store(
                PARTIAL_AMAX + off_hz,
                tl.maximum(
                    tl.max(tl.max(amax_k, axis=1), axis=0),
                    tl.max(tl.max(amax_v, axis=1), axis=0),
                ),
            )


def hstu_attention_delta_bwd_e5m2(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dq: torch.Tensor,
    dvk_fp8: torch.Tensor,
    rows: int,
    seq_offsets: torch.Tensor,
    N: int,
    alpha: float,
    inv_scale: torch.Tensor,
    limit: float,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    block_n: int = _BLOCK_N,
    num_warps: int = _NUM_WARPS,
    reduce_each: bool = _REDUCE_EACH,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Delta attention backward writing dQ (BF16) and the E5M2 dV|dK pack.

    `dvk_fp8` is a contiguous [Mp, H*D_V + H*D_Q] FP8 tensor, Mp >= rows; rows
    [rows, Mp) are zeroed here (K padding for the token-major wgrad). Returns
    (amax, colsum fp32 [n_cols]) of the BF16 dV|dK the stock kernel would have
    stored -- the same triple `bwd_fp8.quantize_amax_colsum` returns.
    """
    assert dvk_fp8.dim() == 2 and dvk_fp8.is_contiguous()
    # Same input handling as the stock wrapper: the kernel takes row/head
    # strides, only the innermost dimension must be unit-stride.
    dout = switch_to_contiguous_if_needed(dout)
    q = switch_to_contiguous_if_needed(q)
    k = switch_to_contiguous_if_needed(k)
    v = switch_to_contiguous_if_needed(v)
    assert dq.stride(-1) == 1
    Z = seq_offsets.numel() - 1
    _, H, DimQ = q.shape
    _, _, DimV = v.shape
    n_cols = H * (DimQ + DimV)
    assert dvk_fp8.shape[1] == n_cols and dvk_fp8.shape[0] >= rows
    assert k.shape[0] == rows and v.shape[0] == rows
    if q.shape[0] != Z or dout.shape[0] != Z:
        raise ValueError("delta attention training requires one query per sequence")
    if dvk_fp8.shape[0] > rows:
        dvk_fp8[rows:].zero_()
    partial_col = torch.empty((max(Z, 1), n_cols), dtype=torch.float32, device=q.device)
    partial_amax = torch.empty(max(Z * H, 1), dtype=torch.float32, device=q.device)
    if Z == 0:
        partial_col.zero_()
        partial_amax.zero_()
        return partial_amax.amax(), partial_col.sum(dim=0)
    if _VKP_V2:
        lpt = _VKP_V2_LPT
        perm = seq_offsets  # placeholder pointer when USE_PERM is False
        if lpt == 2 and Z <= 4096 and N < (1 << 19):
            # longest-processing-time-first launch order (one tiny sort kernel)
            bz = max(16, triton.next_power_of_2(Z))
            perm = torch.empty(Z, dtype=torch.int32, device=q.device)
            _hstu_attn_delta_bwd_e5m2_lpt_perm[(1,)](
                seq_offsets, perm, Z, BLOCK_Z=bz, ZBITS=bz.bit_length() - 1, num_warps=4
            )
        elif lpt:
            lens = seq_offsets[1:] - seq_offsets[:-1]
            perm = torch.argsort(lens, descending=True, stable=True).to(torch.int32)
        if delta_bwd_v2_diet_requested():
            why = _v2_diet_decline_reason(
                q, k, v, dout, dq, dvk_fp8, limit, block_n, num_warps, reduce_each
            )
            if why is None:
                # Item (48)(b'): same grid, PERM and outputs as the v2 launch.
                from generative_recommenders.ops.triton import (
                    hstu_delta_bwd_e5m2_v2_diet as _v2_diet,
                )

                if not _V2_DIET_LOGGED[0]:
                    _V2_DIET_LOGGED[0] = True
                    print(
                        "[hstu_delta_bwd_e5m2] DLRM_HSTU_FP8_DELTA_BWD_V2_DIET=1 "
                        "armed: VALU-diet kernel _hstu_attn_delta_bwd_e5m2_v2_diet "
                        "(bitwise vs _hstu_attn_delta_bwd_e5m2_v2), (stages, peel, "
                        f"sp)={_v2_diet.default_config(k, v, dvk_fp8)}",
                        flush=True,
                    )
                _v2_diet.launch_kernel(
                    dout, q, k, v, dq, dvk_fp8, seq_offsets, N, alpha, inv_scale,
                    limit, k_scale, v_scale, partial_col, partial_amax, perm,
                    bool(lpt),
                )
                _log_vkp_v2_armed(q.device, Z, H, rows, lpt)
                return partial_amax.amax(), partial_col.sum(dim=0)
            if why not in _V2_DIET_DECLINE_LOGGED:
                _V2_DIET_DECLINE_LOGGED.add(why)
                print(
                    "[hstu_delta_bwd_e5m2] DLRM_HSTU_FP8_DELTA_BWD_V2_DIET=1 "
                    f"declined ({why}): v2 kernel",
                    flush=True,
                )
        _hstu_attn_delta_bwd_e5m2_v2[(Z * H,)](
            Q=q,
            K=k,
            V=v,
            seq_offsets=seq_offsets,
            DOut=dout,
            DQ=dq,
            DVK8=dvk_fp8,
            PARTIAL_COL=partial_col,
            PARTIAL_AMAX=partial_amax,
            INV_SCALE=inv_scale,
            stride_qm=q.stride(0),
            stride_qh=q.stride(1),
            stride_kn=k.stride(0),
            stride_kh=k.stride(1),
            stride_vn=v.stride(0),
            stride_vh=v.stride(1),
            stride_dom=dout.stride(0),
            stride_doh=dout.stride(1),
            stride_dqm=dq.stride(0),
            stride_dqh=dq.stride(1),
            stride_dvk=dvk_fp8.stride(0),
            alpha=alpha,
            K_SCALE=k_scale,
            V_SCALE=v_scale,
            H=H,
            MAX_SEQ_LEN=N,
            COL_K0=H * DimV,
            BLOCK_D_Q=DimQ,
            BLOCK_D_V=DimV,
            BLOCK_N=block_n,
            LIMIT=limit,
            REDUCE_EACH=reduce_each,
            PERM=perm,
            USE_PERM=bool(lpt),
            NUM_STAGES=_VKP_V2_NUM_STAGES,
            AMAX_ROWMAX=_VKP_V2_AMAX_ROWMAX,
            DQ_ELEM=_VKP_V2_DQ_ELEM,
            num_warps=num_warps,
        )
        _log_vkp_v2_armed(q.device, Z, H, rows, lpt)
        return partial_amax.amax(), partial_col.sum(dim=0)
    _hstu_attn_delta_bwd_e5m2[(Z * H,)](
        Q=q,
        K=k,
        V=v,
        seq_offsets=seq_offsets,
        DOut=dout,
        DQ=dq,
        DVK8=dvk_fp8,
        PARTIAL_COL=partial_col,
        PARTIAL_AMAX=partial_amax,
        INV_SCALE=inv_scale,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_dom=dout.stride(0),
        stride_doh=dout.stride(1),
        stride_dqm=dq.stride(0),
        stride_dqh=dq.stride(1),
        stride_dvk=dvk_fp8.stride(0),
        alpha=alpha,
        K_SCALE=k_scale,
        V_SCALE=v_scale,
        H=H,
        MAX_SEQ_LEN=N,
        COL_K0=H * DimV,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        BLOCK_N=block_n,
        LIMIT=limit,
        REDUCE_EACH=reduce_each,
        num_warps=num_warps,
    )
    return partial_amax.amax(), partial_col.sum(dim=0)
