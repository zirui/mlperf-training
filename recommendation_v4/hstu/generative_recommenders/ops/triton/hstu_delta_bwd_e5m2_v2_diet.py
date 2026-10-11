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

"""DLRM_HSTU_FP8_DELTA_BWD_V2_DIET: VALU-diet form of the VKP_V2 delta backward.

Bitwise drop-in for `hstu_delta_bwd_e5m2._hstu_attn_delta_bwd_e5m2_v2`
(DLRM_HSTU_FP8_BWD_VKP_V2) in the configuration it was verified in:
the v2 module constants AMAX_ROWMAX=1, DQ_ELEM=1, LPT=2, NUM_STAGES=1, and
REDUCE_EACH=False, BLOCK_N=32, num_warps=4, D_Q = D_V = 128, bf16 q/dout/dQ,
e4m3 K/V, e5m2 payload, gfx950, Triton 3.6.0.
`hstu_delta_bwd_e5m2.hstu_attention_delta_bwd_e5m2` dispatches here only for
exactly that configuration and keeps the v2 kernel for everything else (and
behind DLRM_HSTU_FP8_DELTA_BWD_V2_DIET=0 / absent).

Outputs are byte-identical to the v2 kernel: payload incl. zeroed pad rows,
dQ, PARTIAL_COL, PARTIAL_AMAX and the returned amax / colsum, for both
Triton builds (<= 2 GiB buffer-op and > 2 GiB 64-bit global-memory),
including NaN / sNaN / inf / -0 inputs, product overflow and subnormals,
with NaN payload and sign bits included.

v2 already has the row-max amax and the element-wise dq accumulator. The
remaining VALU work per [32, 128] tile is reduced, each change value-preserving:
 * bf16 round trips (k_real, v_real, silu, dqk, dK_b, dV_b): one
   `v_cvt_pk_bf16_f32 d, 0, x` per element (= f32(bf16_rne(x))) instead of
   the conversion plus an unpack.
 * fp32 products as explicit `v_pk_mul_f32` in the v2 build's operand order;
   the clamp is `v_med3_f32 x, -L, L`.
 * the qk / dqk per-thread partial sums in asm, reproducing the v2 build's
   exact fused/unfused sequence and operand order (read off its ISA/IR):
   LLVM would otherwise pick its own split once the surrounding code changes,
   which differs whenever a product overflows or is subnormal, and the
   operand order decides which NaN payload propagates. The cross-lane
   reduction is the stock tl.sum (with a bitcast -0.0 filler: a `-0.0`
   literal reaches LLVM as +0.0 and would turn a -0.0 partial into +0.0).
 * `1 + qk*(1-sig)` keeps its unfused multiply (asm), as v2 does.
 * the running row maxima of |silu| / |dqk| as one `v_max_f32 acc, acc, |x|`.
 * scalar row addressing; in the > 2 GiB (in-model) build the masked tail
   tile runs after the unmasked full tiles (v2 tile order), with the uniform
   multiplier pairs in SGPRs.
Program -> sequence mapping (LPT PERM) and the v2 epilogue (dq / colsum
reductions, bf16(max m_s * max|dout|), bf16((max m_d * max|q|) * |alpha|))
are v2 verbatim.
Bitwise equality depends on asm that mirrors the current Triton/LLVM's
choices for the v2 kernel: re-verify after any Triton or LLVM change (the
dispatcher declines on any other Triton version).
"""
import torch
import triton
import triton.language as tl

from generative_recommenders.ops.triton.triton_hstu_attention import (
    fast_dividef,
    fast_expf,
)


def _asm_from_spec(spec):
    # Per-thread partial sum over its 16 elements. spec: ("mul0", j) acc = fl(a_j*b_j); ("fma", j)
    # acc = fma(a_j, b_j, acc); ("add", j, acc_first) acc = acc + fl(a_j*b_j) (acc_first: add operand order).
    # $0 acc, $1 scratch; $16+j b_j (k_j / v_j); $32+j a_j (q_j / dout_j).
    a = []
    for s in spec:
        kind, j = s[0], s[1]
        if kind == "mul0":
            a.append(f"v_mul_f32 $0, ${32 + j}, ${16 + j}")
        elif kind == "fma":
            a.append(f"v_fmac_f32 $0, ${32 + j}, ${16 + j}")
        elif kind == "add":
            a.append(f"v_mul_f32 $1, ${32 + j}, ${16 + j}")
            a.append("v_add_f32 $0, $0, $1" if s[2] else "v_add_f32 $0, $1, $0")
        else:
            raise ValueError(s)
    return "\n".join(a)


# The v2 build's sequences (both the <= 2 GiB and the > 2 GiB build, Triton 3.6.0 / gfx950): qk = 16 rounded
# products summed in order; dqk = fl(d1*v1), fma j = 0, 2..5, then rounded products 6..15. Every add takes the
# product as src0 except the first qk add.
_QK_SPEC = [("mul0", 0), ("add", 1, True)] + [("add", j, False) for j in range(2, 16)]
_DQK_SPEC = [("mul0", 1), ("fma", 0), ("fma", 2), ("fma", 3), ("fma", 4), ("fma", 5)] + [
    ("add", j, False) for j in range(6, 16)]
_ASM_QK = tl.constexpr(_asm_from_spec(_QK_SPEC))
_ASM_DQK = tl.constexpr(_asm_from_spec(_DQK_SPEC))
_RED_CONSTRAINTS = tl.constexpr(",".join(["=&v", "=&v"] + ["=v"] * 14 + ["v"] * 32))


@triton.jit
def _bf16_rt(x):
    return tl.inline_asm_elementwise(
        "v_cvt_pk_bf16_f32 $0, 0, $1", "=v,v", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _mul_nc(a, b):
    return tl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2", "=v,v,v", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _vmax_abs(acc, x):
    return tl.inline_asm_elementwise(
        "v_max_f32_e64 $0, $1, |$2|", "=v,v,v", [acc, x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _med3(x, lo, hi):
    return tl.inline_asm_elementwise(
        "v_med3_f32 $0, $1, $2, $3", "=v,v,v,v", [x, lo, hi], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _pair(x):
    x0, x1 = tl.split(tl.reshape(x, [x.shape[0], x.shape[1] // 2, 2]))
    return x0.to(tl.uint32, bitcast=True).to(tl.uint64) | (
        x1.to(tl.uint32, bitcast=True).to(tl.uint64) << 32
    )


@triton.jit
def _unpair(w):
    r0 = w.to(tl.uint32).to(tl.float32, bitcast=True)
    r1 = (w >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    j = tl.join(r0, r1)
    return tl.reshape(j, [j.shape[0], j.shape[1] * 2])


@triton.jit
def _pkmul(wa, wb, SP: tl.constexpr):
    if SP:
        # uniform first operand in an SGPR pair (as the v2 build's `v_pk_mul_f32 v, s[..], v`)
        return tl.inline_asm_elementwise(
            "v_pk_mul_f32 $0, $1, $2", "=v,s,v", [wa, wb], dtype=tl.uint64, is_pure=True, pack=1
        )
    else:
        return tl.inline_asm_elementwise(
            "v_pk_mul_f32 $0, $1, $2", "=v,v,v", [wa, wb], dtype=tl.uint64, is_pure=True, pack=1
        )


@triton.jit
def _pkmul_vv(wa, wb):
    return tl.inline_asm_elementwise(
        "v_pk_mul_f32 $0, $1, $2", "=v,v,v", [wa, wb], dtype=tl.uint64, is_pure=True, pack=1
    )


@triton.jit
def _red16(a, b, ASM: tl.constexpr, CONS: tl.constexpr, first):
    # one asm per thread over its 16 elements; slot 0 of each chunk = partial sum, other slots -> -0.0 so that
    # x + -0.0 = x folds away in tl.sum's in-thread chain (the cross-lane DPP reduction is the stock's).
    # The -0.0 is built by bitcast (0x80000000): a `-0.0` literal reaches LLVM as +0.0 and leaves one
    # `v_add_f32 x, 0, x` per chain, which turns a -0.0 partial sum into +0.0 (v2 keeps -0.0).
    r = tl.inline_asm_elementwise(ASM, CONS, [a, b], dtype=tl.float32, is_pure=True, pack=16)
    neg0 = tl.full(r.shape, -2147483648, tl.int32).to(tl.float32, bitcast=True)
    return tl.sum(tl.where(first, r, neg0), axis=1)


@triton.jit
def _tile(
    K, V, DVK8, k_row, v_row, o_row, kcol, vcol, ocol_v, ocol_k, mask_n,
    q, dout, dob, qb, first, q_p, do_p, ks_p, vs_p, al_p, inv_p, lo_c, hi_c, alpha, MAX_SEQ_LEN, inv,
    dq_acc, col_k, col_v, m_d, m_s,
    BLOCK_N: tl.constexpr, BLOCK_D_Q: tl.constexpr, BLOCK_D_V: tl.constexpr,
    MASKED: tl.constexpr, SP: tl.constexpr, SEL: tl.constexpr,
):
    if MASKED:
        k = tl.load(K + k_row + kcol, mask=mask_n[:, None], other=0.0)
        v = tl.load(V + v_row + vcol, mask=mask_n[:, None], other=0.0)
    else:
        k = tl.load(K + k_row + kcol)
        v = tl.load(V + v_row + vcol)
    # SEL: apply the row-mask selects to silu/dqk (v2 arithmetic); False only for the PEEL=1 full tiles
    k_real = _bf16_rt(_unpair(_pkmul(ks_p, _pair(k.to(tl.float32)), SP)))
    v_real = _bf16_rt(_unpair(_pkmul(vs_p, _pair(v.to(tl.float32)), SP)))
    qk = _red16(k_real, qb, _ASM_QK, _RED_CONSTRAINTS, first) * alpha
    sig = fast_dividef(1.0, 1.0 + tl.exp(-qk))
    silu = fast_dividef(qk, 1.0 + fast_expf(-qk)) * (1.0 / MAX_SEQ_LEN)
    if SEL:
        silu = tl.where(mask_n, silu, 0.0)
    silu = _bf16_rt(silu)
    dqk = _red16(v_real, dob, _ASM_DQK, _RED_CONSTRAINTS, first)
    dqk *= sig * (1.0 + _mul_nc(qk, 1.0 - sig)) * (1.0 / MAX_SEQ_LEN)
    if SEL:
        dqk = tl.where(mask_n, dqk, 0.0)
    dqk = _bf16_rt(dqk)
    dq_acc = tl.fma(tl.broadcast_to(dqk[:, None], [BLOCK_N, BLOCK_D_Q]), k_real, dq_acc)  # v2: fused
    # output path: same instructions / operand order as v2, minus the bf16 unpack
    dv = _unpair(_pkmul_vv(_pair(tl.broadcast_to(silu[:, None], [BLOCK_N, BLOCK_D_V])), do_p))
    dk = _unpair(
        _pkmul(al_p, _pkmul_vv(_pair(tl.broadcast_to(dqk[:, None], [BLOCK_N, BLOCK_D_Q])), q_p), SP)
    )
    dk_b = _bf16_rt(dk)
    dv_b = _bf16_rt(dv)
    col_k += dk_b
    col_v += dv_b
    m_d = _vmax_abs(m_d, dqk)
    m_s = _vmax_abs(m_s, silu)
    dk8 = _med3(_unpair(_pkmul(inv_p, _pair(dk_b), SP)), lo_c, hi_c)
    dv8 = _med3(_unpair(_pkmul(inv_p, _pair(dv_b), SP)), lo_c, hi_c)
    if MASKED:
        tl.store(DVK8 + o_row + ocol_v, dv8.to(DVK8.dtype.element_ty), mask=mask_n[:, None])
        tl.store(DVK8 + o_row + ocol_k, dk8.to(DVK8.dtype.element_ty), mask=mask_n[:, None])
    else:
        tl.store(DVK8 + o_row + ocol_v, dv8.to(DVK8.dtype.element_ty))
        tl.store(DVK8 + o_row + ocol_k, dk8.to(DVK8.dtype.element_ty))
    return dq_acc, col_k, col_v, m_d, m_s


@triton.jit
def _hstu_attn_delta_bwd_e5m2_v2_diet(
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
    PERM,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LIMIT: tl.constexpr,
    USE_PERM: tl.constexpr,
    STAGES: tl.constexpr,
    PEEL: tl.constexpr,
    SP: tl.constexpr,
):
    off_hz = tl.program_id(0)
    if USE_PERM:
        # longest-first (LPT) sequence order; outputs are per-sequence, order-free (v2 verbatim)
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
    dq_acc = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
    m_s = tl.zeros([BLOCK_N], dtype=tl.float32)
    m_d = tl.zeros([BLOCK_N], dtype=tl.float32)
    col_k = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
    col_v = tl.zeros([BLOCK_N, BLOCK_D_V], dtype=tl.float32)

    ks_p = _pair(tl.full([BLOCK_N, BLOCK_D_Q], K_SCALE, tl.float32))
    vs_p = _pair(tl.full([BLOCK_N, BLOCK_D_V], V_SCALE, tl.float32))
    al_p = _pair(tl.full([BLOCK_N, BLOCK_D_Q], alpha, tl.float32))
    inv_p = _pair(tl.full([BLOCK_N, BLOCK_D_Q], inv, tl.float32))
    q_p = _pair(tl.broadcast_to(q[None, :], [BLOCK_N, BLOCK_D_Q]))
    do_p = _pair(tl.broadcast_to(dout[None, :], [BLOCK_N, BLOCK_D_V]))
    dob = tl.broadcast_to(dout[None, :], [BLOCK_N, BLOCK_D_V])
    qb = tl.broadcast_to(q[None, :], [BLOCK_N, BLOCK_D_Q])
    first = ((offs_q % 16) == 0)[None, :]
    lo_c = tl.full([BLOCK_N, BLOCK_D_Q], -LIMIT, tl.float32)
    hi_c = tl.full([BLOCK_N, BLOCK_D_Q], LIMIT, tl.float32)

    rows = tl.arange(0, BLOCK_N)
    k_row0 = (seq_start + rows[:, None]) * stride_kn
    v_row0 = (seq_start + rows[:, None]) * stride_vn
    o_row0 = (seq_start + rows[:, None]) * stride_dvk
    kcol = off_h * stride_kh + offs_q[None, :]
    vcol = off_h * stride_vh + offs_v[None, :]
    ocol_v = off_h * BLOCK_D_V + offs_v[None, :]
    ocol_k = COL_K0 + off_h * BLOCK_D_Q + offs_q[None, :]

    if PEEL >= 1:
        n_full = (seq_len // BLOCK_N) * BLOCK_N
        for start_n in tl.range(0, n_full, BLOCK_N, num_stages=STAGES):
            dq_acc, col_k, col_v, m_d, m_s = _tile(
                K, V, DVK8, k_row0 + start_n * stride_kn, v_row0 + start_n * stride_vn,
                o_row0 + start_n * stride_dvk, kcol, vcol, ocol_v, ocol_k, (start_n + rows) < seq_len,
                q, dout, dob, qb, first, q_p, do_p, ks_p, vs_p, al_p, inv_p, lo_c, hi_c, alpha, MAX_SEQ_LEN, inv,
                dq_acc, col_k, col_v, m_d, m_s,
                BLOCK_N, BLOCK_D_Q, BLOCK_D_V, False, SP, PEEL == 2,
            )
        if n_full < seq_len:
            # the partial last tile, masked, after all full tiles (v2 tile order)
            dq_acc, col_k, col_v, m_d, m_s = _tile(
                K, V, DVK8, k_row0 + n_full * stride_kn, v_row0 + n_full * stride_vn,
                o_row0 + n_full * stride_dvk, kcol, vcol, ocol_v, ocol_k, (n_full + rows) < seq_len,
                q, dout, dob, qb, first, q_p, do_p, ks_p, vs_p, al_p, inv_p, lo_c, hi_c, alpha, MAX_SEQ_LEN, inv,
                dq_acc, col_k, col_v, m_d, m_s,
                BLOCK_N, BLOCK_D_Q, BLOCK_D_V, True, SP, True,
            )
    else:
        for start_n in tl.range(0, seq_len, BLOCK_N, num_stages=STAGES):
            dq_acc, col_k, col_v, m_d, m_s = _tile(
                K, V, DVK8, k_row0 + start_n * stride_kn, v_row0 + start_n * stride_vn,
                o_row0 + start_n * stride_dvk, kcol, vcol, ocol_v, ocol_k, (start_n + rows) < seq_len,
                q, dout, dob, qb, first, q_p, do_p, ks_p, vs_p, al_p, inv_p, lo_c, hi_c, alpha, MAX_SEQ_LEN, inv,
                dq_acc, col_k, col_v, m_d, m_s,
                BLOCK_N, BLOCK_D_Q, BLOCK_D_V, True, SP, True,
            )

    # ---- v2 epilogue (verbatim math)
    dq = tl.sum(dq_acc, axis=0)
    tl.store(
        DQ + off_z * stride_dqm + off_h * stride_dqh + offs_q,
        (dq * alpha).to(DQ.dtype.element_ty),
        mask=offs_q < BLOCK_D_Q,
    )
    col_row = PARTIAL_COL + off_z.to(tl.int64) * stride_dvk
    tl.store(col_row + off_h * BLOCK_D_V + offs_v, tl.sum(col_v, axis=0))
    tl.store(col_row + COL_K0 + off_h * BLOCK_D_Q + offs_q, tl.sum(col_k, axis=0))
    av = (tl.max(m_s, axis=0) * tl.max(tl.abs(dout), axis=0)).to(tl.bfloat16).to(tl.float32)
    ak = ((tl.max(m_d, axis=0) * tl.max(tl.abs(q), axis=0)) * tl.abs(alpha)).to(tl.bfloat16).to(tl.float32)
    tl.store(PARTIAL_AMAX + off_hz, tl.maximum(ak, av))


_MAX_INT_32 = 2**31 - 1
# (stages, peel, sp) per Triton build:
# small (<= 2 GiB storages: buffer-op build) st2+peel0+sp0;
# big (> 2 GiB: 64-bit global-memory build, the in-model case) st1+peel1+sp1.
DEFAULT_SMALL = (2, 0, 0)
DEFAULT_BIG = (1, 1, 1)


def default_config(k: torch.Tensor, v: torch.Tensor, dvk_fp8: torch.Tensor):
    small = all(t.untyped_storage().size() <= _MAX_INT_32 for t in (k, v, dvk_fp8))
    return DEFAULT_SMALL if small else DEFAULT_BIG


def launch_kernel(dout, q, k, v, dq, dvk_fp8, seq_offsets, N, alpha, inv_scale, limit, k_scale, v_scale,
                  partial_col, partial_amax, perm, use_perm):
    """The v2 launch of `hstu_attention_delta_bwd_e5m2` (same grid, same PERM / USE_PERM) with this kernel.
    Inputs as the v2 branch has them (contiguous innermost dims; outputs allocated by the caller)."""
    Z = seq_offsets.numel() - 1
    _, H, DimQ = q.shape
    DimV = v.shape[2]
    stages, peel, sp = default_config(k, v, dvk_fp8)
    return _hstu_attn_delta_bwd_e5m2_v2_diet[(Z * H,)](
        Q=q, K=k, V=v, seq_offsets=seq_offsets, DOut=dout, DQ=dq, DVK8=dvk_fp8,
        PARTIAL_COL=partial_col, PARTIAL_AMAX=partial_amax, INV_SCALE=inv_scale,
        stride_qm=q.stride(0), stride_qh=q.stride(1), stride_kn=k.stride(0), stride_kh=k.stride(1),
        stride_vn=v.stride(0), stride_vh=v.stride(1), stride_dom=dout.stride(0), stride_doh=dout.stride(1),
        stride_dqm=dq.stride(0), stride_dqh=dq.stride(1), stride_dvk=dvk_fp8.stride(0),
        alpha=alpha, K_SCALE=k_scale, V_SCALE=v_scale, H=H, MAX_SEQ_LEN=N, COL_K0=H * DimV, PERM=perm,
        BLOCK_D_Q=DimQ, BLOCK_D_V=DimV, BLOCK_N=32, LIMIT=limit, USE_PERM=use_perm, STAGES=stages, PEEL=peel,
        SP=sp, num_warps=4,
    )
