# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""HSTU jagged-attention BACKWARD dk/dv pass written in FlyDSL for gfx950.

Structure (mirrors the Triton production split: a dk/dv pass + a separate dq pass):

    per (batch, head, n-block):
      load K,V tile (BN x D) once into registers
      for m0 in [n0, seq_len) step BM:
          stage Q,DO tiles in LDS (row-major + transposed)
          qk   = q  @ k^T                 (mfma, acc in f32)
          silu = qk*sigmoid(qk)/MAX_SEQ   (masked: causal + m<seq_len)
          dv  += silu^T @ do              (silu^T is the f32 acc reinterpreted
                                           as an mfma A-operand -- free transpose)
          dp   = do @ v^T
          ds   = dp * sigmoid'(...)       (masked)
          dk  += ds^T @ q
      store dk*alpha, dv

This covers four of the five HSTU backward GEMMs.  dq is NOT computed here.

Offsets: the per-(batch,head) base pointer is advanced in 64-bit before the
buffer resource is built, so every in-kernel voffset stays < 2^31 even though
the full tensors are multi-GB.
"""

import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm, memref
from flydsl._mlir.dialects import rocdl as rocdl_raw
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.compiler.protocol import fly_values
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from generative_recommenders.ops.flydsl import current_stream_launch

LOG2E = 1.4426950408889634

# The shipping configuration, in one place so the benchmark, the correctness
# gate and the autograd wrapper cannot drift apart.
#
# dq16="bf16p" accumulates dq_acc in bf16 through packed atomics.  It is not a
# micro-optimisation: the fused kernel's runtime IS its dq atomic stream, the
# atomic unit is limited per 64 B line touched (atomic_roof.hip), and bf16p is
# the only 16-bit layout that actually halves the lines rather than just the
# bytes.  51.45 -> 37.29 ms.  It costs dq precision: 55.5 -> 45.2 dB against the
# dense fp32 reference, still clear of the 40 dB per-output gate, and dq is a
# bf16 output either way.
#
# sched="iglp2": iglp0 was tuned on the dk/dv-only body, which is now two thirds
# of this one.  Re-swept after bf16p moved the bottleneck off the atomic stream:
# iglp0 36.98, iglp1 36.13, iglp2 35.04 ms.
SHIPPED = {"BN": 128, "fused": True, "dq16": "bf16p", "sched": "iglp2"}


def dq_acc_dtype(dq16=None):
    """torch dtype the dq accumulator must have for a given dq16 mode."""
    import torch

    mode = SHIPPED["dq16"] if dq16 is None else dq16
    return {"": torch.float32, "f16": torch.float16}.get(mode, torch.bfloat16)

def cvt_scale_is_exact(scale):
    """Does `scale` survive `cvt_scalef32_pk_bf16_fp8`'s scale operand intact?

    Only if it is an exact power of two.  That operand is **exponent-only** on
    gfx950 (E8M0): the hardware keeps the biased exponent and discards the
    mantissa, so 2.234e-4 becomes 2**-13 = 1.2207e-4 and every value converted
    through it comes back scaled by a constant 1/1.830.  Nothing traps, nothing
    warns -- the kernel simply returns wrong numbers.  Measured on the
    `q_bf16` + `scale_in_cvt` configuration with a calibrated `amax/448` scale:
    dq/dk/dv = 3.1/6.9/6.9 dB, against 28.1/31.3/31.3 dB for the identical build
    with the scale rounded to a power of two, and 3.08 dB predicted in advance
    from the truncation alone.

    `frexp` returns a mantissa in [0.5, 1), so an exact power of two -- and only
    an exact power of two, subnormals included -- gives exactly 0.5.
    """
    try:
        s = float(scale)
    except (TypeError, ValueError):
        # Better to refuse than to wave through a boxed value we cannot read;
        # a scale that reaches the cvt unchecked is a silent wrong answer.
        return False
    return math.isfinite(s) and s > 0.0 and math.frexp(s)[0] == 0.5


def _require_pow2_cvt_scale(scale):
    """Fail loudly rather than return silently-rescaled gradients.

    Checked at every call, not at compile time, because `scale_in_cvt` is a
    build-time flag while `qkv_scale` is a runtime argument -- a cached kernel
    built under one scale is handed a different one on the next step.
    """
    if not cvt_scale_is_exact(scale):
        raise ValueError(
            "scale_in_cvt=True folds qkv_scale into "
            "cvt_scalef32_pk_bf16_fp8, whose gfx950 scale operand is "
            f"exponent-only, but qkv_scale={scale!r} is not an exact power of "
            "two. It would be truncated to 2**floor(log2(qkv_scale)) and every "
            "gradient would be silently rescaled (measured 3.1 dB). Quantize "
            "q/k/v with a power-of-two scale, or build with scale_in_cvt=False "
            "to apply the scale in the score/epilogue math instead."
        )


PTR_TY = "!llvm.ptr"
LDS_PTR_TY = "!llvm.ptr<3>"
RSRC_TY = "!llvm.ptr<8>"
CDNA_BUFFER_FLAGS = (7 << 12) | (4 << 15)
GEP_DYNAMIC = -2147483648  # kDynamicIndex marker in llvm.getelementptr
CAT8 = [0, 1, 2, 3, 4, 5, 6, 7]  # concatenate two 4-wide fragments into K=32


def _lds_gep(mem, elem_off, elem_ty):
    """`!llvm.ptr<3>` at a dynamic element offset into an LDS memref.

    rocdl.ds.read.tr16.b64 takes a raw LDS pointer, not the (memref, index) pair
    the vector ops accept, so the view has to be reduced to a pointer by hand.
    """
    ptr_ty = ir.Type.parse(LDS_PTR_TY)
    # fly.extract_aligned_pointer_as_index only accepts global tensors, so the
    # LDS view goes through memref's version: index -> i32 -> ptr<3> (an LDS
    # pointer is 32 bits on AMDGPU).
    idx = memref.ExtractAlignedPointerAsIndexOp(mem).result
    base = llvm.IntToPtrOp(ptr_ty, arith.index_cast(T.i32, idx)).result
    return llvm.GEPOp(
        ptr_ty, base, [arith.unwrap(elem_off)], [GEP_DYNAMIC], elem_ty,
        llvm.GEPNoWrapFlags(0),
    ).result


def _atomic_fadd(val, rsrc, byte_off, aux=0):
    """Global atomic add through a buffer resource.

    flydsl's buffer_ops has loads and stores but no atomics, so this goes
    straight to the ROCDL op.  Offset is in BYTES, like the raw op wants.
    `aux` is the cache-policy field (bit0 sc0, bit1 nt, bit4 sc1).
    """
    rocdl_raw.RawPtrBufferAtomicFaddOp(
        arith.unwrap(val),
        rsrc,
        arith.unwrap(byte_off),
        arith.unwrap(_i32(0)),
        arith.unwrap(_i32(aux)),
    )


DPP_QUAD_SWAP_ADJ = 0xB1  # quad_perm[1,0,3,2]: lane l reads lane l^1


def _dpp_swap_adj(val):
    """Lane l's copy of lane (l^1)'s value, as a DPP modifier on a v_mov.

    Costs no extra instruction slot beyond the mov and never touches the LDS
    pipe, unlike ds_bpermute.
    """
    return rocdl_raw.DPPUpdateOp(
        T.f32, arith.unwrap(val), arith.unwrap(val),
        DPP_QUAD_SWAP_ADJ, 0xF, 0xF, True,
    ).result


def _current_stream(t):
    """fx.Stream for the caller's current PyTorch stream on t's device.

    The jitted launch()'s default fx.Stream(None) is HIP's null stream, which
    does not order against PyTorch's non-blocking side streams, so a launch
    issued while a side stream is current raced its own producers.  On the
    default stream this is the same handle as before.  Only called when
    current_stream_launch() ($DLRM_HSTU_FP8_FLYDSL_CURRENT_STREAM) is on.
    """
    import torch

    return fx.Stream(torch.cuda.current_stream(t.device))


def _i32(v):
    return arith.constant(int(v), type=T.i32)


def _i64(v):
    return arith.constant(int(v), type=T.i64)


def _to_i64(v):
    v = arith.unwrap(v)
    return v if v.type == T.i64 else arith.extsi(T.i64, v)


def _f32(v):
    return arith.constant(float(v), type=T.f32)


def _idx(v):
    return arith.index_cast(T.index, v)


def _shifted_rsrc(tensor, elem_offset, elem_bytes, num_bytes=None):
    """Buffer resource whose base is advanced by elem_offset.

    `elem_offset` must ALREADY be i64 when it can exceed 2^31.  It is only
    widened here, so a product formed in i32 has already wrapped by the time it
    arrives: at production's 2.77 M tokens a row stride of 2048 reaches 5.7e9
    elements, and even the packed 512 reaches 1.4e9, within 1.5x of the i32
    ceiling.  Callers build it with _to_i64 for that reason.

    `num_bytes`, when given, becomes the descriptor's num_records: a raw buffer
    load whose voffset reaches it returns zero instead of reading.  That turns
    "row past the end of this sequence" into an exact algebraic zero and removes
    the need for an in-loop bounds mask.
    """
    ptr_ty = ir.Type.parse(PTR_TY)
    base = fly.extract_aligned_pointer_as_index(ptr_ty, fly_values(tensor)[0])
    base_i64 = llvm.PtrToIntOp(T.i64, base).result
    off_i64 = arith.ArithValue(_to_i64(elem_offset))
    byte_i64 = off_i64 * arith.constant(int(elem_bytes), type=T.i64)
    new_i64 = llvm.AddOp(
        arith.unwrap(base_i64), arith.unwrap(byte_i64), llvm.IntegerOverflowFlags(0)
    ).result
    new_ptr = llvm.IntToPtrOp(ptr_ty, new_i64).result
    if num_bytes is None:
        nrec = arith.constant(0xFFFFFFFF, type=T.i64)
    else:
        nrec = arith.extsi(T.i64, arith.unwrap(num_bytes))
    return rocdl_raw.MakeBufferRsrcOp(
        ir.Type.parse(RSRC_TY),
        new_ptr,
        arith.unwrap(arith.constant(0, type=T.i16)),
        arith.unwrap(nrec),
        arith.unwrap(_i32(CDNA_BUFFER_FLAGS)),
    ).result


@functools.lru_cache(maxsize=64)
def compile_hstu_bwd_dkdv(
    heads: int,
    dim: int,
    alpha: float,
    BM: int = 64,
    BN: int = 128,
    abl: str = "",
    lds_pad: int = 0,
    # Bank-conflict padding for the tr16 Q/dO tiles.  The tr16 layout is
    # flat[d_block][m][d%16] with a d_block stride of BM*16 = 1024 halfwords =
    # 2048 B, an exact multiple of the 128 B (32-bank) LDS period, so all 8
    # d_blocks alias the same banks.  write_tiles spreads consecutive threads
    # across d_blocks (st_db), so its b128 stores collide ~16-way.  Adding
    # tr_pad halfwords per d_block (16 B, keeps every d_block 16 B-aligned for
    # the b128 store/load) rotates each d_block onto a distinct bank group and
    # takes the write pass from ~16 cycles to the ideal 4.  Reads (a_frag b128,
    # b_frag_tr) are already conflict-free and stay so.  +256 B LDS total.
    tr_pad: int = 8,
    SQ: int = 136,
    ST: int = 68,
    rm_align: int = 8,
    # iglp1 trips an assertion in LLVM's AMDGPUIGroupLP on some parameter
    # combinations, so it is not the default. iglp0 is the default.
    sched: str = "iglp0",
    tr16: bool = True,
    DQC: int = 1,
    fused: bool = False,
    dq16: str = "",
    dq_scale: float = 1.0,
    gorder: str = "xcd",
    rev: bool = False,
    dq_aux: int = 0,
    # R15 design B (default off): instead of the in-kernel dq MFMA and the
    # bf16p dq atomics, every workgroup stores its e4m3 ds tile (the exact
    # dq-MFMA A fragments, m-major [m 64][n 128]) to a
    # triangular workspace passed in the DQ slot; hstu_dq_split computes
    # dq = sum_n ds K afterwards in f32 with one plain store per row.  Same
    # products, f32 instead of bf16-atomic accumulation across n-blocks.
    dqsplit: bool = False,
    NXCD: int = 8,
    rs: int = 0,
    rs_in: int = 0,
    rs_out: int = 0,
    input_fp8: bool = False,
    scale_in_cvt: bool = False,
    q_bf16: bool = False,
    rs_q: int = 0,
    # ---- true fp8 MFMA on the qk gemm -----------------------------------
    # qk = q @ k^T is the ONLY one of the five gemms whose two operands are
    # both already e4m3 in global memory.  e4m3 -> bf16 is lossless (3 mantissa
    # bits into 7), so today the kernel widens both operands and runs a bf16
    # MFMA over values that are exactly fp8 -- the products are identical
    # either way and only the f32 accumulation order changes.  That makes this
    # one gemm the only numerically FREE place to put an fp8 MFMA: every other
    # gemm has a genuinely-bf16 operand (dO) or an f32-computed one (ds), and
    # rounding those to e4m3 costs 13-21 dB.
    #
    #   ""     incumbent: bf16 16x16x32 everywhere.
    #   "k32"  mfma_f32_16x16x32_fp8_fp8 for qk.  Same (lane, element) -> k
    #          mapping as the bf16 op, so it is a drop-in once both operands
    #          are raw bytes.  The MFMA rate is only 1.06x
    #          -- the win this mode is really testing is the LDS one: the Q
    #          A-operand read halves from b128 to b64 and the staged Q bytes
    #          halve.
    #   "k128" mfma_scale_f32_16x16x128_f8f6f4 for qk, 2.17x the bf16 rate and
    #          4x the K depth, so KSTEP collapses from 4 mfma to 1 and the four
    #          b128 A-operand reads become two.  Same LDS tile as "k32".
    #
    # Both modes keep the bf16 tr16 Q tile, because the dk gemm (dk += ds^T @ q)
    # needs q as a bf16 B-operand and ds cannot leave f32/bf16.  Reading the
    # fp8 tile transposed (ds_read_tr8_b64) and widening in registers was
    # costed and rejected: the Q tile is read 8x more than it is written, so
    # converting at read time quadruples the cvt count instead of removing it.
    fp8_mfma: str = "",
    # Bank-conflict padding, in BYTES, for the fp8 Q tile's 32-byte d-block
    # stride.  Same role as tr_pad for the bf16 tile: BM*32 = 2048 B is an
    # exact multiple of the 128 B LDS period, so without it all four d_blocks
    # alias the same banks and write_tiles (which spreads consecutive threads
    # across d_blocks via st_d // 32) collides 4-way.  32 B rotates each block
    # by 8 banks and keeps every block 16 B-aligned for the b128 reads "k128"
    # issues.
    q8_pad: int = 32,
    # ---- TRACK U1: m-unroll x2 so the dv/dk out-gemms reach K=128 ---------
    # dv = silu^T @ dO and dk = ds^T @ q contract over m.  One m tile is BM=64
    # deep, which is why they are stuck on the bf16 16x16x32 op: the only
    # profitable fp8 shape on gfx950 is mfma_scale_f32_16x16x128_f8f6f4, and
    # that needs contraction depth 128.  Running TWO m tiles per loop body
    # forms K=128.
    #   ""      off.
    #   "bf16"  unroll only; out-gemms stay bf16 K=32.
    #   "k128"  unroll plus mfma_scale_f32_16x16x128_f8f6f4 on dv and dk.
    mu2: str = "",
    mu2_fmt: str = "e4m3",
    mu2_ssc: float = 1.0,   # silu pre-scale (dv A operand)
    mu2_dsc: float = 1.0,   # ds   pre-scale (dk A operand)
    mu2_osc: float = 1.0,   # dO pre-scale (dv B operand)
    mu2_qsc: float = 1.0,   # q  pre-scale (dk B operand); q is already e4m3
                            # in LDS, so 1.0 is a LOSSLESS reinterpretation
    # ---- R1: full-fp8 backward (default OFF) -----------------------------
    #   ""     incumbent.
    #   "all"  ALL FIVE gemms on mfma_scale_f32_16x16x128_f8f6f4: qk (as k128),
    #          dp = dO V^T, dv = silu^T dO, dk = ds^T q, dq = ds k.  dO is an
    #          e4m3 INPUT tensor (producer-emitted), there are no bf16 Q/dO
    #          tiles at all: Q and dO are staged once as e4m3 in the 32 B
    #          d-block layout and the out-gemms' transposed B operands come
    #          straight out of them with ds_read_b64_tr_b8.  ds is exchanged
    #          through LDS as e4m3 ([n][m] bytes, tr8 back), K for dq is staged
    #          once as e4m3 in the bf16p interleave.  Needs k128 + mu2=k128 +
    #          fused + bf16p.  Stage-1 scales: unit except the existing mu2_*sc
    #          pre-scales and qkv_scale; dO's own scale is folded by the caller.
    f8b: str = "",
    ds8_pad: int = 8,       # bytes of padding per n row of the e4m3 ds tile
    # ---- R1 stage 2: runtime dO scale (default OFF, f8b only) -------------
    # When set, the launcher passes a 2-element fp32 DEVICE tensor
    # do_scale = [s_do, 1/s_do] (s_do a power of two) and the e4m3 dO tensor
    # holds dO * s_do.  dp and ds then carry the factor s_do, and the static
    # mu2_dsc is chosen for the NORMALIZED dO (amax in (224, 448]), so the
    # e4m3 ds tile sees a range that no longer depends on the loss scale.
    # All three epilogues (dv, dk, dq) multiply by 1/s_do -- exact, pow2.
    # No host sync: the scale never leaves the device.
    # RESEARCH-ONLY: the production wrapper
    # always passes do_dyn=False/do_blk=True, and no in-tree producer of the
    # [s_do, 1/s_do] scale remains (quantize_do_e4m3 was removed). Training
    # passes do_dyn=False.
    do_dyn: bool = False,
    # ---- C8 option B: per-(sequence, head, token-group) dO scale -----------
    # (default OFF, f8b only, exclusive with do_dyn).  The launcher passes a
    # uint8 tensor E8 [T, H] in the do_scale slot; E8[p, h] = 127 - e_g where
    # dO8 = e4m3(dO * 2**e_g) and the group of seq-relative token p is the
    # HARDWARE MX K-block: G(p) = (p // 64, (p % 16) // 8).  dp / ds stay in
    # the per-group scaled domain in registers (the static mu2_dsc is still
    # right: each group's amax is normalized like the global one was), and
    # the 2**-e_g is undone INSIDE the out-gemm MFMAs through their E8M0 scale
    # operands (mfma_scale_f32_16x16x128_f8f6f4, factor 2**(byte-127)):
    #   dv = silu^T dO8  (K = m): B scale per K-block = that block's group byte
    #   dk = ds^T  q     (K = m): A scale per K-block = group byte
    #   dq = ds    k     (K = n): A scale per ROW m   = row's group byte
    # The unscaled side passes a RUNTIME 127 (a runtime 0 means 2**-127; only
    # a literal 0 is folded to the unscaled op).  No inv_s_do epilogue.
    do_blk: bool = False,
    # ---- R1 stage 2b: saturating e4m3 cast of silu/ds (f8b only) ----------
    # cvt_scalef32_pk_fp8_f32 does NOT saturate: |v*scale| > 448 becomes NaN
    # (silu*ssc reached 1e4 and dv went NaN on
    # the first armed call).  The cvt's own VOP3 clamp bit is NOT available:
    # gfx950's assembler rejects 'clamp' on v_cvt_scalef32_pk_fp8_f32
    # ("invalid operand for instruction").
    # f8_sat clamps each element to +-448/scale first
    # (one v_min + one v_max, or a v_med3, per element) so an out-of-range
    # tile degrades to saturation instead of poisoning the weights.
    f8_sat: bool = False,
    # ---- silu / grad-coefficient epilogue form ---------------------------
    #   "legacy"  silu = (x*sig)*inv_max;
    #             gc   = sig*(1 + x*(1-sig))*inv_max;  ds = dp*vis*gc
    #   "reuse"   hoist the uniform alpha*score_input_scale product out of
    #             the per-element chain and share sig/N between SiLU and its
    #             derivative:  sig_inv = sig*inv_max;  silu = x*sig_inv;
    #             gc = fma(silu, 1-sig, sig_inv)  (= SiLU'(x)/N exactly in
    #             real arithmetic).  Same math, different f32 rounding order
    #             (~1 ulp); fewer VALU ops in the VALU-bound epilogue.
    #   "reuse_pk" (R15; default OFF, env-armed): "reuse" with every non-transcendental
    #             op of the chain (x, exp2 argument, e+1, sig/N, silu, 1-sig,
    #             the gc fma, dp*vis, *gc) issued on gfx950's double-rate
    #             packed f32 (v_pk_mul/add/fma_f32) over two consecutive
    #             accumulator elements; exp2/rcp stay scalar.  Same ops, same
    #             order, IEEE per lane -> bitwise equal to "reuse".  Same
    #             technique as hstu_fwd's epi_pk.
    # Packed f32 over two consecutive accumulator elements. exp2 and rcp stay
    # scalar. Same operations and the same order as "reuse".
    epilogue: str = "reuse",
    # Intra-pair lookahead (dqsplit + mu2 only; 0 = off, the
    # unchanged kernel).  Both m tiles of an mu2 pair are staged in LDS up
    # front (a second q8/o8 buffer), and the qk/dp MFMAs of tile B are issued
    # in the same scheduling region as tile A's silu/ds VALU chain, which they
    # do not depend on.  Same operations on every element; only the program
    # order, the LDS staging and the out-gemm B-operand read point move.
    #   1: next pair's two tiles are prefetched at the top of the body.
    #   2: as 1, but the second tile's prefetch is issued after tile A's ds
    #      exchange (shorter live range of the prefetch registers).
    #   3: as 2, and the out-gemm MFMAs of the last NSTEP/2 d tiles are issued
    #      after the tile-B ds-exchange barrier, next to its readback / ws
    #      store (their q/dO B operands are still read before the barrier,
    #      which is what frees the q8/o8 buffers for the next body).
    #   4: as 3 with all NSTEP d tiles' out-gemm MFMAs after the barrier.
    #   5: as 2, and the out-gemm q/dO B-operand LDS reads of the first
    #      NSTEP/2 d tiles move ahead of tile B's silu/ds chain (same region),
    #      so their latency hides under that VALU work.
    #   6: as 5 with all NSTEP d tiles' B-operand reads.
    pipe: int = 0,
):
    # `abl` is a comma-separated set of debug switches. Those builds are
    # numerically wrong. They are not used by training.
    abl_set = {s for s in abl.split(",") if s}
    ABL_ACT = "act" in abl_set  # drop silu / grad-coeff math
    ABL_MASK = "mask" in abl_set  # drop causal + bounds selects
    ABL_TR = "tr" in abl_set  # feed out-gemms from the row-major tile
    ABL_SCORE = "score" in abl_set  # drop the qk / dp gemms
    ABL_OUT = "out" in abl_set  # drop the dv / dk gemms
    ABL_PREFETCH = "nopf" in abl_set  # no global-load prefetch
    # Attribution only: collapse every LDS read of one operand class onto a
    # single address so CSE leaves exactly one ds_read.  The mfma stream, the
    # elementwise chain and the global traffic are untouched, so the time delta
    # is the cost of that class's LDS read bandwidth and nothing else.
    ABL_LDSA = "ldsa" in abl_set  # score-gemm A operands (q_rm / o_rm)
    ABL_LDSB = "ldsb" in abl_set  # out-gemm B operands (q_tr / o_tr)
    # Clean footprint-only variants of ldsa / ldsb.  `ldsa`/`ldsb` replace the
    # step index by a literal 0, so every read of the class becomes the SAME
    # SSA expression and CSE deletes all but one -- the delta then mixes read
    # INSTRUCTIONS with read BYTES and over-states the bandwidth term.
    # `ldsa2`/`ldsb2` instead multiply the step index by an OPAQUE ZERO (a
    # select on a runtime predicate LLVM cannot fold), so each read keeps a
    # distinct SSA address, every ds_read survives, the operand registers and
    # the mfma stream are untouched, and only the runtime FOOTPRINT collapses
    # onto one tile's worth of lines.  Lane-dependent terms are left alone, so
    # the bank pattern of a single read is identical to the real one.
    #   delta(ldsa2)  = LDS read bandwidth / L1-of-LDS locality of that class
    #   delta(ldsa) - delta(ldsa2) = the ds_read ISSUE cost of that class
    ABL_LDSA2 = "ldsa2" in abl_set
    ABL_LDSB2 = "ldsb2" in abl_set
    # Register-pressure probe for a FUSED backward.  Adds the fifth gemm
    # (dq = ds @ k), its BM x D f32 accumulator and its atomic flush to the
    # loop, with deliberately wrong operand orientations: the result is
    # garbage, but the mfma count, the accumulator size, the address
    # arithmetic and the atomic stream are the ones the real design needs.
    # DQC splits the accumulator into that many d-chunks, flushed in turn, to
    # trade peak register pressure against instruction-level parallelism.
    ABL_DQ = "dq" in abl_set
    # Attribution switches for the fused dq path.  Each collapses exactly one
    # term and leaves the rest of the instruction stream alone, so the time
    # delta is that term's cost.  All of them make dq numerically wrong.
    ABL_DQA = "dqa" in abl_set  # plain store instead of atomic (no RMW read)
    ABL_DQF = "dqf" in abl_set  # flush 1 of every 4 elements (quarter volume)
    ABL_DQR = "dqr" in abl_set  # collapse the ds transpose reads to one address
    ABL_DQW = "dqw" in abl_set  # collapse the ds writes to one address
    ABL_DQB = "dqb" in abl_set  # drop the ds-exchange barrier
    # Drop m0 from the atomic address, so every m iteration of every workgroup
    # read-modify-writes the SAME BM x D tile of dq_acc.  Instruction count,
    # atomic count and coalescing are all unchanged; only the address stream
    # collapses, so the traffic becomes cache-resident.  This is the exact
    # analogue of CK's level-2 dq ablation (`store_tile` without advancing the
    # window), which is what makes the two attributions comparable.
    ABL_DQZ = "dqz" in abl_set
    # MFMA-sensitivity probe for the fp8-MFMA track.  Issues every score-gemm
    # mfma TWICE into the same accumulator: +40% MFMA instructions, and nothing
    # else changes -- same LDS reads, same operands, same register pressure, no
    # dead code for the compiler to remove.  The time delta is the EXPOSED cost
    # of that MFMA issue, which is what a faster MFMA could actually buy back;
    # the analytic issue-time model only bounds it
    # from above, because MFMA that hides under LDS latency is free to remove
    # and free to add.  Numerically wrong on purpose.
    ABL_MFMA2X = "mfma2x" in abl_set
    # Same probe, for the other two gemm classes, because the hi+lo error
    # compensation of H4 does NOT double the MFMA count uniformly -- it doubles
    # exactly the gemms whose inexact operand gets split, and the three classes
    # sit in different parts of the schedule (score gemms wait on the k/v LDS
    # reads, the out gemms on the tr16 transpose, the dq gemm on the ds
    # round-trip through LDS plus its barrier).  An MFMA that hides under one
    # of those stalls is free to add; one that does not is not.  Each switch
    # re-issues that class's mfma into the same accumulator, so operands,
    # register pressure, LDS traffic and the atomic stream are untouched and
    # the time delta is that class's EXPOSED marginal MFMA cost -- which is
    # exactly the quantity a 2x-MFMA scheme has to pay.  Numerically wrong.
    ABL_MFMA2XO = "mfma2xo" in abl_set  # out gemms (dv, dk)
    ABL_MFMA2XQ = "mfma2xq" in abl_set  # dq gemm
    # ---- TRACK X1 CEILING PROBE ----------------------------------------
    # The whole fp8-LDS-staging + ds_read_tr8_b64 escape exists to delete ONE
    # thing: the v_cvt_scalef32_pk_fp8_bf16 that narrow the k128 out-gemms' B
    # operands (dO^T, q^T) in the inner loop, NSTEP times per element instead
    # of once at write_tiles.  Rather than spend the build on the LDS rewrite
    # and then discover the cvts were not the cost, price them first: keep the
    # identical k128 kernel, keep the LDS reads and their data dependence, and
    # replace only the cvt with a register slice of the bf16 fragment that
    # already arrived.  Zero VALU, same loads, same mfma count, same operand
    # widths -- the ONLY delta is the 512 cvts.  Numerically wrong on purpose:
    # it reinterprets bf16 halfwords as fp8 bytes.  The resulting time is the
    # FLOOR the real escape could reach on the B-operand axis (the real thing
    # additionally halves the tr reads and may relieve the spill, so it can go
    # a little lower -- but if this probe cannot beat the baseline, nothing
    # built on top of it can either).
    ABL_BCVT0 = "bcvt0" in abl_set
    # Prices the e4m3->bf16 widening (_widen_raw8). See the branch there.
    ABL_WCVT0 = "wcvt0" in abl_set
    assert dim == 128, "kernel is specialised for D=128"
    assert BM == 64 and BN % 64 == 0
    D = dim
    H = heads
    HD = H * D
    # Elements between consecutive tokens of q/k/v (and dq/dk/dv).  A packed
    # [N, H, D] tensor has H*D, but production never builds one: the UVQK gemm
    # emits a single [N, 2*H*hidden + 2*H*attn] buffer and hands the backward
    # COLUMN SLICES of it, so one token's q sits `rs` elements from the next.
    # Passing that stride is what lets the kernel read the slice in place; with
    # rs unset the caller is promising a packed tensor and gets the old math.
    #
    # The column offset needs no parameter -- a torch slice view carries it in
    # its data_ptr, which is what _shifted_rsrc starts from.  dO is NOT a slice
    # of that buffer (it is the attention output's gradient, its own contiguous
    # tensor), so it keeps the packed stride and is addressed separately.
    RS_IN = rs_in or rs or HD
    RS_OUT = rs_out or rs or HD
    RS_Q = rs_q or RS_IN
    # Eight-wide input loads are 16 B for BF16 and 8 B for FP8. Output gradient
    # loads/stores remain BF16. Both production row strides satisfy the stricter
    # eight-element alignment.
    assert RS_IN % 8 == 0, (
        f"input row stride {RS_IN} must be a multiple of 8 elements"
    )
    assert RS_OUT % 8 == 0, (
        f"output row stride {RS_OUT} must be a multiple of 8 elements"
    )
    assert RS_Q % 8 == 0, (
        f"Q row stride {RS_Q} must be a multiple of 8 elements"
    )
    assert RS_IN >= HD, (
        f"input row stride {RS_IN} cannot be narrower than one H*D row"
    )
    assert RS_OUT >= HD, (
        f"output row stride {RS_OUT} cannot be narrower than one H*D row"
    )
    assert RS_Q >= HD, (
        f"Q row stride {RS_Q} cannot be narrower than one H*D row"
    )
    assert not q_bf16 or input_fp8
    # "k32w" is an attribution probe, not a candidate: k128's TWO b128 A-operand
    # reads feeding k32's FOUR 16x16x32 mfma.  k128 changes two things at once
    # (4 mfma -> 1, and 4 b64 reads -> 2 b128), and this holds the read side at
    # k128 while leaving the mfma side at k32, so the difference k32w - k128 is
    # the mfma-rate term alone and k32 - k32w is the read term alone.
    assert fp8_mfma in ("", "k32", "k32w", "k128"), fp8_mfma
    if fp8_mfma:
        # Both qk operands have to BE e4m3 in memory.  q_bf16 stages a genuinely
        # bf16 Q, so there is nothing to reinterpret and rounding it here would
        # cost precision the incumbent does not pay.
        assert input_fp8 and not q_bf16, (
            "fp8_mfma needs e4m3 q and k (input_fp8=True, q_bf16=False)"
        )
        assert not (ABL_SCORE or ABL_MFMA2X or ABL_LDSA or ABL_DQ), (
            "score-gemm ablations are defined against the bf16 qk path"
        )
        assert tr16, "fp8_mfma reuses the tr16 staging schedule"
    FP8_QK = fp8_mfma
    assert mu2 in ("", "bf16", "k128"), mu2
    MU2 = mu2
    assert mu2_fmt in ("e4m3", "e5m2"), mu2_fmt
    if MU2:
        assert BM == 64 and tr16, "mu2 unrolls the BM=64 tr16 body"
        assert not (ABL_OUT or ABL_TR or ABL_MFMA2XO), (
            "out-gemm ablations are defined against the K=32 path"
        )
        for s in (mu2_ssc, mu2_dsc, mu2_osc, mu2_qsc):
            assert cvt_scale_is_exact(s) or s == 1.0, (
                f"mu2 pre-scale {s} must be an exact power of two: it is "
                "folded into cvt_scalef32's E8M0 (exponent-only) scale field"
            )
    assert f8b in ("", "all"), f8b
    F8B = f8b == "all"
    assert not f8_sat or F8B, "f8_sat needs f8b=all"
    assert epilogue in ("legacy", "reuse", "reuse_pk"), (
        f"unknown epilogue {epilogue!r}")
    EPI_REUSE = epilogue in ("reuse", "reuse_pk")
    EPI_PK = epilogue == "reuse_pk"
    if F8B:
        assert (fp8_mfma == "k128" and mu2 == "k128" and mu2_fmt == "e4m3"
                and fused and dq16 == "bf16p" and input_fp8 and not q_bf16
                and not rev and not scale_in_cvt), (
            "f8b=all needs k128 + mu2=k128(e4m3) + fused bf16p + e4m3 inputs")
        assert not (abl_set - {"bcvt0", "wcvt0"}), "f8b: no ablations"
        # dO / q reach the out-gemms as stored e4m3 bytes, with no re-cast to
        # carry a pre-scale.
        assert mu2_osc == 1.0 and mu2_qsc == 1.0
        assert ds8_pad % 8 == 0
    DQSPLIT = bool(dqsplit)
    if DQSPLIT:
        assert F8B and do_blk and BN == 128 and BM == 64, (
            "dqsplit needs the C8 path (f8b=all, do_blk) at BN=128, BM=64")
    PIPE = int(pipe)
    assert PIPE in (0, 1, 2, 3, 4, 5, 6), pipe
    if PIPE:
        assert DQSPLIT and MU2 == "k128" and fused and not rev and not abl_set, (
            "pipe needs dqsplit + mu2=k128, ascending, no ablations")
    assert not do_dyn or F8B, "do_dyn needs f8b=all"
    assert not do_blk or (F8B and not do_dyn), (
        "do_blk needs f8b=all and excludes do_dyn")
    # Per-group dO scales put ~1e-6 of ds above 448 on real tensors, and the
    # unclamped cvt_scalef32_pk_fp8_f32 returns NaN above 464 (probe P1).
    # The production C8 path (hstu_flydsl_mha) always passes both.
    assert not do_blk or f8_sat, (
        "do_blk needs f8_sat: without the clamp the rare ds > 448 becomes a "
        "NaN gradient")
    # The K-block <-> token-group identity below needs m tiles that start on a
    # multiple of 64 seq-relative tokens (m0 = n0 + 64*it, n0 = BN*n_blk) and
    # pairs (pend, cur) = (m0 - 64, m0): BM=64, BN a multiple of 128.
    assert not do_blk or (BM == 64 and BN % 128 == 0)
    DO_BYTES = 1 if F8B else 2
    INPUT_BYTES = 1 if input_fp8 else 2
    Q_BYTES = 2 if q_bf16 else INPUT_BYTES
    WAVES = 4
    THREADS = WAVES * 64
    NSTEP = D // 16  # 8 d-tiles of the dk/dv accumulators
    MSTEP = BM // 16  # 4 m-tiles
    KSTEP = D // 32  # 4 k-tiles of the score gemms (double-rate 16x16x32)
    # n-tiles owned by one wave.  Every LDS read in the body is indexed by (m, d)
    # or (m, k) and never by n, so widening BN reuses the same staged Q/dO tile
    # against NT times as much math: LDS bytes per flop are 5/BN.  The cost is
    # registers -- accumulators, K/V fragments and score tiles all scale with NT.
    NT = BN // (WAVES * 16)
    NPEEL = BN // BM  # leading m iterations whose tile overlaps the n range
    # ---- fused dq -------------------------------------------------------
    # dq[m][d] = sum_n ds[m][n] k[n][d] contracts over n, so a wave that owns
    # only 16*NT of the block's n rows can produce nothing but a partial, and
    # four partials per workgroup would mean four atomic streams into dq_acc --
    # 4x the global read-modify-write traffic, which is the one thing this
    # design cannot afford.  So dq partitions the waves by D instead: wave w
    # takes d in [DQD*16*w, +DQD*16) and contracts over the FULL BN.  That
    # needs every wave to see every wave's ds, i.e. an exchange through LDS.
    # The exchange is not overhead: ds arrives from the mfma accumulator with n
    # in the lane index and m in the element index, and dq needs the opposite,
    # so the tile has to be transposed anyway.  Writing it in the tr16 layout
    # and reading it back with ds_read_b64_tr_b16 does the exchange and the
    # transpose in one round trip.
    DQD = D // 16 // WAVES  # 16-wide d tiles per wave
    DQK = BN // 32  # 32-wide n slices, one double-rate mfma each
    assert not fused or (D % (16 * WAVES) == 0 and BN % 32 == 0)
    XKIND = {"xcd": 1, "xcdn": 2}.get(gorder, 0)
    # dq_acc element type.  The fused kernel is bandwidth-bound on dq_acc's
    # read-modify-write traffic -- 61 GB each way at BN=128 -- so halving the
    # element halves the dominant cost.  gfx950 has no packed f32 atomic add,
    # but it has buffer_atomic_pk_add_bf16 and _f16, which add two 16-bit
    # values in one dword.  Reaching them needs the accumulator TRANSPOSED so
    # that a lane's four elements are four consecutive d rather than four m;
    # that is one mfma operand swap, see the flush below.
    # "bf16" and "f16" transpose the accumulator so a lane's own elements pair.
    # "bf16c" keeps C[m][d] and pairs with the neighbouring lane.
    # "bf16p" interleaves d in the staged K tile so a lane's two accumulators
    # hold d and d+1, and all 16 lanes pack a pair. Training uses "bf16p".
    assert dq16 in ("", "bf16", "f16", "bf16c", "bf16p")
    # bf16p needs exactly two d tiles per wave to pair up.
    assert dq16 != "bf16p" or not fused or DQD == 2
    DQ_BYTES = 4 if not dq16 else 2
    # Resolved lazily: MLIR types need a live Context, which only exists once
    # the kernel body is being emitted.
    def dq_ty():
        return {"bf16": T.bf16, "bf16c": T.bf16, "bf16p": T.bf16,
                "f16": T.f16}.get(dq16, T.f32)

    def emit_sched():
        """Tell the machine scheduler how to pipeline the loop body.

        At BN=128 the kernel runs one wave per SIMD, so nothing else covers LDS
        latency: left alone, LLVM emits ds_read / s_waitcnt lgkmcnt(0) / v_mfma
        in lockstep and every mfma pays a full LDS round trip.  The hand-written
        assembly issues a whole tile's reads and then waits with a non-zero
        lgkmcnt.  `sched` picks how to ask for the same shape.
        """
        if not sched:
            return
        if sched.startswith("iglp"):
            rocdl.iglp_opt(int(sched[4:]))
            return
        if sched.startswith("sgb"):
            # sgb<reads>x<mfmas>: repeat a (DS_READ n, MFMA m) group until the
            # body's 64 reads and 128 mfmas are consumed.
            r, m = (int(x) for x in sched[3:].split("x"))
            for _ in range(min(64 // r, 128 // m)):
                rocdl.sched_dsrd(r)
                rocdl.sched_mfma(m)
            return
        raise ValueError(f"unknown sched {sched!r}")

    arch = get_rocm_arch()
    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=(f"hstu_dkdv_smem_{abl.replace(',', '_')}_{lds_pad}"
                         f"_{SQ}_{ST}_{rm_align}_{BN}_{int(tr16)}"
                         f"_{int(fused)}_{dq16}_{gorder}{NXCD}"
                         f"_{int(rev)}_{dq_aux}_{RS_IN}_{RS_OUT}"
                         f"_{int(input_fp8)}_{int(scale_in_cvt)}"
                         f"_{int(q_bf16)}_{RS_Q}"
                         f"_{fp8_mfma or 'bf16'}_{q8_pad}"
                         f"_mu{mu2 or 'off'}{mu2_fmt}_{mu2_ssc}_{mu2_dsc}_{mu2_osc}_{mu2_qsc}"
                         f"_f8b{f8b or 'off'}{ds8_pad}{'_dd' if do_dyn else ''}{'_db' if do_blk else ''}{'_sat' if f8_sat else ''}"
                         f"_epi{epilogue}"
                         + ("_dqs" if dqsplit else "")
                         + (f"_pipe{PIPE}" if PIPE else "")),
    )
    # With tr16 there is one copy of each tile, in the layout gfx950's
    # ds_read_b64_tr_b16 wants:  halfword(m, d) = ((d/16)*BM + m)*16 + d%16.
    # A run of 16 halfwords is one m at one 16-wide d block, so 8 of them are
    # still contiguous for the natural read, and 4 consecutive m at one d are
    # exactly the transpose the instruction gathers.  Without tr16 the
    # transposed operand needs a second, hand-packed copy.
    RM = D if tr16 else SQ
    # Halfwords per Q/dO tile.  With tr16 the tile is (D/16) d_blocks each of
    # BM*16 halfwords, plus tr_pad halfwords of bank-conflict padding per block.
    DBS = BM * 16 + tr_pad  # d_block stride in halfwords (tr16)
    RM_HW = (D // 16) * DBS if tr16 else BM * RM
    if F8B:
        RM_HW = 8  # no bf16 tiles; keep a token allocation so the views exist
    off_q_rm = allocator._align(allocator.ptr, 16)
    allocator.ptr = off_q_rm + RM_HW * 2
    off_o_rm = allocator._align(allocator.ptr, 16)
    allocator.ptr = off_o_rm + RM_HW * 2
    # ---- fp8 Q tile ------------------------------------------------------
    # byte(m, d) = (d/32)*Q8BS + m*32 + (d%32): the byte analogue of tr16 but
    # with 32-byte d_blocks instead of 16.  32 is what makes BOTH qk modes
    # conflict-free and contiguous:
    #   k32  wants 8 bytes at d = 8*l4 + 32*ks, i.e. block ks, offset 8*l4
    #        inside the row -- 64 lanes then start at banks
    #        (8*l16 + 2*l4) mod 32, which covers all 32 banks exactly 4 deep,
    #        the floor for a 512 B wavefront read.
    #   k128 wants all 32 bytes of block l4 at one m -- one contiguous 32 B
    #        run, read as two b128.
    # A 16-byte d_block (the direct byte copy of tr16) would leave half the
    # banks idle on the k32 read, which is why the block is widened here.
    Q8BS = BM * 32 + q8_pad  # d_block stride in BYTES
    assert Q8BS % 16 == 0
    off_q8 = 0
    if FP8_QK:
        off_q8 = allocator._align(allocator.ptr, 16)
        allocator.ptr = off_q8 + (D // 32) * Q8BS
    off_o8 = 0
    if F8B:
        off_o8 = allocator._align(allocator.ptr, 16)
        allocator.ptr = off_o8 + (D // 32) * Q8BS
    # pipe: second fp8 Q / dO tile pair (tile B of an mu2 pair).
    off_q8b = off_o8b = 0
    if PIPE:
        off_q8b = allocator._align(allocator.ptr, 16)
        allocator.ptr = off_q8b + (D // 32) * Q8BS
        off_o8b = allocator._align(allocator.ptr, 16)
        allocator.ptr = off_o8b + (D // 32) * Q8BS
    # e4m3 ds exchange: byte(n, m) = n*DS8R + m.
    DS8R = BM + ds8_pad
    if not tr16:
        off_q_tr = allocator._align(allocator.ptr, 16)
        allocator.ptr = off_q_tr + D * ST * 2
        off_o_tr = allocator._align(allocator.ptr, 16)
        allocator.ptr = off_o_tr + D * ST * 2
    if fused:
        # One region serves two purposes in sequence.  Before the m loop it
        # holds K in the tr16 layout, just long enough to pull dq's B operands
        # into registers; from then on its first BM*BN halfwords are the ds
        # exchange buffer.  K needs BN*D and ds needs BM*BN, so the larger one
        # sizes it and the two never overlap in time.
        off_kt = allocator._align(allocator.ptr, 16)
        allocator.ptr = off_kt + (max(BN * D, BN * DS8R) if F8B
                                  else max(BN * D, BM * BN) * 2)
    # Dead LDS, used only to probe how workgroups-per-CU affects this kernel.
    if lds_pad:
        allocator.ptr = allocator._align(allocator.ptr, 16) + lds_pad

    def emit_body(
        Q,
        K,
        V,
        DO,
        DK,
        DV,
        DQ,
        inv_max,
        qkv_scale,
        tid,
        n0,
        head,
        seq_start,
        seq_len,
        DO_SC=None,
        batch_idx=None,
        nbatch=None,
    ):
        mfma = rocdl.mfma_f32_16x16x16bf16_1k
        v4i16 = T.vec(4, T.i16)
        v4bf = T.vec(4, T.bf16)
        v8bf = T.vec(8, T.bf16)
        v2i32 = T.vec(2, T.i32)
        v4i32 = T.vec(4, T.i32)
        v8i32 = T.vec(8, T.i32)
        zero_acc = arith.constant_vector(0.0, T.f32x4)

        def load_raw8(resource, offset):
            """The 8 packed e4m3 bytes at `offset` (in elements), un-widened.

            Exactly the buffer_load load_qkv issues before its cvt chain; the
            fp8 paths keep the dwords instead of widening them.
            """
            return buffer_ops.buffer_load(
                resource,
                arith.ArithValue(offset) // _i32(4),
                vec_width=2,
                dtype=T.i32,
            )

        def _widen_raw8(packed):
            """8 packed e4m3 bytes (vec<2xi32>) -> vec<8xbf16>.

            e4m3 -> bf16 is exact (3 mantissa bits into 7), so this is a pure
            widening whenever conversion_scale is 1.0.
            """
            if ABL_WCVT0:
                # Debug build: reinterpret the loaded dwords as bf16 and skip the
                # convert. Numerically wrong. Not used by training.
                as_bf = vector.bitcast(T.vec(4, T.bf16), packed)
                return vector.shuffle(as_bf, as_bf, CAT8)
            converted = []
            conversion_scale = (
                arith.ArithValue(qkv_scale)
                if scale_in_cvt
                else _f32(1.0)
            )
            for dword in range_constexpr(2):
                source = vector.extract(
                    packed,
                    static_position=[dword],
                    dynamic_position=[],
                )
                for high_word in (False, True):
                    converted.append(
                        rocdl_raw.cvt_scalef32_pk_bf16_fp8(
                            T.vec(2, T.bf16),
                            source,
                            arith.unwrap(conversion_scale),
                            high_word,
                        )
                    )
            low = vector.shuffle(converted[0], converted[1], [0, 1, 2, 3])
            high = vector.shuffle(converted[2], converted[3], [0, 1, 2, 3])
            return vector.shuffle(low, high, CAT8)

        def load_qkv(resource, offset, source_fp8=True):
            if input_fp8 and source_fp8:
                # Every call is eight-byte aligned. Loading two dwords avoids
                # vector<8xi8>, which SelectionDAG cannot legally split for the
                # raw-buffer intrinsic on gfx950.
                return _widen_raw8(load_raw8(resource, offset))
            return buffer_ops.buffer_load(
                resource, offset, vec_width=8, dtype=T.bf16
            )

        score_input_scale = (
            arith.ArithValue(qkv_scale)
            * (
                _f32(1.0)
                if q_bf16
                else arith.ArithValue(qkv_scale)
            )
            if input_fp8 and not scale_in_cvt
            else _f32(1.0)
        )
        value_input_scale = (
            arith.ArithValue(qkv_scale)
            if input_fp8 and not scale_in_cvt
            else _f32(1.0)
        )
        dk_input_scale = (
            _f32(1.0) if q_bf16 else value_input_scale
        )
        # epilogue="reuse": the uniform alpha*score_input_scale product is
        # paid once per kernel invocation instead of once per score element.
        score_alpha_scale = _f32(alpha) * arith.ArithValue(score_input_scale)
        # epilogue="reuse_pk": <2 x f32> splats of the uniform operands, built
        # once per kernel (LLVM keeps them in SGPR/VGPR pairs that
        # v_pk_*_f32 read directly).
        V2F = T.vec(2, T.f32)
        if EPI_PK:
            def _splat2(v):
                v = arith.unwrap(v)
                return vector.from_elements(V2F, [v, v])
            pk_sas2 = _splat2(score_alpha_scale)
            pk_nl2e2 = arith.constant_vector(-LOG2E, V2F)
            pk_one2 = arith.constant_vector(1.0, V2F)
            pk_inv2 = _splat2(inv_max)
            pk_vis2 = _splat2(value_input_scale)
        # R1 stage 2 runtime dO scale.  do_scale = [s_do, 1/s_do].
        inv_s_do = None
        if do_dyn:
            r_dsc = buffer_ops.create_buffer_resource(
                DO_SC, max_size=False,
                num_records_bytes=arith.unwrap(_i32(8)))
            inv_s_do = arith.ArithValue(buffer_ops.buffer_load(
                r_dsc, _i32(1), vec_width=1, dtype=T.f32))

        # Double-rate bf16 MFMA (K=32 per issue) for the score GEMMs.  Reached
        # through the raw ROCDL op: it consumes native v8bf16 A/B fragments and
        # keeps the same 16x16 f32 C layout as the half-rate 16x16x16 variant.
        def mfma2(a, b, acc):
            return rocdl_raw.mfma_f32_16x16x32_bf16(
                T.f32x4,
                arith.unwrap(a),
                arith.unwrap(b),
                arith.unwrap(acc),
                0,
                0,
                0,
            ).result

        # ---- fp8 MFMA for the qk gemm ------------------------------------
        # Both variants take the SAME (lane, element) -> k mapping the bf16
        # 16x16x32 op uses, so the operands are the raw e4m3 bytes in exactly
        # the order the widened bf16 fragments carried them.  No scale is
        # folded into the instruction: qkv_scale**2 is still applied once per
        # score in the epilogue (score_input_scale), which keeps the dequant
        # out of the E8M0-only scale field -- see the cvt_scalef32 finding.
        def mfma_qk_k32(a, b, acc):
            # a/b are i64 (8 packed e4m3).  vector.bitcast cannot produce a
            # scalar, so the dword pair goes through llvm.bitcast.
            return rocdl_raw.mfma_f32_16x16x32_fp8_fp8(
                T.f32x4,
                llvm.BitcastOp(T.i64, arith.unwrap(a)).result,
                llvm.BitcastOp(T.i64, arith.unwrap(b)).result,
                arith.unwrap(acc),
                0,
                0,
                0,
            ).result

        z_i32 = arith.unwrap(arith.constant(0, type=T.i32))

        def mfma_qk_k128(a, b, acc):
            # a/b are vec<8xi32> (32 packed e4m3).  cbsz=0 / blgp=0 select e4m3
            # for both operands; scale=0 is E8M0 exponent bias 127, multiplier
            # 1.0, so the instruction's scale is a no-op.
            return rocdl_raw.mfma_scale_f32_16x16x128_f8f6f4(
                T.f32x4,
                arith.unwrap(a),
                arith.unwrap(b),
                arith.unwrap(acc),
                0,
                0,
                0,
                z_i32,
                0,
                z_i32,
            ).result

        # ---- TRACK U1 helpers: fp8 packing + K=128 out-gemm ---------------
        # v_cvt_scalef32_pk8_{fp8,bf8}_{f32,bf16}: EIGHT elements -> eight fp8
        # bytes (two dwords) in ONE VALU op.  That is what makes the m-unroll
        # affordable at all -- the naive per-pair cvt would cost 4x as many
        # instructions as the mfma saving.  The scale operand is E8M0
        # (exponent-only), hence the power-of-two assert on mu2_*sc.
        MU2_BF8 = mu2_fmt == "e5m2"

        def _res(x):
            return x.result if hasattr(x, "result") else x

        # The pk8 (8-element) forms exist in the ROCDL dialect but are NOT
        # selectable in this toolchain (LLVM ERROR: Cannot select
        # llvm.amdgcn.cvt.scalef32.pk8.fp8.bf16), so the pairwise forms are
        # used: two elements per op, written into the lo or hi half of a
        # dword.  Four ops per four-byte piece pair instead of one.
        # ---- MEASURED SEMANTICS OF cvt_scalef32's SCALE OPERAND -------------
        # The DOWN-convert DIVIDES by the scale: dst = fp8(src / scale).  That
        # is the OCP-MX convention (the scale is the shared block exponent, and
        # the stored element is value/scale), and it is the opposite of the
        # up-convert (cvt_scalef32_pk_bf16_fp8, dst = fp8_to_bf16(src) * scale)
        # this file already used, which is where the sign error came from.
        #
        # Proof, not guesswork.  Two observations from the k128 arm pin it:
        #   * scale = 2**27 on silu (amax 2.47e-6) returned dk and dv
        #     IDENTICALLY ZERO -- 2.47e-6 / 2**27 = 1.8e-14, far under e4m3's
        #     smallest subnormal, so every element flushed.  Had the operand
        #     been multiplied, that arm would have been CORRECT, not zero.
        #   * feeding the reciprocals instead gave dk = -276.95 dB.  Under the
        #     divide reading, the cvt is then right but the epilogue undoes the
        #     scale the wrong way, leaving a constant factor (2**23 * 2**0)**2
        #     = 2**46; -20*log10(2**46) = -276.9476 dB.  Matches to 0.01 dB.
        # `mu2_*sc` therefore keeps its natural caller-facing meaning ("multiply
        # the operand by this before casting") and the reciprocal is formed
        # here.  Exact for a power of two, which is asserted at entry.
        def _cvt_sc(scale):
            return arith.unwrap(_f32(1.0 / scale))

        def _sat(x, lim, mode):
            # mode "hi": min(x, lim) -- silu >= -0.279/N can never reach -lim
            # for any K the launcher allows.  mode "both": min(max(x,-lim),lim)
            # for ds.  NaN-ignoring num-variants lower to v_min/v_max_f32
            # (or one fused v_med3_f32).
            from flydsl._mlir.dialects import arith as _ar
            mn = getattr(_ar, "minnumf", None) or (
                lambda a_, b_: _ar.MinNumFOp(a_, b_).result)
            mx = getattr(_ar, "maxnumf", None) or (
                lambda a_, b_: _ar.MaxNumFOp(a_, b_).result)
            hi = arith.unwrap(_f32(lim))
            if mode == "hi":
                return mn(x, hi)
            return mn(mx(x, arith.unwrap(_f32(-lim))), hi)

        def pk4_from_f32(v4f32, scale, sat=False):
            """4 f32 -> one dword of 4 fp8, in element order."""
            f = (rocdl_raw.cvt_scalef32_pk_bf8_f32 if MU2_BF8
                 else rocdl_raw.cvt_scalef32_pk_fp8_f32)
            acc = arith.unwrap(arith.constant_vector(0, T.vec(2, T.i16)))
            sc = _cvt_sc(scale)
            lim = (57344.0 if MU2_BF8 else 448.0) / scale
            for half in range_constexpr(2):
                e0 = vector.extract(v4f32, static_position=[2 * half],
                                    dynamic_position=[])
                e1 = vector.extract(v4f32, static_position=[2 * half + 1],
                                    dynamic_position=[])
                if sat:
                    e0 = _sat(arith.unwrap(e0), lim, sat)
                    e1 = _sat(arith.unwrap(e1), lim, sat)
                acc = _res(f(T.vec(2, T.i16), acc, e0, e1, sc, half))
            return acc

        def pk4_from_bf16(v4bf16, scale):
            """4 bf16 -> one dword of 4 fp8, in element order."""
            f = (rocdl_raw.cvt_scalef32_pk_bf8_bf16 if MU2_BF8
                 else rocdl_raw.cvt_scalef32_pk_fp8_bf16)
            acc = arith.unwrap(arith.constant_vector(0, T.vec(2, T.i16)))
            sc = _cvt_sc(scale)
            for half in range_constexpr(2):
                acc = _res(f(
                    T.vec(2, T.i16), acc,
                    vector.shuffle(v4bf16, v4bf16, [2 * half, 2 * half + 1]),
                    sc, half))
            return acc

        # Pieces stay as halfword vectors all the way to the mfma: v2i16 (one
        # dword, 4 fp8 = one 16-row m sub-tile) -> v4i16 (a K=32 pair) ->
        # v8i16 (one BM=64 tile's half of the operand) -> v16i16 for the pair,
        # bitcast to the v8i32 the mfma_scale op takes.
        def pk8_from_f32(frag_a, frag_b, scale):
            return vector.shuffle(pk4_from_f32(frag_a, scale),
                                  pk4_from_f32(frag_b, scale), [0, 1, 2, 3])

        def pk8_from_bf16(frag_a, frag_b, scale):
            return vector.shuffle(pk4_from_bf16(frag_a, scale),
                                  pk4_from_bf16(frag_b, scale), [0, 1, 2, 3])

        # cbsz/blgp select the operand format class: 0 = e4m3, 1 = e5m2.
        MU2_FSEL = 1 if MU2_BF8 else 0

        def mfma_out_k128(a, b, acc):
            return rocdl_raw.mfma_scale_f32_16x16x128_f8f6f4(
                T.f32x4,
                arith.unwrap(a),
                arith.unwrap(b),
                arith.unwrap(acc),
                MU2_FSEL,
                MU2_FSEL,
                0,
                z_i32,
                0,
                z_i32,
            ).result

        # do_blk: RUNTIME E8M0 scale operands (byte 0 of an i32, opsel 0).
        # 127 = x1 on the unscaled side; it must be a runtime value -- see the
        # do_blk flag comment (P0: runtime 0 = 2**-127).
        sc127 = arith.unwrap(_i32(127)) if do_blk else None

        def mfma_k128_sc(a, b, acc, sa, sb):
            return rocdl_raw.mfma_scale_f32_16x16x128_f8f6f4(
                T.f32x4,
                arith.unwrap(a),
                arith.unwrap(b),
                arith.unwrap(acc),
                MU2_FSEL,
                MU2_FSEL,
                0,
                arith.unwrap(sa),
                0,
                arith.unwrap(sb),
            ).result

        # ---- R1 f8b: ds_read_b64_tr_b8 --------------------------------------
        # MEASURED on gfx950: within a 16-lane
        # group, lane j receives byte (j % 8) of the 8-byte runs addressed by
        # source lanes 2*e + j//8, e = 0..7, as its byte e.  So source lane j
        # names row e = j//2 for the receiver half k = j%2, and the receiver
        # half k reads 8 consecutive COLUMNS starting at that row address.
        i8_t = ir.IntegerType.get_signless(8)
        _tr8_op = getattr(rocdl_raw, "ds_read_tr8_b64_", None)

        def tr8(lds, byte_off):
            ptr = _lds_gep(lds, byte_off, i8_t)
            if _tr8_op is not None:
                return _res(_tr8_op(v2i32, ptr))
            addr = llvm.PtrToIntOp(T.i32, ptr).result
            return llvm.inline_asm(
                v2i32, [addr], "ds_read_b64_tr_b8 $0, $1", "=v,v",
                has_side_effects=False)

        # i64: seq_start * row-stride is a whole-tensor element offset and overflows i32
        # at production sizes (see _shifted_rsrc).  Only this base shift needs
        # the wider type -- every offset used inside the loop is relative to it
        # and bounded by one sequence.
        head_off = arith.ArithValue(_to_i64(head)) * _i64(D)
        tok_off_in = (
            arith.ArithValue(_to_i64(seq_start)) * _i64(RS_IN) + head_off
        )
        tok_off_q = (
            arith.ArithValue(_to_i64(seq_start)) * _i64(RS_Q) + head_off
        )
        tok_off_out = (
            arith.ArithValue(_to_i64(seq_start)) * _i64(RS_OUT) + head_off
        )
        # dO always carries the packed H*D stride.
        tok_off_do = (
            arith.ArithValue(_to_i64(seq_start)) * _i64(HD) + head_off
        )
        # Q and dO are the only tensors read past the end of the sequence (the m
        # loop runs in whole BM tiles and prefetches one tile beyond the last).
        # Stopping their descriptors at the sequence end makes those reads return
        # zero, and a zero q/dO row contributes exactly zero to both outputs:
        # silu(0) = 0, and dp = 0 forces ds = 0.  So the bounds half of the
        # in-loop mask is unnecessary -- see the m loop below.
        q_bytes = arith.ArithValue(seq_len) * _i32(RS_Q * Q_BYTES)
        do_bytes = arith.ArithValue(seq_len) * _i32(HD * DO_BYTES)
        rQ = _shifted_rsrc(
            Q, tok_off_q, Q_BYTES, num_bytes=q_bytes
        )
        rK = _shifted_rsrc(K, tok_off_in, INPUT_BYTES)
        rV = _shifted_rsrc(V, tok_off_in, INPUT_BYTES)
        rDO = _shifted_rsrc(DO, tok_off_do, DO_BYTES, num_bytes=do_bytes)
        if do_blk:
            # E8 [T, H] uint8, based at (seq_start, head) and stopped at the
            # sequence end: byte offset p*H is token p's group byte, and a
            # token past seq_len reads 0 (2**-127) -- only ever applied to
            # all-zero data (its dO row, hence ds row, is zero).
            rE8 = _shifted_rsrc(
                DO_SC, arith.ArithValue(_to_i64(seq_start)) * _i64(H)
                + _to_i64(head), 1,
                num_bytes=arith.ArithValue(seq_len) * _i32(H))
        rDK = _shifted_rsrc(DK, tok_off_out, 2)
        rDV = _shifted_rsrc(DV, tok_off_out, 2)
        if ABL_DQ:
            # The probe has no dq tensor of its own; it scribbles f32 atomics
            # over DK, bounded by DK's real extent so nothing lands outside the
            # allocation.  Same instruction stream, garbage result, by design.
            rDQ = _shifted_rsrc(
                DK,
                tok_off_out,
                2,
                num_bytes=arith.ArithValue(seq_len) * _i32(RS_OUT * 2),
            )
        if fused:
            # num_records stops the descriptor at this sequence's last token, so
            # the atomics for m rows past seq_len are dropped in hardware and
            # the flush needs no bounds branch.  Those rows carry exactly zero
            # anyway (a zero dO row forces dp = 0 hence ds = 0), so this only
            # avoids touching a neighbour's memory, it does not change values.
            rDQ = _shifted_rsrc(
                DQ, tok_off_out, DQ_BYTES,
                num_bytes=arith.ArithValue(seq_len)
                * _i32(RS_OUT * DQ_BYTES))
        if DQSPLIT:
            # Workspace (passed as DQ): header int32 prefix[B+1] of per-batch
            # tile counts, padded to WS_HDR bytes, then 8 KiB e4m3 ds^T tiles.
            # (b, h) owns H-major nb*(nb+1) tiles, nb = ceil(L/128); n-block j
            # owns tiles t = 2j .. 2nb-1 at slot 2*j*nb - j*(j-1) + t - 2j.
            # Must match hstu_dq_split.py.
            r_hdr = _shifted_rsrc(DQ, _to_i64(_i32(0)), 1)
            ws_pre = arith.ArithValue(buffer_ops.buffer_load(
                r_hdr, batch_idx, vec_width=1, dtype=T.i32))
            ws_nb = (arith.ArithValue(seq_len) + _i32(127)) // _i32(128)
            ws_hdr = ((arith.ArithValue(nbatch) + _i32(1)) * _i32(4)
                      + _i32(255)) // _i32(256) * _i32(256)
            ws_tiles = ws_pre + arith.ArithValue(head) * ws_nb * (ws_nb + _i32(1))
            ws_j = n0 // _i32(128)
            # Bounded to this (b, h)'s nb*(nb+1) tiles: a stray offset is
            # dropped by the buffer unit instead of landing in a neighbour.
            rWS = _shifted_rsrc(
                DQ, arith.ArithValue(_to_i64(ws_tiles)) * _i64(8192)
                + _to_i64(ws_hdr), 1,
                num_bytes=ws_nb * (ws_nb + _i32(1)) * _i32(8192))
            ws_jbase = (_i32(2) * ws_j * ws_nb - ws_j * (ws_j - _i32(1))
                        - _i32(2) * ws_j)

        base = allocator.get_base()
        q_rm = SmemPtr(base, off_q_rm, T.bf16, shape=(RM_HW,)).get()
        o_rm = SmemPtr(base, off_o_rm, T.bf16, shape=(RM_HW,)).get()
        if FP8_QK:
            # Typed as i32 so every offset below is a dword index: the tile is
            # only ever touched 8 or 16 bytes at a time and never element-wise,
            # and a dword view keeps the address arithmetic in the same units
            # as the buffer_loads that fill it.
            q8_lds = SmemPtr(
                base, off_q8, T.i32, shape=((D // 32) * Q8BS // 4,)
            ).get()
        Q8BS_DW = Q8BS // 4
        if F8B:
            o8_lds = SmemPtr(
                base, off_o8, T.i32, shape=((D // 32) * Q8BS // 4,)
            ).get()
            kt8_lds = SmemPtr(base, off_kt, T.i32,
                              shape=(max(BN * D, BN * DS8R) // 4,)).get()
        if PIPE:
            q8b_lds = SmemPtr(
                base, off_q8b, T.i32, shape=((D // 32) * Q8BS // 4,)).get()
            o8b_lds = SmemPtr(
                base, off_o8b, T.i32, shape=((D // 32) * Q8BS // 4,)).get()
        if not tr16:
            q_tr = SmemPtr(base, off_q_tr, T.bf16, shape=(D * ST,)).get()
            o_tr = SmemPtr(base, off_o_tr, T.bf16, shape=(D * ST,)).get()
        if fused:
            kt_lds = SmemPtr(base, off_kt, T.bf16, shape=(BN * D,)).get()
            ds_lds = SmemPtr(base, off_kt, T.bf16, shape=(BM * BN,)).get()

        lane = tid % _i32(64)
        wave = tid // _i32(64)
        l16 = lane % _i32(16)
        l4 = lane // _i32(16)
        seq_last = seq_len - _i32(1)

        # Opaque zero for the ldsa2/ldsb2 footprint ablations: seq_len is a
        # runtime kernel value and is never 0 (the launcher skips empty
        # sequences), so this selects 0 every time, but LLVM cannot prove it
        # and therefore cannot constant-fold `oz * c` or CSE two reads whose
        # step constants differ.
        def _oz_mul(c):
            if c == 0:
                return _i32(0)
            nz = arith.cmpi(arith.CmpIPredicate.ne, seq_len, _i32(0))
            return arith.select(nz, _i32(0), _i32(c))

        # ---- K / V fragments, live across the whole m loop -------------------
        # Wave `wave` owns n rows [n0 + wave*16*NT, +16*NT), as NT 16-row tiles.
        kf = []
        vf = []
        kf8 = []
        vf8 = []  # F8B only: one v8i32 per n tile
        for nt in range_constexpr(NT):
            n_row = n0 + wave * _i32(16 * NT) + _i32(16 * nt) + l16
            n_row_safe = arith.select(
                arith.cmpi(arith.CmpIPredicate.slt, n_row, seq_len), n_row, seq_last
            )
            kv_base = (
                arith.ArithValue(n_row_safe) * _i32(RS_IN) + l4 * _i32(8)
            )
            kf.append([])
            vf.append([])
            kf8.append([])
            for ks in range_constexpr(KSTEP):
                o = kv_base + _i32(32 * ks)
                # K feeds the qk gemm and nothing else (dq's B operands come
                # from the separately staged kt_lds), so under FP8_QK it never
                # needs widening: the cvt chain disappears and the fragment
                # halves from 4 to 2 VGPRs.
                if not FP8_QK:
                    kf[nt].append(load_qkv(rK, o))
                if not F8B:
                    vf[nt].append(load_qkv(rV, o))
            if FP8_QK in ("k32", "k32w"):
                # Same 8-bytes-at-d = 8*l4 + 32*ks slicing the bf16 op used.
                for ks in range_constexpr(KSTEP):
                    kf8[nt].append(load_raw8(rK, kv_base + _i32(32 * ks)))
            elif FP8_QK == "k128":
                # K = 128 wants lane l4 to hold d in [32*l4, 32*l4 + 32): one
                # 32 B run, which is two dwordx4 loads (a 256-bit buffer load
                # is not selectable on gfx950).
                kb = (arith.ArithValue(n_row_safe) * _i32(RS_IN)
                      + l4 * _i32(32)) // _i32(4)
                kf8[nt].append(vector.shuffle(
                    buffer_ops.buffer_load(rK, kb, vec_width=4, dtype=T.i32),
                    buffer_ops.buffer_load(rK, kb + _i32(4), vec_width=4,
                                           dtype=T.i32),
                    [0, 1, 2, 3, 4, 5, 6, 7]))
                if F8B:
                    # dp's B operand: V rows n, k = d, same 32 B run as K.
                    vf8.append(vector.shuffle(
                        buffer_ops.buffer_load(rV, kb, vec_width=4,
                                               dtype=T.i32),
                        buffer_ops.buffer_load(rV, kb + _i32(4), vec_width=4,
                                               dtype=T.i32),
                        [0, 1, 2, 3, 4, 5, 6, 7]))

        st_m = (tid // _i32(16)) * _i32(4)
        st_d = (tid % _i32(16)) * _i32(8)

        def load_tiles(m0):
            """Issue the global loads for the Q and DO tiles at row offset m0.

            No clamp: rows at or past seq_len fall outside the descriptor's
            num_records and read back as zero, which is the value the math wants.
            """
            out = []
            if F8B:
                # Q and dO both arrive as e4m3 and are staged raw; nothing is
                # widened anywhere in the loop.
                for src, src_rs in ((rQ, RS_Q), (rDO, HD)):
                    for r in range_constexpr(4):
                        m = arith.ArithValue(m0 + st_m + _i32(r))
                        out.append(load_raw8(src, m * _i32(src_rs) + st_d))
                if do_blk:
                    # Two group bytes per lane for tile m0, prefetched with the
                    # tile (a load inside one_iter would queue behind the next
                    # tile's prefetch on vmcnt):
                    #   [8] out-gemm K-block byte: group (m0, l4 % 2), i.e. the
                    #       group's first token m0 + 8*(l4 % 2);
                    #   [9] dq row byte: row l16 -> group (m0, l16 // 8).
                    for tok in (m0 + (l4 % _i32(2)) * _i32(8),
                                m0 + (l16 // _i32(8)) * _i32(8)):
                        b = buffer_ops.buffer_load(
                            rE8, arith.ArithValue(tok) * _i32(H),
                            vec_width=1, dtype=T.i8)
                        out.append(arith.extui(T.i32, b))
                return out
            raws = []  # FP8_QK only: Q's un-widened bytes, appended after dO
            for source_index, (src, src_rs) in enumerate(
                ((rQ, RS_Q), (rDO, HD))
            ):
                for r in range_constexpr(4):
                    m = arith.ArithValue(m0 + st_m + _i32(r))
                    off = m * _i32(src_rs) + st_d
                    if input_fp8 and source_index == 0:
                        if FP8_QK:
                            # One load, two consumers.  The raw dwords go to the
                            # fp8 tile for qk; the widened copy still has to be
                            # produced because the dk gemm reads Q back
                            # transposed as a bf16 B-operand, and converting
                            # there instead would quadruple the cvt count (the
                            # tile is read ~8x more than it is written).
                            # Kept as a flat tail rather than a tuple: `staged`
                            # is carried through scf.for as iter_args.
                            raw = load_raw8(src, off)
                            raws.append(raw)
                            out.append(_widen_raw8(raw))
                        else:
                            out.append(
                                load_qkv(src, off, source_fp8=not q_bf16)
                            )
                    else:
                        out.append(
                            buffer_ops.buffer_load(
                                src, off, vec_width=8, dtype=T.bf16
                            )
                        )
            return out + raws

        # d block index and offset within it for this thread's 8-wide store.
        st_db = (tid % _i32(16)) // _i32(2)
        st_din = ((tid % _i32(16)) % _i32(2)) * _i32(8)

        # fp8 Q tile addressing: d_block = st_d // 32 at stride Q8BS_DW dwords,
        # m-row within it at stride 8 dwords (32 B), byte offset (st_d % 32)
        # inside the row -- always a multiple of 8 B because st_d is.
        st8_db = st_d // _i32(32)
        st8_in = (st_d % _i32(32)) // _i32(4)

        def write_tiles(vecs, bufs=None):
            if F8B:
                for t, lds8 in enumerate(bufs or (q8_lds, o8_lds)):
                    for r in range_constexpr(4):
                        lo = (arith.ArithValue(st8_db) * _i32(Q8BS_DW)
                              + (arith.ArithValue(st_m) + _i32(r)) * _i32(8)
                              + st8_in)
                        vector.store(vecs[4 * t + r], lds8, [_idx(lo)],
                                     alignment=8)
                return
            if FP8_QK:
                # vecs[8:12] are Q's un-widened bytes; publish them to the fp8
                # tile, then fall through with the leading 8 bf16 tiles.
                q_raw = list(vecs[8:12])
                for r in range_constexpr(4):
                    lo = (arith.ArithValue(st8_db) * _i32(Q8BS_DW)
                          + (arith.ArithValue(st_m) + _i32(r)) * _i32(8)
                          + st8_in)
                    vector.store(q_raw[r], q8_lds, [_idx(lo)], alignment=8)
            if tr16:
                for t, lds_rm in enumerate((q_rm, o_rm)):
                    rows = vecs[4 * t : 4 * t + 4]
                    for r in range_constexpr(4):
                        # d_block=st_db at stride DBS (padded), m-row within it
                        # at stride 16, d%16 half at st_din.
                        lo = (arith.ArithValue(st_db) * _i32(DBS)
                              + (arith.ArithValue(st_m) + _i32(r)) * _i32(16)
                              + st_din)
                        vector.store(rows[r], lds_rm, [_idx(lo)], alignment=16)
                return
            for t, (lds_rm, lds_tr) in enumerate(((q_rm, q_tr), (o_rm, o_tr))):
                rows = vecs[4 * t : 4 * t + 4]
                for r in range_constexpr(4):
                    lo = (st_m + _i32(r)) * _i32(SQ) + st_d
                    vector.store(rows[r], lds_rm, [_idx(lo)], alignment=rm_align)
                if ABL_TR:
                    continue
                for e in range_constexpr(8):
                    elems = [
                        vector.extract(rows[r], static_position=[e], dynamic_position=[])
                        for r in range_constexpr(4)
                    ]
                    vector.store(
                        vector.from_elements(v4bf, elems),
                        lds_tr,
                        [_idx((st_d + _i32(e)) * _i32(ST) + st_m)],
                        alignment=8,
                    )

        l4h = l4 // _i32(2)  # d block within a 32-wide k slice
        l4l = (l4 % _i32(2)) * _i32(8)  # offset inside that 16-wide d block
        # ds_read_b64_tr_b16, measured on gfx950:
        # for lane 16*g + j, write k = j/4 and s = j%4.  Element e of that lane
        # is the halfword at (address given by lane 16*g + 4*e + k) + s.  So each
        # 16-lane group runs four independent 4x4 transposes, and lane j supplies
        # the address for row e = j/4 of sub-group k = j%4.
        l16q = l16 // _i32(4)
        l16r = (l16 % _i32(4)) * _i32(4)

        def tr_read(lds, row_base, inner, pad_off=0, oz=False):
            """One ds_read_b64_tr_b16 out of a tile in the tr16 layout.

            The layout is halfword(r, c) = ((c/16)*ROWS + r)*16 + c%16, and
            `row_base` selects the (c/16) block as (c/16)*ROWS.  The read hands
            lane j the four r values 4*l4 + 16*inner + 0..3 taken at column
            c = 16*(c/16) + j: a transposed 4-wide slice with the lane index on
            the c axis.  Three tiles use it -- Q and dO with (r, c) = (m, d),
            K with (n, d), and the ds exchange with (n, m).
            """
            cst = _oz_mul if oz else _i32
            if isinstance(row_base, int):
                row_base = cst(row_base)
            off = (arith.ArithValue(row_base) + l4 * _i32(4) + cst(16 * inner)
                   + arith.ArithValue(l16q)) * _i32(16) + l16r
            # pad_off (halfwords) rotates a padded d_block's base; row_base is
            # d_block*ROWS so the padding cannot fold into the *16 above.
            if pad_off:
                off = arith.ArithValue(off) + cst(pad_off)
            return rocdl_raw.ds_read_tr16_b64_(v4bf, _lds_gep(lds, off, T.bf16))

        def tr_pair(lds, row_base, inner2):
            """K=32 fragment: two tr reads whose r values differ by 16.

            The mfma only requires A and B to agree on which logical k each
            (lane, element) slot carries, not that the k values be contiguous,
            so the pair can be concatenated as it comes out.
            """
            return vector.shuffle(tr_read(lds, row_base, 2 * inner2),
                                  tr_read(lds, row_base, 2 * inner2 + 1), CAT8)

        def a_frag(lds, mstep, kstep):
            if ABL_LDSA:
                mstep = kstep = 0
            cst = _oz_mul if ABL_LDSA2 else _i32
            if tr16:
                # d = l4*8 + 32*kstep, so d/16 = l4/2 + 2*kstep and d%16 = 8*(l4%2)
                off = ((arith.ArithValue(l4h) + cst(2 * kstep)) * _i32(DBS)
                       + (l16 + cst(16 * mstep)) * _i32(16) + l4l)
                return vector.load_op(v8bf, lds, [_idx(off)])
            off = (l16 + cst(16 * mstep)) * _i32(SQ) + l4 * _i32(8) + cst(32 * kstep)
            return vector.load_op(v8bf, lds, [_idx(off)])

        def a_frag8(mstep, kstep, lds8=None):
            """Q A-operand as raw e4m3 out of the fp8 tile.

            k32:  8 B at d = 8*l4 + 32*kstep -> block kstep, byte 8*l4 in the
                  row.  One ds_read_b64, half the bytes of the bf16 b128 it
                  replaces.
            k128: all 32 B of block l4 at this m, as two b128 -- four b128
                  reads and four mfma collapse to two reads and one mfma.
            """
            row = (l16 + _i32(16 * mstep)) * _i32(8)
            if FP8_QK == "k32":
                off = (_i32(kstep * Q8BS_DW) + row + l4 * _i32(2))
                return vector.load_op(v2i32, q8_lds, [_idx(off)])
            if FP8_QK == "k32w":
                # k128's read, k32's mfma: pull the full 32 B once and hand out
                # the four K=32 slices.  d = 8*l4 + 32*ks lives in block ks at
                # byte 8*l4, so the four blocks a lane needs are Q8BS_DW apart
                # and each slice is dwords [2*l4, 2*l4+2) of its block -- which
                # is NOT what one 32 B run holds, so the probe reads the two
                # b128 that cover blocks 0..1 and 2..3 of THIS lane's l4 slot.
                off = arith.ArithValue(l4) * _i32(Q8BS_DW) + row
                wide = vector.shuffle(
                    vector.load_op(v4i32, q8_lds, [_idx(off)]),
                    vector.load_op(v4i32, q8_lds,
                                   [_idx(arith.ArithValue(off) + _i32(4))]),
                    [0, 1, 2, 3, 4, 5, 6, 7])
                return vector.shuffle(wide, wide, [2 * kstep, 2 * kstep + 1])
            src8 = q8_lds if lds8 is None else lds8
            off = arith.ArithValue(l4) * _i32(Q8BS_DW) + row
            return vector.shuffle(
                vector.load_op(v4i32, src8, [_idx(off)]),
                vector.load_op(v4i32, src8, [_idx(arith.ArithValue(off)
                                                  + _i32(4))]),
                [0, 1, 2, 3, 4, 5, 6, 7])

        # ---- F8B transposed operand addressing (see tr8 above) ------------
        # Source lane j = l16: row slot e = j//2, receiver half k = j%2.
        t8_e = l16 // _i32(2)
        t8_k8 = (l16 % _i32(2)) * _i32(8)
        # Out-gemm B (dO^T / q^T) for an mu2 HALF: element e is
        # m = 16*(2r + e//4) + 4*l4 + e%4 -- exactly the m order the silu / ds
        # A pieces carry (pk8 of m-subtiles 2r and 2r+1).
        t8_m = ((t8_e // _i32(4)) * _i32(16) + l4 * _i32(4)
                + t8_e % _i32(4))

        def b8_tr(lds8, r, ds):
            off = (_i32((ds // 2) * Q8BS + 16 * (ds % 2) + 32 * 32 * r)
                   + arith.ArithValue(t8_m) * _i32(32) + t8_k8)
            return vector.bitcast(T.vec(4, T.i16), tr8(lds8, off))

        # dq operands: k byte b of lane group l4 is n = 32*l4 + b.
        t8_n = l4 * _i32(32) + t8_e

        def dq8_a(ms):
            """ds as dq's A operand: row m = 16*ms + l16, 32 n per lane."""
            parts = [tr8(kt8_lds, (arith.ArithValue(t8_n) + _i32(8 * r))
                         * _i32(DS8R) + t8_k8 + _i32(16 * ms))
                     for r in range_constexpr(4)]
            return vector.shuffle(vector.shuffle(parts[0], parts[1], [0, 1, 2, 3]),
                                  vector.shuffle(parts[2], parts[3], [0, 1, 2, 3]),
                                  list(range(8)))

        def b_frag_tr(lds, mtile, dstep):
            """4 consecutive m at one d, straight out of the row-major tile.

            ds_read_b64_tr_b16 works on 16-lane groups: lanes 4e of a group supply
            the four row base addresses, and lane j gets element e = row_e + j.
            With the tr16 layout row_e is (m = 4*(lane/16) + 16*mtile + e) at d
            block `dstep`, so lane j receives m = q0..q0+3 at d = 16*dstep + j --
            exactly the transposed B operand, with no second LDS copy.
            """
            if ABL_LDSB:
                mtile = dstep = 0
            # q_rm/o_rm carry tr_pad halfwords of padding per d_block; add the
            # d_block's share (dstep*tr_pad) so the read matches write_tiles.
            return tr_read(lds, dstep * BM, mtile, pad_off=dstep * tr_pad,
                           oz=ABL_LDSB2)

        def b_frag(lds, mtile, dstep):
            cst = _oz_mul if ABL_LDSB2 else _i32
            off = (l16 + cst(16 * dstep)) * _i32(ST) + l4 * _i32(4) + cst(16 * mtile)
            return vector.bitcast(v4i16, vector.load_op(v4bf, lds, [_idx(off)]))

        def b_frag_bf(lds, mtile, dstep):
            # Same transposed B-operand slice as b_frag but left as native bf16
            # so two consecutive m-tiles can be concatenated into a K=32 fragment.
            if ABL_LDSB:
                mtile = dstep = 0
            cst = _oz_mul if ABL_LDSB2 else _i32
            off = (l16 + cst(16 * dstep)) * _i32(ST) + l4 * _i32(4) + cst(16 * mtile)
            return vector.load_op(v4bf, lds, [_idx(off)])

        n_idx_score = [
            n0 + wave * _i32(16 * NT) + _i32(16 * nt) + l16 for nt in range(NT)
        ]

        def act_tile(m0, qk, dp, masked):
            """silu / ds of one m tile from its qk / dp accumulators."""
            silu_f = []
            ds_f = []
            silu32 = []  # MU2 only: the f32 fragments, before any narrowing
            ds32 = []
            ds_raw = []  # same fragments as ds_f, kept as bf16 for the LDS store
            for nt in range_constexpr(NT):
                silu_f.append([])
                ds_f.append([])
                silu32.append([])
                ds32.append([])
                ds_raw.append([])
                for ms in range_constexpr(MSTEP):
                    s_el = []
                    d_el = []
                    pk_silu = [None] * 4
                    pk_ds = [None] * 4
                    if EPI_PK and not ABL_ACT:
                        # The "reuse" chain below, two accumulator elements
                        # per issue.  Op for op (and operand order for the
                        # non-commutative 1-sig and the fma) identical to the
                        # scalar branch, so every lane rounds identically.
                        q4 = qk[nt][ms]
                        p4 = dp[nt][ms]

                        def _ex2(v2, j):
                            return arith.unwrap(vector.extract(
                                v2, static_position=[j], dynamic_position=[]))

                        for h2 in range_constexpr(2):
                            sel = [2 * h2, 2 * h2 + 1]
                            x2 = arith.mulf(vector.shuffle(q4, q4, sel),
                                            pk_sas2)
                            t2 = arith.mulf(x2, pk_nl2e2)
                            e2 = vector.from_elements(V2F, [
                                rocdl.exp2(T.f32, _ex2(t2, j))
                                for j in range_constexpr(2)])
                            d2 = arith.addf(e2, pk_one2)
                            sig2 = vector.from_elements(V2F, [
                                rocdl.rcp(T.f32, _ex2(d2, j))
                                for j in range_constexpr(2)])
                            sinv2 = arith.mulf(sig2, pk_inv2)
                            silu2 = arith.mulf(x2, sinv2)
                            oms2 = arith.subf(pk_one2, sig2)
                            gc2 = llvm.intr_fma(arith.unwrap(silu2),
                                                arith.unwrap(oms2),
                                                arith.unwrap(sinv2))
                            dv2 = arith.mulf(vector.shuffle(p4, p4, sel),
                                             pk_vis2)
                            ds2 = arith.mulf(dv2, gc2)
                            for j in range_constexpr(2):
                                pk_silu[2 * h2 + j] = arith.ArithValue(
                                    _ex2(silu2, j))
                                pk_ds[2 * h2 + j] = arith.ArithValue(
                                    _ex2(ds2, j))
                    for i in range_constexpr(4):
                        raw = arith.ArithValue(
                            vector.extract(qk[nt][ms], static_position=[i],
                                           dynamic_position=[])
                        )
                        dpv = arith.ArithValue(
                            vector.extract(dp[nt][ms], static_position=[i],
                                           dynamic_position=[])
                        )
                        if EPI_PK and not ABL_ACT:
                            silu = pk_silu[i]
                            dsv = pk_ds[i]
                        elif ABL_ACT:
                            silu = raw
                            dsv = dpv
                        else:
                            if EPI_REUSE:
                                x = raw * score_alpha_scale
                            else:
                                x = raw * _f32(alpha) * score_input_scale
                            e = rocdl.exp2(T.f32, arith.unwrap(x * _f32(-LOG2E)))
                            sig = arith.ArithValue(
                                rocdl.rcp(T.f32, arith.unwrap(arith.ArithValue(e) + _f32(1.0)))
                            )
                            # inv_max is a scalar KERNEL ARGUMENT, not a baked
                            # constant: production derives max_seq_len per batch
                            # as max_uih_len + num_candidates, so baking it into
                            # the lru_cache key would JIT a fresh kernel per
                            # batch.  It lands in an SGPR and the multiply is
                            # the same one that was there before, so this is
                            # bit-identical to the constant-folded version for
                            # every max_seq_len, powers of two or not -- unlike
                            # folding it into the epilogue, which would move the
                            # bf16 rounding point.
                            if EPI_REUSE:
                                # SiLU'(x)/N = sig/N + (x*sig/N)*(1-sig):
                                # share sig/N and the SiLU value itself.
                                sig_inv = sig * inv_max
                                silu = arith.ArithValue(x) * sig_inv
                                gc = arith.ArithValue(llvm.intr_fma(
                                    arith.unwrap(silu),
                                    arith.unwrap(_f32(1.0) - sig),
                                    arith.unwrap(sig_inv)))
                            else:
                                silu = arith.ArithValue(x * sig) * inv_max
                                gc = arith.ArithValue(
                                    sig * (_f32(1.0) + arith.ArithValue(x) * (_f32(1.0) - sig))
                                ) * inv_max
                            dsv = (
                                dpv
                                * value_input_scale
                                * arith.ArithValue(gc)
                            )
                        if masked and not ABL_MASK:
                            # Causal only.  The bounds test that used to sit here
                            # is gone: out-of-range q/dO rows arrive as zeros from
                            # the buffer descriptor and are already exactly zero.
                            m_idx = m0 + _i32(16 * ms) + l4 * _i32(4) + _i32(i)
                            ok = arith.cmpi(arith.CmpIPredicate.sge, m_idx,
                                            n_idx_score[nt])
                            silu = arith.select(ok, silu, _f32(0.0))
                            dsv = arith.select(ok, dsv, _f32(0.0))
                        s_el.append(silu)
                        d_el.append(dsv)
                    # One vector truncf per fragment: v4f32 -> v4bf16 lowers to
                    # v_cvt_pk_bf16_f32 pairs that pack two f32 into two bf16
                    # halfwords directly, vs 4 scalar truncs + v_perm_b32 shuffles
                    # to assemble the v4i16.  The silu/ds values are already scalar
                    # out of the scalar-only exp2/rcp chain, so this tail pack does
                    # NOT fight the transcendental repack barrier -- it only removes
                    # the halfword-assembly perms from the issue-bound VALU stream.
                    s_v = vector.from_elements(T.f32x4, s_el)
                    d_v = vector.from_elements(T.f32x4, d_el)
                    silu32[nt].append(s_v)
                    ds32[nt].append(d_v)
                    silu_bf = arith.trunc_f(v4bf, s_v)
                    ds_bf = arith.trunc_f(v4bf, d_v)
                    silu_f[nt].append(vector.bitcast(v4i16, silu_bf))
                    ds_raw[nt].append(ds_bf)
                    ds_f[nt].append(vector.bitcast(v4i16, ds_bf))
            return silu_f, ds_f, silu32, ds32, ds_raw

        def f8b_cast(silu32, ds32):
            s8, d8 = [], []
            for nt in range_constexpr(NT):
                s8.append([pk4_from_f32(silu32[nt][ms], mu2_ssc,
                                        "hi" if f8_sat else False)
                           for ms in range_constexpr(MSTEP)])
                d8.append([pk4_from_f32(ds32[nt][ms], mu2_dsc,
                                        "both" if f8_sat else False)
                           for ms in range_constexpr(MSTEP)])
            return s8, d8

        def f8b_ds_store(d8):
            for nt in range_constexpr(NT):
                for ms in range_constexpr(MSTEP):
                    n_loc = (wave * _i32(16 * NT) + _i32(16 * nt) + l16)
                    off = (arith.ArithValue(n_loc) * _i32(DS8R)
                           + _i32(16 * ms) + l4 * _i32(4)) // _i32(4)
                    vector.store(
                        vector.bitcast(T.vec(1, T.i32), d8[nt][ms]),
                        kt8_lds, [_idx(off)], alignment=4)

        def dqsplit_ws_store(m0):
            """Read the exchanged ds tile back (tr8) and store it to the ws."""
            parts = [tr8(kt8_lds, (arith.ArithValue(t8_n) + _i32(8 * r))
                         * _i32(DS8R) + t8_k8 + wave * _i32(16))
                     for r in range_constexpr(4)]
            a8 = vector.shuffle(
                vector.shuffle(parts[0], parts[1], [0, 1, 2, 3]),
                vector.shuffle(parts[2], parts[3], [0, 1, 2, 3]),
                list(range(8)))
            ws_t = arith.ArithValue(m0) // _i32(64)
            ws_off = ((ws_jbase + ws_t) * _i32(8192)
                      + (wave * _i32(16) + l16) * _i32(128)
                      + l4 * _i32(32))
            for q4 in range_constexpr(2):
                v = vector.shuffle(a8, a8,
                                   [4 * q4 + i for i in range(4)])
                rocdl_raw.RawPtrBufferStoreOp(
                    arith.unwrap(v), rWS,
                    arith.unwrap(arith.ArithValue(ws_off) + _i32(16 * q4)),
                    arith.unwrap(_i32(0)), arith.unwrap(_i32(0)))

        def one_iter(m0, acc_dv, acc_dk, masked, pend=None, e8=None):
            # e8 (do_blk only): [kblock byte, dq row byte] of tile m0, as
            # loaded by load_tiles(m0) -- elements 8:10 of its result.
            assert (e8 is not None) == do_blk
            # A-operands of the score gemms and B-operands of the out gemms are
            # both independent of n, so they are read once here and fed to all NT
            # n-tiles.  That is the whole point of a wide BN.
            if FP8_QK in ("k32", "k32w"):
                a_qk_frag = [
                    [a_frag8(ms, ks) for ks in range_constexpr(KSTEP)]
                    for ms in range_constexpr(MSTEP)
                ]
            elif FP8_QK == "k128":
                a_qk_frag = [[a_frag8(ms, 0)] for ms in range_constexpr(MSTEP)]
            else:
                a_qk_frag = [
                    [a_frag(q_rm, ms, ks) for ks in range_constexpr(KSTEP)]
                    for ms in range_constexpr(MSTEP)
                ]
            if F8B:
                a_dp_frag = [[a_frag8(ms, 0, o8_lds)]
                             for ms in range_constexpr(MSTEP)]
            else:
                a_dp_frag = [[a_frag(o_rm, ms, ks)
                              for ks in range_constexpr(KSTEP)]
                             for ms in range_constexpr(MSTEP)]

            qk = []
            dp = []
            for nt in range_constexpr(NT):
                qk.append([])
                dp.append([])
                for ms in range_constexpr(MSTEP):
                    a_qk = zero_acc
                    a_dp = zero_acc
                    if FP8_QK:
                        # dp keeps the bf16 op: dO is genuinely bf16 and
                        # rounding it to e4m3 costs dq/dk 13 dB.
                        if FP8_QK == "k128":
                            a_qk = mfma_qk_k128(a_qk_frag[ms][0], kf8[nt][0],
                                                a_qk)
                        if F8B:
                            a_dp = mfma_qk_k128(a_dp_frag[ms][0], vf8[nt],
                                                a_dp)
                        for ks in range_constexpr(0 if F8B else KSTEP):
                            if FP8_QK in ("k32", "k32w"):
                                a_qk = mfma_qk_k32(a_qk_frag[ms][ks],
                                                   kf8[nt][ks], a_qk)
                            a_dp = mfma2(a_dp_frag[ms][ks], vf[nt][ks], a_dp)
                    elif not ABL_SCORE:
                        for ks in range_constexpr(KSTEP):
                            a_qk = mfma2(a_qk_frag[ms][ks], kf[nt][ks], a_qk)
                            a_dp = mfma2(a_dp_frag[ms][ks], vf[nt][ks], a_dp)
                            if ABL_MFMA2X:
                                a_qk = mfma2(a_qk_frag[ms][ks], kf[nt][ks],
                                             a_qk)
                                a_dp = mfma2(a_dp_frag[ms][ks], vf[nt][ks],
                                             a_dp)
                    else:
                        a_qk = mfma2(kf[nt][0], kf[nt][1], a_qk)
                        a_dp = mfma2(vf[nt][0], vf[nt][1], a_dp)
                    qk[nt].append(a_qk)
                    dp[nt].append(a_dp)

            silu_f, ds_f, silu32, ds32, ds_raw = act_tile(m0, qk, dp, masked)

            s8, d8 = [], []
            if F8B:
                s8, d8 = f8b_cast(silu32, ds32)
                f8b_ds_store(d8)
            elif fused:
                # Hoisted ds-exchange publish.  ds_raw is complete here and the
                # out-gemm mfmas below do not touch ds_lds, so issuing the LDS
                # writes NOW puts ~128 independent mfmas between them and the
                # cross-wave barrier.  The store is async; without work in
                # between, the s_barrier stalls on the full LDS write latency
                # (KB LDS Fix C, exposed write-read distance), and iglp cannot
                # hoist it itself because an s_barrier is a hard scheduling
                # boundary.  Same region, same barrier, same ds_raw liveness as
                # before, so this only moves the store earlier in program order.
                for nt in range_constexpr(NT):
                    for ms in range_constexpr(MSTEP):
                        if ABL_DQW:
                            off = _i32(0)
                        else:
                            off = (_i32(ms * BN) + wave * _i32(16 * NT)
                                   + _i32(16 * nt) + l16) * _i32(16) + l4 * _i32(4)
                        vector.store(ds_raw[nt][ms], ds_lds, [_idx(off)],
                                     alignment=8)

            new_dv = list(acc_dv)
            new_dk = list(acc_dk)
            cur = None
            if MU2 == "k128":
                # ---- TRACK U1: build this m tile's HALF of the K=128 operands.
                # A: silu / ds arrive as four f32 fragments per n-tile, one per
                #    16-row m sub-tile.  Two fragments = 8 f32 = one pk8 cvt =
                #    two dwords, so a whole BM=64 tile is 4 dwords per n-tile.
                # B: dO / q come out of the tr16 tile as v4bf slices, same
                #    (l4, element) -> m mapping as A, so the pieces concatenate
                #    in the same order and the two operands agree on k.
                v4i32_t = T.vec(4, T.i32)
                a_dv_h, a_dk_h = [], []
                for nt in range_constexpr(NT if F8B else 0):
                    def _cat(p):
                        return vector.shuffle(
                            vector.shuffle(p[0], p[1], [0, 1, 2, 3]),
                            vector.shuffle(p[2], p[3], [0, 1, 2, 3]),
                            list(range(8)))
                    a_dv_h.append(_cat(s8[nt]))
                    a_dk_h.append(_cat(d8[nt]))
                for nt in range_constexpr(0 if F8B else NT):
                    def _pack(frags, sc):
                        halves = [
                            pk8_from_f32(frags[nt][2 * j], frags[nt][2 * j + 1],
                                         sc)
                            for j in range_constexpr(MSTEP // 2)
                        ]
                        return vector.shuffle(halves[0], halves[1],
                                              list(range(8)))
                    a_dv_h.append(_pack(silu32, mu2_ssc))
                    a_dk_h.append(_pack(ds32, mu2_dsc))
                bo_h, bq_h = [], []
                for ds in range_constexpr(NSTEP if F8B else 0):
                    for lds8, out in ((o8_lds, bo_h), (q8_lds, bq_h)):
                        out.append(vector.shuffle(b8_tr(lds8, 0, ds),
                                                  b8_tr(lds8, 1, ds),
                                                  list(range(8))))
                for ds in range_constexpr(0 if F8B else NSTEP):
                    for lds, out, bsc in ((o_rm, bo_h, mu2_osc),
                                          (q_rm, bq_h, mu2_qsc)):
                        if ABL_BCVT0:
                            # v4i16 (4 bf16) + v4i16 -> v4i16 "8 fp8": a pure
                            # register select, no VALU, both LDS reads still
                            # feed it.  Ceiling probe only -- see ABL_BCVT0.
                            halves = [
                                vector.shuffle(b_frag_tr(lds, 2 * j, ds),
                                               b_frag_tr(lds, 2 * j + 1, ds),
                                               [0, 1, 4, 5])
                                for j in range_constexpr(MSTEP // 2)
                            ]
                        else:
                            halves = [
                                pk8_from_bf16(b_frag_tr(lds, 2 * j, ds),
                                              b_frag_tr(lds, 2 * j + 1, ds),
                                              bsc)
                                for j in range_constexpr(MSTEP // 2)
                            ]
                        out.append(vector.shuffle(halves[0], halves[1],
                                                  list(range(8))))
                cur = (a_dv_h, a_dk_h, bo_h, bq_h,
                       e8[0] if do_blk else None)
                if pend is not None:
                    p_dv, p_dk, p_bo, p_bq, p_e8 = pend
                    v8i32_t = T.vec(8, T.i32)
                    CAT32 = list(range(16))

                    def _wide(a, b):
                        return vector.bitcast(
                            v8i32_t, vector.shuffle(a, b, CAT32))
                    sc_kb = None
                    if do_blk:
                        # K=128 operand = [pend half | cur half].  Data (lane
                        # group g, half h) holds m%16 in [4g, 4g+4) of tile h
                        # (d8/s8 dword ms = acc rows 16ms+4g+i; b8_tr's t8_m is
                        # the same order) and is HW K-block 2h + g//2, scaled
                        # by scale lanes 16*(2h + g//2) + l16 (P0.c).  So
                        # scale lane group l4 = K-block = (tile l4//2, sub
                        # l4%2); each tile's load already picked sub l4%2.
                        sc_kb = arith.select(
                            arith.cmpi(arith.CmpIPredicate.ult, l4, _i32(2)),
                            p_e8, e8[0])
                    for ds in range_constexpr(NSTEP):
                        b_o = _wide(p_bo[ds], bo_h[ds])
                        b_q = _wide(p_bq[ds], bq_h[ds])
                        for nt in range_constexpr(NT):
                            i = nt * NSTEP + ds
                            if do_blk:
                                new_dv[i] = mfma_k128_sc(
                                    _wide(p_dv[nt], a_dv_h[nt]),
                                    b_o, acc_dv[i], sc127, sc_kb)
                                new_dk[i] = mfma_k128_sc(
                                    _wide(p_dk[nt], a_dk_h[nt]),
                                    b_q, acc_dk[i], sc_kb, sc127)
                                continue
                            new_dv[i] = mfma_out_k128(
                                _wide(p_dv[nt], a_dv_h[nt]),
                                b_o, acc_dv[i])
                            new_dk[i] = mfma_out_k128(
                                _wide(p_dk[nt], a_dk_h[nt]),
                                b_q, acc_dk[i])
                    cur = None
            # Concatenate two consecutive m-tiles' free-transpose score fragments
            # into a K=32 A-operand; the matching B-operand pair is contiguous in
            # M, so the double-rate mfma2 sums the same M values A and B carry.
            cat = [0, 1, 2, 3, 4, 5, 6, 7]
            for ds in range_constexpr(0 if MU2 == "k128" else NSTEP):
                bo_frag = []
                bq_frag = []
                for j in range_constexpr(MSTEP // 2):
                    if tr16:
                        bo_frag.append(vector.shuffle(
                            b_frag_tr(o_rm, 2 * j, ds),
                            b_frag_tr(o_rm, 2 * j + 1, ds), cat))
                        bq_frag.append(vector.shuffle(
                            b_frag_tr(q_rm, 2 * j, ds),
                            b_frag_tr(q_rm, 2 * j + 1, ds), cat))
                    elif ABL_TR:
                        bo_frag.append(a_frag(o_rm, 2 * j, ds))
                        bq_frag.append(a_frag(q_rm, 2 * j, ds))
                    else:
                        bo_frag.append(vector.shuffle(
                            b_frag_bf(o_tr, 2 * j, ds),
                            b_frag_bf(o_tr, 2 * j + 1, ds), cat))
                        bq_frag.append(vector.shuffle(
                            b_frag_bf(q_tr, 2 * j, ds),
                            b_frag_bf(q_tr, 2 * j + 1, ds), cat))
                for nt in range_constexpr(NT):
                    adv = acc_dv[nt * NSTEP + ds]
                    adk = acc_dk[nt * NSTEP + ds]
                    if ABL_OUT:
                        # Attribution only (Task C, scratch): the old form fed
                        # kf/vf, which are K=32 v8bf16 fragments since the fp8
                        # widening landed, into the half-rate 16x16x16 mfma and
                        # no longer verifies.  Keep j=0 only, so the out-gemm
                        # mfma count halves and nothing else changes.
                        adv = mfma2(
                            vector.bitcast(v8bf, vector.shuffle(
                                silu_f[nt][0], silu_f[nt][1], cat)),
                            bo_frag[0], adv)
                        adk = mfma2(
                            vector.bitcast(v8bf, vector.shuffle(
                                ds_f[nt][0], ds_f[nt][1], cat)),
                            bq_frag[0], adk)
                    else:
                        for j in range_constexpr(MSTEP // 2):
                            a_dv = vector.bitcast(
                                v8bf,
                                vector.shuffle(silu_f[nt][2 * j],
                                               silu_f[nt][2 * j + 1], cat),
                            )
                            a_dk = vector.bitcast(
                                v8bf,
                                vector.shuffle(ds_f[nt][2 * j],
                                               ds_f[nt][2 * j + 1], cat),
                            )
                            adv = mfma2(a_dv, bo_frag[j], adv)
                            adk = mfma2(a_dk, bq_frag[j], adk)
                            if ABL_MFMA2XO:
                                adv = mfma2(a_dv, bo_frag[j], adv)
                                adk = mfma2(a_dk, bq_frag[j], adk)
                    new_dv[nt * NSTEP + ds] = adv
                    new_dk[nt * NSTEP + ds] = adk

            if fused:
                # The ds-exchange publish (tr16 layout, (r, c) = (n, m)) was
                # hoisted above the out-gemm mfmas; only the cross-wave barrier
                # remains here so the ~128 mfmas cover the LDS write latency.
                if not ABL_DQB:
                    gpu.barrier()
                # Read it back transposed and contract over the full BN.  The
                # accumulator is complete for this m tile after one iteration,
                # so it is flushed here rather than carried.
                acc_dq = [zero_acc] * (MSTEP * DQD)
                if DQSPLIT:
                    # The ds tile is complete in LDS after the barrier above.
                    # dq8_a(ms) (here ms = wave) is ds[m = 16*ms + l16][n = 32*l4 .. 32*l4+31]
                    # in natural byte order (tr8 receiver byte e of part r is
                    # n = 32*l4 + 8r + e), so wave w stores rows 16w..16w+15
                    # of an m-major [m 64][n 128] e4m3 tile: the 4 l4 lanes
                    # of a row write one full 128 B line.
                    dqsplit_ws_store(m0)
                for ms in range_constexpr(MSTEP if (F8B and not DQSPLIT) else 0):
                    # One K=128 (= BN) fp8 mfma per (m tile, d tile) instead
                    # of DQK=4 bf16 K=32 ones.  bf16p orientation: ds is A.
                    a8 = dq8_a(ms)
                    for dt in range_constexpr(DQD):
                        if do_blk:
                            # A row m = 16*ms + l16 carries its group's
                            # 2**e_g in every K-block: A scale lane l (row
                            # l16, any K-block) = byte of group (m0, l16//8).
                            acc_dq[ms * DQD + dt] = mfma_k128_sc(
                                a8, kt_frag[dt], acc_dq[ms * DQD + dt],
                                e8[1], sc127)
                            continue
                        acc_dq[ms * DQD + dt] = mfma_qk_k128(
                            a8, kt_frag[dt], acc_dq[ms * DQD + dt])
                for ms in range_constexpr(0 if F8B else MSTEP):
                    for t in range_constexpr(DQK):
                        a_dq = vector.bitcast(
                            v8bf, tr_pair(ds_lds, 0 if ABL_DQR else ms * BN,
                                          0 if ABL_DQR else t))
                        for dt in range_constexpr(DQD):
                            # Operand order decides the accumulator's
                            # orientation.  f32 wants C[m][d], so ds supplies M;
                            # a packed 16-bit flush wants C[d][m], so that a
                            # lane's four elements are four CONSECUTIVE d and
                            # pair up into dwords -- then k supplies M instead.
                            a, b = ((kt_frag[dt][t], a_dq)
                                    if dq16 in ("bf16", "f16")
                                    else (a_dq, kt_frag[dt][t]))
                            acc_dq[ms * DQD + dt] = mfma2(
                                a, b, acc_dq[ms * DQD + dt])
                            if ABL_MFMA2XQ:
                                acc_dq[ms * DQD + dt] = mfma2(
                                    a, b, acc_dq[ms * DQD + dt])
                sc = _f32(alpha * dq_scale / (mu2_dsc if F8B else 1.0))
                if input_fp8:
                    sc = arith.ArithValue(sc) * value_input_scale
                if do_dyn:
                    sc = arith.ArithValue(sc) * inv_s_do
                if dq16 == "bf16p" and not DQSPLIT:
                    # The staged K tile was interleaved so that lane l16's two
                    # accumulators hold d = 32*wave + 2*l16 and +1.  Packing them
                    # into one dword makes the 16 lanes of a row cover
                    # [32*wave, 32*wave+32) in bf16 = 64 contiguous bytes, one
                    # aligned line per request, with every lane live.
                    dqt = dq_ty()
                    v2 = T.vec(2, dqt)
                    for ms in range_constexpr(MSTEP):
                        lo_a = acc_dq[ms * DQD + 0]
                        hi_a = acc_dq[ms * DQD + 1]
                        row = m0 + _i32(16 * ms) + l4 * _i32(4)
                        col = wave * _i32(32) + l16 * _i32(2)
                        for i in range_constexpr(1 if ABL_DQF else 4):
                            pk = vector.from_elements(v2, [
                                arith.trunc_f(dqt, arith.ArithValue(
                                    vector.extract(a, static_position=[i],
                                                   dynamic_position=[])) * sc)
                                for a in (lo_a, hi_a)
                            ])
                            off = ((arith.ArithValue(row) + _i32(i)) * _i32(RS_OUT)
                                   + col) * _i32(2)
                            _atomic_fadd(pk, rDQ, off, dq_aux)
                if dq16 == "bf16c":
                    # Lane l owns d = 16*dt + l16 and l16 runs 0..15, so the
                    # pair (d, d+1) that shares a dword straddles lanes l and
                    # l^1.  One DPP swap brings the partner's value in; the odd
                    # lanes are then redundant and are steered out of range.
                    dqt = dq_ty()
                    v2 = T.vec(2, dqt)
                    keep = arith.cmpi(arith.CmpIPredicate.eq,
                                      l16 % _i32(2), _i32(0))
                    for ms in range_constexpr(MSTEP):
                        for dt in range_constexpr(DQD):
                            a = acc_dq[ms * DQD + dt]
                            row = m0 + _i32(16 * ms) + l4 * _i32(4)
                            col = wave * _i32(16 * DQD) + _i32(16 * dt) + l16
                            # The DPP swap has to run with BOTH lanes of a pair
                            # enabled, so the pack is built before the odd lanes
                            # are masked off.
                            pks = []
                            for i in range_constexpr(1 if ABL_DQF else 4):
                                lo = arith.ArithValue(
                                    vector.extract(a, static_position=[i],
                                                   dynamic_position=[])) * sc
                                hi = _dpp_swap_adj(lo)
                                pks.append(vector.from_elements(v2, [
                                    arith.trunc_f(dqt, lo),
                                    arith.trunc_f(dqt, hi)]))
                            # Steering the odd lanes out of num_records instead
                            # of masking exec faults the queue: unlike loads,
                            # an out-of-range buffer ATOMIC is not discarded.
                            odd_off = scf.IfOp(keep, results_=[], has_else=False)
                            with ir.InsertionPoint(odd_off.then_block):
                                for i, pk in enumerate(pks):
                                    off = ((arith.ArithValue(row) + _i32(i))
                                           * _i32(RS_OUT) + col) * _i32(2)
                                    _atomic_fadd(pk, rDQ, off, dq_aux)
                                scf.YieldOp([])
                for ms in range_constexpr(
                        MSTEP if dq16 not in ("bf16c", "bf16p") else 0):
                    for dt in range_constexpr(DQD):
                        a = acc_dq[ms * DQD + dt]
                        if dq16:
                            m_idx = m0 + _i32(16 * ms) + l16
                            d_idx = (wave * _i32(16 * DQD) + _i32(16 * dt)
                                     + l4 * _i32(4))
                            base = (
                                arith.ArithValue(m_idx) * _i32(RS_OUT) + d_idx
                            )
                            dqt = dq_ty()
                            for p in range_constexpr(1 if ABL_DQF else 2):
                                pk = vector.from_elements(T.vec(2, dqt), [
                                    arith.trunc_f(dqt, arith.ArithValue(
                                        vector.extract(a, static_position=[2 * p + s],
                                                       dynamic_position=[])) * sc)
                                    for s in range(2)
                                ])
                                off = (base + _i32(2 * p)) * _i32(2)
                                _atomic_fadd(pk, rDQ, off, dq_aux)
                            continue
                        row = ((_i32(0) if ABL_DQZ else m0)
                               + _i32(16 * ms) + l4 * _i32(4))
                        col = wave * _i32(16 * DQD) + _i32(16 * dt) + l16
                        for i in range_constexpr(1 if ABL_DQF else 4):
                            off = ((arith.ArithValue(row) + _i32(i)) * _i32(RS_OUT)
                                   + col) * _i32(4)
                            val = arith.ArithValue(
                                vector.extract(a, static_position=[i],
                                               dynamic_position=[])) * sc
                            if ABL_DQA:
                                buffer_ops.buffer_store(
                                    arith.unwrap(val), rDQ, off,
                                    offset_is_bytes=True)
                            else:
                                _atomic_fadd(val, rDQ, off, dq_aux)

            if ABL_DQ:
                # dq[m][d] = sum_n ds[m][n] k[n][d].  A wave owns 16*NT of the
                # block's n rows, so its dq partial covers the FULL BM x D tile
                # and is complete for those n after one m iteration -- it is not
                # loop-carried, it is flushed by atomic fadd every iteration.
                # That is 4 partials per workgroup, hence 4 atomic streams.
                assert NSTEP % DQC == 0
                nd = NSTEP // DQC
                for c in range_constexpr(DQC):
                    dq = [zero_acc] * (MSTEP * nd)
                    for ms in range_constexpr(MSTEP):
                        # K = 32 contraction over this wave's n rows.
                        a_dq = vector.bitcast(
                            v8bf,
                            vector.shuffle(ds_f[0][ms], ds_f[NT - 1][ms], cat),
                        )
                        for dt in range_constexpr(nd):
                            # Every d tile must get a DISTINCT B operand or CSE
                            # folds the accumulators together and the probe
                            # understates the pressure it exists to measure.
                            e = c * nd + dt
                            dq[ms * nd + dt] = mfma2(
                                a_dq, kf[(e // KSTEP) % NT][e % KSTEP],
                                dq[ms * nd + dt],
                            )
                    for ms in range_constexpr(MSTEP):
                        for dt in range_constexpr(nd):
                            row = m0 + _i32(16 * ms) + l4 * _i32(4)
                            col = l16 + _i32(16 * (c * nd + dt))
                            for i in range_constexpr(4):
                                off = (
                                    (row + _i32(i)) * _i32(RS_OUT) + col
                                ) * _i32(4)
                                _atomic_fadd(
                                    vector.extract(dq[ms * nd + dt],
                                                   static_position=[i],
                                                   dynamic_position=[]),
                                    rDQ, off,
                                )
            return new_dv, new_dk, cur

        def pipe_score(lds_q8, lds_o8):
            """qk / dp accumulators of one m tile (F8B k128 path of one_iter)."""
            a_qk_frag = [a_frag8(ms, 0, lds_q8) for ms in range_constexpr(MSTEP)]
            a_dp_frag = [a_frag8(ms, 0, lds_o8) for ms in range_constexpr(MSTEP)]
            qk, dp = [], []
            for nt in range_constexpr(NT):
                qk.append([])
                dp.append([])
                for ms in range_constexpr(MSTEP):
                    qk[nt].append(mfma_qk_k128(a_qk_frag[ms], kf8[nt][0],
                                               zero_acc))
                    dp[nt].append(mfma_qk_k128(a_dp_frag[ms], vf8[nt],
                                               zero_acc))
            return qk, dp

        def pipe_pair(mA, acc_dv, acc_dk, masked, eA, eB, prefetch_b=None):
            """One mu2 pair (tiles mA, mA + BM) with both tiles already in LDS:
            tile A in (q8_lds, o8_lds), tile B in (q8b_lds, o8b_lds).

            Same per-element math as one_iter(A, pend=None) followed by
            one_iter(B, pend=A); only the program order moves:
              score(A), score(B) | act(A) cast store | bar | ws(A)
              | act(B) cast | bar | store(B) out-gemm(A+B) | bar | ws(B)
            score(B) shares the first scheduling region with act(A).
            """
            mB = mA + _i32(BM)
            v8i32_t = T.vec(8, T.i32)
            CAT32 = list(range(16))

            def _wide(a, b):
                return vector.bitcast(v8i32_t, vector.shuffle(a, b, CAT32))
            qkA, dpA = pipe_score(q8_lds, o8_lds)
            qkB, dpB = pipe_score(q8b_lds, o8b_lds)
            _, _, silA, dsA, _ = act_tile(mA, qkA, dpA, masked)
            s8A, d8A = f8b_cast(silA, dsA)
            f8b_ds_store(d8A)
            gpu.barrier()
            dqsplit_ws_store(mA)
            nb = prefetch_b() if prefetch_b is not None else None

            def _bop(ds):
                bo = [vector.shuffle(b8_tr(l8, 0, ds), b8_tr(l8, 1, ds),
                                     list(range(8))) for l8 in (o8_lds, o8b_lds)]
                bq = [vector.shuffle(b8_tr(l8, 0, ds), b8_tr(l8, 1, ds),
                                     list(range(8))) for l8 in (q8_lds, q8b_lds)]
                return (_wide(bo[0], bo[1]), _wide(bq[0], bq[1]))
            n_early = {5: NSTEP // 2, 6: NSTEP}.get(PIPE, 0)
            b_ops = [_bop(ds) for ds in range_constexpr(n_early)]
            _, _, silB, dsB, _ = act_tile(mB, qkB, dpB, masked)
            s8B, d8B = f8b_cast(silB, dsB)
            gpu.barrier()  # every wave has read tile A's ds back
            f8b_ds_store(d8B)

            def _cat(p):
                return vector.shuffle(
                    vector.shuffle(p[0], p[1], [0, 1, 2, 3]),
                    vector.shuffle(p[2], p[3], [0, 1, 2, 3]),
                    list(range(8)))
            sc_kb = arith.select(
                arith.cmpi(arith.CmpIPredicate.ult, l4, _i32(2)), eA[0], eB[0])
            a_dv = [_wide(_cat(s8A[nt]), _cat(s8B[nt]))
                    for nt in range_constexpr(NT)]
            a_dk = [_wide(_cat(d8A[nt]), _cat(d8B[nt]))
                    for nt in range_constexpr(NT)]
            new_dv = list(acc_dv)
            new_dk = list(acc_dk)
            n_late = {3: NSTEP // 2, 4: NSTEP}.get(PIPE, 0)
            for j in range_constexpr(NSTEP - n_early):
                b_ops.append(_bop(n_early + j))

            def _out(ds):
                b_o, b_q = b_ops[ds]
                for nt in range_constexpr(NT):
                    i = nt * NSTEP + ds
                    new_dv[i] = mfma_k128_sc(a_dv[nt], b_o, acc_dv[i],
                                             sc127, sc_kb)
                    new_dk[i] = mfma_k128_sc(a_dk[nt], b_q, acc_dk[i],
                                             sc_kb, sc127)
            for ds in range_constexpr(NSTEP - n_late):
                _out(ds)
            gpu.barrier()
            dqsplit_ws_store(mB)
            for j in range_constexpr(n_late):
                _out(NSTEP - n_late + j)
            return new_dv, new_dk, nb

        # ---- m loop ---------------------------------------------------------
        # The causal test m >= n is only non-trivial while the m tile [m0, m0+BM)
        # overlaps the block's n range [n0, n0+BN).  m0 = n0 + BM*it, so that is
        # the first BN/BM iterations and no others.  Peel exactly those, masked,
        # and run the loop from there with no mask at all -- the hand-written
        # assembly does the same thing in .Lpeel.  Peeling past the end of a short
        # sequence is harmless: those rows read back as zero and contribute zero.
        kt_frag = None
        if F8B:
            # K as e4m3, staged in the bf16p interleave but in BYTES: plane
            # p = 2*(d//32) + d%2 holds [BN rows][16 cols], col = (d%32)//2.
            # A thread's 8 consecutive d split into 4 even + 4 odd bytes, one
            # dword store per parity.
            KROWS = BN * D // (THREADS * 8)
            st_n = (tid // _i32(16)) * _i32(KROWS)
            w32 = st_d // _i32(32)
            col0 = (st_d % _i32(32)) // _i32(2)
            v8i8 = T.vec(8, i8_t)
            for r in range_constexpr(KROWS):
                n_row = st_n + _i32(r)
                n_safe = arith.select(
                    arith.cmpi(arith.CmpIPredicate.slt, n0 + n_row, seq_len),
                    n0 + n_row, seq_last,
                )
                raw = vector.bitcast(v8i8, load_raw8(
                    rK, arith.ArithValue(n_safe) * _i32(RS_IN) + st_d))
                for par in range_constexpr(2):
                    half = vector.bitcast(T.vec(1, T.i32), vector.shuffle(
                        raw, raw, [par, 2 + par, 4 + par, 6 + par]))
                    off = (((arith.ArithValue(w32) * _i32(2) + _i32(par))
                            * _i32(BN) + n_row) * _i32(16) + col0) // _i32(4)
                    vector.store(half, kt8_lds, [_idx(off)], alignment=4)
            gpu.barrier()
            # B operand for d tile dt of wave w: column j <-> d = 32w + 2j + dt,
            # k byte b of lane group l4 <-> n = 32*l4 + b.
            kt_frag = []
            for dt in range_constexpr(DQD):
                parts = [tr8(kt8_lds,
                             ((wave * _i32(2) + _i32(dt)) * _i32(BN)
                              + t8_n + _i32(8 * r)) * _i32(16) + t8_k8)
                         for r in range_constexpr(4)]
                kt_frag.append(vector.shuffle(
                    vector.shuffle(parts[0], parts[1], [0, 1, 2, 3]),
                    vector.shuffle(parts[2], parts[3], [0, 1, 2, 3]),
                    list(range(8))))
            gpu.barrier()  # K is consumed; the same bytes become ds below
        elif fused:
            # Stage K once in the tr16 layout and pull dq's B operands into
            # registers.  They are loop-invariant, so this is paid once per
            # workgroup and the LDS is handed straight over to the ds exchange.
            KROWS = BN * D // (THREADS * 8)
            st_n = (tid // _i32(16)) * _i32(KROWS)
            for r in range_constexpr(KROWS):
                n_row = st_n + _i32(r)
                n_safe = arith.select(
                    arith.cmpi(arith.CmpIPredicate.slt, n0 + n_row, seq_len),
                    n0 + n_row, seq_last,
                )
                vec = load_qkv(
                    rK, arith.ArithValue(n_safe) * _i32(RS_IN) + st_d
                )
                if dq16 == "bf16p":
                    # Interleave d across the two 16-wide tiles a wave owns, so
                    # tile 2w holds the EVEN d of [32w, 32w+32) and tile 2w+1 the
                    # odd ones, both at column (d % 32) / 2.  The mfma then hands
                    # lane j the pair (32w + 2j, 32w + 2j + 1) in its two
                    # accumulators, and 16 lanes packing that pair span a whole
                    # 64 B line -- which is the unit the atomic roofline counts.
                    #
                    # A thread loads 8 consecutive d starting at a multiple of 8,
                    # so d // 32 is constant over them and the even and odd
                    # halves each land on 4 CONSECUTIVE columns: two 8-byte
                    # stores, not eight 2-byte ones.
                    w32 = st_d // _i32(32)
                    col0 = (st_d % _i32(32)) // _i32(2)
                    for par in range_constexpr(2):
                        half = vector.from_elements(v4bf, [
                            vector.extract(vec, static_position=[2 * e + par],
                                           dynamic_position=[])
                            for e in range_constexpr(4)
                        ])
                        off = ((arith.ArithValue(w32) * _i32(2) + _i32(par))
                               * _i32(BN) + n_row) * _i32(16) + col0
                        vector.store(half, kt_lds, [_idx(off)], alignment=8)
                    continue
                off = (arith.ArithValue(st_db) * _i32(BN) + n_row) * _i32(16) + st_din
                vector.store(vec, kt_lds, [_idx(off)], alignment=16)
            gpu.barrier()
            # Wave w owns d in [16*DQD*w, +16*DQD).
            dq_row0 = wave * _i32(DQD * BN)
            kt_frag = [[tr_pair(kt_lds, dq_row0 + _i32(dt * BN), t)
                        for t in range(DQK)] for dt in range(DQD)]
            gpu.barrier()  # K is consumed; the same bytes become ds below

        NACC = NT * NSTEP
        n_iter = (seq_len - n0 + _i32(BM - 1)) // _i32(BM)
        acc = [arith.constant_vector(0.0, T.f32x4) for _ in range(2 * NACC)]
        pk_dv, pk_dk = acc[:NACC], acc[NACC:]
        pend = None  # MU2: the first m tile's half-operands, awaiting a partner
        assert not (MU2 and rev), "mu2 pairs m tiles in ascending order only"
        if rev:
            # Descend: every n_block of a (batch, head) starts at the SAME m,
            # the last tile of the sequence, instead of at its own diagonal.
            # Concurrent workgroups then walk the same dq_acc rows in step, so
            # their read-modify-writes collide in cache rather than each
            # streaming its own private stripe.  Accumulation is a sum, so the
            # order is free; the masked peel just moves to the end.
            # A sequence shorter than the peel makes the loop empty, and then
            # `staged` has to already be the first peel tile.
            last = arith.select(
                arith.cmpi(arith.CmpIPredicate.sgt, n_iter, _i32(NPEEL)),
                n_iter, _i32(NPEEL))
            staged = load_tiles(n0 + (arith.ArithValue(last) - _i32(1)) * _i32(BM))
        elif PIPE:
            # Peel = the first pair, masked.  Both tiles staged, one barrier.
            stA = load_tiles(n0)
            stB = load_tiles(n0 + _i32(BM))
            write_tiles(stA)
            write_tiles(stB, (q8b_lds, o8b_lds))
            gpu.barrier()
            nA = load_tiles(n0 + _i32(2 * BM))
            nB = None
            if PIPE == 1:
                nB = load_tiles(n0 + _i32(3 * BM))
            pk_dv, pk_dk, nb = pipe_pair(
                n0, pk_dv, pk_dk, True, stA[8:10], stB[8:10],
                prefetch_b=(None if PIPE == 1 else
                            (lambda: load_tiles(n0 + _i32(3 * BM)))))
            staged = nA + (nB if PIPE == 1 else nb)
        else:
            staged = load_tiles(n0)
            for p in range(NPEEL):
                # Redundant in the fused path: the previous sync (the K-consumed
                # barrier for p==0, the prior one_iter ds-exchange barrier for
                # p>0) already follows the last q_rm/o_rm read.  See the main loop.
                if not fused:
                    gpu.barrier()
                write_tiles(staged)
                gpu.barrier()
                e8_t = staged[8:10] if do_blk else None
                staged = load_tiles(n0 + _i32(BM * (p + 1)))
                pk_dv, pk_dk, pend = one_iter(n0 + _i32(BM * p), pk_dv, pk_dk,
                                              True, pend, e8=e8_t)

        lb = arith.constant(NPEEL, index=True)
        if MU2:
            # Two m tiles per body.  Round the trip count UP to an even number
            # of tiles past the peel: an extra tile reads q/dO past the
            # descriptor's num_records, gets zeros, and contributes exactly zero
            # to dv/dk/dq -- the same argument the existing one-tile prefetch
            # overrun already relies on.
            rem = arith.select(
                arith.cmpi(arith.CmpIPredicate.sgt, n_iter, _i32(NPEEL)),
                arith.ArithValue(n_iter) - _i32(NPEEL), _i32(0))
            ub = arith.index_cast(
                T.index,
                arith.unwrap(_i32(NPEEL)
                             + ((arith.ArithValue(rem) + _i32(1)) // _i32(2))
                             * _i32(2)))
        else:
            ub = arith.index_cast(T.index, n_iter)
        step = arith.constant(2 if MU2 else 1, index=True)
        init = [arith.unwrap(x) for x in pk_dv + pk_dk + staged]
        loop = scf.ForOp(arith.unwrap(lb), arith.unwrap(ub), arith.unwrap(step), init)
        with ir.InsertionPoint(loop.body):
            it = arith.ArithValue(arith.index_cast(T.i32, loop.induction_variable))
            # Ascending induction either way; when reversed it indexes the tiles
            # from the top down, so the trip count and the loop bounds are
            # unchanged and only the address mapping flips.
            m0 = (n0 + (arith.ArithValue(n_iter) + _i32(NPEEL - 1) - it) * _i32(BM)
                  if rev else n0 + it * _i32(BM))
            acc_dv = list(loop.inner_iter_args[:NACC])
            acc_dk = list(loop.inner_iter_args[NACC : 2 * NACC])
            staged = list(loop.inner_iter_args[2 * NACC :])
            # The top-of-loop barrier only needs to keep write_tiles from
            # overwriting q_rm/o_rm before the PREVIOUS iteration's readers of
            # those tiles finish.  In the FUSED path the previous iteration's
            # ds-exchange barrier (one_iter, `if fused`) already fires AFTER the
            # last q_rm/o_rm read (all a_frag score reads and b_frag_tr out-gemm
            # reads precede it) and before write_tiles here, so this barrier is
            # redundant -- ds_lds (off_kt), the only LDS touched between them, is
            # a disjoint allocation from q_rm/o_rm.  Dropping it removes one of
            # three full-workgroup syncs per iteration on a latency-bound body.
            # The non-fused body has no such barrier, so it must keep this one.
            if not fused:
                gpu.barrier()
            if PIPE:
                NST = len(staged) // 2
                stA, stB = staged[:NST], staged[NST:]
                write_tiles(stA)
                write_tiles(stB, (q8b_lds, o8b_lds))
                gpu.barrier()
                nA = load_tiles(m0 + _i32(2 * BM))
                if PIPE == 1:
                    nB = load_tiles(m0 + _i32(3 * BM))
                acc_dv, acc_dk, nb = pipe_pair(
                    m0, acc_dv, acc_dk, False, stA[8:10], stB[8:10],
                    prefetch_b=(None if PIPE == 1 else
                                (lambda: load_tiles(m0 + _i32(3 * BM)))))
                nxt = nA + (nB if PIPE == 1 else nb)
            else:
                write_tiles(staged)
                gpu.barrier()
                # Issue the next tile's global loads before the math so the HBM
                # latency overlaps the mfma stream instead of stalling on it.
                nxt = load_tiles(m0 - _i32(BM) if rev else m0 + _i32(BM))
                if MU2:
                    # First half of the pair: score math + dq for tile m0, and the
                    # K=128 A/B halves parked in registers.  No out-gemm yet.
                    acc_dv, acc_dk, ph = one_iter(
                        m0, acc_dv, acc_dk, False, None,
                        e8=staged[8:10] if do_blk else None)
                    # Re-stage LDS for tile m0+BM.  The barrier pairing is exactly
                    # the one the un-unrolled body uses between consecutive
                    # iterations; the fused ds-exchange barrier inside one_iter
                    # already separates the previous readers of q_rm/o_rm.
                    write_tiles(nxt)
                    gpu.barrier()
                    e8_2 = nxt[8:10] if do_blk else None
                    nxt = load_tiles(m0 + _i32(2 * BM))
                    # Second half: same score math, then ONE K=128 out-gemm per
                    # (n-tile, d-tile) covering both m tiles.
                    acc_dv, acc_dk, _ = one_iter(m0 + _i32(BM), acc_dv, acc_dk,
                                                 False, ph, e8=e8_2)
                else:
                    acc_dv, acc_dk, _ = one_iter(m0, acc_dv, acc_dk, False)
            emit_sched()
            scf.YieldOp([arith.unwrap(x) for x in acc_dv + acc_dk + nxt])

        res_dv = list(loop.results[:NACC])
        res_dk = list(loop.results[NACC : 2 * NACC])
        if rev:
            # The masked tiles are the ones overlapping the n range, i.e. the
            # BOTTOM NPEEL of the m range, so descending reaches them last.
            staged = list(loop.results[2 * NACC :])
            for p in range(NPEEL - 1, -1, -1):
                if not fused:
                    gpu.barrier()
                write_tiles(staged)
                gpu.barrier()
                staged = load_tiles(n0 + _i32(BM * (p - 1)))
                res_dv, res_dk, _ = one_iter(n0 + _i32(BM * p), res_dv, res_dk,
                                             True)

        # ---- epilogue -------------------------------------------------------
        n_out_base = n0 + wave * _i32(16 * NT) + l4 * _i32(4)
        for nt, ds in [(a, b) for a in range(NT) for b in range(NSTEP)]:
            d_idx = l16 + _i32(16 * ds)
            for i in range_constexpr(4):
                n_idx = n_out_base + _i32(16 * nt) + _i32(i)
                ok = arith.cmpi(arith.CmpIPredicate.slt, n_idx, seq_len)
                st_if = scf.IfOp(ok, results_=[], has_else=False)
                with ir.InsertionPoint(st_if.then_block):
                    off = n_idx * _i32(RS_OUT) + d_idx
                    dvv = vector.extract(
                        res_dv[nt * NSTEP + ds], static_position=[i], dynamic_position=[]
                    )
                    dkv = arith.ArithValue(
                        vector.extract(
                            res_dk[nt * NSTEP + ds], static_position=[i], dynamic_position=[]
                        )
                    ) * _f32(alpha)
                    if MU2 == "k128":
                        # Undo the power-of-two pre-scales the fp8 casts folded
                        # in.  Both are exact, so this is a pure exponent shift.
                        dvv = arith.unwrap(
                            arith.ArithValue(dvv)
                            * _f32(1.0 / (mu2_ssc * mu2_osc)))
                        if do_dyn:
                            dvv = arith.unwrap(
                                arith.ArithValue(dvv) * inv_s_do)
                            dkv = arith.ArithValue(dkv) * inv_s_do
                        dkv = arith.ArithValue(dkv) * _f32(
                            1.0 / (mu2_dsc * mu2_qsc))
                    if input_fp8:
                        dkv = arith.ArithValue(dkv) * dk_input_scale
                    buffer_ops.buffer_store(arith.trunc_f(T.bf16, dvv), rDV, off)
                    buffer_ops.buffer_store(arith.trunc_f(T.bf16, dkv), rDK, off)
                    scf.YieldOp([])

    @flyc.kernel
    def bwd_dkdv_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DO: fx.Tensor,
        DK: fx.Tensor,
        DV: fx.Tensor,
        DQ: fx.Tensor,
        SEQ_OFF: fx.Tensor,
        INV_MAX_ARG: fx.Float32,
        QKV_SCALE_ARG: fx.Float32,
        NBATCH: fx.Int32,
        DO_SC: fx.Tensor,
    ):
        def as_i32(v):
            raw = arith.unwrap(v)
            if not isinstance(raw.type, ir.IndexType):
                return arith.ArithValue(raw)
            return arith.ArithValue(arith.index_cast(T.i32, raw))

        tid = as_i32(fx.thread_idx.x)
        # Grid order.  In the dk/dv-only kernel n_block had to be the SLOWEST
        # dimension: under a causal mask a tile's cost grows with its n_block
        # index, so dispatching n_block fastest correlates cost with dispatch
        # position and leaves the CUs draining unevenly (that ordering was
        # worth -28%).  Fusing adds a competing pull -- every n_block of one
        # (batch, head) read-modify-writes the SAME dq_acc rows, so putting
        # them close in dispatch order is what keeps that traffic in L2 -- so
        # the choice is re-opened here rather than assumed.
        blk = [as_i32(fx.block_idx.x), as_i32(fx.block_idx.y),
               as_i32(fx.block_idx.z)]
        # XKIND is a plain int decided at build time.  flydsl's kernel decorator
        # rewrites `if` statements, and it only constant-folds simple
        # comparisons -- a method call like gorder.startswith(...) in the
        # condition makes it treat the branch as device control flow and the
        # Python assignments inside stop escaping.
        if XKIND != 0:
            # Pin every workgroup of one batch to one XCD.  Workgroups are
            # handed to XCDs round-robin on the LINEAR id, id % NXCD, and the
            # linear id is x + gx*(y + gy*z); when gx is a multiple of NXCD the
            # y and z terms drop out and x % NXCD alone picks the XCD.  Putting
            # the batch index there means a batch's dq_acc rows are only ever
            # read-modify-written from one XCD's L2 instead of bouncing between
            # eight.  What shares the x axis decides what varies FASTEST within
            # an XCD, and so how small its live dq_acc footprint is: "xcd" puts
            # heads there (four heads of a batch in flight at once), "xcdn"
            # puts n_blocks there (one head of a batch at a time, a quarter of
            # the footprint).
            slot = blk[0] % _i32(NXCD)
            fast = blk[0] // _i32(NXCD)
            if XKIND == 1:
                head = fast
                n_blk = blk[1]
            else:
                head = blk[1]
                n_blk = fast
            batch = slot + _i32(NXCD) * blk[2]
        else:
            # gorder names what x, y, z carry, fastest axis first.
            batch, head, n_blk = (blk[gorder.index(c)] for c in "bhn")

        # Rounding the batch axis up to a multiple of NXCD dispatches
        # workgroups for batch indices past the end of the batch, and those
        # must read nothing and write nothing.  Two things are needed for that,
        # and the kernel previously had neither:
        #
        # num_records_bytes bounds the descriptor so an out-of-range index
        # returns 0 instead of whatever memory follows seq_offsets.  max_size
        # sets num_records to the maximum, which DISABLES the range check
        # entirely, and passing max_size=False is not enough on its own --
        # flydsl derives the extent from the memref type and gives up when the
        # shape is dynamic, which it always is here because batch varies.
        #
        # The explicit batch < NBATCH test then rejects the workgroup without
        # depending on that reasoning at all.  It is what actually stops the
        # fault: the loads themselves land in the same page and come back as
        # garbage rather than trapping, and it was USING that garbage as a
        # token offset into q/k/v that walked off the allocation.
        #
        # This only ever looked correct because every shape benchmarked before
        # now had a batch that was a multiple of NXCD, so no padded slot ever
        # existed.  At batch=4 it inflated every output by up to 1.9x; at
        # batch=1 it faulted.
        nbatch = arith.ArithValue(arith.unwrap(NBATCH))
        r_off = buffer_ops.create_buffer_resource(
            SEQ_OFF, max_size=False,
            num_records_bytes=arith.unwrap((nbatch + _i32(1)) * _i32(4)),
        )
        seq_start = arith.ArithValue(
            buffer_ops.buffer_load(r_off, batch, vec_width=1, dtype=T.i32)
        )
        seq_end = arith.ArithValue(
            buffer_ops.buffer_load(r_off, batch + _i32(1), vec_width=1, dtype=T.i32)
        )
        seq_len = seq_end - seq_start
        n0 = n_blk * _i32(BN)

        guard = scf.IfOp(
            arith.andi(
                arith.cmpi(arith.CmpIPredicate.slt, n0, seq_len),
                arith.cmpi(arith.CmpIPredicate.slt, batch, nbatch),
            ),
            results_=[], has_else=False,
        )
        with ir.InsertionPoint(guard.then_block):
            emit_body(Q, K, V, DO, DK, DV, DQ,
                      arith.ArithValue(arith.unwrap(INV_MAX_ARG)),
                      arith.ArithValue(arith.unwrap(QKV_SCALE_ARG)),
                      tid, n0, head, seq_start, seq_len, DO_SC,
                      batch_idx=batch, nbatch=nbatch)
            scf.YieldOp([])

    @flyc.jit
    def launch(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DO: fx.Tensor,
        DK: fx.Tensor,
        DV: fx.Tensor,
        DQ: fx.Tensor,
        SEQ_OFF: fx.Tensor,
        batch: fx.Int32,
        n_blocks: fx.Int32,
        inv_max: fx.Float32,
        qkv_scale: fx.Float32 = fx.Float32(1.0),
        do_scale: fx.Tensor = None,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        if gorder == "xcd":
            grid = (NXCD * heads, n_blocks, (batch + NXCD - 1) // NXCD)
        elif gorder == "xcdn":
            grid = (NXCD * n_blocks, heads, (batch + NXCD - 1) // NXCD)
        else:
            extent = {"b": batch, "h": heads, "n": n_blocks}
            grid = tuple(extent[c] for c in gorder)
        bwd_dkdv_kernel(
            Q,
            K,
            V,
            DO,
            DK,
            DV,
            DQ,
            SEQ_OFF,
            inv_max,
            qkv_scale,
            batch,
            do_scale,
        ).launch(
            grid=grid, block=(THREADS, 1, 1), stream=stream,
        )

    # flydsl's jit only knows how to marshal int, torch.Tensor and Stream from
    # plain Python; a bare float raises "No JitArgument registered".  Float32
    # itself satisfies the JitArgument protocol, so callers can keep passing an
    # ordinary float and we box it here rather than leaking fx types into the
    # integration API.
    def call(*args, **kw):
        if scale_in_cvt:
            _require_pow2_cvt_scale(
                kw["qkv_scale"] if "qkv_scale" in kw
                else (args[11] if len(args) > 11 else 1.0)
            )
        if do_dyn:
            ds_t = kw.get("do_scale", args[12] if len(args) > 12 else None)
            assert ds_t is not None and str(ds_t.dtype) == "torch.float32" \
                and ds_t.numel() >= 2 and ds_t.is_cuda, (
                    "do_dyn kernel needs do_scale=[s_do, 1/s_do] fp32 cuda")
        elif do_blk:
            ds_t = kw.get("do_scale", args[12] if len(args) > 12 else None)
            do_t = kw.get("DO", args[3] if len(args) > 3 else None)
            assert ds_t is not None and str(ds_t.dtype) == "torch.uint8" \
                and ds_t.is_cuda and ds_t.is_contiguous() \
                and ds_t.numel() == do_t.shape[0] * heads, (
                    "do_blk kernel needs do_scale = E8 [T, H] uint8 cuda "
                    "(127 - group exponent per token)")
        elif "do_scale" not in kw and len(args) <= 12:
            # Unused by a static-scale kernel; any tensor fills the slot.
            kw["do_scale"] = args[7]
        if "stream" not in kw and len(args) < 14 and current_stream_launch():
            # Launch on the caller's current stream (see _current_stream); an
            # explicit stream, keyword or 14th positional, still wins.  With
            # $DLRM_HSTU_FP8_FLYDSL_CURRENT_STREAM=0 no stream is passed: the
            # previous null-stream launch arguments.
            kw["stream"] = _current_stream(args[0] if args else kw["Q"])
        args = tuple(fx.Float32(a) if isinstance(a, float) else a for a in args)
        if isinstance(kw.get("inv_max"), float):
            kw["inv_max"] = fx.Float32(kw["inv_max"])
        if isinstance(kw.get("qkv_scale"), float):
            kw["qkv_scale"] = fx.Float32(kw["qkv_scale"])
        return launch(*args, **kw)

    # Callers size the grid's n dimension, so they need the block's n extent.
    call.BN = BN
    call.launch = launch
    return call
