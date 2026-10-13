# pyre-strict
"""The preprocessor L1 dgrads fused into the grouped EC backward segment reduce.

DEFAULT OFF. Gate: DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE=1 (1/true/on/yes), read once
per process at first use (apply_env_bootstrap binds it). "0" is the kill switch.
ROCm builds only.

WHAT. The first Linear of ContextualPreprocessor's content MLP (W1_c [256, 512])
and additional MLP (W1_a [256, 1024]) does not write its input gradient
dX = dh @ W1 ([M, 512] / [M, 1024] bf16). Pair 0 returns dW1, db1 and the
SwishLayerNorm gradients as usual and hands dh [M, 256] to the grouped EC
backward (dlrm_v4/train/ec_bf16_output._ECExpandGroup) through a "dh port": a
stride-0 [M, 256] bf16 output of the EC expand, one per hoisted member, consumed
by pair 0 as an extra Function input (dlrm_v4/train/ec_dgrad_ports.py). The EC
backward then runs,
per hoisted member (item; cat = artist|album interleaved, position p = 2t+half),
one pass over its segment-sorted positions (the EC's own (order, offsets)):
  _ecx_dgrad_pass       one program = 128 consecutive sorted positions: gather
                        the dh rows once, compute every token's dX row on bf16
                        MFMA (K = 256 in one dot, rounded to bf16 like the GEMM
                        output), sum the rows of each embedding row in fp32 with
                        a 0/1 indicator MFMA, STORE each segment that closes in
                        the tile (its closing partial if it began earlier) and
                        write the tile's open last segment to a fragment row;
  _ecx_combine          one program per tile; a run head adds the run's
                        fragment rows to the closing partial in fixed tile
                        order (blocks of 8 rows);
  _ecx_zero_uncovered   rows no hoisted member covers = +0.0; rows covered by
                        BOTH members (an invariant violation) = NaN.
The kept keys' scatter_add_ then accumulates into the same fp32 accumulator.
No atomics, no host syncs, 4 launches. Launch config: BM 128, BN 64, S 16,
num_warps 4, num_stages 1, waves_per_eu 2; combine BF 8, 2 warps; ZR 16.

MLPerf: every token's dX row is still computed (M x 256 x Din MACs per MLP, the
unfused dgrad FLOPs, plus the indicator dot); the segment additions are the
unfused reduce's additions, reassociated. Nothing is skipped or approximated;
(sum_segment dh) @ W1 is NOT this design.

NUMERICS: declared fp32 reduction-order class, NOT bit-identical. Addends are
bitwise the single-pass K=256 dot bf16_RNE(fp32 sum_k dh[t,k] W1[k,c]); the
hipBLASLt dX can differ where a stream-K tail tile splits K before rounding
(one bf16 ulp per affected addend). The per-row sums are fp32 in a different
association (tile sums + fixed-order fragment combine instead of 32-row folds +
hot-piece atomics), and deterministic run to run. A non-finite dX element
reaches every segment sum of its 128-position tile and column chunk (0 x inf in
the indicator MFMA); the unfused reduce confines it to its own row. Uncovered
rows are exact +0.0. dh, dW1, db1 and every other gradient are unchanged.

INVARIANTS the passes rely on (they STORE): member row sets are disjoint, and no
cat row has both artist and album positions. Both are structural (separate
tables), are checked on device on the first 2 fused calls per process (one host
sync each; ec_dgrad_ports), and a row covered by both members is NaN-poisoned
by _ecx_zero_uncovered on every call.

FALLBACKS. Before pair 0's backward, every decline gives exactly the unfused dX
(claim/bind/route below). After it, the EC backward rebuilds dX = dh @ W1 with
the unfused GEMM and runs the unfused reduce (ec_dgrad_ports). A raising launch,
a failed invariant check or a failed prebuild latches the fused path off for the process
(latch_off: one "FAILED CLOSED" line). Decline/fallback lines print once per
(site, reason), every rank.

SPECIALIZATION. Triton 3.6 on AMD compiles one variant per class of each int
argument (== 1, % 16) and of each pointer (16-byte alignment, storage within
2 GiB: buffer ops). T and frag_base (pass) and U (zero) change every step:
do_not_specialize. The
dh row stride is a constexpr (SDH): a layout-class property (256 for the
contiguous dh production produces), which keeps constant-stride
address arithmetic and the register budget. Every pointer production passes is a
fresh allocation or a view into one: the members' orders are views of one
joint sort buffer (onesweep: the first member at offset 0, the
second at an even element offset, so still 16-byte aligned). All are
16-byte aligned, storage within 2 GiB (dh [M,256] bf16 reaches 2 GiB at
M = 2^22 rows, above the per-rank range). That is the class prebuild()
compiles, so each pass compiles once per IS_CAT, before RUN_START. The
debug-only EMIT=True variant (emit_dx) adds one variant per DX storage class
(dX [M,1024] / [M,512] bf16 crosses 2 GiB at M = 2^20 / 2^21 rows).

Top-level imports: os, logging, weakref, collections, torch,
triton (guarded: CPU tests import this module without Triton),
ops/triton/preproc_fp8_snr and ops/utils. Nothing from dlrm_v4.train.
"""

import collections
import logging
import os
import weakref
from typing import Any, Dict, Optional, Set, Tuple

import torch
from generative_recommenders.ops.triton import preproc_fp8_snr
from generative_recommenders.ops.utils import arm_print_all_ranks

try:  # CPU-only test environments have no Triton; only the kernels need it.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]

logger: logging.Logger = logging.getLogger(__name__)

ENV_VAR: str = "DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE"
# Debug only: compare the first 2 fused calls with the late-fallback reference.
CHECK_ENV_VAR: str = "DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE_CHECK"
TAG: str = "[ec-dgrad-segreduce]"
ARMED_PREFIX: str = "armed OK: " + TAG
PORT_ATTR: str = "_dlrmv4_ec_dgrad_port"
HID: int = 256  # L1 out features = dh width = K of the fused dot
WIDTH: int = 512  # EC row width

# Fixed launch config of the passes, combine and zero kernels.
_BM, _BN, _S = 128, 64, 16
_PASS_KW: Dict[str, int] = dict(num_warps=4, num_stages=1, waves_per_eu=2)
_COMBINE_BF, _COMBINE_WARPS = 8, 2
_ZR = 16

_ENABLED: Optional[bool] = None
_BROKEN: Optional[str] = None
_ONCE: Set[Tuple[str, str]] = set()
_BOUND_KINDS: Set[str] = set()
_PORTS_LOGGED: bool = False
# name -> Triton CompiledKernel of the last dgrad_segreduce (tests / diagnostics).
LAST_KERNELS: Dict[str, Any] = {}

Member = collections.namedtuple("Member", ["dh", "w1", "idx", "order", "offsets"])


def _say(msg: str) -> None:
    arm_print_all_ranks(msg)


def _once(site: str, reason: str, msg: str) -> None:
    if (site, reason) not in _ONCE:
        _ONCE.add((site, reason))
        _say(msg)


def enabled() -> bool:
    """Gate on, a ROCm build with Triton, and not latched off (read once)."""
    global _ENABLED
    if _ENABLED is None:
        on = os.environ.get(ENV_VAR, "").strip().lower() in ("1", "true", "on", "yes")
        why = None
        if on and torch.version.hip is None:
            why = "not a ROCm/HIP build (verified on gfx950 only)"
        elif on and _KERNELS_ERROR is not None:
            why = f"Triton kernels unavailable ({_KERNELS_ERROR})"
        if why is not None:
            logger.warning("%s %s=1 IGNORED: %s; the shipped dgrad + EC reduce run",
                           TAG, ENV_VAR, why)
            on = False
        _ENABLED = on
    return _ENABLED and _BROKEN is None


def latch_off(where: str, exc: BaseException) -> None:
    """The fused path off for the rest of the process (every later step: unfused path)."""
    global _BROKEN
    if _BROKEN is None:
        _BROKEN = f"{where}: {type(exc).__name__}: {str(exc)[:300]}"
        _say(f"{TAG} FAILED CLOSED -> shipped dgrad + EC reduce for the rest of "
             f"the run ({_BROKEN})")


# ------------------------------------------------------------------ port protocol
class PortRecord:
    """One hoisted member of one EC invocation.

    Strong refs TO a record: the EC ctx (ex2b_records), the member output's (or
    big's) PORT_ATTR tag, pair 0's ctx. A record holds only weakrefs to tensors
    that have a grad_fn, so nothing here forms a reference cycle (the run
    gc.freeze()s once and relies on refcounting). w1 is pair 0's bf16 W1 copy,
    a forward constant without grad_fn that pair 0 saves anyway.
    """

    __slots__ = ("kind", "member", "rows", "din", "out_ref", "watch", "claimed",
                 "state", "w1", "w1_version", "__weakref__")

    def __init__(self, kind: str, member: int, rows: int, din: int) -> None:
        self.kind = kind  # "cat" | "item"
        self.member = member  # position among the EC call's idxs
        self.rows = rows  # M: pair 0's x rows
        self.din = din  # 1024 (cat) | 512 (item)
        self.out_ref: Optional[weakref.ref] = None  # the tensor pair 0 must get as x
        self.watch: Tuple[weakref.ref, ...] = ()  # hooks/retain_grad here -> decline
        self.claimed = False
        self.state = "new"  # new | bound | routed | declined
        self.w1: Optional[torch.Tensor] = None
        self.w1_version = -1


def _graph_task_id() -> int:
    f = getattr(torch._C, "_current_graph_task_id", None)
    return -1 if f is None else int(f())  # missing API: check skipped


def tracing(x: Optional[torch.Tensor] = None) -> bool:
    from torch.fx._symbolic_trace import is_fx_tracing

    return (
        is_fx_tracing()
        or torch.compiler.is_compiling()
        or (x is not None and isinstance(x, torch.fx.Proxy))
    )


def _watched_has_hooks(rec: PortRecord) -> bool:
    for ref in rec.watch:
        t = ref()
        # Tensor._backward_hooks: register_hook's dict (private; absent -> no hooks seen)
        if t is not None and (getattr(t, "_backward_hooks", None) or t.retains_grad):
            return True
    return False


def claim(
    x: torch.Tensor, lin_w: torch.Tensor
) -> Tuple[Optional[PortRecord], Optional[torch.Tensor]]:
    """Pair-0 call site, before apply: (record, port) when x is a tagged EC
    member output this pair 0 may take dh for, else (None, None)."""
    if not enabled() or tracing(x):
        return None, None
    tag = getattr(x, PORT_ATTR, None)
    if tag is None:
        return None, None
    rec, port = tag
    why = None
    if rec.claimed:
        why = "second consumer of the member output (e.g. a grouped-tail re-run)"
    elif rec.out_ref is None or rec.out_ref() is not x:
        why = "x is not the tagged member output"
    elif not torch.is_grad_enabled():
        why = "grad mode off"
    elif _graph_task_id() != -1:
        why = "inside a backward (reentrant checkpoint recompute)"
    elif preproc_fp8_snr.ENABLED:
        why = "the FP8 numerics probe wants dX"
    elif _watched_has_hooks(rec):
        why = "hooks or retain_grad on the member output"
    elif tuple(lin_w.shape) != (HID, rec.din) or tuple(x.shape) != (rec.rows, rec.din):
        why = f"shapes x={tuple(x.shape)} W1={tuple(lin_w.shape)}"
    if why is not None:
        _once("claim", why, f"{TAG} DECLINED at claim ({rec.kind}): {why} -> shipped dX")
        return None, None
    rec.claimed = True
    return rec, port


def bind(
    rec: Optional[PortRecord], x_c: torch.Tensor, lin_w_c: torch.Tensor
) -> Optional[PortRecord]:
    """Inside pair 0's forward, after the autocast casts: hand W1 over.
    Returns the record for ctx.ex2b, or None (port then gets None)."""
    if rec is None:
        return None
    why = None
    if rec.state != "new" or not rec.claimed:
        why = f"record state {rec.state}"
    elif rec.out_ref is None or rec.out_ref() is not x_c:
        why = "x changed by the autocast cast / contiguous"
    elif (
        lin_w_c.dtype != torch.bfloat16
        or not lin_w_c.is_contiguous()
        or tuple(lin_w_c.shape) != (HID, rec.din)
    ):
        why = f"W1 {lin_w_c.dtype} {tuple(lin_w_c.shape)}"
    if why is not None:
        rec.state = "declined"
        _once("bind", why, f"{TAG} DECLINED at bind ({rec.kind}): {why} -> shipped dX")
        return None
    rec.w1 = lin_w_c
    rec.w1_version = lin_w_c._version
    rec.state = "bound"
    global _PORTS_LOGGED
    _BOUND_KINDS.add(rec.kind)
    if not _PORTS_LOGGED and len(_BOUND_KINDS) == 2:
        _PORTS_LOGGED = True
        _say(f"{ARMED_PREFIX} ports -- content/additional L1 dX leaves as dh "
             f"[M,{HID}] through EC ports (W1 [{HID},{WIDTH}] / [{HID},{2 * WIDTH}]; "
             f"kill switch {ENV_VAR}=0)")
    return rec


def _ec_will_run(ec_node: Any) -> bool:
    """Whether this backward EXECUTES the EC node. The EC
    node alone is not enough: under torch.autograd.grad(loss, [member output])
    it is only captured, yet _will_engine_execute_node reports True for it. It
    executes iff its embeddings edge does (its only differentiable input), so
    both are asked; a leaf capture raises in that API -> False (decline)."""
    f = getattr(torch._C, "_will_engine_execute_node", None)
    if f is None:
        return True  # API missing (documented unsupported: autograd.grad on x)
    if ec_node is None:
        return False
    try:
        nf = ec_node.next_functions
        emb_node = nf[0][0] if nf else None
        return emb_node is not None and bool(f(ec_node)) and bool(f(emb_node))
    except Exception:
        return False


def route(ctx: Any, rec: PortRecord, dh: torch.Tensor, port_idx: int) -> bool:
    """Pair 0's backward, after the LN backward: True -> return dx=None and
    dh for the port; False -> dx = dh @ W1 exactly as unfused."""
    why = None
    if rec.state not in ("bound", "routed"):
        why = f"record state {rec.state}"
    elif not enabled():
        why = "latched off"
    elif torch.is_grad_enabled():
        why = "create_graph backward"
    elif not ctx.needs_input_grad[port_idx]:
        why = "port needs no grad"
    elif (
        dh.dtype != torch.bfloat16
        or dh.dim() != 2
        or tuple(dh.shape) != (rec.rows, HID)
        or dh.stride(1) != 1
    ):
        why = f"dh {dh.dtype} {tuple(dh.shape)} stride {tuple(dh.stride())}"
    elif rec.w1 is None or rec.w1._version != rec.w1_version:
        why = "W1 copy modified in place"
    elif _watched_has_hooks(rec):
        why = "hooks or retain_grad registered after the forward"
    # next_functions has entries for TENSOR inputs only (not index-aligned with
    # needs_input_grad); the port is pair 0's last argument and a tensor here.
    elif not _ec_will_run(ctx.next_functions[-1][0]):
        why = "the EC backward does not run in this backward (autograd.grad / inputs=)"
    if why is not None:
        _once("route", why, f"{TAG} DECLINED at route ({rec.kind}): {why} -> shipped dX")
        return False
    rec.state = "routed"
    return True


# ------------------------------------------------------------------ kernels
# Defined only with Triton. A decoration-time error disables the fused path (enabled()
# names it) instead of breaking the imports of fused_linear_norm & co.
_KERNELS_ERROR: Optional[str] = "Triton is not importable" if triton is None else None
if triton is not None:
    try:

        @triton.jit
        def _ecx_tile(tile, frow, DH, W, IDX, ORDER, OFF, ACC, FRAG_U, FRAG_N, FRAG_V, T, DX,
                      IS_CAT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, KD: tl.constexpr,
                      NOUT: tl.constexpr, SW: tl.constexpr, S: tl.constexpr, SDH: tl.constexpr,
                      EMIT: tl.constexpr):
            """One tile of BM sorted positions (dh row stride SDH): segments closing in the tile are
            STOREd to acc[u] (a segment that began in an earlier tile stores its
            closing partial); the tile's last segment, if open at the tile end, is
            written to fragment row frow (combine adds it later, in tile order)."""
            s0 = tile * BM
            rows = tl.arange(0, BM)
            kk = tl.arange(0, KD)
            cols = tl.arange(0, BN)
            slots = tl.arange(0, S)
            s = s0 + rows
            valid = s < T
            tile_end = tl.minimum(s0 + BM, T)
            o = tl.load(ORDER + s, mask=valid, other=0)
            if IS_CAT:
                t = (o >> 1).to(tl.int32)
                half = (o & 1).to(tl.int32)
                hmin = tl.min(tl.where(valid, half, 1), axis=0)
                hmax = tl.max(tl.where(valid, half, 0), axis=0)
                nh = hmax - hmin + 1
            else:
                t = o.to(tl.int32)
                nh = 1
            A = tl.load(DH + t[:, None] * SDH + kk[None, :], mask=valid[:, None], other=0.0)
            u = tl.load(IDX + o, mask=valid, other=0).to(tl.int32)
            lo = tl.load(OFF + u, mask=valid, other=0).to(tl.int32)
            nseg = tl.sum((valid & ((rows == 0) | (lo == s))).to(tl.int32), axis=0)
            # the tile's last segment: fragment bookkeeping (scalars)
            o_l = tl.load(ORDER + (tile_end - 1))
            u_l = tl.load(IDX + o_l).to(tl.int32)
            lo_l = tl.load(OFF + u_l).to(tl.int32)
            hi_l = tl.load(OFF + u_l + 1).to(tl.int32)
            last_open = hi_l > tile_end
            tl.store(FRAG_U + frow, tl.where(last_open, u_l, -1))
            # run head (segment began in this tile and is open): fragment rows of the run
            tl.store(FRAG_N + frow, tl.where(last_open & (lo_l >= s0), (hi_l - 1) // BM - tile, 0))
            fptr = FRAG_V + frow.to(tl.int64) * NOUT + tl.zeros([S, 1], tl.int64)
            # > S segments or both W1_a halves in the tile: extra passes of the same
            # code (next S slots / other half), each re-running the tile GEMM.
            npass = tl.cdiv(nseg, S) * nh
            for pz in range(npass):
                g = pz // nh
                # per-pass metadata reloaded (L1/L2 hits): nothing extra live across the GEMM
                o2 = tl.load(ORDER + s, mask=valid, other=0)
                u2 = tl.load(IDX + o2, mask=valid, other=0).to(tl.int32)
                lo2 = tl.load(OFF + u2, mask=valid, other=0).to(tl.int32)
                hi2 = tl.load(OFF + u2 + 1, mask=valid, other=0).to(tl.int32)
                kr = tl.cumsum((valid & ((rows == 0) | (lo2 == s))).to(tl.int32), axis=0) - 1
                if IS_CAT:
                    h = hmin + pz % nh
                    rmask = valid & ((o2 & 1).to(tl.int32) == h)
                    hoff = h * NOUT
                else:
                    rmask = valid
                    hoff = 0
                sel = (kr[None, :] == (g * S + slots)[:, None]) & rmask[None, :]
                P = sel.to(tl.bfloat16)
                su = tl.max(tl.where(sel, u2[None, :], -1), axis=1)
                shi = tl.max(tl.where(sel, hi2[None, :], -1), axis=1)
                act = su >= 0
                closed = act & (shi <= tile_end)
                openm = act & (shi > tile_end)
                arow = su.to(tl.int64) * NOUT
                for n in range(NOUT // BN):
                    cb = n * BN
                    b = tl.load(W + kk[:, None] * SW + (hoff + cb + cols)[None, :])
                    xb = tl.dot(A, b).to(tl.bfloat16)  # the GEMM's output rounding (RNE)
                    if EMIT:
                        tl.store(DX + t.to(tl.int64)[:, None] * SW + (hoff + cb + cols)[None, :], xb,
                                 mask=rmask[:, None])
                    val = tl.dot(P, xb)  # 0/1 x bf16 products exact; fp32 accumulation
                    tl.store(ACC + arow[:, None] + (cb + cols)[None, :], val, mask=closed[:, None])
                    if last_open:
                        tl.store(fptr + (cb + cols)[None, :], val, mask=openm[:, None])

        @triton.jit(do_not_specialize=["T", "frag_base"])
        def _ecx_dgrad_pass(DH, W, IDX, ORDER, OFF, ACC, FRAG_U, FRAG_N, FRAG_V, T, frag_base, DX,
                            IS_CAT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                            KD: tl.constexpr, NOUT: tl.constexpr, SW: tl.constexpr,
                            S: tl.constexpr, SDH: tl.constexpr, EMIT: tl.constexpr):
            tile = tl.program_id(0)
            _ecx_tile(tile, frag_base + tile, DH, W, IDX, ORDER, OFF, ACC, FRAG_U, FRAG_N, FRAG_V,
                      T, DX, IS_CAT, BM, BN, KD, NOUT, SW, S, SDH, EMIT)

        @triton.jit
        def _ecx_combine(FRAG_U, FRAG_N, FRAG_V, ACC, NOUT: tl.constexpr, BF: tl.constexpr):
            """Run head a (FRAG_N[a] = n > 0): acc[u] = closing partial (already in
            acc) + fragment rows a .. a+n-1 in fixed order (blocks of BF rows,
            tl.sum per block, block sums sequential). One writer per u."""
            a = tl.program_id(0)
            n = tl.load(FRAG_N + a)
            if n > 0:
                u = tl.load(FRAG_U + a)
                cols = tl.arange(0, NOUT)
                aptr = ACC + u.to(tl.int64) * NOUT + cols
                acc = tl.load(aptr)
                for j0 in range(0, n, BF):
                    jj = j0 + tl.arange(0, BF)
                    v = tl.load(FRAG_V + (a + jj).to(tl.int64)[:, None] * NOUT + cols[None, :],
                                mask=(jj < n)[:, None], other=0.0)
                    acc += tl.sum(v, axis=0)
                tl.store(aptr, acc)

        @triton.jit(do_not_specialize=["U"])
        def _ecx_zero_uncovered(ACC, OFFA, OFFB, U, NOUT: tl.constexpr, ZR: tl.constexpr):
            """acc[u] = +0.0 where no member covers u; NaN where BOTH members do
            (the passes STORE, so overlap must not train silently). Launched last; touches no row a pass writes legitimately."""
            rows = tl.program_id(0) * ZR + tl.arange(0, ZR)
            m = rows < U
            la = tl.load(OFFA + rows + 1, mask=m, other=0) - tl.load(OFFA + rows, mask=m, other=0)
            lb = tl.load(OFFB + rows + 1, mask=m, other=0) - tl.load(OFFB + rows, mask=m, other=0)
            z = m & (la == 0) & (lb == 0)
            both = m & (la > 0) & (lb > 0)
            w = z | both
            if tl.max(w.to(tl.int32), axis=0) > 0:
                cols = tl.arange(0, NOUT)
                nan = tl.full([ZR, NOUT], 0x7FC00000, tl.int32).to(tl.float32, bitcast=True)
                val = tl.where(both[:, None], nan, tl.zeros([ZR, NOUT], tl.float32))
                tl.store(ACC + rows.to(tl.int64)[:, None] * NOUT + cols[None, :], val,
                         mask=w[:, None])
    except Exception as exc:  # pragma: no cover
        _KERNELS_ERROR = f"{type(exc).__name__}: {str(exc)[:200]}"


def _check_member(m: Member, kind: str, dev: torch.device, num_rows: int) -> None:
    din = 2 * WIDTH if kind == "cat" else WIDTH
    T = m.idx.numel()
    rows = T // 2 if kind == "cat" else T
    if kind == "cat" and T % 2:
        raise ValueError(f"cat member has an odd position count {T}")
    dh, w1 = m.dh, m.w1
    if dh.dtype != torch.bfloat16 or dh.dim() != 2 or dh.stride(1) != 1:
        raise ValueError(f"{kind} dh must be 2-D bf16 with unit column stride")
    if tuple(dh.shape) != (rows, HID):
        raise ValueError(f"{kind} dh {tuple(dh.shape)} != ({rows}, {HID})")
    if w1.dtype != torch.bfloat16 or not w1.is_contiguous() or tuple(w1.shape) != (HID, din):
        raise ValueError(f"{kind} W1 must be contiguous bf16 ({HID}, {din})")
    for name, t in (("idx", m.idx), ("order", m.order), ("offsets", m.offsets)):
        if t.dtype != torch.int64 or t.dim() != 1 or not t.is_contiguous():
            raise ValueError(f"{kind} {name} must be contiguous 1-D int64")
    if m.order.numel() != T or m.offsets.numel() != num_rows + 1:
        raise ValueError(f"{kind} order/offsets sizes do not match T={T}, U={num_rows}")
    if T >= 2**31 or max(rows, 1) * dh.stride(0) >= 2**31:
        raise ValueError(f"{kind} too large for int32 positions / dh offsets")
    for t in (dh, w1, m.idx, m.order, m.offsets):
        if t.device != dev:
            raise ValueError(f"{kind} tensors on different devices")


def dgrad_segreduce(
    cat: Member,
    item: Member,
    num_rows: int,
    emit: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """acc fp32 [U, 512]: per row, the sum over both hoisted members of the bf16
    dX rows of its positions (+0.0 for rows neither covers). No host sync.
    cat: Member(dh_a [Mc,256], W1_a [256,1024], idx [2Mc] interleaved,
    order [2Mc], offsets [U+1]); item: Member(dh_c [Mi,256], W1_c [256,512],
    idx [Mi], order, offsets). emit=(dX_a [Mc,1024], dX_c [Mi,512]) bf16 also
    receives every addend by token (debug/tests only). Raises ValueError on an
    unsupported input (the caller fails closed)."""
    if _KERNELS_ERROR is not None:
        raise RuntimeError(f"Triton kernels unavailable ({_KERNELS_ERROR})")
    dev = cat.dh.device
    if dev.type != "cuda" or not isinstance(num_rows, int) or num_rows < 0:
        raise ValueError("CUDA tensors and a nonnegative int num_rows required")
    _check_member(cat, "cat", dev, num_rows)
    _check_member(item, "item", dev, num_rows)
    U = num_rows
    acc = torch.empty((U, WIDTH), dtype=torch.float32, device=dev)
    if U == 0:
        return acc
    Ta, Tc = cat.idx.numel(), item.idx.numel()
    nA, nC = triton.cdiv(Ta, _BM), triton.cdiv(Tc, _BM)
    nt = nA + nC
    frag_u = torch.empty((nt,), dtype=torch.int32, device=dev)
    frag_n = torch.empty((nt,), dtype=torch.int32, device=dev)
    frag_v = torch.empty((nt, WIDTH), dtype=torch.float32, device=dev)
    em = emit is not None
    dxa, dxc = emit if em else (acc, acc)
    ks: Dict[str, Any] = {}
    if Ta:
        ks["cat"] = _ecx_dgrad_pass[(nA,)](
            cat.dh, cat.w1, cat.idx, cat.order, cat.offsets, acc, frag_u, frag_n, frag_v,
            Ta, 0, dxa, True, _BM, _BN, HID, WIDTH, 2 * WIDTH, _S, cat.dh.stride(0), em,
            **_PASS_KW,
        )
    if Tc:
        ks["item"] = _ecx_dgrad_pass[(nC,)](
            item.dh, item.w1, item.idx, item.order, item.offsets, acc, frag_u, frag_n,
            frag_v, Tc, nA, dxc, False, _BM, _BN, HID, WIDTH, WIDTH, _S, item.dh.stride(0),
            em, **_PASS_KW,
        )
    if nt:
        ks["combine"] = _ecx_combine[(nt,)](
            frag_u, frag_n, frag_v, acc, WIDTH, _COMBINE_BF, num_warps=_COMBINE_WARPS
        )
    ks["zero"] = _ecx_zero_uncovered[(triton.cdiv(U, _ZR),)](
        acc, cat.offsets, item.offsets, U, WIDTH, _ZR, num_warps=4
    )
    LAST_KERNELS.clear()
    LAST_KERNELS.update(ks)
    return acc


def emit_dx(cat: Member, item: Member, num_rows: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Debug/tests: the kernel's own bf16 addends by token, (dX_a [Mc,1024],
    dX_c [Mi,512])."""
    dxa = torch.zeros((cat.dh.shape[0], 2 * WIDTH), dtype=torch.bfloat16, device=cat.dh.device)
    dxc = torch.zeros((item.dh.shape[0], WIDTH), dtype=torch.bfloat16, device=item.dh.device)
    dgrad_segreduce(cat, item, num_rows, emit=(dxa, dxc))
    return dxa, dxc


def check_invariants(cat: Member, item: Member, num_rows: int) -> Optional[str]:
    """None, or why the passes' STORE preconditions fail. One host sync."""
    del num_rows
    oc, oi = cat.offsets, item.offsets
    overlap = ((oc[1:] > oc[:-1]) & (oi[1:] > oi[:-1])).sum()
    if cat.order.numel() > 1:
        par = cat.order & 1
        u_s = cat.idx[cat.order]
        mixed = ((par[1:] != par[:-1]) & (u_s[1:] == u_s[:-1])).sum()
        del par, u_s
    else:
        mixed = torch.zeros_like(overlap)
    n_overlap, n_mixed = (int(v) for v in torch.stack([overlap, mixed]).tolist())
    if n_overlap or n_mixed:
        return (f"INVARIANT violated: {n_overlap} rows covered by both members, "
                f"{n_mixed} cat segments with artist and album positions")
    return None


def _synthetic(device: torch.device) -> Tuple[Member, Member, int]:
    """Prebuild input: no RNG, integer-valued dh/W1 in {-1, 0, 1} so every
    addend (|dot| <= 256) and every fp32 sum is exact in any order. Cat Mc=500
    (one mixed-half tile at sorted position 500), item Mi=512; U=96 with
    disjoint pools (item 0-39, artist 40-63, album 64-87, kept-only 88-95);
    a hot row per member spanning 3 tiles; a tile with > 16 segments."""
    Mc, Mi, U = 500, 512, 96
    t = torch.arange(Mi, device=device)
    item_idx = torch.where(t < 300, torch.zeros_like(t), 1 + (t * 7) % 39)
    tc = torch.arange(Mc, device=device)
    artist = torch.where(tc < 400, torch.full_like(tc, 40), 41 + (tc * 5) % 23)
    album = 64 + (tc * 11) % 24
    cat_idx = torch.stack([artist, album], dim=1).reshape(-1)

    def tern(r: int, c: int, a: int, b: int) -> torch.Tensor:
        i = torch.arange(r, device=device)[:, None]
        j = torch.arange(c, device=device)[None, :]
        return ((i * a + j * b) % 3 - 1).to(torch.bfloat16).contiguous()

    def segs(idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        order = torch.argsort(idx, stable=True)
        off = torch.zeros(U + 1, dtype=torch.int64, device=device)
        off[1:] = torch.cumsum(torch.bincount(idx, minlength=U), 0)
        return order, off

    cat = Member(tern(Mc, HID, 13, 7), tern(HID, 2 * WIDTH, 5, 11), cat_idx, *segs(cat_idx))
    item = Member(tern(Mi, HID, 3, 17), tern(HID, WIDTH, 19, 2), item_idx, *segs(item_idx))
    return cat, item, U


def _reference(cat: Member, item: Member, U: int) -> torch.Tensor:
    # fp64 on the CPU: exact for the integer-valued prebuild input, and it keeps
    # device GEMMs (and TunableOp tuning / record-untuned, when on) out of the
    # pre-RUN_START prebuild. Returned on the members' device.
    dev = cat.dh.device
    acc = torch.zeros((U, WIDTH), dtype=torch.float64)
    dxa = (cat.dh.cpu().double() @ cat.w1.cpu().double()).view(-1, WIDTH)
    acc.index_add_(0, cat.idx.cpu(), dxa)
    acc.index_add_(0, item.idx.cpu(), item.dh.cpu().double() @ item.w1.cpu().double())
    return acc.float().to(dev)


def prebuild(device: torch.device) -> str:
    """Before RUN_START (blas_warmup): compile both pass specializations, the
    combine and the zero kernel, and check the result EXACTLY against an
    fp64 reference on a synthetic exact-arithmetic input. Raises on mismatch
    (the caller latches the fused path off). Its tensors are fresh allocations, i.e.
    the production specialization class (SPECIALIZATION above), so the first
    fused backward does not compile."""
    cat, item, U = _synthetic(device)
    if check_invariants(cat, item, U) is not None:
        raise RuntimeError("prebuild input violates the member invariants")
    acc = dgrad_segreduce(cat, item, U)
    ref = _reference(cat, item, U)
    torch.cuda.synchronize(device)
    bad = int((acc.view(torch.int32) != ref.view(torch.int32)).sum().item())
    if bad:
        raise RuntimeError(f"prebuild result differs from the exact reference in {bad} elements")
    msg = (f"{TAG} pre-run_start prebuild OK (_ecx_dgrad_pass cat+item, _ecx_combine, "
           f"_ecx_zero_uncovered; exact check {U}x{WIDTH} bitwise, device={device})")
    _say(msg)
    return msg
