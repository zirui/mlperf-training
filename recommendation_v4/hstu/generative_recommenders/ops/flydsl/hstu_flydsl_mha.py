# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""HSTU jagged attention with FlyDSL forward and backward kernels, each optional.

THE TWO HALVES ARE SELECTED INDEPENDENTLY, because HSTU has no softmax: the
forward hands the backward no lse and no rowsum(dO*O), so nothing couples them.
$HSTU_ATTN_BWD defaults to flydsl. $HSTU_ATTN_FWD defaults to **flydsl** in gin
(FlyDSL e4m3 forward + fused-silu when DLRM_HSTU_FP8_FWD is on).
$HSTU_ATTN_FWD=triton is the opt-out. The bf16 FlyDSL forward is the same knob
with FP8 off. `$HSTU_ATTN_FWD_DUMP_LENS` below dumps the forward lengths.

`hstu_mha` mirrors the signature of
`generative_recommenders.ops.triton.triton_hstu_attention.triton_hstu_mha`, so
it is a drop-in at the call site. Selection lives in `ops/hstu_attention.py`
behind $HSTU_ATTN_BWD, which **defaults to flydsl** (gin:
`apply_env_bootstrap.HSTU_ATTN_BWD`); set it to "triton" to opt out.

TWO KINDS OF "CAN'T RUN", DELIBERATELY HANDLED DIFFERENTLY.

  * The HOST cannot run it -- not a gfx950 GPU, or flydsl is not installed.
    That is a property of the whole process. FlyDSL is the default, so a
    missing install raises `FlyDSLUnavailableError` on first use
    rather than degrading. Opt out explicitly with $HSTU_ATTN_BWD=triton.
  * The SHAPE cannot be served -- see SPECIALIZATIONS below. That is per call
    and legitimately varies across configs and tests, so it falls back to the
    Triton backward. The first fallback of each distinct reason is logged, so
    it is quiet but not silent.

SPECIALIZATIONS.  The FlyDSL kernel is not general. It runs only when ALL of the
following hold, and otherwise this falls back to the Triton backward -- a shape
it cannot serve is slower, never wrong:

  * dim == 128           the mfma fragment layout is written around it
  * BM == 64             likewise; asserted inside the kernel builder
  * bf16 q, k, v, dout   the only dtype with an mfma path here
  * causal, max_attn_len == 0, contextual_seq_len == 0
  * num_targets is None, or every entry is <= 1

  HSTU's target-aware masking clamps positions to max_ids = seq_len - n_targets
  and then applies the ordinary >= mask. At num_targets = 1 the clamp is the
  identity, so plain causal is the same function. At num_targets > 1 it is not.
  yambda-5b sets hstu_config.max_num_candidates = 1. A config that raises it
  falls back rather than computing the wrong mask.

  `heads` is a compile-time constant (it keys the kernel cache) -- not a
  correctness limit, just a fresh JIT per distinct head count. max_seq_len is NOT
  compile-time; it is a runtime scalar, which matters because production derives
  it per batch as max_uih_len + num_candidates.

NON-DETERMINISM.  dq is accumulated with global atomics, so the order of the
float adds varies run to run and dq is not bitwise reproducible. dk and dv are
deterministic. A caller that needs bitwise reproducibility must use the Triton
path.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional, Tuple

import numpy as np
import torch

from generative_recommenders.ops.flydsl import (
    DEFAULT_BWD_BACKEND,
    DEFAULT_FWD_BACKEND,
)
from generative_recommenders.ops.triton.triton_hstu_attention import (
    triton_hstu_attention_bwd,
    triton_hstu_attention_fwd,
)

logger: logging.Logger = logging.getLogger(__name__)

SUPPORTED_DIM = 128
FP8_E4M3 = getattr(torch, "float8_e4m3fn", None)
# The kernel emits gfx950 mfma/ds_read_b64_tr_b16; it is not portable to another
# arch, and there is no meaningful "close enough" here.
SUPPORTED_ARCH = "gfx950"

# Import state for the flydsl kernel, resolved on first backward. Kept lazy so a
# trainer built without flydsl can still import this module (to report why it
# cannot be used, or to run with $HSTU_ATTN_BWD=triton).
_kernel = None
_kernel_error: Optional[BaseException] = None
_backward_calls: int = 0
_logged_fallbacks: set = set()
_logged_layouts: set = set()
_host_ok: bool = False

# The forward kernel is a separate module with its own import latch, so a tree
# that has the backward but not the forward degrades to the Triton forward rather
# than failing the whole attention path.
_fwd_kernel = None
_fwd_kernel_error: Optional[BaseException] = None
_forward_calls: int = 0
_logged_fwd_fallbacks: set = set()
_logged_fwd_layouts: set = set()
_fwd_host_ok: bool = False
# The e4m3 forward keeps its own counters: it is a different kernel behind a
# different knob, and mixing the two would make $HSTU_ATTN_FWD_CHECK's period
# depend on which arm is running.
_fp8_forward_calls: int = 0
_logged_fp8_fwd_layouts: set = set()
_e1_bwd_census: dict = {}
_e1_bwd_calls: int = 0


def _maybe_dump_lengths(seq_offsets: torch.Tensor) -> None:
    """Save one batch's sequence lengths to $HSTU_ATTN_FWD_DUMP_LENS, once.

    Off unless the variable is set. Writes once per process.
    """
    path = os.environ.get("HSTU_ATTN_FWD_DUMP_LENS", "")
    if not path:
        return
    try:
        lens = (seq_offsets[1:] - seq_offsets[:-1]).to("cpu").numpy()
        # Rank-tagged: every rank sees a different slice of the global batch, and
        # one rank's 1024 sequences is a thin sample of the length distribution.
        out = f"{path}.rank{torch.distributed.get_rank()}.npy" \
            if torch.distributed.is_available() and torch.distributed.is_initialized() \
            else f"{path}.npy"
        np.save(out, lens)
        logger.warning(
            "[hstu-attn] dumped %d sequence lengths (mean %.0f max %d) to %s",
            lens.size, float(lens.mean()), int(lens.max()), out,
        )
    except Exception as exc:  # a benchmarking aid must never break a train step
        logger.warning("[hstu-attn] length dump to %s failed: %r", path, exc)

# Memo for the num_targets support check below: (tensor, version, max).
# A strong reference to the tensor is deliberate -- it is one int per row -- and
# is what makes the `is` test sound, since the storage cannot be recycled under
# a live reference.
_targets_max_memo: Tuple[Optional[torch.Tensor], int, int] = (None, -1, 0)


def _max_num_targets(num_targets: torch.Tensor) -> int:
    """`num_targets.max()`, computed once per batch rather than once per layer.

    Every HSTU layer's backward is handed the same num_targets tensor and asks
    the same question of it, but `.item()` is a device-to-host sync: it drains
    the launch queue, and in the backward the queue is at its deepest. Profiled
    at 31.7 ms per call against 0.6 ms for the same call in the forward, so the
    cost is the wait for queued work, not the 4-byte copy. Memoizing collapses
    the per-step cost from one sync per layer to one.

    Keyed on tensor identity plus `_version` so that a fresh batch or any
    in-place write misses and re-reads.
    """
    global _targets_max_memo
    cached, version, value = _targets_max_memo
    if cached is num_targets and version == num_targets._version:
        return value
    value = int(num_targets.max().item())
    _targets_max_memo = (num_targets, num_targets._version, value)
    return value


def seed_max_num_targets(num_targets: torch.Tensor, value: int) -> None:
    """Record ``num_targets.max() == value`` read by the caller's own sync.

    DlrmHSTU.preprocess reads this max (as max_num_candidates) together with
    its other per-batch length stats in a single device-to-host copy; seeding
    the memo here lets the first _max_num_targets call of the batch skip its
    own `.item()`. Same identity/_version key, so any other tensor, or an
    in-place write to this one, still misses and re-reads.
    """
    global _targets_max_memo
    _targets_max_memo = (num_targets, num_targets._version, int(value))


class FlyDSLUnavailableError(RuntimeError):
    """The FlyDSL backward was selected but this host cannot run it.

    Raised rather than falling back because flydsl is the default: a silent
    downgrade to Triton is a ~2x slower backward that no log line would show.
    """


def _device_arch() -> str:
    """gfx arch of the current device, or "" if it is not a ROCm GPU."""
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
    except BaseException:  # noqa: BLE001 - no device, or a driver that cannot say
        return ""
    # ROCm exposes gcnArchName ("gfx950:sramecc+:xnack-"); CUDA has no such field.
    return str(getattr(props, "gcnArchName", "") or "")


def unsupported_hardware_reason() -> Optional[str]:
    """Why this host cannot run the kernel, or None if it can."""
    if not torch.cuda.is_available():
        return "no GPU is visible to torch"
    arch = _device_arch()
    if not arch:
        return (
            "this is not a ROCm device (no gcnArchName); the kernel is "
            f"{SUPPORTED_ARCH} assembly"
        )
    if not arch.startswith(SUPPORTED_ARCH):
        return f"device arch is {arch!r}, the kernel is built for {SUPPORTED_ARCH}"
    return None


def _load_kernel():
    """Return the kernel module, or None if flydsl could not be imported.

    The failure is latched so a broken install is diagnosed once, not once per
    backward. Callers that have *selected* flydsl go through `require_flydsl`,
    which turns None into a raise.
    """
    global _kernel, _kernel_error
    if _kernel is not None or _kernel_error is not None:
        return _kernel
    try:
        from generative_recommenders.ops.flydsl import hstu_bwd_dkdv

        _kernel = hstu_bwd_dkdv
    except BaseException as e:  # noqa: BLE001 - report it, do not mask it
        _kernel_error = e
    return _kernel


def require_flydsl() -> None:
    """Raise `FlyDSLUnavailableError` unless the kernel can run on this host.

    Latched on success only: this sits on the per-backward path and neither the
    arch nor the install changes within a process, but an unsupported host must
    keep raising rather than pass on the second call.
    """
    global _host_ok
    if _host_ok:
        return
    reason = unsupported_hardware_reason()
    if reason is None and _load_kernel() is None:
        reason = f"flydsl import failed ({type(_kernel_error).__name__}: {_kernel_error})"
    if reason is None:
        _host_ok = True
        # The one place both call sites pass through, and the only selection
        # record the FUSED path produces -- it does not go through
        # ops/hstu_attention.py's dispatch, so without this a production run
        # says nothing about which backward it ran.
        # The forward half is reported from the env rather than hardcoded: it is
        # separately selectable and now defaults to flydsl too, so a fixed string
        # here would misreport the run it is supposed to be the record of.
        logger.info(
            "[hstu-attn] backward kernel: flydsl (forward: %s) on %s",
            "flydsl" if flydsl_fwd_selected() else "triton",
            _device_arch(),
        )
        return
    raise FlyDSLUnavailableError(
        f"the FlyDSL HSTU attention backward is selected but cannot run here: "
        f"{reason}. It is the default; to run the Triton backward on this host "
        f"set HSTU_ATTN_BWD=triton in the environment, or bind "
        f"apply_env_bootstrap.HSTU_ATTN_BWD = 'triton' in the gin config."
    )


def unsupported_shape_reason(
    q, k, v, dout, num_targets, max_attn_len, contextual_seq_len,
    num_softmax_heads: int = 0,
) -> Optional[str]:
    """Why the kernel cannot serve this call, or None if it can."""
    if max_attn_len:
        return f"max_attn_len={max_attn_len} (kernel is plain causal)"
    if contextual_seq_len:
        return f"contextual_seq_len={contextual_seq_len} (kernel assumes 0)"
    if num_softmax_heads:
        return f"num_softmax_heads={num_softmax_heads} (kernel assumes 0)"
    if q.dim() != 3 or q.shape[-1] != SUPPORTED_DIM:
        return f"qk dim {tuple(q.shape)} (kernel is written around dim {SUPPORTED_DIM})"
    if v.shape[-1] != SUPPORTED_DIM:
        return f"v dim {v.shape[-1]} (kernel is written around dim {SUPPORTED_DIM})"
    qkv_dtypes = {q.dtype, k.dtype, v.dtype}
    supported_qkv_dtype = qkv_dtypes == {torch.bfloat16} or (
        FP8_E4M3 is not None and qkv_dtypes == {FP8_E4M3}
    ) or (
        FP8_E4M3 is not None
        and q.dtype is torch.bfloat16
        and k.dtype is FP8_E4M3
        and v.dtype is FP8_E4M3
    )
    if not supported_qkv_dtype or dout.dtype is not torch.bfloat16:
        return (
            "dtype "
            f"{tuple(str(t.dtype).replace('torch.', '') for t in (q, k, v, dout))}"
            " (kernel requires BF16 QKV, E4M3 QKV, or BF16 Q with E4M3 K/V)"
        )
    # Layout is deliberately NOT a support condition. On the fused
    # preprocess+attention path q/k/v arrive as column slices of one uvqk buffer
    # and are not contiguous; the kernel's buffer descriptors assume a packed
    # [N, H, D] layout, so those are normalized with .contiguous() below rather
    # than being grounds to fall back.
    if num_targets is not None:
        # Checked here rather than trusted from the caller: that is the
        # difference between falling back and silently computing a different
        # function, which is the exact failure this kernel already shipped once.
        # The sync it costs is amortized to once per batch -- see _max_num_targets.
        n = _max_num_targets(num_targets)
        if n > 1:
            return f"num_targets max={n} (target-aware masking is identity only at <= 1)"
    return None


def flydsl_supports(
    q, k, v, dout, num_targets, max_attn_len, contextual_seq_len,
    num_softmax_heads: int = 0,
) -> bool:
    return unsupported_shape_reason(
        q, k, v, dout, num_targets, max_attn_len, contextual_seq_len,
        num_softmax_heads,
    ) is None


def _common_row_stride(tensors, heads, dim):
    """The shared token-row stride of `tensors`, or None if they cannot share one.

    Every tensor must be [tokens, heads, dim] whose two inner dimensions are
    packed -- stride (rs, dim, 1) -- differing only in `rs`, and every one must
    agree on it, because the kernel carries a single row stride for the whole
    uvqk family. Production satisfies this: q/k/v and dq/dk/dv are all column
    slices of one [tokens, 2048] buffer, so all six have stride (2048, 128, 1).

    rs must also be a multiple of 8 to keep the 8-wide bf16 loads 16 B-aligned
    at every token; the kernel asserts this too, but a caller-side check turns a
    bad layout into a fallback instead of a compile-time crash.
    """
    strides = set()
    for t in tensors:
        if t.dim() != 3 or t.shape[1] != heads or t.shape[2] != dim:
            return None
        rs, sh, sd = t.stride()
        if sd != 1 or sh != dim or rs < heads * dim or rs % 8:
            return None
        strides.add(rs)
    return strides.pop() if len(strides) == 1 else None


def bwd_qk_mfma_mode() -> str:
    """Attention-backward qk GEMM math: 'bf16', 'fp8', or 'fp8_strict'.

    ``DLRM_HSTU_FP8_BWD_QK_MFMA`` default is ``fp8``: k128 fp8 MFMA when this
    call's q and k are already e4m3; calibration stays bf16 math.
    ``bf16`` is the explicit opt-out. ``fp8_strict`` raises instead of skipping.

    Deprecated ``DLRM_HSTU_FP8_BWD_K128=1`` still selects ``fp8_strict``.
    ``K128=0`` is not an opt-out once the named knob defaults to fp8 -- set
    ``QK_MFMA=bf16``.
    """
    aliases = {
        "bf16": "bf16",
        "0": "bf16",
        "off": "bf16",
        "fp8": "fp8",
        "k128": "fp8",
        "auto": "fp8",
        "fp8_strict": "fp8_strict",
        "strict": "fp8_strict",
        "1": "fp8_strict",
    }

    def _parse(name: str, raw: Optional[str]) -> Optional[str]:
        if raw is None or raw.strip() == "":
            return None
        key = raw.strip().lower()
        if key not in aliases:
            raise RuntimeError(
                f"{name}={raw!r} is not one of bf16, fp8, fp8_strict "
                f"(deprecated K128: 0, auto, 1)."
            )
        return aliases[key]

    qk = _parse(
        "DLRM_HSTU_FP8_BWD_QK_MFMA",
        os.environ.get("DLRM_HSTU_FP8_BWD_QK_MFMA"),
    )
    k128 = _parse(
        "DLRM_HSTU_FP8_BWD_K128",
        os.environ.get("DLRM_HSTU_FP8_BWD_K128"),
    )
    if qk == "bf16":
        return "bf16"
    if qk == "fp8_strict" or k128 == "fp8_strict":
        return "fp8_strict"
    return "fp8"


# Removed: block dO scales (DOBLK) and the saturating cast (SAT)
# are the only C8 path now. An off value must not be silently ignored.
_REMOVED_C8_KNOBS = ("DLRM_HSTU_FP8_BWD_FULL_DOBLK", "DLRM_HSTU_FP8_BWD_FULL_SAT")


def check_removed_c8_knobs() -> None:
    """Raise if a removed C8 sub-knob is set to an off value (an on value is
    the only behaviour and is accepted). Same rule as
    dlrm_v4/train/_env_bootstrap._check_removed_c8_knobs."""
    for name in _REMOVED_C8_KNOBS:
        raw = os.environ.get(name)
        if raw is not None and raw.strip() != "" and raw.strip().lower() \
                not in ("1", "true", "on", "yes"):
            raise RuntimeError(
                f"{name}={raw!r}: this knob was removed; DOBLK+SAT is the only "
                "C8 path. Use DLRM_HSTU_FP8_BWD_FULL=0 to turn C8 off.")


def bwd_full_fp8_enabled() -> bool:
    """$DLRM_HSTU_FP8_BWD_FULL (default OFF): full-fp8 attention backward."""
    if _hybrid_force_off[0]:
        return False
    return os.environ.get("DLRM_HSTU_FP8_BWD_FULL", "0").strip().lower() in (
        "1", "true", "on", "yes")


# $DLRM_HSTU_FP8_BWD_FULL_HYBRID = dq | dkdv (default unset). Diagnostic only:
# it runs both backward kernels and keeps one gradient from each.
_hybrid_force_off = [False]
_hybrid_inner = [False]


_full_fp8_stats_calls = [0]


def _full_fp8_stats_left() -> bool:
    """$DLRM_HSTU_FP8_BWD_FULL_STATS=<n>: DEBUG ONLY, host-syncs. Log operand
    ranges for the first n armed calls."""
    n = int(os.environ.get("DLRM_HSTU_FP8_BWD_FULL_STATS", "0") or 0)
    return _full_fp8_stats_calls[0] < n


@torch.no_grad()
def _full_fp8_stats(q, k, v, dout8, do_scale, seq_offsets, N, alpha,
                    qkv_scale, config, dq, dk, dv, dout_orig=None):
    """fp32 reference ranges of the full-fp8 kernel's e4m3 tiles on the longest
    of the first 16 sequences (same math as the kernel: silu/N, ds with the
    NORMALIZED dO), plus finiteness of the kernel outputs. Host syncs."""
    _full_fp8_stats_calls[0] += 1
    # Longest WHOLE sequence <= 2048 among the first 64 (a truncated one would
    # give a wrong dk/dv reference: keys see every later query row).
    offs = seq_offsets[:65].tolist()
    lens = [b - a if b - a <= 2048 else 0 for a, b in zip(offs[:-1], offs[1:])]
    i = int(np.argmax(lens))
    st, L = offs[i], lens[i]
    qd = q[st:st + L].float() * qkv_scale
    kd = k[st:st + L].float() * qkv_scale
    vd = v[st:st + L].float() * qkv_scale
    do_n = dout8[st:st + L].float()
    x = torch.einsum("mhd,nhd->hmn", qd, kd) * alpha
    mask = torch.tril(torch.ones(L, L, device=q.device, dtype=torch.bool))
    sg = torch.sigmoid(x)
    siluN = (x * sg).masked_fill(~mask, 0)
    gc = (sg * (1 + x * (1 - sg))).masked_fill(~mask, 0)
    dsN = torch.einsum("mhd,nhd->hmn", do_n, vd) * gc
    xa = float(x.masked_fill(~mask, 0).abs().max())
    sa = float(siluN.abs().max())
    da = float(dsN.abs().max())
    ssc, dsc = config["mu2_ssc"], config["mu2_dsc"]
    s_t = sa / float(N) * ssc
    d_t = da / float(N) * dsc
    # fraction of tile elements that would overflow / underflow e4m3
    s_tile = siluN / float(N) * ssc
    d_tile = dsN / float(N) * dsc
    nz = mask.sum().item() * x.shape[0]
    def frac(t, lo):
        return (float((t.abs() > 448).sum()) / nz,
                float(((t.abs() < lo) & (t != 0)).sum()) / nz)
    fin = [bool(torch.isfinite(t).all()) for t in (dq, dk, dv)]

    def q8snr(t):
        t = t[mask.expand_as(t)]
        g = t.clamp(-448, 448).to(torch.float8_e4m3fn).float()
        return float(10 * torch.log10(t.pow(2).sum()
                                      / (t - g).pow(2).sum().clamp_min(1e-30)))
    c = int(np.ceil(np.log2(max(int(N), 2))))
    sweep = {}
    for nm, t in (("silu", siluN), ("ds", dsN)):
        r = {kk: q8snr(t * 2.0 ** (kk + c) / float(N)) for kk in range(-6, 12)}
        best = max(r, key=r.get)
        sweep[nm] = "best K=%d %.1fdB | cur %.1fdB | %s" % (
            best, r[best], q8snr(t * (ssc if nm == "silu" else dsc) / float(N)),
            " ".join("%d:%.1f" % (kk, r[kk]) for kk in range(-6, 12, 2)))
    gsnr = "n/a"
    if dout_orig is not None:
        with torch.enable_grad():
            qg, kg, vg = (t.detach().clone().requires_grad_() for t in (qd, kd, vd))
            xg = torch.einsum("mhd,nhd->hmn", qg, kg) * alpha
            pg = torch.nn.functional.silu(xg) / float(N) * mask
            og = torch.einsum("hmn,nhd->mhd", pg, vg)
            rq, rk, rv = torch.autograd.grad(
                (og * dout_orig[st:st + L].float()).sum(), (qg, kg, vg))
        def gs(r_, g_):
            g_ = g_[st:st + L].float()
            return float(10 * torch.log10(r_.pow(2).sum()
                                          / (r_ - g_).pow(2).sum().clamp_min(1e-30)))
        gsnr = "dv %.1f dq %.1f dk %.1f" % (gs(rv, dv), gs(rq, dq), gs(rk, dk))
    print("[hstu-attn][full-fp8 stats] call=%d SNR(kernel vs fp32 ref, L=%d) %s"
          " || silu-op %s || ds-op %s" % (_full_fp8_stats_calls[0], L, gsnr,
                                          sweep["silu"], sweep["ds"]),
          flush=True)
    print(
        "[hstu-attn][full-fp8 stats] call=%d N=%d L=%d qkv_scale=%.4g "
        "s_do=%.4g |x|max=%.4g |silu*N|max=%.4g |ds*N|max=%.4g "
        "silu_tile_max=%.4g (ssc=2^%d) ds_tile_max=%.4g (dsc=2^%d) "
        "silu ovf/sub=%.3g/%.3g ds ovf/sub=%.3g/%.3g "
        "|q|max=%.4g |k|max=%.4g |v|max=%.4g finite(dq,dk,dv)=%s" % (
        _full_fp8_stats_calls[0], int(N), L, qkv_scale,
        # s_do: dO scales are per (seq, head, MX group) E8 bytes; no scalar.
        float("nan"), xa, sa, da, s_t, int(np.log2(ssc)), d_t,
        int(np.log2(dsc)), *frac(s_tile, 2.0 ** -9), *frac(d_tile, 2.0 ** -9),
        float(qd.abs().max()), float(kd.abs().max()), float(vd.abs().max()),
        fin,
    ), flush=True)


_bwd_dump_calls = [0]


def _bwd_dump_dir() -> str:
    """$DLRM_HSTU_FP8_BWD_DUMP=<dir> (default OFF; DEBUG ONLY, host-syncs on
    the selected calls). Rank 0 saves real backward operands for offline
    numerics (C8 L0 harness) on the call indices listed in
    $DLRM_HSTU_FP8_BWD_DUMP_CALLS (comma list, 0-based over every backward
    call of this process). Inert when unset."""
    d = os.environ.get("DLRM_HSTU_FP8_BWD_DUMP", "")
    if not d:
        return ""
    rank = (torch.distributed.get_rank()
            if torch.distributed.is_available()
            and torch.distributed.is_initialized()
            else int(os.environ.get("RANK", "0") or 0))
    return d if rank == 0 else ""


@torch.no_grad()
def _bwd_dump(q, k, v, dout, do_scale, seq_offsets, N, alpha, qkv_scale,
              dq, dk, dv, config) -> None:
    i = _bwd_dump_calls[0]
    _bwd_dump_calls[0] += 1
    want = {int(x) for x in os.environ.get(
        "DLRM_HSTU_FP8_BWD_DUMP_CALLS", "").split(",") if x.strip()}
    if i not in want:
        return
    d = _bwd_dump_dir()
    os.makedirs(d, exist_ok=True)
    offs = seq_offsets.to(torch.int64).cpu()
    lens = offs[1:] - offs[:-1]
    B = lens.numel()
    # Whole sequences only (attention is per sequence, so slicing is exact):
    # the 4 longest plus 12 spread evenly over the batch.
    longest = torch.argsort(lens, descending=True)[:4].tolist()
    spread = torch.linspace(0, B - 1, 12).round().long().tolist()
    sel = sorted(set(longest) | set(spread))
    rows = torch.cat([torch.arange(int(offs[s]), int(offs[s + 1]))
                      for s in sel]).to(q.device)

    def g(t):
        return t.index_select(0, rows).contiguous().cpu()

    dof = dout.float()
    payload = {
        "call": i, "N": int(N), "alpha": float(alpha),
        "qkv_scale": float(qkv_scale), "sel": sel, "seq_offsets": offs,
        "q": g(q), "k": g(k), "v": g(v), "dout": g(dout),
        "dq": g(dq), "dk": g(dk), "dv": g(dv),
        "do_scale": (None if do_scale is None or do_scale.dtype == torch.uint8
                     else do_scale.cpu()),
        # do_blk (C8 option B): the per-token group bytes E8 = 127 - e_g.
        "do_e8": (g(do_scale) if do_scale is not None
                  and do_scale.dtype == torch.uint8 else None),
        # Whole-batch dO magnitude census, per token over heads x dim.
        "do_row_amax": dof.abs().amax(dim=(1, 2)).cpu(),
        "do_row_rms": dof.pow(2).mean(dim=(1, 2)).sqrt().cpu(),
        "config": {kk: (vv if isinstance(vv, (int, float, str, bool))
                        else str(vv)) for kk, vv in config.items()},
    }
    p = os.path.join(d, "bwd_call%05d.pt" % i)
    torch.save(payload, p)
    logger.warning("[hstu-attn] BWD DUMP call=%d -> %s (%d seqs, %d rows)",
                   i, p, len(sel), rows.numel())


_DQ_SPLIT_ORDER_LOGGED = set()


def _dq_split_order(N: int) -> str:
    """Grid order of the dq-split kernel ($DLRM_HSTU_FP8_BWD_DQ_SPLIT_ORDER).

    "xcd" (default) groups each (b, h)'s row blocks on one XCD so its K tiles
    stay in that XCD's L2; "bh" is the row-major order.  Bitwise-identical dq.
    """
    order = os.environ.get("DLRM_HSTU_FP8_BWD_DQ_SPLIT_ORDER", "xcd").strip().lower()
    if order not in ("bh", "xcd"):
        raise ValueError(f"DLRM_HSTU_FP8_BWD_DQ_SPLIT_ORDER={order!r}: use bh|xcd")
    if order not in _DQ_SPLIT_ORDER_LOGGED:
        _DQ_SPLIT_ORDER_LOGGED.add(order)
        logger.warning("[hstu-attn] bwd dq SPLIT grid order: %s", order)
    return order


def flydsl_hstu_attention_bwd(
    dout,
    q,
    k,
    v,
    seq_offsets,
    N,
    alpha,
    out=None,
    qkv_scale: float = 1.0,
):
    """The five-gemm fused backward with BF16 or scaled-E4M3 Q/K/V.

    With `out=(dq, dk, dv)` the gradients are written straight into those
    tensors, which may be strided views; otherwise fresh packed tensors are
    allocated and returned. The in-place form is what avoids the wrapper tax on
    the production path -- see try_flydsl_attention_bwd.

    dq is accumulated across kv blocks with global atomics, so it must start at
    zero -- the zero-fill cannot be folded into the kernel, because nothing
    orders one workgroup's zeroing against another's atomic add. Under the
    shipped dq16 mode dq is bf16 and IS the accumulator, so there is no
    down-convert pass and the zero-fill moves half the bytes.
    """
    _hy = os.environ.get("DLRM_HSTU_FP8_BWD_FULL_HYBRID", "")
    if _hy and not _hybrid_inner[0] and bwd_full_fp8_enabled():
        if _hy not in ("dq", "dkdv"):
            raise ValueError(
                f"DLRM_HSTU_FP8_BWD_FULL_HYBRID={_hy!r}: expected dq|dkdv")
        _hybrid_inner[0] = True
        try:
            _hybrid_force_off[0] = True
            try:
                ref = flydsl_hstu_attention_bwd(
                    dout, q, k, v, seq_offsets, N, alpha, out=None,
                    qkv_scale=qkv_scale)
            finally:
                _hybrid_force_off[0] = False
            dq, dk, dv = flydsl_hstu_attention_bwd(
                dout, q, k, v, seq_offsets, N, alpha, out=out,
                qkv_scale=qkv_scale)
        finally:
            _hybrid_inner[0] = False
        if _hy == "dq":
            dq.copy_(ref[0])
        else:
            dk.copy_(ref[1])
            dv.copy_(ref[2])
        if "f8b_hybrid" not in _logged_layouts:
            _logged_layouts.add("f8b_hybrid")
            logger.warning(
                "[hstu-attn] FULL-FP8 HYBRID=%s ARMED: %s taken from the "
                "bf16-backward kernel (diagnostic; both kernels run)", _hy,
                "dq" if _hy == "dq" else "dk,dv")
        return dq, dk, dv
    kernel = _load_kernel()
    heads, dim = q.shape[1], q.shape[2]
    acc_dtype = kernel.dq_acc_dtype()
    input_fp8 = FP8_E4M3 is not None and k.dtype is FP8_E4M3
    q_bf16 = input_fp8 and q.dtype is torch.bfloat16
    if input_fp8 and not (qkv_scale > 0.0):
        raise ValueError("FP8 Q/K/V requires a positive dequantization scale")

    # Read q/k/v where they already live when the layout allows it. dO is not
    # part of the uvqk buffer (it is the attention output's gradient) and keeps
    # the packed stride, so .contiguous() stays -- a no-op in production.
    if q_bf16:
        rs_in = _common_row_stride((k, v), heads, dim)
        rs_q = _common_row_stride((q,), heads, dim)
    else:
        rs_in = _common_row_stride((q, k, v), heads, dim)
        rs_q = rs_in
    rs_out = None
    if out is not None:
        dq, dk, dv = out
        # Only when dq IS the accumulator. A narrower accumulator than the
        # caller's dq would need a down-convert, and that has nowhere to land
        # in place, so those modes take the allocate-and-copy path.
        if acc_dtype is torch.bfloat16 and dq.dtype is acc_dtype:
            rs_out = _common_row_stride((dq, dk, dv), heads, dim)

    dout = dout.contiguous()
    if q_bf16:
        if rs_in is None:
            k, v = k.contiguous(), v.contiguous()
            rs_in = heads * dim
        if rs_q is None:
            q = q.contiguous()
            rs_q = heads * dim
    elif rs_in is None:
        q, k, v = (t.contiguous() for t in (q, k, v))
        rs_in = heads * dim
        rs_q = rs_in
    # R15 design B ($DLRM_HSTU_FP8_BWD_DQ_SPLIT=1, default OFF): dq is computed
    # by hstu_dq_split from a stored e4m3 ds workspace with plain stores, so
    # the zero-fill below is deferred until the C8 path is known not to take
    # it.  Unset = the original zeroing at the original point.
    _dqs_req = os.environ.get("DLRM_HSTU_FP8_BWD_DQ_SPLIT", "").strip().lower() \
        in ("1", "true", "on", "yes")
    if rs_out is None:
        output_shape = q.shape
        dk = torch.empty(output_shape, dtype=torch.bfloat16, device=q.device)
        dv = torch.empty(output_shape, dtype=torch.bfloat16, device=q.device)
        dq = (torch.empty if _dqs_req else torch.zeros)(
            output_shape, dtype=acc_dtype, device=q.device)
        rs_out = heads * dim
    elif not _dqs_req:
        # Same reasoning: dk/dv are fully overwritten, so only dq needs zeroing,
        # and it is zeroed in place rather than allocated.
        dq.zero_()
    config = dict(kernel.SHIPPED)
    if input_fp8:
        # The packed conversion changes the VALU issue mix enough that the
        # BF16-tuned interleave is no longer optimal on gfx950.
        config["sched"] = "iglp3"
    if q_bf16 and kernel.cvt_scale_is_exact(qkv_scale):
        # ONLY when the scale is an exact power of two. scale_in_cvt folds
        # qkv_scale into cvt_scalef32_pk_bf16_fp8. That instruction's scale
        # operand is exponent-only on gfx950. A scale that is not a power of
        # two would silently rescale the values. The kernel checks this at
        # every launch and raises if the flag is set by hand.
        config["scale_in_cvt"] = True
    elif q_bf16 and "q_bf16_scale_in_cvt" not in _logged_layouts:
        _logged_layouts.add("q_bf16_scale_in_cvt")
        logger.warning(
            "[hstu-attn] qkv_scale=%r is not an exact power of two, so the "
            "BF16-Q/E4M3-K/V backward applies it in the score math rather than "
            "folding it into the fp8 converts. Correct, and slower than the "
            "fold would be; quantize with a power-of-two scale to get both.",
            qkv_scale,
        )
    # True fp8 MFMA on the qk gemm. $DLRM_HSTU_FP8_BWD_QK_MFMA, DEFAULT fp8.
    #   "fp8"        k128 fp8 MFMA when this call's q AND k are already e4m3;
    #                calibration (bf16 q/k) stays bf16 math.
    #   "bf16"       explicit opt-out -- shipped-before-this-default math.
    #   "fp8_strict" engage, RAISE on any call whose q/k are not both e4m3.
    # Deprecated alias: $DLRM_HSTU_FP8_BWD_K128=0|auto|1 -> bf16|fp8|fp8_strict.
    fp8_mfma = ""
    _qk_mode = bwd_qk_mfma_mode()
    if _qk_mode in ("fp8", "fp8_strict"):
        if not input_fp8 or q_bf16:
            if _qk_mode == "fp8_strict":
                raise RuntimeError(
                    "DLRM_HSTU_FP8_BWD_QK_MFMA=fp8_strict requires E4M3 q AND "
                    "k on every backward call, but this call has "
                    f"q={q.dtype} k={k.dtype} (input_fp8={input_fp8}, "
                    f"q_bf16={q_bf16}). Use QK_MFMA=fp8 so calibration "
                    "steps stay on bf16 math, or QK_MFMA=bf16 to opt out."
                )
        else:
            fp8_mfma = "k128"
            config["sched"] = "iglp2"
    # C8 full-fp8 backward. $DLRM_HSTU_FP8_BWD_FULL (absent = OFF; gin ON).
    # All five gemms on mfma_scale_f32_16x16x128_f8f6f4. dO is quantized here
    # to e4m3 with one pow2 scale per (sequence, head, 32-token MX K-block
    # group) -- quantize_do_e4m3_blk, no host sync -- and the kernel undoes
    # them inside the out-gemm MFMAs (do_blk), with the saturating e4m3 cast
    # of silu/ds (f8_sat). That is the ONLY C8 path: the per-tensor dO scale
    # (do_dyn) flushed small dO rows to zero in-model and is no longer
    # reachable from here, and do_blk without f8_sat NaNs gradients.  A call
    # shape do_blk cannot serve takes the bf16-dO kernel below instead.
    do_scale = None
    f8b_full = bwd_full_fp8_enabled()
    if f8b_full:
        check_removed_c8_knobs()
    # do_blk's K-block <-> token-group identity needs BM=64 and BN % 128 == 0
    # (kernel assert), and quantize_do_e4m3_blk needs power-of-two H and H*D.
    _blk_ok = (int(config.get("BM", 64)) == 64
               and int(config.get("BN", 128)) % 128 == 0
               and heads > 0 and heads & (heads - 1) == 0
               and (heads * dim) & (heads * dim - 1) == 0)
    if f8b_full and fp8_mfma == "k128" and input_fp8 and not q_bf16 \
            and config.get("fused") and config.get("dq16") == "bf16p" \
            and _blk_ok:
        # silu and ds both carry the kernel's 1/N, so their e4m3 pre-scales
        # are N-RELATIVE: 2**(K + ceil(log2 N)) puts silu*N (resp. ds*N with
        # normalized dO) times 2**K..2**(K+1) on the e4m3 grid, and only one
        # or two kernels compile (ceil(log2 N) is 12 for every in-model N
        # seen in 51366).  In-model 51366: |silu*N| <= ~2 (|x| <= 2.2),
        # |ds*N| 0.5..323 across layers.  An absolute *_LOG2 overrides.
        # f8_sat (always on) clamps before the cast: gfx950's
        # cvt_scalef32_pk_fp8_f32 returns NaN, not 448, on overflow.
        c = int(math.ceil(math.log2(max(int(N), 2))))
        def _sc(name, k_default):
            ab = os.environ.get(f"DLRM_HSTU_FP8_BWD_FULL_{name}_LOG2")
            if ab not in (None, ""):
                return 2.0 ** int(ab)
            return 2.0 ** (c + int(os.environ.get(
                f"DLRM_HSTU_FP8_BWD_FULL_{name}_K", k_default)))
        config.update(
            f8b="all", mu2="k128", do_dyn=False, do_blk=True, f8_sat=True,
            # silu and ds pre-scales. Override with DLRM_HSTU_FP8_BWD_FULL_*_K.
            mu2_ssc=_sc("SSC", "3"),
            mu2_dsc=_sc("DSC", "-3"),
            ds8_pad=int(os.environ.get("DLRM_HSTU_FP8_BWD_FULL_DS8_PAD", "16")),
        )
        # The silu clamp is upper-only (silu*N >= -0.279), which stays exact
        # while -0.279 * ssc * 2/2**c > -448, i.e. K <= 9.
        assert config["mu2_ssc"] <= 2.0 ** (c + 9), (
            "DLRM_HSTU_FP8_BWD_FULL ssc too large for the upper-only clamp")
        from generative_recommenders.ops.flydsl.hstu_do_fp8 import (
            quantize_do_e4m3_blk,
        )
        if _dqs_req and dq.dtype is torch.bfloat16:
            config["dqsplit"] = True
        dout_orig = dout
        # One pow2 dO scale per (sequence, head, MX token group); do_scale is
        # E8 [T, H] uint8 (127 - e_g), undone inside the kernel's out-gemm
        # MFMAs (no epilogue rescale).
        dout, do_scale = quantize_do_e4m3_blk(dout, seq_offsets, N)
    elif f8b_full:
        # Two once-keys: the expected calibration-step skips (bf16 q/k) must
        # not hide a later block-dO shape fallback, which silently drops a
        # production call from C8.
        _skip_key = "f8b_full_skip" if _blk_ok else "f8b_full_skip_shape"
        if _skip_key not in _logged_layouts:
            _logged_layouts.add(_skip_key)
            logger.warning(
                "[hstu-attn] DLRM_HSTU_FP8_BWD_FULL set but this call cannot "
                "use it (%s; qk_mfma=%r input_fp8=%s q_bf16=%s "
                "do_blk_shape_ok=%s heads=%d dim=%d BM=%s BN=%s); bf16-dO "
                "kernel used.",
                "operand/config" if _blk_ok else "block-dO SHAPE fallback",
                fp8_mfma or "bf16", input_fp8, q_bf16, _blk_ok, heads, dim,
                config.get("BM", 64), config.get("BN", 128),
            )
    # $DLRM_HSTU_FP8_BWD_EPILOGUE = legacy (default when absent) | reuse
    # (gin: apply_env_bootstrap.DLRM_HSTU_FP8_BWD_EPILOGUE).
    _epilogue = os.environ.get("DLRM_HSTU_FP8_BWD_EPILOGUE", "legacy") or "legacy"
    if _epilogue not in ("reuse", "legacy", "reuse_pk"):
        raise ValueError(
            f"DLRM_HSTU_FP8_BWD_EPILOGUE={_epilogue!r}: expected "
            "reuse|legacy|reuse_pk"
        )
    # "reuse_pk" runs the "reuse" silu chain on packed f32. Same ops in the
    # same order. Default off. Set DLRM_HSTU_FP8_BWD_EPILOGUE=reuse_pk or
    # DLRM_HSTU_FP8_BWD_PK=1.
    if (os.environ.get("DLRM_HSTU_FP8_BWD_PK", "").strip().lower()
            in ("1", "true", "on", "yes") and _epilogue == "reuse"):
        _epilogue = "reuse_pk"
    if _epilogue == "reuse_pk" and ("epi", "reuse_pk") not in _logged_layouts:
        _logged_layouts.add(("epi", "reuse_pk"))
        logger.warning(
            "[hstu-attn] bwd_dkdv packed-f32 epilogue ARMED: epilogue=reuse_pk "
            "(DLRM_HSTU_FP8_BWD_PK)")
    # $DLRM_HSTU_FP8_BWD_GORDER = xcd | xcdn. Absent keeps the kernel default
    # "xcd". "xcdn" puts n-blocks fastest inside an XCD. Same math.
    _gorder = os.environ.get("DLRM_HSTU_FP8_BWD_GORDER", "").strip().lower()
    if _gorder:
        if _gorder not in ("xcd", "xcdn"):
            raise ValueError(
                f"DLRM_HSTU_FP8_BWD_GORDER={_gorder!r}: expected xcd|xcdn"
            )
        config["gorder"] = _gorder
        if ("gorder", _gorder) not in _logged_layouts:
            _logged_layouts.add(("gorder", _gorder))
            logger.warning(
                "[hstu-attn] bwd_dkdv grid order ARMED: gorder=%s "
                "(DLRM_HSTU_FP8_BWD_GORDER)", _gorder,
            )
    # Intra-pair lookahead ($DLRM_HSTU_FP8_BWD_PIPE = 1..6,
    # default OFF).  dq-split C8 kernel only: both m tiles of an mu2 pair are
    # staged in LDS up front and tile B's qk/dp MFMAs share a scheduling
    # region with tile A's silu/ds VALU chain.  Same math on every element
    # (dk/dv/dq-workspace bitwise equal to the unpiped split kernel).  Unset
    # (or a non-split call) leaves the config dict, hence the compile cache
    # key and the kernel, unchanged.
    _pipe = os.environ.get("DLRM_HSTU_FP8_BWD_PIPE", "").strip()
    if _pipe and _pipe != "0" and config.get("dqsplit"):
        if _pipe not in ("1", "2", "3", "4", "5", "6"):
            raise ValueError(
                f"DLRM_HSTU_FP8_BWD_PIPE={_pipe!r}: expected 1..6")
        config["pipe"] = int(_pipe)
        if ("pipe", _pipe) not in _logged_layouts:
            _logged_layouts.add(("pipe", _pipe))
            logger.warning(
                "[hstu-attn] bwd_dkdv intra-pair lookahead ARMED: pipe=%s "
                "(DLRM_HSTU_FP8_BWD_PIPE; dq-split mu2 pairs staged together)",
                _pipe)
    fn = kernel.compile_hstu_bwd_dkdv(
        heads,
        dim,
        alpha,
        epilogue=_epilogue,
        rs_in=rs_in,
        rs_out=rs_out,
        input_fp8=input_fp8,
        q_bf16=q_bf16,
        rs_q=rs_q,
        fp8_mfma=fp8_mfma,
        **config,
    )
    # Logged once per distinct layout.
    _e1_period = int(
        os.environ.get("DLRM_HSTU_FP8_BWD_CENSUS")
        or os.environ.get("DLRM_E1_BWD_CENSUS", "0")
        or 0
    )
    if _e1_period:
        global _e1_bwd_calls
        _e1_bwd_calls += 1
        vk = (
            "e4m3" if (input_fp8 and not q_bf16)
            else ("bf16q_e4m3kv" if q_bf16 else "bf16"),
            fp8_mfma or "bf16mfma",
        )
        _e1_bwd_census[vk] = _e1_bwd_census.get(vk, 0) + 1
        if _e1_bwd_calls % _e1_period == 0:
            logger.warning(
                "[e1census] calls=%d %s",
                _e1_bwd_calls,
                sorted(_e1_bwd_census.items()),
            )

    key = (rs_in, rs_q, rs_out, input_fp8, q_bf16, heads, dim, out is not None,
           fp8_mfma, do_scale is not None, bool(config.get("do_blk")),
           _epilogue)
    if key not in _logged_layouts:
        _logged_layouts.add(key)
        logger.warning(
            "[e1] bwd_dkdv layout qk_mfma=%r input_fp8=%s q_bf16=%s "
            "rs_in=%d rs_q=%d rs_out=%d heads=%d dim=%d epilogue=%s",
            fp8_mfma or "bf16", input_fp8, q_bf16, rs_in, rs_q, rs_out,
            heads, dim, _epilogue,
        )
        if do_scale is not None:
            logger.warning(
                "[hstu-attn] FULL-FP8 backward ARMED: f8b=all mu2=k128 "
                "do_dyn=%d do_blk=%d ssc=2^%d dsc=2^%d ds8_pad=%d sat=%d "
                "(dO e4m3, runtime pow2 scale per (seq, head, 32-token MX "
                "group); N=%d)",
                # do_dyn is always 0 here; the field stays for log-grep compat.
                int(bool(config.get("do_dyn"))), int(bool(config.get("do_blk"))),
                int(np.log2(config["mu2_ssc"])), int(np.log2(config["mu2_dsc"])),
                config["ds8_pad"], int(config["f8_sat"]), int(N),
            )
        logger.warning(
            "[hstu-attn] FlyDSL backward reading %s at row stride %d "
            "(Q stride %d) and writing BF16 gradients at row stride %d "
            "(heads=%d dim=%d)",
            "BF16 Q + E4M3 K/V"
            if q_bf16
            else ("E4M3 Q/K/V" if input_fp8 else "BF16 Q/K/V"),
            rs_in,
            rs_q,
            rs_out,
            heads,
            dim,
        )

    # The kernel reads seq_offsets through a 32-bit buffer descriptor;
    # production carries it as int64.
    cu = seq_offsets.to(torch.int32)
    batch = seq_offsets.numel() - 1
    # N, not seq_offsets.diff().max(): the caller's max_seq_len already bounds
    # every sequence, and reading the real max costs a device-to-host sync in
    # the deepest part of the backward queue (profiled at 31.7 ms a call). The
    # two agreed exactly on 1,600 audited calls across 8 ranks -- production
    # derives N as max_uih_len + max_num_candidates, and with
    # max_num_candidates = 1 the pair can differ by at most one token anyway,
    # which moves n_blocks only when N lands one past a block boundary. A larger
    # N would over-launch, never under-launch, so the failure mode is bounded
    # waste rather than a wrong answer.
    n_blocks = int((int(N) + fn.BN - 1) // fn.BN)

    _dqs = bool(config.get("dqsplit"))
    if _dqs_req and not _dqs:
        dq.zero_()
    dq_slot = dq
    if _dqs:
        from generative_recommenders.ops.flydsl import hstu_dq_split
        ws = hstu_dq_split.build_workspace(seq_offsets, heads, q.shape[0], N)
        dq_slot = ws.view(torch.bfloat16)
        if "dq_split" not in _logged_layouts:
            _logged_layouts.add("dq_split")
            logger.warning(
                "[hstu-attn] bwd dq SPLIT ARMED: e4m3 ds workspace %.2f GB + "
                "f32 dq kernel, no dq atomics/zero-fill "
                "(DLRM_HSTU_FP8_BWD_DQ_SPLIT)", ws.numel() / 1e9)
    # The saved q/k/v still carry requires_grad, and flydsl's dlpack export
    # refuses those.
    fn(
        q.detach(),
        k.detach(),
        v.detach(),
        dout.detach(),
        dk,
        dv,
        dq_slot,
        cu,
        batch,
        n_blocks,
        1.0 / float(N),
        qkv_scale,
        **({"do_scale": do_scale} if do_scale is not None else {}),
    )
    if _dqs:
        hstu_dq_split.dq_from_ds(
            ws, k.detach(), do_scale, seq_offsets, dq, N,
            alpha * float(config.get("dq_scale", 1.0)) / config["mu2_dsc"]
            * float(qkv_scale),
            rs_in, rs_out,
            fp8_dot=os.environ.get("DLRM_HSTU_FP8_BWD_DQ_SPLIT_DOT", "fp8")
            != "bf16",
            bmq=int(os.environ.get("DLRM_HSTU_FP8_BWD_DQ_SPLIT_BM", "128")),
            num_warps=int(os.environ.get("DLRM_HSTU_FP8_BWD_DQ_SPLIT_NW", "8")),
            stages=int(os.environ.get("DLRM_HSTU_FP8_BWD_DQ_SPLIT_STAGES", "2")),
            order=_dq_split_order(int(N)))
        del ws, dq_slot
    if do_scale is not None and _full_fp8_stats_left():
        _full_fp8_stats(q, k, v, dout, do_scale, seq_offsets, N, alpha,
                        qkv_scale, config, dq, dk, dv, dout_orig)
    if _bwd_dump_dir():
        _bwd_dump(q, k, v, dout_orig if do_scale is not None else dout,
                  do_scale, seq_offsets, N, alpha, qkv_scale, dq, dk, dv,
                  config)
    if dq.dtype is not torch.bfloat16:
        dq = dq.to(torch.bfloat16)
    return dq, dk, dv


def flydsl_selected() -> bool:
    """True when $HSTU_ATTN_BWD asks for FlyDSL. Says nothing about whether it can run."""
    return (
        os.environ.get("HSTU_ATTN_BWD", DEFAULT_BWD_BACKEND).strip().lower()
        or DEFAULT_BWD_BACKEND
    ) == "flydsl"


def flydsl_enabled() -> bool:
    """True when FlyDSL is selected; raises if it is selected but unusable.

    Selection and usability are separate questions on purpose -- see the
    module docstring. This never returns False for a host problem.
    """
    if not flydsl_selected():
        return False
    require_flydsl()
    return True


def _log_shape_fallback(reason: str) -> None:
    """Log the first fallback of each distinct reason, then stay quiet."""
    if reason in _logged_fallbacks:
        return
    _logged_fallbacks.add(reason)
    logger.warning(
        "[hstu-attn] FlyDSL backward cannot serve this shape (%s); using the "
        "Triton backward for these calls. This is a per-shape fallback, not a "
        "host problem, and is logged once per reason.",
        reason,
    )


def try_flydsl_attention_bwd(
    *, dout, q, k, v, dq, dk, dv, seq_offsets, num_targets, N, alpha,
    max_attn_len, contextual_seq_len, num_softmax_heads, qkv_scale=1.0,
) -> bool:
    """Write the attention backward into dq/dk/dv with FlyDSL.

    Returns False without touching the outputs when FlyDSL is not selected or
    cannot serve this shape, so the caller runs its normal Triton backward.

    Mirrors `triton_hstu_attention_bwd`'s convention of writing into caller-owned
    dq/dk/dv rather than returning them: on the fused preprocess+attention path
    those are views into a single `duvqk` buffer that the rest of the backward
    consumes whole, so the gradients have to land in that buffer.

    The kernel is handed those views directly, and reads q/k/v out of `uvqk`
    where they already sit. It used to require packed tensors, which cost three
    pack copies in and three copy-backs out per call -- with a zero-fill for the
    dq accumulator, seven extra kernels per layer. Passing the row stride
    instead costs +0.16% inside the kernel and leaves only dq's zero-fill.
    Layouts that cannot share one row stride still take the copy path.
    """
    if not flydsl_enabled():
        return False
    reason = unsupported_shape_reason(
        q, k, v, dout, num_targets, max_attn_len, contextual_seq_len,
        num_softmax_heads,
    )
    if reason is not None:
        _log_shape_fallback(reason)
        return False
    global _backward_calls
    out_dq, out_dk, out_dv = flydsl_hstu_attention_bwd(
        dout,
        q,
        k,
        v,
        seq_offsets,
        N,
        alpha,
        out=(dq, dk, dv),
        qkv_scale=qkv_scale,
    )

    _backward_calls += 1
    if _check_due(_backward_calls):
        # sort_by_length_indices is a scheduling hint only, so the reference does
        # not need the caller's; enable_tma likewise does not change the result.
        _log_reference_snr(
            out_dq, out_dk, out_dv,
            _triton_bwd(
                dout, q, k, v, seq_offsets, num_targets, N, alpha,
                max_attn_len, contextual_seq_len, None, False,
            ),
            q.shape,
        )

    # No-ops when the kernel wrote in place; real copies only when the layout
    # forced the packed path and these are distinct tensors.
    for dst, src in ((dq, out_dq), (dk, out_dk), (dv, out_dv)):
        if dst.data_ptr() != src.data_ptr():
            dst.copy_(src)
    return True


# ---------------------------------------------------------------------------
# Forward
#
# Selected independently of the backward ($HSTU_ATTN_FWD vs $HSTU_ATTN_BWD),
# because the two kernels are genuinely decoupled: HSTU has no softmax, so there
# is no lse and no rowsum(dO*O) for the forward to hand the backward. Either can
# run while the other stays on Triton.
# ---------------------------------------------------------------------------


def _load_fwd_kernel():
    """Return the forward kernel module, or None if it could not be imported."""
    global _fwd_kernel, _fwd_kernel_error
    if _fwd_kernel is not None or _fwd_kernel_error is not None:
        return _fwd_kernel
    try:
        from generative_recommenders.ops.flydsl import hstu_fwd

        _fwd_kernel = hstu_fwd
    except BaseException as e:  # noqa: BLE001 - report it, do not mask it
        _fwd_kernel_error = e
    return _fwd_kernel


def require_flydsl_fwd() -> None:
    """Raise unless the FlyDSL forward can run on this host."""
    global _fwd_host_ok
    if _fwd_host_ok:
        return
    reason = unsupported_hardware_reason()
    if reason is None and _load_fwd_kernel() is None:
        reason = (
            f"flydsl forward import failed "
            f"({type(_fwd_kernel_error).__name__}: {_fwd_kernel_error})"
        )
    if reason is None:
        _fwd_host_ok = True
        logger.info(
            "[hstu-attn] forward kernel: flydsl on %s", _device_arch()
        )
        return
    raise FlyDSLUnavailableError(
        f"the FlyDSL HSTU attention forward is selected but cannot run here: "
        f"{reason}. It is the DEFAULT, so raising is deliberate -- set "
        f"HSTU_ATTN_FWD=triton to run the Triton forward instead."
    )


def flydsl_fwd_selected() -> bool:
    """True when $HSTU_ATTN_FWD asks for FlyDSL. Says nothing about usability."""
    return (
        os.environ.get("HSTU_ATTN_FWD", DEFAULT_FWD_BACKEND).strip().lower()
        or DEFAULT_FWD_BACKEND
    ) == "flydsl"


def flydsl_fwd_enabled() -> bool:
    """True when the FlyDSL forward is selected; raises if selected but unusable."""
    if not flydsl_fwd_selected():
        return False
    require_flydsl_fwd()
    return True


def unsupported_fwd_shape_reason(
    q, k, v, num_targets, max_attn_len, contextual_seq_len,
    num_softmax_heads: int = 0,
) -> Optional[str]:
    """Why the forward kernel cannot serve this call, or None if it can.

    Same specialization set as the backward, minus dO (the forward has none) and
    plus the v head dim, because the forward's out-gemm writes D=128 columns.
    """
    if max_attn_len:
        return f"max_attn_len={max_attn_len} (kernel is plain causal)"
    if contextual_seq_len:
        return f"contextual_seq_len={contextual_seq_len} (kernel assumes 0)"
    if num_softmax_heads:
        return f"num_softmax_heads={num_softmax_heads} (kernel assumes 0)"
    if q.dim() != 3 or q.shape[-1] != SUPPORTED_DIM:
        return f"qk dim {tuple(q.shape)} (kernel is written around dim {SUPPORTED_DIM})"
    if v.shape[-1] != SUPPORTED_DIM:
        return f"v dim {v.shape[-1]} (kernel is written around dim {SUPPORTED_DIM})"
    if any(t.dtype is not torch.bfloat16 for t in (q, k, v)):
        return (
            "dtype "
            f"{tuple(str(t.dtype).replace('torch.', '') for t in (q, k, v))}"
            " (kernel has an mfma path for bf16 only)"
        )
    if num_targets is not None:
        n = _max_num_targets(num_targets)
        if n > 1:
            return f"num_targets max={n} (target-aware masking is identity only at <= 1)"
    return None


def flydsl_hstu_attention_fwd(q, k, v, seq_offsets, N, alpha) -> torch.Tensor:
    """The two-gemm forward: out = silu(alpha * q @ k.T)/N masked causally, @ v.

    Returns a freshly allocated packed [tokens, heads, dim] bf16 tensor, matching
    what `triton_hstu_attention_fwd` returns (`torch.empty_like(v)`, which is
    contiguous because v is a non-dense slice of uvqk).

    Unlike the backward there is no accumulator to zero and no atomics: out[m]
    sums over n <= m only, so the m-outer grid gives each workgroup exclusive
    ownership of its output rows. Every token row inside a sequence is written,
    so the output needs no pre-fill.
    """
    kernel = _load_fwd_kernel()
    heads, dim = q.shape[1], q.shape[2]

    # Read q/k/v where they already live. On the fused preprocess+attention path
    # they are column slices of one uvqk buffer at stride 2048, and packing them
    # would copy the attention input once per tensor per layer per call.
    rs = _common_row_stride((q, k, v), heads, dim)
    if rs is None:
        q, k, v = (t.contiguous() for t in (q, k, v))
    fn = kernel.compile_hstu_fwd(heads, dim, alpha, rs=(rs or 0),
                                 **kernel.SHIPPED)

    key = (rs, heads, dim)
    if key not in _logged_fwd_layouts:
        _logged_fwd_layouts.add(key)
        _maybe_dump_lengths(seq_offsets)
        # tokens and N are logged because the grid is sized by N while the
        # work is set by the real lengths.
        logger.warning(
            "[hstu-attn] FlyDSL forward %s (heads=%d dim=%d tokens=%d N=%d "
            "mean_len=%.0f)",
            f"reading q/k/v in place at row stride {rs}" if rs
            else "packing q/k/v to contiguous (row stride %d)" % (heads * dim),
            heads, dim, q.shape[0], int(N),
            q.shape[0] / max(1, seq_offsets.numel() - 1),
        )

    # The kernel reads seq_offsets through a 32-bit buffer descriptor;
    # production carries it as int64.
    cu = seq_offsets.to(torch.int32)
    batch = seq_offsets.numel() - 1
    # N rather than the real max, for the same reason as the backward: reading
    # the true max costs a device-to-host sync deep in the queue, and N already
    # bounds every sequence. Over-launching only wastes workgroups that exit
    # immediately -- it cannot produce a wrong answer. Note this is also why the
    # kernel-level benchmark must derive N the way production does: N sizes this
    # grid, so a padded N would make the comparison meaningless.
    m_blocks = int((int(N) + fn.BM - 1) // fn.BM)

    out = torch.empty((q.shape[0], heads, dim), dtype=q.dtype, device=q.device)
    # q/k/v may still carry requires_grad, and flydsl's dlpack export refuses
    # those.
    fn(q.detach(), k.detach(), v.detach(), out, cu, batch, m_blocks,
       1.0 / float(N))
    return out


def _log_fwd_shape_fallback(reason: str) -> None:
    """Log the first forward fallback of each distinct reason, then stay quiet."""
    if reason in _logged_fwd_fallbacks:
        return
    _logged_fwd_fallbacks.add(reason)
    logger.warning(
        "[hstu-attn] FlyDSL forward cannot serve this shape (%s); using the "
        "Triton forward for these calls. This is a per-shape fallback, not a "
        "host problem, and is logged once per reason.",
        reason,
    )


def try_flydsl_attention_fwd(
    *, q, k, v, seq_offsets, num_targets, N, alpha, max_attn_len,
    contextual_seq_len, num_softmax_heads,
) -> Optional[torch.Tensor]:
    """The attention forward output from FlyDSL, or None to use Triton's.

    Returns None -- without allocating anything -- when FlyDSL is not selected or
    cannot serve this shape, so the caller runs its normal Triton forward.

    Returns a tensor rather than writing into a caller buffer, because that is
    what `triton_hstu_attention_fwd` does: attention's output is its own
    allocation, not a slice of uvqk.
    """
    # e4m3 q/k/v take try_flydsl_attention_fwd_fp8, not this bf16 kernel.
    if os.environ.get("DLRM_HSTU_FP8_FWD", "0") == "1":
        return None
    if not flydsl_fwd_enabled():
        return None
    reason = unsupported_fwd_shape_reason(
        q, k, v, num_targets, max_attn_len, contextual_seq_len,
        num_softmax_heads,
    )
    if reason is not None:
        _log_fwd_shape_fallback(reason)
        return None
    out = flydsl_hstu_attention_fwd(q, k, v, seq_offsets, N, alpha)

    global _forward_calls
    _forward_calls += 1
    period = int(os.environ.get("HSTU_ATTN_FWD_CHECK", "0") or 0)
    if period > 0 and _forward_calls % period == 0:
        # sort_by_length_indices is a scheduling hint only and does not change
        # the result, so the reference does not need the caller's.
        ref = triton_hstu_attention_fwd(
            N=N, alpha=alpha, q=q, k=k, v=v, seq_offsets=seq_offsets,
            num_targets=num_targets, max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            sort_by_length_indices=None, enable_tma=False,
            num_softmax_heads=0,
        )
        logger.info(
            "[hstu-fwd-check] call=%d shape=%s snr_db out=%.1f",
            _forward_calls, tuple(q.shape), _snr_db(ref, out),
        )
    return out


# ---------------------------------------------------------------------------
# FP8 forward (Track I1)
#
# The FP8 arm's fused preprocess hands attention q/k/v that are ALREADY e4m3
# column slices of one vqk buffer, plus ONE per-tensor scale. Today that goes to
# the Triton e4m3 forward. This is the FlyDSL alternative for the same contract.
#
# Selected by $HSTU_ATTN_FWD=flydsl once $DLRM_HSTU_FP8_FWD is on.
# Compiles fp8_mfma="k128", fp8_v=True, plus the fused-silu epilogue.
# $HSTU_ATTN_FWD=triton with FP8 on keeps the Triton e4m3 forward.
# V is already e4m3 from the same projection as q and k.
# ---------------------------------------------------------------------------

_FP8_FWD_CFG = {"fp8_mfma": "k128", "fp8_v": True}

# epi_fold and mask_split are part of this kernel. mask_split is incompatible
# with the SHIPPED iglp1 hint (the n-loop split trips AMDGPUIGroupLP), so this
# config uses iglp0.
_FP8_FWD_E2_CFG = {"epi_fold": True, "mask_split": True, "sched": "iglp0"}


def _fp8_fwd_config() -> dict:
    """Compile kwargs for the e4m3 FlyDSL forward, including the fused silu epi."""
    cfg = dict(_load_fwd_kernel().SHIPPED)
    cfg.update(_FP8_FWD_CFG)
    cfg.update(_FP8_FWD_E2_CFG)
    return cfg


def flydsl_fp8_fwd_selected() -> bool:
    """True when the FlyDSL e4m3 forward (plus fused-silu epi) is requested.

    Natural selector: ``DLRM_HSTU_FP8_FWD=1`` and ``HSTU_ATTN_FWD=flydsl``.
    Deprecated alias: ``DLRM_HSTU_FP8_FWD_FLYDSL=1`` (same arm; still honored).
    """
    if os.environ.get("DLRM_HSTU_FP8_FWD_FLYDSL", "0") == "1":
        return True
    return (
        os.environ.get("DLRM_HSTU_FP8_FWD", "0") == "1"
        and flydsl_fwd_selected()
    )


def unsupported_fp8_fwd_shape_reason(
    q, k, v, num_targets, max_attn_len, contextual_seq_len,
    num_softmax_heads: int = 0,
) -> Optional[str]:
    """Why the e4m3 forward kernel cannot serve this call, or None if it can.

    The bf16 predicate with the dtype clause inverted: q/k/v must all be e4m3,
    and they must share a row stride, because the fp8 kernel reads the vqk slice
    in place exactly as the bf16 one reads the uvqk slice.
    """
    if max_attn_len:
        return f"max_attn_len={max_attn_len} (kernel is plain causal)"
    if contextual_seq_len:
        return f"contextual_seq_len={contextual_seq_len} (kernel assumes 0)"
    if num_softmax_heads:
        return f"num_softmax_heads={num_softmax_heads} (kernel assumes 0)"
    if q.dim() != 3 or q.shape[-1] != SUPPORTED_DIM:
        return f"qk dim {tuple(q.shape)} (kernel is written around dim {SUPPORTED_DIM})"
    if v.shape[-1] != SUPPORTED_DIM:
        return f"v dim {v.shape[-1]} (kernel is written around dim {SUPPORTED_DIM})"
    fp8 = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    if any(t.dtype not in fp8 for t in (q, k, v)):
        return (
            "dtype "
            f"{tuple(str(t.dtype).replace('torch.', '') for t in (q, k, v))}"
            " (the fp8 path needs e4m3 q, k and v)"
        )
    if num_targets is not None:
        n = _max_num_targets(num_targets)
        if n > 1:
            return f"num_targets max={n} (target-aware masking is identity only at <= 1)"
    return None


def flydsl_hstu_attention_fwd_fp8(
    q, k, v, seq_offsets, N, alpha, qkv_scale: float
) -> torch.Tensor:
    """The e4m3 forward. Returns a packed [tokens, heads, dim] BF16 tensor.

    `qkv_scale` is the single per-tensor scale the fused vqk GEMM quantized with.
    The kernel applies it as qkv_scale**2 on the scores (in f32, on the way into
    the silu) and once more on the out accumulator for V -- not folded into the
    MFMA's scale field, which is E8M0 and could not carry it.

    Output dtype is bf16 and NOT q.dtype: attention's output is a normal
    activation, and `triton_hstu_attention_fwd` returns bf16 here too.
    """
    kernel = _load_fwd_kernel()
    heads, dim = q.shape[1], q.shape[2]

    rs = _common_row_stride((q, k, v), heads, dim)
    if rs is None:
        q, k, v = (t.contiguous() for t in (q, k, v))
    cfg = _fp8_fwd_config()
    fn = kernel.compile_hstu_fwd(heads, dim, alpha, rs=(rs or 0), **cfg)

    key = (rs, heads, dim)
    if key not in _logged_fp8_fwd_layouts:
        _logged_fp8_fwd_layouts.add(key)
        logger.warning(
            "[hstu-attn] FlyDSL FP8 forward %s (heads=%d dim=%d tokens=%d N=%d "
            "mean_len=%.0f scale=%.6g cfg=%s)",
            f"reading q/k/v in place at row stride {rs}" if rs
            else "packing q/k/v to contiguous (row stride %d)" % (heads * dim),
            heads, dim, q.shape[0], int(N),
            q.shape[0] / max(1, seq_offsets.numel() - 1), qkv_scale,
            cfg,
        )

    cu = seq_offsets.to(torch.int32)
    batch = seq_offsets.numel() - 1
    m_blocks = int((int(N) + fn.BM - 1) // fn.BM)
    out = torch.empty(
        (q.shape[0], heads, dim), dtype=torch.bfloat16, device=q.device
    )
    fn(q.detach(), k.detach(), v.detach(), out, cu, batch, m_blocks,
       1.0 / float(N), float(qkv_scale))
    return out


def try_flydsl_attention_fwd_fp8(
    *, q, k, v, seq_offsets, num_targets, N, alpha, max_attn_len,
    contextual_seq_len, num_softmax_heads, qkv_scale,
) -> Optional[torch.Tensor]:
    """The e4m3 attention forward from FlyDSL, or None to use Triton's.

    FAIL-CLOSED. Returns None only when the knob is off. Once it is on, every
    other way this could decline -- host cannot run FlyDSL, kernel import
    failed, shape outside the specialization -- RAISES. A knob whose entire
    purpose is the FlyDSL kernel must not silently run Triton instead.
    """
    if not flydsl_fp8_fwd_selected():
        return None
    if os.environ.get("DLRM_HSTU_FP8_FWD", "0") != "1":
        raise RuntimeError(
            "HSTU_ATTN_FWD=flydsl (or DLRM_HSTU_FP8_FWD_FLYDSL=1) needs "
            "DLRM_HSTU_FP8_FWD=1 to run the e4m3 FlyDSL forward. It is a "
            "replacement for the Triton e4m3 forward inside the FP8 arm, not "
            "a way to turn the FP8 arm on."
        )
    require_flydsl_fwd()
    reason = unsupported_fp8_fwd_shape_reason(
        q, k, v, num_targets, max_attn_len, contextual_seq_len,
        num_softmax_heads,
    )
    if reason is not None:
        raise RuntimeError(
            f"HSTU_ATTN_FWD=flydsl with DLRM_HSTU_FP8_FWD=1 but the FlyDSL "
            f"e4m3 forward cannot serve this shape ({reason}). Refusing to "
            f"fall back silently -- set HSTU_ATTN_FWD=triton for the Triton "
            f"FP8 forward."
        )
    out = flydsl_hstu_attention_fwd_fp8(
        q, k, v, seq_offsets, N, alpha, qkv_scale
    )

    global _fp8_forward_calls
    _fp8_forward_calls += 1
    period = int(os.environ.get("HSTU_ATTN_FWD_CHECK", "0") or 0)
    if period > 0 and _fp8_forward_calls % period == 0:
        ref = triton_hstu_attention_fwd(
            N=N, alpha=alpha, q=q, k=k, v=v, seq_offsets=seq_offsets,
            num_targets=num_targets, max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            sort_by_length_indices=None, enable_tma=False,
            num_softmax_heads=0,
            q_scale=qkv_scale, k_scale=qkv_scale, v_scale=qkv_scale,
        )
        logger.info(
            "[hstu-fwd-fp8-check] call=%d shape=%s snr_db out=%.1f",
            _fp8_forward_calls, tuple(q.shape), _snr_db(ref, out),
        )
    return out


def _triton_bwd(
    dout, q, k, v, seq_offsets, num_targets, N, alpha, max_attn_len,
    contextual_seq_len, sort_by_length_indices, enable_tma,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    triton_hstu_attention_bwd(
        dout=dout, q=q, k=k, v=v, dq=dq, dk=dk, dv=dv,
        seq_offsets=seq_offsets, num_targets=num_targets, N=N, alpha=alpha,
        max_attn_len=max_attn_len, contextual_seq_len=contextual_seq_len,
        sort_by_length_indices=sort_by_length_indices, enable_tma=enable_tma,
        num_softmax_heads=0,
    )
    return dq, dk, dv


def _snr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    """Signal-to-noise in dB of `got` against `ref`; +inf when bit-identical.

    Preferred over a max-abs-diff tolerance because these are whole gradient
    tensors: one denormal-scale element disagreeing in the last ulp would fail an
    elementwise bound while being numerically irrelevant, whereas SNR weighs the
    error against the signal actually present.
    """
    ref32, got32 = ref.float(), got.float()
    noise = torch.linalg.vector_norm(ref32 - got32).item()
    if noise == 0.0:
        return float("inf")
    signal = torch.linalg.vector_norm(ref32).item()
    return 20.0 * torch.log10(torch.tensor(signal / noise)).item()


class _FlyDSLAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, N, alpha, q, k, v, seq_offsets, num_targets, max_attn_len,
        contextual_seq_len, sort_by_length, enable_tma,
    ):
        sort_by_length_indices = None
        if sort_by_length:
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            _, sort_by_length_indices = torch.sort(
                seq_lengths, descending=True, stable=False
            )
        saved = [q, k, v, seq_offsets]
        if num_targets is not None:
            saved.append(num_targets)
        if sort_by_length_indices is not None:
            saved.append(sort_by_length_indices)
        ctx.save_for_backward(*saved)
        ctx.alpha = alpha
        ctx.has_multiple_targets = num_targets is not None
        ctx.max_attn_len = max_attn_len
        ctx.N = N
        ctx.contextual_seq_len = contextual_seq_len
        ctx.sort_by_length = sort_by_length
        ctx.enable_tma = enable_tma
        # $HSTU_ATTN_FWD is honoured here too, so the non-fused dispatch and the
        # fused preprocess+attention path agree on which forward they run.
        out = try_flydsl_attention_fwd(
            q=q, k=k, v=v, seq_offsets=seq_offsets, num_targets=num_targets,
            N=N, alpha=alpha, max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len, num_softmax_heads=0,
        )
        if out is not None:
            return out
        return triton_hstu_attention_fwd(
            N=N, alpha=alpha, q=q, k=k, v=v, seq_offsets=seq_offsets,
            num_targets=num_targets, max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            sort_by_length_indices=sort_by_length_indices,
            enable_tma=enable_tma, num_softmax_heads=0,
        )

    @staticmethod
    def backward(ctx, dout):
        global _backward_calls
        with torch.inference_mode():
            q, k, v, seq_offsets = ctx.saved_tensors[:4]
            idx = 4
            num_targets = None
            if ctx.has_multiple_targets:
                num_targets = ctx.saved_tensors[idx]
                idx += 1
            sort_by_length_indices = (
                ctx.saved_tensors[idx] if ctx.sort_by_length else None
            )

            dout = dout.contiguous()
            # Reaching this Function at all means flydsl was selected, so a host
            # that cannot run it is an error, not a fallback. Only the shape
            # decides between the two backends here.
            require_flydsl()
            reason = unsupported_shape_reason(
                q, k, v, dout, num_targets, ctx.max_attn_len,
                ctx.contextual_seq_len,
            )
            use_flydsl = reason is None
            if not use_flydsl:
                _log_shape_fallback(reason)
            if use_flydsl:
                dq, dk, dv = flydsl_hstu_attention_bwd(
                    dout, q, k, v, seq_offsets, ctx.N, ctx.alpha
                )
            else:
                dq, dk, dv = _triton_bwd(
                    dout, q, k, v, seq_offsets, num_targets, ctx.N, ctx.alpha,
                    ctx.max_attn_len, ctx.contextual_seq_len,
                    sort_by_length_indices, ctx.enable_tma,
                )

            _backward_calls += 1
            if use_flydsl and _check_due(_backward_calls):
                _log_reference_snr(
                    dq, dk, dv,
                    _triton_bwd(
                        dout, q, k, v, seq_offsets, num_targets, ctx.N,
                        ctx.alpha, ctx.max_attn_len, ctx.contextual_seq_len,
                        sort_by_length_indices, ctx.enable_tma,
                    ),
                    q.shape,
                )
        return (None, None, dq, dk, dv, None, None, None, None, None, None)


def _check_due(call_index: int) -> bool:
    """True when this backward should also run Triton as a numerical reference.

    $HSTU_ATTN_BWD_CHECK is the period in backward calls: 0 (default) disables
    the cross-check, N>0 runs it every Nth call. It doubles the cost of the
    backward on the calls it fires, so it is a validation knob, not a production
    one.
    """
    period = int(os.environ.get("HSTU_ATTN_BWD_CHECK", "0") or 0)
    return period > 0 and call_index % period == 0


def _log_reference_snr(dq, dk, dv, reference, shape) -> None:
    ref_dq, ref_dk, ref_dv = reference
    logger.info(
        "[hstu-bwd-check] call=%d shape=%s snr_db dq=%.1f dk=%.1f dv=%.1f",
        _backward_calls,
        tuple(shape),
        _snr_db(ref_dq, dq),
        _snr_db(ref_dk, dk),
        _snr_db(ref_dv, dv),
    )


def hstu_mha(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    sort_by_length: bool = False,
    enable_tma: bool = False,
) -> torch.Tensor:
    """Signature-compatible with triton_hstu_mha."""
    return _FlyDSLAttentionFunction.apply(
        N, alpha, q, k, v, seq_offsets, num_targets, max_attn_len,
        contextual_seq_len, sort_by_length, enable_tma,
    )
