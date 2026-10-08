"""FP8 machinery for the HSTU backward projection GEMMs.

Separate from `triton_fp8.py` on purpose: that module is the promoted forward
arm, and the two should not share a blast radius while this one is younger.

Measured deltas are deliberately NOT quoted here. They are a property of a
build, a library version and a host, so a number inlined in a comment is stale
the first time any of those move, and a reader has no way to tell. They live in
`docs/perf_opt.md` -- the evaluation row for the current figure, the
FP8-backward optimization-log row for the kernel-level accounting -- and the
mechanism and campaign history are in `docs/fp8/fp8.md` (§12).

Two things live here, and they are one thing in practice.

**The cast.** `quantize_with_scale` in the forward module is
`(v / scale).clamp(...).to(fp8)`, an elementwise chain that materializes
full-size intermediates, which is expensive enough to put every unfused site
underwater. A one-pass Triton cast is substantially faster at the same shapes
and bit-exact, and that is what lets `output_projection` pay for itself with no
epilogue work.

**The scale.** The gradient amax *grows* over training by orders of magnitude,
so the forward's calibrate-then-freeze policy drifts badly here. This one
refreshes on a cadence at a margin, and the margin does the work: single-step
spikes are large enough that no cadence tracks a spike.

**The format.** E5M2 for gradients, E4M3 for weights. E4M3 was tried for both
and lost samples-to-AUC against the control, because at this margin it rounds a
large fraction of the gradient to zero and Adam gives a flushed coordinate no
update at all. See the format note below.

Refreshing needs an amax, and a separate reduction pass over a multi-GB tensor
would cost about what the GEMM saves. It does not need one: the cast already
reads every element, so the reduction rides along for a negligible share of it.
The controller is therefore effectively free, and it never reads a device value
back to the host in the steady-state step.

Enable with the single switch, exactly as the forward arm is enabled:

    DLRM_HSTU_FP8_BWD=1                (gin: apply_env_bootstrap.DLRM_HSTU_FP8_BWD)

That switch is the whole arm -- there is nothing else to set. The knobs below
are tunables carrying the values this campaign measured; override one to A/B it
against the arm, not to assemble the arm by hand.

    DLRM_HSTU_FP8_BWD_MARGIN=8         headroom over the observed amax
    DLRM_HSTU_FP8_BWD_REFRESH_CALLS    steps between scale refreshes
    DLRM_HSTU_FP8_BWD_WARMUP_CALLS     backward calls run in BF16 first
    DLRM_HSTU_FP8_BWD_GRAD_FORMAT      e5m2 (default) or e4m3, for A/B only
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

import torch

logger: logging.Logger = logging.getLogger(__name__)

# Weights are E4M3 and gradients are E5M2. That split is the standard FP8
# training recipe and this campaign arrived at it the long way round.
#
# E4M3 was picked for the gradients too at first, on spike data -- evidence that
# only describes the *top* of the range. In a float format the margin costs
# nothing in mantissa, so its entire price is paid at the bottom, and scoring
# the bottom against the gradient histograms shows E4M3 flushing a large
# fraction of this gradient to zero. Adam normalizes per parameter, so a flushed
# coordinate receives no update at all, and paired seeds duly came in behind the
# control on samples-to-AUC.
#
# E5M2 spends one mantissa bit for four decades of range, which moves that
# underflow fraction by orders of magnitude and beats E4M3 at any margin.
# Weights keep E4M3: their range is narrow, so for them the mantissa bit is the
# better buy. The underflow table, with the measured fractions per site, is in
# docs/fp8/fp8.md §12.
_FP8_E4M3: Optional[torch.dtype] = getattr(torch, "float8_e4m3fn", None)
_FP8_E5M2: Optional[torch.dtype] = getattr(torch, "float8_e5m2", None)
_E4M3_MAX: float = 448.0
_E5M2_MAX: float = 57344.0

_GRAD_FORMAT: str = os.environ.get("DLRM_HSTU_FP8_BWD_GRAD_FORMAT", "e5m2").lower()
_GRAD_DTYPE: Optional[torch.dtype] = (
    _FP8_E4M3 if _GRAD_FORMAT == "e4m3" else _FP8_E5M2
)
_GRAD_MAX: float = _E4M3_MAX if _GRAD_FORMAT == "e4m3" else _E5M2_MAX

# The whole arm, from one switch. There is one site and it is meaningless
# without the scale controller, so a separate "enable the controller" knob and a
# separate "enable the site" knob could only ever be set together -- they were
# two spellings of this one. The forward arm keeps its two-tier structure
# because seventeen knobs really can be combined into an unmeasured arm; one
# cannot.
_BWD_ENABLED: bool = os.environ.get("DLRM_HSTU_FP8_BWD", "0") == "1"

# Tunables, not gates. Each has the value this campaign measured and is here to
# be moved for an A/B, not to be assembled into the arm.
_BWD_MARGIN: float = float(os.environ.get("DLRM_HSTU_FP8_BWD_MARGIN", "8"))
# Ship-4 submission logging profile: the per-refresh scale line (two blocking
# .item() reads per site every DLRM_HSTU_FP8_BWD_REFRESH_CALLS steps, mid-backward)
# is a diagnostic, not an MLPerf requirement. Off by default; DLRM_HSTU_FP8_BWD_SCALE_LOG=1
# restores it. The scale and clipped_windows state is unchanged either way.
_BWD_SCALE_LOG: bool = os.environ.get("DLRM_HSTU_FP8_BWD_SCALE_LOG", "0") == "1"
_BWD_REFRESH_CALLS: int = max(
    1, int(os.environ.get("DLRM_HSTU_FP8_BWD_REFRESH_CALLS", "200"))
)
# Long enough to see a representative amax, short enough to be a rounding error
# against a 7.5k step run. The forward arm uses 4 for the same reason.
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
        # These gradients reach 5.0e9 elements. A 32-bit element offset wraps at
        # 2.1e9, and the wgrad probe has already produced silent garbage from
        # exactly that, so the offset is built in 64-bit from the start.
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


def quantize_row_major(
    value: torch.Tensor,
    inv_scale: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    limit: Optional[float] = None,
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
        self.published = False

    def accumulate(self, amax: torch.Tensor) -> None:
        torch.maximum(self.pending, amax, out=self.pending)

    def publish(self) -> None:
        """Turn the window's amax into the scale the next window will use.

        Clamped away from zero because a window whose gradient was entirely zero
        would otherwise publish a scale of zero and make every later value
        infinite. Phase C never saw that, but the failure is unrecoverable and
        the guard is one operation.
        """
        self.scale.copy_(
            (self.pending.clamp_min(1.0e-30) * (_BWD_MARGIN / _GRAD_MAX)).reshape(1)
        )
        self.inv_scale.copy_(1.0 / self.scale)
        self.pending.zero_()
        first = not self.published
        self.published = True
        # One host read every _BWD_REFRESH_CALLS steps, not in the step. Off by
        # default (submission logging profile): the gradient amax growth it shows
        # is still available on demand with DLRM_HSTU_FP8_BWD_SCALE_LOG=1 or
        # through site_diagnostics().
        if not _BWD_SCALE_LOG:
            return
        logger.info(
            "[hstu-fp8-bwd] scale %s: site=%s call=%d scale=%.6e clipped_windows=%d",
            "published" if first else "refreshed",
            self.site,
            self.calls,
            float(self.scale.item()),
            int(self.clipped_windows.item()),
        )

    def note_clipping(self, amax: torch.Tensor) -> None:
        # Anything above scale * FP8_MAX was clamped by the cast. The comparison
        # is exact and costs nothing: both sides are already on device and the
        # amax was computed by the pass that did the clamping.
        self.clipped_windows += (amax > self.scale.reshape(()) * _GRAD_MAX).to(
            torch.int64
        )


_sites: Dict[str, _SiteScale] = {}
# A site name alone is the wrong key: the model stacks several STU layers and
# every one of them calls the same site, so keying on the name puts three
# different layers' gradients behind one scale. That is not a correctness bug --
# each tensor is dequantized with whatever quantized it -- but it forces the
# margin to cover the spread *between* layers on top of the spread within one,
# and it was visible in the first smoke run as three refreshes per rank where
# the cadence predicted one.
#
# The discriminator is the weight's storage, which is only a stable identity
# when the weight is the parameter itself. `STU.forward_targets_only` and
# `STU.cached_forward` hand the backward a per-call `.to(dtype)` copy, whose
# address is whatever the caching allocator returns -- so those minted a scale
# state per new address, spent the warmup in BF16, published, and orphaned it
# (`L63` observed against the three states per rank three layers should give).
# `_has_stable_identity` is what keeps this to the path where the key means
# something; see the leaf-parameter note there.
_layer_index: Dict[Tuple[str, int], int] = {}
_engaged_sites: set = set()
_logged_resolved: bool = False


def _log_resolved_once() -> None:
    """Say what the backward arm resolved to, once, in the run's own log.

    Not deferred to D5. A smoke run of D2 produced a sane loss curve and no
    record whatsoever of whether the FP8 path had executed -- at warmup learning
    rates the loss is identical either way -- which is the exact condition that
    cost the forward campaign a full round of runs.
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

    Refusing the unstable case costs nothing that was being collected anyway --
    those calls never survived warmup to reach FP8. Whether they *should* is a
    separate question that needs its own convergence evidence, which is the
    point: it should be decided, not left to allocator behaviour.
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
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Quantize a backward gradient under the Phase C policy.

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

    quantized, amax = quantize_row_major(gradient, state.inv_scale)
    state.note_clipping(amax)
    state.accumulate(amax)
    if state.calls % _BWD_REFRESH_CALLS == 0:
        state.publish()
    return quantized, state.scale


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


def output_projection_dgrad(
    gradient: torch.Tensor, weight: torch.Tensor
) -> Optional[torch.Tensor]:
    """`dy = dout @ weight^T` in FP8, or None to fall back to BF16.

    The only site in the arm. It is the one that needs no epilogue work: the
    gradient is already row-major, which is the layout `_scaled_mm` wants, so
    the cast is cheap enough here to pay for itself out of one GEMM. The wgrads
    would need both operands re-laid-out along K and do not.

    Current measured delta: `docs/perf_opt.md`, FP8-backward optimization-log
    row. Note it is quoted against the *pinned* BF16 baseline this site actually
    runs, which is a smaller win than against a stock GEMM.
    """
    if not _BWD_ENABLED:
        return None
    if not _has_stable_identity(weight):
        return None
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
    weight_fp8, weight_scale = quantize_weight(weight)
    record_engaged("output_projection_dgrad")
    return torch._scaled_mm(
        gradient_fp8,
        weight_fp8.t(),
        scale_a=gradient_scale,
        scale_b=weight_scale,
        out_dtype=gradient.dtype,
    )


def site_diagnostics() -> Dict[str, Dict[str, float]]:
    """Per-site counters, for logging. Reads device values, so not in the step."""
    report: Dict[str, Dict[str, float]] = {}
    for site, state in _sites.items():
        report[site] = {
            "calls": float(state.calls),
            "scale": float(state.scale.item()) if state.published else 0.0,
            "clipped_windows": float(state.clipped_windows.item()),
        }
    return report


def reset_for_testing() -> None:
    _sites.clear()
    _layer_index.clear()
    _engaged_sites.clear()
