"""dO8 consumer hook, contract v1.2: the C8 attention backward takes the
e4m3 dO and E8 block scales that its d(attn) producer already emitted.

Producer: L11 (`l11_outproj_lnmd_bwd`, 'plus' mode, ops/hip/l11_do8.py).
Imports torch and the stdlib only; nothing here launches a kernel, syncs the
device or allocates device memory.

WHAT IT REPLACES. On the C8 path (DLRM_HSTU_FP8_BWD_FULL, e4m3 q and k, k128
qk MFMA, fused dq16=bf16p, do_blk shape) `flydsl_hstu_attention_bwd` runs
    dO8, E8 = hstu_do_fp8.quantize_do_e4m3_blk(dout, seq_offsets, N)
    (dO8 [T, H, D] float8_e4m3fn, E8 [T, H] uint8 = 127 - e_g,
     groups G(p) = (p // 64, (p % 16) // 8) on sequence-relative p)
which re-reads the whole bf16 dO of each ordinary layer. L11 'plus' holds
every bf16 d(attn) row in registers, still writes the bf16 dx, and also
emits dO8/E8 byte-identical to quantize_do_e4m3_blk(dx).
This module carries those bytes from the producer to the consumer. It never
computes, converts or decodes: a record is taken whole, or the quantizer runs
unchanged.

OBJECTS.
  Do8Site    one per ordinary-layer TRAINING attention forward call (its
             activation input x needs a gradient; eval / no_grad forwards get
             none) whose backward will take the C8 do_blk dO quantization
             (decided in the forward with the backward's own predicates,
             hstu_flydsl_mha.c8_do_blk_planned). Holds the seq_offsets tensor
             that backward uses, N, T, H, D. Tagged on the attention output
             `out` (HSTU_DO8_SITE_ATTR) and kept in the attention autograd
             ctx; `carry_site` copies the tag to the [T, H*D] view that
             HSTUComputeOutputFunction actually receives (ops/hstu_compute.py
             views `out` before handing it over).
  Do8Record  the producer's (dO8, E8) for one backward of one site. Tagged on
             the exact dx object the producer returns to autograd
             (HSTU_DO8_RECORD_ATTR) and on the site. Autograd's view backward
             hands the attention backward a NEW [T, H, D] view of that dx, so
             the attention backward `bind`s the site's pending record to the
             dO object it received; take() then checks that this dO really is
             dx or an autograd view of it.

KNOB. DLRM_HSTU_FP8_BWD_DO8, read per call, default OFF. Gin
binding apply_env_bootstrap.DLRM_HSTU_FP8_BWD_DO8 (dlrm_v4/train/
_env_bootstrap.py, whose _bind_bool writes "1" for True and "0" for False); a
pre-set environment variable wins over it, so DLRM_HSTU_FP8_BWD_DO8=0 in the
environment is the kill switch of a gin-armed run. Only the exact value "1"
(surrounding whitespace ignored) arms, as for the sibling DLRM_HSTU_FP8_*
knobs compared with == "1" (DU_E5M2, DQ_E5M2, DKDV_E5M2, DOBLK_WL);
"true", "on" or "yes" do NOT arm it (fail closed, one line):
  "1"              the ordinary-layer attention forward creates sites on
                   training calls whose backward plans the C8 do_blk dO
                   quantization, and that backward may take a published
                   record;
  absent, "", "0"  off: no sites, no takes. A producer that follows the
                   contract (emit only for a site, and only when
                   take_enabled()) then never calls publish(), so nothing is
                   printed or counted and the hook prints, launches and
                   computes nothing.
                   A publish() call made anyway is refused (kill_switch, or
                   site for a None site), counted and printed;
  anything else    off, with one line.
One knob arms the pair: a producer emits only for a site (and take_enabled()).
The retired switch DLRM_HSTU_FP8_BWD_DO8_TAKE cannot turn anything on, but
any value other than "" or "1" forces off, with one line.

PRODUCER API:  site_of(attn) in the forward; in the backward, emit only
  when a site is present and take_enabled(), and call publish(site, dx, do8,
  e8) after the LAST write to dx, on the stream the consumer will run on (the
  current stream). dx is the [T, H*D] tensor returned as d(attn) (attn's
  shape); do8/e8 are allocated, written and published with that same stream
  current (the record frees them after the consumer kernel is enqueued on it).
CONSUMER API:  take(dout, seq_offsets, N) right before the quantizer;
  discard(dout_or_site, reason) on every path that will not take.

FAIL-CLOSED. take() returns None -- and the caller runs the quantizer -- unless
every check holds: a pending (unconsumed) record is on dout; the knob is on;
the producer's dx is still alive; with root = dx if dx._base is None else
dx._base, dout is dx itself, or dout._base is root with dx's data_ptr and
numel (only such an autograd view shares dx's version counter; an alias with
its own counter -- x.data, t.set_(storage), a dtype view, a gradient hook
that returns g.data -- declines even with dx's data_ptr, numel, shape and
dtype); dx's and dout's _version equal the version recorded at publish;
dtypes bf16 / float8_e4m3fn / uint8; dout is [T, H, D] of the site, dO8 has
dout's numel and is contiguous, E8 is a contiguous (T, H), both 16-byte
aligned; one device, and on GPU the current stream is the publish stream;
seq_offsets is the producer's tensor (data_ptr, _version, numel, device) and
N matches. A declined record is released at once (its T*H*D-byte dO8 is not
pinned), and so is one discarded by a path that will not take. An UNTRACKED
write to dx after publish (a raw kernel, or g.data.mul_() in a hook that
returns None) is invisible to the host and breaks the contract.

ONE-SHOT. The first take consumes the record. Any later take of the same dO --
the dkdv_fp8_sink exact re-run of the attention backward, which re-runs it with
identical arguments and no sinks -- declines ("consumed") and quantizes the
same bf16 dO, i.e. the same bytes again.

'ONLY' MODE (bf16_written=False) IS REFUSED with an error: the
HSTU_ATTN_BWD_CHECK reference, the dkdv_fp8_sink exact re-run and the Triton fallback
all read the bf16 dO, so leaving it unwritten needs a contract v2 with an exact
producer re-run.

LOGS (stdout, flush; per process). Creating a site only bumps the quiet
`sites` counter, and a site that reaches the attention backward without a
pending record (the producer did not publish for that backward: no producer,
or L11 declined or latched on that call) only bumps the quiet `unbound`
counter. Every other counter increment (published, taken, declined,
discarded, refused) is an event. With the knob on, one line
at the first knob read that finds it on (the first ordinary-layer attention
forward, since the site guard reads the knob first):
    [hstu-attn] dO8 registry REQUESTED: rank=R pid=P dev=D DLRM_HSTU_FP8_BWD_DO8=1 ...
The summary line
    [hstu-attn] dO8 registry: rank=R pid=P dev=D events=E sites=S
        published=P taken=T unbound=U declined={reason:n,...}
        discarded={...} refused={...} take=on|off contract=v1.2 [final=1]
is printed at events 1, 2, 4, 8, ... (i.e. from the first publish or refusal
on), at the first occurrence of each declined/discarded/refused reason, right
after the arming line, and once at interpreter exit (final=1) when the
process published or refused a record. One arming line on the first take:
    [hstu-attn] dO8 registry ARMED: rank=R pid=P dev=D producer=... ...
Full arming means unbound=0, i.e. every planned site that reached its
backward got a record.
"""

import atexit
import itertools
import os
import weakref
from typing import Dict, Optional, Tuple

import torch

HSTU_DO8_SITE_ATTR: str = "_hstu_do8_site"
HSTU_DO8_RECORD_ATTR: str = "_hstu_do8_record"
# The dO8 knob: "1" = on; absent, "" or "0" = off (the default).
KNOB_ENV: str = "DLRM_HSTU_FP8_BWD_DO8"
# The retired switch: it can only force off (any value but "" and "1").
TAKE_ENV: str = "DLRM_HSTU_FP8_BWD_DO8_TAKE"
CONTRACT: str = "v1.2"
LOG_PREFIX: str = "[hstu-attn] dO8 registry"
# dO8 / E8 base alignment (bytes). The quantizer's outputs are fresh caching-
# allocator blocks; a producer slice must keep 16-byte (dwordx4) alignment.
ALIGN: int = 16

_FP8 = getattr(torch, "float8_e4m3fn", None)
_tokens = itertools.count(1)

_COUNTS: Dict[str, int] = {"sites": 0, "published": 0, "taken": 0, "unbound": 0}
_REASONS: Dict[str, Dict[str, int]] = {"declined": {}, "discarded": {}, "refused": {}}
_STATE: Dict[str, object] = {"events": 0, "dev": "", "atexit": False}
_SEEN: set = set()
_WARNED: set = set()


# --------------------------------------------------------------------------
# Knob


def _knob() -> Tuple[str, Optional[str]]:
    """(state, value) of the dO8 knob, without side effects. state: "on"
    (DLRM_HSTU_FP8_BWD_DO8=1), "off" (absent, "" or "0"), "invalid" (any other
    value; value = that value) or "retired" (=1, but the retired
    DLRM_HSTU_FP8_BWD_DO8_TAKE holds neither "" nor "1"; value = its value)."""
    raw = os.environ.get(KNOB_ENV)
    if raw is None:
        return "off", None
    v = raw.strip()
    if v in ("", "0"):
        return "off", raw
    if v != "1":
        return "invalid", raw
    old = os.environ.get(TAKE_ENV)
    if old is not None and old.strip() not in ("", "1"):
        return "retired", old
    return "on", raw


def take_enabled() -> bool:
    """The dO8 knob, read per call (the gin env bootstrap may run after
    import): True only for DLRM_HSTU_FP8_BWD_DO8=1. The default (absent, ""
    or "0") is False and prints nothing. An invalid value, or the retired
    DLRM_HSTU_FP8_BWD_DO8_TAKE forcing off, is False with one line; the first
    True prints one REQUESTED line."""
    state, raw = _knob()
    if state == "off":
        return False
    key = ("on",) if state == "on" else (state, raw)
    if key not in _WARNED:
        _WARNED.add(key)
        if state == "on":
            _emit(
                f"{LOG_PREFIX} REQUESTED: {_who()} {KNOB_ENV}=1: the ordinary-"
                "layer attention forward creates dO8 sites on calls whose "
                "backward plans the C8 do_blk dO quantization, and that "
                "backward takes a d(attn) producer's dO8/E8 when every check "
                f"holds (contract {CONTRACT} 'plus'; summary lines start at "
                f"the first publish; kill switch {KNOB_ENV}=0)"
            )
        elif state == "invalid":
            _emit(
                f"{LOG_PREFIX}: {KNOB_ENV}={raw!r} is neither 0 nor 1 -> "
                "treated as 0 (off: no sites, no takes)"
            )
        else:
            _emit(
                f"{LOG_PREFIX}: {TAKE_ENV}={raw!r} (retired v1 kill switch) -> "
                f"off: no sites, no takes; set {KNOB_ENV}=0 or 1 instead"
            )
    return state == "on"


def sites_enabled() -> bool:
    """Whether the attention forward may create sites: the knob itself
    (take_enabled()), so a producer that keys its emission on the site emits
    nothing while the consumer is off."""
    return take_enabled()


# --------------------------------------------------------------------------
# Objects


class Do8Site:
    """Forward-to-backward handle of one ordinary-layer attention call.

    seq_offsets: the exact tensor object the backward will pass to the
    quantizer (the producer builds its 64-token tile list from it); N: the
    attention's max_seq_len (the quantizer's N); T, H, D: out's shape;
    record: the latest Do8Record published on this site (None until then)."""

    __slots__ = ("token", "seq_offsets", "N", "T", "H", "D", "device", "record")

    def __init__(
        self,
        seq_offsets: torch.Tensor,
        N: int,
        T: int,
        H: int,
        D: int,
        device: torch.device,
    ) -> None:
        self.token = next(_tokens)
        self.seq_offsets = seq_offsets
        self.N = int(N)
        self.T = int(T)
        self.H = int(H)
        self.D = int(D)
        self.device = device
        self.record: Optional["Do8Record"] = None


class Do8Record:
    """The producer's dO8/E8 for one backward of one site (see publish).

    Holds dO8 and E8 until it is taken, declined or discarded, then drops
    both. Holds NO strong reference to dx (a weakref only): dx carries this
    record in its __dict__, and a strong back-reference would make a tensor
    reference cycle."""

    __slots__ = (
        "token", "site_token", "producer", "do8", "e8", "dx_ref", "dx_ptr",
        "dx_numel", "version", "device", "stream", "offs_key", "T", "H", "D",
        "N", "state",
    )

    def __init__(
        self,
        site: Do8Site,
        dx: torch.Tensor,
        do8: torch.Tensor,
        e8: torch.Tensor,
        producer: str,
    ) -> None:
        self.token = next(_tokens)
        self.site_token = site.token
        self.producer = str(producer)
        self.do8: Optional[torch.Tensor] = do8
        self.e8: Optional[torch.Tensor] = e8
        self.dx_ref = weakref.ref(dx)
        self.dx_ptr = dx.data_ptr()
        self.dx_numel = dx.numel()
        self.version = dx._version
        self.device = dx.device
        self.stream = _stream_key(dx.device)
        self.offs_key = _offs_key(site.seq_offsets)
        self.T, self.H, self.D, self.N = site.T, site.H, site.D, site.N
        # pending -> consumed | declined | discarded
        self.state = "pending"


def _stream_key(device: torch.device) -> Optional[int]:
    """Handle of the current stream on a GPU device; None elsewhere. A host
    query only (no sync). Tests replace it to emulate a stream change."""
    if device.type != "cuda":
        return None
    return int(torch.cuda.current_stream(device).cuda_stream)


def _offs_key(t: torch.Tensor) -> Tuple[int, int, int, str]:
    return (t.data_ptr(), t._version, t.numel(), str(t.device))


# --------------------------------------------------------------------------
# Forward side (consumer's attention forward, producer's forward)


def new_site(
    out: torch.Tensor, seq_offsets: torch.Tensor, N: int
) -> Optional[Do8Site]:
    """Create the site of one attention forward call and tag `out` with it.

    The caller (the ordinary-layer attention Function's forward) calls this
    only when sites_enabled(), ctx.needs_input_grad[0] (a training forward)
    and hstu_flydsl_mha.c8_do_blk_planned(...) hold, and keeps the returned
    site in its ctx. None (no site) for a malformed call: then no producer
    can find a site and nothing changes. Counted in the quiet `sites`
    counter only: a site is no event and prints nothing."""
    if (
        not isinstance(out, torch.Tensor)
        or out.dim() != 3
        or not isinstance(seq_offsets, torch.Tensor)
        or seq_offsets.dim() != 1
        or seq_offsets.numel() < 1
        or seq_offsets.device != out.device
    ):
        return None
    T, H, D = (int(s) for s in out.shape)
    site = Do8Site(seq_offsets, int(N), T, H, D, out.device)
    setattr(out, HSTU_DO8_SITE_ATTR, site)
    _COUNTS["sites"] += 1
    if not _STATE["dev"]:
        _STATE["dev"] = str(out.device)
    return site


def carry_site(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Return `dst` (a view of the attention output `src`) carrying src's
    site tag, if any. For the caller that views `out` before handing it to
    HSTUComputeOutputFunction. Anything that is not a plain tensor pair (an
    fx Proxy during tracing) passes through untouched."""
    if isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor):
        site = getattr(src, HSTU_DO8_SITE_ATTR, None)
        if isinstance(site, Do8Site):
            setattr(dst, HSTU_DO8_SITE_ATTR, site)
    return dst


def site_of(attn: torch.Tensor) -> Optional[Do8Site]:
    """PRODUCER: the site tagged on the `attn` argument of
    HSTUComputeOutputFunction.forward, or None (no emission for this call:
    the targets-only layer, a non-C8 call, the knob off, or any other
    composition). Call it in the forward and keep the result in ctx."""
    if not isinstance(attn, torch.Tensor):
        return None
    site = getattr(attn, HSTU_DO8_SITE_ATTR, None)
    if not isinstance(site, Do8Site):
        return None
    if attn.numel() != site.T * site.H * site.D or attn.device != site.device:
        return None
    return site


# --------------------------------------------------------------------------
# Producer backward


def _publish_check(
    site: Do8Site, dx: torch.Tensor, do8: torch.Tensor, e8: torch.Tensor
) -> Optional[str]:
    if not isinstance(site, Do8Site):
        return "site"
    if not take_enabled():
        return "kill_switch"
    if not all(isinstance(t, torch.Tensor) for t in (dx, do8, e8)):
        return "type"
    if (
        dx.dtype is not torch.bfloat16
        or _FP8 is None
        or do8.dtype is not _FP8
        or e8.dtype is not torch.uint8
    ):
        return "dtype"
    n = site.T * site.H * site.D
    # dx is the d(attn) tensor itself, i.e. attn's [T, H*D] shape (a
    # 3-D gradient for the 2-D attn is rejected by autograd anyway).
    if (
        tuple(dx.shape) != (site.T, site.H * site.D)
        or do8.numel() != n
        or tuple(e8.shape) != (site.T, site.H)
    ):
        return "shape"
    if not (dx.is_contiguous() and do8.is_contiguous() and e8.is_contiguous()):
        return "layout"
    if do8.data_ptr() % ALIGN or e8.data_ptr() % ALIGN:
        return "align"
    if not (dx.device == do8.device == e8.device == site.device):
        return "device"
    return None


def publish(
    site: Do8Site,
    dx: torch.Tensor,
    do8: torch.Tensor,
    e8: torch.Tensor,
    producer: str = "L11",
    bf16_written: bool = True,
) -> Optional[Do8Record]:
    """PRODUCER: hand this backward's dO8/E8 to the C8 attention backward.

    dx  the bf16 d(attn) the producer returns to autograd -- the exact
        object, shape (T, H*D) == attn.shape (anything else: refused
        "shape"); contiguous; ALREADY fully written (publish records
        dx._version, and any later tracked write makes the consumer decline;
        an untracked late write breaks the contract).
    do8 float8_e4m3fn with T*H*D elements ([T, H*D] or [T, H, D]),
        contiguous, == the bytes of quantize_do_e4m3_blk(dx.view(T, H, D),
        site.seq_offsets, site.N)[0] on every row the quantizer covers, i.e.
        computed from the bf16 values stored in dx (not from wider
        intermediates).
    e8  uint8 (T, H), contiguous, == quantize_do_e4m3_blk(...)[1] likewise.
        dO8 and E8 start 16-byte aligned (fresh allocations always do).
    Allocate do8/e8, launch the emission and call this with
    torch.cuda.current_stream(dx.device) current: the record stores that
    stream, take() requires it, and the consumer's free of do8/e8 is ordered
    on it (a block allocated under another stream could be recycled there
    while the consumer kernel still reads it).

    Returns the record (tagged on dx and on the site), or None when refused
    (counted in refused{reason}; nothing is tagged, the consumer quantizes the
    bf16 dx). bf16_written=False ('only') raises ValueError."""
    if not bf16_written:
        _bump_reason("refused", "only", getattr(site, "device", None))
        raise ValueError(
            f"hstu_do8_registry contract {CONTRACT} accepts 'plus' only "
            "(bf16_written=True): the HSTU_ATTN_BWD_CHECK reference, the "
            "dkdv_fp8_sink exact re-run and the Triton fallback all read the bf16 "
            "dO, so an unwritten bf16 dx needs contract v2 (decide before "
            "emission + an exact producer re-run in bf16 mode)."
        )
    why = _publish_check(site, dx, do8, e8)
    if why is not None:
        _bump_reason("refused", why, getattr(dx, "device", None))
        return None
    old = site.record
    if old is not None and old.state == "pending":
        _release(old, "discarded", "superseded")
    rec = Do8Record(site, dx, do8, e8, producer)
    setattr(dx, HSTU_DO8_RECORD_ATTR, rec)
    site.record = rec
    _bump("published", dx.device)
    return rec


# --------------------------------------------------------------------------
# Consumer backward


def bind(site: Optional[Do8Site], dout: torch.Tensor) -> bool:
    """CONSUMER (attention Function backward, before the attention backward
    call): tag the dO object it received with the site's pending record.
    Autograd delivers a fresh view of the producer's dx (the [T, H*D] ->
    [T, H, D] view backward), so the producer's own tag is not on it; take()
    re-checks that this dO is dx or an autograd view of it.

    A site that arrives here without a pending record (the producer did
    not publish for this backward) bumps the quiet `unbound` counter; no
    event, no line, but every summary line carries it, so the arming audit
    sees partial arming. Only training forwards create sites, and only a
    backward calls bind(), so the count is exact."""
    if not isinstance(site, Do8Site) or not isinstance(dout, torch.Tensor):
        return False
    rec = site.record
    if rec is None or rec.state != "pending":
        _COUNTS["unbound"] += 1
        return False
    if getattr(dout, HSTU_DO8_RECORD_ATTR, None) is not rec:
        setattr(dout, HSTU_DO8_RECORD_ATTR, rec)
    return True


def _take_check(
    rec: Do8Record,
    dout: torch.Tensor,
    seq_offsets: torch.Tensor,
    N: Optional[int],
) -> Optional[str]:
    if rec.state == "consumed":
        return "consumed"
    if rec.state != "pending" or rec.do8 is None or rec.e8 is None:
        return "released"
    if not take_enabled():
        return "kill_switch"
    prod = rec.dx_ref()
    if prod is None:
        return "producer_freed"
    # Identity: dout must be dx itself or an autograd view of dx's root
    # tensor. Only those share dx's version counter, which is what makes the
    # version clause below see every tracked write. An alias with its own
    # counter (x.data, t.set_(storage), a dtype view, a gradient hook that
    # returns g.data) declines here even when its data_ptr, numel, shape and
    # dtype -- and, by coincidence, its _version number -- match dx's.
    root = prod if prod._base is None else prod._base
    if prod.data_ptr() != rec.dx_ptr or (
        dout is not prod
        and (
            dout._base is not root
            or dout.data_ptr() != rec.dx_ptr
            or dout.numel() != rec.dx_numel
        )
    ):
        return "storage"
    if prod._version != rec.version or dout._version != rec.version:
        return "version"
    if (
        dout.dtype is not torch.bfloat16
        or rec.do8.dtype is not _FP8
        or rec.e8.dtype is not torch.uint8
    ):
        return "dtype"
    if (
        dout.dim() != 3
        or tuple(dout.shape) != (rec.T, rec.H, rec.D)
        or rec.do8.numel() != dout.numel()
        or tuple(rec.e8.shape) != (rec.T, rec.H)
    ):
        return "shape"
    if not (
        dout.is_contiguous() and rec.do8.is_contiguous() and rec.e8.is_contiguous()
    ):
        return "layout"
    if rec.do8.data_ptr() % ALIGN or rec.e8.data_ptr() % ALIGN:
        return "align"
    if not (dout.device == rec.device == rec.do8.device == rec.e8.device):
        return "device"
    if _stream_key(dout.device) != rec.stream:
        return "stream"
    if not isinstance(seq_offsets, torch.Tensor) or _offs_key(seq_offsets) != rec.offs_key:
        return "seq_offsets"
    if N is not None and int(N) != rec.N:
        return "n"
    return None


def take(
    dout: torch.Tensor, seq_offsets: torch.Tensor, N: Optional[int] = None
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """CONSUMER (C8 branch of flydsl_hstu_attention_bwd, in place of
    quantize_do_e4m3_blk): (dO8 [T, H, D] float8_e4m3fn, E8 [T, H] uint8) of
    the record on `dout`, or None (no record, or any failed check: declined,
    counted, record released; the caller runs the quantizer). One-shot."""
    rec = getattr(dout, HSTU_DO8_RECORD_ATTR, None) if isinstance(
        dout, torch.Tensor) else None
    if not isinstance(rec, Do8Record):
        return None
    why = _take_check(rec, dout, seq_offsets, N)
    if why is not None:
        if rec.state == "pending":
            _release(rec, "declined", why)
        else:
            _bump_reason("declined", why, rec.device)
        return None
    do8, e8 = rec.do8, rec.e8
    rec.do8 = None
    rec.e8 = None
    rec.state = "consumed"
    shape = (rec.T, rec.H, rec.D)
    if tuple(do8.shape) != shape:
        do8 = do8.view(shape)
    first = _COUNTS["taken"] == 0
    _bump("taken", rec.device, quiet=first)
    if first:
        _emit(
            f"{LOG_PREFIX} ARMED: {_who()} producer={rec.producer} first take: "
            f"dO8 [T, H, D] float8_e4m3fn + E8 [T, H] uint8 handed to the C8 "
            f"attention backward instead of quantize_do_e4m3_blk (T={rec.T} "
            f"H={rec.H} D={rec.D} N={rec.N}; contract {CONTRACT} 'plus'; kill "
            f"switch {KNOB_ENV}=0)"
        )
        _emit(summary_line())
    return do8, e8


def discard(dout_or_site: object, reason: str) -> bool:
    """CONSUMER: release a pending record that this path will not take (a
    non-C8 call, a FlyDSL early return, the attention backward's safety net),
    so its dO8/E8 are not pinned. No-op (False) without a pending record."""
    if isinstance(dout_or_site, Do8Site):
        rec = dout_or_site.record
    elif isinstance(dout_or_site, torch.Tensor):
        rec = getattr(dout_or_site, HSTU_DO8_RECORD_ATTR, None)
    else:
        rec = None
    if not isinstance(rec, Do8Record) or rec.state != "pending":
        return False
    _release(rec, "discarded", str(reason))
    return True


def _release(rec: Do8Record, kind: str, reason: str) -> None:
    rec.do8 = None
    rec.e8 = None
    rec.state = kind
    _bump_reason(kind, reason, rec.device)


# --------------------------------------------------------------------------
# Counters and log lines


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:  # noqa: BLE001 - a log field must never raise
        pass
    raw = os.environ.get("RANK", "").strip()
    return int(raw) if raw.lstrip("-").isdigit() else -1


def _who() -> str:
    return f"rank={_rank()} pid={os.getpid()} dev={_STATE['dev'] or '?'}"


def _fmt(d: Dict[str, int]) -> str:
    return "{" + ",".join(f"{k}:{v}" for k, v in sorted(d.items())) + "}"


def summary_line(final: bool = False) -> str:
    """The '[hstu-attn] dO8 registry:' counter line of this process."""
    return (
        f"{LOG_PREFIX}: {_who()} events={_STATE['events']} "
        f"sites={_COUNTS['sites']} published={_COUNTS['published']} "
        f"taken={_COUNTS['taken']} unbound={_COUNTS['unbound']} "
        f"declined={_fmt(_REASONS['declined'])} "
        f"discarded={_fmt(_REASONS['discarded'])} "
        f"refused={_fmt(_REASONS['refused'])} "
        f"take={'on' if _knob()[0] == 'on' else 'off'} contract={CONTRACT}"
        + (" final=1" if final else "")
    )


def _emit(line: str) -> None:
    try:
        print(line, flush=True)
    except Exception:  # noqa: BLE001 - closed stdout at exit, etc.
        pass


def _final() -> None:
    # Only a process that published or refused a record prints the final
    # line (a site alone is quiet).
    if _COUNTS["published"] + sum(_REASONS["refused"].values()) > 0:
        _emit(summary_line(final=True))


def _event(device: Optional[torch.device], force: bool, quiet: bool) -> None:
    if device is not None:
        _STATE["dev"] = str(device)
    n = int(_STATE["events"]) + 1
    _STATE["events"] = n
    if not _STATE["atexit"]:
        _STATE["atexit"] = True
        atexit.register(_final)
    if not quiet and (force or (n & (n - 1)) == 0):
        _emit(summary_line())


def _bump(key: str, device: Optional[torch.device], quiet: bool = False) -> None:
    _COUNTS[key] += 1
    _event(device, False, quiet)


def _bump_reason(kind: str, reason: str, device: Optional[torch.device]) -> None:
    d = _REASONS[kind]
    d[reason] = d.get(reason, 0) + 1
    new = (kind, reason) not in _SEEN
    _SEEN.add((kind, reason))
    _event(device, new, False)


def stats() -> Dict[str, object]:
    """Snapshot of this process's counters."""
    return {
        "sites": _COUNTS["sites"],
        "published": _COUNTS["published"],
        "taken": _COUNTS["taken"],
        "unbound": _COUNTS["unbound"],
        "declined": dict(_REASONS["declined"]),
        "discarded": dict(_REASONS["discarded"]),
        "refused": dict(_REASONS["refused"]),
        "events": int(_STATE["events"]),
    }


def reset_for_testing() -> None:
    """Zero the counters and the once-per-reason / REQUESTED / arming state."""
    for k in _COUNTS:
        _COUNTS[k] = 0
    for d in _REASONS.values():
        d.clear()
    _STATE["events"] = 0
    _STATE["dev"] = ""
    _SEEN.clear()
    _WARNED.clear()
