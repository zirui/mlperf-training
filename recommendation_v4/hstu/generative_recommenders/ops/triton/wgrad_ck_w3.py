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

"""Custom FP8 K=tokens out-projection weight gradient ("wgrad_ck" W3).

DLRM_HSTU_FP8_BWD_WGRAD_CK_W3=1 (default off; read once at import). Independent of DLRM_HSTU_FP8_BWD_WGRAD_CK (W1 / W2, wgrad_ck.py): either
can be on without the other.

What: with the FP8 out-proj wgrad in NN mode (DLRM_HSTU_FP8_BWD_OUT_WGRAD=1,
IMPL=nn), HSTUComputeOutputFunction.backward (triton_hstu_linear.py) computes
dw_t [512, 1536] fp32 = scale_y*scale_d * dout_t[:, :M] @ y8 with one hipBLASLt
NN GEMM (ops/hipblaslt_fp8_nn.fp8_wgrad_nn). With this knob it runs
wgrad_tm_ck_w3.cu instead: the W1 split-K-over-XCD-cohorts design with a
256 x 192 tile (256 WGs, NSPLIT=16), each token chunk of both operands fetched
from HBM once per cohort, fp32 partials [16, 512, 1536] in a persistent workspace per
(device, stream) (48 MiB) and a fixed-order reduction kernel that applies
scale_y*scale_d. Kernels in a trace: wgrad_ck_w3_kernel<1536, 192, 16>,
wgrad_ck_w3_reduce<1536, 16>.

How: the library is built by wgrad_ck.build (same hipcc, flags, /tmp cache root
and atomic publish as W1/W2; own cache entry keyed by this source) and bound with
wgrad_ck.bind_entry (same launcher ABI). `prebuild()` builds and loads it before
RUN_START (triton_fp8.prebuild_extensions -> triton_hstu_linear.prebuild_wgrad_ck_w3,
gin DLRM_HSTU_FP8_PREBUILD). Launched on the caller's current stream.

Contract (`run_w3` declines, i.e. returns None and the caller runs hipBLASLt, otherwise):
y8 float8_e4m3fn 2-D, strides (1536, 1), >= M rows (rows >= M are never read);
dout_t float8_e4m3fn [512, ldm], strides (ldm, 1), ldm % 16 == 0,
ldm >= roundup(M, 16), 256 * ldm <= 0x7fffffff; 16-byte aligned data; scale_y /
scale_d single-element fp32 tensors on the same (cuda) device; M >= 1; a gfx950
device. dout_t columns [M, ldm) are padding (as for fp8_wgrad_nn: never part of
the result). The kernel reads tokens [M, min(ldm, roundup(M, 128))) in whole
16-token chunks and multiplies them by zero y8 rows, so `run_w3` zero-fills that
range of dout_t before every launch (production: the dual cast already wrote
zeros there, <= 15 columns at ldm = roundup(M, 16)); a NaN/Inf left in the
padding therefore cannot reach dW.

Guard: the first call per device also computes the production result
(`reference()`, the call site's fp8_wgrad_nn) and requires the custom result to be
all-finite with rel_l2 <= 1e-4 (wgrad_ck.guard_metrics, wgrad_ck._GUARD_TOL), then
prints "wgrad_ck_w3 guard out_projection on cuda:<i>: ... -> armed" and
"ARMED wgrad_ck w3 on cuda:<i>". A mismatch prints "-> rejected"; a mismatch or
ANY exception latches the custom kernel off for the process ("wgrad_ck_w3
DISABLED: <reason>") and the caller runs hipBLASLt for that call and every later
one. W1/W2 and W3 latch independently.

Launch status: a stale HIP error that an earlier call left on this thread (the
last error is sticky) is drained before the padding fill and by the C entry
(returned negated); it is logged once ("stale HIP error ... drained") and is not
a failure (as wgrad_ck).

Numerics (declared): fp32-order. e4m3 x e4m3 products are exact in fp32; only the
fp32 accumulation order differs from hipBLASLt. Deterministic (fixed reduction
order, no atomics): bitwise-reproducible run to run.

Step-tiled dout_t: DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_TILED=1
(default off; read once at import; effective only with DLRM_HSTU_FP8_BWD_WGRAD_CK_W3=1).
The out-proj backward asks `tiled_active()` BEFORE its dual cast; when True (knob on,
not latched, gfx950, library loaded, y8 inside the contract) the cast writes the
token-major e4m3 copy STEP-TILED, a 3-D tensor [ceil(M/128), 512, 128] (step k =
tokens [128k, 128k+128) of all 512 rows, zero tail tokens [M, ceil(M/128)*128);
bwd_fp8.quantize_row_major_dual(t_tiled=True)), and `run_w3` reads it with the entry
wgrad_ck_w3_t (kernel wgrad_ck_w3_t_kernel<1536, 192, 16>: no row stride, so none of
the row-major kernel's stride sensitivity; bitwise-identical output to the row-major
entry). hipBLASLt cannot read that layout: every other consumer (the first-call
guard's reference, hipBLASLt after a decline or the latch, the TN fallback, the SNR
probe) re-lays it out with `untile_to_row_major` (bitwise the 2-D dual-cast bytes).
The tiled entry has its own first-call guard per device (line "... dout_t=step-tiled
... -> armed", then "ARMED wgrad_ck w3 on cuda:<i> (dout_t step-tiled)"); a failure
latches the whole W3 kernel off, after which the cast writes the 2-D layout again.

Robustness. A decline of a step-tiled dout_t turns tiling off for the
rest of the process (`tiled_active` False from then on, logged once): a steady decline would
otherwise re-lay out every call while the cast keeps tiling. The re-laid-out copy for hipBLASLt
comes from the caller's allocator (`untile_to_row_major` `empty`; triton_hstu_linear passes
fp8_tm_alloc.empty_copy, the private pool when active). When that copy is out of memory on the
first-call guard, `reference()` raises wgrad_ck.GuardDeferred: `run_w3` neither arms nor latches
(no verdict, logged once; the next call runs the guard again) and returns the custom result of
this call if it is all-finite (no allocation-free fallback exists at the out-proj site: the bf16
y is not recomputed in FP8 mode and the TN fallback needs the copy too), else latches as a
rejected guard.
"""

import ctypes
import os
import threading
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
from generative_recommenders.ops.triton import wgrad_ck

_REQUESTED: bool = os.environ.get("DLRM_HSTU_FP8_BWD_WGRAD_CK_W3", "0") == "1"
# The dual cast writes dout_t step-tiled for the custom kernel.
_TILED_REQUESTED: bool = (
    os.environ.get("DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_TILED", "0") == "1"
)

_SOURCE: Path = Path(__file__).with_name("wgrad_tm_ck_w3.cu")
_LIB_NAME: str = "libwgrad_ck_w3.so"
_ENTRY: str = "wgrad_ck_w3"
_ENTRY_T: str = "wgrad_ck_w3_t"  # step-tiled dout_t [ceil(M/128), 512, 128]
_NSPLIT: int = 16
_KX: int = 512  # output rows (dout features)
_N: int = 1536  # output columns (y8 features)
_TOKEN_ALIGN: int = 16  # dout_t ldm granularity (16-token DMA chunks)
_KSTEP: int = 128  # tokens per K step: dout_t is read up to roundup(M, 128)
_LDM_MAX: int = 0x7FFFFFFF // 256  # 31-bit buffer range (wgrad_tm_ck_w3.cu WGCK3_LDM_MAX)
_SITE: str = "out_projection"

_lock = threading.Lock()
_fn: Optional[Callable[..., int]] = None
_fn_t: Optional[Callable[..., int]] = None  # wgrad_ck_w3_t (step-tiled dout_t)
_drain: Optional[Callable[[], int]] = None  # wgrad_ck_w3_drain_last_error
_lib: Optional[ctypes.CDLL] = None
_lib_path: Optional[Path] = None
_broken: Optional[str] = None
_override: Optional[bool] = None
_tiled_override: Optional[bool] = None
_armed: set = set()  # device indices that passed the guard
_armed_t: set = set()  # device indices whose step-tiled entry passed its guard
_tiled_ok: set = set()  # device indices tiled_active() checked (gfx950, library loaded)
_workspaces: Dict[Tuple[int, int], torch.Tensor] = {}  # (device index, stream) -> [16, 512, 1536] fp32
_once: set = set()
# Set by `run_w3` on the first decline of a step-tiled dout_t: the cast stops tiling (S3).
_tiled_declined: Optional[str] = None


def log(message: str) -> None:
    # Rank-local, flushed; the out-proj wgrad's own prefix (triton_hstu_linear).
    print(f"[fp8-out-wgrad] {message}", flush=True)


def _log_once(key: str, message: str) -> None:
    if key in _once:
        return
    _once.add(key)
    log(message)


def log_once(key: str, message: str) -> None:
    """One line per key per process (call sites: triton_hstu_linear)."""
    _log_once(key, message)


if _REQUESTED:
    log("requested: wgrad_ck_w3=1")
if _TILED_REQUESTED:
    log(
        "requested: wgrad_ck_w3_tiled=1"
        + ("" if _REQUESTED else " (INERT without DLRM_HSTU_FP8_BWD_WGRAD_CK_W3=1)")
    )


def requested() -> bool:
    """The knob (or the test-only override) asks for the custom W3 kernel."""
    return _REQUESTED if _override is None else _override


def tiled_requested() -> bool:
    """DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_TILED (or its test-only override) AND the
    W3 knob: the out-proj dual cast should write dout_t step-tiled."""
    tiled = _TILED_REQUESTED if _tiled_override is None else _tiled_override
    return tiled and requested()


def set_tiled_requested_for_testing(value: Optional[bool]) -> None:
    """Test-only: override the import-time tiled knob (None restores it)."""
    global _tiled_override
    _tiled_override = value


def enabled() -> bool:
    """Requested and not latched off: the NN-mode out-proj wgrad tries it first."""
    return requested() and _broken is None


def broken() -> Optional[str]:
    return _broken


def set_requested_for_testing(value: Optional[bool]) -> None:
    """Test-only: override the import-time knob (None restores it)."""
    global _override
    _override = value


def reset_for_testing(unload: bool = False) -> None:
    """Test-only: clear the latch, the per-device arming and the once-lines.

    unload=True also forgets the loaded library (the next call builds/loads again,
    e.g. after changing DLRM_HSTU_FP8_BWD_WGRAD_CK_HIPCC / _CACHE).
    """
    global _broken, _fn, _fn_t, _drain, _lib, _lib_path, _tiled_declined
    _broken = None
    _tiled_declined = None
    _armed.clear()
    _armed_t.clear()
    _tiled_ok.clear()
    _once.clear()
    if unload:
        with _lock:
            _fn = None
            _fn_t = None
            _drain = None
            _lib = None
            _lib_path = None


def _disable(reason: str) -> None:
    global _broken
    if _broken is not None:
        return
    _broken = reason
    log(f"wgrad_ck_w3 DISABLED: {reason}")


def _load() -> None:
    global _fn, _fn_t, _drain, _lib, _lib_path
    if _fn is not None:
        return
    with _lock:
        if _fn is not None:
            return
        so = wgrad_ck.build(_SOURCE, _LIB_NAME)
        lib = ctypes.CDLL(str(so))
        fn = wgrad_ck.bind_entry(lib, so, _ENTRY, _NSPLIT)
        fn_t = wgrad_ck.bind_entry(lib, so, _ENTRY_T, _NSPLIT)
        drain = getattr(lib, f"{_ENTRY}_drain_last_error")
        drain.restype = ctypes.c_int
        drain.argtypes = []
        _lib_path = so
        _lib = lib
        _drain = drain
        _fn_t = fn_t
        _fn = fn


def _log_stale(code: int) -> None:
    _log_once(
        f"stale:{code}",
        f"wgrad_ck_w3: stale HIP error {code} drained before a {_ENTRY} launch "
        "(left on this thread by an earlier call, not ours)",
    )


def library_path() -> Optional[Path]:
    return _lib_path


def prebuild() -> bool:
    """Build and load before RUN_START. False (latched off, logged) on failure or
    when not requested; touches no tensor data."""
    if not enabled():
        return False
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("no GPU")
        wgrad_ck._check_arch(torch.device("cuda", torch.cuda.current_device()))
        _load()
    except Exception as error:
        _disable(f"{type(error).__name__}: {error}")
        return False
    return True


def _decline_reason(
    y8: torch.Tensor,
    dout_t: torch.Tensor,
    scale_y: torch.Tensor,
    scale_d: torch.Tensor,
    m: int,
) -> Optional[Tuple[str, str]]:
    """(category, detail) when this call is outside the kernel's contract."""
    if m < 1:
        return "M", f"M={m}"
    if y8.dtype != torch.float8_e4m3fn or dout_t.dtype != torch.float8_e4m3fn:
        return "dtype", f"y8 {y8.dtype} dout_t {dout_t.dtype}"
    if y8.dim() != 2 or dout_t.dim() not in (2, 3):
        return "rank", f"y8 {tuple(y8.shape)} dout_t {tuple(dout_t.shape)}"
    if int(y8.shape[1]) != _N or y8.stride() != (_N, 1) or int(y8.shape[0]) < m:
        return "y8", f"y8 shape {tuple(y8.shape)} strides {y8.stride()} M={m}"
    if dout_t.dim() == 3:
        # Step-tiled [ceil(M/128), 512, 128], contiguous (wgrad_ck_w3_t).
        kt = (m + _KSTEP - 1) // _KSTEP
        # is_contiguous: dense [KT][512][128] (ignores the stride of a KT == 1 dim)
        if tuple(dout_t.shape) != (kt, _KX, _KSTEP) or not dout_t.is_contiguous():
            return (
                "dout_t",
                f"step-tiled dout_t shape {tuple(dout_t.shape)} strides {dout_t.stride()} "
                f"M={m} expected ({kt}, {_KX}, {_KSTEP}) contiguous",
            )
    else:
        ldm = int(dout_t.stride(0))
        mp = (m + _TOKEN_ALIGN - 1) // _TOKEN_ALIGN * _TOKEN_ALIGN
        if (
            int(dout_t.shape[0]) != _KX
            or dout_t.stride(1) != 1
            or int(dout_t.shape[1]) != ldm
            or ldm % _TOKEN_ALIGN != 0
            or ldm < mp
            or ldm > _LDM_MAX
        ):
            return (
                "dout_t",
                f"dout_t shape {tuple(dout_t.shape)} strides {dout_t.stride()} M={m} Mp16={mp}",
            )
    if not (wgrad_ck._aligned(y8) and wgrad_ck._aligned(dout_t)):
        return "align", f"y8 ptr {y8.data_ptr():#x} dout_t ptr {dout_t.data_ptr():#x}"
    device = y8.device
    if device.type != "cuda":
        return "device", f"y8 on {device}"
    for name, t in (("dout_t", dout_t), ("scale_y", scale_y), ("scale_d", scale_d)):
        if t.device != device:
            return "device", f"{name} on {t.device}, y8 on {device}"
    for name, t in (("scale_y", scale_y), ("scale_d", scale_d)):
        if t.dtype != torch.float32 or t.numel() != 1:
            return "scale", f"{name} {t.dtype} numel={t.numel()}"
    return None


def _launch(
    y8: torch.Tensor,
    dout_t: torch.Tensor,
    scale_y: torch.Tensor,
    scale_d: torch.Tensor,
    m: int,
    index: int,
) -> torch.Tensor:
    tiled = dout_t.dim() == 3
    fn, drain = (_fn_t if tiled else _fn), _drain
    assert fn is not None and drain is not None
    # step-tiled: the entry's `ldm` argument is the step stride 512*128 (checked by it)
    ldm = _KX * _KSTEP if tiled else int(dout_t.stride(0))
    with torch.cuda.device(index):
        stream = torch.cuda.current_stream(y8.device).cuda_stream
        key = (index, int(stream))
        workspace = _workspaces.get(key)
        if workspace is None:
            workspace = torch.empty(
                (_NSPLIT, _KX, _N), dtype=torch.float32, device=y8.device
            )
            _workspaces[key] = workspace
        # A stale HIP error an earlier call left on this thread (sticky last error)
        # would fail the torch fill below: drain it first (logged once, not ours).
        stale = int(drain())
        if stale != 0:
            _log_stale(stale)
        # Padding the kernel reads (see the module docstring): zero it.
        if tiled:
            # step-tiled: tokens [M, KT*128) = the last step's columns from M % 128 on
            # (the dual cast already wrote zeros there).
            rem = m - (int(dout_t.shape[0]) - 1) * _KSTEP
            if rem < _KSTEP:
                dout_t[-1, :, rem:].zero_()
        else:
            hi = min(ldm, (m + _KSTEP - 1) // _KSTEP * _KSTEP)
            if hi > m:
                dout_t[:, m:hi].zero_()
        out = torch.empty((_KX, _N), dtype=torch.float32, device=y8.device)
        rc = fn(
            y8.data_ptr(),
            dout_t.data_ptr(),
            ldm,
            workspace.data_ptr(),
            out.data_ptr(),
            scale_y.data_ptr(),
            scale_d.data_ptr(),
            int(m),
            stream,
        )
    if rc < 0:
        # Both kernels launched; the entry drained a stale error that an earlier
        # call had left on this thread: not a launch failure.
        _log_stale(-rc)
    elif rc != 0:
        raise RuntimeError(f"{_ENTRY} launch returned hip error {rc}")
    return out


def run_w3(
    y8: torch.Tensor,
    dout_t: torch.Tensor,
    scale_y: torch.Tensor,
    scale_d: torch.Tensor,
    m: int,
    reference: Callable[[], torch.Tensor],
) -> Optional[torch.Tensor]:
    """dW fp32 [512, 1536] (= fp8_wgrad_nn's dw_t), or None: the caller runs hipBLASLt.

    None when not enabled (knob off or latched), when this call is outside the
    contract (logged once per reason), when the first-call guard rejects
    (latched) or on any exception (latched). `reference()` returns the
    production (hipBLASLt NN) dw_t of this call (or raises), computed on the
    row-major operand (`untile_to_row_major(dout_t, m)` for a step-tiled one); it
    is called only by the first-call guard of each device and layout.

    dout_t is either the row-major [512, ldm] operand (entry wgrad_ck_w3) or the
    step-tiled [ceil(M/128), 512, 128] one (entry wgrad_ck_w3_t; same output
    bits). A None for a step-tiled dout_t means the caller must re-lay it out
    (`untile_to_row_major`) before running hipBLASLt. A declined step-tiled dout_t
    also turns tiling off (`tiled_active`). A wgrad_ck.GuardDeferred from
    `reference()`: nothing armed or latched, and the (unverified) custom result is
    returned when it is all-finite (see the module docstring), else latched.
    """
    global _tiled_declined
    if not enabled():
        return None
    m = int(m)
    why = _decline_reason(y8, dout_t, scale_y, scale_d, m)
    if why is not None:
        _log_once(
            f"decline:{why[0]}",
            f"wgrad_ck_w3 declined {_SITE} ({why[0]}: {why[1]}) -> hipBLASLt for such calls",
        )
        if dout_t.dim() == 3 and _tiled_declined is None:
            _tiled_declined = f"{why[0]}: {why[1]}"
            log(
                f"wgrad_ck_w3 dout_t tiling OFF: a step-tiled dout_t was declined "
                f"({_tiled_declined}) -> the dual cast writes the 2-D layout from now on"
            )
        return None
    device = y8.device
    index = device.index if device.index is not None else torch.cuda.current_device()
    tiled = dout_t.dim() == 3
    armed = _armed_t if tiled else _armed
    layout = " dout_t=step-tiled" if tiled else ""
    out: Optional[torch.Tensor] = None
    try:
        first = index not in armed
        if first:
            wgrad_ck._check_arch(device)
        _load()
        out = _launch(y8, dout_t, scale_y, scale_d, m, index)
        if first:
            rel, finite = wgrad_ck.guard_metrics(out, reference())
            line = (
                f"wgrad_ck_w3 guard {_SITE} on cuda:{index}: N={_N} tokens={m}{layout} "
                f"rel={rel:.3e}"
            )
            if not (finite and rel <= wgrad_ck._GUARD_TOL):
                log(line + (" -> rejected" if finite else " finite=0 -> rejected"))
                _disable(
                    f"guard rejected {_SITE} on cuda:{index}: N={_N} tokens={m}{layout} "
                    f"rel={rel:.3e} finite={int(finite)}"
                )
                return None
            log(line + " -> armed")
            armed.add(index)
            log(
                f"ARMED wgrad_ck w3 on cuda:{index}"
                + (" (dout_t step-tiled)" if tiled else "")
            )
        return out
    except wgrad_ck.GuardDeferred as error:
        # The guard's reference could not be computed for this call only (its
        # re-laid-out copy is out of memory): no verdict, nothing armed.
        finite = out is not None and bool(torch.isfinite(out).all())
        _log_once(
            f"guard_deferred:{index}{layout}",
            f"wgrad_ck_w3 guard {_SITE} on cuda:{index}:{layout} deferred ({error}) -> "
            + (
                "no verdict, nothing latched; this call uses the custom result "
                "(all-finite, unverified), the next call runs the guard again"
                if finite
                else "the custom result is not all-finite -> latched"
            ),
        )
        if not finite:
            _disable(f"guard deferred on cuda:{index}{layout} and the custom result is not all-finite")
            return None
        return out
    except Exception as error:
        _disable(f"{type(error).__name__}: {error}")
        return None


def tiled_active(
    device: torch.device, y8: Optional[torch.Tensor] = None, kx: Optional[int] = None
) -> bool:
    """Should the out-proj dual cast write dout_t step-tiled for this call?

    Asked BEFORE the cast, from the latch state, so the steady state never re-lays
    the operand out: True iff tiled_requested(), enabled() (W3 knob on, not
    latched), a gfx950 device and the library loaded (loads it on the first ask
    per device -- prebuild() normally did; a failure latches W3 off like run_w3)
    and, when given, y8 inside run_w3's contract and the dout width `kx` == 512
    (otherwise run_w3 would decline every call and each one would re-lay out).
    """
    if not (tiled_requested() and enabled()) or _tiled_declined is not None:
        return False
    if kx is not None and int(kx) != _KX:
        return False
    if y8 is not None and (
        y8.dtype != torch.float8_e4m3fn
        or y8.dim() != 2
        or int(y8.shape[0]) < 1
        or int(y8.shape[1]) != _N
        or y8.stride() != (_N, 1)
        or not wgrad_ck._aligned(y8)
        or y8.device != device
    ):
        return False
    if device.type != "cuda":
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index in _tiled_ok and _fn_t is not None:
        return True
    try:
        wgrad_ck._check_arch(device)
        _load()
    except Exception as error:
        _disable(f"{type(error).__name__}: {error}")
        return False
    _tiled_ok.add(index)
    return True


def untile_to_row_major(
    dout_t: torch.Tensor, m: int, empty: Callable[..., torch.Tensor] = torch.empty
) -> torch.Tensor:
    """The row-major token-major operand of a step-tiled dout_t.

    [ceil(M/128), 512, 128] -> [512, roundup(M, 16)] contiguous (the dual cast's
    pad_to=16 layout). A byte copy: tokens < M are the same bytes and tokens
    [M, roundup(M, 16)) are the tiled copy's zero tail, so the result is bitwise
    the 2-D dual cast's output and hipBLASLt reads exactly what it reads without
    tiling. For every consumer other than wgrad_ck_w3_t (the first-call guard's
    reference, hipBLASLt after a decline or the latch, the TN fallback, the SNR
    probe). A 2-D dout_t is returned unchanged. The copy is allocated with
    `empty(shape, dtype=, device=)` (triton_hstu_linear: fp8_tm_alloc.empty_copy).
    """
    if dout_t.dim() != 3:
        return dout_t
    m = int(m)
    kt, kx, ks = (int(s) for s in dout_t.shape)
    if ks != _KSTEP or kt != (m + _KSTEP - 1) // _KSTEP or not dout_t.is_contiguous():
        raise ValueError(
            f"untile_to_row_major: dout_t {tuple(dout_t.shape)} strides {dout_t.stride()} "
            f"is not step-tiled for M={m}"
        )
    mp = (m + _TOKEN_ALIGN - 1) // _TOKEN_ALIGN * _TOKEN_ALIGN
    out = empty((kx, mp), dtype=dout_t.dtype, device=dout_t.device)
    src = dout_t.view(torch.uint8).permute(1, 0, 2)  # [512, KT, 128] view
    dst = out.view(torch.uint8)
    full = mp // _KSTEP
    if full:
        dst[:, : full * _KSTEP].view(kx, full, _KSTEP).copy_(src[:, :full])
    if mp > full * _KSTEP:
        dst[:, full * _KSTEP :].copy_(src[:, full, : mp - full * _KSTEP])
    return out
