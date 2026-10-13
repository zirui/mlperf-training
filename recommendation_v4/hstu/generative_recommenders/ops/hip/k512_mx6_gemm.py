# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0

# pyre-unsafe

"""MXFP6 forward K=512 GEMMs of the HSTU layers and the fused MXFP6 weight shadow, as hipcc-built HIP libraries.

Behind ``DLRM_HSTU_FP8_FWD_MXFP6=1`` (default off; a DECLARED PRECISION CHANGE). The integration adapter
``ops/hip/mx6_fwd.py`` (module ``$DLRM_HSTU_FP8_FWD_MXFP6_GEMM_MODULE``, default this one) calls::

    SUPPORTED_N                       (1024, 1536), plus 512 when the U source (k512_mx6_u.cu) is present
    kernel(n) -> Kernel               build / load the library for N; RuntimeError when unavailable (latched)
    mx6_weight(weight, layout) -> (w6, ws, sb)
    k512_mx6(x6, xs, w6, ws, bias, sa, sb, M, N, out=None) -> out
    prebuild() -> bool                (PrebuildError on a failed build / self-check)

The GEMM kernel takes pre-tiled "MX6-T" operands and uses the scaled MFMA. The U GEMM (N=512, bf16 + bias) is the
library ``k512_mx6_u.cu`` (C ABI ``k512_mx6_u_bf16_bias``), dispatched from here when its source is present.

Math (k512_mx6_gemm.cu header)::

    acc[m, n] = sum_k dequant(x6, xs)[m, k] * dequant(w6, ws)[n, k]       fp32 inside the MFMA, K = 512
    N in (1536, 1024): out e4m3 [M, N] = e4m3_rne(sat448(acc * (sa * sb) + float(bias[n])))
    N = 512 (U):       out bf16 [M, 512] = bf16_rne(acc * (sa * sb) + float(bias[n]))     (k512_mx6_u.cu)

Operands (uint8, all contiguous): x6 [T, 4, 12288] MX6-T and xs [T, 4, 512] MX6-A-v1 with T = ceil(M / 128) (the
LN producer ``ln_store_mx6`` writes both); w6 [N/256, 2, 4, 12288] MX6-T weight items with the k512 output-column
permutation, ws [N/32, 4, 4] = [N/32][q][S] 2-D 32x32 E8M0 bytes (``mx6_weight`` writes both); bias bf16 [N]; sa / sb
one fp32 value each on the device.

WEIGHT SHADOW (``mx6_weight``). ONE kernel launch per refresh turns the uvqk weight [512, C] (fp32 or bf16, any
strides) and a column layout (``((512, 2048),)`` V/Q/K, ``((512, 1024), (1536, 2048))`` targets V/K, ``((0, 512),)``
U; the ``triton_fp8`` layout convention) into

* ``sb`` fp32 [1, 1]: bitwise ``triton_fp8._get_weight_shadow(weight, layout)[1]`` (calibrated_scale with margin 1.0:
  amax * fp32(1/448), amax clamped at 1e-12; NaN if the slice holds a NaN);
* ``w6`` / ``ws``: byte for byte ``tile_W(quant_W(W^T / sb))`` (``mx6_weight_reference``) with ``W^T / sb`` the fp32
  IEEE quotient the e4m3 shadow quantizes (``quantize_with_scale``), so the codes are relative to the FP8 shadow's
  scale_b (C = acc * (sa * sb) + bias).

The result is cached per (weight identity, device, shape, stride, dtype, layout) and served only while the SAME tensor
is alive with the same ``data_ptr()`` and ``_version``: an optimizer step (any in-place update bumps ``_version``) or
a storage swap refreshes it on the next call. A writer that changes the weight without bumping ``_version`` (raw
pointers, ``.data`` writes) leaves this shadow stale exactly like the FP8 shadows; the adapter's SHADOW_AUDIT covers
that class.

Build: ``/opt/rocm/bin/hipcc`` (the k512_fp8_gemm.py flags), once per library per process, into a per-process directory
under ``/tmp/dlrm_hstu_mx6_k512``: ``libk512_mx6_gemm_N{1536,1024}.so`` (``-DHK_N``), ``libk512_mx6_wshadow.so``
(``-DK512_MX6_WSHADOW``) and, when ``k512_mx6_u.cu`` is present next to this file (or named by
``DLRM_HSTU_FP8_FWD_MXFP6_U_SRC``), ``libk512_mx6_u.so``. ``prebuild()`` (before RUN_START) builds them in parallel and
self-checks them; otherwise each builds on first use. A failed build or self-check is latched for the process.
Optional environment: ``DLRM_HSTU_FP8_FWD_MXFP6_GEMM_SRC`` = another ``.cu`` with this file's ABI.

Every launch is asynchronous on ``torch.cuda.current_stream()``; the scales are read on the device; nothing here
synchronizes the host after the build. Operands are validated on the host from metadata only (dtype, shape,
contiguity, alignment, device): a violation raises ValueError before any launch.
"""

from __future__ import annotations

import concurrent.futures
import ctypes
import hashlib
import os
import re
import subprocess
import tempfile
import threading
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

K: int = 512
TILE: int = 128
KSTEPS: int = K // 128
X6_ITEM_BYTES: int = 12288  # one (128-row tile, K step) MX6-T item: HEAD 8 KiB + TAIL 4 KiB
XS_ITEM_BYTES: int = 512  # one (tile, K step) MX6-A-v1 scale block
GEMM_N: Tuple[int, ...] = (1024, 1536)  # built from k512_mx6_gemm.cu (e4m3 + bias epilogue)
U_N: int = 512  # k512_mx6_u.cu (bf16 + bias epilogue)
HIPCC: str = "/opt/rocm/bin/hipcc"
KNOB: str = "DLRM_HSTU_FP8_FWD_MXFP6"
_SRC_ENV: str = KNOB + "_GEMM_SRC"
_U_SRC_ENV: str = KNOB + "_U_SRC"
U_SOURCE_NAME: str = "k512_mx6_u.cu"
U_SYMBOL: str = "k512_mx6_u_bf16_bias"
WSHADOW_SYMBOL: str = "k512_mx6_wshadow"
WSHADOW_P1: int = 64  # phase-1 workgroups of the shadow kernel (k512mx6w::P1; checked against the library)
WSHADOW_KEY: str = "wshadow"
PERM_K512: int = 0  # w6 output-column permutation of the k512 MX6 kernels (VQK, VK; the U library uses the same)
_BUILD_ROOT: Path = Path("/tmp/dlrm_hstu_mx6_k512")
_BUILD_TIMEOUT_S: int = 600
_INT32_MAX: int = 2**31 - 1
_STALE_ERROR_BASE: int = 100000  # rc = 100000 + e: the launch was clean, an EARLIER HIP call left error e
_MAX_SHADOW_N: int = 1 << 20
# fp32(1/448): what aten multiplies by for "x / 448.0" on a CUDA tensor (k512_mx6_gemm.cu uses 1.0f / 448.0f)
INV_FP8_MAX_F32: float = float(torch.tensor(1.0 / 448.0, dtype=torch.float32).item())

_FP8_E4M3: Optional[torch.dtype] = getattr(torch, "float8_e4m3fn", None)
_Layout = Tuple[Tuple[int, int], ...]


class PrebuildError(RuntimeError):
    """prebuild(): a library failed to build, load or self-check (latched; kernel() then raises for it)."""


@dataclass(frozen=True)
class Kernel:
    """One loaded library: a GEMM for N (1536 / 1024 / 512) or the weight shadow (n = 0)."""

    key: object
    n: int
    path: str
    fn: object  # ctypes function pointer
    impl: str
    source: str
    source_sha256: str
    build_s: float
    resources: str


def source_path() -> Path:
    override = os.environ.get(_SRC_ENV)
    return Path(override) if override else Path(__file__).with_name("k512_mx6_gemm.cu")


def u_source_path() -> Optional[Path]:
    """The U GEMM source (``k512_mx6_u.cu`` next to this file, or ``$DLRM_HSTU_FP8_FWD_MXFP6_U_SRC``); None
    when absent (N = 512 is then not in SUPPORTED_N and U stays on the shipped FP8 GEMM)."""
    override = os.environ.get(_U_SRC_ENV)
    path = Path(override) if override else Path(__file__).with_name(U_SOURCE_NAME)
    return path if path.is_file() else None


SUPPORTED_N: Tuple[int, ...] = ((U_N,) if u_source_path() is not None else ()) + GEMM_N

_lock = threading.Lock()  # guards _build_dir and the lock table
_key_locks: Dict[object, threading.Lock] = {}
_kernels: Dict[object, Kernel] = {}
_failures: Dict[object, str] = {}
_build_dir: Optional[Path] = None
_prebuilt: bool = False


def _print(message: str) -> None:
    print(f"[k512-mx6] {message}", flush=True)


def hipcc_command(source: Path, output: Path, defines: Sequence[str] = (), *, extra_flags: Sequence[str] = ()) -> list:
    """The exact hipcc invocation: k512_fp8_gemm.py's flags (iterative-ilp scheduler, -O3, -ffp-contract=off)."""
    return [
        HIPCC,
        str(source),
        "--offload-arch=gfx950",
        "-std=c++20",
        "-w",
        "-O3",
        "-ffp-contract=off",
        "-I/opt/rocm/include/hip",
        *defines,
        "-mllvm",
        "-amdgpu-sched-strategy=iterative-ilp",
        "-shared",
        "-fPIC",
        "-Rpass-analysis=kernel-resource-usage",
        *extra_flags,
        "-o",
        str(output),
    ]


def _process_build_dir() -> Path:
    global _build_dir
    with _lock:
        if _build_dir is None:
            _BUILD_ROOT.mkdir(parents=True, exist_ok=True)
            _build_dir = Path(tempfile.mkdtemp(prefix=f"pid{os.getpid()}_", dir=str(_BUILD_ROOT)))
        return _build_dir


def _key_lock(key: object) -> threading.Lock:
    with _lock:
        lk = _key_locks.get(key)
        if lk is None:
            lk = _key_locks[key] = threading.Lock()
        return lk


def resource_summary(build_log: str) -> str:
    """One entry per kernel from the -Rpass-analysis=kernel-resource-usage remarks (k512_fp8_gemm.py's parser)."""
    per_kernel: Dict[str, Dict[str, str]] = {}
    current = None
    location = r"(?:\s+\S+:\d+:\d+:)?"
    for line in build_log.splitlines():
        m = re.search(r"remark:" + location + r"\s+Function Name:\s+(\S+)", line)
        if m:
            current = m.group(1)
            per_kernel[current] = {}
            continue
        m = re.search(r"remark:" + location + r"\s+([A-Za-z][A-Za-z /\[\]]*?):\s+(\S+)\s+\[-Rpass", line)
        if m and current is not None:
            per_kernel[current][m.group(1).strip()] = m.group(2)
    keep = (("VGPRs", "vgpr"), ("AGPRs", "agpr"), ("VGPRs Spill", "vspill"), ("SGPRs Spill", "sspill"),
            ("ScratchSize [bytes/lane]", "scratch"), ("Occupancy [waves/SIMD]", "occ"), ("LDS Size [bytes/block]", "lds"))
    parts = []
    for name, fields in per_kernel.items():
        m = re.match(r"_Z(\d+)(\w+)", name)
        short = m.group(2)[: int(m.group(1))] if m else name[:40]
        kv = " ".join(f"{label}={fields[k]}" for k, label in keep if k in fields)
        parts.append(f"{short}: {kv}")
    return "; ".join(parts)


def _spec(key: object) -> Tuple[Path, str, list, str, str]:
    """(source, library file name, -D defines, entry symbol, impl symbol) of a library key."""
    if key == WSHADOW_KEY:
        return source_path(), "libk512_mx6_wshadow.so", ["-DK512_MX6_WSHADOW"], WSHADOW_SYMBOL, "k512_mx6_wshadow_impl"
    if key in GEMM_N:
        return (source_path(), f"libk512_mx6_gemm_N{key}.so", [f"-DHK_N={key}", f"-DHK_K={K}"],
                f"k512_mx6_e4m3_bias_N{key}", "k512_mx6_gemm_impl")
    if key == U_N:
        src = u_source_path()
        if src is None:
            raise RuntimeError(
                f"MX6 U library absent: no {U_SOURCE_NAME} next to {Path(__file__).name} and ${_U_SRC_ENV} unset "
                "(U stays on the shipped FP8 GEMM)")
        return src, "libk512_mx6_u.so", [], U_SYMBOL, "k512_mx6_u_impl"
    raise RuntimeError(f"N={key} is not built (supported: {SUPPORTED_N})")


def _build(key: object) -> Tuple[Path, Path, float, str]:
    source, libname, defines, _, _ = _spec(key)
    out_dir = _process_build_dir()
    output = out_dir / libname
    temporary = out_dir / f".{libname}.{os.getpid()}.tmp.so"
    log_path = out_dir / f"build_{libname}.log"
    command = hipcc_command(source, temporary, defines)
    t0 = time.perf_counter()
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                timeout=_BUILD_TIMEOUT_S, check=False)
    except FileNotFoundError as error:
        raise RuntimeError(f"hipcc not found ({HIPCC})") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"hipcc timed out after {_BUILD_TIMEOUT_S}s") from error
    seconds = time.perf_counter() - t0
    log_path.write_text(" ".join(command) + "\n" + result.stdout)
    if result.returncode != 0 or not temporary.exists():
        lines = result.stdout.strip().splitlines()
        errors = [line.strip() for line in lines if "error" in line.lower()]
        tail = " | ".join((errors or lines)[:3])
        raise RuntimeError(f"hipcc rc={result.returncode} for {libname} (log {log_path}): {tail[:400]}")
    os.replace(temporary, output)
    return source, output, seconds, resource_summary(result.stdout)


def _load(key: object) -> Kernel:
    source, path, seconds, resources = _build(key)
    _, _, _, symbol, impl_symbol = _spec(key)
    blob = source.read_bytes()
    library = ctypes.CDLL(str(path))
    fn = getattr(library, symbol)
    fn.restype = ctypes.c_int
    if key == WSHADOW_KEY:
        fn.argtypes = ([ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int]
                       + [ctypes.c_int] * 8 + [ctypes.c_int] + [ctypes.c_void_p] * 5)
        p1_fn = getattr(library, "k512_mx6_wshadow_p1", None)
        if p1_fn is not None:
            p1_fn.restype = ctypes.c_int
            p1_fn.argtypes = []
            if int(p1_fn()) != WSHADOW_P1:
                raise RuntimeError(f"{path}: k512_mx6_wshadow_p1() = {int(p1_fn())} != WSHADOW_P1 = {WSHADOW_P1}")
    else:
        # GEMM e4m3 and the U library share one C ABI: (x6, xs, w6, ws, out, bias, sa, sb, M, stream)
        fn.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int, ctypes.c_void_p]
    impl = "?"
    impl_fn = getattr(library, impl_symbol, None)
    if impl_fn is not None:
        impl_fn.restype = ctypes.c_char_p
        impl_fn.argtypes = []
        impl = (impl_fn() or b"?").decode(errors="replace")
    return Kernel(key=key, n=key if isinstance(key, int) else 0, path=str(path), fn=fn, impl=impl, source=str(source),
                  source_sha256=hashlib.sha256(blob).hexdigest(), build_s=seconds, resources=resources)


def _get(key: object) -> Kernel:
    cached = _kernels.get(key)
    if cached is not None:
        return cached
    with _key_lock(key):  # per library, so prebuild() compiles them in parallel
        cached = _kernels.get(key)
        if cached is not None:
            return cached
        if key in _failures:
            raise RuntimeError(_failures[key])
        try:
            loaded = _load(key)
        except Exception as error:
            what = "weight shadow" if key == WSHADOW_KEY else f"GEMM N={key}"
            _failures[key] = f"K=512 MXFP6 {what} library unavailable: {error}"
            raise RuntimeError(_failures[key]) from error
        _kernels[key] = loaded
        return loaded


def kernel(n: int) -> Kernel:
    """The loaded GEMM library for N (built on first use). RuntimeError if N is not supported or the library cannot
    be built / loaded; the failure is latched for the process."""
    n = int(n)
    if n not in SUPPORTED_N:
        if n == U_N:
            _spec(U_N)  # raises the "U library absent" reason
        raise RuntimeError(f"N={n} is not built (supported: {SUPPORTED_N})")
    return _get(n)


def wshadow_kernel() -> Kernel:
    """The loaded weight-shadow library (built on first use; failure latched)."""
    return _get(WSHADOW_KEY)


def failure(n: object) -> Optional[str]:
    return _failures.get(n)


# ----------------------------------------------------------------------------------------------- GEMM launch
def _stream_handle(device: torch.device) -> int:
    return int(torch.cuda.current_stream(device).cuda_stream)


def _rc_message(rc: int) -> str:
    if rc >= _STALE_ERROR_BASE:
        return f"clean launch, but an EARLIER HIP call left hipError {rc - _STALE_ERROR_BASE} on this thread"
    return f"hipError {rc}"


def ineligible_reason(x6: torch.Tensor, xs: torch.Tensor, w6: torch.Tensor, ws: torch.Tensor, bias: torch.Tensor,
                      sa: torch.Tensor, sb: torch.Tensor, m: int, n: int, out: Optional[torch.Tensor]) -> Optional[str]:
    """None when k512_mx6 can take these operands, else the reason (metadata only; no host sync)."""
    if _FP8_E4M3 is None:
        return "torch has no float8_e4m3fn"
    if not isinstance(m, int) or isinstance(m, bool) or m < 0 or m > _INT32_MAX:
        return f"M={m!r} (needs 0 <= M < 2^31)"
    if n not in SUPPORTED_N:
        return f"N={n} (supported: {SUPPORTED_N})"
    t = (m + TILE - 1) // TILE
    tensors = {"x6": x6, "xs": xs, "w6": w6, "ws": ws, "bias": bias, "sa": sa, "sb": sb}
    for name, ten in tensors.items():
        if not isinstance(ten, torch.Tensor):
            return f"{name} is {type(ten).__name__}, not a tensor"
    for name, ten, numel, align in (("x6", x6, t * KSTEPS * X6_ITEM_BYTES, 16), ("xs", xs, t * KSTEPS * XS_ITEM_BYTES, 8),
                                     ("w6", w6, (n // 256) * 2 * KSTEPS * X6_ITEM_BYTES, 16), ("ws", ws, (n // 32) * 16, 4)):
        if ten.dtype != torch.uint8 or not ten.is_contiguous() or ten.numel() != numel:
            return (f"{name} {ten.dtype} {tuple(ten.shape)} contiguous={ten.is_contiguous()} "
                    f"(needs contiguous uint8 with {numel} bytes for M={m}, N={n})")
        if ten.data_ptr() % align:
            return f"{name} is not {align}-byte aligned"
    if bias.dtype != torch.bfloat16 or tuple(bias.shape) != (n,) or bias.stride() != (1,) or bias.data_ptr() % 16:
        return f"bias {bias.dtype} {tuple(bias.shape)} stride {bias.stride()} (needs contiguous 16-B aligned bf16 [{n}])"
    for name, scale in (("sa", sa), ("sb", sb)):
        if scale.dtype != torch.float32 or scale.numel() != 1:
            return f"{name} {scale.dtype} {tuple(scale.shape)} (needs one fp32 value)"
    if out is not None:
        want = torch.bfloat16 if n == U_N else _FP8_E4M3
        if out.dtype != want or tuple(out.shape) != (m, n) or not out.is_contiguous() or out.data_ptr() % 16:
            return (f"out {out.dtype} {tuple(out.shape)} contiguous={out.is_contiguous()} "
                    f"(needs contiguous 16-B aligned {want} [{m}, {n}])")
    device = x6.device
    if device.type != "cuda" or any(ten.device != device for ten in list(tensors.values()) + ([out] if out is not None else [])):
        return "operands are not all on one CUDA device"
    if device.index is not None and device.index != torch.cuda.current_device():
        return f"operands on {device}, current device cuda:{torch.cuda.current_device()}"
    return None


def launch(kern: Kernel, x6: torch.Tensor, xs: torch.Tensor, w6: torch.Tensor, ws: torch.Tensor, bias: torch.Tensor,
           sa: torch.Tensor, sb: torch.Tensor, m: int, out: torch.Tensor) -> torch.Tensor:
    """Launch on operands that passed ineligible_reason() (stream-ordered on the current stream)."""
    if m == 0:
        return out
    rc = kern.fn(x6.data_ptr(), xs.data_ptr(), w6.data_ptr(), ws.data_ptr(), out.data_ptr(), bias.data_ptr(),
                 sa.data_ptr(), sb.data_ptr(), m, _stream_handle(x6.device))
    if rc != 0:
        raise RuntimeError(f"k512_mx6 N={kern.n} M={m} failed: {_rc_message(rc)} (lib={kern.path})")
    return out


def k512_mx6(x6: torch.Tensor, xs: torch.Tensor, w6: torch.Tensor, ws: torch.Tensor, bias: torch.Tensor,
             sa: torch.Tensor, sb: torch.Tensor, M: int, N: int, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """out = the MXFP6 GEMM of (x6, xs) x (w6, ws) with the shipped epilogue (module docstring), written in place into
    ``out`` (allocated when None: e4m3 [M, N], or bf16 [M, 512] for U) and returned. Asynchronous on the current
    stream. ValueError for operands it cannot take; RuntimeError when the library is unavailable or a launch fails."""
    m, n = int(M), int(N)
    reason = ineligible_reason(x6, xs, w6, ws, bias, sa, sb, m, n, out)
    if reason is not None:
        raise ValueError(f"k512_mx6: {reason}")
    kern = kernel(n)
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16 if n == U_N else _FP8_E4M3, device=x6.device)
    return launch(kern, x6, xs, w6, ws, bias, sa, sb, m, out)


# ----------------------------------------------------------------------------------------------- weight shadow
@dataclass
class _ShadowEntry:
    ref: "weakref.ReferenceType[torch.Tensor]"
    version: int
    data_ptr: int
    w6: torch.Tensor
    ws: torch.Tensor
    sb: torch.Tensor


_shadow_lock = threading.Lock()
_shadows: Dict[Tuple[Any, ...], _ShadowEntry] = {}
_shadow_stats: Dict[str, int] = {"hits": 0, "refreshes": 0}
_workspaces: Dict[Tuple[int, int], torch.Tensor] = {}


def _normalize_layout(weight: torch.Tensor, layout: Optional[Sequence[Sequence[int]]]) -> _Layout:
    """triton_fp8._normalize_layout, plus int-normalization (a layout is a cache key)."""
    if layout is None:
        return ((0, int(weight.shape[1])),)
    norm = tuple((int(a), int(b)) for a, b in layout)
    if not norm:
        raise ValueError("MX6 weight layout cannot be empty")
    for start, end in norm:
        if start < 0 or end <= start or end > weight.shape[1]:
            raise ValueError(f"invalid MX6 weight column range {(start, end)} for shape {tuple(weight.shape)}")
    return norm


def shadow_ineligible_reason(weight: torch.Tensor, layout: _Layout) -> Optional[str]:
    if not isinstance(weight, torch.Tensor) or weight.dim() != 2 or weight.shape[0] != K:
        return f"weight {getattr(weight, 'shape', None)} (needs [{K}, C])"
    if weight.dtype not in (torch.float32, torch.bfloat16):
        return f"weight dtype {weight.dtype} (needs float32 or bfloat16)"
    if weight.device.type != "cuda":
        return f"weight on {weight.device} (needs CUDA)"
    if weight.device.index is not None and weight.device.index != torch.cuda.current_device():
        return f"weight on {weight.device}, current device cuda:{torch.cuda.current_device()}"
    if not 1 <= len(layout) <= 4:
        return f"layout {layout} (1..4 column ranges)"
    n = sum(b - a for a, b in layout)
    if n % 256 or n > _MAX_SHADOW_N:
        return f"layout {layout}: N={n} (needs a multiple of 256, <= {_MAX_SHADOW_N})"
    return None


def _workspace(device: torch.device) -> torch.Tensor:
    """The zero-initialized 64-byte barrier workspace of (device, current stream); the kernel leaves it zero."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (int(index), _stream_handle(device))
    ws = _workspaces.get(key)
    if ws is None:
        ws = torch.zeros((16,), dtype=torch.int32, device=device)  # one-time init, ordered on this stream
        _workspaces[key] = ws
    return ws


def refresh_weight_shadow(weight: torch.Tensor, layout: _Layout, *, perm: int = PERM_K512,
                          out: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One uncached shadow refresh = ONE launch of k512_mx6_wshadow: (w6 [N/256, 2, 4, 12288] uint8, ws [N/32, 4, 4]
    uint8, sb [1, 1] fp32) of `weight`'s column `layout` (module docstring). `out` reuses caller-provided buffers."""
    layout = _normalize_layout(weight, layout)
    reason = shadow_ineligible_reason(weight, layout)
    if reason is not None:
        raise ValueError(f"mx6_weight: {reason}")
    kern = wshadow_kernel()
    n = sum(b - a for a, b in layout)
    device = weight.device
    if out is None:
        w6 = torch.empty((n // 256, 2, KSTEPS, X6_ITEM_BYTES), dtype=torch.uint8, device=device)
        ws = torch.empty((n // 32, 4, 4), dtype=torch.uint8, device=device)
        sb = torch.empty((1, 1), dtype=torch.float32, device=device)
    else:
        w6, ws, sb = out
    ranges = list(layout) + [(0, 0)] * (4 - len(layout))
    args = []
    for a, b in ranges:
        args += [a, b - a]
    wk = _workspace(device)
    rc = kern.fn(weight.data_ptr(), int(weight.dtype == torch.bfloat16), int(weight.stride(0)), int(weight.stride(1)),
                 n, *args, int(perm), w6.data_ptr(), ws.data_ptr(), sb.data_ptr(), wk.data_ptr(),
                 _stream_handle(device))
    if rc != 0:
        raise RuntimeError(f"k512_mx6_wshadow N={n} failed: {_rc_message(rc)} (lib={kern.path})")
    return w6, ws, sb


def mx6_weight(weight: torch.Tensor, layout: Optional[Sequence[Sequence[int]]] = None
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(w6, ws, sb) of `weight`'s column `layout`, cached until `weight` changes (its identity, data_ptr or
    _version); a refresh is one kernel launch on the current stream, no host sync. See the module docstring."""
    layout = _normalize_layout(weight, layout)
    index = weight.device.index
    if index is None:
        index = torch.cuda.current_device() if weight.is_cuda else -1
    key = (id(weight), int(index), tuple(weight.shape), tuple(weight.stride()), weight.dtype, layout, PERM_K512)
    version = int(getattr(weight, "_version", 0))
    with _shadow_lock:
        entry = _shadows.get(key)
        if (entry is not None and entry.ref() is weight and entry.version == version
                and entry.data_ptr == weight.data_ptr()):
            _shadow_stats["hits"] += 1
            return entry.w6, entry.ws, entry.sb
    w6, ws, sb = refresh_weight_shadow(weight, layout)
    with _shadow_lock:
        for k in [k for k, e in _shadows.items() if e.ref() is None]:  # drop entries of freed weights
            del _shadows[k]
        _shadows[key] = _ShadowEntry(ref=weakref.ref(weight), version=version, data_ptr=weight.data_ptr(),
                                     w6=w6, ws=ws, sb=sb)
        _shadow_stats["refreshes"] += 1
    return w6, ws, sb


def clear_shadow_cache() -> None:
    with _shadow_lock:
        _shadows.clear()
        _shadow_stats.update(hits=0, refreshes=0)


def shadow_cache_stats() -> Dict[str, int]:
    with _shadow_lock:
        return {"entries": len(_shadows), **_shadow_stats}


# ----------------------------------------------------------------------------------------------- torch reference
# Torch reference: q_e2m3 / enc_e2m3 / e8m0_exp / pack_rows / quant_W / lds_map / tile_W with an explicit N, plus
# triton_fp8's calibrated_scale chain. Used by prebuild()'s self-check; never on the training path. No MX library.
def _ref_q_e2m3(v: torch.Tensor) -> torch.Tensor:
    a = v.abs()
    step = torch.where(a < 2, torch.full_like(a, 0.125), torch.where(a < 4, torch.full_like(a, 0.25), torch.full_like(a, 0.5)))
    q = torch.clamp(torch.round(a / step) * step, max=7.5)
    return torch.copysign(q, v)


def _ref_enc_e2m3(q: torch.Tensor) -> torch.Tensor:
    a = q.abs()
    s = (q < 0).to(torch.int64)
    e = torch.where(a < 1, torch.zeros_like(a), torch.floor(torch.log2(torch.clamp(a, min=1.0))) + 1)
    m = torch.where(a < 1, a / 0.125, (a / torch.exp2(e - 1) - 1.0) * 8.0)
    return (s << 5) | (e.to(torch.int64) << 3) | torch.round(m).to(torch.int64)


def _ref_e8m0_exp(amax: torch.Tensor, fmax: float = 7.5) -> torch.Tensor:
    e = torch.ceil(torch.log2(torch.clamp(amax, min=1e-38) / fmax))
    for _ in range(2):
        e = torch.where(amax / torch.exp2(e) > fmax, e + 1, e)
        e = torch.where(amax / torch.exp2(e - 1) <= fmax, e - 1, e)
    e = torch.where(amax == 0, torch.zeros_like(e), e)
    return torch.clamp(e, -127, 127)


def _ref_pack_rows(codes: torch.Tensor) -> torch.Tensor:
    r = codes.shape[0]
    c = codes.view(r, 16, 4, 8)
    sh = torch.arange(8, device=codes.device, dtype=torch.int64) * 6
    v = (c << sh).sum(-1)
    by = ((v.unsqueeze(-1) >> (torch.arange(6, device=codes.device, dtype=torch.int64) * 8)) & 255).to(torch.uint8)
    blk = by.view(r, 4, 4, 24)
    head = blk[..., :16].reshape(r, 4, 64)
    tail = blk[..., 16:].reshape(r, 4, 32)
    return torch.cat([head, tail], -1).reshape(r, 384).contiguous()


def _ref_quant_w(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """quant_W: w fp32 [N, 512] -> (MX6 v1 rows [N, 384], ws [N/32][q][S] flat)"""
    n = w.shape[0]
    wb = w.view(n // 32, 32, 16, 32)
    e = _ref_e8m0_exp(wb.abs().amax(dim=(1, 3)))
    q = _ref_q_e2m3(torch.clamp(wb / torch.exp2(e)[:, None, :, None], -7.5, 7.5))
    codes = _ref_enc_e2m3(q).view(n, 16, 32)
    ws = (e.to(torch.int32) + 127).to(torch.uint8).view(n // 32, 4, 4).permute(0, 2, 1).contiguous().view(-1)
    return _ref_pack_rows(codes), ws


def _ref_lds_map(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    d = torch.arange(12288, device=device, dtype=torch.int64)
    b = d % 16
    lane = (d // 16) % 64
    w = (d // 1024) % 4
    i = d // 4096
    j = 4 * i + w
    head = j < 8
    rh = 16 * j + lane // 4
    qh = (lane & 3) ^ ((rh >> 2) & 3)
    rt = 32 * (j - 8) + lane // 2
    pt = (lane & 1) ^ ((rt >> 3) & 1)
    rr = torch.where(head, rh, rt)
    cb = torch.where(head, 16 * qh + b, 64 + 16 * pt + b)
    return rr, cb


def _ref_tile_w(w6_rows: torch.Tensor, n: int) -> torch.Tensor:
    """tile_W with N explicit: MX6 v1 rows [n, 384] -> [n/256, 2, 4, 12288] (k512 output-column permutation)"""
    rr, cb = _ref_lds_map(w6_rows.device)
    perm = lambda h: 64 * (rr >> 5) + 16 * ((rr & 15) >> 2) + 4 * ((rr >> 4) & 1) + (rr & 3) + 8 * h  # noqa: E731
    flat = w6_rows.reshape(-1)
    parts = []
    for tn in range(n // 256):
        for h in (0, 1):
            for s in range(4):
                parts.append(flat[(tn * 256 + perm(h)) * 384 + s * 96 + cb])
    return torch.stack(parts, 0).contiguous().view(n // 256, 2, KSTEPS, X6_ITEM_BYTES)


def mx6_activation_reference(y: torch.Tensor, chunk: int = 65536) -> Tuple[torch.Tensor, torch.Tensor]:
    """quant_A + tile_A + a_scale_native of fp32 y [R, 512] (row 0 = a 128-row tile boundary): (x6 [T, 4, 12288],
    xs [T, 4, 512]), pad rows zero codes / byte 127. The bytes the LN producer must write for y; not used on the
    training path."""
    r = int(y.shape[0])
    t = (r + TILE - 1) // TILE
    rows = torch.zeros((t * TILE, 384), dtype=torch.uint8, device=y.device)
    ee = torch.zeros((t * TILE, 16), dtype=torch.int32, device=y.device)
    for r0 in range(0, r, chunk):
        xb = y[r0:r0 + chunk].reshape(-1, 16, 32)
        e = _ref_e8m0_exp(xb.abs().amax(-1))
        q = _ref_q_e2m3(torch.clamp(xb / torch.exp2(e).unsqueeze(-1), -7.5, 7.5))
        rows[r0:r0 + xb.shape[0]] = _ref_pack_rows(_ref_enc_e2m3(q))
        ee[r0:r0 + xb.shape[0]] = e.to(torch.int32)
    rr, cb = _ref_lds_map(y.device)
    flat = rows.view(t, TILE * 384)
    x6 = torch.stack([flat[:, rr * 384 + s * 96 + cb] for s in range(KSTEPS)], 1).contiguous()
    xs = (ee + 127).to(torch.uint8).view(t, 8, 16, 4, 4).permute(0, 3, 4, 2, 1).contiguous().view(t, KSTEPS, 512)
    return x6, xs


def mx6_weight_reference(weight: torch.Tensor, layout: Optional[Sequence[Sequence[int]]] = None
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The torch-op chain mx6_weight must equal byte for byte: triton_fp8's FP8 shadow scale (_pack_weight_layout +
    calibrated_scale(margin=1.0)), then quant_W(packed^T / scale) + tile_W. Slow; used by the prebuild self-check."""
    layout = _normalize_layout(weight, layout)
    w = weight.detach()
    packed = w[:, layout[0][0]:layout[0][1]] if len(layout) == 1 else torch.cat([w[:, a:b] for a, b in layout], 1)
    amax = packed.abs().amax().float().clamp_min(1.0e-12)
    # calibrated_scale's "amax * 1.0 / 448.0" on a CUDA tensor is amax * fp32(1/448) (aten div_true_kernel_cuda with
    # a CPU scalar; triton_fp8_wshadow.py); written out so the reference also holds on the CPU (true division there)
    sb = (amax * torch.tensor(INV_FP8_MAX_F32, dtype=torch.float32, device=amax.device)).reshape(1, 1)
    wnt = packed.t().contiguous() / sb  # fp32 [N, 512] IEEE quotient (bf16 / fp32 promotes to fp32)
    rows, ws = _ref_quant_w(wnt)
    n = wnt.shape[0]
    return _ref_tile_w(rows, n), ws.view(n // 32, 4, 4), sb


# ----------------------------------------------------------------------------------------------- prebuild
def _ramp_weight(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Deterministic [512, 2048] weight (no RNG): many exponents per block, zero blocks, a 448 peak (sb = 1 exactly
    is NOT forced), negative values."""
    k = torch.arange(K, device=device, dtype=torch.int64).view(-1, 1)
    c = torch.arange(2048, device=device, dtype=torch.int64).view(1, -1)
    mant = ((k * 131 + c * 29) % 211 - 105).to(torch.float32) / 64.0
    expo = (((k // 32) * 7 + (c // 32) * 3) % 13 - 6).to(torch.float32)
    w = mant * torch.exp2(expo) * 0.01
    w[(k // 32 == 3) & (c // 32 == 5)] = 0.0  # all-zero 32x32 blocks (one per layout)
    w[(k // 32 == 9) & (c // 32 == 37)] = 0.0
    w[(k // 32 == 12) & (c // 32 == 50)] = 0.0
    return w.to(dtype).contiguous()


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Same shape and the same bytes (fp32 compared as int32 bits, so -0.0 != +0.0 and NaN payloads count)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype == torch.float32:
        return bool(torch.equal(a.view(torch.int32), b.view(torch.int32)))
    if a.dtype == torch.uint8:
        return bool(torch.equal(a, b))
    return bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))


def _selfcheck_shadow(device: torch.device) -> Optional[str]:
    """The shadow on deterministic ramps, fp32 and bf16, the three production layouts: every byte of w6 / ws and the
    bits of sb vs mx6_weight_reference."""
    for dtype in (torch.float32, torch.bfloat16):
        w = _ramp_weight(device, dtype)
        for layout in (((512, 2048),), ((512, 1024), (1536, 2048)), ((0, 512),)):
            got = refresh_weight_shadow(w, layout)
            ref = mx6_weight_reference(w, layout)
            bad = [nm for nm, g, r in zip(("w6", "ws", "sb"), got, ref) if not bitwise_equal(g, r)]
            if bad:
                return f"weight shadow self-check {dtype} layout {layout}: {bad} differ from the torch reference"
    return None


def _selfcheck_gemm(device: torch.device, n: int, rows: int = 257) -> Optional[str]:
    """GEMM N on zeros (codes 0, E8M0 127) into a 0xFF (e4m3 / bf16 NaN) prefill: every output byte must become +0."""
    u8 = torch.uint8
    t = (rows + TILE - 1) // TILE
    x6 = torch.zeros((t, KSTEPS, X6_ITEM_BYTES), dtype=u8, device=device)
    xs = torch.full((t, KSTEPS, XS_ITEM_BYTES), 127, dtype=u8, device=device)
    one = torch.ones((1, 1), dtype=torch.float32, device=device)
    w6 = torch.zeros(((n // 256), 2, KSTEPS, X6_ITEM_BYTES), dtype=u8, device=device)
    ws = torch.full((n // 32, 4, 4), 127, dtype=u8, device=device)
    bias = torch.zeros((n,), dtype=torch.bfloat16, device=device)
    dtype = torch.bfloat16 if n == U_N else _FP8_E4M3
    width = n * (2 if n == U_N else 1)
    out = torch.full((rows, width), 0xFF, dtype=u8, device=device).view(dtype)
    k512_mx6(x6, xs, w6, ws, bias, one, one, rows, n, out=out)
    torch.cuda.synchronize(device)
    nonzero = int(torch.count_nonzero(out.view(u8)).item())
    if nonzero:
        return f"GEMM N={n} self-check on zeros wrote {nonzero} nonzero bytes ({_kernels[n].path})"
    return None


def prebuild(warm_rows: int = 257) -> bool:
    """Build and load the N=1536 / 1024 GEMMs and the weight shadow (and the U GEMM when present) in parallel before
    RUN_START, then self-check: the shadow on deterministic ramps (fp32 / bf16, the three production layouts) against
    the torch reference, byte for byte, and every GEMM on zeros into a NaN prefill. No data, model state or RNG.
    Returns True when 1536 / 1024 / the shadow are ready (U is optional: its failure is latched and logged, and U stays
    FP8), False when skipped (no gfx950 GPU). Raises PrebuildError otherwise (latched)."""
    global _prebuilt
    del warm_rows  # signature kept compatible with k512_fp8_gemm.prebuild
    if not torch.cuda.is_available():
        _print("prebuild skipped: no GPU")
        return False
    device = torch.device("cuda", torch.cuda.current_device())
    arch = str(getattr(torch.cuda.get_device_properties(device), "gcnArchName", ""))
    if not arch.startswith("gfx950"):
        _print(f"prebuild skipped: arch {arch or 'unknown'} (built for gfx950)")
        return False
    keys = list(GEMM_N) + [WSHADOW_KEY] + ([U_N] if U_N in SUPPORTED_N else [])
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(keys)) as pool:
        futures = {key: pool.submit(_get, key) for key in keys}
    errors, u_error = [], None
    for key, future in futures.items():
        try:
            future.result()
        except Exception as error:
            if key == U_N:
                u_error = str(error)
            else:
                errors.append(f"{key}: {error}")
    if errors:
        _print("prebuild FAILED: " + " | ".join(errors))
        raise PrebuildError("K=512 MXFP6 prebuild failed: " + " | ".join(errors))
    t_build = time.perf_counter() - t0
    reason = _selfcheck_shadow(device)
    for n in GEMM_N:
        reason = reason or _selfcheck_gemm(device, n)
    if reason is not None:
        for key in keys:
            with _key_lock(key):
                _kernels.pop(key, None)
                _failures[key] = f"K=512 MXFP6 self-check failed: {reason}"
        _print("prebuild FAILED: " + reason)
        raise PrebuildError("K=512 MXFP6 prebuild failed: " + reason)
    if U_N in _kernels:
        try:
            u_reason = _selfcheck_gemm(device, U_N)
        except RuntimeError as error:  # a failed U launch (rc != 0); a device fault surfaces in the next sync anyway
            u_reason = f"GEMM N={U_N} self-check raised: {error}"
        if u_reason is not None:  # U is optional: latch it, keep VQK / VK
            with _key_lock(U_N):
                _kernels.pop(U_N, None)
                _failures[U_N] = f"K=512 MXFP6 U self-check failed: {u_reason}"
            u_error = u_reason
    detail = "; ".join(f"{k} {_kernels[k].build_s:.1f}s [{_kernels[k].resources}]" for k in keys if k in _kernels)
    _print(f"prebuilt: N={','.join(str(n) for n in GEMM_N)} + weight shadow"
           + (f" + U N={U_N}" if U_N in _kernels else f" (U N={U_N}: {u_error or 'source absent'})")
           + f" in {t_build:.1f}s (dir {_process_build_dir()}); self-check (shadow ramps vs torch reference, GEMMs on "
           f"zeros) OK; {detail}")
    _prebuilt = True
    return True
