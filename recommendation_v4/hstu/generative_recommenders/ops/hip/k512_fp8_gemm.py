# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0

# pyre-unsafe

"""K=512 FP8 forward projection GEMM (HSTU V/Q/K and targets V/K) as a hipcc-built HIP library.

Behind ``DLRM_HSTU_FP8_FWD_K512=1``, ``triton_fp8.scaled_mm_fp8_out_preq`` calls this
instead of the hipBLASLt GEMM it runs in the ``input_scale`` backend mode::

    torch._scaled_mm(A, Bw.t(), scale_a=sa, scale_b=sb, bias=bias_bf16,
                     out_dtype=torch.float8_e4m3fn, use_fast_accum=False)

    C[M, N] e4m3 = e4m3_rne(sat448(acc * (sa * sb) + float(bias[n])))
    acc = sum_k A[m, k] * Bw[n, k] in fp32; K = 512; N in {1024, 1536}

The replacement is bit-identical to that call on hipBLASLt's data-parallel tiles.
At in-model M hipBLASLt runs a persistent stream-K grid whose tail tiles add split-K
partial sums in fp32 through its workspace, so an order-sensitive row there can round
differently: an fp32 summation-order difference.

C ABI of the library (``k512_fp8_gemm.cu``; one source compiled per N with ``-DHK_N=N``)::

    int k512_fp8_gemm_e4m3_bias_N{N}(const void* A,     /* e4m3 [M, 512] row-major */
                                     const void* B,     /* e4m3 [N, 512] row-major */
                                     void* C,           /* e4m3 [M, N] row-major, rows < M */
                                     const void* bias,  /* bf16 [N] */
                                     const float* sa,   /* fp32 device scalar */
                                     const float* sb,   /* fp32 device scalar */
                                     int M,             /* >= 1, any value */
                                     void* stream);     /* hipStream_t; returns hipGetLastError() */
    const char* k512_fp8_gemm_impl(void);               /* optional implementation tag */

The in-tree ``k512_fp8_gemm.cu`` is a self-contained HIP kernel (raw intrinsics, no
library headers):
- two resident 4-wave workgroups per CU, so one WG's epilogue overlaps the other's MFMA;
- a 128x256 tile and a 5-slot 80 KiB LDS ring fed by LDS-DMA loads;
- operand-swapped ``v_mfma_f32_16x16x128_f8f6f4`` with the same per-element K order as hipBLASLt;
- a packed mul-then-add epilogue whose fp8 converts saturate under MODE.FP16_OVFL, so it needs
  no clamp;
- ``buffer_store_dwordx4`` stores, with M-tail rows dropped by the buffer range;
- XCD-aware tile order.

Build: ``/opt/rocm/bin/hipcc``, once per N per process, into a per-process
directory under ``/tmp/dlrm_hstu_fp8_k512``. The build runs from ``prebuild()``
(``triton_fp8.prebuild_extensions``, before MLPerf RUN_START), or lazily on the
first call. A failed build is latched, so it is not retried on every call.
Optional environment:

* ``DLRM_HSTU_FP8_FWD_K512_SRC``: another ``.cu`` with the same ABI, used instead
  of the in-tree source.

Every launch is asynchronous on ``torch.cuda.current_stream()``. The scales are
read on the device, so there is no host sync.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch

K: int = 512
SUPPORTED_N: Tuple[int, ...] = (1024, 1536)
HIPCC: str = "/opt/rocm/bin/hipcc"
KNOB: str = "DLRM_HSTU_FP8_FWD_K512"
_SRC_ENV: str = "DLRM_HSTU_FP8_FWD_K512_SRC"
_BUILD_ROOT: Path = Path("/tmp/dlrm_hstu_fp8_k512")
_BUILD_TIMEOUT_S: int = 600
_INT32_MAX: int = 2**31 - 1

_FP8_E4M3: Optional[torch.dtype] = getattr(torch, "float8_e4m3fn", None)


class PrebuildError(RuntimeError):
    """prebuild(): a library failed to build, load or self-check (latched; the
    forward then declines that N). A device fault is not wrapped in this."""


@dataclass(frozen=True)
class Kernel:
    """One loaded library (one N)."""

    n: int
    path: str
    fn: object  # ctypes function pointer
    impl: str
    source: str
    source_sha256: str
    build_s: float
    resources: str


_lock = threading.Lock()  # guards _build_dir
_n_locks: Dict[int, threading.Lock] = {n: threading.Lock() for n in SUPPORTED_N}
_kernels: Dict[int, Kernel] = {}
_failures: Dict[int, str] = {}
_build_dir: Optional[Path] = None
_armed_logged: set = set()
_decline_logged: set = set()


def _print(message: str) -> None:
    print(f"[k512-fwd] {message}", flush=True)


def source_path() -> Path:
    override = os.environ.get(_SRC_ENV)
    return Path(override) if override else Path(__file__).with_name("k512_fp8_gemm.cu")


def entry_symbol(n: int) -> str:
    return f"k512_fp8_gemm_e4m3_bias_N{n}"


def hipcc_command(
    n: int,
    source: Path,
    output: Path,
    *,
    extra_flags: Sequence[str] = (),
) -> list:
    """The exact hipcc invocation: the flags the kernel was gated with
    (iterative-ilp scheduler, -O3) plus -ffp-contract=off."""

    return [
        HIPCC,
        str(source),
        "--offload-arch=gfx950",
        "-std=c++20",
        "-w",
        "-O3",
        # Bit parity with hipBLASLt needs acc*s and +bias as two roundings; the
        # source also says so with '#pragma clang fp contract(off)'.
        "-ffp-contract=off",
        "-I/opt/rocm/include/hip",
        f"-DHK_N={n}",
        f"-DHK_K={K}",
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
            _build_dir = Path(
                tempfile.mkdtemp(prefix=f"pid{os.getpid()}_", dir=str(_BUILD_ROOT))
            )
        return _build_dir


def resource_summary(build_log: str) -> str:
    """Condense the hipcc -Rpass-analysis=kernel-resource-usage remarks to one
    entry per kernel: VGPRs, AGPRs, spills, scratch, occupancy, LDS."""

    per_kernel: Dict[str, Dict[str, str]] = {}
    current = None
    # "<loc>: remark: Function Name: X" or, with --save-temps, "remark: <loc>: ..."
    location = r"(?:\s+\S+:\d+:\d+:)?"
    for line in build_log.splitlines():
        m = re.search(r"remark:" + location + r"\s+Function Name:\s+(\S+)", line)
        if m:
            current = m.group(1)
            per_kernel[current] = {}
            continue
        m = re.search(
            r"remark:" + location + r"\s+([A-Za-z][A-Za-z /\[\]]*?):\s+(\S+)\s+\[-Rpass",
            line,
        )
        if m and current is not None:
            per_kernel[current][m.group(1).strip()] = m.group(2)
    keep = (
        ("VGPRs", "vgpr"),
        ("AGPRs", "agpr"),
        ("VGPRs Spill", "vspill"),
        ("SGPRs Spill", "sspill"),
        ("ScratchSize [bytes/lane]", "scratch"),
        ("Occupancy [waves/SIMD]", "occ"),
        ("LDS Size [bytes/block]", "lds"),
    )
    parts = []
    for name, fields in per_kernel.items():
        m = re.match(r"_Z(\d+)(\w+)", name)
        short = m.group(2)[: int(m.group(1))] if m else name[:40]
        kv = " ".join(f"{label}={fields[key]}" for key, label in keep if key in fields)
        parts.append(f"{short}: {kv}")
    return "; ".join(parts)


def _build(n: int, source: Path) -> Tuple[Path, float, str]:
    out_dir = _process_build_dir()
    output = out_dir / f"libk512_fp8_gemm_N{n}.so"
    temporary = out_dir / f".libk512_fp8_gemm_N{n}.{os.getpid()}.so"
    log_path = out_dir / f"build_N{n}.log"
    command = hipcc_command(n, source, temporary)
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=_BUILD_TIMEOUT_S,
            check=False,
        )
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
        raise RuntimeError(
            f"hipcc rc={result.returncode} for N={n} (log {log_path}): {tail[:400]}"
        )
    os.replace(temporary, output)
    return output, seconds, resource_summary(result.stdout)


def _load(n: int) -> Kernel:
    source = source_path()
    blob = source.read_bytes()
    path, seconds, resources = _build(n, source)
    library = ctypes.CDLL(str(path))
    fn = getattr(library, entry_symbol(n))
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int, ctypes.c_void_p]
    impl = "?"
    impl_fn = getattr(library, "k512_fp8_gemm_impl", None)
    if impl_fn is not None:
        impl_fn.restype = ctypes.c_char_p
        impl_fn.argtypes = []
        impl = (impl_fn() or b"?").decode(errors="replace")
    return Kernel(
        n=n,
        path=str(path),
        fn=fn,
        impl=impl,
        source=str(source),
        source_sha256=hashlib.sha256(blob).hexdigest(),
        build_s=seconds,
        resources=resources,
    )


def kernel(n: int) -> Kernel:
    """The loaded library for N (built on first use). Raises RuntimeError if it
    cannot be built or loaded; the failure is latched for the process."""

    cached = _kernels.get(n)
    if cached is not None:
        return cached
    if n not in SUPPORTED_N:
        raise RuntimeError(f"N={n} is not built (supported: {SUPPORTED_N})")
    with _n_locks[n]:  # per N, so prebuild() compiles both N in parallel
        cached = _kernels.get(n)
        if cached is not None:
            return cached
        if n in _failures:
            raise RuntimeError(_failures[n])
        try:
            loaded = _load(n)
        except Exception as error:
            _failures[n] = f"K=512 FP8 GEMM library for N={n} unavailable: {error}"
            raise RuntimeError(_failures[n]) from error
        _kernels[n] = loaded
        return loaded


def failure(n: int) -> Optional[str]:
    return _failures.get(n)


def ineligible_reason(
    activation: torch.Tensor,
    weight_nk: torch.Tensor,
    bias: Optional[torch.Tensor],
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> Optional[Tuple[str, str]]:
    """(kind, detail) if the kernel cannot take these operands, else None.

    kind is from a small fixed set (it keys the one-time decline log line)."""

    if _FP8_E4M3 is None:
        return ("dtype", "torch has no float8_e4m3fn")
    if (
        activation.dtype != _FP8_E4M3
        or activation.dim() != 2
        or activation.shape[1] != K
        or not activation.is_contiguous()
    ):
        return (
            "activation",
            f"activation {activation.dtype} {tuple(activation.shape)} stride "
            f"{activation.stride()} (needs contiguous e4m3 [M, {K}])",
        )
    n = weight_nk.shape[0] if weight_nk.dim() == 2 else -1
    if n not in SUPPORTED_N:
        return ("shape", f"N={n} (built for N in {SUPPORTED_N})")
    if (
        weight_nk.dtype != _FP8_E4M3
        or tuple(weight_nk.shape) != (n, K)
        or weight_nk.stride() != (K, 1)
    ):
        return (
            "weight",
            f"weight {weight_nk.dtype} {tuple(weight_nk.shape)} stride "
            f"{weight_nk.stride()} (needs e4m3 [N, {K}] row-major)",
        )
    if bias is None:
        return ("bias", "no bias (the kernel always adds one)")
    if bias.dtype != torch.bfloat16 or tuple(bias.shape) != (n,) or bias.stride() != (1,):
        return (
            "bias",
            f"bias {bias.dtype} {tuple(bias.shape)} stride {bias.stride()} "
            f"(needs contiguous bf16 [{n}])",
        )
    for name, scale in (("scale_a", scale_a), ("scale_b", scale_b)):
        if scale.dtype != torch.float32 or scale.numel() != 1:
            return (
                "scale",
                f"{name} {scale.dtype} {tuple(scale.shape)} (needs one fp32 value)",
            )
    device = activation.device
    if device.type != "cuda" or any(
        t.device != device for t in (weight_nk, bias, scale_a, scale_b)
    ):
        return ("device", "operands are not all on one CUDA device")
    if device.index is not None and device.index != torch.cuda.current_device():
        return (
            "device",
            f"operands on {device}, current device cuda:{torch.cuda.current_device()}",
        )
    if activation.shape[0] > _INT32_MAX:
        return ("rows", f"M={activation.shape[0]} exceeds int32")
    if activation.data_ptr() % 16 or weight_nk.data_ptr() % 16:
        return ("align", "activation/weight not 16-byte aligned")
    return None


def _empty_output(m: int, n: int, device: torch.device) -> torch.Tensor:
    # A separate function so the unit test can hand out a NaN-prefilled buffer.
    return torch.empty((m, n), dtype=_FP8_E4M3, device=device)


def launch(
    kern: Kernel,
    activation: torch.Tensor,
    weight_nk: torch.Tensor,
    bias: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the kernel on operands that passed ineligible_reason(); returns `out`
    or a new e4m3 [M, N] tensor (stream-ordered on the current stream)."""

    m = activation.shape[0]
    if out is None:
        out = _empty_output(m, kern.n, activation.device)
    if m == 0:
        return out
    stream = torch.cuda.current_stream(activation.device).cuda_stream
    rc = kern.fn(
        activation.data_ptr(),
        weight_nk.data_ptr(),
        out.data_ptr(),
        bias.data_ptr(),
        scale_a.data_ptr(),
        scale_b.data_ptr(),
        m,
        stream,
    )
    if rc != 0:
        raise RuntimeError(
            f"{entry_symbol(kern.n)} failed: hipError {rc} (M={m}, lib={kern.path})"
        )
    return out


def k512_gemm_e4m3(
    a_fp8: torch.Tensor,
    weight_nk_fp8: torch.Tensor,
    bias_bf16: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """e4m3 [M, N] = e4m3_rne(sat448(a @ weight_nk.T * (scale_a * scale_b) + bias)).

    a_fp8 e4m3 [M, 512] contiguous; weight_nk_fp8 e4m3 [N, 512] contiguous
    (N in 1024/1536); bias_bf16 [N]; scale_a / scale_b one fp32 value each, on the
    same (current) device. Raises ValueError for anything else."""

    reason = ineligible_reason(a_fp8, weight_nk_fp8, bias_bf16, scale_a, scale_b)
    if reason is not None:
        raise ValueError(f"k512_gemm_e4m3: {reason[1]}")
    return launch(
        kernel(weight_nk_fp8.shape[0]), a_fp8, weight_nk_fp8, bias_bf16, scale_a, scale_b
    )


def log_armed_once(kern: Kernel, m: int) -> None:
    """The one arming line per N per process (printed at the first dispatch)."""

    if kern.n in _armed_logged:
        return
    _armed_logged.add(kern.n)
    _print(
        f"armed: N={kern.n} lib={kern.path} ({KNOB}=1) impl={kern.impl!r} "
        f"src={os.path.basename(kern.source)}@{kern.source_sha256[:12]} "
        f"device=cuda:{torch.cuda.current_device()} first_M={m}"
    )


def log_decline_once(key: str, message: str) -> None:
    """One line per decline kind per process: this call stays on hipBLASLt."""

    if key in _decline_logged:
        return
    _decline_logged.add(key)
    _print(f"declined -> hipBLASLt ({KNOB}=1): {message}")


def prebuild(warm_rows: int = 257) -> bool:
    """Build and load both N before RUN_START, then run each once on synthetic
    zeros (warm_rows rows: one full tile plus a tail row, so the code objects
    are resident and both paths ran) into a 0xFF-prefilled (e4m3 NaN) buffer,
    and check that every output byte became +0.

    No data, model state or RNG is touched. Returns True when both N are ready
    and False when skipped (no gfx950). Raises PrebuildError on a failed build
    or self-check (latched, so the dispatch then declines that N). An error from
    the synchronize is a sticky device fault and propagates unwrapped, as in the
    FlyDSL LayerNorm-backward prebuild."""

    if not torch.cuda.is_available():
        _print("prebuild skipped: no GPU")
        return False
    device = torch.device("cuda", torch.cuda.current_device())
    arch = str(getattr(torch.cuda.get_device_properties(device), "gcnArchName", ""))
    if not arch.startswith("gfx950"):
        _print(f"prebuild skipped: arch {arch or 'unknown'} (built for gfx950)")
        return False
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(SUPPORTED_N)) as pool:
        futures = {n: pool.submit(kernel, n) for n in SUPPORTED_N}
    errors = []
    for n, future in futures.items():
        try:
            future.result()
        except Exception as error:
            errors.append(f"N={n}: {error}")
    if errors:
        _print("prebuild FAILED: " + " | ".join(errors))
        raise PrebuildError("K=512 FP8 GEMM prebuild failed: " + " | ".join(errors))
    t_build = time.perf_counter() - t0
    u8 = torch.uint8
    a = torch.zeros((warm_rows, K), dtype=u8, device=device).view(_FP8_E4M3)
    one = torch.ones((1, 1), dtype=torch.float32, device=device)
    for n in SUPPORTED_N:
        kern = _kernels[n]
        w = torch.zeros((n, K), dtype=u8, device=device).view(_FP8_E4M3)
        bias = torch.zeros((n,), dtype=torch.bfloat16, device=device)
        out = torch.full((warm_rows, n), 0xFF, dtype=u8, device=device).view(_FP8_E4M3)
        launch(kern, a, w, bias, one, one, out=out)
        torch.cuda.synchronize(device)
        nonzero = int(torch.count_nonzero(out.view(u8)).item())
        if nonzero:
            message = (
                f"N={n}: self-check on zeros wrote {nonzero} nonzero bytes "
                f"({kern.path})"
            )
            with _n_locks[n]:
                _kernels.pop(n, None)
                _failures[n] = f"K=512 FP8 GEMM self-check failed: {message}"
            _print("prebuild FAILED: " + message)
            raise PrebuildError("K=512 FP8 GEMM prebuild failed: " + message)
    detail = "; ".join(
        f"N={n} {_kernels[n].build_s:.1f}s [{_kernels[n].resources}]"
        for n in SUPPORTED_N
    )
    _print(
        f"prebuilt: N={','.join(str(n) for n in SUPPORTED_N)} in {t_build:.1f}s "
        f"(dir {_process_build_dir()}, impl={_kernels[SUPPORTED_N[0]].impl!r}); "
        f"self-check M={warm_rows} OK; {detail}"
    )
    return True
