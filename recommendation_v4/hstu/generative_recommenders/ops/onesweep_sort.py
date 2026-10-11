"""Stable LSD radix sort pairs for bounded integer keys (HIP extension).

onesweep_sort_kernels.hip runs rocPRIM's onesweep device code (the code torch.sort runs) under a leaner host
orchestration: the bit range is the caller's static key bound, the digit width is chosen per caller, the values are
a counting iterator (no iota kernel), and one fill per sort zeroes what rocPRIM zeroes with 1 + 2 fills per pass.
An LSD radix sort is stable and the stable sorting permutation is unique, so for keys in [0, 2^end_bit) the result
equals torch.sort(stable=True), which is what torch.sort runs for every 1-D CUDA input above 4096 elements.

Users (each behind its own default-off switch, see the callers):
  * DLRM_HSTU_FP8_POSENC_TS_ONESWEEP -- ops/triton/triton_position.py: the posenc timestamp-bucket sort (int16 keys
    <= ts table rows - 1, 6-bit digits, 2 passes).
  * DLRM_HSTU_FP8_EC_JOINT_ONESWEEP -- dlrm_v4/train/ec_offsets.py: the grouped EC backward's joint argsort (int32
    keys < 2U read from the two members' int64 ids, 9-bit digits, 2 passes at U <= 2^17) plus both members'
    segment offsets from the sorted keys.

The extension is built by torch.utils.cpp_extension into /tmp/dlrm_hstu_onesweep_sort (sources copied there first,
since the ROCm build hipifies next to its inputs) for the current device's arch only. train_ranker builds it before
RUN_START (DLRM_HSTU_FP8_PREBUILD). A build or load failure returns None from every entry point (the callers keep
torch.sort and each logs its own fail-closed line) and logs one warning here.

Build lock: the ranks of a node serialize the build/load on an flock of <dir>/build.flock, which the kernel releases
when its holder dies (SIGKILL included), with a timeout. torch's own <dir>/lock (FileBaton: an O_EXCL file polled
without a timeout) survives a process killed mid-build and would hang every later build in that container; since
every user of this dir takes the flock first, a torch lock found while holding it is stale and is removed.

One sort is one onesweep batch: rocPRIM's lookback state keeps 30 bits of count (its own host code splits sorts
into batches of < 2^30 items), so inputs of 2^30 or more items return None (the caller's fallback sorts them).
"""
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch

logger: logging.Logger = logging.getLogger(__name__)

_SOURCES = ("onesweep_sort_kernels.hip", "onesweep_sort_bind.cpp")
_BUILD_DIR = Path("/tmp/dlrm_hstu_onesweep_sort")
_BUILD_LOCK_TIMEOUT_S = 900.0  # a hung builder fails the waiters closed
_MAX_ITEMS = 1 << 30  # exclusive: one onesweep batch (30-bit lookback counts)
_EXT = None
_FAILED: Optional[str] = None
_SWITCHES = ("DLRM_HSTU_FP8_POSENC_TS_ONESWEEP", "DLRM_HSTU_FP8_EC_JOINT_ONESWEEP")


def _switch_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "on", "yes")


def wanted() -> bool:
    """Whether any switch that uses the extension is on (prebuild gate)."""
    return any(_switch_on(n) for n in _SWITCHES)


def _arch() -> Optional[str]:
    try:
        name = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
        return name.split(":")[0] or None
    except Exception:
        return None


def _stage_sources() -> List[str]:
    src_dir = _BUILD_DIR / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).parent
    out = []
    for f in _SOURCES:
        data = (here / f).read_bytes()
        dst = src_dir / f
        if not dst.exists() or dst.read_bytes() != data:
            tmp = src_dir / f"{f}.{os.getpid()}.tmp"
            tmp.write_bytes(data)
            os.replace(tmp, dst)  # atomic: concurrent ranks stage identical bytes
        out.append(str(dst))
    return out


def _locked(fn):
    """fn() under an exclusive flock on <dir>/build.flock (see the module docstring), after removing a stale
    torch build lock. Raises TimeoutError after _BUILD_LOCK_TIMEOUT_S."""
    import fcntl

    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(_BUILD_DIR / "build.flock"), os.O_RDWR | os.O_CREAT, 0o666)
    try:
        t0 = time.monotonic()
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - t0 > _BUILD_LOCK_TIMEOUT_S:
                    raise TimeoutError(
                        f"{_BUILD_DIR}/build.flock still held after {_BUILD_LOCK_TIMEOUT_S:.0f} s"
                    ) from None
                time.sleep(0.05)
        stale = _BUILD_DIR / "lock"
        if stale.exists():
            # no live user of this dir can hold it now: left by a process killed mid-build
            logger.warning("DLRMV4 onesweep sort: removed a stale torch build lock %s", stale)
            stale.unlink(missing_ok=True)
        return fn()
    finally:
        os.close(fd)  # releases the flock


def extension():
    """The built module, or None (build/load failed: callers keep torch.sort)."""
    global _EXT, _FAILED
    if _EXT is not None or _FAILED is not None:
        return _EXT
    try:
        from torch.utils.cpp_extension import load

        arch = _arch()
        cuda_flags = ["-O3", "-U__HIP_NO_HALF_CONVERSIONS__", "-U__HIP_NO_HALF_OPERATORS__"]
        if arch:
            cuda_flags.append(f"--offload-arch={arch}")
        _EXT = _locked(
            lambda: load(
                name="dlrm_hstu_onesweep_sort",
                sources=_stage_sources(),
                extra_cflags=["-O3"],
                extra_cuda_cflags=cuda_flags,
                build_directory=str(_BUILD_DIR),
                verbose=False,
            )
        )
    except Exception as e:  # fail closed: torch.sort everywhere
        _FAILED = f"{type(e).__name__}: {e}"[:400]
        logger.warning(
            "DLRMV4 onesweep sort extension unavailable -> torch.sort / incumbent "
            "sort paths for the rest of the run (%s)",
            _FAILED,
        )
    return _EXT


def prebuild() -> bool:
    """Build/load the extension when a switch that uses it is on (train_ranker, before RUN_START)."""
    if not wanted():
        return False
    t0 = time.perf_counter()
    ok = extension() is not None
    if ok:
        logger.info(
            "armed OK: prebuild before run_start: onesweep_sort_ext=1 %.1fs", time.perf_counter() - t0
        )
    else:
        logger.warning(
            "DLRMV4 NOT armed: prebuild before run_start: onesweep_sort_ext=0 %.1fs (%s)",
            time.perf_counter() - t0,
            _FAILED,
        )
    return ok


def _on_current_device(t: torch.Tensor) -> bool:
    return t.is_cuda and t.device.index == torch.cuda.current_device()


def sort16_pairs(keys: torch.Tensor, max_key: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """(sorted keys, int64 permutation) == torch.sort(keys) for contiguous 1-D int16 keys in [0, max_key] with
    4096 < numel < 2^30, or None (extension unavailable / out of scope)."""
    n = keys.numel()
    if keys.dtype != torch.int16 or keys.dim() != 1 or not keys.is_contiguous() or not _on_current_device(keys):
        return None
    if n <= 4096 or n >= _MAX_ITEMS or max_key < 0:
        return None
    ext = extension()
    if ext is None:
        return None
    end_bit = max(int(max_key).bit_length(), int(ext.radix_bits16()) + 1)
    if end_bit > 15:
        return None
    dev = keys.device
    kout = torch.empty_like(keys)
    vout = torch.empty(n, dtype=torch.int64, device=dev)
    ktmp = torch.empty_like(keys)
    vtmp = torch.empty(n, dtype=torch.int64, device=dev)
    ws = torch.empty(int(ext.ws16(n, end_bit)), dtype=torch.uint8, device=dev)
    ext.sort16(keys, end_bit, kout, vout, ktmp, vtmp, ws)
    return kout, vout


def ec_joint_segments(
    a: torch.Tensor, b: torch.Tensor, num_rows: int
) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
    """[(order_a, offsets_a), (order_b, offsets_b)] == ec_offsets' fallback joint build for valid int64 ids
    (0 <= id < num_rows) with fewer than 2^30 ids in total, or None (extension unavailable / out of scope). The
    two orders are 16-byte aligned views of one int64 buffer; the offsets are int64 [num_rows + 1]."""
    if a.dtype != torch.int64 or b.dtype != torch.int64 or a.dim() != 1 or b.dim() != 1:
        return None
    if not (_on_current_device(a) and a.device == b.device):
        return None
    ta, tb = a.numel(), b.numel()
    n = ta + tb
    if ta == 0 or tb == 0 or n >= _MAX_ITEMS or num_rows <= 0 or 2 * num_rows > 2**31 - 1:
        return None
    ext = extension()
    if ext is None:
        return None
    end_bit = max(int(2 * num_rows - 1).bit_length(), int(ext.radix_bits32()) + 1)
    sh = ta & 1
    dev = a.device
    kout = torch.empty(n, dtype=torch.int32, device=dev)
    vout = torch.empty(n + sh, dtype=torch.int64, device=dev)
    ktmp = torch.empty(n, dtype=torch.int32, device=dev)
    vtmp = torch.empty(n, dtype=torch.int64, device=dev)
    ws = torch.empty(int(ext.ws32(n, end_bit)), dtype=torch.uint8, device=dev)
    off_a = torch.empty(num_rows + 1, dtype=torch.int64, device=dev)
    off_b = torch.empty(num_rows + 1, dtype=torch.int64, device=dev)
    gaps = torch.empty(3 * int(ext.gap_capacity(num_rows)), dtype=torch.int64, device=dev)
    ext.ec_joint(a, b, num_rows, end_bit, kout, vout, ktmp, vtmp, ws, off_a, off_b, gaps)
    return [(vout[:ta], off_a), (vout[ta + sh:], off_b)]
