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

# pyre-strict
"""DO8: the attention-backward dO quantization folded into L11's e5m2-du mode.

DLRM_HSTU_FP8_L11_DO8 (yambda_5b.gin; only "1" arms it, "0" or any other value is off). Producer side of the dO8
contract in ops/flydsl/hstu_do8_registry.py ('plus' only).

WHAT. On an ordinary-layer backward call that L11 runs in its e5m2-du mode (DLRM_HSTU_FP8_BWD_DU_E5M2), and whose
attention backward will quantize dO on the do_blk path (the attention forward tagged a dO8 site on `out`), ONE launch
of a separate kernel, l11_outproj_lnmd_bwd_do8 (l11_do8.cu, its own .so), replaces L11's E5
launch. It writes everything L11's E5 launch writes (bf16 d(attn), the e5m2 u block + pad rows, amax / u-colsum /
dW / dB partials), on sequence-aligned 64-token tiles, and in addition the attention backward's e4m3 dO (dO8 [M, 512])
and E8M0 group scales (E8 [M, 4] = 127 - e_g), byte-identical to hstu_do_fp8.quantize_do_e4m3_blk(d(attn),
seq_offsets, N) on every row that quantizer writes. The tiles come from a device work list (l11_do8_wl_build) built
from the attention call's own seq_offsets tensor (the site's), once per step: both ordinary layers receive the same
x_offsets object, so the second layer reuses the first layer's list (memo keyed on that object, its _version, numel,
dtype, device, M and the stream). After the last write to d(attn) the backward publishes (dO8, E8) on the site
(hstu_do8_registry.publish, bf16_written=True); the attention backward takes them instead of running the quantizer
on the bf16 d(attn), or -- any failed check -- quantizes as before (same bytes).

WHY A SEPARATE .so. L11's loader latches ALL of L11 off unless every instantiation in its .so has
0 VGPR spill / 0 scratch, and the E5 + emission instantiation crashes clang's greedy register allocator under L11's
iterative-ilp scheduler. So DO8 is its own translation unit with its own flags (no -amdgpu-sched-strategy: the default
scheduler), cache (/tmp/dlrm_hstu_fp8_l11_do8/<key>), resource gate, prebuild
self-check and latch. Any DO8 failure (build, gate, self-check, launch, registry) latches only DO8; L11 keeps running
its own launches (and the attention backward its quantizer).

VD SOURCE. The consumer instruction reductions of l11_outproj_lnmd_vd.cu (C1 bf16-domain keep masks, C2
one Newton step, C3 unsigned-max fallback predicate, C4 packed dW/dB chains, LANEFIX, PIN) are also in DO8's
kernel body: l11_do8.cu carries them (impl tag "+vd(C1,C2,C3,C4,LANEFIX,PIN)"), l11_do8_novd.cu is the same source
without them. The source follows L11's own selector
(l11_outproj_lnmd.vd_requested(), DLRM_HSTU_FP8_L11_VD == "1"), so DO8 and L11 always build the same consumer
variant; each source has its own cache key (the key covers the source bytes). The l11_do8.cu launch rejects a dscale
outside (0, FLT_MAX], as l11_outproj_lnmd_vd.cu's launches do.

DECIDE BEFORE EMISSION. `emission_site` is evaluated on host metadata only (no sync, no allocation, no launch) right
before the launch: knob on, prebuilt, not latched, L11 enabled and prebuilt, the call is in the e5m2-du mode (the du
e5m2 plan), a site on attn, the consumer knob on (registry.take_enabled()), site T == M and (H, D) == (4, 128),
seq_offsets int32/int64, contiguous, on the device, B <= 8192. Otherwise L11's own E5 launch runs (no emission). The
bf16-du mode (the du e5m2 plan declined, e.g. warm-up calls) has no DO8 variant: counted, never printed.

LINES (every rank, "[rank N]", ops.utils.arm_print_all_ranks, once each per process; only with the knob on). All start
with FAMILY. "armed OK: <FAMILY> prebuild before run_start ...", "armed OK: <FAMILY> ACTIVE mode=e5 plus ..." at the
first published record, "<FAMILY> NOT armed: <reason> ..." once per unexpected reason, "<FAMILY> DISABLED: ...",
"<FAMILY> INERT: ...", "<FAMILY> prebuild DEVICE FAULT: ...". Counters ("[l11-do8] counters ...") at events 1, 2, 4,
8, ... and once at exit.
"""

import atexit
import ctypes
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import weakref
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from generative_recommenders.ops.hip import l11_outproj_lnmd as l11
from generative_recommenders.ops.utils import arm_print_all_ranks

KNOB: str = "DLRM_HSTU_FP8_L11_DO8"
SCHED_ENV: str = KNOB + "_SCHED"  # "default" (code default) | "iterative-ilp" (diagnostic)
KERNEL_NAME: str = "l11_outproj_lnmd_bwd_do8"
BUILD_KERNEL_NAME: str = "l11_do8_wl_build"
FAMILY: str = "L11 DO8 dO8/E8 emission"
ITEM: str = "DO8"

_SOURCE: Path = Path(__file__).with_name("l11_do8.cu")  # with the consumer instruction reductions (DLRM_HSTU_FP8_L11_VD=1)
_SOURCE_NOVD: Path = Path(__file__).with_name("l11_do8_novd.cu")  # without them (DLRM_HSTU_FP8_L11_VD off)
_ARCH: str = "gfx950"
# L11's flags (l11_outproj_lnmd._FLAGS) WITHOUT "-mllvm -amdgpu-sched-strategy=iterative-ilp": the E5 + emission
# instantiation builds only with the default scheduler (iterative-ilp crashes clang-22's greedy register allocator).
_BASE_FLAGS: Tuple[str, ...] = (
    "-DKITTENS_CDNA4",
    "-std=c++20",
    "-w",
    "-O3",
    "-ffp-contract=off",
    "--save-temps",
    "-shared",
    "-fPIC",
    "-Rpass-analysis=kernel-resource-usage",
)
_SCHED_FLAGS: Dict[str, Tuple[str, ...]] = {
    "default": (),
    "iterative-ilp": ("-mllvm", "-amdgpu-sched-strategy=iterative-ilp"),
}
_CACHE_ROOT: str = "/tmp/dlrm_hstu_fp8_l11_do8"
_LIB_NAME: str = "libl11_do8.so"
_BUILD_TIMEOUT_S: int = 300
_ABI_VERSION: int = 1
LDS_BYTES: int = 163840  # the 'plus' instantiation: L11's 96 KiB + the 64 KiB d(attn) tile staging
H: int = 4  # heads: lane >> 4 of the kernel's 64 lanes x 8 columns
HD: int = 128  # head dim (H * HD = 512 = L11's D)
B_MAX: int = 8192  # sequences per list build (1024 threads x 8)
# The production instantiation's template arguments in the mangled name (E5 1, WLIST 2, EM 1, RET 0, EMV 2, TM 0).
_PROD_MANGLE: str = "ILi1ELi2ELi0ELi0ELi1ELi2ELi1ELi0ELi1ELi2ELi1ELi0ELi2ELi0EE"

_lock = threading.Lock()
_requested: Optional[bool] = None
_override: Optional[bool] = None
_broken: Optional[str] = None
_lib: Optional[ctypes.CDLL] = None
_lib_path: Optional[Path] = None
_fn_launch: Optional[Callable[..., int]] = None
_fn_wl_build: Optional[Callable[..., int]] = None
_fn_wl_cap: Optional[Callable[..., int]] = None
_fn_plan: Optional[Callable[..., int]] = None
_fn_drain: Optional[Callable[[], int]] = None
_impl: str = ""
_resources: Dict[str, Any] = {}
_build_secs: float = 0.0
_prebuilt: bool = False
_active_logged: bool = False
_once: set = set()
_plan_cache: Dict[int, Tuple[int, int, int, int]] = {}
_registry_mod: Any = None
_registry_tried: bool = False
# Work-list memo (one entry): the list of the latest seq_offsets object (weakref) + its key.
_WL_MEMO: Dict[str, Any] = {}
# Per-call engagement counters (per process).
_COUNTS: Dict[str, int] = {
    "calls": 0,
    "emitted": 0,
    "published": 0,
    "refused": 0,
    "failed": 0,
    "wl_built": 0,
    "wl_reused": 0,
    "bf16_mode": 0,
}
_DECLINED: Dict[str, int] = {}
_STATE: Dict[str, Any] = {"events": 0, "atexit": False}


# ---------------------------------------------------------------------------- knob / latch / lines
def requested() -> bool:
    """DLRM_HSTU_FP8_L11_DO8 == "1" (read once, lazily: the gin env bootstrap runs after import)."""
    global _requested
    if _override is not None:
        return _override
    if _requested is None:
        _requested = os.environ.get(KNOB, "0") == "1"
    return _requested


def enabled() -> bool:
    return requested() and _broken is None


def broken() -> Optional[str]:
    return _broken


def prebuilt() -> bool:
    return _prebuilt


def resources() -> Dict[str, Any]:
    return dict(_resources)


def impl() -> str:
    return _impl


def library_path() -> Optional[Path]:
    return _lib_path


def counters() -> Dict[str, Any]:
    out: Dict[str, Any] = dict(_COUNTS)
    out["declined"] = dict(_DECLINED)
    out["events"] = int(_STATE["events"])
    return out


def set_requested_for_testing(value: Optional[bool]) -> None:
    global _override, _requested
    _override = value
    _requested = None


def set_prebuilt_for_testing(value: bool) -> None:
    global _prebuilt
    _prebuilt = value


def reset_for_testing(unload: bool = False) -> None:
    """Tests only: clear the latch, once-lines, counters, memo, prebuilt flag; unload=True also forgets the library."""
    global _broken, _prebuilt, _active_logged, _registry_mod, _registry_tried
    global _lib, _lib_path, _fn_launch, _fn_wl_build, _fn_wl_cap, _fn_plan, _fn_drain, _impl
    _broken = None
    _prebuilt = False
    _active_logged = False
    _registry_mod = None
    _registry_tried = False
    _once.clear()
    _plan_cache.clear()
    _WL_MEMO.clear()
    for k in _COUNTS:
        _COUNTS[k] = 0
    _DECLINED.clear()
    _STATE["events"] = 0
    if unload:
        with _lock:
            _lib = None
            _lib_path = None
            _fn_launch = _fn_wl_build = _fn_wl_cap = _fn_plan = _fn_drain = None
            _impl = ""
            _resources.clear()


def _fail_closed() -> bool:
    """DLRM_HSTU_FP8_L11_DO8_FAIL_CLOSED=1 (smoke/debug only): raise instead of latching off."""
    return os.environ.get(KNOB + "_FAIL_CLOSED", "0") == "1"


def _print_once(key: str, message: str) -> None:
    if key in _once:
        return
    _once.add(key)
    arm_print_all_ranks(message)


def summary_line(final: bool = False) -> str:
    dec = ",".join(f"{k}:{v}" for k, v in sorted(_DECLINED.items()))
    c = _COUNTS
    return (
        f"[l11-do8] counters: events={_STATE['events']} calls={c['calls']} emitted={c['emitted']} "
        f"published={c['published']} refused={c['refused']} failed={c['failed']} wl_built={c['wl_built']} "
        f"wl_reused={c['wl_reused']} bf16_mode={c['bf16_mode']} declined={{{dec}}} "
        f"latched={int(_broken is not None)}" + (" final=1" if final else "")
    )


def _final() -> None:
    if _STATE["events"]:
        arm_print_all_ranks(summary_line(final=True))


def _event() -> None:
    """One per-call event (emitted / published / refused / declined / failed); log-spaced counter lines."""
    _STATE["events"] += 1
    n = int(_STATE["events"])
    if not _STATE["atexit"]:
        _STATE["atexit"] = True
        atexit.register(_final)
    if n & (n - 1) == 0:
        arm_print_all_ranks(summary_line())


def _count(key: str, event: bool = True) -> None:
    _COUNTS[key] += 1
    if event:
        _event()


def _disable(reason: str) -> None:
    """Latch DO8 (only DO8) off for this process and print one DISABLED line."""
    global _broken
    if _fail_closed():
        raise RuntimeError(f"{FAMILY} failed with {KNOB}_FAIL_CLOSED=1: {reason}")
    if _broken is not None:
        return
    _broken = reason
    _COUNTS["failed"] += 1
    _event()
    arm_print_all_ranks(
        f"{FAMILY} DISABLED: {reason} -> L11's own launches + quantize_do_e4m3_blk for this process "
        "(L11 unaffected)"
    )


def decline_key(reason: str) -> str:
    tokens = reason.split()
    key = tokens[0] if tokens else reason
    if len(tokens) > 1:
        key += " " + re.split(r"[=:(\d]", tokens[1], maxsplit=1)[0]
    return key


def note_decline(reason: str, quiet: bool = False) -> None:
    """Count a declined call; one 'NOT armed' line per (reason key, process) unless quiet (expected states)."""
    k = decline_key(reason)
    _DECLINED[k] = _DECLINED.get(k, 0) + 1
    _event()
    if not quiet:
        _print_once(
            "decline:" + k,
            f"{FAMILY} NOT armed: {reason} -> L11's own launch; the C8 attention backward quantizes dO",
        )


def note_bf16_mode(ctx: Any) -> None:
    """The seam's bf16-du branch (the du e5m2 plan declined this call): DO8 has no variant there. Counted only when a
    dO8 site exists (an emission was possible in principle); never printed (an expected state, e.g. warm-up calls)."""
    if requested() and getattr(ctx, "do8_site", None) is not None:
        _COUNTS["bf16_mode"] += 1


# ---------------------------------------------------------------------------- registry (hstu_do8_registry)
def registry() -> Any:
    """The consumer's registry module, or None when this tree has no registry (one INERT line, knob on only)."""
    global _registry_mod, _registry_tried
    if _registry_tried:
        return _registry_mod
    _registry_tried = True
    try:
        from generative_recommenders.ops.flydsl import hstu_do8_registry as reg

        _registry_mod = reg
    except ImportError as error:
        _registry_mod = None
        if requested():
            _print_once(
                "inert-registry",
                f"{FAMILY} INERT: no ops/flydsl/hstu_do8_registry ({error}); the consumer hook is not in this "
                "tree, so no dO8 site exists and nothing is emitted",
            )
    return _registry_mod


def site_of(attn: torch.Tensor) -> Any:
    """FORWARD (HSTUComputeOutputFunction.forward only, registry contract rule (a)): the dO8 site tagged on `attn`, or None.
    Knob off: None, nothing imported, nothing printed."""
    if not requested():
        return None
    reg = registry()
    if reg is None:
        return None
    return reg.site_of(attn)


# ---------------------------------------------------------------------------- build / load
def sched() -> str:
    raw = os.environ.get(SCHED_ENV, "").strip() or "default"
    if raw not in _SCHED_FLAGS:
        _print_once("sched", f"{FAMILY}: {SCHED_ENV}={raw!r} is not one of {sorted(_SCHED_FLAGS)} -> 'default'")
        return "default"
    return raw


def _source() -> Path:
    """DLRM_HSTU_FP8_L11_DO8_SRC overrides the source. Otherwise L11's own source selector decides: l11_do8.cu when
    DLRM_HSTU_FP8_L11_VD == "1", else l11_do8_novd.cu."""
    override = os.environ.get(KNOB + "_SRC", "").strip()
    if override:
        return Path(override)
    return _SOURCE if l11.vd_requested() else _SOURCE_NOVD


def _flags(src: bytes) -> List[str]:
    sha = hashlib.sha256(src).hexdigest()
    return [f"--offload-arch={_ARCH}", *_BASE_FLAGS, *_SCHED_FLAGS[sched()], f'-DL11DO8_SRC_SHA256="{sha}"']


def build_command(source: Path, out: Path) -> List[str]:
    """The exact hipcc command of `build`."""
    return [l11._hipcc(), str(source.resolve()), *_flags(source.read_bytes()), "-o", str(out.resolve())]


def cache_key(source: Optional[Path] = None) -> str:
    source = source or _source()
    src = source.read_bytes()
    digest = hashlib.sha256()
    digest.update(src)
    digest.update(b"\0" + "\0".join(_flags(src)).encode())
    digest.update(b"\0" + _ARCH.encode())
    digest.update(b"\0" + l11._toolchain(l11._hipcc()).encode())
    return digest.hexdigest()[:16]


def _cache_dir(source: Optional[Path] = None) -> Path:
    root = os.environ.get(KNOB + "_CACHE", "").strip() or _CACHE_ROOT
    return Path(root) / cache_key(source)


def check_resources(kernels: List[Dict[str, str]]) -> Dict[str, Any]:
    """DO8's own gate (L11's is untouched): the 'plus' kernel and the list builder must report 0 VGPR spill and 0
    scratch, and the 'plus' kernel exactly LDS_BYTES of LDS; raises otherwise (the caller latches DO8 off)."""
    mine = [k for k in kernels if KERNEL_NAME in k.get("Function Name", "")]
    wlb = [k for k in kernels if BUILD_KERNEL_NAME in k.get("Function Name", "")]
    if len(mine) != 1:
        raise RuntimeError(f"{len(mine)} {KERNEL_NAME} resource remarks in the build log (need exactly 1)")
    if len(wlb) < 1:
        raise RuntimeError(f"no {BUILD_KERNEL_NAME} resource remarks in the build log")
    for k in mine + wlb:
        spill = int(k.get("VGPRs Spill", "-1"))
        scratch = int(k.get("ScratchSize [bytes/lane]", "-1"))
        if spill != 0 or scratch != 0:
            raise RuntimeError(
                f"resource check: {k.get('Function Name')} VGPR spill {spill}, scratch {scratch} B/lane (need 0/0)"
            )
    p = mine[0]
    if _PROD_MANGLE not in p.get("Function Name", ""):
        raise RuntimeError(f"resource check: unexpected instantiation {p.get('Function Name')}")
    lds = int(p.get("LDS Size [bytes/block]", "-1"))
    if lds != LDS_BYTES:
        raise RuntimeError(f"resource check: {KERNEL_NAME} LDS {lds} B (expected {LDS_BYTES})")
    return {
        "vgpr": int(p.get("VGPRs", "-1")),
        "sgpr": int(p.get("TotalSGPRs", "-1")),
        "vgpr_spill": int(p.get("VGPRs Spill", "-1")),
        "sgpr_spill": int(p.get("SGPRs Spill", "-1")),
        "scratch": int(p.get("ScratchSize [bytes/lane]", "-1")),
        "lds": lds,
        "wl_build_vgpr": max(int(k.get("VGPRs", "-1")) for k in wlb),
        "sched": sched(),
        "kernels": len(mine) + len(wlb),
    }


def build(source: Optional[Path] = None) -> Tuple[Path, Dict[str, Any]]:
    """Compile (cache hit: no compile) and return (library path, resource summary). Raises on any failure (missing
    hipcc, compile error or compiler crash, timeout, VGPR spill, scratch, LDS mismatch). Same race-safe cache protocol
    as L11's loader (per-host/pid/thread temp file, atomic renames, sidecar before the .so)."""
    source = source or _source()
    so = _cache_dir(source) / _LIB_NAME
    meta = so.with_name(_LIB_NAME + ".resources.json")
    if so.is_file() and meta.is_file():
        summary = json.loads(meta.read_text())
        check_resources(summary["kernels_raw"])
        return so, summary
    so.parent.mkdir(parents=True, exist_ok=True)
    tag = f"tmp-{socket.gethostname()}-{os.getpid()}-{threading.get_ident()}"
    tmp = so.parent / f"{tag}-{_LIB_NAME}"
    tmp_meta = so.parent / f"{tag}-{_LIB_NAME}.resources.json"
    work = so.parent / f"{tag}-build"
    src = source.read_bytes()
    cmd = build_command(source, tmp)
    try:
        work.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=_BUILD_TIMEOUT_S, cwd=str(work))
        log = (proc.stderr or "") + (proc.stdout or "")
        if proc.returncode != 0 or not tmp.is_file():
            raise RuntimeError(f"hipcc rc={proc.returncode} (sched={sched()}): {log.strip()[-1200:]}")
        kernels = l11.parse_resource_remarks(log)
        summary = check_resources(kernels)
        summary["kernels_raw"] = kernels
        summary["src_sha256"] = hashlib.sha256(src).hexdigest()
        tmp_meta.write_text(json.dumps(summary))
        os.replace(tmp_meta, meta)
        os.replace(tmp, so)
    finally:
        for t in (tmp, tmp_meta):
            try:
                t.unlink()
            except FileNotFoundError:
                pass
        shutil.rmtree(work, ignore_errors=True)
    return so, summary


def _bind(lib: ctypes.CDLL) -> None:
    global _fn_launch, _fn_wl_build, _fn_wl_cap, _fn_plan, _fn_drain, _impl
    abi = lib.l11do8_abi_version
    abi.restype = ctypes.c_int
    abi.argtypes = []
    got = int(abi())
    if got != _ABI_VERSION:
        raise RuntimeError(f"l11do8_abi_version()={got}, expected {_ABI_VERSION}")
    launch = lib.l11do8_e5_plus_launch
    launch.restype = ctypes.c_int
    launch.argtypes = (
        [ctypes.c_void_p] * 4  # G8, W8, sa, sb
        + [ctypes.c_int]  # M
        + [ctypes.c_void_p] * 9  # X, U, MK, MEAN, RSTD, WN, BN, DX, DZ8
        + [ctypes.c_int]  # rows_out
        + [ctypes.c_void_p] * 5  # INVS, AMAXP, DWP, DBP, DUCS
        + [ctypes.c_longlong]  # P
        + [ctypes.c_float, ctypes.c_float]  # dscale, invD
        + [ctypes.c_void_p] * 2  # WL, CNT
        + [ctypes.c_longlong]  # cap
        + [ctypes.c_void_p] * 2  # DO8, E8
        + [ctypes.c_void_p]  # stream
    )
    wlb = lib.l11do8_wl_build_launch
    wlb.restype = ctypes.c_int
    wlb.argtypes = [
        ctypes.c_void_p,  # OFFS
        ctypes.c_int,  # offs_i64
        ctypes.c_int,  # B
        ctypes.c_int,  # M
        ctypes.c_int,  # nprog
        ctypes.c_void_p,  # WL
        ctypes.c_longlong,  # cap
        ctypes.c_void_p,  # CNT
        ctypes.c_void_p,  # stream
    ]
    cap = lib.l11do8_wl_cap
    cap.restype = ctypes.c_longlong
    cap.argtypes = [ctypes.c_longlong, ctypes.c_longlong]
    plan_ = lib.l11do8_plan
    plan_.restype = ctypes.c_int
    plan_.argtypes = [ctypes.c_longlong, ctypes.c_void_p]
    drain = lib.l11do8_drain_last_error
    drain.restype = ctypes.c_int
    drain.argtypes = []
    tag = lib.l11do8_impl
    tag.restype = ctypes.c_char_p
    tag.argtypes = []
    _impl = (tag() or b"").decode()
    _fn_launch, _fn_wl_build, _fn_wl_cap, _fn_plan, _fn_drain = launch, wlb, cap, plan_, drain


def _load() -> None:
    global _lib, _lib_path, _build_secs
    if _lib is not None:
        return
    with _lock:
        if _lib is not None:
            return
        t0 = time.perf_counter()
        so, summary = build()
        lib = ctypes.CDLL(str(so))
        _bind(lib)
        _resources.clear()
        _resources.update({k: v for k, v in summary.items() if k != "kernels_raw"})
        _build_secs = time.perf_counter() - t0
        _lib_path = so
        _lib = lib


def wl_cap(m: int, b: int) -> int:
    """List capacity (entries) for m rows and b sequences, from the .so (cdiv(m, 64) + b + 2)."""
    assert _fn_wl_cap is not None
    cap = int(_fn_wl_cap(int(m), int(b)))
    if cap <= 0:
        raise RuntimeError(f"l11do8_wl_cap(M={m}, B={b}) = {cap}")
    return cap


def plan(cap: int) -> Tuple[int, int, int, int]:
    """(G, fl, nf, P) for a list of at most cap tiles, from the .so."""
    got = _plan_cache.get(cap)
    if got is not None:
        return got
    assert _fn_plan is not None
    out = (ctypes.c_longlong * 4)()
    rc = int(_fn_plan(int(cap), ctypes.addressof(out)))
    if rc != 0:
        raise RuntimeError(f"l11do8_plan(cap={cap}) rc={rc}")
    got = (int(out[0]), int(out[1]), int(out[2]), int(out[3]))
    if len(_plan_cache) < 4096:
        _plan_cache[cap] = got
    return got


# ---------------------------------------------------------------------------- decide before emission
def _offsets_ok(offs: Any, device: torch.device) -> Optional[str]:
    if not isinstance(offs, torch.Tensor):
        return "seq_offsets not a tensor"
    if offs.dim() != 1 or offs.dtype not in (torch.int32, torch.int64) or not offs.is_contiguous():
        return f"seq_offsets {offs.dtype} {tuple(offs.shape)} contiguous={offs.is_contiguous()}"
    if offs.device != device:
        return f"seq_offsets on {offs.device}, d(attn) on {device}"
    if not (1 <= offs.numel() <= B_MAX + 1):
        return f"seq_offsets B={offs.numel() - 1} outside [0, {B_MAX}]"
    if offs.data_ptr() % offs.element_size() != 0:
        return "seq_offsets misaligned"
    return None


def emission_site(ctx: Any, attn: torch.Tensor) -> Any:
    """DECIDE BEFORE EMISSION (host metadata only: no sync, no allocation, no launch), called by the seam right before
    L11's e5m2-du launch. Returns the dO8 site to emit for, or None (L11's own launch runs; nothing is published)."""
    if not requested():
        return None
    site = getattr(ctx, "do8_site", None)
    if site is None:
        # no site: the attention backward does not plan the do_blk quantization for this call (or the consumer
        # knob was off at the forward) -- an expected state, not counted as a call
        return None
    _COUNTS["calls"] += 1
    if _broken is not None:
        note_decline(f"latched off: {_broken}", quiet=True)
        return None
    if not _prebuilt:
        note_decline("D11 not prebuilt before RUN_START (no build/self-check inside the timed run)")
        return None
    if not (l11.enabled() and l11.prebuilt()):
        note_decline("L11 not enabled/prebuilt", quiet=True)
        return None
    reg = registry()
    if reg is None:
        note_decline("no hstu_do8_registry", quiet=True)
        return None
    if not reg.take_enabled():
        note_decline("consumer knob DLRM_HSTU_FP8_BWD_DO8 off")
        return None
    m = int(attn.shape[0])
    if int(site.T) != m or (int(site.H), int(site.D)) != (H, HD) or int(attn.shape[-1]) != H * HD:
        note_decline(f"site T={site.T} H={site.H} D={site.D} vs d(attn) [{m}, {int(attn.shape[-1])}] (kernel: H=4, D=128)")
        return None
    if site.device != attn.device:
        note_decline(f"site device {site.device} vs d(attn) {attn.device}")
        return None
    why = _offsets_ok(site.seq_offsets, attn.device)
    if why is not None:
        note_decline(why)
        return None
    return site


# ---------------------------------------------------------------------------- device work list
def work_list(offs: torch.Tensor, m: int, n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """(WL int32 [cap, 2], CNT int32 [2], cap) of the sequence-aligned 64-token tile list for m rows, built on the
    current stream by l11_do8_wl_build from `offs` (the attention's own seq_offsets tensor), or reused: one entry,
    keyed on the offs OBJECT (weakref identity: a freed and re-allocated offsets tensor never matches) and its
    data_ptr, _version, numel, dtype, device, m and the stream. Both ordinary layers pass the same x_offsets object,
    so the list is built once per step."""
    assert _fn_wl_build is not None
    stream = torch.cuda.current_stream(device)
    key = (offs.data_ptr(), offs._version, int(offs.numel()), str(offs.dtype), str(offs.device), int(m),
           int(stream.cuda_stream))
    ref = _WL_MEMO.get("ref")
    if ref is not None and ref() is offs and _WL_MEMO.get("key") == key:
        _COUNTS["wl_reused"] += 1
        return _WL_MEMO["wl"], _WL_MEMO["cnt"], int(_WL_MEMO["cap"])
    b = int(offs.numel()) - 1
    cap = wl_cap(m, b)
    wl = torch.empty((cap, 2), dtype=torch.int32, device=device)
    cnt = torch.empty((2,), dtype=torch.int32, device=device)
    nprog = max(1, min(128, (int(n) + 63) // 64))  # cdiv(N, 64) programs, capped at 128; longer sequences: grid-stride
    rc = int(
        _fn_wl_build(
            offs.data_ptr(), int(offs.dtype == torch.int64), b, int(m), nprog, wl.data_ptr(), cap, cnt.data_ptr(),
            stream.cuda_stream,
        )
    )
    if rc > 0:
        raise RuntimeError(f"l11do8_wl_build_launch rc={rc} (hipError) M={m} B={b}")
    _WL_MEMO.clear()
    _WL_MEMO.update(ref=weakref.ref(offs), key=key, wl=wl, cnt=cnt, cap=cap)
    _COUNTS["wl_built"] += 1
    return wl, cnt, cap


# ---------------------------------------------------------------------------- launch / run
def launch_e5_plus(
    g8: torch.Tensor,
    w8: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    attn: torch.Tensor,
    u: torch.Tensor,
    random_mask: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    dropout_ratio: float,
    inv_scale: torch.Tensor,
    dz8: torch.Tensor,
    wl: torch.Tensor,
    cnt: torch.Tensor,
    cap: int,
    do8: torch.Tensor,
    e8: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """One DO8 launch on the current stream into the caller's dz8 / do8 / e8 (raw: tests and the self-check keep the
    partials). Returns (dx, DWP, DBP, DUCS, AMAXP, P); raises on a launch error (rc > 0). `run_e5_plus` is the
    production entry."""
    _load()
    assert _fn_launch is not None and _fn_drain is not None
    m = int(attn.shape[0])
    rows_out = int(dz8.shape[0])
    if not (dz8.dtype == torch.float8_e5m2 and dz8.dim() == 2 and int(dz8.shape[1]) == l11.DU_ROW
            and dz8.stride() == (l11.DU_ROW, 1) and l11._a16(dz8) and m <= rows_out < m + l11.E5_MAX_PAD):
        raise RuntimeError(f"dz8 layout {dz8.dtype} {tuple(dz8.shape)} {dz8.stride()} (M={m})")
    if not (do8.dtype == torch.float8_e4m3fn and tuple(do8.shape) == (m, l11.D) and do8.is_contiguous()
            and l11._a16(do8)):
        raise RuntimeError(f"do8 layout {do8.dtype} {tuple(do8.shape)} {do8.stride()} (M={m})")
    if not (e8.dtype == torch.uint8 and tuple(e8.shape) == (m, H) and e8.is_contiguous() and l11._a16(e8)):
        raise RuntimeError(f"e8 layout {e8.dtype} {tuple(e8.shape)} {e8.stride()} (M={m})")
    if not (wl.dtype == torch.int32 and tuple(wl.shape) == (cap, 2) and wl.is_contiguous()
            and cnt.dtype == torch.int32 and cnt.numel() == 2 and cnt.is_contiguous()):
        raise RuntimeError(f"work list layout {tuple(wl.shape)} {wl.dtype} / {tuple(cnt.shape)} {cnt.dtype}")
    G, fl, nf, P = plan(cap)
    device = attn.device
    index = device.index if device.index is not None else torch.cuda.current_device()
    with torch.cuda.device(index):
        stale = int(_fn_drain())
        if stale != 0:
            _print_once(f"stale:{stale}", f"{FAMILY}: stale HIP error {stale} drained before the launch (not ours)")
        stream = torch.cuda.current_stream(device).cuda_stream
        dx = torch.empty_like(attn)
        dwp = torch.empty((P, l11.D), dtype=torch.float32, device=device)
        dbp = torch.empty((P, l11.D), dtype=torch.float32, device=device)
        ducs = torch.empty((P, l11.D), dtype=torch.float32, device=device)
        amaxp = torch.empty((G * 8,), dtype=torch.float32, device=device)
        rc = int(
            _fn_launch(
                g8.data_ptr(), w8.data_ptr(), sa.data_ptr(), sb.data_ptr(), m,
                attn.data_ptr(), u.data_ptr(), random_mask.data_ptr(), mean.data_ptr(), rstd.data_ptr(),
                norm_weight.data_ptr(), norm_bias.data_ptr(), dx.data_ptr(), dz8.data_ptr(), rows_out,
                inv_scale.data_ptr(), amaxp.data_ptr(), dwp.data_ptr(), dbp.data_ptr(), ducs.data_ptr(), int(P),
                l11.dropout_scale(dropout_ratio), 1.0 / l11.D, wl.data_ptr(), cnt.data_ptr(), int(cap),
                do8.data_ptr(), e8.data_ptr(), stream,
            )
        )
    if rc > 0:
        raise RuntimeError(f"l11do8_e5_plus_launch rc={rc} (hipError) M={m}")
    if rc < 0:
        _print_once(f"stale:{-rc}", f"{FAMILY}: stale HIP error {-rc} drained at the launch (not ours; result used)")
    return dx, dwp, dbp, ducs, amaxp, P


def run_e5_plus(
    g8: torch.Tensor,
    w8: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    BLOCK_D: int,
    num_warps: int,
    dropout_ratio: float,
    silu_u: bool,
    random_mask: torch.Tensor,
    du_row_width: int,
    site_key: int,
    inv_scale: torch.Tensor,
    cast_scale: torch.Tensor,
    generation: Optional[Tuple[object, int]],
    site: Any,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any, torch.Tensor, torch.Tensor]]:
    """PRODUCTION: l11_outproj_lnmd.run_e5's exact contract and return (dx, placeholder du, dweight, dbias,
    DuE5m2Emit) -- the same checks (l11._operands_ok, l11.e5_decline_reason), the same dz8 / record construction --
    plus (dO8 [M, 512] e4m3fn, E8 [M, 4] uint8) for `site` (from emission_site). None = declined (its own line) or
    failed (DO8 latched, DISABLED line): the caller then runs L11's own run_e5 on the SAME operands, which writes only
    fresh buffers too, so nothing of this attempt is visible."""
    if not (enabled() and _prebuilt and l11.enabled() and l11.prebuilt()):
        return None
    m = int(x.shape[0])
    why = l11._operands_ok(g8, w8, sa, sb, m)
    try:
        from generative_recommenders.ops.triton import bwd_fp8, hstu_du_e5m2

        grad_dtype, limit = bwd_fp8.grad_dtype_and_limit()
        pad = int(bwd_fp8.uvqk_tm_pad())
        if why is None:
            why = l11.e5_decline_reason(x, u, BLOCK_D, num_warps, silu_u, du_row_width, grad_dtype, limit, pad,
                                        inv_scale)
        if why is not None:
            note_decline("E5 " + why, quiet=True)  # L11's own run_e5 runs the same checks and prints its line
            return None
        if _lib is None:
            raise RuntimeError("library not loaded (no prebuild)")
        wl, cnt, cap = work_list(site.seq_offsets, m, int(site.N), x.device)
        rows_out = (m + pad - 1) // pad * pad
        dz8 = torch.empty((rows_out, du_row_width), dtype=grad_dtype, device=x.device)
        do8 = torch.empty((m, l11.D), dtype=torch.float8_e4m3fn, device=x.device)
        e8 = torch.empty((m, H), dtype=torch.uint8, device=x.device)
        dx, dwp, dbp, ducs, amaxp, P = launch_e5_plus(
            g8, w8, sa, sb, x, u, random_mask, mean, rstd, weight, bias, dropout_ratio, inv_scale, dz8, wl, cnt, cap,
            do8, e8,
        )
        dweight = torch.empty((l11.D,), dtype=weight.dtype, device=x.device)
        dbias = torch.empty((l11.D,), dtype=weight.dtype, device=x.device)
        l11.dwdb_finalize(dwp, dbp, dweight, dbias, P)
        du = hstu_du_e5m2.placeholder_du(u, m, l11.D)
        emit = hstu_du_e5m2.DuE5m2Emit(
            dz8=dz8,
            u_cols=(0, l11.D),
            u_amax=amaxp.amax(),
            u_colsum=ducs.sum(dim=0),
            inv_scale=inv_scale,
            cast_scale=cast_scale,
            rows=m,
            width=du_row_width,
            site_key=site_key,
            generation=generation,
            buffer_ptr=0,  # no d(uvqk) buffer handed over (placeholder du), as run_e5
        )
    except Exception as error:  # noqa: BLE001 -- latch DO8 only; the caller runs L11's own launch
        _disable(f"e5 plus: {type(error).__name__}: {str(error)[:600]}")
        return None
    _count("emitted")
    if not l11._active_e5_logged:
        # L11's own e5 ACTIVE line, once per process, printed here because this call ran L11's e5m2-du mode through
        # the DO8 kernel (l11.run_e5, which prints it, did not run). Same text as l11_outproj_lnmd.run_e5 plus a DO8
        # tail; impl= names the kernel that ran (DO8's own impl, "+vd(...)" only for the l11_do8.cu build), not
        # L11's .so.
        l11._active_e5_logged = True
        arm_print_all_ranks(
            f"armed OK: {l11.FAMILY} ACTIVE mode=e5 (du e5m2 producer contract) ({l11.KNOB}, with "
            f"DLRM_HSTU_FP8_BWD_DU_E5M2): one {l11.KERNEL_NAME} E5 launch replaces hipBLASLt F8B8BS NTB0 out-proj "
            f"dgrad + _ln_mul_dropout_bwd_dx_du_rng_v3 (du e5m2 EMIT: e5m2 u block + pad rows of dz8, amax/colsum "
            f"partials; no bf16 du, dy never in HBM); impl={_impl} M={m} Mp={int(dz8.shape[0])} P={P} "
            f"device={x.device}; launched as {KERNEL_NAME} ({ITEM}, + dO8/E8)"
        )
    return dx, du, dweight, dbias, emit, do8, e8


def publish(site: Any, dattn: torch.Tensor, do8: torch.Tensor, e8: torch.Tensor) -> bool:
    """BACKWARD, after the LAST write to dattn (the exact tensor returned as d(attn)), on the current stream: hand the
    record to the registry ('plus': bf16_written=True). True = published. A refusal is counted (the consumer then
    quantizes the bf16 d(attn): dO8/E8 exact) and never raises. A refused (or publish-latched) call is not
    bit-identical to DO8 off: DO8's launch already ran, so its dW/dB/u-colsum keep DO8's tile-walk fp32 order
    (the same fp32 summation-order change as L11's); d(attn), dz8 and amax are bitwise."""
    reg = registry()
    if reg is None or site is None:
        return False
    try:
        rec = reg.publish(site, dattn, do8, e8, producer="L11", bf16_written=True)
    except Exception as error:  # noqa: BLE001 -- the registry never raises for 'plus'; be fail-safe anyway
        _disable(f"publish: {type(error).__name__}: {str(error)[:300]}")
        return False
    if rec is None:
        _count("refused")
        _print_once(
            "refused",
            f"{FAMILY} NOT armed: hstu_do8_registry.publish refused a record (its refused{{...}} counter names the "
            "check) -> the C8 attention backward quantizes the bf16 d(attn)",
        )
        return False
    _count("published")
    global _active_logged
    if not _active_logged:
        _active_logged = True
        G, _, _, P = plan(int(_WL_MEMO.get("cap", 0))) if _WL_MEMO.get("cap") else (0, 0, 0, 0)
        arm_print_all_ranks(
            f"armed OK: {FAMILY} ACTIVE mode=e5 plus ({KNOB}, {ITEM}, with DLRM_HSTU_FP8_BWD_OUTPROJ_LNMD_FUSED, "
            f"DLRM_HSTU_FP8_BWD_DU_E5M2 and hstu_do8_registry): one {KERNEL_NAME} launch replaces "
            f"l11_outproj_lnmd_bwd (E5) and also emits the C8 attention backward's e4m3 dO + E8 (quantize_do_e4m3_blk "
            f"skipped when taken; bf16 d(attn) still written); impl={_impl} M={int(dattn.shape[0])} cap={_WL_MEMO.get('cap')} G={G} P={P} "
            f"device={dattn.device}"
        )
    return True


# ---------------------------------------------------------------------------- self-check / prebuild
_SELF_CHECK_SEED = 20261010
# Sequence lengths of the self-check (cycled to >= 32,768 rows): every in-tile group boundary (1, 7, 8, 9, 15, 16, 17,
# 63, 64, 65), multi-tile sequences, a zero-length sequence, and lengths beyond one tile row of a WG.
_SELF_CHECK_LENS = (1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 127, 128, 0, 200, 640, 1000, 4096, 2288, 333)


def self_check() -> Tuple[bool, str]:
    """DO8 vs L11's own E5 launch and the stock quantizer, on seeded SYNTHETIC tensors (a private generator: no
    training data, no global RNG draw, no FP8 scale state, no registry). Gates: the work list (count, status, entries
    vs a host list); DO8's dx, dz8 u block + pad rows, guard columns (prefilled), amax bitwise vs L11's launch_e5 on
    the same operands; dO8 / E8 (prefilled) bitwise vs quantize_do_e4m3_blk(L11's dx) on every row; u colsum, dW, dB
    relL2 vs L11's <= 1e-5 (fp32 summation order); all-zero token groups included (E8 127); finite data (as L11's own
    self-check: a NaN row would make every dW/dB column NaN and the relL2 gates vacuous)."""
    from generative_recommenders.ops.flydsl.hstu_do_fp8 import quantize_do_e4m3_blk
    from generative_recommenders.ops.triton import bwd_fp8
    from generative_recommenders.ops.triton import triton_hstu_linear as thl

    dev = torch.device("cuda", torch.cuda.current_device())
    lens: List[int] = []
    while sum(lens) < 32768:
        lens.extend(_SELF_CHECK_LENS)
    M, Dm, p = sum(lens), l11.D, 0.1
    offs_h = [0]
    for n_ in lens:
        offs_h.append(offs_h[-1] + n_)
    offs = torch.tensor(offs_h, dtype=torch.int64, device=dev)
    N = max(lens)
    g = torch.Generator(device=dev)
    g.manual_seed(_SELF_CHECK_SEED)
    g8f = torch.randn((M, Dm), generator=g, device=dev) * 1024.0
    # zero groups: sequence 5's first 8 rows (G8 zero -> dy zero -> d(attn) zero rows -> e = 0, E8 127)
    s5 = offs_h[5]
    g8f[s5:s5 + min(8, lens[5])] = 0.0
    g8 = g8f.clamp_(-57344.0, 57344.0).to(torch.float8_e5m2)
    del g8f
    w8 = (torch.randn((l11.NOUT, Dm), generator=g, device=dev) * 90.0).clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)
    sa = torch.full((1,), 2.0**-19, dtype=torch.float32, device=dev)
    sb = torch.full((1,), 0.0613 / 448.0, dtype=torch.float32, device=dev)
    x = (torch.randn((M, Dm), generator=g, device=dev) * 0.5).to(torch.bfloat16)
    u = torch.randn((M, Dm), generator=g, device=dev).to(torch.bfloat16)
    wn = (1.0 + 0.05 * torch.randn((Dm,), generator=g, device=dev)).to(torch.bfloat16)
    bn = (0.05 * torch.randn((Dm,), generator=g, device=dev)).to(torch.bfloat16)
    xf = x.float()
    mean = xf.mean(dim=1)
    rstd = torch.rsqrt(xf.var(dim=1, unbiased=False) + 1e-6)
    del xf
    mask = thl._create_dropout_mask(
        N=M, D=Dm, BLOCK_D=Dm, concat_u=True, concat_x=True, dropout_ratio=p, seed=_SELF_CHECK_SEED, device=dev,
    )
    grad_dtype, limit = bwd_fp8.grad_dtype_and_limit()
    pad = int(bwd_fp8.uvqk_tm_pad())
    rows_out = (M + pad - 1) // pad * pad
    inv = torch.full((1,), 0.7371 * 2.0**-3, dtype=torch.float32, device=dev)  # not a power of two: rounds
    # L11's own E5 launch (its .so, its flags), into a prefilled dz8
    dz8_s = torch.full((rows_out, l11.DU_ROW), 0x5A, dtype=torch.uint8, device=dev)
    dx_s, dwp_s, dbp_s, ducs_s, amaxp_s, P_s = l11.launch_e5(
        g8, w8, sa, sb, x, u, mask, mean, rstd, wn, bn, p, inv, dz8_s.view(grad_dtype)
    )
    # DO8: the device list from this offsets tensor (raw: the production memo stays untouched), prefilled outputs
    b = len(lens)
    cap = wl_cap(M, b)
    wl = torch.full((cap, 2), -7, dtype=torch.int32, device=dev)
    cnt = torch.full((2,), -7, dtype=torch.int32, device=dev)
    assert _fn_wl_build is not None
    rc = int(_fn_wl_build(offs.data_ptr(), 1, b, M, max(1, min(128, (N + 63) // 64)), wl.data_ptr(), cap,
                          cnt.data_ptr(), torch.cuda.current_stream(dev).cuda_stream))
    if rc > 0:
        return False, f"l11do8_wl_build_launch rc={rc}"
    dz8_d = torch.full((rows_out, l11.DU_ROW), 0x5A, dtype=torch.uint8, device=dev)
    do8 = torch.full((M, Dm), 0x5A, dtype=torch.uint8, device=dev)
    e8 = torch.full((M, H), 0xA5, dtype=torch.uint8, device=dev)
    dx_d, dwp_d, dbp_d, ducs_d, amaxp_d, P_d = launch_e5_plus(
        g8, w8, sa, sb, x, u, mask, mean, rstd, wn, bn, p, inv, dz8_d.view(grad_dtype), wl, cnt, cap,
        do8.view(torch.float8_e4m3fn), e8,
    )
    q8, qe8 = quantize_do_e4m3_blk(dx_s.view(M, H, HD), offs, N)
    dw_s = torch.empty((Dm,), dtype=wn.dtype, device=dev)
    db_s = torch.empty((Dm,), dtype=wn.dtype, device=dev)
    dw_d = torch.empty((Dm,), dtype=wn.dtype, device=dev)
    db_d = torch.empty((Dm,), dtype=wn.dtype, device=dev)
    l11.dwdb_finalize(dwp_s, dbp_s, dw_s, db_s, P_s)
    l11.dwdb_finalize(dwp_d, dbp_d, dw_d, db_d, P_d)
    torch.cuda.synchronize()

    def nbytes(a: torch.Tensor, c: torch.Tensor) -> int:
        a8 = a.contiguous().view(torch.uint8).reshape(-1)
        c8 = c.contiguous().view(torch.uint8).reshape(-1)
        if a8.numel() != c8.numel():
            return -1
        return int((a8 != c8).sum())

    def rel(a: torch.Tensor, c: torch.Tensor) -> float:
        a64, c64 = a.double(), c.double()
        fin = torch.isfinite(a64) & torch.isfinite(c64)
        return float((a64[fin] - c64[fin]).norm() / c64[fin].norm().clamp_min(1e-300))

    # host reference list (sequence-major, every tile, no gaps: offs[0] == 0, offs[B] == M)
    ref: List[Tuple[int, int]] = []
    for i, n_ in enumerate(lens):
        for t in range(0, n_, 64):
            ref.append((offs_h[i] + t, min(64, n_ - t)))
    cnt_h = cnt.tolist()
    wl_ok = cnt_h == [len(ref), 0] and wl[: len(ref)].tolist() == [list(e) for e in ref]
    dx_m = nbytes(dx_d, dx_s)
    dz_m = nbytes(dz8_d[:, :Dm], dz8_s[:, :Dm])
    guard = int((dz8_d[:, Dm:] != 0x5A).sum())
    amax_eq = bool(torch.equal(amaxp_d.amax().reshape(1).view(torch.int32), amaxp_s.amax().reshape(1).view(torch.int32)))
    do8_m = nbytes(do8, q8)
    e8_m = nbytes(e8, qe8)
    r_cs, r_dw, r_db = rel(ducs_d.sum(0), ducs_s.sum(0)), rel(dw_d, dw_s), rel(db_d, db_s)
    e8_127 = int((qe8 == 127).sum())
    finite = all(bool(torch.isfinite(t).all()) for t in (dx_d.float(), dw_d.float(), db_d.float(), ducs_d))
    detail = (
        f"M={M} B={b} cap={cap} tiles={cnt_h[0]} status={cnt_h[1]} P={P_d}: list ok {int(wl_ok)}; vs L11 E5 dx mism "
        f"{dx_m}, dz8 u+pad mism {dz_m}, guard bytes changed {guard}, amax eq {int(amax_eq)}; vs quantize_do_e4m3_blk "
        f"dO8 mism {do8_m}, E8 mism {e8_m} (E8=127 bytes {e8_127}); relL2 colsum {r_cs:.2e} dW {r_dw:.2e} dB "
        f"{r_db:.2e}; finite {int(finite)}"
    )
    ok = (
        wl_ok
        and dx_m == 0
        and dz_m == 0
        and guard == 0
        and amax_eq
        and do8_m == 0
        and e8_m == 0
        and r_cs <= 1e-5
        and r_dw <= 1e-5
        and r_db <= 1e-5
        and e8_127 > 0
        and finite
    )
    return ok, detail


def prebuild(l11_ready: bool, e5_expected: bool, self_check_fn: Optional[Callable[[], Tuple[bool, str]]] = None) -> bool:
    """Before RUN_START (triton_fp8.prebuild_extensions -> triton_hstu_linear.prebuild_l11_do8, AFTER L11's prebuild):
    arch check, build + load + DO8's resource gate, then the self-check. Knob off: no-op, no line. Without a prebuilt
    L11 or without L11's e5m2-du mode (DLRM_HSTU_FP8_BWD_DU_E5M2): one INERT line, nothing built. Any build / load / gate / self-check failure
    latches DO8 off (DISABLED line; L11 untouched). A self-check exception whose text is a device fault re-raises
    (abort before RUN_START, L11's policy)."""
    global _prebuilt
    if not requested():
        return False
    if not l11_ready:
        _print_once(
            "inert-l11",
            f"{FAMILY} INERT: needs L11 prebuilt and enabled in this process "
            "(DLRM_HSTU_FP8_BWD_OUTPROJ_LNMD_FUSED=1, prebuild OK); nothing built",
        )
        return False
    if not e5_expected:
        _print_once(
            "inert-e5",
            f"{FAMILY} INERT: needs the du e5m2 mode (DLRM_HSTU_FP8_BWD_DU_E5M2=1, not latched): DO8 exists only for "
            "L11's e5m2-du mode; nothing built",
        )
        return False
    if _broken is not None:
        return False
    t0 = time.perf_counter()
    try:
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise RuntimeError("no ROCm GPU")
        arch = l11.device_arch(torch.cuda.current_device())
        if arch != _ARCH:
            raise RuntimeError(f"arch {arch or '<unknown>'} is not {_ARCH}")
        _load()
    except Exception as error:  # noqa: BLE001 -- build / gate / load: never a device fault, always latch DO8
        _disable(f"prebuild build/load: {type(error).__name__}: {str(error)[:600]}")
        return False
    try:
        ok, detail = (self_check_fn or self_check)()
        torch.cuda.synchronize()
    except Exception as error:  # noqa: BLE001
        msg = f"{type(error).__name__}: {error}"
        if l11._DEVICE_FAULT.search(msg):
            arm_print_all_ranks(f"{FAMILY} prebuild DEVICE FAULT: {msg[:600]}")
            raise
        _disable(f"prebuild self-check: {msg[:600]}")
        return False
    if not ok:
        _disable(f"prebuild self-check mismatch: {detail}")
        return False
    _prebuilt = True
    so_key = _lib_path.parent.name if _lib_path is not None else "?"
    arm_print_all_ranks(
        f"armed OK: {FAMILY} prebuild before run_start: built+loaded+self-check OK {time.perf_counter() - t0:.1f}s "
        f"(VGPR {_resources.get('vgpr')}, spill {_resources.get('vgpr_spill')}, scratch {_resources.get('scratch')}, "
        f"LDS {_resources.get('lds')}, SGPR spill {_resources.get('sgpr_spill')}, sched={_resources.get('sched')}; "
        f"so {so_key}; {detail})"
    )
    return True
