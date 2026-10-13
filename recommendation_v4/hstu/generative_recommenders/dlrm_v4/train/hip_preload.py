# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict
"""Targeted HIP code-object preload before RUN_START (default ON via the yambda_5b.gin binding).

ROCm 7.2.1 loads statically registered kernels lazily (HIP deferred loading, HIP_ENABLE_DEFERRED_LOADING unset = 1):
the FIRST launch of any kernel of a translation unit (TU) of libtorch_hip.so / fbgemm_gpu*.so digests that TU's fat
binary (COMGR unbundle + zstd decompression for libtorch_hip) and loads its gfx950 code object on the current device.
The TUs a production step needs are first launched inside training step 1, i.e. inside the timed region.

hipFuncGetAttributes(&attr, host_stub) on one kernel stub per needed TU does exactly that load and nothing else: it
launches no kernel, allocates no tensor, and touches no RNG, model, optimizer or data state. The stubs come from the
checked-in table hip_preload_table.jsonl (generated from 1n/2n/4n production traces, plus the rows listed in its
header "amendments": step-1-only first-use code objects that steady-step traces cannot show), keyed by library
basename + mangled symbol:
  * mode "dlsym":  dlsym in the ALREADY-LOADED library (dlopen RTLD_NOLOAD; a library that is not loaded is skipped);
  * mode "symtab": local stubs (anonymous-namespace kernels are not exported): load base + st_value, used only when
    the loaded library's GNU build-id (read from its mapped ELF notes) equals the table's, and the base derived from
    a dlsym'd anchor symbol equals the library's offset-0 mapping in /proc/self/maps.
Every entry is fail-open: a missing library/symbol, a build-id mismatch or a HIP error is counted and skipped; a HIP
error is cleared with hipGetLastError so no later torch launch check sees it. Never raises.

MLPerf (training_rules.adoc "Run Results" L545-549): model initialization before the clock may include work that does
not touch the dataset; code-object loading is library initialization (same class as the Triton/HIP compile, BLAS and
FlyDSL warmups already run before RUN_START). It reads no sample and leaves all training state bit-identical.

Knob (gin apply_env_bootstrap.DLRM_HSTU_FP8_HIP_PRELOAD; the env bootstrap writes it into the environment before this
module reads it):
  DLRM_HSTU_FP8_HIP_PRELOAD   default ON via the yambda_5b.gin binding (True -> "1"); code default unset =
                              off (no HIP lookup, no log line). A pre-set env value wins over the binding (_bind_bool):
                              0/off = off, i.e. "0" is the kill switch; 1/on = the checked-in table, entries of this run
                              shape (<nodes>n) only; "all" = every entry; any other value = path of a table file ("all"
                              and a path are env-only values).
  DLRM_HSTU_FP8_HIP_PRELOAD_REPORT=<dir>  optional per-rank JSON with per-entry milliseconds (diagnostic, env only).
Arming line (rank 0): "armed OK: [hip-preload] ...", or "[hip-preload] NOT armed: ..." when nothing could be loaded.
"""

import ctypes
import json
import logging
import os
import struct
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger: logging.Logger = logging.getLogger(__name__)

ENV = "DLRM_HSTU_FP8_HIP_PRELOAD"
REPORT_ENV = "DLRM_HSTU_FP8_HIP_PRELOAD_REPORT"
DEFAULT_TABLE: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hip_preload_table.jsonl")
ARMED = "armed OK: [hip-preload]"
NOT_ARMED = "[hip-preload] NOT armed:"
_OFF = ("", "0", "off", "false", "no", "none")
_ON = ("1", "on", "true", "yes")
# An entry faster than this was already loaded (a hash lookup).
ALREADY_LOADED_MS = 0.05

RTLD_NOLOAD: int = getattr(os, "RTLD_NOLOAD", 4)
RTLD_LAZY: int = getattr(os, "RTLD_LAZY", 1)


def resolve_table(value: Optional[str]) -> Tuple[Optional[str], bool]:
    """env value -> (table path or None = off, use every shape). Pure."""
    v = (value or "").strip()
    if v.lower() in _OFF:
        return None, False
    if v.lower() in _ON:
        return DEFAULT_TABLE, False
    if v.lower() == "all":
        return DEFAULT_TABLE, True
    return v, True


def load_table(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], int]:
    """(header, entries, n_bad_lines). A bad line is counted, not raised; a missing/unreadable file raises OSError."""
    header: Dict[str, Any] = {}
    entries: List[Dict[str, Any]] = []
    bad = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if not isinstance(row, dict):
                bad += 1
                continue
            kind = row.get("kind")
            if kind == "header":
                header = row
            elif (kind == "entry" and isinstance(row.get("lib"), str) and isinstance(row.get("sym"), str)
                  and row.get("mode") in ("dlsym", "symtab")
                  and (row["mode"] == "dlsym" or isinstance(row.get("value"), int))):
                entries.append(row)
            else:
                bad += 1
    return header, entries, bad


def select_entries(header: Dict[str, Any], entries: List[Dict[str, Any]], shape: Optional[str],
                   use_all: bool) -> Tuple[List[Dict[str, Any]], int]:
    """Entries for this run shape ('1n', '2n', ...): those whose 'sets' contain it. A shape the table was not generated
    for, or use_all, keeps every entry. Returns (selected, n_other_shape)."""
    known = set((header.get("traces") or {}).keys())
    if use_all or not shape or shape not in known:
        return list(entries), 0
    sel = [e for e in entries if shape in (e.get("sets") or [shape])]
    return sel, len(entries) - len(sel)


# ----------------------------------------------------------------------------------------- process introspection
def mapped_libraries() -> Dict[str, Dict[str, int]]:
    """{path: {'base': start of the offset-0 mapping}} of file-backed mappings of this process."""
    out: Dict[str, Dict[str, int]] = {}
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.split(None, 5)
                if len(parts) < 6:
                    continue
                path = parts[5].strip()
                if not path.startswith("/"):
                    continue
                lo = int(parts[0].split("-")[0], 16)
                if int(parts[2], 16) == 0:
                    cur = out.setdefault(path, {"base": lo})
                    cur["base"] = min(cur["base"], lo)
    except OSError:
        pass
    return out


def find_loaded(basename: str, maps: Dict[str, Dict[str, int]]) -> List[str]:
    return sorted(p for p in maps if os.path.basename(p) == basename)


def memory_build_id(base: int) -> Optional[str]:
    """GNU build-id from the ELF image mapped at `base` (ELF header + PT_NOTE of a loaded library). The caller has
    checked that `base` is the start of the library's offset-0 mapping."""
    try:
        ehdr = ctypes.string_at(base, 64)
        if ehdr[:4] != b"\x7fELF" or ehdr[4] != 2:
            return None
        e_phoff = struct.unpack_from("<Q", ehdr, 0x20)[0]
        e_phentsize, e_phnum = struct.unpack_from("<HH", ehdr, 0x36)
        ph = ctypes.string_at(base + e_phoff, e_phentsize * e_phnum)
        for i in range(e_phnum):
            p_type, _fl, _off, p_vaddr, _pa, _fs, p_memsz, _al = struct.unpack_from("<IIQQQQQQ", ph, i * e_phentsize)
            if p_type != 4:  # PT_NOTE
                continue
            notes = ctypes.string_at(base + p_vaddr, p_memsz)
            q = 0
            while q + 12 <= len(notes):
                namesz, descsz, ntype = struct.unpack_from("<III", notes, q)
                name = notes[q + 12:q + 12 + namesz]
                dq = q + 12 + ((namesz + 3) & ~3)
                if ntype == 3 and name.rstrip(b"\0") == b"GNU":
                    return notes[dq:dq + descsz].hex()
                q = dq + ((descsz + 3) & ~3)
    except Exception:  # noqa: BLE001
        return None
    return None


class _Lib:
    """One already-loaded library: dlsym handle, and (if verified) the base for symtab entries."""

    def __init__(self, handle: Optional[ctypes.CDLL], path: Optional[str], symtab_base: Optional[int],
                 reason: str) -> None:
        self.handle = handle
        self.path = path
        self.symtab_base = symtab_base
        self.reason = reason  # why symtab entries are unusable ('' = usable)


def _dlsym(handle: ctypes.CDLL, sym: str) -> Optional[int]:
    try:
        return ctypes.cast(handle[sym], ctypes.c_void_p).value
    except (AttributeError, ValueError, TypeError):
        return None


def open_library(basename: str, meta: Dict[str, Any], maps: Dict[str, Dict[str, int]]) -> _Lib:
    paths = find_loaded(basename, maps)
    if not paths:
        return _Lib(None, None, None, "nolib")
    want = meta.get("build_id")
    chosen: Optional[_Lib] = None
    for path in paths:
        try:
            handle = ctypes.CDLL(path, mode=RTLD_NOLOAD | RTLD_LAZY)
        except OSError:
            continue
        base = maps[path]["base"]
        reason = ""
        anchor = meta.get("anchor") or {}
        a_addr = _dlsym(handle, anchor["sym"]) if anchor.get("sym") else None
        if a_addr is None or not isinstance(anchor.get("value"), int):
            reason = "anchor"
        elif a_addr - int(anchor["value"]) != base:
            reason = "base"
        elif not want or memory_build_id(base) != want:
            reason = "buildid"
        lib = _Lib(handle, path, base if not reason else None, reason)
        if not reason:
            return lib
        chosen = chosen or lib
    return chosen or _Lib(None, None, None, "nolib")


class _Hip:
    """hipFuncGetAttributes / hipGetLastError / hipGetDevice of the HIP runtime the kernels are registered with.
    Resolved with dlsym through the already-loaded libtorch_hip.so handle (its own dependency tree, i.e. exactly the
    libamdhip64 libtorch_hip binds to), else from the loaded libamdhip64 itself. Never dlopens a new library."""

    def __init__(self, maps: Dict[str, Dict[str, int]]) -> None:
        lib: Optional[ctypes.CDLL] = None
        self.path: str = ""
        for via in find_loaded("libtorch_hip.so", maps) + sorted(
                p for p in maps if os.path.basename(p).startswith("libamdhip64.so")):
            try:
                cand = ctypes.CDLL(via, mode=RTLD_NOLOAD | RTLD_LAZY)
                cand.hipFuncGetAttributes  # noqa: B018  (dlsym probe)
            except (OSError, AttributeError):
                continue
            lib, self.path = cand, via
            break
        if lib is None:
            raise OSError("no loaded library resolves hipFuncGetAttributes (libtorch_hip/libamdhip64 not loaded)")
        self._fga = lib.hipFuncGetAttributes
        self._fga.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._fga.restype = ctypes.c_int
        self._gle = lib.hipGetLastError
        self._gle.argtypes = []
        self._gle.restype = ctypes.c_int
        self._gdev = lib.hipGetDevice
        self._gdev.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self._gdev.restype = ctypes.c_int
        # hipFuncAttributes is ~100 bytes in ROCm 7.x; 1 KiB leaves room for any layout growth.
        self._buf = ctypes.create_string_buffer(1024)

    def func_attributes(self, addr: int) -> int:
        return int(self._fga(self._buf, ctypes.c_void_p(addr)))

    def clear_error(self) -> int:
        return int(self._gle())

    def device(self) -> int:
        d = ctypes.c_int(-1)
        return d.value if self._gdev(ctypes.byref(d)) == 0 else -1


def preload_entries(entries: List[Dict[str, Any]], libs_meta: Dict[str, Dict[str, Any]],
                    func_attributes: Callable[[int], int], clear_error: Callable[[], int],
                    maps: Optional[Dict[str, Dict[str, int]]] = None,
                    opener: Callable[[str, Dict[str, Any], Dict[str, Dict[str, int]]], _Lib] = open_library,
                    ) -> Dict[str, Any]:
    """Core loop (testable without a GPU: pass fake func_attributes/clear_error). Fail-open per entry."""
    maps = mapped_libraries() if maps is None else maps
    cnt = {"ok": 0, "already": 0, "nolib": 0, "nosym": 0, "symtab_unverified": 0, "hip": 0, "error": 0}
    rows: List[Dict[str, Any]] = []
    libs: Dict[str, _Lib] = {}
    t_all = time.perf_counter()
    for e in entries:
        name = e["lib"]
        try:
            if name not in libs:
                libs[name] = opener(name, libs_meta.get(name) or {}, maps)
            lib = libs[name]
            if lib.handle is None:
                cnt["nolib"] += 1
                rows.append({"lib": name, "bundle": e.get("bundle"), "res": "nolib"})
                continue
            if e["mode"] == "dlsym":
                addr = _dlsym(lib.handle, e["sym"])
            elif lib.symtab_base is not None:
                addr = lib.symtab_base + int(e["value"])
            else:
                cnt["symtab_unverified"] += 1
                rows.append({"lib": name, "bundle": e.get("bundle"), "res": "symtab:" + lib.reason})
                continue
            if not addr:
                cnt["nosym"] += 1
                rows.append({"lib": name, "bundle": e.get("bundle"), "res": "nosym"})
                continue
            t = time.perf_counter()
            rc = func_attributes(addr)
            ms = (time.perf_counter() - t) * 1e3
            if rc != 0:
                clear_error()
                cnt["hip"] += 1
                rows.append({"lib": name, "bundle": e.get("bundle"), "res": f"hip{rc}", "ms": ms})
                continue
            cnt["ok"] += 1
            if ms < ALREADY_LOADED_MS:
                cnt["already"] += 1
            rows.append({"lib": name, "bundle": e.get("bundle"), "res": "ok", "ms": round(ms, 4),
                         "est_ms": e.get("est_ms")})
        except Exception as error:  # noqa: BLE001  (fail-open per entry)
            cnt["error"] += 1
            rows.append({"lib": name, "bundle": e.get("bundle"), "res": f"error:{type(error).__name__}"})
            try:
                clear_error()
            except Exception:  # noqa: BLE001
                pass
    return {"counts": cnt, "rows": rows, "seconds": time.perf_counter() - t_all,
            "libs": {k: {"path": v.path, "symtab": v.symtab_base is not None, "reason": v.reason}
                     for k, v in libs.items()}}


def preload_from_env(rank: int = 0, world_size: int = 1, local_world_size: int = 1,
                     env: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    """train_ranker hook: after torch.cuda.set_device + the library imports + every warmup, right before RUN_START.
    Off (code default when the env is unset; yambda_5b.gin binds it on) = returns None without importing or logging
    anything. Never raises."""
    env = os.environ if env is None else env
    path, use_all = resolve_table(env.get(ENV))
    if path is None:
        return None
    t0 = time.perf_counter()
    try:
        header, entries, bad = load_table(path)
        shape = f"{max(1, world_size // max(1, local_world_size))}n"
        sel, other = select_entries(header, entries, shape, use_all)
        maps = mapped_libraries()
        hip = _Hip(maps)
        dev = hip.device()
        res = preload_entries(sel, header.get("libs") or {}, hip.func_attributes, hip.clear_error, maps)
    except Exception as error:  # noqa: BLE001
        if rank == 0:
            logger.warning("%s %s: %s (run continues; step 1 loads the code objects lazily)", NOT_ARMED,
                           type(error).__name__, error)
        return {"armed": False, "error": repr(error)}
    c = res["counts"]
    loaded_ms = sum(r.get("ms", 0.0) for r in res["rows"] if r.get("res") == "ok")
    failed = c["nolib"] + c["nosym"] + c["symtab_unverified"] + c["hip"] + c["error"]
    summary = {"armed": c["ok"] > 0, "table": path, "shape": shape, "selected": len(sel), "other_shape": other,
               "bad_lines": bad, "device": dev, "counts": c, "failed": failed, "seconds": time.perf_counter() - t0,
               "hip_seconds": res["seconds"], "loaded_ms": loaded_ms, "hip_via": hip.path, "libs": res["libs"]}
    report = env.get(REPORT_ENV, "")
    if report and report.lower() not in _OFF:
        try:
            os.makedirs(report, exist_ok=True)
            with open(os.path.join(report, f"hip_preload_rank{rank}.json"), "w") as f:
                json.dump(summary | {"rows": res["rows"]}, f, indent=1)
        except OSError as error:
            logger.warning("[hip-preload] report not written: %s", error)
    msg = (f"table={os.path.basename(path)} shape={shape} entries={len(sel)} (other_shape={other} bad_lines={bad}) "
           f"ok={c['ok']} already_loaded={c['already']} failed={failed} (nolib={c['nolib']} nosym={c['nosym']} "
           f"symtab_unverified={c['symtab_unverified']} hip={c['hip']} error={c['error']}) "
           f"in {summary['seconds']:.2f}s (hip {res['seconds']:.2f}s) dev={dev} before run_start; "
           f"no kernel launch, no tensor/RNG state")
    if rank == 0:
        if c["ok"] > 0:
            logger.info("%s %s", ARMED, msg)
        else:
            logger.warning("%s %s", NOT_ARMED, msg)
    elif failed:
        logger.warning("[hip-preload] rank %d: %s", rank, msg)
    return summary
