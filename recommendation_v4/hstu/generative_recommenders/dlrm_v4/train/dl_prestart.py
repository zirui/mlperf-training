"""Streaming DataLoader pre-start bookkeeping: startup timeline + read probes.

Two jobs, both one-time and startup-only (no per-step cost beyond a cached
bool check per worker fetch):

1. **Startup timeline** (always on, rank 0, ONE log line). The trainer's
   ``[boundary] train_ts=<first> train first-batch data-wait=`` line times the
   whole first ``train_pipeline.progress()`` call: the pipeline fill (the first
   batch pulls, i.e. the real data wait), the fx trace and the first
   forward/backward with every first-use JIT. ``mark(name)`` records the first wall time
   of each named startup point; ``report()`` prints them relative to RUN_START
   (``set_run_start``) together with the worker-side first-sample-read stamps,
   so the wait splits into data vs compute.

2. **Worker first-read stamps** (the compliance proof for the pre-start).
   ``new_read_slot(name)`` allocates a shared double in the main process BEFORE
   the DataLoader forks; ``note_read(slot)`` is called by the dataset wrapper in
   the worker on its first fetch and stores the wall time once. With
   ``DL_PRESTART=1`` the worker pools are forked before INIT_STOP/RUN_START
   with an EMPTY sampler window, so no index reaches a worker and no sample is
   read, prefetched or collated before RUN_START; the report line shows
   ``workers_ready`` < 0 and ``train_first_read`` > 0 relative to RUN_START, and
   any read stamped before RUN_START is reported as a COMPLIANCE VIOLATION.

Wall times (``time.time()``) are used so worker and main-process stamps share
one clock (MLLOG ``time_ms`` is also wall time).
"""

from __future__ import annotations

import logging
import multiprocessing as _mp
import os
import time
from typing import Dict, List, Optional, Tuple

logger: logging.Logger = logging.getLogger(__name__)

# name -> first wall time (main process only).
_MARKS: Dict[str, float] = {}
# (label, wall) reference; RUN_START on a cold start, else loop entry.
_REF: Optional[Tuple[str, float]] = None
# slot name -> multiprocessing.Value('d'), created before the pools fork.
_READ_SLOTS: Dict[str, "_mp.sharedctypes.Synchronized"] = {}
_REPORTED: bool = False
# True until the first-step report; hot-path marks check this bool only.
ACTIVE: bool = True

# Report order (anything else marked is appended after these).
_ORDER: List[str] = [
    "prestart_begin",
    "prestart_workers_ready",
    "loop_entry",
    "dataset_built",
    "eval_pool_ready",
    "train_pools_ready",
    "first_progress_start",
    "pull1",
    "fx_begin",
    "idist0_begin",
    "idist0_done",
    "pull2",
    "fwd_begin",
    "fwd_end",
    "pull3",
    "enq3_done",
    "idist1_begin",
    "bwd_begin",
    "bwd_end",
    "opt_begin",
    "opt_end",
    "deq_wait_begin",
    "first_progress_done",
]


def set_run_start(wall: Optional[float] = None) -> None:
    """Record the RUN_START reference (call right after the RUN_START marker)."""
    global _REF
    _REF = ("run_start", time.time() if wall is None else float(wall))


def run_start_wall() -> Optional[float]:
    return _REF[1] if _REF is not None and _REF[0] == "run_start" else None


_EXPECT_RUN_START: List[bool] = [False]


def expect_run_start() -> None:
    """This process will log RUN_START (MLPerf cold start): from now on no
    sample may be handed out or prefetched until ``set_run_start`` ran."""
    _EXPECT_RUN_START[0] = True


def require_run_start(what: str) -> None:
    """Hard guard for the sampler release / cold prefetch: refuse to touch the
    dataset before RUN_START was logged (rules L549), instead of relying on
    call order."""
    if _EXPECT_RUN_START[0] and run_start_wall() is None:
        raise RuntimeError(
            f"[dlstart] {what} before RUN_START: the dataset must not be touched "
            "before the clock starts"
        )


def page_cache_snapshot(root: str, rank: int) -> Optional[str]:
    """Count page-cache-resident pages of every .npy under ``root`` and log
    them, separating the header page of each file from sample pages. Uses
    mincore(2) on a PROT_READ mapping, which reads nothing. Called right before
    RUN_START: after the launcher's cache flush, the sample-page count is the
    evidence that no data was read before the clock. Diagnostics only."""
    import ctypes
    import glob

    import numpy as np

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int, ctypes.c_long]
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        page = os.sysconf("SC_PAGE_SIZE")
        prot_read, map_shared, chunk = 0x1, 0x01, 1 << 30
        files = sorted(glob.glob(os.path.join(root, "**", "*.npy"), recursive=True))
        total = header = sample = 0
        hot: List[str] = []
        for p in files:
            size = os.path.getsize(p)
            if size == 0:
                continue
            total += size
            fd = os.open(p, os.O_RDONLY)
            try:
                addr = libc.mmap(None, size, prot_read, map_shared, fd, 0)
            finally:
                os.close(fd)
            if addr in (None, ctypes.c_void_p(-1).value):
                raise OSError(ctypes.get_errno(), f"mmap {p}")
            res = 0
            try:
                for off in range(0, size, chunk):
                    n = min(chunk, size - off)
                    vec = (ctypes.c_ubyte * ((n + page - 1) // page))()
                    if libc.mincore(ctypes.c_void_p(addr + off), n, vec) != 0:
                        raise OSError(ctypes.get_errno(), f"mincore {p}")
                    bits = np.frombuffer(vec, dtype=np.uint8) & 1
                    r = int(bits.sum())
                    if off == 0 and bits[0]:
                        header += page
                        r -= 1
                    res += r
            finally:
                libc.munmap(ctypes.c_void_p(addr), size)
            sample += res * page
            if res:
                hot.append(f"{os.path.basename(p)}:{res * page >> 10}KiB")
        line = (
            f"[dlstart] page cache at RUN_START (node of rank {rank}): sample pages "
            f"resident {sample >> 10} KiB, header pages {header >> 10} KiB, over "
            f"{len(files)} .npy files ({total / 2**30:.1f} GiB)"
            + (f"; resident: {', '.join(hot[:8])}" if hot else "")
        )
        logger.warning(line)
        return line
    except Exception as e:  # diagnostics only; never fails the run
        logger.warning("[dlstart] page-cache snapshot skipped: %r", e)
        return None


def mark(name: str) -> None:
    """Record the first wall time of a named startup point (idempotent)."""
    if name not in _MARKS:
        _MARKS[name] = time.time()


def marked(name: str) -> bool:
    return name in _MARKS


def new_read_slot(name: str):
    """Shared first-read stamp for one worker pool. MUST be created in the main
    process before that pool forks (fork inherits the shared page)."""
    slot = _READ_SLOTS.get(name)
    if slot is None:
        slot = _mp.get_context("fork").Value("d", 0.0)
        _READ_SLOTS[name] = slot
    return slot


def note_read(slot) -> None:
    """Worker side: stamp the first sample read of this pool (first writer wins)."""
    if slot is None:
        return
    with slot.get_lock():
        if slot.value == 0.0:
            slot.value = time.time()


def read_stamps() -> Dict[str, float]:
    """Pool name -> first-read wall time (pools that read nothing are omitted)."""
    out: Dict[str, float] = {}
    for name, slot in _READ_SLOTS.items():
        v = float(slot.value)
        if v > 0.0:
            out[name] = v
    return out


def compliance_violations() -> List[str]:
    """Read stamps that precede RUN_START (only meaningful on a cold start)."""
    rs = run_start_wall()
    if rs is None:
        return []
    return [
        f"{name}@{v - rs:+.3f}s" for name, v in sorted(read_stamps().items()) if v < rs
    ]


def report(rank: int, prestarted: bool) -> Optional[str]:
    """One summary line (rank 0) + a violation warning on ANY rank. Idempotent."""
    global _REPORTED, ACTIVE
    ACTIVE = False
    if _REPORTED:
        return None
    _REPORTED = True
    bad = compliance_violations()
    if bad:
        logger.error(
            "[dlstart] COMPLIANCE VIOLATION rank=%d: sample read before RUN_START: %s",
            rank,
            ", ".join(bad),
        )
    if rank != 0:
        return None
    if _REF is not None:
        ref_label, ref = _REF
    else:
        ref_label, ref = "loop_entry", _MARKS.get("loop_entry", time.time())
    names = [n for n in _ORDER if n in _MARKS] + sorted(
        n for n in _MARKS if n not in _ORDER
    )
    parts = [f"{n}={_MARKS[n] - ref:+.3f}" for n in names]
    reads = read_stamps()
    parts += [f"{n}_first_read={v - ref:+.3f}" for n, v in sorted(reads.items())]
    # Split of the first progress(): data (start -> first batch at the main
    # process), pipeline fill (first -> third pull), the rest (fx trace +
    # first fwd/bwd/opt incl. first-use JIT; may include later pulls).
    split = []
    t0 = _MARKS.get("first_progress_start")
    t1 = _MARKS.get("pull1")
    tn = _MARKS.get("pull3", _MARKS.get("pull2", t1))
    te = _MARKS.get("first_progress_done")
    if t0 is not None and t1 is not None and te is not None:
        split.append(f"data_wait={t1 - t0:.3f}")
        split.append(f"fill_pulls={tn - t1:.3f}")
        split.append(f"after_pulls={te - tn:.3f}")
        split.append(f"first_progress={te - t0:.3f}")
    # Host-side sub-split of the first progress() (no syncs added; each segment
    # ends at a host return, so async GPU tails land in the next segment that
    # blocks on the device).
    for lab, a, b in (
        ("fx", "fx_begin", "idist0_begin"),
        ("idist0", "idist0_begin", "idist0_done"),
        ("fwd", "fwd_begin", "fwd_end"),
        ("enq3", "fwd_end", "enq3_done"),
        ("idist1", "idist1_begin", "bwd_begin"),
        ("bwd", "bwd_begin", "bwd_end"),
        ("opt", "opt_begin", "opt_end"),
        ("deq", "opt_end", "first_progress_done"),
    ):
        if a in _MARKS and b in _MARKS:
            split.append(f"{lab}={_MARKS[b] - _MARKS[a]:.3f}")
    line = (
        f"[dlstart] timeline s rel. {ref_label} (prestart={'on' if prestarted else 'off'}): "
        + " ".join(parts)
        + (" | split " + " ".join(split) if split else "")
        + (" | COMPLIANCE VIOLATION" if bad else (" | compliance OK" if ref_label == "run_start" else ""))
    )
    logger.info(line)
    return line


def _reset_for_tests() -> None:
    global _REF, _REPORTED
    global ACTIVE
    _MARKS.clear()
    _READ_SLOTS.clear()
    _REF = None
    _REPORTED = False
    ACTIVE = True
