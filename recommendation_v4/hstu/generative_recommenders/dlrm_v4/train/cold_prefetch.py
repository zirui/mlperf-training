"""DL_COLD_PREFETCH: post-RUN_START parallel readahead of the first batches.

With the host page cache flushed before the trainer (MLPerf "Data State at
Start of Run"), the first training batches are built from cold storage by ONE
DataLoader worker per batch, i.e. a serial chain of synchronous mmap page
faults (queue depth ~1): ``positions[g] -> flat_uid[pos] -> user_start[uid] ->``
the per-sample scan window ``[max(user_start, pos - scan_window), pos]`` of five
``flat_*`` columns plus three scalar columns at ``pos``. In the START_TS=0
production shape the first batches come from tiny windows (1 batch each), so the
pipeline fill is several such serial batches back to back.

This module computes, in the rank's main process, exactly the byte ranges the
rank's first ``K`` batches will touch (same per-rank slice as
``StreamingWindowSampler.set_window``, same drop_last batching) and hands them to
the block layer in parallel (``posix_fadvise(WILLNEED)`` from a small thread
pool, or parallel ``pread`` in mode 2). The workers then hit the page cache.

Streaming read-ahead (``DL_COLD_PREFETCH_STREAM=1``, default off). The one-shot
prefetch covers only the first ``DL_COLD_PREFETCH_BATCHES`` batches. Early
windows hold one or two batches per rank. With the two-pool loader (DL_SINGLE_POOL=0) a window's pool
opens only when the window two places earlier has finished, so its
cross-window lookahead is ~2-4 batches; with the single chained pool
(DL_SINGLE_POOL=1, the gin default) the loader's lookahead crosses windows at
the full prefetch depth. Either way each later cold batch is still one
worker's serial fault chain. In streaming mode the same thread keeps walking the run's windows in
order after the first batch. It recomputes each window's per-rank batches
exactly as the sampler does (same indices, per-rank slice, drop_last, per-window
cap) and issues the same parallel WILLNEED per group of
``DL_COLD_PREFETCH_GROUP`` batches (default 4). It stays at most
``DL_COLD_PREFETCH_LEAD`` batches (default 64) ahead of the batches the training
pipeline has pulled (``note_pull`` / ``note_window_done``), skips batches already
pulled, and runs until the stream ends (``stop``); ``DL_COLD_PREFETCH_HORIZON``
caps it at that many per-rank batches.

Why the whole stream, not a horizon: the walk must not stop mid-run. The
WILLNEED walk caches exactly the pages each batch needs, whereas without it the
workers' first faults on a file region read the kernel's mmap read-around,
which also caches neighbouring users that later batches need. A walk that stops
then leaves the workers faulting on regions nothing has touched yet, and every
local rank hits the device at once. Stream mode preads the index levels inline
(``_gather``) rather than on the thread pool.

Rules (MLPerf):
  * Nothing here runs before ``release()``, which the loader calls when the
    sampler windows are installed, i.e. after RUN_START (inside the timed
    region, where overlapping I/O with compute is legal). ``start()`` refuses to
    run otherwise, and every file access goes through ``_open``/``_pread``/
    ``_fadvise`` so tests can prove it.
  * Advisory only: it never changes the bytes, order, windows or samples the
    model sees (the workers still read through their own mmaps).
  * Default off. Armed by ``DL_COLD_PREFETCH=1`` (fadvise) / ``=2`` (parallel
    pread), or by ``DL_PRESTART=2`` (pre-start + cold prefetch, for launchers
    that forward DL_PRESTART but not DL_COLD_PREFETCH). Optional
    ``DL_COLD_PREFETCH_BATCHES`` (default 8) and ``DL_COLD_PREFETCH_THREADS``
    (default 16). ``DL_COLD_PREFETCH_STREAM=1`` adds the streaming read-ahead
    to an armed prefetcher; it has no effect when the prefetcher is off.
Failures are swallowed with one warning line; the run is unaffected.
"""

from __future__ import annotations

import io
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)

PAGE: int = int(os.sysconf("SC_PAGE_SIZE")) if hasattr(os, "sysconf") else 4096
_CHUNK: int = 1 << 20  # max bytes per pread/fadvise call (spreads big runs over the pool)
_GAP_PAGES: int = 4  # merge runs separated by <= this many pages (fewer, larger requests)

# File-access hooks (tests monkeypatch these to prove nothing runs pre-release).
_open = os.open
_close = os.close
_pread = os.pread
_fadvise = getattr(os, "posix_fadvise", None)
_WILLNEED: int = getattr(os, "POSIX_FADV_WILLNEED", 3)

_RELEASED = threading.Event()
_RELEASE_WALL: List[float] = []
_THREADS: List[threading.Thread] = []

# Streaming read-ahead pacing. All three are written by the training (main)
# thread and only read by the prefetch thread; plain list cells, no locks
# (pacing is approximate by design).
#   _PULLED[0]  batches pulled so far by the chained window stream (warm-keep)
#   _DONE       [windows fully pulled, _PULLED value at that moment]
_PULLED: List[int] = [0]
_DONE: List[int] = [0, 0]
_STOP = threading.Event()
# Streaming mode merges like the one-shot (4 pages): ranges are ~one per user
# per column and users sit far apart in the int64 columns.
_STREAM_GAP_PAGES: int = _GAP_PAGES
_WAIT_S: float = 0.05  # pacing poll interval
# Default streaming horizon: the whole stream (see the module docstring for why a
# mid-run stop hurts). $DL_COLD_PREFETCH_HORIZON caps it in per-rank batches.
HORIZON_ALL: int = 1 << 62


def note_pull() -> None:
    """One batch pulled by the training pipeline (hot path: one int add)."""
    _PULLED[0] += 1


def note_window_done(n_done: int) -> None:
    """``n_done`` windows of the stream have been pulled completely."""
    _DONE[1] = _PULLED[0]
    _DONE[0] = int(n_done)


def stop() -> None:
    """End of the window stream: the streaming read-ahead exits."""
    _STOP.set()


def stream_on() -> bool:
    return os.environ.get("DL_COLD_PREFETCH_STREAM", "").strip().lower() in (
        "1", "true", "on", "yes",
    )

# yambda flat-store columns: (name, touched as the scan window [ss, pos] or the
# scalar [pos, pos]).
_SCAN_COLS: Tuple[str, ...] = (
    "flat_item_ids", "flat_timestamps",
    "flat_is_listen_plus", "flat_is_like", "flat_is_skip",
)
_SCALAR_COLS: Tuple[str, ...] = (
    "flat_played_ratio", "flat_event_types", "flat_is_organic",
)


# Pre-start level resolved by gin (prestart_streaming_loaders.enabled, which
# reads $DL_PRESTART first), set by utils.prestart_streaming_loaders. Level 2 =
# pre-start plus this prefetcher; an unset env var must not disable it.
_PRESTART_LEVEL: List[int] = [0]


def set_prestart_level(level: int) -> None:
    _PRESTART_LEVEL[0] = int(level)


def mode() -> int:
    """0 = off, 1 = fadvise WILLNEED, 2 = parallel pread. $DL_COLD_PREFETCH
    overrides; otherwise on (WILLNEED) when the pre-start level is 2."""
    raw = os.environ.get("DL_COLD_PREFETCH", "").strip()
    if raw:
        try:
            return max(0, min(2, int(raw)))
        except ValueError:
            return 0
    return 1 if _PRESTART_LEVEL[0] == 2 else 0


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, "") or default)
    except ValueError:
        return default


def release() -> None:
    """Mark the sampler release (after RUN_START). Idempotent. Refuses to run
    before RUN_START was logged when this process logs it (hard guard)."""
    from generative_recommenders.dlrm_v4.train import dl_prestart as _dlstart

    _dlstart.require_run_start("sampler release / cold prefetch")
    if not _RELEASED.is_set():
        _RELEASE_WALL.append(time.time())
        _RELEASED.set()


def released() -> bool:
    return _RELEASED.is_set()


class _NpyFile:
    """fd + header geometry of one .npy column (metadata only)."""

    def __init__(self, path: str, expect_len: Optional[int]) -> None:
        self.fd: int = _open(path, os.O_RDONLY)
        try:
            self.size: int = os.fstat(self.fd).st_size
            head = _pread(self.fd, 4096, 0)
            f = io.BytesIO(head)
            ver = np.lib.format.read_magic(f)
            rd = (
                np.lib.format.read_array_header_1_0
                if ver[0] == 1
                else np.lib.format.read_array_header_2_0
            )
            shape, _, dtype = rd(f)
            self.hdr: int = f.tell()
            self.dtype: np.dtype = np.dtype(dtype)
            self.isz: int = int(self.dtype.itemsize)
            n = int(shape[0]) if shape else 0
            if expect_len is not None and n != int(expect_len):
                raise ValueError(f"{path}: header len {n} != array len {expect_len}")
            # Short file: only the elements actually present are addressable.
            self.n: int = max(0, min(n, (self.size - self.hdr) // max(1, self.isz)))
            self.ra_cap: int = _willneed_cap(os.fstat(self.fd).st_dev)
        except Exception:
            _close(self.fd)
            raise

    def close(self) -> None:
        try:
            _close(self.fd)
        except OSError:
            pass


def _page_runs(lo_b: np.ndarray, hi_b: np.ndarray, size: int, gap_pages: int):
    """Byte ranges -> sorted, merged, page-aligned [start, end) runs within size."""
    lo = (lo_b // PAGE) * PAGE
    hi = np.minimum(((hi_b + PAGE - 1) // PAGE) * PAGE, size)
    keep = hi > lo
    lo, hi = lo[keep], hi[keep]
    if lo.size == 0:
        e = np.empty(0, dtype=np.int64)
        return e, e
    order = np.argsort(lo, kind="stable")
    lo, hi = lo[order], hi[order]
    cm = np.maximum.accumulate(hi)
    new = np.ones(lo.size, dtype=bool)
    new[1:] = lo[1:] > cm[:-1] + gap_pages * PAGE
    idx = np.flatnonzero(new)
    return lo[idx].astype(np.int64), np.maximum.reduceat(hi, idx).astype(np.int64)


_RA_CAPS: Dict[int, int] = {}


def _willneed_cap(st_dev: int) -> int:
    """Bytes one posix_fadvise(WILLNEED) call reads at most on this device.

    Linux clamps each WILLNEED request to max(io_pages, ra_pages) pages from
    its start (mm/readahead.c force_page_cache_ra; io_pages = max_sectors_kb,
    ra_pages = read_ahead_kb of the backing queue) and silently drops the rest,
    so longer runs are split into pieces of at most this size. Read once per
    device from sysfs; 128 KiB (the kernel's default read_ahead_kb) when the
    queue cannot be found. Never above _CHUNK."""
    cap = _RA_CAPS.get(st_dev)
    if cap is not None:
        return cap
    cap = 128 << 10
    try:
        base = f"/sys/dev/block/{os.major(st_dev)}:{os.minor(st_dev)}"
        q = os.path.join(base, "queue")
        if not os.path.isdir(q):  # a partition: the queue lives on its disk
            q = os.path.join(base, "..", "queue")
        kb = []
        for name in ("max_sectors_kb", "read_ahead_kb"):
            with open(os.path.join(q, name)) as fh:
                kb.append(int(fh.read().strip()))
        if max(kb) > 0:
            cap = max(kb) << 10
    except (OSError, ValueError):
        pass
    cap = max(PAGE, min(_CHUNK, cap))
    _RA_CAPS[st_dev] = cap
    return cap


def _chunks(starts: np.ndarray, ends: np.ndarray, chunk: int = _CHUNK):
    for s, e in zip(starts.tolist(), ends.tolist()):
        while s < e:
            n = min(chunk, e - s)
            yield s, n
            s += n


def _pmap(pool, fn, items) -> List[Any]:
    """``[fn(x) for x in items]`` on the pool, one task per contiguous slice
    rather than per item: per-item executor dispatch (a future, a queue hop and
    GIL hand-offs per call) costs more Python time than the tiny fadvise/pread
    calls themselves, in the training process. Results keep the item order."""
    items = list(items)
    if not items:
        return []
    n = max(1, min(len(items), int(getattr(pool, "_max_workers", 16) or 16)))
    step = -(-len(items) // n)
    out: List[Any] = []
    for part in pool.map(lambda sl: [fn(x) for x in sl],
                         [items[k:k + step] for k in range(0, len(items), step)]):
        out.extend(part)
    return out


def _willneed_loop(f: _NpyFile, starts, ends) -> None:
    """WILLNEED on [starts, ends) from the calling thread. The call only queues
    readahead, so one thread keeps up, and a single thread contends for the GIL with the training thread
    far less than a pool of busy workers would."""
    fd = f.fd
    for s, n in _chunks(starts, ends, getattr(f, "ra_cap", _CHUNK)):
        _fadvise(fd, s, n, _WILLNEED)


def _issue(pool, f: _NpyFile, starts, ends, how: int, serial: bool = False) -> int:
    """Hand [starts, ends) to the block layer in parallel (``serial``: WILLNEED
    from the calling thread instead). Returns bytes."""
    if how == 1 and _fadvise is not None:
        if serial:
            _willneed_loop(f, starts, ends)
            return int((ends - starts).sum())
        fn = lambda a: _fadvise(f.fd, a[0], a[1], _WILLNEED)  # noqa: E731
        chunk = getattr(f, "ra_cap", _CHUNK)  # the kernel drops WILLNEED bytes past it
    else:
        fn = lambda a: len(_pread(f.fd, a[1], a[0]))  # noqa: E731
        chunk = _CHUNK
    _pmap(pool, fn, _chunks(starts, ends, chunk))
    return int((ends - starts).sum())


def _gather(pool, f: _NpyFile, idx: np.ndarray, stats: Dict[str, int],
            serial: bool = False) -> np.ndarray:
    """Values f[idx] read after a WILLNEED pass over their pages, so a dependent
    index level costs ~one I/O latency instead of len(idx) faults. ``serial``
    (the streaming walk): WILLNEED and the preads both from the calling thread.
    The WILLNEED pass already has every read in flight, and pooled preads add
    process CPU and GIL hand-offs against the training thread. Otherwise both
    go through the pool. Out-of-range
    idx yield -1 (the caller drops them)."""
    idx = np.asarray(idx, dtype=np.int64)
    ok = (idx >= 0) & (idx < f.n)
    u = np.unique(idx[ok])
    out = np.full(idx.shape, -1, dtype=np.int64)
    if u.size == 0:
        return out
    off = f.hdr + u * f.isz
    starts, ends = _page_runs(off, off + f.isz, f.size, 0)
    if _fadvise is not None:
        if serial:
            _willneed_loop(f, starts, ends)
        else:
            _pmap(pool, lambda a: _fadvise(f.fd, a[0], a[1], _WILLNEED),
                  _chunks(starts, ends, getattr(f, "ra_cap", _CHUNK)))
    if serial:
        bufs = [_pread(f.fd, e - s, s) for s, e in zip(starts.tolist(), ends.tolist())]
    else:
        bufs = _pmap(pool, lambda se: _pread(f.fd, se[1] - se[0], se[0]),
                     zip(starts.tolist(), ends.tolist()))
    stats["index_runs"] += int(starts.size)
    stats["index_bytes"] += int((ends - starts).sum())
    base = np.zeros(len(bufs), dtype=np.int64)
    base[1:] = np.cumsum([len(b) for b in bufs])[:-1]
    flat = np.frombuffer(b"".join(bufs), dtype=np.uint8)
    run = np.searchsorted(starts, off, side="right") - 1
    pos = base[run] + (off - starts[run])
    if np.any(pos + f.isz > flat.size):  # short read at EOF
        good = pos + f.isz <= flat.size
        u, pos = u[good], pos[good]
    if u.size == 0:
        return out
    vals = flat[pos[:, None] + np.arange(f.isz)].copy().view(f.dtype).reshape(-1)
    vals = vals.astype(np.int64)
    sel = np.searchsorted(u, idx).clip(0, u.size - 1)
    hit = ok & (u[sel] == idx)  # elements lost to a short read stay -1
    out[hit] = vals[sel[hit]]
    return out


def _window_ids(ds: Any, ts: int) -> np.ndarray:
    """``ds.train_window_indices(ts)`` without its per-window log line when the
    dataset supports ``quiet`` (the loader computes and logs the same window)."""
    fn = ds.train_window_indices
    try:
        import inspect

        quiet = "quiet" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        quiet = False
    return np.asarray(fn(ts, quiet=True) if quiet else fn(ts), dtype=np.int64)


def rank_window_batches(
    ds: Any, ts: int, rank: int, world_size: int, batch_size: int, skip: int = 0
) -> Tuple[np.ndarray, int]:
    """(per-rank anchor indices, batch count) of window ``ts``, exactly as
    StreamingWindowSampler.set_window + drop_last see them."""
    g = _window_ids(ds, ts)
    n = (len(g) // world_size) * world_size
    per_rank = g[:n][rank::world_size]
    if skip:
        per_rank = per_rank[skip:]
    return per_rank, len(per_rank) // batch_size


def first_rank_samples(
    ds: Any,
    ts_list: Sequence[int],
    rank: int,
    world_size: int,
    batch_size: int,
    k_batches: int,
    first_skip_samples: int = 0,
    max_windows: int = 64,
) -> Tuple[List[np.ndarray], List[int]]:
    """This rank's first ``k_batches`` batches as global anchor indices, one
    array per batch, mirroring StreamingWindowSampler.set_window + drop_last."""
    batches: List[np.ndarray] = []
    used_ts: List[int] = []
    for i, ts in enumerate(list(ts_list)[:max_windows]):
        g = _window_ids(ds, ts)
        n = (len(g) // world_size) * world_size
        per_rank = g[:n][rank::world_size]
        if i == 0 and first_skip_samples:
            per_rank = per_rank[first_skip_samples:]
        nb = len(per_rank) // batch_size
        for b in range(nb):
            batches.append(per_rank[b * batch_size:(b + 1) * batch_size])
            if not used_ts or used_ts[-1] != ts:
                used_ts.append(int(ts))
            if len(batches) >= k_batches:
                return batches, used_ts
    return batches, used_ts


def _files(ds: Any) -> Dict[str, _NpyFile]:
    """Open the store columns + positions (missing/mismatched files skipped)."""
    out: Dict[str, _NpyFile] = {}
    cache_dir = getattr(ds, "_cache_dir", None)
    store = getattr(ds, "store", None)
    if not cache_dir or store is None:
        return out
    names = [("positions", getattr(ds, "_positions_name", None), getattr(ds, "_positions", None))]
    for col in ("flat_uid",) + _SCAN_COLS + _SCALAR_COLS:
        names.append((col, f"{col}.npy", getattr(store, col, None)))
    names.append(("user_start", "user_start.npy", getattr(store, "user_start", None)))
    for role, fname, arr in names:
        if not fname or arr is None:
            continue
        try:
            out[role] = _NpyFile(os.path.join(cache_dir, fname), len(arr))
        except Exception:  # missing / unreadable / mismatched: skip this column
            pass
    return out


def _ranges(pool, files, samples: np.ndarray, scan_window: int, how: int,
            stats: Dict[str, int], gap_pages: int = _GAP_PAGES,
            serial: bool = False) -> None:
    pos_f, uid_f, us_f = files.get("positions"), files.get("flat_uid"), files.get("user_start")
    if pos_f is None:
        return
    fp = _gather(pool, pos_f, samples, stats, serial)
    fp = fp[fp >= 0]
    if fp.size == 0:
        return
    if uid_f is not None and us_f is not None:
        uid = _gather(pool, uid_f, fp, stats, serial)
        us = _gather(pool, us_f, uid, stats, serial)
        us = np.where(us >= 0, us, fp)  # unknown user start: scalar only
    else:
        us = fp
    ss = np.maximum(us, fp - int(scan_window))
    for col in _SCAN_COLS + _SCALAR_COLS:
        f = files.get(col)
        if f is None:
            continue
        lo = ss if col in _SCAN_COLS else fp
        lo_b = f.hdr + lo * f.isz
        hi_b = f.hdr + (fp + 1) * f.isz
        starts, ends = _page_runs(lo_b, hi_b, f.size, gap_pages)
        stats["ranges"] += int(starts.size)
        stats["bytes"] += _issue(pool, f, starts, ends, how, serial)


def _ahead(cum_at: Dict[int, int], i: int, b: int) -> Optional[int]:
    """Batches between the consumer and batch ``b`` of walked window ``i``
    (None: window ``i`` is already pulled completely). The consumer is in
    window ``_DONE[0]`` (every earlier window pulled), ``_PULLED - _DONE[1]``
    batches in (always 0 without warm-keep pull counting)."""
    d = _DONE[0]
    if i < d:
        return None
    p = max(0, _PULLED[0] - _DONE[1])
    if i == d:
        return b - p
    # d < i: the walk visits windows in order and only skips windows already
    # pulled, so window d was walked and has a start index.
    return cum_at[i] + b - (cum_at.get(d, cum_at[i]) + p)


def _stream(pool, files, ds, ts_list, rank, world_size, batch_size,
            first_skip_samples, window_cap, how, scan_window, start_batch,
            horizon, lead, group, st: Dict[str, Any]) -> None:
    """Paced read-ahead of the stream's batches [start_batch, horizon), in the
    order the training pipeline pulls them (see the module docstring)."""
    cum_at: Dict[int, int] = {}
    walked = 0  # batches walked (issued or found already pulled), the horizon budget
    skip_left = int(start_batch)  # leading batches the first wave already issued
    pulled0 = _PULLED[0]
    # A mid-window resume re-enters window 0 at batch K: the stream pulls only
    # cap - K of its batches under a per-window cap (_WarmWindowStream._open_next).
    k0 = int(first_skip_samples) // max(1, int(batch_size))
    for i, ts in enumerate(ts_list):
        # Horizon: batches walked, or batches the consumer pulled (late windows
        # are skipped without being walked).
        if _STOP.is_set() or walked >= horizon or _PULLED[0] - pulled0 >= horizon:
            break
        if i < _DONE[0]:  # pulled completely before we got here
            st["late_windows"] += 1
            continue
        per_rank, nb = rank_window_batches(
            ds, int(ts), rank, world_size, batch_size,
            first_skip_samples if i == 0 else 0,
        )
        if window_cap and int(window_cap) > 0:
            nb = min(nb, max(0, int(window_cap) - (k0 if i == 0 else 0)))
        cum_at[i] = walked
        st["windows"] += 1
        st["last_ts"] = int(ts)
        b = min(nb, skip_left)  # batches the first wave already issued
        skip_left -= b
        walked += b
        while b < nb and walked < horizon and not _STOP.is_set():
            k = min(group, nb - b, horizon - walked)
            ahead = _ahead(cum_at, i, b)
            if ahead is None:  # the consumer finished this window meanwhile
                st["late_batches"] += nb - b
                walked += nb - b
                break
            if ahead + k <= 0:  # whole group already pulled
                st["late_batches"] += k
            else:
                if ahead > lead:  # far enough ahead: wait for the consumer
                    w0 = time.time()
                    while not _STOP.is_set():
                        ahead = _ahead(cum_at, i, b)
                        if ahead is None or ahead <= lead:
                            break
                        _STOP.wait(_WAIT_S)
                    st["wait_s"] += time.time() - w0
                    if _STOP.is_set():
                        break
                    if ahead is None:
                        continue  # re-evaluated at the loop top as consumed
                t1 = time.time()
                _ranges(pool, files, per_rank[b * batch_size:(b + k) * batch_size],
                        scan_window, how, st, gap_pages=_STREAM_GAP_PAGES,
                        serial=True)
                st["stream_issue_s"] += time.time() - t1
                st["stream_batches"] += k
            b += k
            walked += k


def _run(ds, ts_list, rank, world_size, batch_size, first_skip_samples, how,
         k_batches, threads, stream=False, window_cap=None, lead=64, horizon=1024,
         group=4) -> None:
    t0 = time.time()
    c0 = time.thread_time()
    rel = _RELEASE_WALL[0] if _RELEASE_WALL else t0
    files: Dict[str, _NpyFile] = {}

    def _rs() -> Optional[float]:
        try:
            from generative_recommenders.dlrm_v4.train import dl_prestart as _dls

            return _dls.run_start_wall()
        except Exception:
            return None

    try:
        if not _RELEASED.is_set():  # belt and braces: never before release
            logger.warning("[cold-prefetch] not released; skipped (rank=%d)", rank)
            return
        # Streaming mode: the first batch alone, then the paced walk from batch 2
        # on (one numbering); one-shot mode: the first k_batches in two waves.
        batches, used_ts = first_rank_samples(
            ds, ts_list, rank, world_size, batch_size, 1 if stream else k_batches,
            first_skip_samples,
        )
        files = _files(ds)
        stats: Dict[str, Any] = {"ranges": 0, "bytes": 0, "index_runs": 0, "index_bytes": 0}
        b1 = dict(stats)
        scan_window = int(getattr(ds, "_scan_window", 0))
        if not (batches and files):
            stream = False
        # (truthiness: gin binds the flag through env_int, so "off" is the int 0)
        if stream and not getattr(ds, "_use_offline_ts_indices", True):
            # Without the offline window index every window is an O(N) mask
            # over the whole anchor array: the walk would repeat the loader's
            # scans and still fall behind tiny windows. One-shot only.
            stream = False
            if rank == 0:
                logger.warning("[cold-prefetch] stream off: no offline window index "
                               "(DL_WINDOW_INDEX=0); one-shot prefetch only")
        with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="cold-prefetch") as pool:
            if batches and files:
                # Wave 1: the first batch alone (the critical one); wave 2: the rest.
                _ranges(pool, files, batches[0], scan_window, how, stats)
                b1 = dict(stats)
                if len(batches) > 1:
                    _ranges(pool, files, np.concatenate(batches[1:]), scan_window, how, stats)
            if rank == 0:
                rs = _rs()
                logger.warning(
                    "[cold-prefetch] armed mode=%s rank=%d batches=%d windows=%s files=%d "
                    "first_ranges=%d bytes=%d batch1_ranges=%d batch1_bytes=%d "
                    "index_runs=%d index_bytes=%d t_after_release=%+.3f issue_s=%.3f%s%s",
                    "fadvise" if how == 1 else "pread", rank, len(batches),
                    (f"{used_ts[0]}..{used_ts[-1]}" if used_ts else "-"), len(files),
                    stats["ranges"], stats["bytes"], b1["ranges"], b1["bytes"],
                    stats["index_runs"], stats["index_bytes"], t0 - rel, time.time() - t0,
                    f" t_rel_run_start={time.time() - rs:+.3f}" if rs else "",
                    (f" stream=on lead={lead}"
                     f" horizon={'all' if horizon >= HORIZON_ALL else horizon} group={group}"
                     f" cap={window_cap or 0}") if stream else "",
                )
            if stream:
                st: Dict[str, Any] = {
                    "ranges": 0, "bytes": 0, "index_runs": 0, "index_bytes": 0,
                    "windows": 0, "late_windows": 0, "late_batches": 0,
                    "stream_batches": 0, "stream_issue_s": 0.0, "wait_s": 0.0,
                    "last_ts": -1,
                }
                ts0 = time.time()
                _stream(pool, files, ds, ts_list, rank, world_size, batch_size,
                        first_skip_samples, window_cap, how, scan_window, len(batches),
                        horizon, lead, group, st)
                if rank == 0:
                    rs = _rs()
                    logger.warning(
                        "[cold-prefetch] stream done rank=%d batches=%d windows=%d "
                        "last_ts=%d late_windows=%d late_batches=%d ranges=%d bytes=%d "
                        "index_runs=%d index_bytes=%d issue_s=%.3f wait_s=%.1f "
                        "walk_s=%.1f cpu_s=%.2f pulled=%d stopped=%d%s",
                        rank, st["stream_batches"], st["windows"], st["last_ts"],
                        st["late_windows"], st["late_batches"], st["ranges"],
                        st["bytes"], st["index_runs"], st["index_bytes"],
                        st["stream_issue_s"], st["wait_s"], time.time() - ts0,
                        time.thread_time() - c0, _PULLED[0], int(_STOP.is_set()),
                        f" t_rel_run_start={time.time() - rs:+.3f}" if rs else "",
                    )
    except Exception as e:  # advisory only: never fail or slow the run
        logger.warning("[cold-prefetch] failed (ignored) rank=%d: %r", rank, e)
    finally:
        for f in files.values():
            f.close()


def start(
    ds: Any,
    ts_list: Sequence[int],
    rank: int,
    world_size: int,
    batch_size: int,
    first_skip_samples: int = 0,
    window_cap: Optional[int] = None,
) -> Optional[threading.Thread]:
    """Launch the background prefetch (one-shot, plus the paced streaming
    read-ahead under DL_COLD_PREFETCH_STREAM=1) if armed and released.
    ``window_cap`` is the stream's per-window batch cap (num_train_batches)."""
    how = mode()
    if not how:
        return None
    if not _RELEASED.is_set():
        logger.warning("[cold-prefetch] start() before release ignored (rank=%d)", rank)
        return None
    _STOP.clear()
    _DONE[0], _DONE[1] = 0, _PULLED[0]  # the pacing counts from this stream's start
    # Horizon: the whole stream by default (see the module docstring for why a
    # mid-run stop hurts); $DL_COLD_PREFETCH_HORIZON caps it in per-rank batches.
    horizon = _env_int("DL_COLD_PREFETCH_HORIZON", HORIZON_ALL)
    th = threading.Thread(
        target=_run,
        args=(ds, list(ts_list), rank, world_size, batch_size, first_skip_samples, how,
              max(1, _env_int("DL_COLD_PREFETCH_BATCHES", 8)),
              max(1, _env_int("DL_COLD_PREFETCH_THREADS", 16))),
        kwargs=dict(
            stream=stream_on(),
            window_cap=window_cap,
            lead=max(1, _env_int("DL_COLD_PREFETCH_LEAD", 64)),
            horizon=max(1, horizon),
            group=max(1, _env_int("DL_COLD_PREFETCH_GROUP", 4)),
        ),
        name="cold-prefetch",
        daemon=True,
    )
    th.start()
    _THREADS.append(th)
    return th


def _warm(ds: Any, samples: np.ndarray, rank: int, label: str) -> None:
    t0 = time.time()
    files: Dict[str, _NpyFile] = {}
    try:
        files = _files(ds)
        if not files:
            return
        stats: Dict[str, Any] = {"ranges": 0, "bytes": 0, "index_runs": 0, "index_bytes": 0}
        _ranges(None, files, samples, int(getattr(ds, "_scan_window", 0)), 1, stats,
                serial=True)
        if rank == 0:
            logger.warning(
                "[cold-prefetch] %s warm rank=%d samples=%d ranges=%d bytes=%d "
                "index_runs=%d index_bytes=%d issue_s=%.3f",
                label, rank, int(samples.size), stats["ranges"], stats["bytes"],
                stats["index_runs"], stats["index_bytes"], time.time() - t0,
            )
    except Exception as e:  # advisory only: never fail or slow the run
        logger.warning("[cold-prefetch] %s warm failed (ignored) rank=%d: %r", label, rank, e)
    finally:
        for f in files.values():
            f.close()


def warm_samples(
    ds: Any, samples: Any, rank: int, label: str = "holdout"
) -> Optional[threading.Thread]:
    """WILLNEED over the store pages ``samples`` read, from one background
    thread. Page cache only; no-op before release or without posix_fadvise."""
    if _fadvise is None or not _RELEASED.is_set():
        return None
    th = threading.Thread(
        target=_warm,
        args=(ds, np.asarray(samples, dtype=np.int64), int(rank), label),
        name=f"cold-prefetch-{label}",
        daemon=True,
    )
    th.start()
    return th


def _reset_for_tests() -> None:
    _STOP.set()
    _RELEASED.clear()
    _RELEASE_WALL.clear()
    for th in _THREADS:
        th.join(timeout=10)
    _THREADS.clear()
    _STOP.clear()
    _PULLED[0] = 0
    _DONE[0] = 0
    _DONE[1] = 0
