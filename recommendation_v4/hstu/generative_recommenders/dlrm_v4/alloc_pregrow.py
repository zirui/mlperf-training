# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Caching-allocator pre-grow before RUN_START, and a growth report.

Why: with ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` (launch_slurm.sh)
every growth of a stream's expandable segment calls hipMemCreate/hipMemMap/
hipMemSetAccess, and on ROCm hipMemMap waits for the device to drain. Under the
host-sync-free step the host runs about one step ahead, so a growth stall costs
120-270 ms of host time on the rank that grows (seen under
``## start_sparse_data_dist ## > ## dedup_ec_indices ## >
fbgemm::jagged_unique_indices > aten::empty > hipMemMap``). That rank then
reaches its next KJT splits all-to-all late, and every peer's comm kernel spins
over its forward GEMMs while it waits.

The caching allocator never unmaps (no empty_cache in this tree) and caches
blocks per stream. So mapping each train-pipeline stream's segment once, up
front -- one allocate-and-free on that stream -- removes the mid-run hipMemMap
calls without changing any computed value.

When: ``pregrow_before_run_start()`` runs in train_ranker.py during
initialization, before INIT_STOP/RUN_START (like the C++ prebuild), so the
mapping cost is outside the timed region. The pipeline's memcpy / data_dist
streams do not exist yet at that point, so this module creates them (same
priority torchrec uses), grows their pools, and the train pipeline adopts them
in its constructor (``adopt_streams``). The eval pipeline keeps its own streams.

Env (forwarded by launch_slurm.sh through the DLRM_HSTU_FP8_ prefix; the gin
default is bound by apply_env_bootstrap.DLRM_HSTU_FP8_ALLOC_PREGROW):
  DLRM_HSTU_FP8_ALLOC_PREGROW=spec  comma list of <pool>=<GiB>, pool in
                                    {memcpy, data_dist, main} (+ "_small" for the
                                    <=1 MiB pools) and margin=<GiB>. The value
                                    "rest" (one pool at most) takes all free HBM
                                    left after the other pools and the margin,
                                    so one spec fits 1n/2n/4n. "0" or "" = off.
  DLRM_HSTU_FP8_ALLOC_REPORT=1      log every step on which reserved memory
                                    grew, with per-stream totals (host only).
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

_GIB = 1 << 30
# torchrec TrainPipelineSparseDist creates its memcpy / data_dist streams with
# this priority; adopt_streams() checks it against the streams it replaces.
_PIPE_STREAM_PRIORITY = -1

_KEYS = ("memcpy", "data_dist", "main", "memcpy_small", "data_dist_small",
         "main_small", "eval_memcpy", "eval_data_dist", "eval_memcpy_small",
         "eval_data_dist_small", "margin")

_pregrown = False
# Pre-grown (memcpy, data_dist) stream pairs, handed out in pipeline build
# order: the train pipeline first, then the eval pipeline when eval_* keys are set.
_side_streams: List[Dict[str, Any]] = []
_n_pairs = 0
_reporter_made = False


def _spec() -> str:
    spec = os.environ.get("DLRM_HSTU_FP8_ALLOC_PREGROW", "").strip()
    return "" if spec in ("0", "off", "false", "no") else spec


def report_on() -> bool:
    return os.environ.get("DLRM_HSTU_FP8_ALLOC_REPORT", "0") not in ("", "0")


def _rank() -> int:
    try:
        import torch.distributed as dist

        return dist.get_rank() if dist.is_initialized() else -1
    except Exception:
        return -1


def _parse_spec(spec: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, val = part.partition("=")
        name = name.strip()
        if name not in _KEYS:
            raise ValueError(f"DLRM_HSTU_FP8_ALLOC_PREGROW: unknown key {name!r}")
        val = val.strip()
        if val == "rest":
            if name == "margin":
                raise ValueError("DLRM_HSTU_FP8_ALLOC_PREGROW: margin cannot be 'rest'")
            out[name] = float("inf")
        else:
            f = float(val)
            if f != f or f in (float("inf"), float("-inf")) or f < 0:
                raise ValueError(
                    f"DLRM_HSTU_FP8_ALLOC_PREGROW: {name}={val!r} must be finite and >= 0"
                )
            out[name] = f
    if sum(1 for v in out.values() if v == float("inf")) > 1:
        raise ValueError("DLRM_HSTU_FP8_ALLOC_PREGROW: at most one 'rest' pool")
    return out


def _per_stream_reserved(names: Dict[int, str]) -> Dict[str, float]:
    tot: Dict[str, float] = {}
    for seg in torch.cuda.memory_snapshot():
        key = names.get(int(seg.get("stream", 0)), "s%x" % int(seg.get("stream", 0)))
        key += "/" + str(seg.get("segment_type", "?"))
        tot[key] = tot.get(key, 0.0) + seg["total_size"] / _GIB
    return tot


def _pool_free_bytes(stream_id: int) -> int:
    n = 0
    for seg in torch.cuda.memory_snapshot():
        if int(seg.get("stream", 0)) != stream_id or seg.get("segment_type") != "large":
            continue
        n += sum(b["size"] for b in seg.get("blocks", []) if b.get("state") == "inactive")
    return n


def _pool_stream(key: str) -> str:
    return key[: -len("_small")] if key.endswith("_small") else key


def pregrow_before_run_start() -> None:
    """Create the pipeline side streams and grow the named pools (once).

    Called during initialization, before INIT_STOP/RUN_START. Allocates and
    frees device memory only; touches no data.
    """
    global _pregrown, _side_streams, _n_pairs
    spec = _spec()
    if _pregrown or not spec or not torch.cuda.is_available():
        return
    _pregrown = True
    t0 = time.perf_counter()
    want = _parse_spec(spec)
    streams: Dict[str, Any] = {
        "main": torch.cuda.current_stream(),
        "memcpy": torch.cuda.Stream(priority=_PIPE_STREAM_PRIORITY),
        "data_dist": torch.cuda.Stream(priority=_PIPE_STREAM_PRIORITY),
    }
    _side_streams = [{k: streams[k] for k in ("memcpy", "data_dist")}]
    if any(k.startswith("eval_") for k in want):
        for k in ("eval_memcpy", "eval_data_dist"):
            streams[k] = torch.cuda.Stream(priority=_PIPE_STREAM_PRIORITY)
        _side_streams.append(
            {"memcpy": streams["eval_memcpy"], "data_dist": streams["eval_data_dist"]}
        )
    _n_pairs = len(_side_streams)
    names = {int(st.cuda_stream): n for n, st in streams.items()}
    before = _per_stream_reserved(names)
    free, total = torch.cuda.mem_get_info()
    # Never leave less than `margin` GiB unmapped: the eval pipeline's own
    # streams, RCCL lazy peer buffers and non-torch device allocations
    # (hipBLASLt/Triton workspaces) still need room.
    budget = free - want.get("margin", 56.0) * _GIB
    # Fixed-size pools first; a "rest" pool (inf) takes what is left.
    for name, gib in sorted(want.items(), key=lambda kv: kv[1] == float("inf")):
        if name == "margin" or gib <= 0:
            continue
        small = name.endswith("_small")
        st = streams[_pool_stream(name)]
        have = before.get(_pool_stream(name) + ("/small" if small else "/large"), 0.0)
        if gib == float("inf"):
            grow = int(budget)
            if small:
                grow = min(grow, 1 << 30)  # 1 MiB blocks: cap the small pool at 1 GiB
        else:
            grow = min(int((gib - have) * _GIB), int(budget))
        if grow <= 0:
            continue
        with torch.cuda.stream(st):
            if small:
                # Small pool (<=1 MiB requests): 1 MiB blocks, all live at once
                # so each one maps fresh pages, then freed back to the pool.
                ts = [
                    torch.empty(1 << 20, dtype=torch.uint8, device="cuda")
                    for _ in range(grow >> 20)
                ]
                del ts
            else:
                # One block of the full target on that stream, then free it: the
                # expandable segment stays mapped and the block returns to that
                # stream's pool, where later allocations on the stream reuse it.
                # Asking for grow + the pool's cached free bytes makes the segment
                # extend by ~grow (the trailing free block is absorbed first).
                # For "rest" the request itself is capped at the budget: cached
                # free blocks that are not at the segment tail are not absorbed,
                # so grow + free could otherwise map past free - margin.
                req = grow + _pool_free_bytes(int(st.cuda_stream))
                if gib == float("inf"):
                    req = grow
                try:
                    t = torch.empty(req, dtype=torch.uint8, device="cuda")
                    del t
                except torch.cuda.OutOfMemoryError:
                    logger.warning(
                        "DLRMV4 alloc pregrow: rank=%d %s request %.2f GiB OOM; "
                        "pool left as is", _rank(), name, req / _GIB
                    )
                    break
        budget -= grow
    torch.cuda.synchronize()
    after = _per_stream_reserved(names)
    free2, _ = torch.cuda.mem_get_info()
    logger.info(
        "armed OK: alloc pregrow before run_start: rank=%d %.1fs spec=%s "
        "free %.1f->%.1f GiB of %.1f; per-stream reserved GiB before=%s after=%s",
        _rank(),
        time.perf_counter() - t0,
        spec,
        free / _GIB,
        free2 / _GIB,
        total / _GIB,
        {k: round(v, 2) for k, v in before.items()},
        {k: round(v, 2) for k, v in after.items()},
    )


def adopt_streams(pipeline: Any) -> None:
    """Hand the next pre-grown stream pair to a pipeline (train, then eval)."""
    if not _side_streams:
        return
    which = "train" if len(_side_streams) == _n_pairs else "eval"
    s = _side_streams.pop(0)
    mine = getattr(pipeline, "_memcpy_stream", None)
    dd = getattr(pipeline, "_data_dist_stream", None)
    if mine is None or dd is None or {mine.priority, dd.priority} != {_PIPE_STREAM_PRIORITY}:
        # torchrec changed how it builds the streams: keep its streams (the
        # side pools then grow on demand) rather than change their priority.
        logger.warning(
            "DLRMV4 alloc pregrow: pipeline streams not adopted (torchrec "
            "priorities %s/%s, expected %d)",
            getattr(mine, "priority", None),
            getattr(dd, "priority", None),
            _PIPE_STREAM_PRIORITY,
        )
        return
    pipeline._memcpy_stream = s["memcpy"]
    pipeline._data_dist_stream = s["data_dist"]
    logger.info("armed OK: alloc pregrow streams adopted by the %s pipeline", which)


class GrowthReporter:
    """Per-step host check of reserved bytes; logs only on growth."""

    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline
        self._last = torch.cuda.memory_reserved()
        self._step = 0
        self._events = 0

    def after_step(self) -> None:
        self._step += 1
        cur = torch.cuda.memory_reserved()
        if cur == self._last:
            return
        self._events += 1
        p = self._pipeline
        names = {int(torch.cuda.current_stream().cuda_stream): "main"}
        for n in ("memcpy", "data_dist"):
            st = getattr(p, "_%s_stream" % n, None)
            if st is not None:
                names[int(st.cuda_stream)] = n
        logger.info(
            "DLRMV4 alloc growth: rank=%d step=%d events=%d reserved %.3f->%.3f GiB "
            "max_alloc %.3f GiB per-stream=%s",
            _rank(),
            self._step,
            self._events,
            self._last / _GIB,
            cur / _GIB,
            torch.cuda.max_memory_allocated() / _GIB,
            {k: round(v, 3) for k, v in _per_stream_reserved(names).items()},
        )
        self._last = cur


def make_reporter(pipeline: Any) -> Optional[GrowthReporter]:
    """First (train) pipeline only; None unless DLRM_HSTU_FP8_ALLOC_REPORT is set."""
    global _reporter_made
    if _reporter_made or not report_on() or not torch.cuda.is_available():
        return None
    _reporter_made = True
    logger.info("DLRMV4 alloc growth report armed")
    return GrowthReporter(pipeline)
