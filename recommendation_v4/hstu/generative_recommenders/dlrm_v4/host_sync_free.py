# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) Advanced Micro Devices, Inc.
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
"""Drain-free training step: move the per-step host reads off the GPU queue.

A blocking device read after ``progress()`` empties the launch queue, so the
host reads that used to sit on that queue are taken from values the host
already has. The three reads move together:

1. ``Samples.to`` (CPU -> device) computes the batch's length statistics from
   the CPU lengths it is about to copy (max/sum of num_candidates, per-key max
   of the UIH lengths) and tags them onto the DEVICE lengths tensors.
2. ``fx_length_stats`` and GAUC's ``max_length`` read those tags (validated
   against the exact view geometry; anything unrecognised falls back to the
   device read). The integers are identical by construction: the same
   reductions over the same values, before vs after the H2D copy.
3. The MLPerf train-loss value is read one step late from a non-blocking
   pinned D2H copy; the pending event is emitted before ANY other MLLOG event
   (pre-emit hook on MLPerfLogger), so ordering and metadata are unchanged.

4. torchrec's KJT input_dist reads ``length_per_key`` back from
   the device after the values all_to_all. The per-key counts are host-known
   on the sender, so
   they ride along as K extra rows of the splits all_to_all (non-fused and
   fused ``FusedKJTListSplitsAwaitable`` paths; one pinned H2D for the rows,
   and torchrec's debug ``_safe_tolist`` is answered from the host copy).
   Knob ``DLRM_HSTU_FP8_SYNCFREE_LPK`` (default 1).
5. The forward's ``uih_features[key]`` / ``candidates`` KJT access
   computed ``length_per_key`` with a device read. ``Samples.to`` caches it
   from the CPU lengths
   (``KeyedJaggedTensor.to`` carries ``_length_per_key``). Knob
   ``DLRM_HSTU_FP8_SYNCFREE_FWDLPK`` (default 1). Same integers.

6. ``dist_init_on_main`` moves the forward's dist_init wait to the main
   stream (with the record_stream bookkeeping that makes that safe) and
   ``split_wait_poll_on`` issues the deferred splits wait only once the
   collective has finished. ``enabled()`` follows the knob at every node count.

A one-step lag fence (``StepFence``) keeps the host at most one step ahead of
the GPU, which bounds queued memory/allocator state the same way the drain
used to, without emptying the queue.

Same math on every sample: nothing is skipped, cached across steps, or
reordered on the device. Kill switch: ``DLRM_HSTU_FP8_SYNCFREE=0`` (the
``DLRM_HSTU_FP8_`` prefix is what launch_slurm.sh forwards into the
container; the knob itself has nothing to do with FP8). Debug:
``DLRM_HSTU_FP8_SYNCFREE_VERIFY=N`` cross-checks the host values against the
device read for the first N uses of each kind and raises on a mismatch.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

logger: logging.Logger = logging.getLogger(__name__)

_ENV = "DLRM_HSTU_FP8_SYNCFREE"
_VERIFY_ENV = "DLRM_HSTU_FP8_SYNCFREE_VERIFY"
_HOST_CAND = "_dlrm_host_cand_stats"  # (stride, max_c, sum_c) on cand lengths
_HOST_UIH = "_dlrm_host_key_max"  # (stride, [max per key]) on uih lengths

_enabled_logged: Optional[bool] = None
_verify_left = {"len": 0, "gauc": 0, "lpk": 0, "fwdlpk": 0}
_verify_init = False
_announced = set()


def _multi_node() -> bool:
    try:
        if int(os.environ.get("NNODES", "1") or 1) > 1:
            return True
    except ValueError:
        pass
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size() > max(1, torch.cuda.device_count())
    except Exception:
        pass
    return False


_multinode_noop_logged = False


def enabled() -> bool:
    """On when ``DLRM_HSTU_FP8_SYNCFREE=1``, at every node count."""
    # Read per call (cheap): the gin env bootstrap may set it after import.
    global _enabled_logged, _multinode_noop_logged
    on = os.environ.get(_ENV, "0") == "1"
    if not _multinode_noop_logged and os.environ.get(_ENV + "_MULTINODE"):
        _multinode_noop_logged = True
        logger.info(
            "%s_MULTINODE=%s is a no-op: the host-sync-free step no longer "
            "has a node-count gate (only %s decides)",
            _ENV, os.environ.get(_ENV + "_MULTINODE"), _ENV,
        )
    if _enabled_logged is not on:
        _enabled_logged = on
        logger.info(
            "DLRMV4 host-sync-free step: active=%d (%s=%s, multi_node=%d)",
            int(on), _ENV, os.environ.get(_ENV, "0"), int(_multi_node()),
        )
    return on


_PRIO_ENV = _ENV.replace("SYNCFREE", "PIPE_STREAM_PRIO")  # DLRM_HSTU_FP8_PIPE_STREAM_PRIO
_DISTINIT_ENV = _ENV + "_DISTINIT_MAIN"  # DLRM_HSTU_FP8_SYNCFREE_DISTINIT_MAIN


def pipe_stream_priority() -> Optional[int]:
    """Explicit priority for torchrec's pipeline memcpy / data-dist streams.

    None keeps torchrec's own stream priority. ``DLRM_HSTU_FP8_PIPE_STREAM_PRIO``
    overrides it.
    """
    v = os.environ.get(_PRIO_ENV)
    if v is not None and v != "":
        return int(v)
    return None


def dist_init_on_main() -> bool:
    """Run the forward's input-dist completion on the main stream.

    On when the host-sync-free step is active. ``DLRM_HSTU_FP8_SYNCFREE_DISTINIT_MAIN=0``
    keeps torchrec's data-dist stream.
    """
    return enabled() and os.environ.get(_DISTINIT_ENV, "1") != "0"


_SPLITPOLL_ENV = _ENV + "_SPLITWAIT_POLL"  # DLRM_HSTU_FP8_SYNCFREE_SPLITWAIT_POLL


def split_wait_poll_on() -> bool:
    """Issue the splits all_to_all wait only after the collective has finished.

    On with ``dist_init_on_main()``. ``DLRM_HSTU_FP8_SYNCFREE_SPLITWAIT_POLL=0``
    uses the stock wait.
    """
    return dist_init_on_main() and os.environ.get(_SPLITPOLL_ENV, "1") != "0"


def _poll_until_done(work: Any) -> None:
    # Bounded only by the collective itself (the process-group watchdog still
    # aborts a hung collective); sleep releases the GIL for the loader threads.
    try:
        while not work.is_completed():
            time.sleep(5e-5)
    except Exception:  # noqa: BLE001 -- any query problem: fall back to wait()
        return


def _verify_budget(kind: str) -> bool:
    global _verify_init
    if not _verify_init:
        n = int(os.environ.get(_VERIFY_ENV, "0") or 0)
        _verify_left["len"] = n
        _verify_left["gauc"] = n
        _verify_left["lpk"] = n
        _verify_left["fwdlpk"] = n
        _verify_init = True
    if _verify_left.get(kind, 0) > 0:
        _verify_left[kind] -= 1
        return True
    return False


def _announce(kind: str) -> None:
    if kind not in _announced:
        _announced.add(kind)
        if "FALLBACK" in kind:
            logger.info("DLRMV4 host-sync-free: %s", kind)
        else:
            logger.info("DLRMV4 host-sync-free: %s armed (host value used)", kind)


def _fixed_stride_rows(kjt: Any) -> Optional[Tuple[int, int]]:
    """(num_keys, stride) when the KJT lengths are a dense [keys, stride]."""
    try:
        if kjt.variable_stride_per_key():
            return None
        nk = len(kjt.keys())
        stride = int(kjt.stride())
        lengths = kjt.lengths()
    except Exception:
        return None
    if nk <= 0 or stride <= 0 or lengths.numel() != nk * stride:
        return None
    return nk, stride


def host_stats_from_cpu(
    uih_kjt: Any, cand_kjt: Any
) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, List[int]]]]:
    """Length statistics of a CPU batch, or None if it is not a plain CPU batch."""
    gc = _fixed_stride_rows(cand_kjt)
    gu = _fixed_stride_rows(uih_kjt)
    if gc is None or gu is None:
        return None
    cl = cand_kjt.lengths()
    ul = uih_kjt.lengths()
    if cl.device.type != "cpu" or ul.device.type != "cpu":
        return None
    nkc, sc = gc
    nku, su = gu
    # numpy, not torch: a torch CPU reduction contends with the dataloader threads.
    # Integer max/sum: identical values either way.
    cn = cl.numpy().reshape(nkc, sc)[0]
    max_c = int(cn.max()) if sc > 0 else 0
    sum_c = int(cn.sum(dtype="int64"))
    key_max = [int(v) for v in ul.numpy().reshape(nku, su).max(axis=1)]
    return (sc, max_c, sum_c), (su, key_max)


def _fwd_lpk_on() -> bool:
    # DLRM_HSTU_FP8_SYNCFREE_FWDLPK=0 reads length_per_key from the device.
    return os.environ.get(_ENV + "_FWDLPK", "1") == "1"


def seed_host_length_per_key(*kjts: Any) -> None:
    """Cache length_per_key on CPU KJTs before the H2D copy.

    ``KeyedJaggedTensor.to`` carries ``_length_per_key`` to the device copy, so
    the forward's ``kjt[key]`` (offset_per_key -> length_per_key) needs no
    device-to-host read. Same integers: a per-key sum of the same CPU lengths.
    """
    if not _fwd_lpk_on():
        return
    for kjt in kjts:
        try:
            if getattr(kjt, "_length_per_key", None) is not None:
                continue
            g = _fixed_stride_rows(kjt)
            if g is None or kjt.lengths().device.type != "cpu":
                continue
            nk, stride = g
            a = kjt.lengths().numpy().reshape(nk, stride).sum(axis=1, dtype="int64")
            kjt._length_per_key = [int(v) for v in a]
        except Exception:  # pragma: no cover - never fatal, stock path
            continue


def verify_host_length_per_key(*kjts: Any) -> None:
    """VERIFY mode: compare the carried host length_per_key with the device."""
    if not _fwd_lpk_on():
        return
    for kjt in kjts:
        host = getattr(kjt, "_length_per_key", None)
        g = _fixed_stride_rows(kjt)
        if host is None or g is None or not _verify_budget("fwdlpk"):
            continue
        nk, _ = g
        dev = [int(v) for v in kjt.lengths().view(nk, -1).sum(1).tolist()]
        if dev != list(host):
            raise RuntimeError(
                f"host-sync-free forward length_per_key mismatch: host={host} device={dev}"
            )
        logger.info("DLRMV4 host-sync-free VERIFY fwdlpk OK %s", host)
    _announce("forward length_per_key")


def tag_device_lengths(uih_kjt: Any, cand_kjt: Any, stats: Any) -> None:
    """Attach the CPU-computed statistics to the device lengths tensors."""
    if stats is None:
        return
    cand_stats, uih_stats = stats
    try:
        setattr(cand_kjt.lengths(), _HOST_CAND, cand_stats)
        setattr(uih_kjt.lengths(), _HOST_UIH, uih_stats)
    except Exception:  # pragma: no cover - never fatal, just falls back
        return


def _row0_of_tagged(t: torch.Tensor, attr: str) -> Optional[Any]:
    base = t._base
    if base is None:
        return None
    tag = getattr(base, attr, None)
    if tag is None:
        return None
    stride = tag[0]
    if t.dim() != 1 or t.numel() != stride:
        return None
    if t.storage_offset() != base.storage_offset():
        return None
    return tag


def host_max_candidates(num_candidates: torch.Tensor) -> Optional[int]:
    """Host-known max(num_candidates) for row 0 of a tagged candidates lengths."""
    if not enabled():
        return None
    tag = _row0_of_tagged(num_candidates, _HOST_CAND)
    if tag is None:
        return None
    return tag[1]


def host_length_stats(
    num_candidates: torch.Tensor, uih_seq_lengths: torch.Tensor
) -> Optional[Tuple[int, int, int]]:
    """(max_c, max_u, sum_c) from tags, or None (caller does the device read)."""
    if not enabled():
        return None
    ctag = _row0_of_tagged(num_candidates, _HOST_CAND)
    if ctag is None:
        return None
    ubase = uih_seq_lengths._base
    if ubase is None or uih_seq_lengths.dim() != 1:
        return None
    utag = getattr(ubase, _HOST_UIH, None)
    if utag is None:
        return None
    ustride, key_max = utag
    if uih_seq_lengths.numel() != ustride:
        return None
    off = uih_seq_lengths.storage_offset() - ubase.storage_offset()
    if off < 0 or off % ustride != 0:
        return None
    k = off // ustride
    if k >= len(key_max):
        return None
    out = (ctag[1], key_max[k], ctag[2])
    if _verify_budget("len"):
        dev = torch.stack(
            [
                num_candidates.max().to(torch.int64),
                uih_seq_lengths.max().to(torch.int64),
                num_candidates.sum().to(torch.int64),
            ]
        ).tolist()
        if tuple(dev) != out:
            raise RuntimeError(
                f"host-sync-free length stats mismatch: host={out} device={tuple(dev)}"
            )
        logger.info("DLRMV4 host-sync-free VERIFY len OK %s", out)
    _announce("fx_length_stats")
    return out


def install_gauc_patch() -> None:
    """Route GAUC's max_length through the host tag (exact same value)."""
    import torchrec.metrics.gauc as gauc

    if getattr(gauc, "_dlrm_hostlen_patched", False):
        return
    orig = gauc.get_auc_states

    @torch.compiler.disable
    def get_auc_states(labels, predictions, weights, num_candidates):
        max_length = host_max_candidates(num_candidates)
        if max_length is None:
            return orig(labels, predictions, weights, num_candidates)
        if _verify_budget("gauc"):
            dev = int(num_candidates.max().item())
            if dev != max_length:
                raise RuntimeError(
                    f"host-sync-free GAUC max_length mismatch: host={max_length} device={dev}"
                )
            logger.info("DLRMV4 host-sync-free VERIFY gauc OK %d", max_length)
        _announce("gauc max_length")
        predictions_perm = predictions.permute(1, 0)
        labels_perm = labels.permute(1, 0)
        weights_perm = weights.permute(1, 0)
        predictions_3d = gauc.to_3d(
            predictions_perm, num_candidates, max_length
        ).permute(2, 0, 1)
        labels_3d = gauc.to_3d(labels_perm, num_candidates, max_length).permute(
            2, 0, 1
        )
        weights_3d = gauc.to_3d(weights_perm, num_candidates, max_length).permute(
            2, 0, 1
        )
        return gauc.compute_gauc_3d(predictions_3d, labels_3d, weights_3d)

    gauc.get_auc_states = get_auc_states
    gauc._dlrm_hostlen_patched = True


class PendingScalar:
    """A device scalar copied to pinned host memory without blocking."""

    def __init__(self, value: torch.Tensor) -> None:
        self._dev = value.detach()
        if self._dev.is_cuda:
            self._host = self._dev.to("cpu", non_blocking=True)
            self._event = torch.cuda.Event()
            self._event.record()
        else:
            self._host = self._dev
            self._event = None

    def get(self) -> float:
        if self._event is not None:
            self._event.synchronize()
        return float(self._host)


class StepFence:
    """Keep the host at most ``lag`` steps ahead of the GPU without draining.

    ``step()`` records an event on the current stream and waits for the one
    recorded ``lag`` calls ago, so the queue always still holds the newest
    ``lag`` steps of work.
    """

    def __init__(self, lag: int = 1) -> None:
        self._lag = max(1, int(lag))
        self._events: List[Any] = []

    def step(self) -> bool:
        """Record this step's end; wait for the one ``lag`` steps back.

        Returns True once there was an earlier step to wait on (every call
        after the first ``lag``), False before that.
        """
        if not torch.cuda.is_available():
            return True
        ev = torch.cuda.Event()
        ev.record()
        self._events.append(ev)
        waited = False
        while len(self._events) > self._lag:
            self._events.pop(0).synchronize()
            waited = True
        return waited

    def drain(self) -> None:
        while self._events:
            self._events.pop(0).synchronize()


# ---------------------------------------------------------------------------
# Host-known length_per_key for the KJT input_dist all-to-all.
#
# Every sender already knows its per-key value counts on the host, so K extra
# rows are added to the splits all_to_all that start_sparse_data_dist already issues and
# reads back. Row j carries, for each destination rank r, the value count of
# the j-th key that r receives. The receiver sums row j over sources to get the
# length_per_key of its post-all_to_all KJT and builds it with that list, so
# sync() reads nothing from the device. Host check every call:
# sum(length_per_key) == values.numel() (both host ints). VERIFY mode also
# compares against the device reduction.
# ---------------------------------------------------------------------------

_LPK_ATTR = "_dlrm_host_lpk"
_lpk_pending: List[Any] = []


def _lpk_rows(input_kjt: Any, splits: List[int]) -> Optional[List[List[int]]]:
    """K rows of len(splits): row j[r] = value count of key offset_r + j."""
    if input_kjt.variable_stride_per_key():
        return None
    lpk = input_kjt.length_per_key()  # cached by dist_splits; host list
    k = max(splits) if splits else 0
    rows = [[0] * len(splits) for _ in range(k)]
    off = 0
    for r, n in enumerate(splits):
        for j in range(n):
            rows[j][r] = int(lpk[off + j])
        off += n
    return rows


def _lpk_on() -> bool:
    # DLRM_HSTU_FP8_SYNCFREE_LPK=0 reads length_per_key from the device.
    return os.environ.get(_ENV + "_LPK", "1") == "1"


def install_kjt_lpk_patch() -> None:
    """Patch torchrec's KJT all_to_all to carry host length_per_key (gated)."""
    import torchrec.distributed.dist_data as dd
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    if getattr(dd, "_dlrm_lpk_patched", False):
        return

    split_init = dd.SplitsAllToAllAwaitable.__init__
    split_wait = dd.SplitsAllToAllAwaitable._wait_impl
    kjts_init = dd.KJTAllToAllSplitsAwaitable.__init__
    kjts_wait = dd.KJTAllToAllSplitsAwaitable._wait_impl
    kjtt_wait = dd.KJTAllToAllTensorsAwaitable._wait_impl

    def p_split_init(self, input_tensors, pg, *args, **kwargs):
        extra = _lpk_pending.pop() if _lpk_pending else None
        self._dlrm_lpk_n = 0
        if extra and input_tensors:
            dev = input_tensors[0].device
            dt = input_tensors[0].dtype
            input_tensors = list(input_tensors) + [
                torch.tensor(row, device=dev, dtype=dt) for row in extra
            ]
            self._dlrm_lpk_n = len(extra)
        split_init(self, input_tensors, pg, *args, **kwargs)

    def p_split_wait(self):
        work = getattr(self, "_splits_awaitable", None)
        if split_wait_poll_on() and work is not None and hasattr(work, "is_completed"):
            # split_wait_poll_on(): issue the stream wait + .tolist() copy only
            # once the splits all_to_all has finished, so the high-priority
            # data-dist stream never holds a blocked barrier.
            with torch.autograd.profiler.record_function("## splits poll ##"):
                _poll_until_done(work)
        ret = split_wait(self)
        n = getattr(self, "_dlrm_lpk_n", 0)
        if n:
            self._dlrm_lpk_rows = ret[-n:]
            ret = ret[:-n]
        return ret

    def p_kjts_init(self, pg, input, splits, *args, **kwargs):
        rows = None
        if enabled() and _lpk_on() and pg.size() > 1:
            rows = _lpk_rows(input, list(splits))
        _lpk_pending.clear()
        if rows:
            _lpk_pending.append(rows)
        try:
            kjts_init(self, pg, input, splits, *args, **kwargs)
        finally:
            _lpk_pending.clear()
        self._dlrm_lpk_k = len(rows) if rows else 0

    def p_kjts_wait(self):
        out = kjts_wait(self)
        k = getattr(self, "_dlrm_lpk_k", 0)
        sa = getattr(self, "_splits_awaitable", None)
        src = getattr(sa, "_dlrm_lpk_rows", None)
        if k and src is not None:
            mine = self._splits[self._pg.rank()]
            # row j, column s: count of my j-th key from source s
            setattr(out, _LPK_ATTR, [int(sum(src[j])) for j in range(mine)])
        return out

    def p_kjtt_wait(self):
        if dist_init_on_main():
            # This wait now runs on the main stream, but every buffer of the
            # lengths/values all_to_all lives in the data-dist stream's
            # allocator pool (dist_data.py KJTAllToAllTensorsAwaitable):
            # - outputs: read here by the recat permute;
            # - inputs: c10d releases its stash of them in work.wait(), which
            #   assumes the waiting stream is the one they were allocated on
            #   (on the default path data-dist itself waits, so its next
            #   allocations are ordered after the NCCL send).
            # With the wait on main, the next batch's input dist (issued on
            # data-dist right after the forward, host a step ahead) can reuse
            # either set while NCCL is still sending or before the permute has
            # read it. Mark both in use on the current stream
            # BEFORE work.wait(), so the allocator's reuse event is recorded
            # after main's wait on the collective (and after the permute).
            cur = torch.cuda.current_stream()
            extra = [self._recat] if self._recat is not None else []
            for t in list(self._output_tensors) + list(self._input_tensors or []) + extra:
                if t is not None and t.is_cuda:
                    t.record_stream(cur)
        lpk = getattr(self, _LPK_ATTR, None)
        if (
            lpk is None
            or self._workers == 1
            or self._recat is None
            or self._stride_per_rank is None
            or self._input.variable_stride_per_key()
            or len(lpk) != len(self._keys)
        ):
            if self._workers > 1 and enabled() and _lpk_on():
                _announce(
                    "kjt length_per_key FALLBACK (device read) keys=%d lpk=%s recat=%s"
                    % (len(self._keys), "none" if lpk is None else len(lpk),
                       self._recat is not None)
                )
            return kjtt_wait(self)
        for aw in self._awaitables:
            aw.wait()
        tensors = self._output_tensors
        lengths, values = tensors[0], tensors[1]
        weights = tensors[2] if len(tensors) == 3 else None
        if sum(lpk) != values.numel():
            raise RuntimeError(
                f"host length_per_key {lpk} sums to {sum(lpk)} != values {values.numel()}"
            )
        spr = self._stride_per_rank
        stride = spr[0]
        with torch.autograd.profiler.record_function("## dist_init:recat_values ##"):
            if all(s == stride for s in spr):
                if stride > 0:
                    lengths, values, weights = torch.ops.fbgemm.permute_2D_sparse_data(
                        self._recat, lengths.view(-1, stride), values, weights,
                        values.numel(),
                    )
                    lengths = lengths.view(-1)
            else:
                lengths, values, weights = torch.ops.fbgemm.permute_1D_sparse_data(
                    self._recat, lengths.view(-1), values, weights, values.numel(),
                )
        kjt = KeyedJaggedTensor(
            keys=self._keys,
            values=values,
            weights=weights,
            lengths=lengths,
            stride=sum(spr),
            length_per_key=list(lpk),
        )
        if _verify_budget("lpk"):
            dev = (
                [int(v) for v in lengths.view(len(self._keys), -1).sum(1).tolist()]
                if lengths.numel()
                else [0] * len(self._keys)
            )
            if dev != list(lpk):
                raise RuntimeError(
                    f"host-sync-free length_per_key mismatch: host={lpk} device={dev}"
                )
            logger.info("DLRMV4 host-sync-free VERIFY lpk OK %s", lpk)
        _announce("kjt length_per_key")
        return kjt.sync()

    # ---- Fused splits path. ----------------------------------------------
    # The train pipeline's start_sparse_data_dist does NOT build a
    # KJTAllToAllSplitsAwaitable per KJT: runtime_forwards.KJTAllToAllForward
    # returns a KJTSplitsAllToAllMeta and FusedKJTListSplitsAwaitable issues ONE
    # splits all_to_all for all of them, then constructs the
    # KJTAllToAllTensorsAwaitable directly. The hooks above never see that path.
    # Append K rows to each meta's splits_tensors (K is
    # max(splits), rank-invariant), strip them per meta after the fused wait,
    # and hand the summed list to the tensors awaitable built for that input.
    import torchrec.distributed.embedding_sharding as es

    fused_init = es.FusedKJTListSplitsAwaitable.__init__
    fused_wait = es.FusedKJTListSplitsAwaitable._wait_impl
    kjtt_init = dd.KJTAllToAllTensorsAwaitable.__init__
    Meta = es.KJTSplitsAllToAllMeta
    _POST = "_dlrm_lpk_post"

    class _Done:
        def __init__(self, v: Any) -> None:
            self._v = v

        def wait(self) -> Any:
            return self._v

    def p_fused_init(self, requests, contexts, pg):
        ks = []
        host_vals: Dict[int, List[int]] = {}
        for req in requests:
            for aw in req.awaitables:
                k = 0
                if (
                    isinstance(aw, Meta)
                    and enabled()
                    and _lpk_on()
                    and aw.pg is not None
                    and aw.pg.size() > 1
                    and aw.splits_tensors
                ):
                    rows = _lpk_rows(aw._input, list(aw.splits))
                    if rows:
                        dev = aw.splits_tensors[0].device
                        dt = aw.splits_tensors[0].dtype
                        # ONE pinned, non-blocking H2D for all K rows: a per-row
                        # torch.tensor(..., device=) would issue K pageable
                        # synchronous copies per step.
                        blk = torch.tensor(rows, dtype=dt).pin_memory()
                        blk = blk.to(dev, non_blocking=True)
                        extra = list(blk.unbind(0))
                        for t, r in zip(extra, rows):
                            host_vals[id(t)] = r
                        aw.splits_tensors = list(aw.splits_tensors) + extra
                        k = len(rows)
                ks.append(k)
        # The stock init runs its debug overflow check, _safe_tolist (a D2H plus
        # a stream sync), on EVERY splits tensor. For the appended rows the values
        # are already on the host, so return them without a device read.
        prev = es._safe_tolist
        if host_vals:
            def _tolist(t):
                v = host_vals.get(id(t))
                return list(v) if v is not None else prev(t)

            es._safe_tolist = _tolist
        try:
            fused_init(self, requests, contexts, pg)
        finally:
            es._safe_tolist = prev
        self._dlrm_lpk_ks = ks

    def p_fused_wait(self):
        ks = getattr(self, "_dlrm_lpk_ks", None)
        if not ks or not any(ks) or not self._splits_awaitable:
            return fused_wait(self)
        full = self._splits_awaitable.wait()
        new: List[Any] = []
        lens: List[int] = []
        off = 0
        for aw, n, k in zip(self._awaitables, self._lengths, ks):
            ch = full[off:off + n]
            off += n
            if k:
                rows, ch = ch[-k:], ch[:-k]
                mine = aw.splits[aw.pg.rank()]
                lpk = [int(sum(rows[j])) for j in range(mine)]
                try:
                    setattr(aw._input, _POST, lpk)
                except Exception:  # noqa: BLE001 -- slotted KJT: stock path
                    pass
            new.extend(ch)
            lens.append(len(ch))
        self._splits_awaitable = _Done(new)
        self._lengths = lens
        _announce("kjt length_per_key (fused splits)")
        return fused_wait(self)

    def p_kjtt_init(self, *args, **kwargs):
        kjtt_init(self, *args, **kwargs)
        inp = getattr(self, "_input", None)
        lpk = getattr(inp, _POST, None)
        if lpk is not None:
            try:
                delattr(inp, _POST)
            except Exception:  # noqa: BLE001
                pass
            setattr(self, _LPK_ATTR, lpk)

    es.FusedKJTListSplitsAwaitable.__init__ = p_fused_init
    es.FusedKJTListSplitsAwaitable._wait_impl = p_fused_wait
    dd.KJTAllToAllTensorsAwaitable.__init__ = p_kjtt_init

    dd.SplitsAllToAllAwaitable.__init__ = p_split_init
    dd.SplitsAllToAllAwaitable._wait_impl = p_split_wait
    dd.KJTAllToAllSplitsAwaitable.__init__ = p_kjts_init
    dd.KJTAllToAllSplitsAwaitable._wait_impl = p_kjts_wait
    dd.KJTAllToAllTensorsAwaitable._wait_impl = p_kjtt_wait
    dd._dlrm_lpk_patched = True
