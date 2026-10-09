# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""bf16 EC output: cast the unique rows, then expand.

``construct_jagged_tensors`` receives a unique-row embedding tensor plus
``reverse_indices`` and gathers one row per token. This patch casts those
unique rows to bf16 before the gather when the consumer is bf16.

The forward matches ``gather`` then ``to(bf16)``: gather is a copy, so the
cast commutes with it. The backward upcasts the incoming gradient to fp32 and
reduces in fp32.

Segments smaller than ``min_rows`` stay in the original dtype. The candidate
item MLP runs outside autocast, so a bf16 tensor there would not match its
fp32 weight.
"""

import logging
from typing import Dict, List, Optional, Tuple

import gin
import torch

logger: logging.Logger = logging.getLogger(__name__)

_PATCHED: bool = False
_LOGGED: bool = False
# Set on the first hoisted call to the dtype the EC really emits, so
# DlrmHSTU._user_forward can still restore the transducer output to it.
_HOISTED_FROM: Optional[torch.dtype] = None

# Rows per backward chunk; the backward needs an fp32 copy of the gradient and
# slicing bounds that temporary. 0 disables chunking.
_BWD_CHUNK_ROWS: int = 0
# A key-segment smaller than this keeps its original dtype (see the docstring).
_MIN_ROWS: int = 100_000

# Replace the backward's scatter_add_ with the kernel in ec_segment_reduce.
# Set by enable_segment_reduce(), which make_model calls from the gin param.
_SEGMENT_REDUCE: bool = False
# Dump the reverse_indices segment-length distribution. Default off.
_SEG_HIST: bool = False
_SEG_ARMED_LOGGED: bool = False
# Log the histogram on a new maximum token count, so the last line is the
# largest call rather than whichever call happens first.
_SEG_HIST_MAX_T: int = 0


# ---------------------------------------------------------------------------
# UIH+candidate merge folded into the expand (make_model.ec_merge_uih_candidate).
#
# DlrmHSTU.main_forward merges each (uih, candidate) embedding pair with
# concat_2D_jagged: a full [T_uih + B, 512] bf16 copy per pair in the forward
# (3 pairs, ~2.4M rows each at the yambda-5b shape) plus a split_2D_jagged
# (and a zeros() of the left gradient) per pair in the backward. The rows it
# copies were themselves written one op earlier by this module's gather. A
# concat of gathered rows is a gather with a permuted index, so the merge can
# be done on the 8-byte index instead of the 1 KiB row: build the merged
# per-sample order [uih_b..., cand_b...] of ``reverse_indices`` and let the
# ONE gather write the merged layout directly. Its backward (scatter_add_ /
# segment-reduce in fp32) then consumes the merged gradient with no split.
#
# Each expanded key also gets its OWN _CastExpandFp32Bwd, instead of one
# expand followed by torch.split. That deletes the SplitWithSizesBackward0
# ``aten::cat`` that re-assembled the per-key gradients into one [T, 512]
# buffer for the single backward; each key's reduction reads its own gradient.
#
# Numerics: forward bitwise identical (the concat is a pure copy and cast;
# the candidate rows are rounded fp32->bf16 by the gather's up-front cast
# rather than by the concat's store -- both RNE; a bit-level check confirms
# raw bits). Backward: the same fp32 addends in fp32 accumulators, summed in
# a different order (per key instead of one pass; candidate-row merged
# gradients join the segment reduce instead of the fp32 gather backward).
# A REDUCTION-ORDER change only, on a backward that is already
# atomic/nondeterministic.
# ---------------------------------------------------------------------------
_MERGE_PAIRS: List[Tuple[str, str]] = []
_MERGE_LOGGED: bool = False
# Marker set on a merged expand output. DlrmHSTU.main_forward skips its
# concat_2D_jagged ONLY for a tensor carrying it, so any call that fell back
# (pair not in this call, uih key below min_rows, CW-sharded key, VBE, ...)
# is merged by main_forward exactly as before.
PREMERGED_ATTR: str = "_dlrmv4_uih_cand_premerged"


def enable_ec_merge_uih_candidate(pairs: List[Tuple[str, str]]) -> None:
    """Register the (uih, candidate) embedding pairs to merge in the expand."""
    global _MERGE_PAIRS
    _MERGE_PAIRS = [(str(u), str(c)) for (u, c) in pairs]
    logger.info(
        "DLRMV4 ec_merge_uih_candidate: registered pairs=%s. The 'armed OK' "
        "line prints on the first expand that actually merges.",
        _MERGE_PAIRS,
    )


# DLRM_HSTU_FP8_EC_CAT_FUSE (default on; =0 disables): when every key of the
# registered group is merged in one call with equal row counts, one expand
# writes the concatenated layout. Its [k*N, W] output viewed as [N, k*W] is
# the cat. Each key is a column view, tagged with CATFUSED_ATTR so the
# preprocessor uses that base instead of torch.cat. The backward is one
# segment reduce over the contiguous gradient.
_CAT_GROUP: List[str] = []
CATFUSED_ATTR: str = "_dlrmv4_catfused"
_CAT_LOGGED: bool = False


def enable_ec_cat_fuse(names: List[str]) -> None:
    global _CAT_GROUP
    _CAT_GROUP = [str(n) for n in names]
    logger.info("DLRMV4 ec_cat_fuse: registered group=%s", _CAT_GROUP)


def is_premerged(t: torch.Tensor) -> bool:
    return bool(getattr(t, PREMERGED_ATTR, False))


def _merged_index(
    ri_u: torch.Tensor,
    ri_c: torch.Tensor,
    len_u: torch.Tensor,
    len_c: torch.Tensor,
    n_u: int,
    n_c: int,
) -> torch.Tensor:
    """reverse_indices in merged per-sample order [uih_b..., cand_b...].

    uih token at flat index i of sample b lands at i + off_c[b];
    candidate token at flat index k of sample b lands at k + off_u[b + 1].
    No host sync: the totals are the host-side length_per_key entries.
    """
    dev = ri_u.device
    len_u = len_u.to(torch.int64)
    len_c = len_c.to(torch.int64)
    off_u = torch.ops.fbgemm.asynchronous_complete_cumsum(len_u)
    off_c = torch.ops.fbgemm.asynchronous_complete_cumsum(len_c)
    samples = torch.arange(len_u.numel(), device=dev, dtype=torch.int64)
    seg_u = torch.repeat_interleave(samples, len_u, output_size=n_u)
    seg_c = torch.repeat_interleave(samples, len_c, output_size=n_c)
    dst_u = torch.arange(n_u, device=dev, dtype=torch.int64) + off_c[seg_u]
    dst_c = torch.arange(n_c, device=dev, dtype=torch.int64) + off_u[seg_c + 1]
    out = torch.empty(n_u + n_c, dtype=ri_u.dtype, device=dev)
    out.scatter_(0, dst_u, ri_u)
    out.scatter_(0, dst_c, ri_c)
    return out


def _merged_construct(
    embeddings: torch.Tensor,
    feats,
    embedding_names: List[str],
    reverse_indices: torch.Tensor,
    out_dtype: torch.dtype,
    length_per_key: List[int],
    offs: List[int],
    kept: List[str],
):
    """construct_jagged_tensors with per-key expands and pre-merged pairs.

    Returns None (caller falls back to the per-call-split path) unless every
    key is 1:1 and at least one registered pair is mergeable in this call.
    """
    from torchrec.sparse.jagged_tensor import JaggedTensor

    if len(set(embedding_names)) != len(embedding_names):
        return None
    pos = {name: i for i, name in enumerate(embedding_names)}
    kept_set = set(kept)
    # A pair merges only when BOTH keys are in this call and the uih key is
    # hoisted: then today's concat output is bf16 (concat takes values_left's
    # dtype), which is exactly what the bf16 gather produces.
    merge = {
        u: c
        for (u, c) in _MERGE_PAIRS
        if u in pos and c in pos and u not in kept_set
    }
    if not merge:
        return None

    w = embeddings.size(-1)
    lengths = feats.lengths().view(-1, feats.stride())
    lengths_tuple = torch.unbind(lengths, dim=0)

    def seg(i: int) -> torch.Tensor:
        return reverse_indices[offs[i]:offs[i] + length_per_key[i]]

    fused_vals: Dict[str, torch.Tensor] = {}
    grp = _CAT_GROUP
    if len(grp) > 1 and all(g in merge for g in grp):
        sizes = [
            length_per_key[pos[g]] + length_per_key[pos[merge[g]]] for g in grp
        ]
        if len(set(sizes)) == 1:
            idxs = [
                _merged_index(
                    seg(pos[g]), seg(pos[merge[g]]),
                    lengths_tuple[pos[g]], lengths_tuple[pos[merge[g]]],
                    length_per_key[pos[g]], length_per_key[pos[merge[g]]],
                )
                for g in grp
            ]
            comb = torch.stack(idxs, dim=1).reshape(-1)
            k = len(grp)
            big = _CastExpandFp32Bwd.apply(embeddings, comb, out_dtype).view(
                sizes[0], k * w
            )
            for q, g in enumerate(grp):
                v = big[:, q * w:(q + 1) * w]
                setattr(v, CATFUSED_ATTR, (big, q, k))
                fused_vals[g] = v
            global _CAT_LOGGED
            if not _CAT_LOGGED:
                _CAT_LOGGED = True
                logger.info(
                    "DLRMV4 armed OK: ec_cat_fuse ACTIVE -- one interleaved "
                    "expand for %s -> [%d, %d]", grp, sizes[0], k * w,
                )

    ret: Dict[str, "JaggedTensor"] = {}
    for i, name in enumerate(embedding_names):
        n = length_per_key[i]
        if name in fused_vals:
            j = pos[merge[name]]
            values = fused_vals[name]
            setattr(values, PREMERGED_ATTR, True)
            jt_lengths = lengths_tuple[i] + lengths_tuple[j]
        elif name in merge:
            j = pos[merge[name]]
            idx = _merged_index(
                seg(i), seg(j), lengths_tuple[i], lengths_tuple[j],
                n, length_per_key[j],
            )
            values = _CastExpandFp32Bwd.apply(embeddings, idx, out_dtype)
            setattr(values, PREMERGED_ATTR, True)
            jt_lengths = lengths_tuple[i] + lengths_tuple[j]
        elif name in kept_set:
            # Unchanged: small segments stay in the EC's own dtype.
            values = torch.gather(embeddings, 0, seg(i).unsqueeze(1).expand(-1, w))
            jt_lengths = lengths_tuple[i]
        else:
            values = _CastExpandFp32Bwd.apply(embeddings, seg(i), out_dtype)
            jt_lengths = lengths_tuple[i]
        ret[name] = JaggedTensor(lengths=jt_lengths, values=values, weights=None)

    global _MERGE_LOGGED
    if not _MERGE_LOGGED:
        _MERGE_LOGGED = True
        logger.info(
            "DLRMV4 armed OK: ec_merge_uih_candidate ACTIVE -- expand writes "
            "merged uih+candidate layout for %s; per-key expands (no split "
            "backward cat); main_forward skips concat_2D_jagged for these",
            sorted(merge.items()),
        )
    return ret


def hoisted_from_dtype() -> Optional[torch.dtype]:
    """The dtype the EC output had before the hoist, or None if inactive."""
    return _HOISTED_FROM


@gin.configurable
def configure_ec_bf16_output(
    bwd_chunk_rows: int = 0,
    min_rows: int = 100_000,
) -> None:
    """Tuning for the bf16 hoist itself. Called by `enable_bf16_ec_output()`.

    Both were `os.environ` reads at import once; they are gin now so
    a run's operative config records what it was tuned with.

    Args:
      bwd_chunk_rows: rows per backward chunk.  The backward needs an fp32 copy
        of the incoming gradient, and chunking bounds that temporary at the cost
        of several passes.  0 disables chunking, which is the shipped setting --
        the fp32 copy fits.  MUTUALLY EXCLUSIVE with `ec_segment_reduce`, which
        reads the gradient in place and never materialises the copy chunking
        exists to bound; arming both raises rather than silently picking one.
      min_rows: a key-segment shorter than this keeps its original dtype.  The
        hoist pays off in proportion to the expansion it skips, and the small
        contextual/candidate features (~1024 rows) expand too little to matter
        while `_item_embedding_mlp` runs outside autocast and would raise on a
        bf16 input.  100k keeps those in fp32 and hoists only the big tables.
    """
    global _BWD_CHUNK_ROWS, _MIN_ROWS
    _BWD_CHUNK_ROWS, _MIN_ROWS = bwd_chunk_rows, min_rows


@gin.configurable
def enable_segment_reduce(
    seg_hist: bool = False,
    split_above: int = 1024,
    block_s: int = 32,
    num_warps: int = 2,
    block_w_cap: int = 512,
    gpu_planner: bool = False,
    fixed_size_offsets: bool = False,
) -> None:
    """Arm the sort-then-segment-reduce backward for the hoisted EC expand.

    It sorts token rows so each output row is a contiguous segment, then folds
    that segment with `tl.sum`. Cold segments store once. Hot pieces still use
    an atomic add.

    Turn it on with `make_model.ec_segment_reduce = True`. `make_model` checks
    that `ec_bf16_output` is on; this function does not.

    Reduction order differs from `scatter_add_`. Both are nondeterministic.

    Args:
      seg_hist: dump the `reverse_indices` segment-length distribution.
        Default off.
      gpu_planner: keep hot-piece planning on the GPU. Default off.
      fixed_size_offsets: construct offsets on the GPU using the unchanged
        argsort order. Requires valid IDs in [0, num_rows). Default off.
      split_above, block_s, num_warps, block_w_cap: launch geometry, forwarded
        to `ec_segment_reduce`. They are bound here because that module imports
        triton and is loaded lazily, after gin parses.
    """
    if type(gpu_planner) is not bool or type(fixed_size_offsets) is not bool:
        raise ValueError("gpu_planner and fixed_size_offsets must be booleans")
    global _SEGMENT_REDUCE, _SEG_HIST
    _SEGMENT_REDUCE, _SEG_HIST = True, seg_hist
    # Importing here rather than at module scope keeps Triton off the import
    # path of every run that does not arm this -- which is also why the geometry
    # knobs are arguments of THIS function and get forwarded, rather than being
    # gin-bound over there where gin could never see them.
    from generative_recommenders.dlrm_v4.train.ec_segment_reduce import (
        configure_segment_reduce,
    )

    configure_segment_reduce(
        split_above=split_above,
        block_s=block_s,
        num_warps=num_warps,
        block_w_cap=block_w_cap,
        gpu_planner=gpu_planner,
        fixed_size_offsets=fixed_size_offsets,
    )


def _log_seg_histogram(reverse_indices: torch.Tensor, num_rows: int) -> None:
    """Dump the real segment-length distribution once, then never again."""
    cnt = torch.bincount(reverse_indices, minlength=num_rows).to(torch.float64)
    q = torch.tensor([0.5, 0.9, 0.99, 0.999], device=cnt.device, dtype=cnt.dtype)
    p = torch.quantile(cnt, q).tolist()
    logger.info(
        "DLRMV4 ec_segment_reduce histogram: T=%d U=%d dup=%.2fx  len "
        "min=%d p50=%.0f p90=%.0f p99=%.0f p999=%.0f max=%d  "
        "top-1%%-of-rows carries %.1f%% of tokens",
        reverse_indices.numel(),
        num_rows,
        reverse_indices.numel() / max(num_rows, 1),
        int(cnt.min()),
        p[0],
        p[1],
        p[2],
        p[3],
        int(cnt.max()),
        100.0
        * torch.topk(cnt, max(num_rows // 100, 1)).values.sum().item()
        / max(cnt.sum().item(), 1.0),
    )


def _segment_reduce_backward(
    ctx, reverse_indices: torch.Tensor, grad_out: torch.Tensor
) -> tuple:
    """``scatter_add_`` replaced by sort-then-segment-reduce.

    Does not fall back. A request that cannot run raises.
    """
    global _SEG_ARMED_LOGGED, _SEG_HIST_MAX_T
    if _BWD_CHUNK_ROWS > 0:
        raise RuntimeError(
            "EC_SEGMENT_REDUCE=1 is incompatible with "
            f"DLRMV4_EC_BF16_BWD_CHUNK={_BWD_CHUNK_ROWS}: the segment kernel "
            "reads the incoming gradient in place and never materialises the "
            "fp32 copy that chunking exists to bound. Unset one of them."
        )
    from generative_recommenders.dlrm_v4.train.ec_segment_reduce import (
        segment_reduce_rows,
    )

    # Arming stays one-shot -- it answers "is this path live", which the first
    # call settles.  The histogram does not: it answers "what shape does this
    # path actually see", and only the largest call seen so far can answer that.
    if _SEG_HIST and reverse_indices.numel() > _SEG_HIST_MAX_T:
        _SEG_HIST_MAX_T = reverse_indices.numel()
        _log_seg_histogram(reverse_indices, ctx.num_rows)
    if not _SEG_ARMED_LOGGED:
        _SEG_ARMED_LOGGED = True
        logger.info(
            "DLRMV4 armed OK: ec_segment_reduce backward active "
            "(T=%d -> U=%d rows, W=%d, grad dtype=%s, accum fp32 -> %s)",
            reverse_indices.numel(),
            ctx.num_rows,
            ctx.width,
            grad_out.dtype,
            ctx.in_dtype,
        )
    # fp32 accumulator out of the kernel; .to() is a no-op when in_dtype is
    # fp32, which is the whole point of this module's backward.
    grad_emb = segment_reduce_rows(grad_out, reverse_indices, ctx.num_rows)
    return grad_emb.to(ctx.in_dtype), None, None


class _CastExpandFp32Bwd(torch.autograd.Function):
    """bf16 expand forward; fp32 ``scatter_add_`` backward.

    forward:  out[t] = embeddings[reverse_indices[t]].to(out_dtype)
    backward: grad_emb = zeros(U, W, in_dtype).scatter_add_(
                  0, idx, grad_out.to(in_dtype))
    """

    @staticmethod
    # pyre-ignore[14]
    def forward(  # type: ignore[override]
        ctx,
        embeddings: torch.Tensor,
        reverse_indices: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        ctx.save_for_backward(reverse_indices)
        ctx.in_dtype = embeddings.dtype
        ctx.num_rows = embeddings.size(0)
        ctx.width = embeddings.size(-1)
        expanded = reverse_indices.unsqueeze(1).expand(-1, embeddings.size(-1))
        # Cast FIRST -- 69k rows instead of 7.1M -- then expand.
        return torch.gather(embeddings.to(out_dtype), 0, expanded)

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        (reverse_indices,) = ctx.saved_tensors
        w = ctx.width
        if _SEGMENT_REDUCE:
            return _segment_reduce_backward(ctx, reverse_indices, grad_out)
        grad_emb = torch.zeros(
            (ctx.num_rows, w), dtype=ctx.in_dtype, device=grad_out.device
        )
        n = grad_out.size(0)
        chunk = _BWD_CHUNK_ROWS if _BWD_CHUNK_ROWS > 0 else n
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            idx = reverse_indices[lo:hi].unsqueeze(1).expand(-1, w)
            # .to(ctx.in_dtype) is the whole point: the accumulation, and the
            # accumulator, stay fp32.
            grad_emb.scatter_add_(0, idx, grad_out[lo:hi].to(ctx.in_dtype))
        return grad_emb, None, None


def _log_once(
    embeddings: torch.Tensor, out_dtype: torch.dtype, hoisted: List[str],
    kept: List[str],
) -> None:
    global _LOGGED
    if _LOGGED:
        return
    _LOGGED = True
    logger.info(
        "armed OK: ec_bf16_output ACTIVE -- construct_jagged_tensors saw "
        "embeddings dtype=%s shape=%s at the call site; expanding in %s. "
        "hoisted keys=%s ; kept at %s (below %d rows)=%s ; "
        "backward accumulates in %s (bwd_chunk_rows=%d)",
        embeddings.dtype,
        tuple(embeddings.shape),
        out_dtype,
        hoisted,
        embeddings.dtype,
        _MIN_ROWS,
        kept,
        embeddings.dtype,
        _BWD_CHUNK_ROWS,
    )


def enable_bf16_ec_output(out_dtype: torch.dtype = torch.bfloat16) -> bool:
    """Patch torchrec so the dedup expand runs in ``out_dtype``. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return True

    # Pick up `configure_ec_bf16_output`'s gin bindings (or its defaults) before
    # the patched path can run. Calling it is what makes the bindings take
    # effect -- gin applies them at call time, not at import.
    configure_ec_bf16_output()

    import torchrec.distributed.embedding as tde
    import torchrec.modules.utils as tmu
    from torchrec.sparse.jagged_tensor import JaggedTensor

    original = tmu.construct_jagged_tensors

    def _wrapped(
        embeddings: torch.Tensor,
        features,
        embedding_names,
        need_indices: bool = False,
        features_to_permute_indices=None,
        original_features=None,
        reverse_indices: Optional[torch.Tensor] = None,
        seq_vbe_ctx=None,
        use_gather_select: bool = False,
    ):
        # Dedup on, gather-select on, no VBE reindex, and an fp32 input.
        if (
            reverse_indices is None
            or not use_gather_select
            or seq_vbe_ctx is not None
            or embeddings.dtype == out_dtype
            or not embeddings.is_floating_point()
        ):
            return original(
                embeddings=embeddings,
                features=features,
                embedding_names=embedding_names,
                need_indices=need_indices,
                features_to_permute_indices=features_to_permute_indices,
                original_features=original_features,
                reverse_indices=reverse_indices,
                seq_vbe_ctx=seq_vbe_ctx,
                use_gather_select=use_gather_select,
            )

        feats = original_features if original_features is not None else features
        length_per_key: List[int] = feats.length_per_key()

        hoisted, kept = [], []
        offs, acc = [], 0
        for i, name in enumerate(embedding_names):
            offs.append(acc)
            acc += length_per_key[i]
            (hoisted if length_per_key[i] >= _MIN_ROWS else kept).append(name)

        global _HOISTED_FROM
        _HOISTED_FROM = embeddings.dtype
        _log_once(embeddings, out_dtype, hoisted, kept)

        if _MERGE_PAIRS and not need_indices:
            merged = _merged_construct(
                embeddings, feats, embedding_names, reverse_indices, out_dtype,
                length_per_key, offs, kept,
            )
            if merged is not None:
                return merged

        expanded = _CastExpandFp32Bwd.apply(embeddings, reverse_indices, out_dtype)
        ret: Dict[str, JaggedTensor] = original(
            embeddings=expanded,
            features=features,
            embedding_names=embedding_names,
            need_indices=need_indices,
            features_to_permute_indices=features_to_permute_indices,
            original_features=original_features,
            reverse_indices=None,
            seq_vbe_ctx=None,
            use_gather_select=use_gather_select,
        )
        if not kept:
            return ret

        # Re-expand the small segments from the ORIGINAL fp32 tensor so their
        # consumers see exactly the bits they see today. These are 1024-row
        # segments; the cost is noise. Only the 1:1 (non-CW) case is handled --
        # a key appearing at several indices is concatenated inside the
        # original and is left on the bf16 path.
        seen: Dict[str, List[int]] = {}
        for i, name in enumerate(embedding_names):
            seen.setdefault(name, []).append(i)
        w = embeddings.size(-1)
        for name in kept:
            idxs = seen[name]
            if len(idxs) != 1:
                continue
            i = idxs[0]
            lo, n = offs[i], length_per_key[i]
            if n == 0:
                ret[name] = JaggedTensor(
                    lengths=ret[name].lengths(),
                    values=ret[name].values().to(embeddings.dtype),
                    weights=ret[name].weights_or_none(),
                )
                continue
            sub = reverse_indices[lo:lo + n].unsqueeze(1).expand(-1, w)
            ret[name] = JaggedTensor(
                lengths=ret[name].lengths(),
                values=torch.gather(embeddings, 0, sub),
                weights=ret[name].weights_or_none(),
            )
        return ret

    tmu.construct_jagged_tensors = _wrapped
    if getattr(tde, "construct_jagged_tensors", None) is original:
        tde.construct_jagged_tensors = _wrapped
    _PATCHED = True
    logger.info(
        "DLRMV4 ec_bf16_output: patched construct_jagged_tensors in "
        "torchrec.modules.utils and torchrec.distributed.embedding "
        "(out_dtype=%s, min_rows=%d). The 'armed OK' line prints on the first "
        "call that actually hoists.",
        out_dtype,
        _MIN_ROWS,
    )
    return True
