"""bf16 EC output: cast the *unique* rows, not the expanded ones.

TorchRec's index dedup hands
``torchrec.modules.utils.construct_jagged_tensors`` a **unique-row** embedding
tensor plus ``reverse_indices``, and expands it to one row per token *inside*
the call::

    expanded_indices = reverse_indices.unsqueeze(1).expand(-1, embeddings.size(-1))
    embeddings = torch.gather(embeddings, 0, expanded_indices)

At this model's yambda-5b shape that is 69,256 unique rows against 7,108,550
token rows -- **102.6x**. Everything downstream of the expand on the three big
UIH features runs in bf16 anyway (``modules/dlrm_hstu.py:645`` casts
``item_id`` explicitly; autocast casts the ``cat`` of ``artist_id`` and
``album_id`` at ``_additional_embedding_mlp``), so the fp32 rows are produced
only to be rounded one op later.

What the hoist removes, measured on a both-arms trace
(steady steps 53-55, per rank per
step; the join keys ``(rank, External id)`` because External id is a
per-process namespace):

=========================================================  =========  =========
component                                                     today     hoisted
=========================================================  =========  =========
``aten::gather`` -- the expand                              3.838 ms    ~1.98
``embedding.to(bf16)``  [2.37M, 512]   (dlrm_hstu.py:645)   1.207 ms    deleted
autocast cast [2.37M, 1024] at ``_additional_embedding_mlp``2.521 ms    deleted
``aten::cat`` dim=1 of the two additional features          3.189 ms    ~1.77
new: cast of the unique rows [69k, 512]                          --     0.04 ms
``aten::cat`` of 11 keys under ``SplitWithSizesBackward0``  5.797 ms    ~2.9
the two ``ToCopyBackward0`` partners (1.248 + 2.521)        3.769 ms    deleted
new: one bf16->fp32 upcast of the [T,512] gradient               --     ~3.7 ms
``aten::scatter_add_``                                     18.394 ms  unchanged
=========================================================  =========  =========

**The forward is bitwise exact.** ``gather`` is a pure copy, so
``gather(x, i).to(bf16)`` and ``gather(x.to(bf16), i)`` produce identical bit
patterns, and the same commutation holds for the ``cat`` on the
additional-feature path. ``_bf16ec_verify.py`` checks both as raw int16 bit
patterns, and checks the patched ``construct_jagged_tensors`` against the
unpatched one at production shape.

**The backward is deliberately NOT changed.** Letting the expand's backward run
in bf16 would accumulate ~103 duplicates per unique row in bf16 instead of
fp32 -- a real numerics change needing a convergence A/B, not an SNR check.
:class:`_CastExpandFp32Bwd` therefore owns both halves: the forward casts and
gathers in bf16, and the backward **upcasts the incoming bf16 gradient to fp32
and runs ``scatter_add_`` in fp32**. The accumulator dtype, the accumulator
shape and the set of atomic adds are exactly what they are today. That upcast
is relocated, not added: today the same 7.1M x 512 of rows are upcast anyway,
as the ``ToCopyBackward0`` partners of the two deleted forward casts.

**Only large key-segments are hoisted.** ``construct_jagged_tensors`` splits the
expanded tensor by ``length_per_key``, and not every key's consumer is in bf16:
``DlrmHSTU._item_forward`` runs ``_item_embedding_mlp`` on the candidate-side
features **outside any autocast**, so handing it bf16 would raise on the fp32
weight. Segments below ``min_rows`` are therefore re-expanded from the original
fp32 tensor and substituted back, which is bitwise identical to today for those
keys. At this model's shape that is 8 segments of 1024 rows against 3 of
~2.37M -- 0.1% of the rows, and the re-gather is free.
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

# Replace the backward's ``scatter_add_`` with the sort-then-segment-reduce
# kernel in ``ec_segment_reduce``.  Default OFF: this is the arm of a ladder,
# not a default, until the ladder says otherwise.  Standalone at the production
# shape it is 16.763 -> 2.668 ms (6.28x) sort included; see that module.
#
# Set by ``enable_segment_reduce()``, which ``make_model`` calls when the gin
# param ``make_model.ec_segment_reduce`` is true -- NOT by an import-time read
# of ``os.environ``.  An import-time constant cannot be reported in gin's
# operative config, so a run could not say what it had been built with; every
# other arm in this stack is a gin param and this one now matches.
_SEGMENT_REDUCE: bool = False
# One-shot dump of the REAL reverse_indices segment-length distribution.  The
# kernel's standalone numbers were taken on a synthetic uniform index draw, and
# both the sort cost and the load balance depend on the real shape, so this is
# how that assumption gets checked against the training path rather than argued.
_SEG_HIST: bool = False
_SEG_ARMED_LOGGED: bool = False
# The histogram is a RUNNING MAX on T, not a one-shot.  A "log the first call"
# guard samples whichever call happens to come first, and at step 0 of window 0
# that is the SMALLEST call in the run -- the first probe reported T=12288
# U=4076 dup=3.01x, an early short-history window, when the call this
# optimization exists for is ~580x larger.  Logging only on a new maximum means
# the LAST line printed describes the dominant call.
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
# rather than by the concat's store -- both RNE; eval/eval_ec_merge.py checks
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


# DLRM_HSTU_FP8_EC_CAT_FUSE (default on; =0 disables): the ContextualPreprocessor concatenates the
# merged artist_id/album_id expands along dim=1 ([N,512]+[N,512] -> [N,1024],
# 1.68 ms/rank/step at 1n) and the cat's backward hands each expand a
# column-slice gradient that the segment reduce then copies contiguous
# (2 x 0.94 ms). When every key of the registered group is merged in one call
# with equal row counts, ONE expand is issued with the per-key indices
# interleaved row-wise ([idx_a[0], idx_b[0], idx_a[1], ...]); its [k*N, W]
# output viewed as [N, k*W] IS the cat, byte for byte. Each key's values are
# column views of it, tagged with CATFUSED_ATTR so the preprocessor can use
# the base instead of torch.cat. The backward is one segment reduce over the
# contiguous [k*N, W] gradient (same per-row addends; the keys index disjoint
# unique rows, and the stable argsort keeps each segment in ascending token
# order, so the sums are the incumbent's up to the hot-piece atomics).
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

    This is the third rung of the measured ladder (`+4.37%` e2e on top of
    `ec_bf16_output`).  It replaces `scatter_add_` -- which serialises ~7.9M
    token rows into ~54k unique rows through global atomics -- with a sort that
    makes each output row a contiguous segment one program folds with `tl.sum`,
    so there is no atomic on the cold path.

    Arm it with `make_model.ec_segment_reduce = True` (the gin binds
    `$EC_SEGMENT_REDUCE`), never by calling this directly: `make_model` is where
    the `ec_bf16_output` precondition is enforced, and this function does not
    re-check it.

    NOT bit-identical to `scatter_add_`, and cannot be: atomics accumulate in
    nondeterministic order, so there are no bits to match.  Both were scored
    against an fp64 `index_add_` reference instead and the segment kernel's max
    error is 0.61x the incumbent's -- better conditioned than what it replaces.
    It carries a 3-seed convergence gate (delta 0) for that reason; see
    `docs/perf_opt.md`.

    Args:
      seg_hist: dump the REAL `reverse_indices` segment-length distribution.
        Diagnostic, default off.  The kernel's standalone 6.28x was measured on
        a synthetic UNIFORM index draw while production is violently skewed
        (p50 43, p99 1,439, max 64,326), and both the sort cost and the load
        balance depend on that shape -- this is how the assumption gets checked
        against the training path rather than argued.  Costs a `bincount` plus a
        `quantile` on each new maximum-size call, so leave it off for timing.

      gpu_planner: keep hot-piece planning on the GPU. Preserves the host
        planner's piece boundaries and fp32 reduction arithmetic; hot-piece
        atomic accumulation remains nondeterministic. Default off.
      fixed_size_offsets: construct offsets on the GPU using the unchanged
        argsort order. Requires the EC caller's valid IDs in [0, num_rows).
        Independent of gpu_planner; default off.

      split_above, block_s, num_warps, block_w_cap: the kernel's launch
        geometry, forwarded verbatim to `ec_segment_reduce`.  They live on THIS
        function rather than on that module's own `configure_segment_reduce`
        because `ec_segment_reduce` imports triton and so is imported lazily,
        which puts it past the point where gin parses -- a knob bound there
        would raise "No configurable matching".  Full per-argument notes,
        including the sweep each default came from, are in that function's
        docstring; the headline is that the 30-point sweep spans 10x
        (1.488 ms at (32, 2, 512) against 14.50 ms at (8, 8, 256)), so these
        are MEASURED defaults, not free parameters.  `num_warps` is the sharp
        one: 2 wins every cell of the sweep by 1.9-4.8x because 8 warps spill
        the register tile to scratch on gfx950.
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

    Deliberately does NOT fall back.  A knob that silently does nothing and a
    knob that does nothing useful are indistinguishable in a step time, and
    this one is measured as an A/B arm -- an inert arm would be read as a null
    result rather than as a broken one.
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
        # Engage only on the path this was measured on: dedup on (there is an
        # expand to hoist above), gather-select on (so the backward being
        # preserved really is scatter_add_), no VBE reindex (fbgemm's permute
        # runs after the expand and is out of scope), and an fp32 input.
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
