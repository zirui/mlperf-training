"""Sequence-embedding all-to-all permutes at row granularity, in the wire dtype.

TorchRec's ``All2All_Seq_Req`` reorders the looked-up sequence embeddings
from (source rank, feature) order into (feature, source rank) order before
the forward all-to-all, and undoes that reordering after the backward
all-to-all. It does this with ``fbgemm.permute_2D_sparse_data`` on the fp32
values, with lengths ``[local_T * world_size, B] * D``. Those lengths can have
fewer than one embedding row per segment, so
``permute_2D_data_kernel_vec<false,long,float>`` spends most of its time on
per-segment overhead.

Each row of the 2-D lengths covers one contiguous block of whole embedding
rows (all B samples of one feature from one source rank). This module
therefore:

* builds a row index ``fidx`` with ``permute_2D_sparse_data`` on per-block row
  counts ``[local_T * world_size, 1]`` and ``values = arange(rows)``. That is a
  one segment per block over a small int64 array, instead of B segments per
  block over ``rows * D`` floats;
* forward: ``wire = encode(x)[fidx]``. The codec's encode is elementwise, so
  encoding before the permutation instead of after it gives bit-identical
  results, and the gather runs on the 2-byte wire dtype;
* backward: ``bidx`` comes from ``backward_recat_tensor`` and the permuted
  per-block counts in the same way. Then ``grad = decode(wire[bidx])`` and
  ``div_`` as in stock.

Every output element is copied from exactly one input element. The outputs
are bit-identical to stock, including NaN payloads, -0.0 and the fp16 clamp.
``a2ai.permuted_lengths_after_sparse_data_all2all`` keeps its stock value: the
D-scaled ``[local_T * world_size, B]`` lengths, permuted by rows. Code in the
backward pass and elsewhere reads it.

The stock path still runs, with the original methods, in these cases:
variable batch size, ``local_T == 0``, ``forward_recat_tensor is None``,
``numel % D != 0``, a non-contiguous input, or a codec that is not an
elementwise cast. Elementwise casts are fbgemm ``QuantizedCommCodec`` with
FP32/FP16/BF16 precision (optionally loss-scaled) and torchrec
``NoOpQuantizedCommCodec``. FP8/INT8/MX4 are row-wise and are never re-ordered.

Trace attribution: the index build plus the gather run under the stock
``## All2All_Seq_fwd_permute ##`` / ``## All2All_Seq_bwd_permute ##``
record_function labels. Keep those labels so traces stay attributable.

Env (the ``DLRM_HSTU_FP8_`` prefix is forwarded by the launcher):
  DLRM_HSTU_FP8_SEQ_A2A_ROWPERM=1          enable (absent = off in code; the
                                           yambda_5b.gin apply_env_bootstrap
                                           binding turns it on; =0 opts out).
                                           Read once, at install() time.
  DLRM_HSTU_FP8_SEQ_A2A_ROWPERM_CHECK=N    first N forward calls check
                                           sum(block rows) == rows (one host
                                           sync each, default 1); on failure the
                                           module falls back to stock permanently
  DLRM_HSTU_FP8_SEQ_A2A_ROWPERM_VERIFY=N   first N forward/backward calls also
                                           run the stock math and compare bits
                                           (default 0; costs an extra fp32 copy).
                                           On a mismatch they use the stock
                                           result and disable the module.
"""

import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_ENV = "DLRM_HSTU_FP8_SEQ_A2A_ROWPERM"
_CHECK_ENV = "DLRM_HSTU_FP8_SEQ_A2A_ROWPERM_CHECK"
_VERIFY_ENV = "DLRM_HSTU_FP8_SEQ_A2A_ROWPERM_VERIFY"

_installed = False
_orig: Dict[str, Any] = {}
_state: Dict[str, Any] = {
    "disabled": False,
    "announced": False,
    "check_left": None,
    "verify_fwd_left": None,
    "verify_bwd_left": None,
}

# Grow-only constant 0..n-1 buffers per (device, dtype); slicing == arange(n).
_arange_cache: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}


def enabled() -> bool:
    return os.environ.get(_ENV, "0").strip().lower() in ("1", "true", "on", "yes")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or 0)
    except ValueError:
        return default


def _arange(n: int, device: torch.device, dtype: torch.dtype = torch.int64) -> torch.Tensor:
    key = (device, dtype)
    buf = _arange_cache.get(key)
    if buf is None or buf.numel() < n:
        cap = max(n, 2 * buf.numel() if buf is not None else 0)
        buf = torch.arange(cap, device=device, dtype=dtype)
        _arange_cache[key] = buf
    return buf[:n]


# --------------------------------------------------------------------------
# Pure math (no process group); unit-testable / benchmarkable directly.
# --------------------------------------------------------------------------
def block_rows(tok_lengths: torch.Tensor, n_blocks: int) -> torch.Tensor:
    """Rows per contiguous block: token lengths [n_blocks, B] summed -> [n_blocks, 1]."""
    return tok_lengths.view(n_blocks, -1).sum(dim=1, keepdim=True)


def row_index(
    recat: torch.Tensor, blk: torch.Tensor, rows: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(permuted block rows [n_blocks, 1], row index [rows]) for a block permute."""
    pblk, idx, _ = torch.ops.fbgemm.permute_2D_sparse_data(
        recat, blk, _arange(rows, blk.device), None, rows
    )
    return pblk, idx


def fwd_rowperm(
    recat: torch.Tensor,
    tok_lengths: torch.Tensor,
    lengths_d: torch.Tensor,
    emb: torch.Tensor,
    n_blocks: int,
    D: int,
    encode: Optional[Callable[[torch.Tensor], torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (permuted D-scaled lengths [n_blocks, B], wire 1-D, permuted blk).

    Equals stock ``permute_2D_sparse_data(recat, lengths_d.view(n_blocks,-1),
    emb.view(-1), None, numel)`` followed by ``encode`` (bitwise), for any
    elementwise ``encode``.
    """
    rows = emb.numel() // D
    permuted_lengths_d = torch.index_select(
        lengths_d.view(n_blocks, -1), 0, recat
    )
    pblk, fidx = row_index(recat, block_rows(tok_lengths, n_blocks), rows)
    x = emb.view(-1)
    if encode is not None:
        x = encode(x)
    wire = torch.index_select(x.view(rows, D), 0, fidx).view(-1)
    return permuted_lengths_d, wire, pblk


def bwd_rowperm(
    bwd_recat: torch.Tensor,
    pblk: torch.Tensor,
    wire: torch.Tensor,
    D: int,
    decode: Optional[Callable[[torch.Tensor], torch.Tensor]],
) -> torch.Tensor:
    """Equals stock ``decode`` then ``permute_2D_sparse_data(bwd_recat, ...)`` (1-D)."""
    rows = wire.numel() // D
    _, bidx = row_index(bwd_recat, pblk, rows)
    g = torch.index_select(wire.view(rows, D), 0, bidx).view(-1)
    if decode is not None:
        g = decode(g)
    return g


def stock_fwd(recat, lengths_d, emb, n_blocks, encode):
    """Reference: exactly the stock All2All_Seq_Req forward permute + encode."""
    pl, v, _ = torch.ops.fbgemm.permute_2D_sparse_data(
        recat, lengths_d.view(n_blocks, -1), emb.view(-1), None, emb.numel()
    )
    if encode is not None:
        v = encode(v)
    return pl, v


def stock_bwd(bwd_recat, permuted_lengths_d, wire, decode):
    """Reference: exactly the stock All2All_Seq_Req backward decode + permute."""
    g = decode(wire) if decode is not None else wire
    _, g, _ = torch.ops.fbgemm.permute_2D_sparse_data(
        bwd_recat, permuted_lengths_d, g, None, g.numel()
    )
    return g


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype.is_floating_point:
        it = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
        return bool(torch.equal(a.view(it), b.view(it)))
    return bool(torch.equal(a, b))


# --------------------------------------------------------------------------
# Codec eligibility
# --------------------------------------------------------------------------
def _codec_elementwise(codec: Any) -> bool:
    name = type(codec).__name__
    if name == "NoOpQuantizedCommCodec":
        return True
    if name != "QuantizedCommCodec":
        return False
    try:
        from fbgemm_gpu.split_embedding_configs import SparseType
    except Exception:  # noqa: BLE001
        return False
    prec = getattr(codec, "_comm_precision", None)
    return prec in (SparseType.FP32, SparseType.FP16, SparseType.BF16)


def _codecs_ok(codecs: Any) -> bool:
    if codecs is None:
        return True
    return _codec_elementwise(codecs.forward) and _codec_elementwise(codecs.backward)


# --------------------------------------------------------------------------
# Patched autograd Function methods
# --------------------------------------------------------------------------
def _eligible(a2ai: Any, emb: torch.Tensor) -> bool:
    if _state["disabled"] or not enabled():
        return False
    if a2ai.variable_batch_size or a2ai.forward_recat_tensor is None:
        return False
    if a2ai.backward_recat_tensor is None:
        return False
    if a2ai.lengths_after_sparse_data_all2all.shape[0] == 0:
        return False
    D = a2ai.embedding_dim
    if D <= 0 or emb.numel() % D != 0 or not emb.is_contiguous():
        return False
    return _codecs_ok(a2ai.codecs)


def _forward(ctx, pg, myreq, a2ai, sharded_input_embeddings):  # noqa: C901
    if not _eligible(a2ai, sharded_input_embeddings):
        ctx._seq_rowperm = None
        return _orig["forward"](ctx, pg, myreq, a2ai, sharded_input_embeddings)

    import torch.distributed as dist
    from torch.autograd.profiler import record_function

    world_size = dist.get_world_size(pg)
    my_rank = dist.get_rank(pg)
    D = a2ai.embedding_dim
    tok_lengths = a2ai.lengths_after_sparse_data_all2all
    lengths_d = tok_lengths * D
    input_splits = [i * D for i in a2ai.output_splits]
    output_splits = [i * D for i in a2ai.input_splits]
    local_T = lengths_d.shape[0]
    n_blocks = local_T * world_size
    rows = sharded_input_embeddings.numel() // D

    if _state["check_left"] is None:
        _state["check_left"] = _env_int(_CHECK_ENV, 1)
    if _state["check_left"] > 0:
        _state["check_left"] -= 1
        tot = int(block_rows(tok_lengths, n_blocks).sum().item())
        if tot != rows:
            _state["disabled"] = True
            logger.warning(
                "DLRMV4 seq a2a rowperm: sum(lengths)=%d != rows=%d; "
                "stock permute for the rest of the run",
                tot,
                rows,
            )
            ctx._seq_rowperm = None
            return _orig["forward"](ctx, pg, myreq, a2ai, sharded_input_embeddings)

    a2ai.input_splits = input_splits
    a2ai.output_splits = output_splits

    codecs = a2ai.codecs
    if codecs is not None:
        qcomm_ctx = codecs.forward.create_context()
        encode = lambda t: codecs.forward.encode(t, qcomm_ctx)  # noqa: E731
    else:
        qcomm_ctx = None
        encode = None

    if not _state["announced"]:
        _state["announced"] = True
        logger.info(
            "DLRMV4 armed OK: seq a2a rowperm: row-index permute in wire dtype "
            "(rows=%d, D=%d, blocks=%d, world=%d, codec fwd=%s bwd=%s)",
            rows,
            D,
            n_blocks,
            world_size,
            getattr(getattr(codecs, "forward", None), "_comm_precision", "none"),
            getattr(getattr(codecs, "backward", None), "_comm_precision", "none"),
        )

    forward_recat_tensor = a2ai.forward_recat_tensor
    with record_function("## All2All_Seq_fwd_permute ##"):
        permuted_lengths_d, wire, pblk = fwd_rowperm(
            forward_recat_tensor,
            tok_lengths,
            lengths_d,
            sharded_input_embeddings,
            n_blocks,
            D,
            encode,
        )

    if _state["verify_fwd_left"] is None:
        _state["verify_fwd_left"] = _env_int(_VERIFY_ENV, 0)
    if _state["verify_fwd_left"] > 0:
        _state["verify_fwd_left"] -= 1
        ref_pl, ref_wire = stock_fwd(
            forward_recat_tensor, lengths_d, sharded_input_embeddings, n_blocks, encode
        )
        ok = bits_equal(ref_pl, permuted_lengths_d) and bits_equal(ref_wire, wire)
        logger.info("DLRMV4 seq a2a rowperm verify fwd: bitwise=%s", ok)
        if not ok:
            _state["disabled"] = True
            permuted_lengths_d, wire = ref_pl, ref_wire
            pblk = None
            logger.warning("DLRMV4 seq a2a rowperm: fwd MISMATCH; using stock")

    if codecs is not None:
        output_splits = [
            codecs.forward.calc_quantized_size(x, qcomm_ctx) for x in output_splits
        ]
        input_splits = [
            codecs.forward.calc_quantized_size(x, qcomm_ctx) for x in input_splits
        ]

    sharded_output_embeddings = torch.empty(
        sum(output_splits), dtype=wire.dtype, device=wire.device
    )
    with record_function("## All2All_Seq_fwd ##"):
        req = dist.all_to_all_single(
            output=sharded_output_embeddings,
            input=wire,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=pg,
            async_op=True,
        )
    from torchrec.distributed import comm_ops

    a2ai.permuted_lengths_after_sparse_data_all2all = permuted_lengths_d
    myreq.req = req
    myreq.tensor = sharded_output_embeddings
    myreq.a2ai = a2ai
    myreq.wait_function = comm_ops.All2All_Seq_Req_Wait
    ctx.myreq = myreq
    myreq.qcomm_ctx = qcomm_ctx
    ctx.pg = pg
    ctx.my_rank = my_rank
    ctx.world_size = world_size
    ctx._seq_rowperm = pblk  # None => stock backward
    return myreq.dummy_tensor


def _backward(ctx, *unused):
    pblk = getattr(ctx, "_seq_rowperm", None)
    if pblk is None:
        return _orig["backward"](ctx, *unused)

    from torch.autograd.profiler import record_function
    from torchrec.distributed import comm_ops

    myreq = ctx.myreq
    a2ai = myreq.a2ai
    D = a2ai.embedding_dim
    assert myreq.req is not None
    myreq.req.wait()
    wire = myreq.tensor
    codecs = a2ai.codecs
    qcomm_ctx = myreq.qcomm_ctx
    decode = (
        (lambda t: codecs.backward.decode(t, qcomm_ctx)) if codecs is not None else None
    )
    myreq.req = None
    myreq.tensor = None
    myreq.dummy_tensor = None

    with record_function("## All2All_Seq_bwd_permute ##"):
        g = bwd_rowperm(a2ai.backward_recat_tensor, pblk, wire, D, decode)

    if _state["verify_bwd_left"] is None:
        _state["verify_bwd_left"] = _env_int(_VERIFY_ENV, 0)
    if _state["verify_bwd_left"] > 0:
        _state["verify_bwd_left"] -= 1
        ref = stock_bwd(
            a2ai.backward_recat_tensor,
            a2ai.permuted_lengths_after_sparse_data_all2all,
            wire,
            decode,
        )
        ok = bits_equal(ref, g)
        logger.info("DLRMV4 seq a2a rowperm verify bwd: bitwise=%s", ok)
        if not ok:
            _state["disabled"] = True
            g = ref
            logger.warning("DLRMV4 seq a2a rowperm: bwd MISMATCH; using stock")

    if comm_ops.GRADIENT_DIVISION:
        import torch.distributed as dist

        g.div_(dist.get_world_size(ctx.pg))
    return (None, None, None, g.view(-1, D))


def install() -> None:
    """Patch All2All_Seq_Req.forward/backward (idempotent; originals kept)."""
    global _installed
    if _installed or not enabled():
        return
    try:
        from torchrec.distributed import comm_ops
    except Exception:  # noqa: BLE001
        return
    cls = comm_ops.All2All_Seq_Req
    _orig["forward"] = cls.forward
    _orig["backward"] = cls.backward
    cls.forward = staticmethod(_forward)
    cls.backward = staticmethod(_backward)
    _installed = True
    logger.info("DLRMV4 seq a2a rowperm: All2All_Seq_Req patched")
