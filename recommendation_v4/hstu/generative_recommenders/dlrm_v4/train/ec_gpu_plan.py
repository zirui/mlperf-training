"""GPU planning for EC segmented reduction.

Hot-piece counts and their prefix sums stay on the caller's current stream.
The reducer preserves the host planner's piece boundaries, fp32 tile arithmetic
and launch geometry; hot atomic accumulation order remains nondeterministic.
"""
import torch
import triton
import triton.language as tl

from . import ec_segment_reduce as incumbent


def hot_capacity(tokens, limit):
    # For every hot segment n>L, ceil(n/L) <= 2*n/L. Sum over hot
    # segments, then floor: P <= floor(2*T/L). No device scalar is read.
    return 0 if limit <= 0 else (2 * tokens) // limit


@triton.jit(do_not_specialize=["U"])
def _piece_counts(OFF, COUNTS, U, LIMIT: tl.constexpr,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lo = tl.load(OFF + row, row < U, other=0)
    hi = tl.load(OFF + row + 1, row < U, other=0)
    length = hi - lo
    pieces = tl.where(length > LIMIT, (length + LIMIT - 1) // LIMIT, 0)
    tl.store(COUNTS + row, pieces, row < U)


@triton.jit
def _piece_location(OFF, PREFIX, piece, U, LIMIT: tl.constexpr):
    left = 0
    right = U
    while left < right:
        middle = (left + right) // 2
        endpoint = tl.load(PREFIX + middle)
        advance = endpoint <= piece
        left = tl.where(advance, middle + 1, left)
        right = tl.where(advance, right, middle)
    row = left
    before = tl.load(PREFIX + row - 1, row > 0, other=0)
    lo = tl.load(OFF + row) + (piece - before) * LIMIT
    hi = tl.minimum(lo + LIMIT, tl.load(OFF + row + 1))
    return row, lo, hi


@triton.jit(do_not_specialize=["U"])
def _reduce(grad_ptr, order_ptr, offsets_ptr, prefix_ptr, out_ptr,
            T, W, U, stride_g, LIMIT: tl.constexpr,
            BLOCK_S: tl.constexpr, BLOCK_W: tl.constexpr, HOT: tl.constexpr):
    p = tl.program_id(0)
    cb = tl.program_id(1)
    if HOT:
        valid = p < tl.load(prefix_ptr + U - 1)
    else:
        start0 = tl.load(offsets_ptr + p)
        end0 = tl.load(offsets_ptr + p + 1)
        valid = (end0 - start0 <= LIMIT) | (LIMIT <= 0)
    if valid:
        if HOT:
            row, start, end = _piece_location(offsets_ptr, prefix_ptr, p, U, LIMIT)
        else:
            row, start, end = p, start0, end0
        # Preserve the incumbent reduction operations, tile sizes and piece
        # boundaries. Cold rows store; hot pieces alone use atomic additions.
        cols = cb * BLOCK_W + tl.arange(0, BLOCK_W)
        cmask = cols < W
        acc = tl.zeros([BLOCK_W], dtype=tl.float32)
        for s0 in range(start, end, BLOCK_S):
            srows = s0 + tl.arange(0, BLOCK_S)
            smask = srows < end
            src = tl.load(order_ptr + srows, mask=smask, other=0)
            tile = tl.load(grad_ptr + src[:, None] * stride_g + cols[None, :],
                           mask=smask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(tile, axis=0)
        if HOT:
            tl.atomic_add(out_ptr + row * W + cols, acc, mask=cmask)
        else:
            tl.store(out_ptr + row * W + cols, acc, mask=cmask)


def make_prefix(offsets, num_rows, limit):
    if num_rows == 0 or limit <= 0:
        return None
    counts = torch.empty(num_rows, dtype=torch.int64, device=offsets.device)
    _piece_counts[(triton.cdiv(num_rows, 256),)](offsets, counts, num_rows, limit, 256)
    return torch.cumsum(counts, dim=0)


def segment_reduce_rows(grad_out, reverse_indices, num_rows,
                        out_dtype=torch.float32, order=None, offsets=None):
    # Rows are addressed as grad_out + row * stride(0) + col; any tensor whose
    # columns are unit-stride (or that has <= 1 column / no elements, where the
    # column stride is never used) is read correctly in place.
    assert grad_out.dim() == 2 and (
        grad_out.stride(1) == 1 or grad_out.size(1) <= 1 or grad_out.numel() == 0
    ), grad_out.stride()
    tokens, width = grad_out.shape
    if order is None or offsets is None:
        order, offsets = incumbent.build_segments(reverse_indices, num_rows)
    out = torch.zeros((num_rows, width), dtype=torch.float32, device=grad_out.device)
    limit = incumbent._SPLIT_ABOVE
    prefix = make_prefix(offsets, num_rows, limit)
    block_w = min(triton.next_power_of_2(width), incumbent._BLOCK_W_CAP)
    columns = triton.cdiv(width, block_w)
    if num_rows and columns:
        _reduce[(num_rows, columns)](grad_out, order, offsets,
            prefix if prefix is not None else offsets, out, tokens, width,
            num_rows, grad_out.stride(0), limit, incumbent._BLOCK_S, block_w, False,
            num_warps=incumbent._NUM_WARPS, num_stages=1)
        capacity = hot_capacity(tokens, limit)
        if capacity:
            _reduce[(capacity, columns)](grad_out, order, offsets, prefix, out,
                tokens, width, num_rows, grad_out.stride(0), limit, incumbent._BLOCK_S, block_w, True,
                num_warps=incumbent._NUM_WARPS, num_stages=1)
    return out if out_dtype == torch.float32 else out.to(out_dtype)
