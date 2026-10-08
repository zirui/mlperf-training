"""GPU offsets for the existing EC caller's already-valid integer row IDs.

The caller must establish 0 <= ID < num_rows (the EC forward gather does).
This is deliberately not a general invalid-input-compatible bincount API.
The incumbent argsort, including duplicate tie order, is retained unchanged.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _search(IDX, ORDER, query, active, T, STRIDE):
    left = tl.full(query.shape, 0, tl.int64)
    right = tl.full(query.shape, T, tl.int64)
    remaining = T.to(tl.int64)
    while remaining > 0:
        live = active & (left < right)
        middle = left + ((right - left) >> 1)
        source = tl.load(ORDER + middle, live, other=0).to(tl.int64)
        value = tl.load(IDX + source * STRIDE.to(tl.int64), live, other=0).to(tl.int64)
        advance = live & (value < query)
        left = tl.where(advance, middle + 1, left)
        right = tl.where(live & ~advance, middle, right)
        remaining = remaining >> 1
    return left


@triton.jit(do_not_specialize=["T", "U", "STRIDE"])
def _offsets(IDX, ORDER, OUT, T, U, STRIDE, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    tokens = T.to(tl.int64)
    rows = U.to(tl.int64)
    location = _search(IDX, ORDER, row, (row > 0) & (row < rows), tokens, STRIDE)
    # The valid-ID precondition proves both endpoints, including T=U=0.
    location = tl.where(row == rows, tokens, location)
    tl.store(OUT + row, location, row <= rows)


def _metadata(reverse_indices, num_rows):
    if not isinstance(num_rows, int) or isinstance(num_rows, bool) or num_rows < 0:
        raise ValueError("num_rows must be a nonnegative Python integer")
    if reverse_indices.device.type != 'cuda' or reverse_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("valid EC IDs must be CUDA int32 or int64")
    idx = reverse_indices.reshape(-1)
    if num_rows == 0 and idx.numel():
        raise ValueError("nonempty IDs cannot be valid with num_rows=0")
    return idx


def offsets_from_order_valid_ids(reverse_indices, order, num_rows, out=None):
    """Preserve a supplied incumbent permutation; optional out is for redzones."""
    idx = _metadata(reverse_indices, num_rows)
    if order.dtype != torch.int64 or order.device != idx.device or order.dim() != 1 or not order.is_contiguous() or order.numel() != idx.numel():
        raise ValueError("order must be the contiguous int64 argsort of the flattened IDs")
    if out is None:
        out = torch.empty(num_rows + 1, dtype=torch.int64, device=idx.device)
    elif out.dtype != torch.int64 or out.device != idx.device or out.shape != (num_rows + 1,) or not out.is_contiguous():
        raise ValueError("offsets output must be contiguous int64 of length num_rows+1")
    _offsets[(triton.cdiv(num_rows + 1, 128),)](
        idx, order, out, idx.numel(), num_rows, idx.stride(0), 128, num_warps=2,
    )
    return out


def build_segments_valid_ids(reverse_indices, num_rows, order=None):
    """Return exact incumbent order and fixed-size offsets for valid EC IDs."""
    idx = _metadata(reverse_indices, num_rows)
    if order is None:
        order = torch.argsort(idx)
    return order, offsets_from_order_valid_ids(idx, order, num_rows)
