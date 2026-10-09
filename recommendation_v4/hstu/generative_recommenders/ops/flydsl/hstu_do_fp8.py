# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""C8: bf16 dO -> e4m3 dO with device-side power-of-two block scales.

Used only by the full-fp8 attention backward (f8b="all", do_blk=True), which
is gated behind $DLRM_HSTU_FP8_BWD_FULL (absent = OFF; gin default ON).

One pow2 scale per (sequence, head, 32-token MX K-block group), no host sync.
The earlier per-tensor variant (quantize_do_e4m3, kernel do_dyn) was removed
once block scales landed: in-model dO rows span ~1e4 in magnitude and the single scale
flushed 87-96% of dO elements to zero, and nothing
in the tree used it once the block scales became the only C8 path.
"""

import torch
import triton
import triton.language as tl

E4M3_MAX = 448.0


# ---- C8 option B: per-(sequence, head, token-group) pow2 dO scale ---------
# The group of seq-relative token p is the HARDWARE MX K-block grouping the
# full-fp8 kernel's out-gemms see (hstu_bwd_dkdv do_blk):
#     G(p) = (p // 64, (p % 16) // 8)
# i.e. inside each 64-token tile, tokens with p % 16 in [0, 8) form one group
# and [8, 16) the other (32 tokens each).  Per (sequence, head, group):
#     e_g = floor(log2(448 / amax_g)), clamped to [-60, 60]; 0 if amax_g == 0
#     dO8 = e4m3(dO * 2**e_g)        E8[p, h] = 127 - e_g  (E8M0 byte of 2**-e_g)
# e_g comes from the fp32 bit pattern (exact; a float log2 is off by one at
# exact powers of two, which would give amax * s = 896 -> e4m3 NaN):
#     amax = 1.f * 2**E:  448/amax = (1.75/1.f) * 2**(8-E)
#     -> e = 8 - E if 1.f <= 1.75 else 7 - E,   so amax * 2**e in (224, 448].


@triton.jit
def _grp_exp(a):
    bits = a.to(tl.int32, bitcast=True)
    E = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    e = tl.where(mant <= 0x600000, 135 - E, 134 - E)
    e = tl.minimum(tl.maximum(e, -60), 60)
    e = tl.where(E == 0, 60, e)          # fp32 subnormal amax
    return tl.where(a == 0.0, 0, e)      # all-zero group


@triton.jit
def _blk_kernel(x_ptr, y_ptr, e8_ptr, offs_ptr, H: tl.constexpr,
                D: tl.constexpr):
    b = tl.program_id(0)
    t = tl.program_id(1)
    st = tl.load(offs_ptr + b).to(tl.int64)
    L = tl.load(offs_ptr + b + 1).to(tl.int64) - st
    p0 = t * 64
    if p0 < L:
        rows = tl.arange(0, 64)
        cols = tl.arange(0, H * D)
        valid = (p0 + rows) < L
        tok = st + p0 + rows                                    # [64] int64
        x = tl.load(x_ptr + tok[:, None] * (H * D) + cols[None, :],
                    mask=valid[:, None], other=0.0).to(tl.float32)
        x3 = tl.reshape(x, (64, H, D))
        ram = tl.max(tl.abs(x3), axis=2)                         # [64, H]
        sub0 = ((rows % 16) // 8) == 0                           # [64]
        a0 = tl.max(tl.where(sub0[:, None], ram, 0.0), axis=0)   # [H]
        a1 = tl.max(tl.where(sub0[:, None], 0.0, ram), axis=0)
        e0 = _grp_exp(a0)
        e1 = _grp_exp(a1)
        e_row = tl.where(sub0[:, None], e0[None, :], e1[None, :])
        # 2**e exactly, from the exponent field (e in [-60, 60]).
        s = ((e_row + 127) << 23).to(tl.float32, bitcast=True)   # [64, H]
        y = tl.reshape(x3 * s[:, :, None], (64, H * D))
        tl.store(y_ptr + tok[:, None] * (H * D) + cols[None, :],
                 y.to(tl.float8e4nv), mask=valid[:, None])
        hh = tl.arange(0, H)
        tl.store(e8_ptr + tok[:, None] * H + hh[None, :],
                 (127 - e_row).to(tl.uint8), mask=valid[:, None])


def quantize_do_e4m3_blk(dout: torch.Tensor, seq_offsets: torch.Tensor,
                         N: int):
    """Per-(sequence, head, MX token group) pow2 e4m3 quantization of dO.

    dout [T, H, D] bf16/f32, seq_offsets [B+1] (any int dtype, device),
    N >= every sequence length (the launcher's max_seq_len; host int, no
    sync).  Returns (dO8 [T, H, D] e4m3, E8 [T, H] uint8 = 127 - e_g).
    """
    x = dout.contiguous()
    T, H, D = x.shape
    assert (H * D) & (H * D - 1) == 0 and H & (H - 1) == 0, (H, D)
    y = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    e8 = torch.empty((T, H), dtype=torch.uint8, device=x.device)
    offs = seq_offsets.to(device=x.device, dtype=torch.int64).contiguous()
    B = offs.numel() - 1
    if B > 0 and T > 0:
        _blk_kernel[(B, triton.cdiv(int(N), 64))](
            x, y, e8, offs, H=H, D=D, num_warps=8)
    return y, e8
