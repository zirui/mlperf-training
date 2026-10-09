# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""R15 design B: dq for the full-FP8 (C8) attention backward without atomics.

Default OFF ($DLRM_HSTU_FP8_BWD_DQ_SPLIT=1 arms it; see hstu_flydsl_mha).

The fused C8 kernel (hstu_bwd_dkdv, dqsplit=True) skips its dq MFMA and the
bf16p dq atomics and instead stores each workgroup's e4m3 ds tile -- the exact
A fragments the dq MFMA would have consumed -- into a workspace.  This module
owns the workspace layout and the dq kernel that consumes it:

    dq[m, h, :] = sc * 2**(E8[m, h] - 127) * sum_n ds8[m, n] * k8[n, h, :]

with f32 accumulation over ALL n (the shipped kernel accumulates per-n-block
partials in bf16 through atomics), one plain bf16 store per row, and no dq
zero-fill.  Same products, same scales; deterministic.

Workspace (uint8):
    [0, WS_HDR)        int32 prefix[B+1] of per-batch tile counts
    [WS_HDR, ...)      8 KiB tiles, tile = ds [m 64][n 128] e4m3 (m-major)
(b, h) owns nb*(nb+1) consecutive tiles (nb = ceil(L_b / 128)), H-major inside
the batch; n-block j owns tiles t = 2j .. 2nb-1 (64-row m tiles) at slot
2*j*nb - j*(j-1) + (t - 2j).  Slots (j, 2u) and (j, 2u+1) are adjacent, so the
128 rows [128u, 128u+128) of n-block j are one contiguous [128][128] block.
Must match hstu_bwd_dkdv (DQSPLIT).

Caveat: rows of dq past seq_offsets[-1] (a padded token tail) are not written;
the production call has q.shape[0] == seq_offsets[-1].
"""

import torch
import triton
import triton.language as tl

TILE_BYTES = 64 * 128


def ws_header_bytes(batch: int) -> int:
    return ((batch + 1) * 4 + 255) // 256 * 256


# Persistent grow-only workspace ($DLRM_HSTU_FP8_BWD_DQ_WS_PERSIST, default 1).
# The workspace is ~10-19 GB and its size changes every call with the jagged
# token count.  A fresh torch.empty each call made the expandable-segments
# allocator map new pages (hipMemMap) about once per rank-step inside the
# attention backward, and under the host-sync-free step every map is a GPU
# bubble (hipMemMap host time 0.56 -> 2.1 s per pairing).
# Keeping one buffer per device, rounded up to 1 GiB with geometric headroom
# and regrown only when a call needs more, removes the per-call maps.  Contents are
# never read before the fused kernel writes them (same as torch.empty), and
# every use is on the current stream, so stream order makes the reuse safe.
_WS_CACHE = {}
_WS_ROUND = 1 << 30


def _workspace(nbytes: int, device) -> torch.Tensor:
    import os
    if os.environ.get("DLRM_HSTU_FP8_BWD_DQ_WS_PERSIST", "1").strip() in (
            "0", "false", "off"):
        return torch.empty(nbytes, dtype=torch.uint8, device=device)
    device = torch.device(device)
    key = (device.type, device.index if device.index is not None
           or device.type != "cuda" else torch.cuda.current_device())
    buf = _WS_CACHE.get(key)
    if buf is None or buf.numel() < nbytes:
        old = 0 if buf is None else buf.numel()
        _WS_CACHE.pop(key, None)
        del buf
        # Geometric growth: need + 1/4, and at least 1.5x the previous size.
        # Streaming sequence lengths grow over the run; 1/8 headroom with
        # 1 GiB rounding regrew 11-14 times per device in a 200-step bench
        # and every regrow is an allocator map stall.
        want = max(nbytes + nbytes // 4, old + old // 2)
        want = (want + _WS_ROUND - 1) // _WS_ROUND * _WS_ROUND
        buf = torch.empty(want, dtype=torch.uint8, device=device)
        _WS_CACHE[key] = buf
        import logging
        logging.getLogger(__name__).warning(
            "[hstu-attn] dq-split ds workspace PERSIST grow: %.2f GiB "
            "(need %.2f GiB) on %s", want / 2**30, nbytes / 2**30, key)
    return buf[:nbytes]


def build_workspace(seq_offsets: torch.Tensor, heads: int, total_tokens: int,
                    N: int) -> torch.Tensor:
    """Allocate the ds workspace and write its prefix header, no host sync.

    The tile count is bounded on the host from (total tokens, batch, N),
    assuming every sequence length <= N (the launcher's contract):
    sum_b nb_b (nb_b + 1) <= (nb_max + 1) * sum_b nb_b and
    sum_b nb_b <= (T + 127 B) / 128.
    """
    B = seq_offsets.numel() - 1
    nb_max = (int(N) + 127) // 128
    sum_nb = (int(total_tokens) + 127 * B) // 128
    tiles_bound = heads * (nb_max + 1) * sum_nb
    hdr = ws_header_bytes(B)
    ws = _workspace(hdr + tiles_bound * TILE_BYTES, seq_offsets.device)
    offs = seq_offsets.to(torch.int64)
    nb = (offs[1:] - offs[:-1] + 127) // 128
    pre = ws[: 4 * (B + 1)].view(torch.int32)
    pre[0].zero_()
    pre[1:].copy_(torch.cumsum(heads * nb * (nb + 1), 0))
    return ws


@triton.jit
def _dq_from_ds_kernel(ws_ptr, hdr_ptr, k_ptr, e8_ptr, cu_ptr, dq_ptr, sc,
                       WS_HDR, rs_k, rs_dq, U_TILES,
                       H: tl.constexpr, D: tl.constexpr, BMQ: tl.constexpr,
                       FP8_DOT: tl.constexpr, STAGES: tl.constexpr,
                       NBLK: tl.constexpr, XCD: tl.constexpr):
    """BMQ (64/128/256) dq rows per program, NBLK n-blocks (K = 128*NBLK)
    per dot; heavy (late) row blocks first and consecutive logical programs
    share (b, h) so K stays in L2.

    XCD=1 remaps the hardware pid so each of the 8 XCDs (which receive pids
    round-robin, pid % 8) runs one contiguous slice of the logical grid; the
    row blocks of a (b, h) then share one XCD's private L2 for their K tiles
    instead of missing in all eight.  Pure scheduling: same programs, same
    math, bitwise-identical output."""
    pid = tl.program_id(0)
    if XCD:
        G8 = tl.num_programs(0) // 8
        pid = tl.where(pid < G8 * 8, (pid % 8) * G8 + pid // 8, pid)
    u = U_TILES - 1 - pid % U_TILES
    bh = pid // U_TILES
    h = bh % H
    b = bh // H
    st = tl.load(cu_ptr + b).to(tl.int64)
    L = tl.load(cu_ptr + b + 1).to(tl.int64) - st
    m0 = u * BMQ
    if m0 < L:
        nb = (L + 127) // 128
        pre = tl.load(hdr_ptr + b).to(tl.int64)
        base = WS_HDR + (pre + h * nb * (nb + 1)) * 8192
        t0 = m0 // 64                     # first 64-row tile of the block
        offs_m = tl.arange(0, BMQ)
        offs_k = tl.arange(0, 128 * NBLK)
        offs_d = tl.arange(0, D)
        kb = offs_k // 128                # n-block within the K chunk
        kr = offs_k % 128
        acc = tl.zeros((BMQ, D), dtype=tl.float32)
        nj = (t0 + BMQ // 64 - 1) // 2 + 1  # n-blocks touching this row block
        for jc in tl.range(0, (nj + NBLK - 1) // NBLK, num_stages=STAGES):
            j = jc * NBLK + kb                                    # [K]
            slot = 2 * j * nb - j * (j - 1) + (t0 - 2 * j)        # [K]
            ok = ((t0 + offs_m // 64)[:, None] >= 2 * j[None, :]) & \
                (j < nj)[None, :]
            a = tl.load(ws_ptr + base + slot[None, :] * 8192
                        + offs_m[:, None] * 128 + kr[None, :], mask=ok, other=0)
            a = a.to(tl.float8e4nv, bitcast=True)
            kn = 128 * j + kr
            kt = tl.load(k_ptr + (st + kn)[:, None] * rs_k + h * D
                         + offs_d[None, :], mask=(kn < L)[:, None], other=0)
            kt = kt.to(tl.float8e4nv, bitcast=True)
            if FP8_DOT:
                acc = tl.dot(a, kt, acc)
            else:
                # e4m3 -> bf16 is exact, so the products are identical.
                acc = tl.dot(a.to(tl.bfloat16), kt.to(tl.bfloat16), acc)
        rows = m0 + offs_m
        rmask = rows < L
        e8 = tl.load(e8_ptr + (st + rows) * H + h, mask=rmask, other=127)
        rs = (e8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        out = acc * (rs * sc)[:, None]
        tl.store(dq_ptr + (st + rows)[:, None] * rs_dq + h * D + offs_d[None, :],
                 out.to(tl.bfloat16), mask=rmask[:, None])


def dq_from_ds(ws: torch.Tensor, k: torch.Tensor, e8: torch.Tensor,
               seq_offsets: torch.Tensor, dq: torch.Tensor, N: int,
               sc: float, rs_k: int, rs_dq: int, fp8_dot: bool = True,
               bmq: int = 128, num_warps: int = 8, stages: int = 2,
               nblk: int = 1, order: str = "bh", **launch_kw) -> None:
    """dq (bf16, row stride rs_dq elements, head stride D) from the workspace.

    k: e4m3 [T, H, D] view with row stride rs_k elements; e8: [T, H] uint8.
    order: "bh" = (b, h)-major grid; "xcd" = same grid with
    the XCD-grouped pid remap (dense lens 6.6 -> 4.4 ms).
    """
    assert order in ("bh", "xcd"), order
    B = seq_offsets.numel() - 1
    H, D = k.shape[1], k.shape[2]
    # 256 is excluded: its tile range t0..t0+3 is not bounded by 2*nb-1, so the
    # last (b, h) can read past the workspace end.
    assert bmq in (64, 128), bmq
    U = triton.cdiv(int(N), bmq)
    if B <= 0 or U <= 0:
        return
    hdr = ws[: 4 * (B + 1)].view(torch.int32)
    _dq_from_ds_kernel[(B * H * U,)](
        ws, hdr, k.view(torch.uint8), e8, seq_offsets, dq, float(sc),
        ws_header_bytes(B), int(rs_k), int(rs_dq), U,
        H=H, D=D, BMQ=int(bmq), FP8_DOT=bool(fp8_dot), STAGES=int(stages),
        NBLK=int(nblk), XCD=int(order == "xcd"),
        num_warps=int(num_warps), **launch_kw)
