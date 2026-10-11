"""dq for the full-FP8 attention backward without atomics.

Default OFF ($DLRM_HSTU_FP8_BWD_DQ_SPLIT=1 arms it; see hstu_flydsl_mha).

The fused full-FP8 kernel (hstu_bwd_dkdv, dqsplit=True) skips its dq MFMA and the
bf16p dq atomics and instead stores each workgroup's e4m3 ds tile -- the exact
A fragments the dq MFMA would have consumed -- into a workspace.  This module
owns the workspace layout and the dq kernel that consumes it:

    dq[m, h, :] = sc * 2**(E8[m, h] - 127) * sum_n ds8[m, n] * k8[n, h, :]

with f32 accumulation over ALL n (the non-split kernel accumulates
per-n-block partials in bf16 through atomics), one plain bf16 store per row, and no dq
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
# The workspace is large and its size changes every call with the jagged
# token count.  A fresh torch.empty each call makes the expandable-segments
# allocator map new pages (hipMemMap) inside the attention backward, and
# under the host-sync-free step every map is a GPU bubble.
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
        # Streaming sequence lengths grow over the run, and every regrow is
        # an allocator map stall.
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


@triton.jit
def _dq_unit(ws_ptr, hdr_ptr, k_ptr, e8_ptr, cu_ptr, dq_ptr, sc, WS_HDR, rs_k,
             rs_dq, b, h, u, H: tl.constexpr, D: tl.constexpr,
             BMQ: tl.constexpr, FP8_DOT: tl.constexpr, STAGES: tl.constexpr,
             NBLK: tl.constexpr, RM: tl.constexpr = 0, CMW: tl.constexpr = "",
             EVW: tl.constexpr = "", CMK: tl.constexpr = "",
             EVK: tl.constexpr = "", SPTR: tl.constexpr = 0,
             WS_C64: tl.constexpr = False, RSK: tl.constexpr = 0):
    """One (b, h, row block u) program body of _dq_from_ds_kernel, verbatim.

    Load knobs ($DLRM_HSTU_FP8_BWD_DQ_SPLIT_OPT, default off, all
    bitwise: same tiles, same dot order).  RM=1: also mask ws rows past the
    sequence end (their dq is never stored; skips fetching dead 128-B lines).
    CMW/EVW, CMK/EVK: tl.load cache_modifier / eviction_policy of the
    read-once ws stream / the reused K tiles ("" = Triton default; CMW/EVW
    are also set by $DLRM_HSTU_FP8_DQ_SPLIT_LD).  SPTR=1 (BMQ=128, NBLK=1):
    scalar per-tile base pointers + int32 offsets.  WS_C64: ws_c64 layout.

    dqpf (default off, SPTR only): RSK > 0 = K row stride as a
    constexpr (== rs_k) so the K tile offsets are provably small/non-negative
    and K becomes buffer loads.  Same tiles, same dots in the same order:
    bitwise dq."""
    st = tl.load(cu_ptr + b).to(tl.int64)
    L = tl.load(cu_ptr + b + 1).to(tl.int64) - st
    m0 = u * BMQ
    if m0 < L:
        nb = (L + 127) // 128
        pre = tl.load(hdr_ptr + b).to(tl.int64)
        base = WS_HDR + (pre + h * nb * (nb + 1)) * 8192
        t0 = m0 // 64
        offs_m = tl.arange(0, BMQ)
        offs_k = tl.arange(0, 128 * NBLK)
        offs_d = tl.arange(0, D)
        kb = offs_k // 128
        kr = offs_k % 128
        if WS_C64:
            # ws_c64 byte position of n = kr (16 B chunk permutation).
            kp = ((kr // 16) % 2) * 64 + (kr // 32) * 16 + kr % 16
        else:
            kp = kr
        acc = tl.zeros((BMQ, D), dtype=tl.float32)
        nj = (t0 + BMQ // 64 - 1) // 2 + 1
        for jc in tl.range(0, (nj + NBLK - 1) // NBLK, num_stages=STAGES):
            if SPTR:
                # BMQ=128, NBLK=1: tiles t0, t0+1 >= 2*jc for every jc < nj,
                # so the ws mask is all-true; scalar int64 tile base + int32
                # tensor offsets lets the backend use buffer loads (masks via
                # OOB offsets, no exec branches).  Same bytes, same dot.
                pa = ws_ptr + base + (2 * jc * nb - jc * (jc - 1)
                                      + (t0 - 2 * jc)) * 8192
                aoff = offs_m[:, None] * 128 + kp[None, :]
                if RM:
                    a = tl.load(pa + aoff, mask=((m0 + offs_m) < L)[:, None],
                                other=0, cache_modifier=CMW,
                                eviction_policy=EVW)
                else:
                    a = tl.load(pa + aoff, cache_modifier=CMW,
                                eviction_policy=EVW)
                pk = k_ptr + (st + 128 * jc) * rs_k + h * D
                if RSK:
                    pk = pk + (kr[:, None] * RSK + offs_d[None, :])
                else:
                    pk = pk + kr[:, None] * rs_k + offs_d[None, :]
                kt = tl.load(pk,
                             mask=((128 * jc + kr) < L)[:, None], other=0,
                             cache_modifier=CMK, eviction_policy=EVK)
            else:
                j = jc * NBLK + kb
                slot = 2 * j * nb - j * (j - 1) + (t0 - 2 * j)
                ok = ((t0 + offs_m // 64)[:, None] >= 2 * j[None, :]) & \
                    (j < nj)[None, :]
                if RM:
                    ok = ok & ((m0 + offs_m) < L)[:, None]
                a = tl.load(ws_ptr + base + slot[None, :] * 8192
                            + offs_m[:, None] * 128 + kp[None, :], mask=ok,
                            other=0, cache_modifier=CMW, eviction_policy=EVW)
                kn = 128 * j + kr
                kt = tl.load(k_ptr + (st + kn)[:, None] * rs_k + h * D
                             + offs_d[None, :], mask=(kn < L)[:, None],
                             other=0, cache_modifier=CMK, eviction_policy=EVK)
            a = a.to(tl.float8e4nv, bitcast=True)
            kt = kt.to(tl.float8e4nv, bitcast=True)
            if FP8_DOT:
                acc = tl.dot(a, kt, acc)
            else:
                acc = tl.dot(a.to(tl.bfloat16), kt.to(tl.bfloat16), acc)
        rows = m0 + offs_m
        rmask = rows < L
        e8 = tl.load(e8_ptr + (st + rows) * H + h, mask=rmask, other=127)
        rs = (e8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        out = acc * (rs * sc)[:, None]
        tl.store(dq_ptr + (st + rows)[:, None] * rs_dq + h * D
                 + offs_d[None, :], out.to(tl.bfloat16), mask=rmask[:, None])


@triton.jit
def _dq_bc_bounds_kernel(pre_ptr, bnd_ptr, B, STEPS):
    """bc schedule: bnd[x] = first sequence whose ws tile prefix reaches x/8 of
    the total (x = 0..8, bnd[8] = B), one tiny program, no host sync."""
    kx = tl.arange(0, 16)
    total = tl.load(pre_ptr + B).to(tl.int64)
    tgt = (total * kx + 7) // 8
    lo = tl.zeros((16,), dtype=tl.int32)
    hi = tl.zeros((16,), dtype=tl.int32) + B
    for _ in range(STEPS):
        mid = (lo + hi) // 2
        v = tl.load(pre_ptr + mid).to(tl.int64)
        right = v < tgt
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where(right, hi, mid)
    lo = tl.where(kx >= 8, B, lo)
    tl.store(bnd_ptr + kx, lo, mask=kx < 9)


@triton.jit
def _dq_from_ds_sched_kernel(ws_ptr, hdr_ptr, k_ptr, e8_ptr, cu_ptr, dq_ptr,
                             perm_ptr, sc, WS_HDR, rs_k, rs_dq, U_TILES, BH,
                             H: tl.constexpr, D: tl.constexpr,
                             BMQ: tl.constexpr, FP8_DOT: tl.constexpr,
                             STAGES: tl.constexpr, NBLK: tl.constexpr,
                             LPT: tl.constexpr, MODE: tl.constexpr,
                             RM: tl.constexpr = 0, CMW: tl.constexpr = "",
                             EVW: tl.constexpr = "", CMK: tl.constexpr = "",
                             EVK: tl.constexpr = "", SPTR: tl.constexpr = 0,
                             WS_C64: tl.constexpr = False,
                             RSK: tl.constexpr = 0):
    """XCD-balanced schedules ($DLRM_HSTU_FP8_BWD_DQ_SPLIT_SCHED).

    The 8 XCDs receive hardware pids round-robin (pid % 8) and each XCD gets
    the same NUMBER of programs no matter how much work they carry.  The "xcd"
    order hands XCD x one contiguous slice of B/8 sequences; in-model batches
    are runs of consecutive anchors of the same user (similar lengths), so the
    slices differ in work and the slowest XCD sets the kernel time.

    MODE 0 (il/lpt): XCD x = pid % 8 owns the (b, h) units s = rank*H + h with
    s % 8 == x (rank: identity, or by descending length when LPT=1).
    MODE 1 (ilb): XCD x owns sequences r % 8 == x, all H heads back to back.
    MODE 2 (ilr): il, each XCD starting at a different eighth of its units.
    MODE 3 (bc): XCD x owns the CONTIGUOUS sequences [bnd[x], bnd[x+1]) whose
    boundaries split the ws tile prefix into 8 equal parts (perm_ptr = bnd);
    a program walks units J, J + NJ, ... of its range (variable unit count).
    Inside one XCD a unit's U row blocks stay consecutive (heavy first), so
    its K tiles stay in that XCD's L2.  Every program body is that of
    _dq_from_ds_kernel (_dq_unit): bitwise-identical dq.
    """
    pid = tl.program_id(0)
    x = pid % 8
    i = pid // 8
    u = U_TILES - 1 - i % U_TILES
    J = i // U_TILES                       # unit index inside XCD x
    if MODE == 3:
        NJ = tl.num_programs(0) // 8 // U_TILES
        lo = tl.load(perm_ptr + x) * H
        hi = tl.load(perm_ptr + x + 1) * H
        for s in range(lo + J, hi, NJ):
            _dq_unit(ws_ptr, hdr_ptr, k_ptr, e8_ptr, cu_ptr, dq_ptr, sc,
                     WS_HDR, rs_k, rs_dq, s // H, s % H, u, H, D, BMQ,
                     FP8_DOT, STAGES, NBLK, RM, CMW, EVW, CMK, EVK, SPTR,
                     WS_C64, RSK)
    else:
        if MODE == 2:
            NJ = (BH + 7) // 8
            J = (J + x * (NJ // 8)) % NJ
        if MODE == 1:
            r = (J // H) * 8 + x
            h = J % H
            s = r * H + h
        else:
            s = J * 8 + x
            h = s % H
            r = s // H
        if s < BH:
            if LPT:
                b = tl.load(perm_ptr + r)
            else:
                b = r
            _dq_unit(ws_ptr, hdr_ptr, k_ptr, e8_ptr, cu_ptr, dq_ptr, sc,
                     WS_HDR, rs_k, rs_dq, b, h, u, H, D, BMQ, FP8_DOT,
                     STAGES, NBLK, RM, CMW, EVW, CMK, EVK, SPTR, WS_C64, RSK)


def dq_from_ds(ws: torch.Tensor, k: torch.Tensor, e8: torch.Tensor,
               seq_offsets: torch.Tensor, dq: torch.Tensor, N: int,
               sc: float, rs_k: int, rs_dq: int, fp8_dot: bool = True,
               bmq: int = 128, num_warps: int = 8, stages: int = 2,
               nblk: int = 1, order: str = "bh", sched: str = "",
               rm: int = 0, cmw: str = "", evw: str = "", cmk: str = "",
               evk: str = "", sptr: int = 0, ws_cm: str = "", ws_ev: str = "",
               ws_c64: bool = False, rsk: int = 0,
               **launch_kw) -> None:
    """dq (bf16, row stride rs_dq elements, head stride D) from the workspace.

    k: e4m3 [T, H, D] view with row stride rs_k elements; e8: [T, H] uint8.
    order: "bh" = (b, h)-major grid; "xcd" = same grid with the XCD-grouped
    pid remap.
    sched: "" (default) = the order above; "il" / "lpt" / "ilb" / "ilr" /
    "bc" = XCD-balanced schedules (_dq_from_ds_sched_kernel; "lpt" also walks
    sequences longest first, "ilb" keeps a sequence's heads on one XCD, "ilr"
    starts each XCD at a different eighth of its units, "bc" keeps contiguous
    per-XCD slices but cuts them at equal ws-tile work).  Same programs and
    math, bitwise-identical dq; overrides order.
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
    # ws_cm/ws_ev ($DLRM_HSTU_FP8_DQ_SPLIT_LD) and cmw/evw
    # ($DLRM_HSTU_FP8_BWD_DQ_SPLIT_OPT w*) set the same ws load policy.
    assert not (ws_cm and cmw and ws_cm != cmw), (ws_cm, cmw)
    assert not (ws_ev and evw and ws_ev != evw), (ws_ev, evw)
    cmw, evw = cmw or ws_cm, evw or ws_ev
    opt = {}
    if rm or cmw or evw or cmk or evk or sptr or ws_c64:
        # Load knobs: sched body only; omitted when off so the
        # default launch is unchanged.
        assert sched, ("rm/cmw/evw/cmk/evk/sptr, DLRM_HSTU_FP8_DQ_SPLIT_LD and "
                       "WS_LAYOUT=c64 need a dq-split sched")
        assert not sptr or (bmq == 128 and nblk == 1), (bmq, nblk)
        opt = dict(RM=int(rm), CMW=cmw, EVW=evw, CMK=cmk, EVK=evk,
                   SPTR=int(bool(sptr)))
        if ws_c64:
            opt["WS_C64"] = True
    if rsk:
        # kb (sched + sptr body only; omitted when off).
        assert sched and sptr, ("dq-split kb needs sp and a sched")
        assert int(rs_k) > 0, rs_k
        opt["RSK"] = int(rs_k)
    if sched:
        assert sched in ("il", "lpt", "ilb", "ilr", "bc"), sched
        BH = B * H
        if sched == "bc":
            perm = torch.empty(16, dtype=torch.int32, device=ws.device)
            _dq_bc_bounds_kernel[(1,)](hdr, perm, B, int(B).bit_length() + 1,
                                       num_warps=1)
        elif sched == "lpt":
            lens = seq_offsets[1:] - seq_offsets[:-1]
            perm = torch.argsort(lens, descending=True, stable=True).to(
                torch.int32)
        else:
            perm = hdr  # unused (LPT=0)
        units = (triton.cdiv(B, 8) * H if sched == "ilb"
                 else triton.cdiv(BH, 8))  # units per XCD
        _dq_from_ds_sched_kernel[(units * 8 * U,)](
            ws, hdr, k.view(torch.uint8), e8, seq_offsets, dq, perm,
            float(sc), ws_header_bytes(B), int(rs_k), int(rs_dq), U, BH,
            H=H, D=D, BMQ=int(bmq), FP8_DOT=bool(fp8_dot), STAGES=int(stages),
            NBLK=int(nblk), LPT=int(sched == "lpt"),
            MODE={"il": 0, "lpt": 0, "ilb": 1, "ilr": 2, "bc": 3}[sched],
            num_warps=int(num_warps), **opt, **launch_kw)
        return
    _dq_from_ds_kernel[(B * H * U,)](
        ws, hdr, k.view(torch.uint8), e8, seq_offsets, dq, float(sc),
        ws_header_bytes(B), int(rs_k), int(rs_dq), U,
        H=H, D=D, BMQ=int(bmq), FP8_DOT=bool(fp8_dot), STAGES=int(stages),
        NBLK=int(nblk), XCD=int(order == "xcd"),
        num_warps=int(num_warps), **launch_kw)
