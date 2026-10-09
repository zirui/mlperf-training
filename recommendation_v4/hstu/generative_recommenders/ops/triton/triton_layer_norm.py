# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

#!/usr/bin/env python3


import os
from typing import List, Optional, Tuple

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl
from generative_recommenders.common import (
    switch_to_contiguous_if_needed,
    triton_autotune,
)
from generative_recommenders.ops.triton._autotune_pinning import pinned_or_full
from generative_recommenders.ops.utils import (
    is_sm100_plus,
    is_sm90,
    maybe_register_custom_op,
)

try:
    # @manual=//triton:triton
    from triton.language.extra.libdevice import fast_dividef, rsqrt as libdevice_rsqrt
except ImportError:
    try:
        # @manual=//triton:triton
        from triton.language.extra.cuda.libdevice import (
            fast_dividef,
            rsqrt as libdevice_rsqrt,
        )
    except ImportError:
        # pyre-ignore: Undefined import [21]
        # @manual=//triton:triton
        from triton.language.math import fast_dividef, rsqrt as libdevice_rsqrt


def _get_layer_norm_fwd_configs() -> List[triton.Config]:
    """Generate autotune configs for multi-row LayerNorm kernels."""
    configs = []
    block_ns = [4, 8, 16] if is_sm100_plus() else [1, 2, 4, 8]
    for BLOCK_N in block_ns:
        for num_warps in [1, 2, 4, 8]:
            configs.append(
                triton.Config(
                    {"BLOCK_N": BLOCK_N},
                    num_warps=num_warps,
                )
            )
    return configs


def _bwd_pre_hook(nargs):
    nargs["DW"].zero_()
    if "DB" in nargs:
        nargs["DB"].zero_()


def _get_norm_bwd_configs() -> List[triton.Config]:
    """Generate autotune configs for multi-row LayerNorm kernels."""
    configs = []
    if is_sm100_plus():
        block_ns = [8, 16]
        num_shards_list = [8, 16]
        num_warps_list = [2, 4]
    elif is_sm90():
        block_ns = [2, 4]
        num_shards_list = [8]
        num_warps_list = [2, 4]
    else:
        block_ns = [1, 2]
        num_shards_list = [8]
        num_warps_list = [2, 4]
    for BLOCK_N in block_ns:
        for num_warps in num_warps_list:
            for num_shards in num_shards_list:
                configs.append(
                    triton.Config(
                        {"BLOCK_N": BLOCK_N, "SHARDS_PER_SM": num_shards},
                        num_warps=num_warps,
                    )
                )
    return configs


@triton_autotune(
    configs=_get_layer_norm_fwd_configs(),
    key=["BLOCK_D"],
)
@triton.jit
def _layer_norm_fwd(
    X,
    Y,
    Mean,
    Rstd,
    N,
    D,
    eps,
    stride_x,
    stride_y,
    TRAINING: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    COMPUTE_MEAN_AND_RSTD: tl.constexpr,
):
    block_id = tl.program_id(0)
    start_row = block_id * BLOCK_N

    X_block_ptr = tl.make_block_ptr(
        base=X,
        shape=(N, D),
        strides=(stride_x, 1),
        offsets=(start_row, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    Y_block_ptr = tl.make_block_ptr(
        base=Y,
        shape=(N, D),
        strides=(stride_y, 1),
        offsets=(start_row, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    x_block = tl.load(X_block_ptr, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )

    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    rows = start_row + tl.arange(0, BLOCK_N)
    row_mask = rows < N

    if COMPUTE_MEAN_AND_RSTD:
        mean = tl.sum(x_block, axis=1) / D
        if TRAINING:
            tl.store(Mean + rows, mean, row_mask)
        mean = tl.expand_dims(mean, 1)
    else:
        mean = tl.load(Mean + rows, row_mask, other=0.0)
        mean = tl.expand_dims(mean, 1)

    x_mean = x_block - mean
    x_mean = tl.where(row_mask[:, None] & col_mask[None, :], x_mean, 0.0)

    if COMPUTE_MEAN_AND_RSTD:
        _var = x_mean * x_mean
        var = tl.sum(_var, axis=1) / D
        rstd = 1 / tl.sqrt(var + eps)
        if TRAINING:
            tl.store(Rstd + rows, rstd, row_mask)
    else:
        rstd = tl.load(Rstd + rows, row_mask, other=0.0)

    rstd = tl.expand_dims(rstd, 1)
    y = x_mean * rstd

    tl.store(Y_block_ptr, y.to(Y.dtype.element_ty), boundary_check=(0, 1))


@triton_autotune(
    configs=_get_layer_norm_fwd_configs(),
    key=["BLOCK_D"],
)
@triton.jit
def _weighted_layer_norm_fwd(
    X,
    Y,
    Y_FP8,
    W,
    B,
    Mean,
    Rstd,
    N,
    D,
    eps,
    stride_x,
    stride_y,
    stride_y_fp8,
    Y_INV_SCALE,
    # F2: optional residual folded into this kernel's epilogue, so that
    # `LN(h) + r` costs one pass instead of a full [M, D] read-modify-write
    # against HBM immediately after the norm has just written that same tensor.
    # Mirrors the machinery _weighted_layer_norm_bwd_dx already carries.
    RESIDUAL,
    stride_residual,
    IS_SWISH: tl.constexpr,
    TRAINING: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    COMPUTE_MEAN_AND_RSTD: tl.constexpr,
    OUTPUT_FP8: tl.constexpr,
    OUTPUT_BOTH: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
):
    # Get the block ID and calculate starting row
    block_id = tl.program_id(0)
    start_row = block_id * BLOCK_N

    # Load weight and bias once (shared across all rows in this block)
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=col_mask, other=0.0).to(tl.float32)

    # Create block pointers for X and Y
    X_block_ptr = tl.make_block_ptr(
        base=X,
        shape=(N, D),
        strides=(stride_x, 1),
        offsets=(start_row, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    Y_block_ptr = tl.make_block_ptr(
        base=Y,
        shape=(N, D),
        strides=(stride_y, 1),
        offsets=(start_row, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )
    if OUTPUT_BOTH:
        Y_FP8_block_ptr = tl.make_block_ptr(
            base=Y_FP8,
            shape=(N, D),
            strides=(stride_y_fp8, 1),
            offsets=(start_row, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )

    x_block = tl.load(X_block_ptr, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )

    rows = start_row + tl.arange(0, BLOCK_N)
    row_mask = rows < N

    if COMPUTE_MEAN_AND_RSTD:
        mean = tl.sum(x_block, axis=1) / D
        if TRAINING:
            tl.store(Mean + rows, mean, row_mask)
        mean = tl.expand_dims(mean, 1)
    else:
        mean = tl.load(Mean + rows, row_mask, other=0.0)
        mean = tl.expand_dims(mean, 1)

    x_mean = x_block - mean
    x_mean = tl.where(row_mask[:, None] & col_mask[None, :], x_mean, 0.0)

    if COMPUTE_MEAN_AND_RSTD:
        _var = x_mean * x_mean
        var = tl.sum(_var, axis=1) / D
        rstd = libdevice_rsqrt(var + eps)
        if TRAINING:
            tl.store(Rstd + rows, rstd, row_mask)
    else:
        rstd = tl.load(Rstd + rows, row_mask, other=0.0)

    rstd = tl.expand_dims(rstd, 1)
    y = x_mean * rstd
    y = y * w[None, :] + b[None, :]

    if IS_SWISH:
        y = tl.sigmoid(y) * x_block

    if ADD_RESIDUAL:
        # ONE insertion point covers ALL THREE store sites below, because every
        # one of them derives from this same `y`. Patching the three stores
        # separately would triplicate this and get the OUTPUT_FP8 case wrong:
        # there `y` is rescaled by Y_INV_SCALE *after* this point, and the
        # residual belongs before that scaling (the fp8 tensor is a scaled copy
        # of the true output, which is LN(x) + r).
        #
        # ROUND BEFORE ADDING. The shipped path this replaces stores the norm
        # output at Y's dtype and only then adds the residual to it, so
        # rounding here reproduces that arithmetic BIT FOR BIT rather than
        # merely closely. _weighted_layer_norm_bwd_dx:487 already answers the
        # identical question the identical way. Adding in fp32 instead would be
        # marginally more accurate and would turn a free fusion into one owing
        # a convergence arm.
        RESIDUAL_block_ptr = tl.make_block_ptr(
            base=RESIDUAL,
            shape=(N, D),
            strides=(stride_residual, 1),
            offsets=(start_row, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )
        y = y.to(Y.dtype.element_ty).to(tl.float32) + tl.load(
            RESIDUAL_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        ).to(tl.float32)

    if OUTPUT_BOTH:
        tl.store(Y_block_ptr, y.to(Y.dtype.element_ty), boundary_check=(0, 1))
        y_fp8 = y * Y_INV_SCALE
        tl.store(
            Y_FP8_block_ptr,
            y_fp8.to(Y_FP8.dtype.element_ty),
            boundary_check=(0, 1),
        )
    else:
        if OUTPUT_FP8:
            # Plan 08 Phi1: the normalized value is already resident in fp32.
            # Scale before the typed E4M3 store so no standalone cast pass is
            # needed by the downstream scaled GEMM.
            y = y * Y_INV_SCALE

        tl.store(Y_block_ptr, y.to(Y.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _layer_norm_bwd_dx(
    DX,
    DY,
    X,
    Mean,
    Rstd,
    stride_dx,
    stride_dy,
    stride_x,
    D,
    eps,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    X += row.to(tl.int64) * stride_x
    DY += row.to(tl.int64) * stride_dy
    DX += row.to(tl.int64) * stride_dx

    # Load data to SRAM
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)

    # Compute dx
    xhat = (x - mean) * rstd
    xhat = tl.where(mask, xhat, 0.0)
    dy = tl.where(mask, dy, 0.0)
    c1 = tl.sum(xhat * dy, axis=0) / D
    c2 = tl.sum(dy, axis=0) / D
    dx = (dy - (xhat * c1 + c2)) * rstd
    # Write dx
    tl.store(DX + cols, dx, mask=mask)


@triton_autotune(
    configs=pinned_or_full(
        [
            triton.Config({"BLOCK_N": 1}, num_warps=1),  # bs=32 winner
            triton.Config({"BLOCK_N": 8}, num_warps=1),  # bs=1024 winner
        ],
        _get_layer_norm_fwd_configs,
    ),
    key=["BLOCK_D"],
)
@triton.jit
def _weighted_layer_norm_bwd_dx(
    DX,
    DY,
    RESIDUAL,
    DW,
    DB,
    DXSUM,
    X,
    W,
    B,
    Mean,
    Rstd,
    stride_dx,
    stride_dy,
    stride_residual,
    stride_x,
    D,
    eps,
    IS_SWISH: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    SUM_DX: tl.constexpr,
    N,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # SUM_DX additionally emits sum(dx, axis=0) into DXSUM, using the same
    # [tile_num, D] partial-buffer scheme the kernel already runs for dw/db.
    # SUM_DX also writes the column sum of dx, which is the preceding Linear's
    # bias gradient. The sum uses the fp32 dx still in registers. aten::sum
    # reduces the stored bf16 dx, so the two are not bit-identical.
    # SUM_DX is a tl.constexpr. With it false, Triton drops the accumulator.
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    num_blocks = tl.cdiv(N, BLOCK_N)
    blocks_per_tile = num_blocks // tile_num
    if pid < num_blocks % tile_num:
        blocks_per_tile += 1

    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)

    acc_dw = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc_db = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc_dxsum = tl.zeros([BLOCK_D], dtype=tl.float32)

    start_block = pid

    for idx in range(blocks_per_tile):
        current_block = start_block + idx * tile_num
        start_row = current_block * BLOCK_N

        X_block_ptr = tl.make_block_ptr(
            base=X,
            shape=(N, D),
            strides=(stride_x, 1),
            offsets=(start_row, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )

        DX_block_ptr = tl.make_block_ptr(
            base=DX,
            shape=(N, D),
            strides=(stride_dx, 1),
            offsets=(start_row, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )

        DY_block_ptr = tl.make_block_ptr(
            base=DY,
            shape=(N, D),
            strides=(stride_dy, 1),
            offsets=(start_row, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )
        RESIDUAL_block_ptr = tl.make_block_ptr(
            base=RESIDUAL,
            shape=(N, D),
            strides=(stride_residual, 1),
            offsets=(start_row, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )

        # Load data blocks
        x_block = tl.load(X_block_ptr, boundary_check=(0, 1), padding_option="zero").to(
            tl.float32
        )
        dy_block = tl.load(
            DY_block_ptr, boundary_check=(0, 1), padding_option="zero"
        ).to(tl.float32)

        # Load mean and rstd for all rows in this block
        rows = start_row + tl.arange(0, BLOCK_N)
        row_mask = rows < N
        mean = tl.load(Mean + rows, row_mask, other=0.0)
        rstd = tl.load(Rstd + rows, row_mask, other=0.0)

        # Expand dimensions for broadcasting
        mean = tl.expand_dims(mean, 1)
        rstd = tl.expand_dims(rstd, 1)

        xhat = (x_block - mean) * rstd

        xhat = tl.where(row_mask[:, None] & col_mask[None, :], xhat, 0.0)
        wdy = w[None, :] * dy_block
        wdy = tl.where(row_mask[:, None] & col_mask[None, :], wdy, 0.0)

        # Compute dx
        if IS_SWISH:
            b = tl.load(B + cols, mask=col_mask, other=0.0).to(tl.float32)
            sigmoid_layer_norm = tl.sigmoid(xhat * w[None, :] + b[None, :])
            sigmoid_layer_norm = tl.where(
                row_mask[:, None] & col_mask[None, :], sigmoid_layer_norm, 0.0
            )

            sigmoid_deriv = sigmoid_layer_norm * (1 - sigmoid_layer_norm)
            x_ = wdy * x_block * sigmoid_deriv
            x_ = tl.where(row_mask[:, None] & col_mask[None, :], x_, 0.0)

            c1 = tl.sum(xhat * x_, axis=1) / D
            c2 = tl.sum(x_, axis=1) / D
            c1 = tl.expand_dims(c1, 1)
            c2 = tl.expand_dims(c2, 1)
            dx = (x_ - (xhat * c1 + c2)) * rstd

            dx = dy_block * sigmoid_layer_norm + dx
            if ADD_RESIDUAL:
                dx = dx.to(DX.dtype.element_ty).to(tl.float32) + tl.load(
                    RESIDUAL_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                ).to(tl.float32)
            # Write dx
            tl.store(DX_block_ptr, dx.to(DX.dtype.element_ty), boundary_check=(0, 1))
            partial_dw = tl.sum(dy_block * x_block * xhat * sigmoid_deriv, axis=0)
            partial_db = tl.sum(dy_block * x_block * sigmoid_deriv, axis=0)
        else:
            c1 = tl.sum(xhat * wdy, axis=1) / D
            c2 = tl.sum(wdy, axis=1) / D
            c1 = tl.expand_dims(c1, 1)
            c2 = tl.expand_dims(c2, 1)
            dx = (wdy - (xhat * c1 + c2)) * rstd
            if ADD_RESIDUAL:
                dx = dx.to(DX.dtype.element_ty).to(tl.float32) + tl.load(
                    RESIDUAL_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                ).to(tl.float32)
            # Write dx
            tl.store(DX_block_ptr, dx.to(DX.dtype.element_ty), boundary_check=(0, 1))
            partial_dw = tl.sum(dy_block * xhat, axis=0)
            partial_db = tl.sum(dy_block, axis=0)

        # Accumulate partial sums in shared memory
        acc_dw += partial_dw
        acc_db += partial_db
        if SUM_DX:
            # dx is defined by both arms of the IS_SWISH branch above and is
            # still resident. Rows >= N need no mask for the same reason
            # partial_db does not: every term feeding them is either padded to
            # zero on load or explicitly row-masked, so dx is zero there.
            acc_dxsum += tl.sum(dx, axis=0)

    # Store accumulated sums back to global memory
    dw_ptrs = DW + pid.to(tl.int64) * D + cols
    db_ptrs = DB + pid.to(tl.int64) * D + cols
    tl.store(dw_ptrs, acc_dw, mask=col_mask)
    tl.store(db_ptrs, acc_db, mask=col_mask)
    if SUM_DX:
        dxsum_ptrs = DXSUM + pid.to(tl.int64) * D + cols
        tl.store(dxsum_ptrs, acc_dxsum, mask=col_mask)


@triton.jit
def _weighted_layer_norm_bwd_dx_v2(
    DX,
    DY,
    RESIDUAL,
    DW,
    DB,
    DXSUM,
    X,
    W,
    B,
    Mean,
    Rstd,
    stride_dx,
    stride_dy,
    stride_residual,
    stride_x,
    D,
    eps,
    IS_SWISH: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    SUM_DX: tl.constexpr,
    N,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    CONTIG: tl.constexpr,
    EVEN_D: tl.constexpr,
    FAST_SIGMOID: tl.constexpr = False,
):
    """Pipelined variant of _weighted_layer_norm_bwd_dx (same math per element).

    Differences from the v1 kernel, none of which change the arithmetic:
      * raw-pointer loads/stores instead of tl.make_block_ptr, and
        tl.range(num_stages=NUM_STAGES) so the loads of block i+1 are issued
        while block i is still being reduced (v1's plain range() loop is not
        software-pipelined on gfx950);
      * B is loaded once per program rather than once per row block;
      * CONTIG selects contiguous row-block ranges per program instead of the
        v1 strided walk; it only changes which program's dw/db/dxsum partial
        a row lands in (fp32 reduction order), never dx;
      * EVEN_D (BLOCK_D == D) drops the column masks, which are all-true.
    FAST_SIGMOID (swish only) is the one arithmetic option: sigmoid via
    fast_dividef(1, 1 + exp(-z)) instead of tl.sigmoid; see
    _wln_bwd_v2_fast_sigmoid for its numerics.
    The [tile_num, D] dw/db/dxsum partial-buffer contract is unchanged, so the
    same _layer_norm_bwd_dwdb / _layer_norm_bwd_dwdbdx finalize consumes it.
    Gated by DLRM_HSTU_FP8_WLN_BWD_V2 (gin default on, yambda_5b.gin item (14); absent or "0" = v1).
    """
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    num_blocks = tl.cdiv(N, BLOCK_N)
    blocks_per_tile = num_blocks // tile_num
    extra = num_blocks % tile_num
    if CONTIG:
        start_block = pid * blocks_per_tile + tl.minimum(pid, extra)
        step = 1
    else:
        start_block = pid
        step = tile_num
    if pid < extra:
        blocks_per_tile += 1

    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
    if IS_SWISH:
        b = tl.load(B + cols, mask=col_mask, other=0.0).to(tl.float32)

    acc_dw = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc_db = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc_dxsum = tl.zeros([BLOCK_D], dtype=tl.float32)

    for idx in tl.range(0, blocks_per_tile, num_stages=NUM_STAGES):
        current_block = start_block + idx * step
        start_row = current_block * BLOCK_N
        rows = start_row + tl.arange(0, BLOCK_N)
        row_mask = rows < N
        rows64 = rows.to(tl.int64)
        if EVEN_D:
            m2 = row_mask[:, None] & (cols < BLOCK_D)[None, :]
        else:
            m2 = row_mask[:, None] & col_mask[None, :]
        x_block = tl.load(
            X + rows64[:, None] * stride_x + cols[None, :], mask=m2, other=0.0
        ).to(tl.float32)
        dy_block = tl.load(
            DY + rows64[:, None] * stride_dy + cols[None, :], mask=m2, other=0.0
        ).to(tl.float32)
        mean = tl.load(Mean + rows, row_mask, other=0.0)
        rstd = tl.load(Rstd + rows, row_mask, other=0.0)
        mean = tl.expand_dims(mean, 1)
        rstd = tl.expand_dims(rstd, 1)

        # Masked rows/cols load x=dy=mean=rstd=0, so every term below is
        # already zero there; the tl.where's are kept on the !EVEN_D path to
        # mirror v1 exactly where padding columns exist.
        xhat = (x_block - mean) * rstd
        xhat = xhat if EVEN_D else tl.where(m2, xhat, 0.0)
        wdy = w[None, :] * dy_block
        wdy = wdy if EVEN_D else tl.where(m2, wdy, 0.0)

        if IS_SWISH:
            if FAST_SIGMOID:
                sigmoid_layer_norm = fast_dividef(
                    1.0, 1.0 + tl.exp(-(xhat * w[None, :] + b[None, :]))
                )
            else:
                sigmoid_layer_norm = tl.sigmoid(xhat * w[None, :] + b[None, :])
            sigmoid_layer_norm = (
                sigmoid_layer_norm
                if EVEN_D
                else tl.where(m2, sigmoid_layer_norm, 0.0)
            )
            sigmoid_deriv = sigmoid_layer_norm * (1 - sigmoid_layer_norm)
            x_ = wdy * x_block * sigmoid_deriv
            x_ = x_ if EVEN_D else tl.where(m2, x_, 0.0)
            c1 = tl.sum(xhat * x_, axis=1) / D
            c2 = tl.sum(x_, axis=1) / D
            c1 = tl.expand_dims(c1, 1)
            c2 = tl.expand_dims(c2, 1)
            dx = (x_ - (xhat * c1 + c2)) * rstd
            dx = dy_block * sigmoid_layer_norm + dx
            partial_dw = tl.sum(dy_block * x_block * xhat * sigmoid_deriv, axis=0)
            partial_db = tl.sum(dy_block * x_block * sigmoid_deriv, axis=0)
        else:
            c1 = tl.sum(xhat * wdy, axis=1) / D
            c2 = tl.sum(wdy, axis=1) / D
            c1 = tl.expand_dims(c1, 1)
            c2 = tl.expand_dims(c2, 1)
            dx = (wdy - (xhat * c1 + c2)) * rstd
            partial_dw = tl.sum(dy_block * xhat, axis=0)
            partial_db = tl.sum(dy_block, axis=0)
        if ADD_RESIDUAL:
            dx = dx.to(DX.dtype.element_ty).to(tl.float32) + tl.load(
                RESIDUAL + rows64[:, None] * stride_residual + cols[None, :],
                mask=m2,
                other=0.0,
            ).to(tl.float32)
        tl.store(
            DX + rows64[:, None] * stride_dx + cols[None, :],
            dx.to(DX.dtype.element_ty),
            mask=m2,
        )
        acc_dw += partial_dw
        acc_db += partial_db
        if SUM_DX:
            acc_dxsum += tl.sum(dx, axis=0)

    tl.store(DW + pid.to(tl.int64) * D + cols, acc_dw, mask=col_mask)
    tl.store(DB + pid.to(tl.int64) * D + cols, acc_db, mask=col_mask)
    if SUM_DX:
        tl.store(DXSUM + pid.to(tl.int64) * D + cols, acc_dxsum, mask=col_mask)


def _get_bwd_dwdb_configs() -> List[triton.Config]:
    configs = []
    BLOCK_N_CHOICES = [32, 64, 128, 256]
    if is_sm100_plus():
        BLOCK_N_CHOICES = [128, 256, 512, 1024]
    for BLOCK_N in BLOCK_N_CHOICES:
        for num_warps in [8, 16] + ([] if torch.ops.hip else [32]):
            configs.append(
                triton.Config(
                    {"BLOCK_N": BLOCK_N},
                    num_warps=num_warps,
                )
            )
    return configs


@triton_autotune(
    configs=pinned_or_full(
        [triton.Config({"BLOCK_N": 128}, num_warps=8)],
        _get_bwd_dwdb_configs,
    ),
    key=["D"],
)
@triton.jit
def _layer_norm_bwd_dwdb(
    DW,
    DB,
    FINAL_DW,
    FINAL_DB,
    N,
    D,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    dw = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    db = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for i in range(0, N, BLOCK_N):
        rows = i + tl.arange(0, BLOCK_N)
        # pyre-fixme[16]: `int` has no attribute `__getitem__`.
        mask = (rows[:, None] < N) & (cols[None, :] < D)
        offs = rows[:, None] * D + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
        db += tl.load(DB + offs, mask=mask, other=0.0)

    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw.to(FINAL_DW.dtype.element_ty), mask=cols < D)
    tl.store(FINAL_DB + cols, sum_db.to(FINAL_DB.dtype.element_ty), mask=cols < D)


@triton_autotune(
    configs=pinned_or_full(
        [triton.Config({"BLOCK_N": 128}, num_warps=8)],
        _get_bwd_dwdb_configs,
    ),
    key=["D"],
)
@triton.jit
def _layer_norm_bwd_dwdbdx(
    DW,
    DB,
    DXSUM,
    FINAL_DW,
    FINAL_DB,
    FINAL_DXSUM,
    N,
    D,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """F1 finalize: _layer_norm_bwd_dwdb plus the dx column-sum.

    A separate kernel from _layer_norm_bwd_dwdb, so the path without the column
    sum is unchanged.

    The reduction is the same [tile_num, D] -> [D] shape the dw/db partials
    already use, so DXSUM costs one more strided read of an array the bwd_dx
    kernel has just written, not a new pass over the [N, D] activations.
    """
    pid = tl.program_id(0)
    cols = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    dw = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    db = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dxsum = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for i in range(0, N, BLOCK_N):
        rows = i + tl.arange(0, BLOCK_N)
        # pyre-fixme[16]: `int` has no attribute `__getitem__`.
        mask = (rows[:, None] < N) & (cols[None, :] < D)
        offs = rows[:, None] * D + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
        db += tl.load(DB + offs, mask=mask, other=0.0)
        dxsum += tl.load(DXSUM + offs, mask=mask, other=0.0)

    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    sum_dxsum = tl.sum(dxsum, axis=0)
    tl.store(FINAL_DW + cols, sum_dw.to(FINAL_DW.dtype.element_ty), mask=cols < D)
    tl.store(FINAL_DB + cols, sum_db.to(FINAL_DB.dtype.element_ty), mask=cols < D)
    tl.store(
        FINAL_DXSUM + cols,
        sum_dxsum.to(FINAL_DXSUM.dtype.element_ty),
        mask=cols < D,
    )


def compute_BLOCK_D(x: torch.Tensor) -> int:
    """Compute the BLOCK_D parameter for layer norm kernels."""
    D = x.shape[-1]
    MAX_FUSED_SIZE = 65536 // x.element_size()
    return min(MAX_FUSED_SIZE, triton.next_power_of_2(D))


@maybe_register_custom_op(
    "generative_recommenders::triton_weighted_layer_norm_fwd", mutates_args=()
)
def triton_weighted_layer_norm_fwd(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
    mean: Optional[torch.Tensor] = None,
    rstd: Optional[torch.Tensor] = None,
    output_fp8: bool = False,
    inv_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.dim() == 2, f"x.dim() == {x.dim()}, expected 2"
    x = switch_to_contiguous_if_needed(x)
    N, D = x.shape
    learnable = weight is not None
    if learnable:
        assert bias is not None and weight is not None
        assert weight.dim() == 1
        assert bias.dim() == 1
        assert weight.numel() == D
        assert bias.numel() == D

    if output_fp8:
        assert learnable, "FP8 LayerNorm output requires learnable affine parameters"
        y = torch.empty(
            (N, D),
            dtype=torch.float8_e4m3fn,
            device=x.device,
        )
    else:
        y = torch.empty_like(x)
    compute_mean_and_rstd = mean is None or rstd is None
    # Always allocate new tensors to avoid aliasing inputs with outputs
    out_mean = torch.empty((N,), dtype=torch.float32, device=x.device)
    out_rstd = torch.empty((N,), dtype=torch.float32, device=x.device)
    if not compute_mean_and_rstd:
        assert mean is not None and rstd is not None
        out_mean.copy_(mean)
        out_rstd.copy_(rstd)

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_D: int = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BLOCK_D:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    if N == 0:
        return y, out_mean, out_rstd

    # pyre-ignore[28]
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)  # noqa E731
    if learnable:
        _weighted_layer_norm_fwd[grid](
            x,
            y,
            y,
            weight,
            bias,
            out_mean,
            out_rstd,
            N,
            D,
            eps,
            x.stride(0),
            y.stride(0),
            y.stride(0),
            float(inv_scale),
            None,  # RESIDUAL
            0,  # stride_residual
            IS_SWISH=False,
            TRAINING=True,
            BLOCK_D=BLOCK_D,
            COMPUTE_MEAN_AND_RSTD=compute_mean_and_rstd,
            OUTPUT_FP8=output_fp8,
            OUTPUT_BOTH=False,
            ADD_RESIDUAL=False,
        )
    else:
        _layer_norm_fwd[grid](
            x,
            y,
            out_mean,
            out_rstd,
            N,
            D,
            eps,
            x.stride(0),
            y.stride(0),
            TRAINING=True,
            BLOCK_D=BLOCK_D,
            COMPUTE_MEAN_AND_RSTD=compute_mean_and_rstd,
        )

    return y, out_mean, out_rstd


def triton_weighted_layer_norm_fp8_save_bf16_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    inv_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward LayerNorm emitting scaled E4M3, BF16, mean and rstd in one pass.

    Store-instead-of-recompute (DLRM_HSTU_FP8_SAVE_NORMED_X): the FP8 forward
    used to write only the E4M3 copy, so backward re-ran this kernel with the
    saved mean/rstd (COMPUTE_MEAN_AND_RSTD=False) to rebuild the BF16 copy for
    the projection wgrad. Here the forward also writes the BF16 copy. Both
    outputs derive from the same fp32 ``y = (x - mean) * rstd * w + b`` using
    the very fp32 mean/rstd that are stored and that the backward recompute
    would load, so the BF16 tensor equals that recompute bit for bit and the
    E4M3 tensor equals the OUTPUT_FP8 store (``y * inv_scale`` then cast).
    Returns (y_bf16, y_fp8, mean, rstd).
    """
    assert x.dim() == 2
    x = switch_to_contiguous_if_needed(x)
    N, D = x.shape
    assert weight.dim() == 1 and weight.numel() == D
    assert bias.dim() == 1 and bias.numel() == D
    y = torch.empty_like(x)
    y_fp8 = torch.empty((N, D), dtype=torch.float8_e4m3fn, device=x.device)
    out_mean = torch.empty((N,), dtype=torch.float32, device=x.device)
    out_rstd = torch.empty((N,), dtype=torch.float32, device=x.device)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_D: int = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BLOCK_D:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    if N == 0:
        return y, y_fp8, out_mean, out_rstd
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)  # noqa E731
    _weighted_layer_norm_fwd[grid](
        x,
        y,
        y_fp8,
        weight,
        bias,
        out_mean,
        out_rstd,
        N,
        D,
        eps,
        x.stride(0),
        y.stride(0),
        y_fp8.stride(0),
        float(inv_scale),
        None,  # RESIDUAL
        0,  # stride_residual
        IS_SWISH=False,
        TRAINING=True,
        BLOCK_D=BLOCK_D,
        COMPUTE_MEAN_AND_RSTD=True,
        OUTPUT_FP8=False,
        OUTPUT_BOTH=True,
        ADD_RESIDUAL=False,
    )
    return y, y_fp8, out_mean, out_rstd


def triton_weighted_layer_norm_dual_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    inv_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Recompute affine LayerNorm once and store BF16 plus scaled E4M3."""
    assert x.dim() == 2
    x = switch_to_contiguous_if_needed(x)
    N, D = x.shape
    assert weight.dim() == 1 and weight.numel() == D
    assert bias.dim() == 1 and bias.numel() == D
    y = torch.empty_like(x)
    y_fp8 = torch.empty((N, D), dtype=torch.float8_e4m3fn, device=x.device)
    if N == 0:
        return y, y_fp8

    BLOCK_D = compute_BLOCK_D(x)
    if D > BLOCK_D:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)  # noqa E731
    _weighted_layer_norm_fwd[grid](
        x,
        y,
        y_fp8,
        weight,
        bias,
        mean,
        rstd,
        N,
        D,
        eps,
        x.stride(0),
        y.stride(0),
        y_fp8.stride(0),
        float(inv_scale),
        None,  # RESIDUAL
        0,  # stride_residual
        IS_SWISH=False,
        TRAINING=False,
        BLOCK_D=BLOCK_D,
        COMPUTE_MEAN_AND_RSTD=False,
        OUTPUT_FP8=False,
        OUTPUT_BOTH=True,
        ADD_RESIDUAL=False,
    )
    return y, y_fp8


@triton_weighted_layer_norm_fwd.register_fake
def _(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
    mean: Optional[torch.Tensor] = None,
    rstd: Optional[torch.Tensor] = None,
    output_fp8: bool = False,
    inv_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N = x.shape[0]
    y = torch.empty(
        x.shape,
        dtype=torch.float8_e4m3fn if output_fp8 else x.dtype,
        device=x.device,
    )
    # Always allocate new tensors to avoid aliasing inputs with outputs
    out_mean = torch.empty((N,), dtype=torch.float32, device=x.device)
    out_rstd = torch.empty((N,), dtype=torch.float32, device=x.device)
    return y, out_mean, out_rstd


_LN_BWD_SWISH_SMALL_D_TILE_MULT = 32


def _ln_bwd_tile_num(N: int, D: int, sms: int, is_swish: bool) -> int:
    """Persistent-grid size for _weighted_layer_norm_bwd_dx.

    Swish with D<=256 uses a larger persistent grid. Other cases use sms*8.
    Rows are independent, so dx for a fixed BLOCK_N does not depend on the
    grid. dw/db/dxsum sum fp32 partials, so their reduction order follows
    tile_num.
    """
    small_swish = is_swish and triton.next_power_of_2(D) <= 256
    mult = _LN_BWD_SWISH_SMALL_D_TILE_MULT if small_swish else 8
    return max(1, min(sms * mult, N // 4))


# DLRM_HSTU_FP8_WLN_BWD_V2 (gin default on, item (14); absent = v1): route every launch of the weighted
# LayerNorm backward dx kernel through _weighted_layer_norm_bwd_dx_v2. "0"
# restores the v1 kernel, its grid and its autotune exactly. Read once, cached.
_WLN_BWD_V2_ENABLED: Optional[bool] = None


def _wln_bwd_v2_enabled() -> bool:
    global _WLN_BWD_V2_ENABLED
    if _WLN_BWD_V2_ENABLED is None:
        _WLN_BWD_V2_ENABLED = os.environ.get("DLRM_HSTU_FP8_WLN_BWD_V2", "0") != "0"
    return _WLN_BWD_V2_ENABLED


# Sub-gate (only consulted when v2 is on): DLRM_HSTU_FP8_WLN_BWD_V2_FAST_SIGMOID
# (gin default on, item (14); absent = tl.sigmoid) computes the swish sigmoid as fast_dividef(1, 1 + exp(-z)).
# At the production swish shape (N=2.36M, D=256) dx error vs an fp64
# reference is unchanged (max abs 1.5625e-2, rel L2 1.65968667e-3 for both);
# dx differs from tl.sigmoid in ~2.5e-5 of elements by one bf16 ulp.
_WLN_BWD_V2_FAST_SIGMOID: Optional[bool] = None


def _wln_bwd_v2_fast_sigmoid() -> bool:
    global _WLN_BWD_V2_FAST_SIGMOID
    if _WLN_BWD_V2_FAST_SIGMOID is None:
        _WLN_BWD_V2_FAST_SIGMOID = (
            os.environ.get("DLRM_HSTU_FP8_WLN_BWD_V2_FAST_SIGMOID", "0") != "0"
        )
    return _WLN_BWD_V2_FAST_SIGMOID


def _wln_bwd_v2_config(
    BLOCK_D: int, is_swish: bool, add_residual: bool, sum_dx: bool
) -> Tuple[int, int, int, bool, int, bool]:
    """(tile multiplier, BLOCK_N, NUM_STAGES, CONTIG, num_warps, FAST_SIGMOID)
    for v2; the grid is min(sms * multiplier, cdiv(N, 4 * BLOCK_N)).

      D=512 residual (no swish):  sms*8,  BLOCK_N=1, 2 stages
      D=512 SUM_DX (no swish):    sms*4,  BLOCK_N=4, 2 stages
      D=256 swish SUM_DX:         sms*32, BLOCK_N=1, 1 stage
    The non-swish non-SUM_DX config is shared by the residual and plain
    calls so a fused residual does not change dw/db. Rows wider than 512 keep
    BLOCK_N=1 (register pressure) and add warps only past 2048 columns.
    """
    if BLOCK_D > 512:
        return (8, 1, 1, False, max(1, BLOCK_D // 2048), False)
    if is_swish:
        return (32, 1, 1, False, 1, _wln_bwd_v2_fast_sigmoid())
    if sum_dx:
        return (4, 4, 2, False, 1, False)
    return (8, 1, 2, False, 1, False)


def _wln_bwd_dx_partials(
    dx: torch.Tensor,
    dy: torch.Tensor,
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    eps: float,
    is_swish: bool,
    add_residual: bool,
    sum_dx: bool,
    BLOCK_D: int,
    sms: int,
) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """v2 launch: writes dx, returns (tile_num, dw, db, dxsum) [tile_num, D]
    fp32 partials for the unchanged finalize kernels. When not sum_dx, the
    dxsum slot aliases the db partial (as the v1 call sites do). N > 0."""
    N, D = x.shape
    mult, block_n, num_stages, contig, num_warps, fast_sigmoid = _wln_bwd_v2_config(
        BLOCK_D, is_swish, add_residual, sum_dx
    )
    # >= ~4 row blocks per program, as v1's N // 4 cap does for BLOCK_N=1.
    tile_num = max(1, min(sms * mult, triton.cdiv(N, 4 * block_n)))
    _dweight = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
    _dbias = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
    if sum_dx:
        _dxsum = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
    else:
        _dxsum = _dbias
    _weighted_layer_norm_bwd_dx_v2[(tile_num,)](
        dx,
        dy,
        residual,
        _dweight,
        _dbias,
        _dxsum,
        x,
        weight,
        bias,
        mean,
        rstd,
        dx.stride(0),
        dy.stride(0),
        residual.stride(0) if add_residual else 0,
        x.stride(0),
        D,
        eps,
        IS_SWISH=is_swish,
        ADD_RESIDUAL=add_residual,
        SUM_DX=sum_dx,
        N=N,
        BLOCK_D=BLOCK_D,
        BLOCK_N=block_n,
        NUM_STAGES=num_stages,
        CONTIG=contig,
        EVEN_D=BLOCK_D == D,
        FAST_SIGMOID=fast_sigmoid,
        num_warps=num_warps,
    )
    return tile_num, _dweight, _dbias, _dxsum


@maybe_register_custom_op(
    "generative_recommenders::triton_weighted_layer_norm_bwd", mutates_args=()
)
def _triton_weighted_layer_norm_bwd_impl(
    dy: torch.Tensor,
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    learnable: bool,
    add_residual: bool,
    eps: float,
    BLOCK_D: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_warps: int = min(max(BLOCK_D // 256, 1), 8)
    if learnable:
        N, D = x.shape
        dx = torch.empty_like(x)
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        use_v2 = _wln_bwd_v2_enabled()
        if not use_v2:
            tile_num = _ln_bwd_tile_num(N, D, sms, is_swish=False)
            _dweight = torch.empty(
                (tile_num, D), dtype=torch.float32, device=x.device
            )
            _dbias = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
        # F1 (bias-grad fold) is not plumbed through this entry point yet; the
        # kernel gate stays off so the compiled control path is unchanged.
        sum_dx = False
        dweight = torch.empty((D,), dtype=weight.dtype, device=x.device)
        dbias = torch.empty((D,), dtype=weight.dtype, device=x.device)
        if N == 0:
            dweight.zero_()
            dbias.zero_()
            return dx, dweight, dbias
        if use_v2:
            tile_num, _dweight, _dbias, _ = _wln_bwd_dx_partials(
                dx,
                dy,
                residual,
                x,
                weight,
                bias,
                mean,
                rstd,
                eps,
                is_swish=False,
                add_residual=add_residual,
                sum_dx=sum_dx,
                BLOCK_D=BLOCK_D,
                sms=sms,
            )
        else:
            _dxsum = _dbias
            # pyre-ignore[28]
            _weighted_layer_norm_bwd_dx[(tile_num,)](
                dx,
                dy,
                residual,
                _dweight,
                _dbias,
                _dxsum,
                x,
                weight,
                bias,
                mean,
                rstd,
                dx.stride(0),
                dy.stride(0),
                residual.stride(0) if add_residual else 0,
                x.stride(0),
                D,
                eps,
                IS_SWISH=False,
                ADD_RESIDUAL=add_residual,
                SUM_DX=sum_dx,
                N=N,
                BLOCK_D=BLOCK_D,
            )

        def grid(META):
            return (triton.cdiv(D, META["BLOCK_D"]),)

        blocks = triton.next_power_of_2(sms * 4)
        BLOCK_D = triton.next_power_of_2(triton.cdiv(D, blocks))
        BLOCK_D = min(max(BLOCK_D, 4), 128)
        _layer_norm_bwd_dwdb[grid](
            _dweight,
            _dbias,
            dweight,
            dbias,
            tile_num,
            D,
            BLOCK_D=BLOCK_D,
        )

        return dx, dweight, dbias
    else:
        N, D = x.shape
        dx = torch.empty_like(x)
        # Return empty tensors as sentinels for None
        dweight = torch.empty(0, dtype=x.dtype, device=x.device)
        dbias = torch.empty(0, dtype=x.dtype, device=x.device)
        if N == 0:
            return dx, dweight, dbias
        # pyre-ignore[28]
        _layer_norm_bwd_dx[(N,)](
            dx,
            dy,
            x,
            mean,
            rstd,
            dx.stride(0),
            dy.stride(0),
            x.stride(0),
            D,
            eps,
            BLOCK_D=BLOCK_D,
            num_warps=num_warps,
        )
        return dx, dweight, dbias


@_triton_weighted_layer_norm_bwd_impl.register_fake
def _(
    dy: torch.Tensor,
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    learnable: bool,
    add_residual: bool,
    eps: float,
    BLOCK_D: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dx = torch.empty_like(x)
    if learnable:
        D = x.shape[-1]
        dweight = torch.empty((D,), dtype=weight.dtype, device=x.device)
        dbias = torch.empty((D,), dtype=weight.dtype, device=x.device)
    else:
        dweight = torch.empty(0, dtype=x.dtype, device=x.device)
        dbias = torch.empty(0, dtype=x.dtype, device=x.device)
    return dx, dweight, dbias


def triton_weighted_layer_norm_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    mean: torch.Tensor,
    rstd: torch.Tensor,
    learnable: bool,
    eps: float,
    BLOCK_D: int,
    residual: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    # Use sentinel tensors for custom_op compatibility (can't return Optional[Tensor])
    _weight = (
        weight if weight is not None else torch.empty(0, dtype=x.dtype, device=x.device)
    )
    _bias = bias if bias is not None else torch.empty(0, dtype=x.dtype, device=x.device)
    _residual = (
        residual
        if residual is not None
        else torch.empty(0, dtype=x.dtype, device=x.device)
    )
    dx, dweight, dbias = _triton_weighted_layer_norm_bwd_impl(
        dy=dy,
        residual=_residual,
        x=x,
        weight=_weight,
        bias=_bias,
        mean=mean,
        rstd=rstd,
        learnable=learnable,
        add_residual=residual is not None,
        eps=eps,
        BLOCK_D=BLOCK_D,
    )
    if not learnable:
        return dx, None, None
    return dx, dweight, dbias


def triton_weighted_layer_norm_bwd_dxsum(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    eps: float,
    BLOCK_D: int,
    is_swish: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """F1: the norm backward, additionally returning sum(dx, axis=0).

    A SEPARATE entry point from triton_weighted_layer_norm_bwd, deliberately.
    That function returns a 3-tuple and is unpacked at six call sites
    (layer_norm_test.py:56,67; triton_hstu_preprocess_and_attention.py:855,1384;
    and twice in this file). Widening its arity would touch all six to buy
    nothing for the five that do not want the extra value, so the fused path
    gets its own 4-tuple and the shipped path is left exactly as it was.

    The Linear's bias gradient is the column sum of dx. This computes that sum
    from the fp32 dx still in registers. aten::sum reduces the stored bf16 dx,
    so the two are not bit-identical.
    """
    N, D = x.shape
    dx = torch.empty_like(x)
    dweight = torch.empty((D,), dtype=weight.dtype, device=x.device)
    dbias = torch.empty((D,), dtype=weight.dtype, device=x.device)
    # dxsum is the preceding Linear's bias gradient, so it is fp32 to match what
    # autograd would otherwise hand that parameter from aten::sum.
    dxsum = torch.empty((D,), dtype=torch.float32, device=x.device)
    if N == 0:
        dweight.zero_()
        dbias.zero_()
        dxsum.zero_()
        return dx, dweight, dbias, dxsum

    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    if _wln_bwd_v2_enabled():
        tile_num, _dweight, _dbias, _dxsum = _wln_bwd_dx_partials(
            dx,
            dy,
            dx,
            x,
            weight,
            bias,
            mean,
            rstd,
            eps,
            is_swish=is_swish,
            add_residual=False,
            sum_dx=True,
            BLOCK_D=BLOCK_D,
            sms=sms,
        )
    else:
        tile_num = _ln_bwd_tile_num(N, D, sms, is_swish)
        _dweight = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
        _dbias = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
        _dxsum = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)

        # pyre-ignore[28]
        _weighted_layer_norm_bwd_dx[(tile_num,)](
            dx,
            dy,
            dx,
            _dweight,
            _dbias,
            _dxsum,
            x,
            weight,
            bias,
            mean,
            rstd,
            dx.stride(0),
            dy.stride(0),
            0,
            x.stride(0),
            D,
            eps,
            IS_SWISH=is_swish,
            ADD_RESIDUAL=False,
            SUM_DX=True,
            N=N,
            BLOCK_D=BLOCK_D,
        )

    def grid(META):
        return (triton.cdiv(D, META["BLOCK_D"]),)

    blocks = triton.next_power_of_2(sms * 4)
    FINAL_BLOCK_D = triton.next_power_of_2(triton.cdiv(D, blocks))
    FINAL_BLOCK_D = min(max(FINAL_BLOCK_D, 4), 128)
    _layer_norm_bwd_dwdbdx[grid](
        _dweight,
        _dbias,
        _dxsum,
        dweight,
        dbias,
        dxsum,
        tile_num,
        D,
        BLOCK_D=FINAL_BLOCK_D,
    )
    return dx, dweight, dbias, dxsum


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        eps: float,
    ) -> torch.Tensor:
        y, mean, rstd = triton_weighted_layer_norm_fwd(
            x=x,
            weight=weight,
            bias=bias,
            eps=eps,
        )
        BLOCK_D = compute_BLOCK_D(x)
        learnable = weight is not None
        if learnable:
            ctx.save_for_backward(x, weight, bias, mean, rstd)
        else:
            ctx.save_for_backward(x, mean, rstd)
        ctx.BLOCK_D = BLOCK_D
        ctx.eps = eps
        ctx.learnable = learnable
        return y

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dy: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], None]:
        if ctx.learnable:
            x, weight, bias, mean, rstd = ctx.saved_tensors
        else:
            x, mean, rstd = ctx.saved_tensors
            weight, bias = None, None
        dx, dweight, dbias = triton_weighted_layer_norm_bwd(
            dy=dy,
            x=x,
            weight=weight,
            bias=bias,
            mean=mean,
            rstd=rstd,
            learnable=ctx.learnable,
            eps=ctx.eps,
            BLOCK_D=ctx.BLOCK_D,
        )
        return dx, dweight, dbias, None


def _get_rms_norm_fwd_configs() -> List[triton.Config]:
    """Generate autotune configs for multi-row RMSNorm kernels."""
    configs = []
    for BLOCK_N in [1, 4, 16]:
        for num_warps in [2, 4]:
            configs.append(
                triton.Config(
                    {"BLOCK_N": BLOCK_N},
                    num_warps=num_warps,
                )
            )
    return configs


@triton.autotune(
    configs=_get_rms_norm_fwd_configs(),
    key=["BLOCK_D", "SILU"],
)
@triton.jit
def _weighted_rms_norm_fwd(
    X,
    Y,
    W,
    Rstd,
    N,
    D: tl.constexpr,
    eps,
    stride_x,
    stride_y,
    SILU: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block_id = tl.program_id(0)
    start_row = block_id * BLOCK_N

    # Load weight once (shared across all rows in this block)
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)

    # Create block pointers for X and Y
    X_block_ptr = tl.make_block_ptr(
        base=X,
        shape=(N, D),
        strides=(stride_x, 1),
        offsets=(start_row, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    Y_block_ptr = tl.make_block_ptr(
        base=Y,
        shape=(N, D),
        strides=(stride_y, 1),
        offsets=(start_row, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    x_block = tl.load(X_block_ptr, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )

    rows = start_row + tl.arange(0, BLOCK_N)
    row_mask = rows < N

    # Compute variance (RMS norm uses x directly, not x - mean)
    x_masked = tl.where(row_mask[:, None] & col_mask[None, :], x_block, 0.0)
    _var = x_masked * x_masked
    var = tl.sum(_var, axis=1) / D
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + rows, rstd, row_mask)

    # Normalize and apply linear transformation
    rstd = tl.expand_dims(rstd, 1)
    y = x_block * rstd
    y = y * w[None, :]

    if SILU:
        # pyre-ignore[16]: Module `triton.language.math` has no attribute `fast_dividef`
        y = fast_dividef(y, 1.0 + tl.exp(-y))

    tl.store(Y_block_ptr, y.to(Y.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _weighted_rms_norm_bwd_dx(
    DX,
    DY,
    DW,
    X,
    W,
    Rstd,
    Lock,
    stride_dx,
    stride_dy,
    stride_x,
    D: tl.constexpr,
    eps,
    GROUP_N,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    X += row.to(tl.int64) * stride_x
    DY += row.to(tl.int64) * stride_dy
    DX += row.to(tl.int64) * stride_dx

    # Load data to SRAM
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    rstd = tl.load(Rstd + row)

    # Compute dx
    xhat = x * rstd
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    wdy = w * dy

    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=0) / D
    dx = (wdy - (xhat * c1)) * rstd
    # Write dx
    tl.store(DX + cols, dx, mask=mask)

    # Offset locks and weights/biases gradient pointer for parallel reduction
    lock_id = row % GROUP_N
    Lock += lock_id
    Count = Lock + GROUP_N
    DW = DW + lock_id * D + cols
    # Accumulate partial sums for dw/db
    partial_dw = dy * xhat
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    # First store doesn't accumulate
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    # Release the lock
    tl.atomic_xchg(Lock, 0)


@triton_autotune(
    configs=_get_norm_bwd_configs(),
    key=["BLOCK_D", "SILU"],
    reset_to_zero=["DW"],
)
@triton.jit
def _weighted_rms_norm_bwd(
    DX,
    DY,
    DW,
    X,
    W,
    Rstd,
    stride_dx,
    stride_dy,
    stride_x,
    D,
    eps,
    N,
    SILU: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SHARDS_PER_SM: tl.constexpr,
):
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    num_blocks = tl.cdiv(N, BLOCK_N)
    blocks_per_tile = num_blocks // tile_num
    if pid < num_blocks % tile_num:
        blocks_per_tile += 1

    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)

    start_block = pid

    acc_dw = tl.zeros([BLOCK_D], dtype=tl.float32)

    for idx in range(blocks_per_tile):
        current_block = start_block + idx * tile_num
        start_row = current_block * BLOCK_N

        X_block_ptr = tl.make_block_ptr(
            base=X,
            shape=(N, D),
            strides=(stride_x, 1),
            offsets=(start_row, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )

        DX_block_ptr = tl.make_block_ptr(
            base=DX,
            shape=(N, D),
            strides=(stride_dx, 1),
            offsets=(start_row, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )

        DY_block_ptr = tl.make_block_ptr(
            base=DY,
            shape=(N, D),
            strides=(stride_dy, 1),
            offsets=(start_row, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )

        # Load data blocks
        x_block = tl.load(X_block_ptr, boundary_check=(0, 1), padding_option="zero").to(
            tl.float32
        )
        dy_block = tl.load(
            DY_block_ptr, boundary_check=(0, 1), padding_option="zero"
        ).to(tl.float32)

        # Load rstd for all rows in this block
        rows = start_row + tl.arange(0, BLOCK_N)
        row_mask = rows < N
        rstd = tl.load(Rstd + rows, row_mask, other=0.0)

        # Expand dimensions for broadcasting
        rstd = tl.expand_dims(rstd, 1)

        # Compute dx
        xhat = x_block * rstd

        # Apply SILU backward if enabled
        if SILU:
            y_before_silu = xhat * w[None, :]
            # pyre-fixme[16]
            sig_y = fast_dividef(1.0, 1.0 + tl.exp(-y_before_silu))
            # SILU derivative: sigmoid(y) + y * sigmoid(y) * (1 - sigmoid(y))
            dy_block = dy_block * (sig_y + y_before_silu * sig_y * (1.0 - sig_y))

        wdy = w[None, :] * dy_block

        c1 = tl.sum(xhat * wdy, axis=1) / D
        c1 = tl.expand_dims(c1, 1)
        dx = (wdy - (xhat * c1)) * rstd

        # Write dx
        tl.store(DX_block_ptr, dx.to(DX.dtype.element_ty), boundary_check=(0, 1))

        # Accumulate partial sums for dw
        # Compute dw for all rows, then sum locally before atomic operation
        partial_dw_block = dy_block * xhat
        # Local reduction: sum across all rows in this block
        partial_dw = tl.sum(partial_dw_block, axis=0)
        acc_dw += partial_dw

    DW_ptr = DW + cols
    tl.atomic_add(DW_ptr, acc_dw, col_mask)


@triton_autotune(
    configs=_get_bwd_dwdb_configs(),
    key=["D"],
)
@triton.jit
def _rms_norm_bwd_dwdb(
    DW,
    FINAL_DW,
    N,
    D,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    dw = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for i in range(0, N, BLOCK_N):
        rows = i + tl.arange(0, BLOCK_N)
        # pyre-fixme[16]: `int` has no attribute `__getitem__`.
        mask = (rows[:, None] < N) & (cols[None, :] < D)
        offs = rows[:, None] * D + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)

    sum_dw = tl.sum(dw, axis=0)
    tl.store(FINAL_DW + cols, sum_dw.to(FINAL_DW.dtype.element_ty), mask=cols < D)


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        silu: bool,
    ) -> torch.Tensor:
        assert x.dim() == 2
        x = switch_to_contiguous_if_needed(x)
        N, D = x.shape
        assert weight.dim() == 1
        assert weight.numel() == D

        y = torch.empty_like(x)
        rstd = torch.empty((N,), dtype=torch.float32, device=x.device)

        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_D = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
        if D > BLOCK_D:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

        ctx.save_for_backward(x, weight, rstd)
        ctx.silu = silu
        if N == 0:
            return y

        # pyre-ignore[28]
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)  # noqa E731
        _weighted_rms_norm_fwd[grid](
            x,
            y,
            weight,
            rstd,
            N,
            D,
            eps,
            x.stride(0),
            y.stride(0),
            SILU=silu,
            BLOCK_D=BLOCK_D,
        )

        ctx.BLOCK_D = BLOCK_D
        ctx.eps = eps
        return y

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dy: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None, None]:
        x, weight, rstd = ctx.saved_tensors
        N, D = x.shape
        dx = torch.empty_like(x)
        dweight = torch.zeros((D,), dtype=weight.dtype, device=x.device)
        if N == 0:
            dweight.zero_()
            return dx, dweight, None, None

        sms = torch.cuda.get_device_properties(x.device).multi_processor_count

        # pyre-ignore[28]
        grid = lambda meta: (  # noqa E731
            max(1, min(sms * meta["SHARDS_PER_SM"], N // 4)),
        )
        _weighted_rms_norm_bwd[grid](
            dx,
            dy,
            dweight,
            x,
            weight,
            rstd,
            dx.stride(0),
            dy.stride(0),
            x.stride(0),
            D,
            ctx.eps,
            N=N,
            SILU=ctx.silu,
            BLOCK_D=ctx.BLOCK_D,
        )

        return dx, dweight, None, None


class SwishLayerNormFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        assert x.dim() == 2, f"x.dim() == {x.dim()}, expected 2"
        x = switch_to_contiguous_if_needed(x)
        N, D = x.shape

        assert bias is not None and weight is not None
        assert weight.dim() == 1
        assert bias.dim() == 1
        assert weight.numel() == D
        assert bias.numel() == D

        y = torch.empty_like(x)
        mean = torch.empty((N,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((N,), dtype=torch.float32, device=x.device)

        BLOCK_D = triton.next_power_of_2(D)
        num_warps = min(max(BLOCK_D // 256, 1), 8)

        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_D = BLOCK_D
        ctx.num_warps = num_warps
        ctx.eps = eps
        if N == 0:
            return y

        # pyre-ignore[28]
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)  # noqa E731
        _weighted_layer_norm_fwd[grid](
            x,
            y,
            y,
            weight,
            bias,
            mean,
            rstd,
            N,
            D,
            eps,
            x.stride(0),
            y.stride(0),
            y.stride(0),
            1.0,
            None,  # RESIDUAL
            0,  # stride_residual
            IS_SWISH=True,
            TRAINING=True,
            BLOCK_D=BLOCK_D,
            COMPUTE_MEAN_AND_RSTD=True,
            OUTPUT_FP8=False,
            OUTPUT_BOTH=False,
            ADD_RESIDUAL=False,
        )

        return y

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dy: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], None]:
        x, weight, bias, mean, rstd = ctx.saved_tensors
        N, D = x.shape
        dx = torch.empty_like(x)
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        use_v2 = _wln_bwd_v2_enabled()
        if not use_v2:
            tile_num = _ln_bwd_tile_num(N, D, sms, is_swish=True)
            _dweight = torch.empty(
                (tile_num, D), dtype=torch.float32, device=x.device
            )
            _dbias = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
        # F1 (bias-grad fold) is not plumbed through this entry point yet; the
        # kernel gate stays off so the compiled control path is unchanged.
        sum_dx = False
        dweight = torch.empty((D,), dtype=weight.dtype, device=x.device)
        dbias = torch.empty((D,), dtype=weight.dtype, device=x.device)
        if N == 0:
            dweight.zero_()
            dbias.zero_()
            return dx, dweight, dbias, None
        if use_v2:
            tile_num, _dweight, _dbias, _ = _wln_bwd_dx_partials(
                dx,
                dy,
                dx,
                x,
                weight,
                bias,
                mean,
                rstd,
                ctx.eps,
                is_swish=True,
                add_residual=False,
                sum_dx=sum_dx,
                BLOCK_D=ctx.BLOCK_D,
                sms=sms,
            )
        else:
            _dxsum = _dbias
            # pyre-ignore[28]
            _weighted_layer_norm_bwd_dx[(tile_num,)](
                dx,
                dy,
                dx,
                _dweight,
                _dbias,
                _dxsum,
                x,
                weight,
                bias,
                mean,
                rstd,
                dx.stride(0),
                dy.stride(0),
                0,
                x.stride(0),
                D,
                ctx.eps,
                IS_SWISH=True,
                ADD_RESIDUAL=False,
                SUM_DX=sum_dx,
                N=N,
                BLOCK_D=ctx.BLOCK_D,
            )

        def grid(META):
            return (triton.cdiv(D, META["BLOCK_D"]),)

        blocks = triton.next_power_of_2(sms * 4)
        BLOCK_D = triton.next_power_of_2(triton.cdiv(D, blocks))
        BLOCK_D = min(max(BLOCK_D, 4), 128)
        _layer_norm_bwd_dwdb[grid](
            _dweight,
            _dbias,
            dweight,
            dbias,
            tile_num,
            D,
            BLOCK_D=BLOCK_D,
        )

        return dx, dweight, dbias, None


@torch.jit.unused
@torch.fx.wrap
def triton_layer_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    return LayerNormFunction.apply(x, weight, bias, eps)


@torch.jit.unused
@torch.fx.wrap
def triton_rms_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
    silu: bool = False,
) -> torch.Tensor:
    return RMSNormFunction.apply(x, weight, eps, silu)


@torch.jit.unused
@torch.fx.wrap
def triton_swish_layer_norm(
    x: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    return SwishLayerNormFunction.apply(x, weight, bias, eps)
