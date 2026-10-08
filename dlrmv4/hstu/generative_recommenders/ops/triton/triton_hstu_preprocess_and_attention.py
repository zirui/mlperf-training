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

#!/usr/bin/env python3

# pyre-strict

import functools
import os
from typing import Callable, Optional, Tuple

import torch
import triton
import triton.language as tl
from generative_recommenders.ops.blas_solution_pins import install_for_tokens
from generative_recommenders.ops.triton.triton_addmm import (
    maybe_triton_addmm_fwd,
    triton_addmm_bwd,
    triton_addmm_fwd,
)
from generative_recommenders.ops.triton.triton_hstu_attention import (
    _should_enable_tma,
    triton_cached_hstu_mha,
    triton_hstu_attention_bwd,
    triton_hstu_attention_delta_bwd,
    triton_hstu_attention_fwd,
)
from generative_recommenders.ops.triton.triton_fp8 import (
    activation_scale_value,
    calibrate_scale_buffer,
    fp8_scale_tensor,
    fp8_supported,
    hstu_fp8_attention_enabled,
    hstu_fp8_fail_closed,
    hstu_fp8_fuse_ln_enabled,
    hstu_fp8_fuse_qkv_enabled,
    packed_weight_for_layout,
    refuse_fp8_with_flydsl_attention_forward,
    record_fp8_capability,
    record_fp8_contract,
    scaled_addmm_fp8_preq,
    scaled_mm_fp8_out_preq,
)
from generative_recommenders.ops.triton.triton_hstu_linear import (
    HSTU_DUVQK_BUFFER_ATTR,
    HSTU_DUVQK_WIDTH_ATTR,
)
from generative_recommenders.ops.triton.triton_layer_norm import (
    compute_BLOCK_D,
    triton_weighted_layer_norm_bwd,
    triton_weighted_layer_norm_dual_fwd,
    triton_weighted_layer_norm_fp8_save_bf16_fwd,
    triton_weighted_layer_norm_fwd,
)
from torch.nn import functional as F
from torch.autograd.profiler import record_function


_SAVE_FP8_UVQK = os.environ.get("DLRM_HSTU_FP8_SAVE_UVQK", "0") == "1"
_EARLY_RELEASE_FP8_UVQK = (
    os.environ.get("DLRM_HSTU_FP8_EARLY_RELEASE_UVQK", "0") == "1"
)
# Store-instead-of-recompute (instance F, user-approved class 2026-10-05): the
# FP8 forward LayerNorm also writes the BF16 normalized input, which is saved
# for the projection wgrad so backward no longer re-runs the LayerNorm. The
# stored tensor is bit-identical to the recompute from the stored fp32
# mean/rstd (L0, iter 2). The mean/rstd themselves come from a separately
# compiled/autotuned kernel variant, so their last fp32 bit can differ from
# the flag-off kernel, exactly as it already differs between autotune configs
# of the flag-off kernel: a declared reduction-order change, same formula.
# Costs 2 B per element of x held from forward to backward (~1 KiB/token/layer).
_SAVE_BF16_NORMED_X = (
    os.environ.get("DLRM_HSTU_FP8_SAVE_NORMED_X", "0") == "1"
)
@triton.jit
def _dequantize_fp8_kernel(
    value,
    output,
    scale,
    num_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = (
        tl.program_id(0).to(tl.int64) * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    )
    mask = offsets < num_elements
    value_block = tl.load(value + offsets, mask=mask).to(tl.float32)
    dequant_scale = tl.load(scale).to(tl.float32)
    tl.store(output + offsets, value_block * dequant_scale, mask=mask)


@triton.jit
def _dequantize_fp8_strided_output_kernel(
    value,
    output,
    scale,
    rows: tl.constexpr,
    columns: tl.constexpr,
    input_row_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    columns_offset = (
        tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ).to(tl.int64)
    mask = columns_offset < columns
    input_offsets = row * input_row_stride + columns_offset
    output_offsets = row * output_row_stride + columns_offset
    value_block = tl.load(value + input_offsets, mask=mask).to(tl.float32)
    dequant_scale = tl.load(scale).to(tl.float32)
    tl.store(output + output_offsets, value_block * dequant_scale, mask=mask)


def _dequantize_fp8(
    value: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    *,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not value.is_contiguous():
        raise ValueError("FP8 dequantization requires contiguous input")
    if output is None:
        output = torch.empty_like(value, dtype=dtype)
    elif output.shape != value.shape or output.dtype != dtype:
        raise ValueError("FP8 dequantization output contract mismatch")
    block_size = 1024
    if output.is_contiguous():
        _dequantize_fp8_kernel[(triton.cdiv(value.numel(), block_size),)](
            value,
            output,
            scale,
            num_elements=value.numel(),
            BLOCK_SIZE=block_size,
        )
    else:
        if value.dim() != 2 or output.dim() != 2 or output.stride(1) != 1:
            raise ValueError("strided FP8 dequantization requires row-major matrices")
        _dequantize_fp8_strided_output_kernel[
            (value.shape[0], triton.cdiv(value.shape[1], block_size))
        ](
            value,
            output,
            scale,
            rows=value.shape[0],
            columns=value.shape[1],
            input_row_stride=value.stride(0),
            output_row_stride=output.stride(0),
            BLOCK_SIZE=block_size,
        )
    return output


def _adopt_duvqk(
    dsilu_u: Optional[torch.Tensor],
    rows: int,
    width: int,
    u_width: int,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Return the d(uvqk) buffer dsilu_u already lives in, or None.

    With deferred SiLU, the incoming dsilu_u is the finished du. When the
    output-projection backward wrote it as the leading u_width columns of a
    (rows, width) buffer (it tags the view with HSTU_DUVQK_BUFFER_ATTR), that
    buffer is used as d(uvqk) directly, removing the du copy. Any mismatch
    (autograd replaced the gradient, other consumer, other layout) returns None
    and the caller falls back to allocate-and-copy.
    """
    if dsilu_u is None:
        return None
    buf = getattr(dsilu_u, HSTU_DUVQK_BUFFER_ATTR, None)
    if buf is None:
        return None
    if (
        buf.shape != (rows, width)
        or buf.stride() != (width, 1)
        or buf.dtype != dtype
        or dsilu_u.dtype != dtype
        or dsilu_u.shape != (rows, u_width)
        or dsilu_u.stride() != (width, 1)
        or dsilu_u.data_ptr() != buf.data_ptr()
    ):
        return None
    return buf


@functools.lru_cache(maxsize=1)
def _flydsl_bwd() -> Callable[..., bool]:
    """The FlyDSL in-place attention backward.

    This is the backward the production TRITON path actually runs (the non-fused
    ops/hstu_attention.py dispatch is a different call site), so $HSTU_ATTN_BWD
    has to be honoured here too. Only the *import* is cached: the callee re-reads
    the selection per call, which is what lets a test flip backends in-process.

    The import is NOT guarded. hstu_flydsl_mha pulls in only torch and triton --
    the flydsl library itself loads lazily, behind a hardware check that raises
    with an actionable message -- so a failure here means a broken tree, not an
    unsupported host, and swallowing it would silently cost the default backward.
    """
    from generative_recommenders.ops.flydsl.hstu_flydsl_mha import (
        try_flydsl_attention_bwd,
    )

    return try_flydsl_attention_bwd


@functools.lru_cache(maxsize=1)
def _flydsl_fwd() -> Callable[..., Optional[torch.Tensor]]:
    """The FlyDSL attention forward, selected by $HSTU_ATTN_FWD.

    Separate from _flydsl_bwd because the two halves are independently
    selectable: HSTU has no softmax, so the forward hands the backward nothing,
    and running one on FlyDSL with the other on Triton is both valid and the only
    way to A/B a single half. Returns None per call when not selected or when the
    shape is not specialized, and the caller falls back to Triton.
    """
    from generative_recommenders.ops.flydsl.hstu_flydsl_mha import (
        try_flydsl_attention_fwd,
    )

    return try_flydsl_attention_fwd


@functools.lru_cache(maxsize=1)
def _flydsl_fwd_fp8() -> Callable[..., Optional[torch.Tensor]]:
    """The FlyDSL e4m3 attention forward.

    Selected by DLRM_HSTU_FP8_FWD=1 plus HSTU_ATTN_FWD=flydsl (deprecated
    alias: DLRM_HSTU_FP8_FWD_FLYDSL=1). Separate from _flydsl_fwd because it
    is a different kernel specialization: e4m3 q/k/v plus the fused vqk GEMM's
    per-tensor scale, vs bf16 and no scale. Returns None only when the knob
    is off -- once on it is fail-closed and raises rather than falling back.
    """
    from generative_recommenders.ops.flydsl.hstu_flydsl_mha import (
        try_flydsl_attention_fwd_fp8,
    )

    return try_flydsl_attention_fwd_fp8


class _HSTUPreprocessAndAttentionFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore [14]
    def forward(
        ctx,  # pyre-ignore [2]
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        norm_eps: float,
        num_heads: int,
        attn_dim: int,
        hidden_dim: int,
        uvqk_weight: torch.Tensor,
        uvqk_bias: torch.Tensor,
        max_seq_len: int,
        seq_offsets: torch.Tensor,
        attn_alpha: float,
        num_targets: Optional[torch.Tensor],
        max_attn_len: int,
        contextual_seq_len: int,
        recompute_uvqk_in_backward: bool,
        recompute_normed_x_in_backward: bool,
        sort_by_length: bool,
        enable_tma: bool,
        fp8_input_scale: Optional[torch.Tensor] = None,
        fp8_qkv_scale: Optional[torch.Tensor] = None,
        defer_silu: bool = False,
        allow_save_fp8_uvqk: bool = True,
        num_softmax_heads: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert num_softmax_heads == 0, "Softmax attention is not supported"
        refuse_fp8_with_flydsl_attention_forward()
        fp8_scale_value = activation_scale_value(fp8_input_scale)
        fp8_scale = (
            fp8_scale_tensor(fp8_input_scale)
            if fp8_input_scale is not None
            else None
        )
        qkv_scale_value = activation_scale_value(fp8_qkv_scale)
        fp8_requested = hstu_fp8_fuse_ln_enabled() and fp8_input_scale is not None
        use_fp8 = (
            fp8_requested
            and recompute_normed_x_in_backward
            and fp8_supported(x, uvqk_weight)
        )
        if fp8_requested and not use_fp8 and hstu_fp8_fail_closed():
            raise RuntimeError(
                "Plan 08 input FP8 path requested for an unsupported contract"
            )
        fp8_qkv_feature_requested = (
            use_fp8
            and hstu_fp8_fuse_qkv_enabled()
            and hstu_fp8_attention_enabled()
        )
        fp8_qkv_requested = (
            fp8_qkv_feature_requested and recompute_uvqk_in_backward
        )
        if (
            fp8_qkv_feature_requested
            and not fp8_qkv_requested
            and hstu_fp8_fail_closed()
        ):
            raise RuntimeError(
                "Plan 08 QKV FP8 requires BF16 UVQK recompute in backward"
            )
        use_fp8_qkv = fp8_qkv_requested and fp8_qkv_scale is not None
        if fp8_qkv_requested and not use_fp8_qkv and hstu_fp8_fail_closed():
            raise RuntimeError("Plan 08 QKV FP8 scale state is missing")
        fp8_qkv_forward = False
        # The projection GEMMs' tuned BLAS solutions are keyed on the token
        # count, which changes every step, so they have to be registered per
        # batch. x.shape[0] is that count -- the LN below is row-wise, so the
        # projection sees the same M -- and registering here covers the
        # out-projection and the backward GEMMs too, since all the pins go in
        # together. It has to precede the branch rather than sit after the
        # projection, or the forward pins would miss on every new count. Under
        # fx tracing the shape is a Proxy, hence the guard.
        num_tokens = x.shape[0]
        if isinstance(num_tokens, int):
            install_for_tokens(num_tokens)
        saved_normed_x_bf16 = None
        if use_fp8 and fp8_scale_value is not None:
            assert fp8_scale is not None
            # Only worth storing when backward would otherwise take the plain
            # BF16 LayerNorm recompute, i.e. when the e4m3 V/Q/K pack is
            # saved (the no-save path recomputes BF16+E4M3 in one dual pass).
            # needs_input_grad[0] (x) is False for eval/no_grad forwards,
            # which must not pay the extra BF16 write.
            save_bf16_normed_x = (
                _SAVE_BF16_NORMED_X
                and bool(ctx.needs_input_grad[0])
                and recompute_normed_x_in_backward
                and recompute_uvqk_in_backward
                and _SAVE_FP8_UVQK
                and allow_save_fp8_uvqk
                and use_fp8_qkv
                and qkv_scale_value is not None
            )
            if save_bf16_normed_x:
                (
                    saved_normed_x_bf16,
                    normed_x,
                    x_mean,
                    x_rstd,
                ) = triton_weighted_layer_norm_fp8_save_bf16_fwd(
                    x=x,
                    weight=norm_weight,
                    bias=norm_bias,
                    eps=norm_eps,
                    inv_scale=1.0 / fp8_scale_value,
                )
            else:
                normed_x, x_mean, x_rstd = triton_weighted_layer_norm_fwd(
                    x=x,
                    weight=norm_weight,
                    bias=norm_bias,
                    eps=norm_eps,
                    output_fp8=True,
                    inv_scale=1.0 / fp8_scale_value,
                )
            if use_fp8_qkv and qkv_scale_value is not None:
                assert fp8_qkv_scale is not None
                u_columns = hidden_dim * num_heads
                u = scaled_addmm_fp8_preq(
                    activation_fp8=normed_x,
                    scale_a=fp8_scale,
                    weight=uvqk_weight,
                    bias_or_residual=uvqk_bias[:u_columns],
                    out_dtype=x.dtype,
                    weight_layout=((0, u_columns),),
                ).contiguous()
                vqk = scaled_mm_fp8_out_preq(
                    activation_fp8=normed_x,
                    scale_a=fp8_scale,
                    weight=uvqk_weight,
                    bias=uvqk_bias[u_columns:],
                    weight_layout=((u_columns, uvqk_weight.shape[1]),),
                    output_scale=fp8_qkv_scale,
                    scale_a_value=fp8_scale_value,
                    output_scale_value=qkv_scale_value,
                ).contiguous()
                v, q, k = vqk.split(
                    [
                        hidden_dim * num_heads,
                        attn_dim * num_heads,
                        attn_dim * num_heads,
                    ],
                    dim=1,
                )
                uvqk = torch.empty(0, dtype=x.dtype, device=x.device)
                fp8_qkv_forward = True
            else:
                uvqk = scaled_addmm_fp8_preq(
                    activation_fp8=normed_x,
                    scale_a=fp8_scale,
                    weight=uvqk_weight,
                    bias_or_residual=uvqk_bias,
                    out_dtype=x.dtype,
                ).contiguous()
                u, v, q, k = uvqk.split(
                    [
                        hidden_dim * num_heads,
                        hidden_dim * num_heads,
                        attn_dim * num_heads,
                        attn_dim * num_heads,
                    ],
                    dim=1,
                )
                if use_fp8_qkv:
                    assert fp8_qkv_scale is not None
                    calibrate_scale_buffer(
                        fp8_qkv_scale,
                        uvqk[:, hidden_dim * num_heads :],
                    )
        else:
            normed_x, x_mean, x_rstd = triton_weighted_layer_norm_fwd(
                x=x,
                weight=norm_weight,
                bias=norm_bias,
                eps=norm_eps,
            )
            if use_fp8:
                calibrate_scale_buffer(fp8_input_scale, normed_x)
            uvqk = maybe_triton_addmm_fwd(
                x=normed_x,
                w=uvqk_weight.to(normed_x.dtype),
                y=uvqk_bias.to(normed_x.dtype),
            ).contiguous()
            u, v, q, k = uvqk.split(
                [
                    hidden_dim * num_heads,
                    hidden_dim * num_heads,
                    attn_dim * num_heads,
                    attn_dim * num_heads,
                ],
                dim=1,
            )
            if use_fp8_qkv:
                assert fp8_qkv_scale is not None
                calibrate_scale_buffer(
                    fp8_qkv_scale,
                    uvqk[:, hidden_dim * num_heads :],
                )
        if use_fp8 and fp8_scale_value is not None:
            record_fp8_capability("phi1_input_ln_projection")
            record_fp8_contract(
                "phi1_input_ln_projection",
                normed_x=normed_x,
                u=u,
            )
        BLOCK_D = compute_BLOCK_D(x)
        q = q.view(-1, num_heads, attn_dim)
        k = k.view(-1, num_heads, attn_dim)
        v = v.view(-1, num_heads, hidden_dim)
        attention_qkv_scale = (
            qkv_scale_value if fp8_qkv_forward and qkv_scale_value else 1.0
        )
        fuse_silu = defer_silu
        silu_u = u if fuse_silu else F.silu(u)
        sort_by_length_indices = None
        if sort_by_length:
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            _, sort_by_length_indices = torch.sort(
                seq_lengths, descending=True, stable=False
            )
        out = None
        if fp8_qkv_forward:
            # FlyDSL e4m3 forward when HSTU_ATTN_FWD=flydsl (or the deprecated
            # DLRM_HSTU_FP8_FWD_FLYDSL alias). q/k/v are e4m3 column slices of
            # the vqk buffer; attention_qkv_scale is the scale they were
            # quantized with -- the same tensors the Triton FP8 forward gets.
            out = _flydsl_fwd_fp8()(
                q=q,
                k=k,
                v=v,
                seq_offsets=seq_offsets,
                num_targets=num_targets,
                N=max_seq_len,
                alpha=attn_alpha,
                max_attn_len=max_attn_len,
                contextual_seq_len=contextual_seq_len,
                num_softmax_heads=num_softmax_heads,
                qkv_scale=attention_qkv_scale,
            )
        else:
            # bf16 FlyDSL forward. The e4m3 path is the branch above; this
            # returns None while DLRM_HSTU_FP8_FWD=1 so we never feed e4m3
            # into the bf16 kernel.
            out = _flydsl_fwd()(
                q=q,
                k=k,
                v=v,
                seq_offsets=seq_offsets,
                num_targets=num_targets,
                N=max_seq_len,
                alpha=attn_alpha,
                max_attn_len=max_attn_len,
                contextual_seq_len=contextual_seq_len,
                num_softmax_heads=num_softmax_heads,
            )
        if out is None:
            out = triton_hstu_attention_fwd(
                N=max_seq_len,
                alpha=attn_alpha,
                q=q,
                k=k,
                v=v,
                seq_offsets=seq_offsets,
                num_targets=num_targets,
                max_attn_len=max_attn_len,
                contextual_seq_len=contextual_seq_len,
                sort_by_length_indices=sort_by_length_indices,
                enable_tma=enable_tma,
                num_softmax_heads=num_softmax_heads,
                q_scale=attention_qkv_scale,
                k_scale=attention_qkv_scale,
                v_scale=attention_qkv_scale,
            )
        if fp8_qkv_forward:
            record_fp8_capability("phi2_qkv_attention")
            record_fp8_contract(
                "phi2_qkv_attention",
                attention_output=out,
                k=k,
                q=q,
                u=u,
                v=v,
            )
        # update ctx
        saved_tensors = [
            x,
            norm_weight,
            norm_bias,
            x_mean,
            x_rstd,
            uvqk_weight,
            seq_offsets,
        ]
        if num_targets is not None:
            saved_tensors.append(num_targets)
        if not recompute_normed_x_in_backward:
            saved_tensors.append(normed_x)
        save_fp8_uvqk = (
            recompute_uvqk_in_backward
            and fp8_qkv_forward
            and _SAVE_FP8_UVQK
            and allow_save_fp8_uvqk
        )
        early_release_fp8_uvqk = save_fp8_uvqk and _EARLY_RELEASE_FP8_UVQK
        if save_fp8_uvqk:
            if early_release_fp8_uvqk:
                if not fuse_silu:
                    saved_tensors.append(u)
                # Retained-graph repeated backward recomputes V/Q/K after the
                # one-shot ephemeral pack has been consumed.
                saved_tensors.append(uvqk_bias)
            elif fuse_silu:
                saved_tensors.append(vqk)
            else:
                saved_tensors.extend((u, vqk))
        elif recompute_uvqk_in_backward:
            saved_tensors.append(uvqk_bias)
        else:
            saved_tensors.append(uvqk)
        if sort_by_length:
            saved_tensors.append(sort_by_length_indices)
        # Must stay the LAST saved tensor: backward reads it as [-1].
        ctx.saved_bf16_normed_x = (
            save_fp8_uvqk and saved_normed_x_bf16 is not None
        )
        if ctx.saved_bf16_normed_x:
            saved_tensors.append(saved_normed_x_bf16)
        saved_normed_x_bf16 = None
        ctx.save_for_backward(*saved_tensors)
        ctx.attn_alpha = attn_alpha
        ctx.has_multiple_targets = num_targets is not None
        ctx.max_seq_len = max_seq_len
        ctx.max_attn_len = max_attn_len
        ctx.recompute_normed_x_in_backward = recompute_normed_x_in_backward
        ctx.recompute_uvqk_in_backward = recompute_uvqk_in_backward
        ctx.hidden_dim = hidden_dim
        ctx.attn_dim = attn_dim
        ctx.num_heads = num_heads
        ctx.uvqk_bias_1d = uvqk_bias.dim() == 1
        ctx.norm_eps = norm_eps
        ctx.norm_BLOCK_D = BLOCK_D
        ctx.contextual_seq_len = contextual_seq_len
        ctx.sort_by_length = sort_by_length
        ctx.enable_tma = enable_tma
        ctx.num_softmax_heads = num_softmax_heads
        ctx.fp8_forward = fp8_qkv_forward
        ctx.fuse_silu = fuse_silu
        ctx.save_fp8_uvqk = save_fp8_uvqk
        ctx.early_release_fp8_uvqk = early_release_fp8_uvqk
        ctx._ephemeral_fp8_vqk = vqk if early_release_fp8_uvqk else None
        ctx.fp8_input_scale_value = fp8_scale_value
        ctx.fp8_qkv_scale_value = qkv_scale_value
        if fuse_silu:
            # Ask the consumer's backward (HSTUComputeOutputFunction) to write
            # d(silu_u) into the leading columns of a d(uvqk)-wide buffer, which
            # backward below adopts instead of copying du into a fresh one.
            setattr(silu_u, HSTU_DUVQK_WIDTH_ATTR, uvqk_weight.shape[1])
        return silu_u, out, x

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx,  # pyre-ignore[2]
        dsilu_u: torch.Tensor,
        dout: torch.Tensor,
        dresidual: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,  # d_x
        torch.Tensor,  # d_norm_weight
        torch.Tensor,  # d_norm_bias
        None,
        None,
        None,
        None,
        torch.Tensor,  # d_uvqk_weight
        torch.Tensor,  # d_uvqk_bias
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        x, norm_weight, norm_bias, x_mean, x_rstd, uvqk_weight, seq_offsets = (
            ctx.saved_tensors[:7]
        )
        idx = 7
        if ctx.has_multiple_targets:
            num_targets = ctx.saved_tensors[idx]
            idx += 1
        else:
            num_targets = None
        ephemeral_vqk_fp8 = (
            getattr(ctx, "_ephemeral_fp8_vqk", None)
            if ctx.early_release_fp8_uvqk
            else None
        )
        recompute_saved_fp8 = (
            ctx.save_fp8_uvqk
            and ctx.early_release_fp8_uvqk
            and ephemeral_vqk_fp8 is None
        )
        recompute_fp8_qkv = (
            ctx.recompute_uvqk_in_backward
            and ctx.fp8_forward
            and (not ctx.save_fp8_uvqk or recompute_saved_fp8)
        )
        normed_x_fp8 = None
        release_fp8_after_attention = (
            ctx.recompute_normed_x_in_backward
            and ctx.fp8_forward
            and (recompute_fp8_qkv or ctx.early_release_fp8_uvqk)
        )
        if ctx.recompute_normed_x_in_backward:
            if recompute_fp8_qkv:
                if ctx.fp8_input_scale_value is None:
                    raise RuntimeError(
                        "missing frozen FP8 input scale for LayerNorm recompute"
                    )
                normed_x, normed_x_fp8 = triton_weighted_layer_norm_dual_fwd(
                    x=x,
                    weight=norm_weight,
                    bias=norm_bias,
                    eps=ctx.norm_eps,
                    mean=x_mean,
                    rstd=x_rstd,
                    inv_scale=1.0 / ctx.fp8_input_scale_value,
                )
            elif getattr(ctx, "saved_bf16_normed_x", False):
                # Forward stored exactly what this LayerNorm would rebuild.
                normed_x = ctx.saved_tensors[-1]
            else:
                normed_x, _, _ = triton_weighted_layer_norm_fwd(
                    x=x,
                    weight=norm_weight,
                    bias=norm_bias,
                    eps=ctx.norm_eps,
                    mean=x_mean,
                    rstd=x_rstd,
                )
        else:
            normed_x = ctx.saved_tensors[idx]
            idx += 1
        vqk_fp8 = None
        attention_qkv_scale = 1.0
        if ctx.recompute_uvqk_in_backward:
            if ctx.save_fp8_uvqk:
                if ctx.early_release_fp8_uvqk:
                    if not ctx.fuse_silu:
                        u = ctx.saved_tensors[idx]
                        idx += 1
                    uvqk_bias = ctx.saved_tensors[idx]
                    idx += 1
                    vqk_fp8 = ephemeral_vqk_fp8
                elif ctx.fuse_silu:
                    vqk_fp8 = ctx.saved_tensors[idx]
                    idx += 1
                else:
                    u = ctx.saved_tensors[idx]
                    vqk_fp8 = ctx.saved_tensors[idx + 1]
                    idx += 2
                if ctx.fp8_qkv_scale_value is None:
                    raise RuntimeError(
                        "missing frozen FP8 Q/K/V scale for saved activations"
                    )
                qkv_scale = torch.full(
                    (1, 1),
                    ctx.fp8_qkv_scale_value,
                    dtype=torch.float32,
                    device=x.device,
                )
                if vqk_fp8 is not None:
                    v, q, k = vqk_fp8.split(
                        [
                            ctx.hidden_dim * ctx.num_heads,
                            ctx.attn_dim * ctx.num_heads,
                            ctx.attn_dim * ctx.num_heads,
                        ],
                        dim=1,
                    )
                uvqk = None
                attention_qkv_scale = ctx.fp8_qkv_scale_value
            else:
                uvqk_bias = ctx.saved_tensors[idx]
                idx += 1
            if recompute_fp8_qkv:
                if (
                    ctx.fp8_input_scale_value is None
                    or ctx.fp8_qkv_scale_value is None
                ):
                    raise RuntimeError(
                        "missing frozen FP8 scales for Q/K/V backward recompute"
                    )
                input_scale = torch.full(
                    (1, 1),
                    ctx.fp8_input_scale_value,
                    dtype=torch.float32,
                    device=x.device,
                )
                qkv_scale = torch.full(
                    (1, 1),
                    ctx.fp8_qkv_scale_value,
                    dtype=torch.float32,
                    device=x.device,
                )
                if normed_x_fp8 is None:
                    normed_x_fp8, _, _ = triton_weighted_layer_norm_fwd(
                        x=x,
                        weight=norm_weight,
                        bias=norm_bias,
                        eps=ctx.norm_eps,
                        mean=x_mean,
                        rstd=x_rstd,
                        output_fp8=True,
                        inv_scale=1.0 / ctx.fp8_input_scale_value,
                    )
                u_columns = ctx.hidden_dim * ctx.num_heads
                if not ctx.fuse_silu and not ctx.save_fp8_uvqk:
                    u = scaled_addmm_fp8_preq(
                        activation_fp8=normed_x_fp8,
                        scale_a=input_scale,
                        weight=uvqk_weight,
                        bias_or_residual=uvqk_bias[:u_columns],
                        out_dtype=x.dtype,
                        weight_layout=((0, u_columns),),
                    ).contiguous()
                vqk_fp8 = scaled_mm_fp8_out_preq(
                    activation_fp8=normed_x_fp8,
                    scale_a=input_scale,
                    weight=uvqk_weight,
                    bias=uvqk_bias[u_columns:],
                    weight_layout=((u_columns, uvqk_weight.shape[1]),),
                    output_scale=qkv_scale,
                    scale_a_value=ctx.fp8_input_scale_value,
                    output_scale_value=ctx.fp8_qkv_scale_value,
                ).contiguous()
                v, q, k = vqk_fp8.split(
                    [
                        ctx.hidden_dim * ctx.num_heads,
                        ctx.attn_dim * ctx.num_heads,
                        ctx.attn_dim * ctx.num_heads,
                    ],
                    dim=1,
                )
                uvqk = None
                attention_qkv_scale = ctx.fp8_qkv_scale_value
                # scaled_mm has consumed this input. Drop the full-row FP8
                # LayerNorm output before allocating the much larger gradient
                # projection scratch below.
                normed_x_fp8 = None
            elif not ctx.save_fp8_uvqk:
                uvqk = maybe_triton_addmm_fwd(
                    x=normed_x,
                    w=uvqk_weight.to(normed_x.dtype),
                    y=uvqk_bias.to(normed_x.dtype),
                )
        else:
            uvqk = ctx.saved_tensors[idx]
            idx += 1
        if ctx.sort_by_length:
            sort_by_length_indices = ctx.saved_tensors[idx]
        else:
            sort_by_length_indices = None

        duvqk = (
            _adopt_duvqk(
                dsilu_u,
                rows=x.shape[0],
                width=uvqk_weight.shape[1],
                u_width=ctx.hidden_dim * ctx.num_heads,
                dtype=x.dtype,
            )
            if ctx.fuse_silu
            else None
        )
        du_in_place = duvqk is not None
        if duvqk is None:
            duvqk = torch.empty(
                (x.shape[0], uvqk_weight.shape[1]),
                dtype=x.dtype,
                device=x.device,
            )
        du, dv, dq, dk = duvqk.split(
            [
                ctx.hidden_dim * ctx.num_heads,
                ctx.hidden_dim * ctx.num_heads,
                ctx.attn_dim * ctx.num_heads,
                ctx.attn_dim * ctx.num_heads,
            ],
            dim=1,
        )
        if uvqk is not None:
            u, v, q, k = uvqk.split(
                [
                    ctx.hidden_dim * ctx.num_heads,
                    ctx.hidden_dim * ctx.num_heads,
                    ctx.attn_dim * ctx.num_heads,
                    ctx.attn_dim * ctx.num_heads,
                ],
                dim=1,
            )
        q = q.view(-1, ctx.num_heads, ctx.attn_dim)
        k = k.view(-1, ctx.num_heads, ctx.attn_dim)
        v = v.view(-1, ctx.num_heads, ctx.hidden_dim)
        dq = dq.view(-1, ctx.num_heads, ctx.attn_dim)
        dk = dk.view(-1, ctx.num_heads, ctx.attn_dim)
        dv = dv.view(-1, ctx.num_heads, ctx.hidden_dim)
        # Note: the operations below update duvqk in place. The FlyDSL backward
        # is the default ($HSTU_ATTN_BWD) but declines any shape it cannot serve,
        # so the Triton call below is still both the opt-out and the fallback.
        # It is *not* the fallback for an unsupported host: that raises inside.
        vqk_bf16 = None
        try:
            served = _flydsl_bwd()(
                dout=dout,
                q=q,
                k=k,
                v=v,
                dq=dq,
                dk=dk,
                dv=dv,
                seq_offsets=seq_offsets,
                num_targets=num_targets,
                N=ctx.max_seq_len,
                alpha=ctx.attn_alpha,
                max_attn_len=ctx.max_attn_len,
                contextual_seq_len=ctx.contextual_seq_len,
                num_softmax_heads=ctx.num_softmax_heads,
                qkv_scale=attention_qkv_scale,
            )
            if not served:
                if vqk_fp8 is not None:
                    vqk_bf16 = _dequantize_fp8(
                        vqk_fp8,
                        qkv_scale,
                        x.dtype,
                    )
                    v, q, k = vqk_bf16.split(
                        [
                            ctx.hidden_dim * ctx.num_heads,
                            ctx.attn_dim * ctx.num_heads,
                            ctx.attn_dim * ctx.num_heads,
                        ],
                        dim=1,
                    )
                    q = q.view(-1, ctx.num_heads, ctx.attn_dim)
                    k = k.view(-1, ctx.num_heads, ctx.attn_dim)
                    v = v.view(-1, ctx.num_heads, ctx.hidden_dim)
                triton_hstu_attention_bwd(
                    dout=dout,
                    q=q,
                    k=k,
                    v=v,
                    dq=dq,
                    dk=dk,
                    dv=dv,
                    seq_offsets=seq_offsets,
                    num_targets=num_targets,
                    N=ctx.max_seq_len,
                    max_attn_len=ctx.max_attn_len,
                    alpha=ctx.attn_alpha,
                    contextual_seq_len=ctx.contextual_seq_len,
                    sort_by_length_indices=sort_by_length_indices,
                    enable_tma=ctx.enable_tma,
                    num_softmax_heads=ctx.num_softmax_heads,
                )
        finally:
            if ctx.early_release_fp8_uvqk:
                ctx._ephemeral_fp8_vqk = None
        if ctx.fuse_silu:
            if not du_in_place:
                du.copy_(dsilu_u)
        else:
            torch.ops.aten.silu_backward(dsilu_u, u, grad_input=du)
        if release_fp8_after_attention:
            # Q/K/V are dead after attention backward. Releasing their packed
            # full-row storage and its context owner before projection backward
            # prevents the saved or recomputed FP8 input from inflating that
            # operation's HBM high-water mark. The BF16 LayerNorm result
            # remains live for the projection gradient.
            ephemeral_vqk_fp8 = None
            q = None
            k = None
            v = None
            vqk_fp8 = None
            vqk_bf16 = None
        assert normed_x is not None
        with record_function("## hstu_ordinary_uvqk_backward ##"):
            d_normed_x, d_uvqk_weight, d_uvqk_bias = triton_addmm_bwd(
                x=normed_x,
                w=uvqk_weight.to(normed_x.dtype),
                dz=duvqk,
                is_y_1d=ctx.uvqk_bias_1d,
            )
        d_uvqk_weight = d_uvqk_weight.to(uvqk_weight.dtype)
        d_uvqk_bias = d_uvqk_bias.to(uvqk_weight.dtype)
        d_x, d_norm_weight, d_norm_bias = triton_weighted_layer_norm_bwd(
            dy=d_normed_x,
            x=x,
            weight=norm_weight,
            bias=norm_bias,
            mean=x_mean,
            rstd=x_rstd,
            learnable=True,
            eps=ctx.norm_eps,
            BLOCK_D=ctx.norm_BLOCK_D,
            residual=dresidual,
        )
        # pyre-ignore[7]
        return (
            d_x,
            d_norm_weight,
            d_norm_bias,
            None,
            None,
            None,
            None,
            d_uvqk_weight,
            d_uvqk_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _HSTUPreprocessAndAttentionTargetsOnlyFunction(torch.autograd.Function):
    """Final-layer training path with full K/V and one compact U/Q per sequence."""

    @staticmethod
    # pyre-ignore [14]
    def forward(
        ctx,  # pyre-ignore [2]
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        norm_eps: float,
        num_heads: int,
        attn_dim: int,
        hidden_dim: int,
        uvqk_weight: torch.Tensor,
        uvqk_bias: torch.Tensor,
        max_seq_len: int,
        seq_offsets: torch.Tensor,
        attn_alpha: float,
        num_targets: torch.Tensor,
        enable_tma: bool,
        fp8_input_scale: Optional[torch.Tensor],
        fp8_qkv_scale: Optional[torch.Tensor],
        allow_save_fp8_vk: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        u_dim = hidden_dim * num_heads
        v_dim = hidden_dim * num_heads
        q_dim = attn_dim * num_heads
        k_dim = attn_dim * num_heads
        refuse_fp8_with_flydsl_attention_forward()
        fp8_scale_value = activation_scale_value(fp8_input_scale)
        fp8_scale = (
            fp8_scale_tensor(fp8_input_scale)
            if fp8_input_scale is not None
            else None
        )
        qkv_scale_value = activation_scale_value(fp8_qkv_scale)
        fp8_requested = (
            hstu_fp8_fuse_qkv_enabled()
            and hstu_fp8_attention_enabled()
        )
        use_fp8_vk = (
            fp8_requested
            and fp8_input_scale is not None
            and fp8_qkv_scale is not None
            and fp8_supported(x, uvqk_weight)
        )
        if fp8_requested and not use_fp8_vk and hstu_fp8_fail_closed():
            raise RuntimeError(
                "Plan 08 targets-only FP8 path requested for an unsupported contract"
            )
        input_fp8_forward = use_fp8_vk and fp8_scale_value is not None
        qkv_fp8_forward = input_fp8_forward and qkv_scale_value is not None
        # See _SAVE_BF16_NORMED_X: keep the BF16 normalized input from the
        # forward LayerNorm for the backward VK wgrad / compact UQ rows
        # instead of re-running the LayerNorm there.
        saved_normed_x_bf16 = None
        if (
            input_fp8_forward
            and _SAVE_BF16_NORMED_X
            and bool(ctx.needs_input_grad[0])
        ):
            assert fp8_scale is not None
            (
                saved_normed_x_bf16,
                normed_x,
                x_mean,
                x_rstd,
            ) = triton_weighted_layer_norm_fp8_save_bf16_fwd(
                x=x,
                weight=norm_weight,
                bias=norm_bias,
                eps=norm_eps,
                inv_scale=1.0 / fp8_scale_value,
            )
        elif input_fp8_forward:
            assert fp8_scale is not None
            normed_x, x_mean, x_rstd = triton_weighted_layer_norm_fwd(
                x=x,
                weight=norm_weight,
                bias=norm_bias,
                eps=norm_eps,
                output_fp8=True,
                inv_scale=1.0 / fp8_scale_value,
            )
        else:
            normed_x, x_mean, x_rstd = triton_weighted_layer_norm_fwd(
                x=x,
                weight=norm_weight,
                bias=norm_bias,
                eps=norm_eps,
            )
            if use_fp8_vk:
                calibrate_scale_buffer(fp8_input_scale, normed_x)
        BLOCK_D = compute_BLOCK_D(x)
        w_u, w_v, w_q, w_k = uvqk_weight.split(
            [u_dim, v_dim, q_dim, k_dim], dim=1
        )
        b_u, b_v, b_q, b_k = uvqk_bias.split(
            [u_dim, v_dim, q_dim, k_dim], dim=0
        )
        candidate_rows = seq_offsets[1:] - 1
        candidate_normed_x_bf16 = None
        if saved_normed_x_bf16 is not None:
            # Same values as the per-row LayerNorm below (same fp32
            # mean/rstd, same kernel arithmetic), without the gathers + launch.
            candidate_normed_x_bf16 = torch.index_select(
                saved_normed_x_bf16, 0, candidate_rows
            )
        elif input_fp8_forward:
            candidate_x = torch.index_select(x, 0, candidate_rows)
            candidate_normed_x_bf16, _, _ = triton_weighted_layer_norm_fwd(
                x=candidate_x,
                weight=norm_weight,
                bias=norm_bias,
                eps=norm_eps,
                mean=torch.index_select(x_mean, 0, candidate_rows),
                rstd=torch.index_select(x_rstd, 0, candidate_rows),
            )

        b_vk = torch.cat((b_v, b_k), dim=0)
        if qkv_fp8_forward:
            assert fp8_scale is not None
            assert fp8_qkv_scale is not None
            vk = scaled_mm_fp8_out_preq(
                activation_fp8=normed_x,
                scale_a=fp8_scale,
                weight=uvqk_weight,
                bias=b_vk,
                weight_layout=(
                    (u_dim, u_dim + v_dim),
                    (u_dim + v_dim + q_dim, uvqk_weight.shape[1]),
                ),
                output_scale=fp8_qkv_scale,
                scale_a_value=fp8_scale_value,
                output_scale_value=qkv_scale_value,
            ).contiguous()
            assert candidate_normed_x_bf16 is not None
            candidate_normed_x = candidate_normed_x_bf16
        elif input_fp8_forward:
            assert fp8_scale is not None
            vk = scaled_addmm_fp8_preq(
                activation_fp8=normed_x,
                scale_a=fp8_scale,
                weight=uvqk_weight,
                bias_or_residual=b_vk,
                out_dtype=x.dtype,
                weight_layout=(
                    (u_dim, u_dim + v_dim),
                    (u_dim + v_dim + q_dim, uvqk_weight.shape[1]),
                ),
            ).contiguous()
            assert candidate_normed_x_bf16 is not None
            candidate_normed_x = candidate_normed_x_bf16
        else:
            # Packing only the small weights keeps the large activation
            # projection at [L, 2D] without changing checkpoint layout.
            if use_fp8_vk:
                w_vk = packed_weight_for_layout(
                    uvqk_weight,
                    (
                        (u_dim, u_dim + v_dim),
                        (
                            u_dim + v_dim + q_dim,
                            uvqk_weight.shape[1],
                        ),
                    ),
                    dtype=normed_x.dtype,
                )
            else:
                w_vk = torch.cat((w_v, w_k), dim=1).to(normed_x.dtype)
            vk = maybe_triton_addmm_fwd(
                x=normed_x,
                w=w_vk,
                y=b_vk.to(normed_x.dtype),
            ).contiguous()
            candidate_normed_x = torch.index_select(
                normed_x, 0, candidate_rows
            )
        if use_fp8_vk and not qkv_fp8_forward:
            assert fp8_qkv_scale is not None
            calibrate_scale_buffer(fp8_qkv_scale, vk)
        v, k = vk.split([v_dim, k_dim], dim=1)
        if use_fp8_vk:
            w_uq = packed_weight_for_layout(
                uvqk_weight,
                (
                    (0, u_dim),
                    (u_dim + v_dim, u_dim + v_dim + q_dim),
                ),
                dtype=candidate_normed_x.dtype,
            )
        else:
            w_uq = torch.cat((w_u, w_q), dim=1).to(
                candidate_normed_x.dtype
            )
        b_uq = torch.cat((b_u, b_q), dim=0)
        uq = maybe_triton_addmm_fwd(
            x=candidate_normed_x,
            w=w_uq,
            y=b_uq.to(candidate_normed_x.dtype),
        ).contiguous()
        if input_fp8_forward:
            record_fp8_capability(
                "phi1_targets_compact_uq_banked_bf16"
            )
            record_fp8_contract(
                "phi1_targets_compact_uq_banked_bf16",
                compact_normed_x=candidate_normed_x,
                compact_uq=uq,
            )
        u, q = uq.split([u_dim, q_dim], dim=1)
        q = q.view(-1, num_heads, attn_dim)
        k = k.view(-1, num_heads, attn_dim)
        v = v.view(-1, num_heads, hidden_dim)
        attention_qkv_scale = (
            qkv_scale_value if qkv_fp8_forward and qkv_scale_value else 1.0
        )
        out = triton_cached_hstu_mha(
            N=max_seq_len,
            alpha=attn_alpha,
            delta_q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            max_attn_len=0,
            contextual_seq_len=0,
            enable_tma=enable_tma,
            q_scale=1.0,
            k_scale=attention_qkv_scale,
            v_scale=attention_qkv_scale,
        )
        if input_fp8_forward:
            record_fp8_capability("phi1_targets_input_ln")
            record_fp8_contract(
                "phi1_targets_input_ln",
                compact_q=q,
                compact_u=u,
                normed_x=normed_x,
            )
        if qkv_fp8_forward:
            record_fp8_capability("phi2_targets_vk_attention")
            record_fp8_contract(
                "phi2_targets_vk_attention",
                attention_output=out,
                compact_q=q,
                compact_u=u,
                full_k=k,
                full_v=v,
            )

        ctx.saved_bf16_normed_x = saved_normed_x_bf16 is not None
        ctx.save_for_backward(
            x,
            norm_weight,
            norm_bias,
            x_mean,
            x_rstd,
            uvqk_weight,
            uvqk_bias,
            seq_offsets,
            *((saved_normed_x_bf16,) if ctx.saved_bf16_normed_x else ()),
        )
        saved_normed_x_bf16 = None
        ctx.attn_alpha = attn_alpha
        ctx.max_seq_len = max_seq_len
        ctx.hidden_dim = hidden_dim
        ctx.attn_dim = attn_dim
        ctx.num_heads = num_heads
        ctx.norm_eps = norm_eps
        ctx.norm_BLOCK_D = BLOCK_D
        ctx.fp8_contract = use_fp8_vk
        ctx.fp8_input_forward = input_fp8_forward
        ctx.fp8_qkv_forward = qkv_fp8_forward
        ctx.fp8_input_scale_value = fp8_scale_value
        ctx.fp8_qkv_scale_value = qkv_scale_value
        # Hand the forward's e4m3 V/K pack to backward instead of rebuilding
        # it there (full-row FP8 LayerNorm + [L,512]x[512,1024] scaled GEMM),
        # under the same SAVE_UVQK / layer-mask / early-release policy as the
        # ordinary layers. It is a one-shot ctx attribute, not a saved tensor:
        # backward drops it after attention backward, so it never coexists
        # with the BF16 normalized input rematerialized for projection
        # backward, and a retained-graph second backward falls back to the
        # recompute. Backward then consumes exactly the values the forward
        # attention used.
        ctx._ephemeral_fp8_vk = (
            vk
            if (
                qkv_fp8_forward
                and _SAVE_FP8_UVQK
                and _EARLY_RELEASE_FP8_UVQK
                and allow_save_fp8_vk
            )
            else None
        )
        return F.silu(u), out, x

    @staticmethod
    # pyre-ignore [14]
    def backward(
        ctx,  # pyre-ignore [2]
        dsilu_u: torch.Tensor,
        dout: torch.Tensor,
        dresidual: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,  # norm_eps
        None,  # num_heads
        None,  # attn_dim
        None,  # hidden_dim
        torch.Tensor,
        torch.Tensor,
        None,  # max_seq_len
        None,  # seq_offsets
        None,  # attn_alpha
        None,  # num_targets
        None,  # enable_tma
        None,  # fp8_input_scale
        None,  # fp8_qkv_scale
        None,  # allow_save_fp8_vk
    ]:
        (
            x,
            norm_weight,
            norm_bias,
            x_mean,
            x_rstd,
            uvqk_weight,
            uvqk_bias,
            seq_offsets,
        ) = ctx.saved_tensors[:8]
        saved_normed_x_bf16 = (
            ctx.saved_tensors[8]
            if getattr(ctx, "saved_bf16_normed_x", False)
            else None
        )
        normed_x = None
        if not ctx.fp8_input_forward:
            normed_x, _, _ = triton_weighted_layer_norm_fwd(
                x=x,
                weight=norm_weight,
                bias=norm_bias,
                eps=ctx.norm_eps,
                mean=x_mean,
                rstd=x_rstd,
            )
        u_dim = ctx.hidden_dim * ctx.num_heads
        v_dim = ctx.hidden_dim * ctx.num_heads
        q_dim = ctx.attn_dim * ctx.num_heads
        k_dim = ctx.attn_dim * ctx.num_heads
        w_u, w_v, w_q, w_k = uvqk_weight.split(
            [u_dim, v_dim, q_dim, k_dim], dim=1
        )
        b_u, b_v, b_q, b_k = uvqk_bias.split(
            [u_dim, v_dim, q_dim, k_dim], dim=0
        )
        candidate_rows = seq_offsets[1:] - 1
        attention_vk_scale = 1.0
        if saved_normed_x_bf16 is not None:
            candidate_normed_x = torch.index_select(
                saved_normed_x_bf16, 0, candidate_rows
            )
        elif ctx.fp8_input_forward:
            candidate_normed_x, _, _ = triton_weighted_layer_norm_fwd(
                x=torch.index_select(x, 0, candidate_rows),
                weight=norm_weight,
                bias=norm_bias,
                eps=ctx.norm_eps,
                mean=torch.index_select(x_mean, 0, candidate_rows),
                rstd=torch.index_select(x_rstd, 0, candidate_rows),
            )
        else:
            candidate_normed_x = torch.index_select(
                normed_x, 0, candidate_rows
            )
        if ctx.fp8_contract:
            w_vk = packed_weight_for_layout(
                uvqk_weight,
                (
                    (u_dim, u_dim + v_dim),
                    (u_dim + v_dim + q_dim, uvqk_weight.shape[1]),
                ),
                dtype=x.dtype,
            )
        else:
            assert normed_x is not None
            w_vk = torch.cat((w_v, w_k), dim=1).to(normed_x.dtype)
        b_vk = torch.cat((b_v, b_k), dim=0).to(x.dtype)
        if ctx.fp8_contract:
            w_uq = packed_weight_for_layout(
                uvqk_weight,
                (
                    (0, u_dim),
                    (u_dim + v_dim, u_dim + v_dim + q_dim),
                ),
                dtype=x.dtype,
            )
        else:
            assert normed_x is not None
            w_uq = torch.cat((w_u, w_q), dim=1).to(normed_x.dtype)
        b_uq = torch.cat((b_u, b_q), dim=0).to(x.dtype)
        vk_fp8 = None
        saved_vk_fp8 = getattr(ctx, "_ephemeral_fp8_vk", None)
        ctx._ephemeral_fp8_vk = None
        if saved_vk_fp8 is not None:
            if ctx.fp8_qkv_scale_value is None:
                raise RuntimeError(
                    "missing frozen V/K scale for saved targets-only V/K"
                )
            vk_fp8 = saved_vk_fp8
            saved_vk_fp8 = None
            vk = vk_fp8
            attention_vk_scale = ctx.fp8_qkv_scale_value
        elif ctx.fp8_input_forward:
            if ctx.fp8_input_scale_value is None:
                raise RuntimeError(
                    "missing frozen input scale for targets-only backward"
                )
            input_scale = torch.full(
                (1, 1),
                ctx.fp8_input_scale_value,
                dtype=torch.float32,
                device=x.device,
            )
            normed_x_fp8, _, _ = triton_weighted_layer_norm_fwd(
                x=x,
                weight=norm_weight,
                bias=norm_bias,
                eps=ctx.norm_eps,
                mean=x_mean,
                rstd=x_rstd,
                output_fp8=True,
                inv_scale=1.0 / ctx.fp8_input_scale_value,
            )
            vk_layout = (
                (u_dim, u_dim + v_dim),
                (u_dim + v_dim + q_dim, uvqk_weight.shape[1]),
            )
            if ctx.fp8_qkv_forward:
                if ctx.fp8_qkv_scale_value is None:
                    raise RuntimeError(
                        "missing frozen V/K scale for targets-only backward"
                    )
                qkv_scale = torch.full(
                    (1, 1),
                    ctx.fp8_qkv_scale_value,
                    dtype=torch.float32,
                    device=x.device,
                )
                vk_fp8 = scaled_mm_fp8_out_preq(
                    activation_fp8=normed_x_fp8,
                    scale_a=input_scale,
                    weight=uvqk_weight,
                    bias=torch.cat((b_v, b_k), dim=0),
                    weight_layout=vk_layout,
                    output_scale=qkv_scale,
                    scale_a_value=ctx.fp8_input_scale_value,
                    output_scale_value=ctx.fp8_qkv_scale_value,
                ).contiguous()
                vk = vk_fp8
                attention_vk_scale = ctx.fp8_qkv_scale_value
            else:
                vk = scaled_addmm_fp8_preq(
                    activation_fp8=normed_x_fp8,
                    scale_a=input_scale,
                    weight=uvqk_weight,
                    bias_or_residual=torch.cat((b_v, b_k), dim=0),
                    out_dtype=x.dtype,
                    weight_layout=vk_layout,
                ).contiguous()
        else:
            assert normed_x is not None
            vk = maybe_triton_addmm_fwd(
                x=normed_x, w=w_vk, y=b_vk
            ).contiguous()
        v, k = vk.split([v_dim, k_dim], dim=1)
        uq = maybe_triton_addmm_fwd(
            x=candidate_normed_x, w=w_uq, y=b_uq
        ).contiguous()
        u, q = uq.split([u_dim, q_dim], dim=1)

        q = q.view(-1, ctx.num_heads, ctx.attn_dim)
        k = k.view(-1, ctx.num_heads, ctx.attn_dim)
        v = v.view(-1, ctx.num_heads, ctx.hidden_dim)
        dq = torch.empty_like(q)
        # Write dV/dK directly in the packed column order consumed by the
        # full-row projection backward. The delta kernel honors row strides,
        # so this avoids an O(LD) torch.cat materialization.
        dvk = torch.empty(
            (x.shape[0], v_dim + k_dim),
            dtype=x.dtype,
            device=x.device,
        )
        dv = dvk[:, :v_dim].view(-1, ctx.num_heads, ctx.hidden_dim)
        dk = dvk[:, v_dim:].view(-1, ctx.num_heads, ctx.attn_dim)
        triton_hstu_attention_delta_bwd(
            dout=dout,
            q=q,
            k=k,
            v=v,
            dq=dq,
            dk=dk,
            dv=dv,
            seq_offsets=seq_offsets,
            N=ctx.max_seq_len,
            alpha=ctx.attn_alpha,
            k_scale=attention_vk_scale,
            v_scale=attention_vk_scale,
        )

        du = torch.empty_like(u)
        torch.ops.aten.silu_backward(dsilu_u, u, grad_input=du)
        duq = torch.cat((du, dq.flatten(1, 2)), dim=1)
        with record_function("## hstu_targets_compact_uq_backward ##"):
            d_candidate_x, d_w_uq, d_b_uq = triton_addmm_bwd(
                x=candidate_normed_x,
                w=w_uq,
                dz=duq,
                is_y_1d=True,
            )
        if ctx.fp8_input_forward:
            # The full-row FP8 normalized input and V/K pack are dead once
            # attention backward has produced dV/dK. Release them before
            # materializing the BF16 normalized input needed by projection
            # backward so the two representations do not overlap at peak HBM.
            q = None
            k = None
            v = None
            vk = None
            vk_fp8 = None
            normed_x_fp8 = None
            if saved_normed_x_bf16 is not None:
                normed_x = saved_normed_x_bf16
                saved_normed_x_bf16 = None
            else:
                normed_x, _, _ = triton_weighted_layer_norm_fwd(
                    x=x,
                    weight=norm_weight,
                    bias=norm_bias,
                    eps=ctx.norm_eps,
                    mean=x_mean,
                    rstd=x_rstd,
                )
        assert normed_x is not None
        with record_function("## hstu_targets_full_vk_backward ##"):
            d_normed_x, d_w_vk, d_b_vk = triton_addmm_bwd(
                x=normed_x,
                w=w_vk,
                dz=dvk,
                is_y_1d=True,
            )
        d_normed_x.index_add_(0, candidate_rows, d_candidate_x)

        d_w_u, d_w_q = d_w_uq.split([u_dim, q_dim], dim=1)
        d_w_v, d_w_k = d_w_vk.split([v_dim, k_dim], dim=1)
        d_uvqk_weight = torch.cat((d_w_u, d_w_v, d_w_q, d_w_k), dim=1)
        d_b_u, d_b_q = d_b_uq.split([u_dim, q_dim], dim=0)
        d_b_v, d_b_k = d_b_vk.split([v_dim, k_dim], dim=0)
        d_uvqk_bias = torch.cat((d_b_u, d_b_v, d_b_q, d_b_k), dim=0)
        d_uvqk_weight = d_uvqk_weight.to(uvqk_weight.dtype)
        d_uvqk_bias = d_uvqk_bias.to(uvqk_bias.dtype)

        d_x, d_norm_weight, d_norm_bias = triton_weighted_layer_norm_bwd(
            dy=d_normed_x,
            x=x,
            weight=norm_weight,
            bias=norm_bias,
            mean=x_mean,
            rstd=x_rstd,
            learnable=True,
            eps=ctx.norm_eps,
            BLOCK_D=ctx.norm_BLOCK_D,
            residual=dresidual,
        )
        return (
            d_x,
            d_norm_weight,
            d_norm_bias,
            None,
            None,
            None,
            None,
            d_uvqk_weight,
            d_uvqk_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,  # allow_save_fp8_vk
        )


def triton_hstu_preprocess_and_attention(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    num_heads: int,
    attn_dim: int,
    hidden_dim: int,
    uvqk_weight: torch.Tensor,
    uvqk_bias: torch.Tensor,
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    attn_alpha: float,
    num_targets: Optional[torch.Tensor],
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    recompute_uvqk_in_backward: bool = False,
    recompute_normed_x_in_backward: bool = False,
    sort_by_length: bool = False,
    enable_tma: Optional[bool] = None,
    fp8_input_scale: Optional[torch.Tensor] = None,
    fp8_qkv_scale: Optional[torch.Tensor] = None,
    defer_silu: bool = False,
    allow_save_fp8_uvqk: bool = True,
    return_residual_proxy: bool = False,
):
    # When the caller does not specify enable_tma, auto-detect whether the
    # TMA / TLX fast path is safe on this device. Resolving here (vs inside
    # the autograd Function.forward) keeps a concrete bool flowing through
    # ctx.save_for_backward / ctx attributes.
    if enable_tma is None:
        enable_tma = _should_enable_tma()
    outputs = _HSTUPreprocessAndAttentionFunction.apply(
        x,
        norm_weight,
        norm_bias,
        norm_eps,
        num_heads,
        attn_dim,
        hidden_dim,
        uvqk_weight,
        uvqk_bias,
        max_seq_len,
        seq_offsets,
        attn_alpha,
        num_targets,
        max_attn_len,
        contextual_seq_len,
        recompute_uvqk_in_backward,
        recompute_normed_x_in_backward,
        sort_by_length,
        enable_tma,
        fp8_input_scale,
        fp8_qkv_scale,
        defer_silu,
        allow_save_fp8_uvqk,
    )
    return outputs if return_residual_proxy else outputs[:2]


def triton_hstu_preprocess_and_attention_targets_only(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    num_heads: int,
    attn_dim: int,
    hidden_dim: int,
    uvqk_weight: torch.Tensor,
    uvqk_bias: torch.Tensor,
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    attn_alpha: float,
    num_targets: torch.Tensor,
    enable_tma: Optional[bool] = None,
    fp8_input_scale: Optional[torch.Tensor] = None,
    fp8_qkv_scale: Optional[torch.Tensor] = None,
    return_residual_proxy: bool = False,
    allow_save_fp8_vk: bool = True,
):
    if enable_tma is None:
        enable_tma = _should_enable_tma()
    outputs = _HSTUPreprocessAndAttentionTargetsOnlyFunction.apply(
        x,
        norm_weight,
        norm_bias,
        norm_eps,
        num_heads,
        attn_dim,
        hidden_dim,
        uvqk_weight,
        uvqk_bias,
        max_seq_len,
        seq_offsets,
        attn_alpha,
        num_targets,
        enable_tma,
        fp8_input_scale,
        fp8_qkv_scale,
        allow_save_fp8_vk,
    )
    return outputs if return_residual_proxy else outputs[:2]
