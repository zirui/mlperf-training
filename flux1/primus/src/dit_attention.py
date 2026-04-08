# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Extracted from NeMo v2.6.0: nemo/collections/diffusion/models/dit/dit_attention.py
# Dependencies: Megatron-core + Transformer Engine only (no NeMo).

import logging
from dataclasses import dataclass
from typing import Union

import torch
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.core.transformer.attention import Attention, SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from megatron.core.transformer.custom_layers.transformer_engine import SplitAlongDim
except ImportError:
    SplitAlongDim = None
    logging.warning(
        "Failed to import SplitAlongDim from Transformer Engine. "
        "Falling back to torch.split."
    )


@dataclass
class JointSelfAttentionSubmodules:
    """Submodules for Joint Self-attention layer."""

    linear_qkv: Union[ModuleSpec, type] = None
    added_linear_qkv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None
    added_q_layernorm: Union[ModuleSpec, type] = None
    added_k_layernorm: Union[ModuleSpec, type] = None


class JointSelfAttention(Attention):
    """Joint Self-attention layer for MMDIT-like transformer blocks."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: JointSelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        context_pre_only: bool = False,
        **kwargs,
    ):
        config.normalization = "RMSNorm"
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            **kwargs,
        )

        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            self.query_projection_size + 2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
        )

        if submodules.added_linear_qkv is not None:
            self.added_linear_qkv = build_module(
                submodules.added_linear_qkv,
                self.config.hidden_size,
                self.query_projection_size + 2 * self.kv_projection_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_qkv_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='qkv',
            )

        if not context_pre_only:
            self.added_linear_proj = build_module(
                submodules.linear_proj,
                self.query_projection_size,
                self.config.hidden_size,
                config=self.config,
                init_method=self.config.output_layer_init_method,
                bias=self.config.add_bias_linear,
                input_is_parallel=True,
                skip_bias_add=True,
                is_expert=False,
                tp_comm_buffer_name='proj',
            )

        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

        if submodules.added_q_layernorm is not None:
            self.added_q_layernorm = build_module(
                submodules.added_q_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.added_q_layernorm = None

        if submodules.added_k_layernorm is not None:
            self.added_k_layernorm = build_module(
                submodules.added_k_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.added_k_layernorm = None

    def _split_qkv(self, mixed_qkv):
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        if SplitAlongDim is not None:
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
        else:
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        return query, key, value

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        query, key, value = self._split_qkv(mixed_qkv)

        if self.config.test_mode:
            self.run_realtime_tests()

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)
        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        return query, key, value

    def get_added_query_key_value_tensors(self, added_hidden_states, key_value_states=None):
        mixed_qkv, _ = self.added_linear_qkv(added_hidden_states)
        query, key, value = self._split_qkv(mixed_qkv)

        if self.config.test_mode:
            self.run_realtime_tests()

        if self.added_q_layernorm is not None:
            query = self.added_q_layernorm(query)
        if self.added_k_layernorm is not None:
            key = self.added_k_layernorm(key)

        return query, key, value

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
        additional_hidden_states=None,
    ):
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        query, key, value = self.get_query_key_value_tensors(hidden_states)
        added_query, added_key, added_value = self.get_added_query_key_value_tensors(additional_hidden_states)

        query = torch.cat([added_query, query], dim=0)
        key = torch.cat([added_key, key], dim=0)
        value = torch.cat([added_value, value], dim=0)

        query, key, value, rotary_pos_emb, attn_mask_type, *_ = self._adjust_key_value_for_inference(
            inference_params, query, key, value, rotary_pos_emb
        )

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None
            query = apply_rotary_pos_emb(query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q)
            key = apply_rotary_pos_emb(key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv)

        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask,
                attn_mask_type=attn_mask_type, packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query, key, value, attention_mask,
                attn_mask_type=attn_mask_type, packed_seq_params=packed_seq_params,
            )

        if packed_seq_params is not None:
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        encoder_attention_output = core_attn_out[: additional_hidden_states.shape[0], :, :]
        attention_output = core_attn_out[additional_hidden_states.shape[0] :, :, :]

        output, bias = self.linear_proj(attention_output)
        encoder_output, encoder_bias = self.added_linear_proj(encoder_attention_output)

        output = output + bias
        encoder_output = encoder_output + encoder_bias

        return output, encoder_output


class FluxSingleAttention(SelfAttention):
    """Self-attention layer for Flux single transformer blocks."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str = None,
        **kwargs,
    ):
        config.normalization = "RMSNorm"
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            **kwargs,
        )
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='proj',
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
    ):
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)

        query, key, value, rotary_pos_emb, attn_mask_type, *_ = self._adjust_key_value_for_inference(
            inference_params, query, key, value, rotary_pos_emb
        )

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            if packed_seq_params is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None
            query = apply_rotary_pos_emb(query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q)
            key = apply_rotary_pos_emb(key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv)

        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask,
                attn_mask_type=attn_mask_type, packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query, key, value, attention_mask,
                attn_mask_type=attn_mask_type, packed_seq_params=packed_seq_params,
            )

        if packed_seq_params is not None:
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        output, _ = self.linear_proj(core_attn_out)
        return output
