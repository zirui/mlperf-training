# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Extracted from NeMo v2.6.0: nemo/collections/diffusion/models/dit/dit_layer_spec.py
# Dependencies: Megatron-core + Transformer Engine only (no NeMo).
# Only Flux-relevant classes retained (MMDiTLayer, FluxSingleTransformerBlock,
# AdaLN, AdaLNContinuous, RMSNorm, and their spec builders).

import copy
import logging
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
from megatron.core.jit import jit_fuser
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.cuda_graphs import CudaGraphManager
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import make_viewless_tensor

from dit_attention import (
    FluxSingleAttention,
    JointSelfAttention,
    JointSelfAttentionSubmodules,
)

try:
    from megatron.core.transformer.custom_layers.transformer_engine import (
        TEColumnParallelLinear,
        TEDotProductAttention,
        TENorm,
        TERowParallelLinear,
    )
except ImportError:
    logging.warning(
        "Failed to import Transformer Engine dependencies. "
        "`from megatron.core.transformer.custom_layers.transformer_engine import *`"
    )


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, config, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class AdaLN(MegatronModule):
    """Adaptive Layer Normalization Module for DiT."""

    def __init__(
        self,
        config: TransformerConfig,
        n_adaln_chunks=9,
        norm=nn.LayerNorm,
        modulation_bias=False,
        use_second_norm=False,
    ):
        super().__init__(config)
        if norm == TENorm:
            self.ln = norm(config, config.hidden_size, config.layernorm_epsilon)
        else:
            self.ln = norm(config.hidden_size, elementwise_affine=False, eps=self.config.layernorm_epsilon)
        self.n_adaln_chunks = n_adaln_chunks
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            ColumnParallelLinear(
                config.hidden_size,
                self.n_adaln_chunks * config.hidden_size,
                config=config,
                init_method=nn.init.normal_,
                bias=modulation_bias,
                gather_output=True,
            ),
        )
        self.use_second_norm = use_second_norm
        if self.use_second_norm:
            self.ln2 = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=1e-6)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)

        setattr(self.adaLN_modulation[-1].weight, "sequence_parallel", config.sequence_parallel)

    @jit_fuser
    def forward(self, timestep_emb):
        output, bias = self.adaLN_modulation(timestep_emb)
        output = output + bias if bias else output
        return output.chunk(self.n_adaln_chunks, dim=-1)

    @jit_fuser
    def modulate(self, x, shift, scale):
        return x * (1 + scale) + shift

    @jit_fuser
    def scale_add(self, residual, x, gate):
        return residual + gate * x

    @jit_fuser
    def modulated_layernorm(self, x, shift, scale, layernorm_idx=0):
        if self.use_second_norm and layernorm_idx == 1:
            layernorm = self.ln2
        else:
            layernorm = self.ln
        input_layernorm_output = layernorm(x).type_as(x)
        return self.modulate(input_layernorm_output, shift, scale)

    @jit_fuser
    def scaled_modulated_layernorm(self, residual, x, gate, shift, scale, layernorm_idx=0):
        hidden_states = self.scale_add(residual, x, gate)
        shifted_pre_mlp_layernorm_output = self.modulated_layernorm(hidden_states, shift, scale, layernorm_idx)
        return hidden_states, shifted_pre_mlp_layernorm_output


class AdaLNContinuous(MegatronModule):
    """A variant of AdaLN used for Flux output normalization."""

    def __init__(
        self,
        config: TransformerConfig,
        conditioning_embedding_dim: int,
        modulation_bias: bool = True,
        norm_type: str = "layer_norm",
    ):
        super().__init__(config)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(conditioning_embedding_dim, config.hidden_size * 2, bias=modulation_bias)
        )
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=1e-6, bias=modulation_bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(config.hidden_size, config=config, eps=1e-6)
        else:
            raise ValueError("Unknown normalization type {}".format(norm_type))

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.adaLN_modulation(conditioning_embedding)
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class MMDiTLayer(TransformerLayer):
    """Multi-modal transformer layer for Flux double blocks.

    MMDiT layer from https://arxiv.org/pdf/2403.03206.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        context_pre_only: bool = False,
    ):
        hidden_size = config.hidden_size
        super().__init__(config=config, submodules=submodules, layer_number=layer_number)

        if config.enable_cuda_graph and config.cuda_graph_scope != "full_iteration":
            self.cudagraph_manager = CudaGraphManager(config, share_cudagraph_io_buffers=False)

        self.adaln = AdaLN(config, modulation_bias=True, n_adaln_chunks=6, use_second_norm=True)

        self.context_pre_only = context_pre_only
        context_norm_type = "ada_norm_continuous" if context_pre_only else "ada_norm_zero"

        if context_norm_type == "ada_norm_continuous":
            self.adaln_context = AdaLNContinuous(config, hidden_size, modulation_bias=True, norm_type="layer_norm")
        elif context_norm_type == "ada_norm_zero":
            self.adaln_context = AdaLN(config, modulation_bias=True, n_adaln_chunks=6, use_second_norm=True)
        else:
            raise ValueError(
                f"Unknown context_norm_type: {context_norm_type}, "
                f"currently only support `ada_norm_continuous`, `ada_norm_zero`"
            )

        cp_override_config = copy.deepcopy(config)
        cp_override_config.context_parallel_size = 1
        cp_override_config.tp_comm_overlap = False

        if not context_pre_only:
            self.context_mlp = build_module(
                submodules.mlp,
                config=cp_override_config,
            )
        else:
            self.context_mlp = None

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        inference_params=None,
        packed_seq_params=None,
        emb=None,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln(emb)

        norm_hidden_states = self.adaln.modulated_layernorm(
            hidden_states, shift=shift_msa, scale=scale_msa, layernorm_idx=0
        )
        if self.context_pre_only:
            norm_encoder_hidden_states = self.adaln_context(encoder_hidden_states, emb)
        else:
            c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.adaln_context(emb)
            norm_encoder_hidden_states = self.adaln_context.modulated_layernorm(
                encoder_hidden_states, shift=c_shift_msa, scale=c_scale_msa, layernorm_idx=0
            )

        attention_output, encoder_attention_output = self.self_attention(
            norm_hidden_states,
            attention_mask=attention_mask,
            key_value_states=None,
            additional_hidden_states=norm_encoder_hidden_states,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = self.adaln.scale_add(hidden_states, x=attention_output, gate=gate_msa)
        norm_hidden_states = self.adaln.modulated_layernorm(
            hidden_states, shift=shift_mlp, scale=scale_mlp, layernorm_idx=1
        )

        mlp_output, mlp_output_bias = self.mlp(norm_hidden_states)
        hidden_states = self.adaln.scale_add(hidden_states, x=(mlp_output + mlp_output_bias), gate=gate_mlp)

        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            encoder_hidden_states = self.adaln_context.scale_add(
                encoder_hidden_states, x=encoder_attention_output, gate=c_gate_msa
            )
            norm_encoder_hidden_states = self.adaln_context.modulated_layernorm(
                encoder_hidden_states, shift=c_shift_mlp, scale=c_scale_mlp, layernorm_idx=1
            )

            context_mlp_output, context_mlp_output_bias = self.context_mlp(norm_encoder_hidden_states)
            encoder_hidden_states = self.adaln.scale_add(
                encoder_hidden_states, x=(context_mlp_output + context_mlp_output_bias), gate=c_gate_mlp
            )

        return hidden_states, encoder_hidden_states

    def __call__(self, *args, **kwargs):
        if hasattr(self, 'cudagraph_manager'):
            return self.cudagraph_manager(self, args, kwargs)
        return super(MegatronModule, self).__call__(*args, **kwargs)


class FluxSingleTransformerBlock(TransformerLayer):
    """Flux single transformer block using Megatron-core."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        mlp_ratio: int = 4,
        n_adaln_chunks: int = 3,
        modulation_bias: bool = True,
    ):
        super().__init__(config=config, submodules=submodules, layer_number=layer_number)

        if config.enable_cuda_graph and config.cuda_graph_scope != "full_iteration":
            self.cudagraph_manager = CudaGraphManager(config, share_cudagraph_io_buffers=False)
        self.adaln = AdaLN(
            config=config, n_adaln_chunks=n_adaln_chunks, modulation_bias=modulation_bias, use_second_norm=False
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        inference_params=None,
        packed_seq_params=None,
        emb=None,
    ):
        residual = hidden_states

        shift, scale, gate = self.adaln(emb)

        norm_hidden_states = self.adaln.modulated_layernorm(hidden_states, shift=shift, scale=scale)

        mlp_hidden_states, mlp_bias = self.mlp(norm_hidden_states)

        attention_output = self.self_attention(
            norm_hidden_states, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb
        )

        hidden_states = mlp_hidden_states + mlp_bias + attention_output

        hidden_states = self.adaln.scale_add(residual, x=hidden_states, gate=gate)

        return hidden_states, None

    def __call__(self, *args, **kwargs):
        if hasattr(self, 'cudagraph_manager'):
            return self.cudagraph_manager(self, args, kwargs)
        return super(MegatronModule, self).__call__(*args, **kwargs)


# ----- Spec builders for Flux -----

def get_flux_single_transformer_engine_spec() -> ModuleSpec:
    return ModuleSpec(
        module=FluxSingleTransformerBlock,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=FluxSingleAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    q_layernorm=TENorm,
                    k_layernorm=TENorm,
                    linear_proj=TERowParallelLinear,
                ),
            ),
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
        ),
    )


def get_flux_double_transformer_engine_spec() -> ModuleSpec:
    return ModuleSpec(
        module=MMDiTLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=JointSelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=JointSelfAttentionSubmodules(
                    q_layernorm=TENorm,
                    k_layernorm=TENorm,
                    added_q_layernorm=TENorm,
                    added_k_layernorm=TENorm,
                    linear_qkv=TEColumnParallelLinear,
                    added_linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                ),
            ),
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
        ),
    )
