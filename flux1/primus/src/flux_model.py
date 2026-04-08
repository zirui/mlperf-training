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
# Extracted from NeMo v2.6.0: nemo/collections/diffusion/models/flux/model.py
# Dependencies: Megatron-core + Transformer Engine + PyTorch only (no NeMo).
# Includes CustomFlux and CustomFluxConfig inlined from flux1/nemo/src/custom_flux.py.

import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Optional

import numpy as np
import torch
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import openai_gelu, sharded_state_dict_default
from torch import nn

from dit_layer_spec import (
    AdaLNContinuous,
    FluxSingleTransformerBlock,
    MMDiTLayer,
    get_flux_double_transformer_engine_spec,
    get_flux_single_transformer_engine_spec,
)
from flux_layers import EmbedND, MLPEmbedder, TimeStepEmbedder
from custom_weight_inits import (
    mlpembedder_weight_init,
    init_single_block_weights,
    init_double_block_weights,
    linear_weight_init,
)

logger = logging.getLogger(__name__)


def flux_data_step(dataloader_iter):
    batch = next(dataloader_iter)
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch
    _batch['loss_mask'] = torch.ones(1, device="cuda")
    return _batch


@dataclass
class FluxConfig(TransformerConfig):
    """Transformer-related Flux configuration."""

    num_layers: int = 1  # dummy setting
    num_joint_layers: int = 19
    num_single_layers: int = 38
    hidden_size: int = 3072
    num_attention_heads: int = 24
    activation_func: Callable = openai_gelu
    normalization: str = 'RMSNorm'
    add_qkv_bias: bool = True
    in_channels: int = 64
    context_dim: int = 4096
    model_channels: int = 256
    patch_size: int = 1
    guidance_embed: bool = False
    vec_in_dim: int = 768
    rotary_interleaved: bool = True
    apply_rope_fusion: bool = False
    layernorm_epsilon: float = 1e-06
    hidden_dropout: float = 0
    attention_dropout: float = 0
    use_cpu_initialization: bool = True
    gradient_accumulation_fusion: bool = False
    enable_cuda_graph: bool = False
    cuda_graph_scope: Optional[str] = None
    use_te_rng_tracker: bool = False
    cuda_graph_warmup_steps: int = 2

    guidance_scale: float = 3.5
    data_step_fn: Callable = flux_data_step
    ckpt_path: Optional[str] = None
    load_dist_ckpt: bool = False
    do_convert_from_hf: bool = False
    save_converted_model_to: Optional[str] = None

    def configure_model(self):
        model = Flux(config=self)
        return model


@dataclass
class FluxModelParams:
    """Minimal Flux model params (VAE/T5/CLIP encoders removed -- we use pre-cached data)."""

    flux_config: FluxConfig = field(default_factory=FluxConfig)
    scheduler_steps: int = 1000
    device: str = 'cuda'


class Flux(VisionModule):
    """
    Standalone Flux DiT model using Megatron-core blocks.

    Extracted from NeMo v2.6.0 with NeMo-specific mixins removed.
    """

    def __init__(self, config: FluxConfig):
        super().__init__(config)
        self.input_tensor = None
        self.out_channels = config.in_channels
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.guidance_embed = config.guidance_embed
        self._use_activation_checkpointing = (
            os.environ.get("ACTIVATION_CHECKPOINTING", "False").lower() == "true"
        )
        self._checkpoint_every_n = int(os.environ.get("CHECKPOINT_EVERY_N", "1"))

        if self._use_activation_checkpointing:
            from transformer_engine.pytorch.distributed import checkpoint as te_checkpoint
            self._te_checkpoint = te_checkpoint
            if self._checkpoint_every_n > 1:
                logger.info(
                    f"Selective activation checkpointing ENABLED — every {self._checkpoint_every_n}th block"
                )
            else:
                logger.info("Full activation checkpointing ENABLED (all blocks)")

        self.pos_embed = EmbedND(dim=self.hidden_size, theta=10000, axes_dim=[16, 56, 56])
        self.img_embed = nn.Linear(config.in_channels, self.hidden_size)
        self.txt_embed = nn.Linear(config.context_dim, self.hidden_size)
        self.timestep_embedding = TimeStepEmbedder(config.model_channels, self.hidden_size)
        self.vector_embedding = MLPEmbedder(in_dim=config.vec_in_dim, hidden_dim=self.hidden_size)
        if config.guidance_embed:
            self.guidance_embedding = (
                MLPEmbedder(in_dim=config.model_channels, hidden_dim=self.hidden_size)
                if config.guidance_embed
                else nn.Identity()
            )

        self.double_blocks = nn.ModuleList(
            [
                MMDiTLayer(
                    config=config,
                    submodules=get_flux_double_transformer_engine_spec().submodules,
                    layer_number=i,
                    context_pre_only=False,
                )
                for i in range(config.num_joint_layers)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    config=config,
                    submodules=get_flux_single_transformer_engine_spec().submodules,
                    layer_number=i,
                )
                for i in range(config.num_single_layers)
            ]
        )

        self.norm_out = AdaLNContinuous(config=config, conditioning_embedding_dim=self.hidden_size)
        self.proj_out = nn.Linear(self.hidden_size, self.patch_size * self.patch_size * self.out_channels, bias=True)
        if self.config.ckpt_path is not None:
            self.load_from_pretrained(
                self.config.ckpt_path,
                do_convert_from_hf=self.config.do_convert_from_hf,
                load_dist_ckpt=self.config.load_dist_ckpt,
                save_converted_model_to=self.config.save_converted_model_to,
            )

    def set_input_tensor(self, input_tensor):
        """Required by Megatron's pipeline-parallel scheduler (even when PP=1)."""
        self.input_tensor = input_tensor

    def get_fp8_context(self):
        if not self.config.fp8:
            fp8_context = nullcontext()
        else:
            from transformer_engine.common.recipe import (
                DelayedScaling,
                Float8BlockScaling,
                Float8CurrentScaling,
                Format,
                MXFP8BlockScaling,
            )
            from transformer_engine.pytorch import fp8_autocast

            if self.config.fp8 == "e4m3":
                fp8_format = Format.E4M3
            elif self.config.fp8 == "hybrid":
                fp8_format = Format.HYBRID
            else:
                raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

            if not self.config.fp8_recipe:
                self.config.fp8_recipe = "delayed"

            if self.config.fp8_recipe == "delayed":
                fp8_recipe = DelayedScaling(
                    margin=self.config.fp8_margin,
                    interval=self.config.fp8_interval,
                    fp8_format=fp8_format,
                    amax_compute_algo=self.config.fp8_amax_compute_algo,
                    amax_history_len=self.config.fp8_amax_history_len,
                    override_linear_precision=(False, False, not self.config.fp8_wgrad),
                )
            elif self.config.fp8_recipe == "current":
                fp8_recipe = Float8CurrentScaling(fp8_format=fp8_format)
            elif self.config.fp8_recipe == "block":
                fp8_recipe = Float8BlockScaling(fp8_format=fp8_format)
            elif self.config.fp8_recipe == "mxfp8":
                fp8_recipe = MXFP8BlockScaling(fp8_format=fp8_format)
            else:
                raise ValueError(f"Unsupported FP8 recipe: {self.config.fp8_recipe}")

            fp8_group = None
            if parallel_state.model_parallel_is_initialized():
                fp8_group = parallel_state.get_amax_reduction_group(with_context_parallel=True)
            fp8_context = fp8_autocast(enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group)
        return fp8_context

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor = None,
        y: torch.Tensor = None,
        timesteps: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        controlnet_double_block_samples: torch.Tensor = None,
        controlnet_single_block_samples: torch.Tensor = None,
    ):
        hidden_states = self.img_embed(img)
        encoder_hidden_states = self.txt_embed(txt)

        timesteps = timesteps.to(img.dtype) * 1000
        vec_emb = self.timestep_embedding(timesteps)

        if guidance is not None:
            vec_emb = vec_emb + self.guidance_embedding(self.timestep_embedding.time_proj(guidance * 1000))
        vec_emb = vec_emb + self.vector_embedding(y)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        rotary_pos_emb = self.pos_embed(ids)

        hidden_states, encoder_hidden_states = self._forward_with_optional_ckpt(
            hidden_states, encoder_hidden_states, rotary_pos_emb, vec_emb,
            controlnet_double_block_samples, controlnet_single_block_samples,
        )

        hidden_states = hidden_states[encoder_hidden_states.shape[0] :, ...]

        hidden_states = self.norm_out(hidden_states, vec_emb)
        output = self.proj_out(hidden_states)

        return output

    def _forward_with_optional_ckpt(
        self, hidden_states, encoder_hidden_states, rotary_pos_emb, vec_emb,
        controlnet_double_block_samples, controlnet_single_block_samples,
    ):
        for id_block, block in enumerate(self.double_blocks):
            use_ckpt = (
                self._use_activation_checkpointing
                and id_block % self._checkpoint_every_n == 0
            )
            if use_ckpt:
                with self.get_fp8_context():
                    hidden_states, encoder_hidden_states = self._te_checkpoint(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        rotary_pos_emb=rotary_pos_emb,
                        emb=vec_emb,
                        use_reentrant=False,
                    )
            else:
                with self.get_fp8_context():
                    hidden_states, encoder_hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        rotary_pos_emb=rotary_pos_emb,
                        emb=vec_emb,
                    )

            if controlnet_double_block_samples is not None:
                interval_control = len(self.double_blocks) / len(controlnet_double_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_double_block_samples[id_block // interval_control]

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=0)

        for id_block, block in enumerate(self.single_blocks):
            use_ckpt = (
                self._use_activation_checkpointing
                and id_block % self._checkpoint_every_n == 0
            )
            if use_ckpt:
                with self.get_fp8_context():
                    hidden_states, _ = self._te_checkpoint(
                        block,
                        hidden_states,
                        rotary_pos_emb=rotary_pos_emb,
                        emb=vec_emb,
                        use_reentrant=False,
                    )
            else:
                with self.get_fp8_context():
                    hidden_states, _ = block(
                        hidden_states=hidden_states,
                        rotary_pos_emb=rotary_pos_emb,
                        emb=vec_emb,
                    )

            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = torch.cat(
                    [
                        hidden_states[: encoder_hidden_states.shape[0]],
                        hidden_states[encoder_hidden_states.shape[0] :]
                        + controlnet_single_block_samples[id_block // interval_control],
                    ]
                )

        return hidden_states, encoder_hidden_states

    def load_from_pretrained(
        self, ckpt_path, do_convert_from_hf=False, save_converted_model_to=None, load_dist_ckpt=False
    ):
        if load_dist_ckpt:
            from megatron.core import dist_checkpointing

            sharded_sd_metadata = dist_checkpointing.load_content_metadata(ckpt_path)
            if sharded_sd_metadata is None:
                sharded_sd_metadata = {}
            sharded_state_dict = dict(
                state_dict=self.sharded_state_dict(prefix="module.", metadata=sharded_sd_metadata)
            )
            loaded_state_dict = dist_checkpointing.load(
                sharded_state_dict=sharded_state_dict, checkpoint_dir=ckpt_path
            )
            ckpt = {k.removeprefix("module."): v for k, v in loaded_state_dict["state_dict"].items()}
        else:
            from safetensors.torch import load_file as load_safetensors
            ckpt = load_safetensors(ckpt_path)
        missing, unexpected = self.load_state_dict(ckpt, strict=False)
        missing = [k for k in missing if not k.endswith('_extra_state')]
        if len(missing) > 0:
            logger.info(
                f"The following keys are missing during checkpoint loading, "
                f"please check the ckpt provided or the image quality may be compromised.\n {missing}"
            )
        logger.info(f"Found unexpected keys: \n {unexpected}")
        logger.info(f"Restored flux model weights from {ckpt_path}")

    def sharded_state_dict(self, prefix='', sharded_offsets: tuple = (), metadata: dict = None) -> ShardedStateDict:
        sharded_state_dict = {}
        layer_prefix = f'{prefix}double_blocks.'
        for layer in self.double_blocks:
            offset = layer._get_layer_offset(self.config)
            global_layer_offset = layer.layer_number
            state_dict_prefix = f'{layer_prefix}{global_layer_offset - offset}.'
            sharded_prefix = f'{layer_prefix}{global_layer_offset}.'
            sharded_pp_offset = []
            layer_sharded_state_dict = layer.sharded_state_dict(state_dict_prefix, sharded_pp_offset, metadata)
            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)
            sharded_state_dict.update(layer_sharded_state_dict)

        layer_prefix = f'{prefix}single_blocks.'
        for layer in self.single_blocks:
            offset = layer._get_layer_offset(self.config)
            global_layer_offset = layer.layer_number
            state_dict_prefix = f'{layer_prefix}{global_layer_offset - offset}.'
            sharded_prefix = f'{layer_prefix}{global_layer_offset}.'
            sharded_pp_offset = []
            layer_sharded_state_dict = layer.sharded_state_dict(state_dict_prefix, sharded_pp_offset, metadata)
            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)
            sharded_state_dict.update(layer_sharded_state_dict)

        for name, module in self.named_children():
            if not (module is self.single_blocks or module is self.double_blocks):
                sharded_state_dict.update(
                    sharded_state_dict_default(module, f'{prefix}{name}.', sharded_offsets, metadata)
                )
        return sharded_state_dict


# ---- CustomFlux / CustomFluxConfig (inlined from flux1/nemo/src/custom_flux.py) ----


class CustomFlux(Flux):
    """Flux with custom weight initialization for MLPerf training."""

    def __init__(self, config: FluxConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.init_weights()

    def init_weights(self):
        linear_weight_init(self.img_embed)
        linear_weight_init(self.txt_embed)

        mlpembedder_weight_init(self.timestep_embedding.time_embedder)
        mlpembedder_weight_init(self.vector_embedding)

        for layer in self.single_blocks:
            init_single_block_weights(layer)
        for layer in self.double_blocks:
            init_double_block_weights(layer)

        nn.init.constant_(self.norm_out.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.norm_out.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)
        self.norm_out.norm.reset_parameters()


class CustomFluxConfig(FluxConfig):
    """FluxConfig that uses GELU and instantiates CustomFlux."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activation_func = torch.nn.functional.gelu

    def configure_model(self):
        model = CustomFlux(config=self)
        return model


# ---- Utility functions (extracted from MegatronFluxModel) ----

@lru_cache
def prepare_latent_image_ids(
    batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype
):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
    latent_image_ids = latent_image_ids.reshape(
        batch_size, latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype, non_blocking=True)


def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


def unpack_latents(latents, height, width):
    batch_size, num_patches, channels = latents.shape
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // 4, height, width)
    return latents
