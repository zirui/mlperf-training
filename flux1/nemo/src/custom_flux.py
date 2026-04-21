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

from itertools import repeat
import logging
import os
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import default_collate
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import parallel_state
from nemo.collections.diffusion.models.flux.model import (
    Flux,
    FluxConfig,
    FluxModelParams,
    MegatronFluxModel,
)
from nemo.collections.diffusion.vae.autoencoder import AutoEncoder, AutoEncoderConfig
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
)
from nemo.lightning.megatron_parallel import MegatronLossReduction
from nemo.lightning.pytorch.optim import OptimizerModule

from custom_weight_inits import (
    mlpembedder_weight_init,
    init_single_block_weights,
    init_double_block_weights,
    linear_weight_init,
)

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _maybe_cudagraph_mark_step_begin():
    if not _env_flag("COMPILE_DIT", False):
        return
    if not _env_flag("COMPILE_DIT_MARK_STEP", True):
        return
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
        torch.compiler.cudagraph_mark_step_begin()


def _rank_zero() -> bool:
    return int(os.getenv("RANK", "0")) == 0


def _configure_compile_debugging() -> Dict[str, str]:
    if not _env_flag("COMPILE_DIT_DEBUG", False):
        return {}

    debug_settings: Dict[str, str] = {}
    os.environ.setdefault("TORCH_LOGS", "graph_breaks,recompiles")
    os.environ.setdefault("TORCHDYNAMO_VERBOSE", "1")
    debug_settings["TORCH_LOGS"] = os.environ["TORCH_LOGS"]
    debug_settings["TORCHDYNAMO_VERBOSE"] = os.environ["TORCHDYNAMO_VERBOSE"]

    try:
        import torch._dynamo.config as dynamo_config

        dynamo_config.verbose = True
        dynamo_config.suppress_errors = False
    except Exception as exc:  # pragma: no cover - best effort safeguard
        logger.warning("Failed to enable torch.compile debug observability: %s", exc)

    return debug_settings


def _format_compile_kwargs(compile_kwargs: Dict[str, object]) -> str:
    return ", ".join(f"{key}={value}" for key, value in compile_kwargs.items())


class CustomFlux(Flux):
    def __init__(self, config: FluxConfig, **kwargs):
        super().__init__(config, **kwargs)
        self._compile_strategy = "disabled"
        self._compile_kwargs: Dict[str, object] = {}
        self._compile_debug_settings: Dict[str, str] = {}
        self._compiled_regions: list[str] = []
        self._double_block_stack_runner: Callable = self._run_double_block_stack
        self._single_block_stack_runner: Callable = self._run_single_block_stack
        self._output_head_runner: Callable = self._run_output_head
        self._full_dit_runner: Callable = self._run_full_dit
        self._controlnet_fallback_logged = False
        self.init_weights()
        self.maybe_configure_dit_compile()

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

    def _build_compile_kwargs(self) -> Dict[str, object]:
        compile_kwargs: Dict[str, object] = {
            "backend": os.getenv("TORCH_COMPILE_BACKEND", "inductor"),
            "fullgraph": _env_flag("TORCH_COMPILE_FULLGRAPH", False),
            "dynamic": _env_flag("TORCH_COMPILE_DYNAMIC", False),
            "mode": os.getenv("TORCH_COMPILE_MODE", "default"),
        }
        return compile_kwargs

    def _compile_region(self, fn: Callable, region_name: str) -> Callable:
        self._compiled_regions.append(region_name)
        return torch.compile(fn, **self._compile_kwargs)

    def _log_compile_configuration(self):
        if not _rank_zero():
            return

        logger.info(
            "COMPILE_DIT enabled: strategy=%s, compiled_regions=%s, compile_kwargs=(%s), "
            "enable_cuda_graph=%s, cpu_offloading=%s, cpu_offloading_activations=%s",
            self._compile_strategy,
            self._compiled_regions or ["none"],
            _format_compile_kwargs(self._compile_kwargs),
            self.config.enable_cuda_graph,
            getattr(self.config, "cpu_offloading", None),
            getattr(self.config, "cpu_offloading_activations", None),
        )
        if self._compile_debug_settings:
            logger.info(
                "COMPILE_DIT debug observability active: %s",
                ", ".join(
                    f"{key}={value}" for key, value in self._compile_debug_settings.items()
                ),
            )

    def maybe_configure_dit_compile(self):
        self._double_block_stack_runner = self._run_double_block_stack
        self._single_block_stack_runner = self._run_single_block_stack
        self._output_head_runner = self._run_output_head
        self._full_dit_runner = self._run_full_dit
        self._compiled_regions = []
        self._compile_kwargs = {}
        self._compile_debug_settings = {}

        if not _env_flag("COMPILE_DIT", False):
            return

        if not hasattr(torch, "compile"):
            raise RuntimeError("COMPILE_DIT is set but torch.compile is unavailable")

        raw_strategy = os.getenv("COMPILE_DIT_STRATEGY", "stack")
        self._compile_strategy = raw_strategy.strip().lower()
        valid_strategies = {"per_block", "single_stack", "double_stack", "stack", "full_dit"}

        if self._compile_strategy not in valid_strategies:
            raise ValueError(
                f"Unsupported COMPILE_DIT_STRATEGY={raw_strategy!r}. "
                f"Expected one of {sorted(valid_strategies)}."
            )
        self._compile_debug_settings = _configure_compile_debugging()
        self._compile_kwargs = self._build_compile_kwargs()

        if self._compile_strategy == "per_block":
            for idx, layer in enumerate(self.double_blocks):
                self.double_blocks[idx] = torch.compile(layer, **self._compile_kwargs)
            for idx, layer in enumerate(self.single_blocks):
                self.single_blocks[idx] = torch.compile(layer, **self._compile_kwargs)
            self._compiled_regions.extend(
                [
                    f"double_blocks[{len(self.double_blocks)}]",
                    f"single_blocks[{len(self.single_blocks)}]",
                ]
            )
        elif self._compile_strategy == "single_stack":
            self._single_block_stack_runner = self._compile_region(
                self._run_single_block_stack, "single_block_stack"
            )
        elif self._compile_strategy == "double_stack":
            self._double_block_stack_runner = self._compile_region(
                self._run_double_block_stack, "double_block_stack"
            )
        elif self._compile_strategy == "stack":
            self._double_block_stack_runner = self._compile_region(
                self._run_double_block_stack, "double_block_stack"
            )
            self._single_block_stack_runner = self._compile_region(
                self._run_single_block_stack, "single_block_stack"
            )
        elif self._compile_strategy == "full_dit":
            self._full_dit_runner = self._compile_region(
                self._run_full_dit, "full_dit"
            )

        if _env_flag("COMPILE_DIT_OUTPUT_HEAD", False) and self._compile_strategy in {
            "single_stack",
            "double_stack",
            "stack",
        }:
            self._output_head_runner = self._compile_region(
                self._run_output_head, "output_head"
            )

        self._log_compile_configuration()

    def _run_double_block_stack(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        vec_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.double_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                emb=vec_emb,
            )
        return hidden_states, encoder_hidden_states

    def _run_single_block_stack(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        vec_emb: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.single_blocks:
            hidden_states, _ = block(
                hidden_states=hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                emb=vec_emb,
            )
        return hidden_states

    def _run_output_head(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        vec_emb: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states[encoder_hidden_states.shape[0] :, ...]
        hidden_states = self.norm_out(hidden_states, vec_emb)
        return self.proj_out(hidden_states)

    def _run_full_dit(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        vec_emb: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states, encoder_hidden_states = self._run_double_block_stack(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            vec_emb=vec_emb,
        )
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=0)
        hidden_states = self._run_single_block_stack(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            vec_emb=vec_emb,
        )
        return self._run_output_head(hidden_states, encoder_hidden_states, vec_emb)

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
        if self._compile_strategy in {"disabled", "per_block"}:
            return super().forward(
                img=img,
                txt=txt,
                y=y,
                timesteps=timesteps,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                controlnet_double_block_samples=controlnet_double_block_samples,
                controlnet_single_block_samples=controlnet_single_block_samples,
            )

        if (
            controlnet_double_block_samples is not None
            or controlnet_single_block_samples is not None
        ):
            if not self._controlnet_fallback_logged and _rank_zero():
                logger.info(
                    "ControlNet samples detected; falling back to eager Flux.forward for "
                    "COMPILE_DIT_STRATEGY=%s compatibility.",
                    self._compile_strategy,
                )
                self._controlnet_fallback_logged = True
            return super().forward(
                img=img,
                txt=txt,
                y=y,
                timesteps=timesteps,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                controlnet_double_block_samples=controlnet_double_block_samples,
                controlnet_single_block_samples=controlnet_single_block_samples,
            )

        hidden_states = self.img_embed(img)
        encoder_hidden_states = self.txt_embed(txt)

        timesteps = timesteps.to(img.dtype) * 1000
        vec_emb = self.timestep_embedding(timesteps)

        if guidance is not None:
            vec_emb = vec_emb + self.guidance_embedding(
                self.timestep_embedding.time_proj(guidance * 1000)
            )
        vec_emb = vec_emb + self.vector_embedding(y)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        rotary_pos_emb = self.pos_embed(ids)

        if self._compile_strategy == "full_dit":
            with self.get_fp8_context():
                return self._full_dit_runner(
                    hidden_states,
                    encoder_hidden_states,
                    rotary_pos_emb,
                    vec_emb,
                )

        with self.get_fp8_context():
            hidden_states, encoder_hidden_states = self._double_block_stack_runner(
                hidden_states,
                encoder_hidden_states,
                rotary_pos_emb,
                vec_emb,
            )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=0)

        with self.get_fp8_context():
            hidden_states = self._single_block_stack_runner(
                hidden_states,
                rotary_pos_emb,
                vec_emb,
            )

        return self._output_head_runner(hidden_states, encoder_hidden_states, vec_emb)


class CustomFluxConfig(FluxConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activation_func = torch.nn.functional.gelu

    def configure_model(self):
        model = CustomFlux(config=self)
        return model


class CustomMegatronFluxModel(MegatronFluxModel):
    """
    This class overrides a few important functionalities.

    Validation:
    - It overrides the validation_step to generate the timesteps for validation.
    - It overrides the validation_loss_reduction to accumulate the loss per timestep and the
        count of samples evaluated per timestep across dp ranks.
    - It overrides the reduce function to sum the losses and counts across all micro-batches.

    Training:
    - It allows the option for the loss to not be reduced.
    - It samples from mean and logvar to get the latents for the VAE.

    VAE:
    - Introduces the vae_scale and shift for sampling from mean and logvar.
    """

    def __init__(
        self,
        flux_params: FluxModelParams,
        empty_encodings_path: str,
        seed: int,
        classifier_free_guidance_prob: float = 0.1,
        always_zero_timestep: bool = False,
        optim: Optional[OptimizerModule] = None,
        generate_random_empty_encodings: bool = False,
    ):
        super().__init__(flux_params, optim)
        self.validation_timestep = 0
        self._validation_loss_reduction = None
        self._training_loss_reduction = None
        self.empty_encodings_path = empty_encodings_path
        self.classifier_free_guidance_prob = classifier_free_guidance_prob
        self.always_zero_timestep = always_zero_timestep
        self.seed = seed

        if generate_random_empty_encodings:
            logger.warning(
                "Generating random empty encodings. Results are not valid training."
            )
            empty_t5_encodings = torch.randn(256, 1, self.config.context_dim)
            empty_clip_encodings = torch.randn(self.config.vec_in_dim)
        else:
            empty_t5_encodings = torch.from_numpy(
                np.load(empty_encodings_path + "/t5_empty.npy")
            )[0].unsqueeze(1)
            empty_clip_encodings = torch.from_numpy(
                np.load(empty_encodings_path + "/clip_empty.npy")
            )[0]
        self.register_buffer("empty_t5_encodings", empty_t5_encodings)
        self.register_buffer("empty_clip_encodings", empty_clip_encodings)

    def _unpack_latents(self, latents, height, width):
        # pylint: disable=C0116
        batch_size, num_patches, channels = latents.shape

        # adjust h and w for patching
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // 4, height, width)

        return latents

    def configure_vae(self, vae):
        # pylint: disable=C0116
        if isinstance(vae, nn.Module):
            self.vae = vae.eval().cuda()
            self.vae_scale_factor = 2 ** (len(self.vae.params.ch_mult) - 1)
            self.vae_channels = self.vae.params.z_channels
            for param in self.vae.parameters():
                param.requires_grad = False
        elif isinstance(vae, AutoEncoderConfig):
            self.vae = AutoEncoder(vae).eval().cuda()
            self.vae_scale_factor = 2 ** (len(vae.ch_mult) - 1)
            self.vae_channels = vae.z_channels
            for param in self.vae.parameters():
                param.requires_grad = False
        else:
            logging.info("Vae not provided, assuming the image input is precached...")
            self.vae = None
            self.vae_scale_factor = 8
            self.vae_channels = 16
            self.vae_scale = 0.3611
            self.vae_shift = 0.1159

    def setup(self, stage=None):
        super().setup(stage)
        torch.manual_seed(self.seed + 100 * parallel_state.get_data_parallel_rank())

    def get_synthetic_batch(self, micro_batch_size: int):
        self.init_global_step = self.trainer.global_step
        batch = default_collate(
            [
                {
                    "mean": torch.randn(
                        self.vae_channels,
                        256 // self.vae_scale_factor,
                        256 // self.vae_scale_factor,
                    ),
                    "logvar": torch.randn(
                        self.vae_channels,
                        256 // self.vae_scale_factor,
                        256 // self.vae_scale_factor,
                    ),
                    "prompt_embeds": torch.randn(256, self.config.context_dim),
                    "pooled_prompt_embeds": torch.randn(self.config.vec_in_dim),
                    "text_ids": torch.zeros(256, 3),
                    "timestep": torch.randint(0, 8, ()),
                }
            ]
            * micro_batch_size
        )
        return repeat(batch)

    def prepare_image_latent_like_reference(
        self, latents: torch.Tensor, timesteps: torch.Tensor | None = None
    ):
        """
        Unlike the paper, the reference:
        - samples timesteps uniformly
        - does not use a schedule for the noise
        """
        noise = torch.randn_like(latents)
        if timesteps is None:
            if self.always_zero_timestep:
                timesteps = torch.zeros(
                    (latents.shape[0],), device=latents.device, dtype=latents.dtype
                )
            else:
                timesteps = torch.rand(
                    (latents.shape[0],), device=latents.device, dtype=latents.dtype
                )
        else:
            timesteps = timesteps.to(latents, non_blocking=True)
        sigmas = timesteps.view(-1, 1, 1, 1)
        noisy_model_input = (1 - sigmas) * latents + sigmas * noise
        packed_noisy_model_input = self._pack_latents(
            noisy_model_input,
            batch_size=latents.shape[0],
            num_channels_latents=latents.shape[1],
            height=latents.shape[2],
            width=latents.shape[3],
        )
        latent_image_ids = self._prepare_latent_image_ids(
            1,
            latents.shape[2],
            latents.shape[3],
            latents.device,
            latents.dtype,
        )
        guidance_vec = None

        return (
            latents.transpose(0, 1),
            packed_noisy_model_input.transpose(0, 1),
            latent_image_ids,
            noise.transpose(0, 1),
            guidance_vec,
            timesteps,
        )

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)

        # the above represent the indices of the timesteps. They correspond to evenly spaced values between 0 and 1000
        timestep_values = (
            batch["timestep"].to(
                device="cuda", dtype=self.autocast_dtype, non_blocking=True
            )
            / 8.0
        )
        loss = self.forward_step(batch, timesteps=timestep_values, reduction="none")
        loss_per_sample = loss.mean(dim=(1, 2, 3))
        loss_sum = loss_per_sample.sum()
        sample_count = torch.empty(1, device="cuda", dtype=self.autocast_dtype).fill_(
            loss_per_sample.numel()
        )
        return loss_sum, sample_count

    def forward_step(self, batch, timesteps=None, reduction="mean") -> torch.Tensor:
        # pylint: disable=C0116
        if self.optim.config.bf16:
            self.autocast_dtype = torch.bfloat16
        elif self.optim.config.fp16:
            self.autocast_dtype = torch.float
        else:
            self.autocast_dtype = torch.float32

        if self.image_precached:
            mean = batch["mean"].to(
                device="cuda", dtype=self.autocast_dtype, non_blocking=True
            )
            logvar = batch["logvar"].to(
                device="cuda", dtype=self.autocast_dtype, non_blocking=True
            )
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(mean, device=mean.device, dtype=mean.dtype) * std
            latents = mean + eps
            latents = self.vae_scale * (latents - self.vae_shift)
        else:
            img = batch["images"].to(
                device="cuda", dtype=self.autocast_dtype, non_blocking=True
            )
            latents = self.vae.encode(img)

        with torch.no_grad():
            (
                latents,
                packed_noisy_model_input,
                latent_image_ids,
                noise,
                guidance_vec,
                timesteps,
            ) = self.prepare_image_latent_like_reference(latents, timesteps=timesteps)

        if self.text_precached:
            prompt_embeds = (
                batch["prompt_embeds"]
                .to(dtype=self.autocast_dtype, device="cuda", non_blocking=True)
                .transpose(0, 1)
            )
            pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(
                dtype=self.autocast_dtype, device="cuda", non_blocking=True
            )
            text_ids = torch.zeros(
                (1, prompt_embeds.shape[0], 3),
                device=prompt_embeds.device,
                dtype=prompt_embeds.dtype,
            )
        else:
            txt = batch["txt"]
            prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
                txt, device=latents.device, dtype=self.autocast_dtype
            )

        dropout_mask = (
            torch.rand(pooled_prompt_embeds.shape[0], device="cuda")
            < self.classifier_free_guidance_prob
        )

        empty_encodings = self.empty_t5_encodings.to(
            device="cuda", dtype=prompt_embeds.dtype, non_blocking=True
        )
        expanded_mask = dropout_mask.view(1, -1, 1).expand_as(prompt_embeds)
        expanded_empty = empty_encodings.expand_as(prompt_embeds)
        prompt_embeds = torch.where(expanded_mask, expanded_empty, prompt_embeds)

        empty_clip = self.empty_clip_encodings.to(
            device="cuda", dtype=pooled_prompt_embeds.dtype, non_blocking=True
        )
        expanded_mask = dropout_mask.view(-1, 1).expand_as(pooled_prompt_embeds)
        expanded_empty = empty_clip.expand_as(pooled_prompt_embeds)
        pooled_prompt_embeds = torch.where(
            expanded_mask, expanded_empty, pooled_prompt_embeds
        )

        with torch.cuda.amp.autocast(
            self.autocast_dtype in (torch.half, torch.bfloat16),
            dtype=self.autocast_dtype,
        ):
            _maybe_cudagraph_mark_step_begin()
            noise_pred = self.forward(
                img=packed_noisy_model_input,
                txt=prompt_embeds,
                y=pooled_prompt_embeds,
                timesteps=timesteps,
                img_ids=latent_image_ids,
                txt_ids=text_ids,
                guidance=guidance_vec,
            )
            noise_pred = self._unpack_latents(
                noise_pred.transpose(0, 1),
                latents.shape[2],
                latents.shape[3],
            )
            target = noise.transpose(0, 1) - latents.transpose(0, 1)
            loss = F.mse_loss(noise_pred.float(), target.float(), reduction=reduction)
        return loss

    @property
    def training_loss_reduction(self):
        if not self._training_loss_reduction:
            self._training_loss_reduction = FluxTrainLossReduction()

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self):
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = FluxValLossReduction()

        return self._validation_loss_reduction


class FluxTrainLossReduction(MegatronLossReduction):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, batch: Dict[str, torch.Tensor], forward_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

        # averaged_loss = average_losses_across_data_parallel_group([forward_out])
        return forward_out, forward_out

    def reduce(self, losses_reduced_per_micro_batch) -> torch.Tensor:
        return torch.stack(losses_reduced_per_micro_batch).mean(dim=0)


class FluxValLossReduction(MegatronLossReduction):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, batch: Dict[str, torch.Tensor], forward_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        This function receives the output of the validation_step defined above.
        It accumulates the loss sum and the count of samples across dp ranks.
        The return value is defined as such in order to work with Megatron's loss calculation.
        The first value is a dummy value, which is not used in validation passes.
        The final values are what will be passed on to the reduce function in this class.
        """
        loss_sum, sample_count = forward_out
        dist.all_reduce(loss_sum, group=parallel_state.get_data_parallel_group())
        dist.all_reduce(sample_count, group=parallel_state.get_data_parallel_group())
        return torch.ones(1), (loss_sum, sample_count)

    def reduce(self, losses_reduced_per_micro_batch) -> torch.Tensor:
        """
        This function receives the output of the forward function for all micro-batches and reduces them.

        losses_reduced_per_micro_batch is a tuple with:
        - loss_sum: a tensor of shape (1,)
        - sample_count: a tensor of shape (1,)

        We need to sum the losses and counts across all micro-batches.
        """
        loss, count = zip(*losses_reduced_per_micro_batch)
        return torch.stack(loss).sum(dim=0), torch.stack(count).sum(dim=0)
