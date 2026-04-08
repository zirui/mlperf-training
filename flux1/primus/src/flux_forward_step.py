# Copyright (c) 2025, NVIDIA CORPORATION & AMD. All rights reserved.
#
# Rectified-flow forward step for Flux training.
# Ported from flux1/nemo/src/custom_flux.py (CustomMegatronFluxModel.forward_step).
# Standalone -- no NeMo dependencies.
#
# Handles both training and validation via data-driven branching:
#   - Training batches: random timesteps, CFG dropout
#   - Validation batches: carry "timestep" key (int 0-7), no CFG dropout

import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from flux_model import pack_latents, unpack_latents, prepare_latent_image_ids

logger = logging.getLogger(__name__)

VAE_SCALE = 0.3611
VAE_SHIFT = 0.1159


class FluxForwardContext:
    """
    Holds state shared across forward_step calls: empty encodings for CFG dropout,
    autocast dtype, and configuration flags.

    Uses lazy initialization for CUDA tensors to avoid placing them on the wrong
    device before Megatron's initialize_megatron() calls torch.cuda.set_device().
    """

    def __init__(
        self,
        empty_encodings_path: str,
        context_dim: int = 4096,
        vec_in_dim: int = 768,
        classifier_free_guidance_prob: float = 0.1,
        autocast_dtype: torch.dtype = torch.bfloat16,
        generate_random_empty_encodings: bool = False,
    ):
        self.empty_encodings_path = empty_encodings_path
        self.context_dim = context_dim
        self.vec_in_dim = vec_in_dim
        self.classifier_free_guidance_prob = classifier_free_guidance_prob
        self.autocast_dtype = autocast_dtype
        self.generate_random_empty_encodings = generate_random_empty_encodings

        self._empty_t5_encodings = None
        self._empty_clip_encodings = None

    @property
    def empty_t5_encodings(self):
        if self._empty_t5_encodings is None:
            self._load_encodings()
        return self._empty_t5_encodings

    @property
    def empty_clip_encodings(self):
        if self._empty_clip_encodings is None:
            self._load_encodings()
        return self._empty_clip_encodings

    def _load_encodings(self):
        if self.generate_random_empty_encodings:
            logger.warning("Generating random empty encodings -- NOT valid for training.")
            self._empty_t5_encodings = torch.randn(256, 1, self.context_dim).cuda()
            self._empty_clip_encodings = torch.randn(self.vec_in_dim).cuda()
        else:
            self._empty_t5_encodings = torch.from_numpy(
                np.load(os.path.join(self.empty_encodings_path, "t5_empty.npy"))
            )[0].unsqueeze(1).cuda()
            self._empty_clip_encodings = torch.from_numpy(
                np.load(os.path.join(self.empty_encodings_path, "clip_empty.npy"))
            )[0].cuda()
        logger.info("Loaded empty encodings for CFG dropout (device=%s)", self._empty_t5_encodings.device)


def prepare_image_latent_like_reference(latents, timesteps=None, always_zero_timestep=False):
    """
    Unlike the paper, the MLPerf reference:
    - Samples timesteps uniformly in [0, 1)
    - Does not use a schedule for the noise
    """
    noise = torch.randn_like(latents)
    if timesteps is None:
        if always_zero_timestep:
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
    packed_noisy_model_input = pack_latents(
        noisy_model_input,
        batch_size=latents.shape[0],
        num_channels_latents=latents.shape[1],
        height=latents.shape[2],
        width=latents.shape[3],
    )
    latent_image_ids = prepare_latent_image_ids(
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


def _sample_latents(batch, ctx):
    """Decode VAE latents from cached mean/logvar."""
    mean = batch["mean"].to(device="cuda", dtype=ctx.autocast_dtype, non_blocking=True)
    logvar = batch["logvar"].to(device="cuda", dtype=ctx.autocast_dtype, non_blocking=True)
    std = torch.exp(0.5 * logvar)
    latents = mean + torch.randn_like(mean) * std
    return VAE_SCALE * (latents - VAE_SHIFT)


def _prepare_text_embeds(batch, ctx):
    """Move text embeddings to GPU and transpose to (seq, batch, dim)."""
    prompt_embeds = (
        batch["prompt_embeds"]
        .to(dtype=ctx.autocast_dtype, device="cuda", non_blocking=True)
        .transpose(0, 1)
    )
    pooled = batch["pooled_prompt_embeds"].to(
        dtype=ctx.autocast_dtype, device="cuda", non_blocking=True
    )
    text_ids = torch.zeros(
        (1, prompt_embeds.shape[0], 3),
        device=prompt_embeds.device,
        dtype=prompt_embeds.dtype,
    )
    return prompt_embeds, pooled, text_ids


def _apply_cfg_dropout(prompt_embeds, pooled, ctx):
    """Apply classifier-free guidance dropout: replace some samples with empty encodings."""
    dropout_mask = (
        torch.rand(pooled.shape[0], device="cuda")
        < ctx.classifier_free_guidance_prob
    )

    empty_t5 = ctx.empty_t5_encodings.to(dtype=prompt_embeds.dtype, non_blocking=True)
    expanded_mask = dropout_mask.view(1, -1, 1).expand_as(prompt_embeds)
    prompt_embeds = torch.where(expanded_mask, empty_t5.expand_as(prompt_embeds), prompt_embeds)

    empty_clip = ctx.empty_clip_encodings.to(dtype=pooled.dtype, non_blocking=True)
    expanded_mask = dropout_mask.view(-1, 1).expand_as(pooled)
    pooled = torch.where(expanded_mask, empty_clip.expand_as(pooled), pooled)

    return prompt_embeds, pooled


_debug_forward_count = 0


def _run_model_forward(model, packed_input, prompt_embeds, pooled, timesteps,
                       img_ids, text_ids, guidance_vec, latents_t, noise_t, ctx,
                       reduction="mean"):
    """Shared model forward + MSE loss computation."""
    global _debug_forward_count
    _debug_forward_count += 1
    is_first = (_debug_forward_count == 1)

    with torch.amp.autocast(
        "cuda",
        enabled=ctx.autocast_dtype in (torch.half, torch.bfloat16),
        dtype=ctx.autocast_dtype,
    ):
        noise_pred = model(
            img=packed_input,
            txt=prompt_embeds,
            y=pooled,
            timesteps=timesteps,
            img_ids=img_ids,
            txt_ids=text_ids,
            guidance=guidance_vec,
        )

        if is_first:
            _log_first_forward_diagnostics(model, noise_pred, latents_t, noise_t,
                                           packed_input, prompt_embeds, timesteps)

        noise_pred = unpack_latents(
            noise_pred.transpose(0, 1),
            latents_t.shape[2],
            latents_t.shape[3],
        )
        target = noise_t.transpose(0, 1) - latents_t.transpose(0, 1)
        loss = F.mse_loss(noise_pred.float(), target.float(), reduction=reduction)

        if is_first:
            _log_loss_diagnostics(noise_pred, target, loss, reduction)

    return loss


def _log_first_forward_diagnostics(model, raw_noise_pred, latents_t, noise_t,
                                   packed_input, prompt_embeds, timesteps):
    """One-time diagnostic dump on the very first forward pass (rank 0 only)."""
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank != 0:
        return

    def _stats(name, t):
        t_f = t.float()
        return (f"  {name}: shape={list(t.shape)}, dtype={t.dtype}, "
                f"mean={t_f.mean().item():.6f}, std={t_f.std().item():.6f}, "
                f"min={t_f.min().item():.6f}, max={t_f.max().item():.6f}")

    lines = [
        "=" * 70,
        "DEBUG: First forward pass diagnostics",
        "=" * 70,
        _stats("packed_input (img)", packed_input),
        _stats("prompt_embeds (txt)", prompt_embeds),
        _stats("timesteps", timesteps),
        _stats("latents_t (C,B,H,W)", latents_t),
        _stats("noise_t (C,B,H,W)", noise_t),
        _stats("raw_model_output (before unpack)", raw_noise_pred),
    ]

    proj_out = getattr(model, "proj_out", None)
    if proj_out is not None:
        w = proj_out.weight
        b = proj_out.bias
        lines.append(f"  proj_out.weight: norm={w.float().norm().item():.8f}, "
                      f"max_abs={w.float().abs().max().item():.8f}")
        if b is not None:
            lines.append(f"  proj_out.bias:   norm={b.float().norm().item():.8f}, "
                          f"max_abs={b.float().abs().max().item():.8f}")

    fp8_val = getattr(getattr(model, "config", None), "fp8", "N/A")
    norm_val = getattr(getattr(model, "config", None), "normalization", "N/A")
    lines.append(f"  config.fp8={fp8_val}, config.normalization={norm_val}")
    lines.append("=" * 70)

    print("\n".join(lines), flush=True)


def _log_loss_diagnostics(noise_pred, target, loss, reduction):
    """Log loss-related diagnostics on the first forward pass (rank 0 only)."""
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank != 0:
        return

    per_elem_sq = (noise_pred.float() - target.float()).pow(2)
    lines = [
        "--- Loss diagnostics ---",
        f"  noise_pred (unpacked): shape={list(noise_pred.shape)}, "
        f"mean={noise_pred.float().mean().item():.6f}, std={noise_pred.float().std().item():.6f}",
        f"  target:                shape={list(target.shape)}, "
        f"mean={target.float().mean().item():.6f}, std={target.float().std().item():.6f}",
        f"  per-element MSE:       mean={per_elem_sq.mean().item():.6f}, "
        f"max={per_elem_sq.max().item():.6f}",
        f"  E[noise_pred^2]={noise_pred.float().pow(2).mean().item():.6f}",
        f"  E[target^2]={target.float().pow(2).mean().item():.6f}",
        f"  loss (reduction={reduction}): {loss.item():.6f}",
        "------------------------",
    ]
    print("\n".join(lines), flush=True)


def _training_forward(batch, model, ctx):
    """Training path: random timesteps, CFG dropout."""
    latents = _sample_latents(batch, ctx)

    with torch.no_grad():
        (latents_t, packed, img_ids, noise_t, guidance_vec, timesteps) = \
            prepare_image_latent_like_reference(latents)

    prompt_embeds, pooled, text_ids = _prepare_text_embeds(batch, ctx)
    prompt_embeds, pooled = _apply_cfg_dropout(prompt_embeds, pooled, ctx)

    loss = _run_model_forward(
        model, packed, prompt_embeds, pooled, timesteps,
        img_ids, text_ids, guidance_vec, latents_t, noise_t, ctx,
    )
    return loss


def _validation_forward(batch, model, ctx):
    """Validation path: fixed timesteps from dataset (int 0-7 / 8.0), no CFG dropout.

    Returns (loss_sum, sample_count) matching NeMo's FluxValLossReduction:
      - loss_sum: sum of per-sample MSE losses (mean over C,H,W per sample, then sum over batch)
      - sample_count: number of samples in this micro-batch
    """
    latents = _sample_latents(batch, ctx)
    timesteps = batch["timestep"].to(device="cuda", dtype=ctx.autocast_dtype) / 8.0

    with torch.no_grad():
        (latents_t, packed, img_ids, noise_t, guidance_vec, _) = \
            prepare_image_latent_like_reference(latents, timesteps=timesteps)

    prompt_embeds, pooled, text_ids = _prepare_text_embeds(batch, ctx)

    loss = _run_model_forward(
        model, packed, prompt_embeds, pooled, timesteps,
        img_ids, text_ids, guidance_vec, latents_t, noise_t, ctx,
        reduction="none",
    )
    loss_per_sample = loss.mean(dim=(1, 2, 3))
    loss_sum = loss_per_sample.sum()
    sample_count = torch.empty(
        1, device=loss_sum.device, dtype=loss_sum.dtype
    ).fill_(loss_per_sample.numel())
    return loss_sum, sample_count


def make_forward_step_func(ctx: FluxForwardContext):
    """
    Returns a Megatron-compatible forward_step_func closure.

    Branches on whether the batch contains a "timestep" key:
    - Present  -> validation (fixed timesteps, no CFG dropout)
    - Absent   -> training (random timesteps, CFG dropout)

    Both paths return (output_tensor, loss_func) with the same signature
    expected by Megatron's forward-backward scheduler.

    For validation, loss_func propagates loss_sum and sample_count through
    Megatron's loss dict so the eval hook can compute a properly weighted
    global average (matching NeMo's FluxValLossReduction).
    """

    def forward_step_func(data_iterator, model):
        batch = next(data_iterator)

        is_validation = "timestep" in batch

        if is_validation:
            loss_sum, sample_count = _validation_forward(batch, model, ctx)

            def loss_func(output_tensor):
                avg = output_tensor / sample_count
                # 2-element tensor [numerator, denominator] triggers Megatron's
                # evaluate() numel==2 branch which all-reduces across DP ranks,
                # giving a correctly weighted global validation loss.
                loss_pair = torch.cat([
                    output_tensor.reshape(1),
                    sample_count.reshape(1),
                ])
                return avg, {"loss": loss_pair.detach().clone()}

            return loss_sum, loss_func
        else:
            loss = _training_forward(batch, model, ctx)

            def loss_func(output_tensor):
                return output_tensor, {"loss": output_tensor.detach().clone()}

            return loss, loss_func

    return forward_step_func
