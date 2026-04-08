# Copyright (c) 2025, NVIDIA CORPORATION & AMD. All rights reserved.
#
# FP8 warmup for Primus Megatron-direct training path.
# Ported from NeMo's warmup.py — adapted for Megatron's pretrain() flow.
#
# NeMo runs 2 synthetic train + 2 val warmup steps before real training,
# then resets FP8 state. Without this, the first step's all-zero output
# produces degenerate FP8 amax=0, causing scale=inf and NaN on step 2+.
#
# Since Megatron's pretrain() controls the training loop, we monkey-patch
# train_step() to intercept the first N steps as warmup. During warmup the
# optimizer is neutered (betas=[1,1], weight_decay=0, bias_correction=False)
# so model weights are NOT modified. After warmup, FP8 meta tensors are reset.

import os
import sys

import torch
import torch.distributed


def _log(msg):
    """Print to stdout with flush, prefixed for easy grep."""
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        print(f"[FP8_WARMUP] {msg}", flush=True, file=sys.stderr)


def _manual_reset_fp8_meta(module):
    """Manually reset FP8 amax history and scaling tensors.

    Fallback for TE versions where reset_fp8_meta_tensors() doesn't exist.
    Iterates over fp8_meta dict entries and resets amax_history to zero,
    scale to 1.0, and scale_inv to 1.0.
    """
    if not hasattr(module, "fp8_meta"):
        return False

    meta = module.fp8_meta
    reset_count = 0
    for key in ("scaling_fwd", "scaling_bwd"):
        if key not in meta:
            continue
        tensor_meta = meta[key]
        if hasattr(tensor_meta, "amax_history"):
            tensor_meta.amax_history.fill_(0.0)
            reset_count += 1
        if hasattr(tensor_meta, "scale"):
            tensor_meta.scale.fill_(1.0)
            reset_count += 1
        if hasattr(tensor_meta, "scale_inv"):
            tensor_meta.scale_inv.fill_(1.0)
            reset_count += 1
    return reset_count > 0


def _seed_fp8_amax(model, seed_value=1.0):
    """Set all FP8 amax_history tensors to a safe non-zero value.

    After reset_fp8_meta_tensors() zeros the amax history, delayed scaling
    computes scale = fp8_max / max(amax_history) = fp8_max / 0 = inf, which
    causes NaN on the very next forward pass.

    By seeding amax_history with a positive value (default 1.0), we get
    scale = fp8_max / 1.0 = fp8_max (e.g. 448 for E4M3). This is a safe
    initial scale that allows delayed scaling to adapt normally.
    """
    count = 0

    def _seed(m):
        nonlocal count
        if not hasattr(m, "fp8_meta"):
            return
        meta = m.fp8_meta
        for key in ("scaling_fwd", "scaling_bwd"):
            if key not in meta:
                continue
            tensor_meta = meta[key]
            if hasattr(tensor_meta, "amax_history"):
                tensor_meta.amax_history.fill_(seed_value)
                count += 1

    model.apply(_seed)
    return count


def reset_fp8_state(model, reset_fp8_meta_tensors=False):
    """Reset FP8 initialization flag on every TE layer, forcing re-init.

    Ported from NeMo warmup.py. When reset_fp8_meta_tensors=True, also
    clears the amax history and scaling factors so FP8 starts fresh
    with safe defaults instead of degenerate values from warmup.
    """
    count = 0
    has_method = 0
    manual_reset = 0

    def _reset(m):
        nonlocal count, has_method, manual_reset
        if hasattr(m, "fp8_initialized"):
            m.fp8_initialized = False
            count += 1
            if reset_fp8_meta_tensors:
                if hasattr(m, "reset_fp8_meta_tensors"):
                    m.reset_fp8_meta_tensors()
                    has_method += 1
                elif _manual_reset_fp8_meta(m):
                    manual_reset += 1

    model.apply(_reset)
    _log(f"reset_fp8_state: {count} FP8 modules found, "
         f"{has_method} via reset_fp8_meta_tensors(), "
         f"{manual_reset} via manual reset")
    return count


def _neuter_optimizer(optimizer):
    """Set optimizer to no-op mode for warmup.

    With betas=[1,1], the momentum/variance update becomes:
      m_t = 1.0 * m_{t-1} + 0.0 * g_t = m_{t-1}  (unchanged, stays at 0)
      v_t = 1.0 * v_{t-1} + 0.0 * g_t^2 = v_{t-1}  (unchanged, stays at 0)
    So the Adam weight update is 0 — model weights don't change.
    """
    saved = []
    inner = getattr(optimizer, "optimizer", optimizer)
    _log(f"Neutering optimizer: type={type(optimizer).__name__}, "
         f"inner={type(inner).__name__}, "
         f"param_groups={len(inner.param_groups)}")
    for group in inner.param_groups:
        state = {}
        for key in ("betas", "weight_decay", "bias_correction", "pre_mult_wd"):
            if key in group:
                state[key] = group[key]
        saved.append(state)

        if "betas" in group:
            group["betas"] = [1.0, 1.0]
        if "weight_decay" in group:
            group["weight_decay"] = 0.0
        if "bias_correction" in group:
            group["bias_correction"] = False
        if "pre_mult_wd" in group:
            group["pre_mult_wd"] = 0.0
    return saved


def _restore_optimizer(optimizer, saved):
    """Restore optimizer params after warmup and clear step counters."""
    inner = getattr(optimizer, "optimizer", optimizer)
    for group, state in zip(inner.param_groups, saved):
        for key, val in state.items():
            group[key] = val
        if "step" in group:
            del group["step"]
    _log("Restored optimizer parameters")


def install_fp8_warmup_hook(warmup_steps=None):
    """Monkey-patch Megatron's train_step() to run FP8 warmup on the first N steps.

    During warmup steps:
    - Optimizer is neutered (no weight updates)
    - FP8 amax history gets populated with non-degenerate values
    After warmup:
    - Optimizer is restored
    - FP8 state is fully reset (fp8_initialized=False + meta tensors cleared)
    - Subsequent steps start with clean FP8 scaling
    """
    if warmup_steps is None:
        warmup_steps = int(os.environ.get("WARMUP_TRAIN_STEPS", "2"))

    fp8_mode = os.environ.get("FP8", "")
    if not fp8_mode or warmup_steps <= 0:
        _log(f"SKIPPED (FP8={fp8_mode!r}, warmup_steps={warmup_steps})")
        return

    import megatron.training.training as mt
    _orig_train_step = mt.train_step
    _count = [0]
    _saved_opt = [None]
    _saved_params = [None]

    _log(f"Patching megatron.training.training.train_step "
         f"(id={id(_orig_train_step):#x})")

    def _save_model_params(models):
        """Save a copy of all model parameters before warmup.

        NeMo's warmup zeros gradients between backward and optimizer.step()
        to prevent 0*NaN=NaN corruption. Since Megatron's train_step() is
        monolithic (backward + optimizer in one call), we can't do that.
        Instead, we snapshot parameters before warmup and restore after.
        """
        saved = {}
        for m in models:
            for name, p in m.named_parameters():
                saved[name] = p.data.clone()
        return saved

    def _restore_model_params(models, saved):
        """Restore model parameters from snapshot, undoing any NaN corruption."""
        restored = 0
        for m in models:
            for name, p in m.named_parameters():
                if name in saved:
                    p.data.copy_(saved[name])
                    restored += 1
        return restored

    def _hooked_train_step(*args, **kwargs):
        _count[0] += 1
        step = _count[0]

        optimizer = args[3] if len(args) > 3 else kwargs.get("optimizer")
        model = args[2] if len(args) > 2 else kwargs.get("model")

        if step == 1:
            models = model if isinstance(model, (list, tuple)) else [model]
            _log(f"Step {step}/{warmup_steps}: saving model parameters "
                 f"(NaN-protection snapshot)")
            _saved_params[0] = _save_model_params(models)
            _log(f"Saved {len(_saved_params[0])} parameter tensors")

            _log(f"Step {step}/{warmup_steps}: neutering optimizer")
            _saved_opt[0] = _neuter_optimizer(optimizer)

        _log(f"Step {step}/{warmup_steps}: calling original train_step "
             f"(model type={type(model).__name__})")

        result = _orig_train_step(*args, **kwargs)

        if step <= warmup_steps:
            loss_dict = result[0] if isinstance(result, tuple) else result
            loss_val = "?"
            if isinstance(loss_dict, dict):
                for k, v in loss_dict.items():
                    if hasattr(v, 'item'):
                        loss_val = f"{k}={v.item()}"
                    else:
                        loss_val = f"{k}={v}"
                    break
            _log(f"Step {step}/{warmup_steps}: train_step returned "
                 f"(loss_dict type={type(loss_dict).__name__}, {loss_val})")

        if step == warmup_steps:
            _log(f"Step {step}: warmup complete, restoring optimizer")
            _restore_optimizer(optimizer, _saved_opt[0])
            _saved_opt[0] = None

            models = model if isinstance(model, (list, tuple)) else [model]

            _log("Restoring model parameters from pre-warmup snapshot")
            n_restored = _restore_model_params(models, _saved_params[0])
            _saved_params[0] = None
            _log(f"Restored {n_restored} parameter tensors")

            fp8_recipe = os.environ.get("FP8_RECIPE", "delayed")
            _log(f"Resetting FP8 state on {len(models)} model(s), "
                 f"recipe={fp8_recipe}")
            total_reset = 0
            for i, m in enumerate(models):
                _log(f"  Model[{i}]: type={type(m).__name__}")
                total_reset += reset_fp8_state(
                    m, reset_fp8_meta_tensors=(fp8_recipe == "delayed"))

            if hasattr(optimizer, "reload_model_params"):
                optimizer.reload_model_params()
                _log("Called optimizer.reload_model_params()")

            torch.cuda.synchronize()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            nan_params = 0
            for m in models:
                for name, p in m.named_parameters():
                    if p.data.is_floating_point() and torch.isnan(p.data).any():
                        nan_params += 1
            _log(f"Post-restore parameter check: nan_params={nan_params}")

            _log(f"FP8 warmup DONE: reset {total_reset} modules across "
                 f"{len(models)} model(s). Restoring original train_step.")

            mt.train_step = _orig_train_step

        return result

    mt.train_step = _hooked_train_step

    verify = mt.train_step is _hooked_train_step
    _log(f"Installed FP8 warmup hook (warmup_steps={warmup_steps}, "
         f"patch_verified={verify})")
