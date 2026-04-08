# Copyright (c) 2025, AMD. All rights reserved.
#
# NaN detection hook for distributed optimizer debugging.
#
# Instruments the DDP backward hooks to detect exactly which parameter's
# gradient first becomes NaN, and whether NaN originates in autograd (param.grad)
# or in the DDP accumulation (param.main_grad).
#
# Usage in train.py:
#   from nan_detect import install_nan_detect_hooks
#   # After pretrain() creates the model but before training starts,
#   # or as a monkey-patch before pretrain():
#   install_nan_detect_hooks()

import os
import logging
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_NAN_DETECT_STATE = {
    "enabled": False,
    "step": 0,
    "microbatch": 0,
    "first_nan_param": None,
    "nan_report": [],
}


def _rank():
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def install_nan_detect_hooks():
    """Monkey-patch Megatron DDP to add NaN detection in backward hooks.

    Must be called BEFORE pretrain() — it patches the DDP class so that when
    DDP.__init__ registers backward hooks, they include NaN checking.
    """
    from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel

    _orig_make_hook = DistributedDataParallel._make_backward_post_hook

    def _nan_detecting_backward_post_hook(self, param):
        orig_hook = _orig_make_hook(self, param)
        param_name = None
        for name, p in self.module.named_parameters():
            if p is param:
                param_name = name
                break

        def hook(*unused):
            state = _NAN_DETECT_STATE
            if not state["enabled"]:
                return orig_hook(*unused)

            grad = param.grad
            main_grad = getattr(param, 'main_grad', None)

            grad_has_nan = False
            main_grad_has_nan_before = False
            main_grad_has_nan_after = False

            if grad is not None:
                grad_has_nan = torch.isnan(grad).any().item()

            if main_grad is not None:
                main_grad_has_nan_before = torch.isnan(main_grad).any().item()

            orig_hook(*unused)

            if main_grad is not None:
                main_grad_has_nan_after = torch.isnan(main_grad).any().item()

            if grad_has_nan or main_grad_has_nan_after:
                nan_frac_grad = 0.0
                nan_frac_main = 0.0
                if grad is not None and grad_has_nan:
                    nan_frac_grad = torch.isnan(grad).float().mean().item()
                if main_grad is not None and main_grad_has_nan_after:
                    nan_frac_main = torch.isnan(main_grad).float().mean().item()

                entry = {
                    "rank": _rank(),
                    "step": state["step"],
                    "microbatch": state["microbatch"],
                    "param": param_name,
                    "shape": list(param.shape),
                    "grad_nan": grad_has_nan,
                    "grad_nan_frac": nan_frac_grad,
                    "main_grad_nan_before_accum": main_grad_has_nan_before,
                    "main_grad_nan_after_accum": main_grad_has_nan_after,
                    "main_grad_nan_frac": nan_frac_main,
                }
                state["nan_report"].append(entry)

                if state["first_nan_param"] is None:
                    state["first_nan_param"] = param_name
                    print(
                        f"\n{'='*80}\n"
                        f"[NaN DETECT] rank={_rank()} step={state['step']} "
                        f"microbatch={state['microbatch']}\n"
                        f"  FIRST NaN in: {param_name} {list(param.shape)}\n"
                        f"  param.grad NaN: {grad_has_nan} (frac={nan_frac_grad:.6f})\n"
                        f"  main_grad NaN before accum: {main_grad_has_nan_before}\n"
                        f"  main_grad NaN after accum:  {main_grad_has_nan_after} "
                        f"(frac={nan_frac_main:.6f})\n"
                        f"{'='*80}",
                        flush=True,
                    )

        return hook

    DistributedDataParallel._make_backward_post_hook = _nan_detecting_backward_post_hook
    logger.info("[NaN detect] Patched DDP._make_backward_post_hook")


def nan_detect_enable():
    """Enable NaN detection for the current step."""
    _NAN_DETECT_STATE["enabled"] = True
    _NAN_DETECT_STATE["first_nan_param"] = None
    _NAN_DETECT_STATE["nan_report"] = []


def nan_detect_disable():
    """Disable NaN detection."""
    _NAN_DETECT_STATE["enabled"] = False


def nan_detect_set_step(step, microbatch=0):
    """Update step/microbatch counters."""
    _NAN_DETECT_STATE["step"] = step
    _NAN_DETECT_STATE["microbatch"] = microbatch


def nan_detect_report():
    """Return and clear the NaN report for the current step."""
    report = _NAN_DETECT_STATE["nan_report"]
    first = _NAN_DETECT_STATE["first_nan_param"]
    return {"first_nan_param": first, "entries": report}


def install_training_nan_hooks():
    """Full-auto install: patches DDP + patches training_log to report per-step.

    Call this BEFORE pretrain(). It will:
    1. Patch DDP backward hooks for NaN detection
    2. Patch the training loop to enable/disable detection per step
    """
    install_nan_detect_hooks()

    max_detect_steps = int(os.environ.get("NAN_DETECT_STEPS", "5"))

    from megatron.training import training as _mt_training

    _orig_train_step = _mt_training.train_step

    def _nan_detecting_train_step(*args, **kwargs):
        step = getattr(_nan_detecting_train_step, '_step_counter', 0)
        _nan_detecting_train_step._step_counter = step + 1

        if step < max_detect_steps:
            nan_detect_enable()
            nan_detect_set_step(step)

        result = _orig_train_step(*args, **kwargs)

        if step < max_detect_steps:
            report = nan_detect_report()
            nan_detect_disable()

            if report["first_nan_param"] is not None:
                n_nan = len(report["entries"])
                print(
                    f"[NaN DETECT] rank={_rank()} step={step}: "
                    f"{n_nan} params with NaN. "
                    f"First: {report['first_nan_param']}",
                    flush=True,
                )

                if _rank() == 0 and step == 0:
                    print("\n[NaN DETECT] Full report for step 0:", flush=True)
                    for e in report["entries"]:
                        print(
                            f"  {e['param']}: grad_nan={e['grad_nan']} "
                            f"(frac={e['grad_nan_frac']:.6f}) "
                            f"main_grad_before={e['main_grad_nan_before_accum']} "
                            f"main_grad_after={e['main_grad_nan_after_accum']} "
                            f"(frac={e['main_grad_nan_frac']:.6f})",
                            flush=True,
                        )
            elif _rank() == 0:
                print(f"[NaN DETECT] step={step}: All gradients finite.", flush=True)

        return result

    _mt_training.train_step = _nan_detecting_train_step
    if _rank() == 0:
        print(f"[NaN detect] Patched train_step, will monitor first {max_detect_steps} steps",
              flush=True)
