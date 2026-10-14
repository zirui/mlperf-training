"""Switch every A6W4-forward MXFP4Linear to an FP8 forward once, at FLUX_FPROP_SWITCH_STEP.

Installs the switch module as primus/backends/diffusion/models/quantization/flux_fprop_switch.py
and hooks the trainer twice:

* around ``self._mlperf_warmup()``: before it, ``prepare`` builds the FP8 scale slots, so the
  A6W4 graphs the warmup compiles already see them (only an FP8 forward reads them); after it,
  ``precompile`` switches to FP8, runs the warmup again so the FP8 train and eval graphs are
  compiled before run_start, and switches back to A6W4;
* right after ``self.global_step += 1``: the step that completes update number
  FLUX_FPROP_SWITCH_STEP is the last A6W4 one and the next forward is FP8. Backward passes are
  untouched (dgrad and wgrad stay MXFP4). The switch also wraps ``validate_loss`` to restore the
  FP8 delayed scales afterwards, since an eval forward updates them from eval activations.

Switching swaps each module between two pre-built configs, so Dynamo finds the graphs compiled
for that config instead of tracing new ones. A weight on the FP4 gather (FLUX_FP4_ALL_GATHER=1)
also has its FSDP2 local shard swapped between the A6W4 wrapper, which gathers the A6W4 packs,
and the FP8 wrapper, which gathers the FP8 fprop copy and the MXFP4 Dgrad col pack. Both wrap
the same fp32 shard, so the optimizer state is unaffected. FSDP2 caches a parameter's gather
outputs and unsharded tensor after its first gather, sized for the old payloads; the swap clears
them so the next gather allocates for the new ones. The FP8 delayed scales the warmup left behind
are dropped, so the first FP8 forward bootstraps each from the live amax as at step 0.

FLUX_FPROP_SWITCH_PRECOMPILE=0 is the first version: no precompile, a torch._dynamo.reset() at
the switch, and FLUX_FP4_ALL_GATHER=0 for the whole run.

Usage: python flux_fprop_switch.py <PRIMUS_PATH>
"""

import re
import sys
from pathlib import Path

MARKER = "_flux_fprop_switch"

MODULE = '''"""FLUX_FPROP_SWITCH_STEP runtime: see patches/flux_fprop_switch.py."""

import logging
import os
import time
from dataclasses import replace

import torch

from primus.backends.diffusion.models.quantization import mxfp4_linear as mx

logger = logging.getLogger(__name__)

_PRECOMPILE = os.getenv("FLUX_FPROP_SWITCH_PRECOMPILE", "1") == "1"
# (module, A6W4 config, FP8 config, FP8 gather eligible)
_MODS: list = []
# id(FSDPParam) -> the local shard wrapper for each mode, kept so a switch back restores the
# original object.
_A6W4_LOCAL: dict = {}
_FP8_LOCAL: dict = {}
_READY = False


def _log(msg):
    logger.info(msg)
    print(msg, flush=True)


def switch_step():
    value = os.getenv("FLUX_FPROP_SWITCH_STEP", "").strip()
    if not value:
        return None
    step = int(value)
    if step <= 0:
        raise ValueError(f"FLUX_FPROP_SWITCH_STEP must be > 0, got {step}")
    fp4_ag = os.getenv("FLUX_FP4_ALL_GATHER", "1") == "1"
    if not _PRECOMPILE and fp4_ag:
        raise RuntimeError("FLUX_FPROP_SWITCH_PRECOMPILE=0 needs FLUX_FP4_ALL_GATHER=0")
    if _PRECOMPILE and fp4_ag and os.getenv("FLUX_FP8_ALL_GATHER", "0") != "1":
        raise RuntimeError(
            "FLUX_FPROP_SWITCH_STEP with the FP4 gather needs FLUX_FP8_ALL_GATHER=1, which runs "
            "the per-step amax refresh the FP8 gather quantizes with"
        )
    return step


def _raise_dynamo_limits():
    import torch._dynamo.config as dc

    for name, floor in (
        ("recompile_limit", 64),
        ("cache_size_limit", 64),
        ("accumulated_recompile_limit", 2048),
        ("accumulated_cache_size_limit", 2048),
    ):
        if hasattr(dc, name):
            setattr(dc, name, max(getattr(dc, name), floor))


def prepare(model):
    """Before the warmup: the config pair and FP8 scale slots of every A6W4 linear."""
    if not _PRECOMPILE:
        return
    _MODS.clear()
    for module in model.modules():
        if isinstance(module, mx.MXFP4Linear) and module.config.fprop == "a6w4":
            a6w4 = module.config
            fp8 = replace(a6w4, fprop="fp8")
            module.config = fp8
            module._build_fp8_slots(module.weight.device)
            eligible = mx._fp8_all_gather_eligible(module)
            module.config = a6w4
            _MODS.append((module, a6w4, fp8, eligible))
    _raise_dynamo_limits()
    _log(f"FLUX_FPROP_SWITCH_STEP: {len(_MODS)} A6W4 linears prepared for the FP8 switch "
         f"({sum(m[3] for m in _MODS)} FP8-gather eligible)")


def _fsdp_params(model):
    """id(owning module) -> its FSDPParam for ``weight``. Keyed on the module rather than on
    ``module.weight``, which is the unsharded parameter whenever the group is unsharded."""
    out = {}
    for module in model.modules():
        get = getattr(module, "_get_fsdp_state", None)
        if get is None:
            continue
        group = get()._fsdp_param_group
        if group is None:
            continue
        for fp in group.fsdp_params:
            info = fp._module_info
            if info.param_name == "weight":
                out[id(info.module)] = fp
    return out


def _reset_gather_state(fp):
    from torch.distributed.fsdp._fully_shard._fsdp_param import ShardedState

    if fp.sharded_state != ShardedState.SHARDED:
        raise RuntimeError(f"FLUX_FPROP_SWITCH_STEP: gather swap on a {fp.sharded_state} parameter")
    fp.all_gather_outputs = []
    if hasattr(fp, "_unsharded_param"):
        del fp._unsharded_param
    fp._unsharded_inner_tensors = []
    ext = getattr(fp, "_extensions_data", None)
    if ext is not None:
        ext.clear()


def _swap_gather(model, to_fp8):
    # Groups with reshard_after_forward=False (the root) stay unsharded between steps; the swap
    # needs every gather state to be the sharded one.
    for module in model.modules():
        if hasattr(module, "_get_fsdp_state") and hasattr(module, "reshard"):
            module.reshard()
    fps = _fsdp_params(model)
    swapped = plain = 0
    skipped: dict = {}
    for module, _, _, eligible in _MODS:
        fp = fps.get(id(module))
        if fp is None:
            skipped["no FSDPParam"] = skipped.get("no FSDPParam", 0) + 1
            continue
        weight = fp.sharded_param
        cur = weight._local_tensor
        key = id(fp)
        if to_fp8:
            if not isinstance(cur, mx._A6W4AllGatherWeight):
                kind = f"local {type(cur).__name__}"
                skipped[kind] = skipped.get(kind, 0) + 1
                continue
            _A6W4_LOCAL[key] = cur
            if eligible:
                new = _FP8_LOCAL.get(key)
                if new is None or new._tensor is not cur._tensor:
                    new = mx._FP8AllGatherWeight(cur._tensor)
                    _FP8_LOCAL[key] = new
            else:
                # Gathers the bf16 weight, as with FLUX_FP4_ALL_GATHER=0.
                new = cur._tensor
                plain += 1
        else:
            old = _A6W4_LOCAL.get(key)
            if old is None:
                continue
            if cur is old:
                continue
            if old._tensor is not getattr(cur, "_tensor", cur):
                raise RuntimeError("FLUX_FPROP_SWITCH_STEP: a gathered shard was rebound since the swap")
            new = old
        weight._local_tensor = new
        _reset_gather_state(fp)
        swapped += 1
    if skipped:
        _log(f"FLUX_FPROP_SWITCH_STEP: gather swap to {'fp8' if to_fp8 else 'a6w4'} skipped {skipped}")
    if to_fp8 and os.getenv("FLUX_FP4_ALL_GATHER", "1") == "1" and swapped == 0 and _MODS:
        raise RuntimeError("FLUX_FPROP_SWITCH_STEP: the FP4 gather is on but no gather was swapped")
    return swapped, plain


def _set_mode(model, fp8):
    for module, a6w4_cfg, fp8_cfg, _ in _MODS:
        module.config = fp8_cfg if fp8 else a6w4_cfg
    return _swap_gather(model, fp8)


def _refresh_fp8_gather(model):
    if os.getenv("FLUX_FP8_ALL_GATHER", "0") == "1":
        mx.precompute_fp8_all_gather_amax(model)


def precompile(trainer):
    """After the warmup: compile the FP8 graphs and the FP8 gather, then return to A6W4."""
    global _READY
    if not _PRECOMPILE:
        return
    t0 = time.time()
    model = trainer.model
    saved = (trainer.mlperf_warmup_train_steps, trainer.mlperf_warmup_validation_steps)
    trainer.mlperf_warmup_train_steps = max(1, saved[0])
    trainer.mlperf_warmup_validation_steps = max(1, saved[1])
    try:
        swapped, plain = _set_mode(model, True)
        _refresh_fp8_gather(model)
        trainer._mlperf_warmup()
        if mx._FP8_AG_PREQUANT:
            # The second refresh captures the prequant graphs; the second warmup replays them.
            _refresh_fp8_gather(model)
            trainer._mlperf_warmup()
        _set_mode(model, False)
    finally:
        trainer.mlperf_warmup_train_steps, trainer.mlperf_warmup_validation_steps = saved
    dropped = 0
    for module, _, _, _ in _MODS:
        for slot in (module.fp8_act_slot, module.fp8_wgt_slot):
            if slot is not None and mx._FP8_DUAL_SCALE.pop(slot.data_ptr(), None) is not None:
                dropped += 1
    torch.cuda.synchronize()
    _READY = True
    _log(f"FLUX_FPROP_SWITCH_STEP: precompiled the FP8 forward for {len(_MODS)} linears in "
         f"{time.time() - t0:.1f}s ({swapped} gathers swapped, {plain} of them to bf16; "
         f"{dropped} warmup FP8 scales dropped)")


def switch_fprop(model, global_step, target="fp8"):
    t0 = time.time()
    if _PRECOMPILE:
        if not _READY:
            raise RuntimeError("FLUX_FPROP_SWITCH_STEP: the FP8 forward was not precompiled "
                               "(the MLPerf warmup did not run)")
        swapped, plain = _set_mode(model, True)
        _refresh_fp8_gather(model)
        switched = len(_MODS)
        how = f"precompiled; {swapped} gathers swapped to FP8, {plain} to bf16"
    else:
        switched = 0
        for module in model.modules():
            if isinstance(module, mx.MXFP4Linear) and module.config.fprop == "a6w4":
                module.config = replace(module.config, fprop=target)
                module._build_fp8_slots(module.weight.device)
                switched += 1
        torch._dynamo.reset()
        how = "dynamo reset"
    _log(f"FLUX_FPROP_SWITCH_STEP: {switched} linears a6w4 -> {target} fprop after step "
         f"{global_step} ({how}, {time.time() - t0:.2f}s)")
    return switched


def guard_eval_scales(trainer):
    """Restore the FP8 delayed scales after each validation, so no scale derived from eval
    activations reaches a training step."""
    validate = trainer.validate_loss

    def validate_loss():
        saved = dict(mx._FP8_DUAL_SCALE)
        try:
            return validate()
        finally:
            mx._FP8_DUAL_SCALE.clear()
            mx._FP8_DUAL_SCALE.update(saved)

    trainer.validate_loss = validate_loss
'''

STEP_HOOK = (
    "{indent}_fsw = getattr(self, \"_flux_fprop_switch\", False)\n"
    "{indent}if _fsw is False:\n"
    "{indent}    from primus.backends.diffusion.models.quantization.flux_fprop_switch import switch_step\n"
    "\n"
    "{indent}    _fsw = switch_step()\n"
    "{indent}    self._flux_fprop_switch = _fsw\n"
    "{indent}if _fsw is not None and self.global_step >= _fsw:\n"
    "{indent}    from primus.backends.diffusion.models.quantization.flux_fprop_switch import (\n"
    "{indent}        guard_eval_scales,\n"
    "{indent}        switch_fprop,\n"
    "{indent}    )\n"
    "\n"
    "{indent}    switch_fprop(self.model, self.global_step)\n"
    "{indent}    guard_eval_scales(self)\n"
    "{indent}    self._flux_fprop_switch = None\n"
)

WARMUP_HOOK = (
    "{indent}from primus.backends.diffusion.models.quantization import flux_fprop_switch as _fsw_mod\n"
    "\n"
    "{indent}_fsw_on = _fsw_mod.switch_step() is not None\n"
    "{indent}if _fsw_on:\n"
    "{indent}    _fsw_mod.prepare(self.model)\n"
    "{indent}self._mlperf_warmup()\n"
    "{indent}if _fsw_on:\n"
    "{indent}    _fsw_mod.precompile(self)\n"
)


def _patch_once(src, pattern, build, what):
    anchor = re.compile(pattern, re.M)
    hits = anchor.findall(src)
    if len(hits) != 1:
        raise SystemExit(f"flux_fprop_switch: expected one {what}, found {len(hits)}")
    return anchor.sub(lambda m: build(m), src, count=1)


_NT4_BIAS_BAD = "_nt4_geoms(M, N, _NT4_BANDS, _NT4_RECT_BANDS)"
_NT4_BIAS_GOOD = "_nt4_geoms(M, N, beta_is_one, _NT4_BANDS, _NT4_RECT_BANDS)"


def _patch_turbo_nt4() -> None:
    """The FP8 forward's bias NT autotune calls `_nt4_geoms` without `beta_is_one` in some Turbo
    pins, a TypeError the first time an FP8 Linear with an unfused bias runs."""
    import importlib.util

    spec = importlib.util.find_spec("primus_turbo")
    if spec is None or not spec.submodule_search_locations:
        return
    kern = Path(spec.submodule_search_locations[0]) / "flydsl/gemm/gemm_fp8_kernel_flux.py"
    if not kern.exists():
        return
    src = kern.read_text()
    if _NT4_BIAS_BAD in src:
        kern.write_text(src.replace(_NT4_BIAS_BAD, _NT4_BIAS_GOOD))
        print(f"flux_fprop_switch: passed beta_is_one to _nt4_geoms in {kern}")


def main() -> int:
    root = Path(sys.argv[1])
    _patch_turbo_nt4()
    target = root / "primus/backends/diffusion/trainers/base.py"
    dest = root / "primus/backends/diffusion/models/quantization/flux_fprop_switch.py"
    dest.write_text(MODULE)
    src = target.read_text()
    if MARKER in src:
        if "guard_eval_scales" not in src:
            raise SystemExit(f"flux_fprop_switch: {target} has an older patch; start from a clean Primus")
        print(f"flux_fprop_switch: already patched {target}")
        return 0
    src = _patch_once(
        src, r"^([ \t]+)self\.global_step \+= 1\n",
        lambda m: m.group(0) + STEP_HOOK.format(indent=m.group(1)),
        "`self.global_step += 1`",
    )
    src = _patch_once(
        src, r"^([ \t]+)self\._mlperf_warmup\(\)\n",
        lambda m: WARMUP_HOOK.format(indent=m.group(1)),
        "`self._mlperf_warmup()` call",
    )
    target.write_text(src)
    print(f"flux_fprop_switch: patched {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
