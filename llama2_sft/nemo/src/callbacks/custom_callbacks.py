import os
import time
import logging
from math import ceil
from typing import Any, List

import lightning as pl
import numpy as np
import torch
import transformer_engine.common.recipe
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from custom_llama import reset_fp8_state, run_training_warmup
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.num_microbatches_calculator import (
    reconfigure_num_microbatches_calculator,
)
from mlperf_common.callbacks import LoggingCallback, MLPerfLogger
from omegaconf import DictConfig, OmegaConf
from prof_handler import TOTAL_ACTIVE_STEPS, TOTAL_WARMUP_STEPS
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer, Float8Tensor
from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer

class RankZeroFilter(logging.Filter):
    def __init__(self, rank):
        self.rank = rank

    def filter(self, record):
        return self.rank == 0

rank = int(os.environ.get("RANK", "0"))
logger = logging.getLogger(__name__)
logger.addFilter(RankZeroFilter(rank))        

def _env_enabled(name: str) -> bool:
    val = os.getenv(name, "").strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    return False


def _uniform_recompute_chunk_size(num_layers_per_pp_rank: int, preferred: int = 72) -> int:
    """Chunk size for Megatron ``recompute_method=uniform``.

    Each checkpoint segment is ``[layer_idx, layer_idx + recompute_num_layers)``. The loop
    advances ``layer_idx`` by ``recompute_num_layers`` until ``layer_idx < num_layers``;
    if ``num_layers`` is not divisible by ``recompute_num_layers``, the last segment still
    uses a full-sized span and indexes past ``layers[-1]`` (e.g. 80 layers and chunk 72
    runs ``range(72, 144)`` → ``index 80 is out of range``).
    """
    cap = min(preferred, num_layers_per_pp_rank)
    for c in range(cap, 0, -1):
        if num_layers_per_pp_rank % c == 0:
            return c
    return 1


def get_context_parallel_world_size_override():
    return 1


original_get_context_parallel_world_size = parallel_state.get_context_parallel_world_size


class ThroughputTimer:
    def __init__(self, gbs):
        self.start_time = None
        self.stop_time = None
        self.elapsed_time = 0
        self.samples = 0
        self.gbs = gbs

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.stop_time = time.time()
        self.samples += self.gbs
        self.elapsed_time += self.stop_time - self.start_time

    def get_throughput(self):
        throughput = self.samples / self.elapsed_time
        self.samples = 0
        self.elapsed_time = 0
        return throughput


def _extract_module(model):
    """Unwrap the model from MegatronParallel / DDP wrappers to get the GPT module."""
    m = model
    while hasattr(m, "module"):
        m = m.module
    return m


class CustomCallback(LoggingCallback):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.gbs = cfg.model.global_batch_size
        self.iter_time = time.time()
        self.enable_perf_debug = int(os.environ.get("ENABLE_PERF_DEBUG", "0"))
        self.enable_mem_profile = int(os.environ.get("ENABLE_MEMORY_PROFILE", "0"))
        if self.enable_mem_profile:
            torch.cuda.memory._record_memory_history(max_entries=100000)
        self.cp_group = None
        self.cp_global_ranks = None
        self.cp_stream = None
        self.logging_interval = int(os.environ.get("LOGGING_INTERVAL", "10"))

        self.throughput_timer = ThroughputTimer(self.gbs)
        self.train_start_time = None

        self.healing_precision = cfg.model.healing_precision
        self.healing_lambda = None

        self.pre_quantized_model = cfg.model.pre_quantized_model
        self.fp8_quantizer = None
        self.mxfp4_quantizer = None
        self.fp8_cpu_params = []
        self.store_quantized_params_on_gpu = cfg.model.store_gpu

        self.first_last_layers_bf16 = cfg.model.first_last_layers_bf16
        self.num_layers_at_start_in_bf16 = cfg.model.num_layers_at_start_in_bf16
        self.num_layers_at_end_in_bf16 = cfg.model.num_layers_at_end_in_bf16

        self.rpd = None
        self.rpd_status = None
        self.profiler = os.getenv("PROFILER", "")

        if self.profiler == "rpd":
            from rpdTracerControl import rpdTracerControl

            rpdTracerControl.setFilename(name="trace.rpd", append=True)
            self.rpd = rpdTracerControl()

    def __deepcopy__(self, memo):
        return CustomCallback(self.cfg)

    @staticmethod
    def _gpu_mem_mb():
        return torch.cuda.memory_allocated() / 1e6, torch.cuda.memory_reserved() / 1e6

    def _reset_full_iteration_cuda_graphs(self, trainer) -> None:
        """Drop Megatron full-iteration CUDA graph captures before FP8 healing (MLPerf pattern).

        After MXFP4→FP8 weight swap and recipe change, replaying old graphs is invalid.
        Always clears stored graphs and results. If ``reset_cg_after_healing`` is true,
        also resets per-stage warmup step counters so ``FullCudaGraphWrapper`` re-runs
        ``cuda_graph_warmup_steps`` before re-capture.
        """
        try:
            from megatron.core.full_cuda_graph import FullCudaGraphWrapper
        except ImportError:
            if trainer.global_rank == 0:
                print(
                    "[healing] FullCudaGraphWrapper unavailable — skip full-iteration "
                    "CUDA graph reset",
                    flush=True,
                )
            return

        FullCudaGraphWrapper.cuda_graph["training"] = None
        FullCudaGraphWrapper.cuda_graph["validation"] = None
        FullCudaGraphWrapper.result["training"] = None
        FullCudaGraphWrapper.result["validation"] = None

        if bool(getattr(self.cfg.model, "reset_cg_after_healing", False)):
            FullCudaGraphWrapper.curr_iteration["training"] = 0
            FullCudaGraphWrapper.curr_iteration["validation"] = 0

    def _healing_setup(self):
        if self.healing_precision == "FP8_DS":
            self.healing_lambda = lambda config: transformer_engine.common.recipe.DelayedScaling(
                amax_history_len=self.cfg.model.fp8_amax_history_len,
                amax_compute_algo=self.cfg.model.fp8_amax_compute_algo,
                reduce_amax=self.cfg.model.reduce_amax,
                fp8_dpa=self.cfg.model.fp8_dot_product_attention,
            )
        elif self.healing_precision == "MXFP8":
            self.healing_lambda = lambda config: transformer_engine.common.recipe.MXFP8BlockScaling()
        else:
            raise ValueError(f"Unsupported healing precision: {self.healing_precision}")

    def _pre_quantize_model(self, trainer, pl_module):
        """Pre-quantize model: store FP8 clones on CPU for healing, replace weights with MXFP4."""
        _t0 = time.time()
        device = next(_extract_module(trainer.model).parameters()).device

        # Clone current FP8 weights to CPU for healing.
        # Must happen BEFORE MXFP4 replacement since we reference the live FP8 params.
        # NOTE: this stays on even when HEALING_ITER<=0. Skipping it caused the
        # MXFP4 path that follows to produce different post-quant weights
        # (~12 init eval vs ~3.7 with the FP8 clone), so we eat the ~28s /
        # ~142GB CPU-RAM cost for parity even when the snapshot is unused.
        self.fp8_quantizer = Float8Quantizer(
            scale=torch.ones(1, dtype=torch.float32, device=device),
            amax=torch.zeros(1, dtype=torch.float32, device=device),
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=_env_enabled("ENABLE_TRANSPOSE_CACHE"),
        )
        self.fp8_cpu_params = self._get_quantized_params_cpu(
            trainer.model, self.fp8_quantizer, "FP8_DS"
        )

        # Replace weights with MXFP4 (both orientations for fwd + bwd GEMMs)
        self.mxfp4_quantizer = MXFP4Quantizer(
            rowwise=True,
            columnwise=True,
            shuffle_rowwise_data=True,
            shuffle_columnwise_data=True,
            with_gemm_swizzled_scales=True,
            use_hadamard=_env_enabled("NVTE_MXFP4_USE_HADAMARD"),
        )
        self._get_quantized_params_cpu(
            trainer.model, self.mxfp4_quantizer, "MXFP4", replace=True
        )

        torch.cuda.empty_cache()
        self.pre_quantized_model = True

    def _get_quantized_params_cpu(self, model, quantizer, qtype: str, replace: bool = False) -> List:
        extracted_module = _extract_module(model)
        layers = extracted_module.decoder.layers
        layer_count = len(layers)

        quantized_params = []
        for layer_idx, layer in enumerate(layers):
            if self.first_last_layers_bf16:
                if (
                    layer_idx < self.num_layers_at_start_in_bf16
                    or layer_idx >= layer_count - self.num_layers_at_end_in_bf16
                ):
                    quantized_params.append([])
                    continue

            quantized_layer_params = []
            for name, module in layer.named_modules():
                if not isinstance(module, (te.Linear, te.LayerNormLinear, te.LayerNormMLP)):
                    continue
                if not hasattr(module, "weight"):
                    continue
                param = module.weight
                with torch.no_grad():
                    if qtype == "MXFP4":
                        if isinstance(param, QuantizedTensor):
                            bf16_param = param.dequantize().detach()
                        else:
                            bf16_param = param.detach().to(torch.bfloat16)
                        bf16_param = bf16_param.contiguous()
                        qparam = quantizer(bf16_param)
                        del bf16_param
                        assert qparam._rowwise_data is not None, "No rowwise data."
                        assert qparam._columnwise_data is not None, "No columnwise data."
                    elif qtype == "FP8_DS":
                        fp8_data = param.data.to(torch.float8_e4m3fn).view(torch.uint8)
                        scale_inv = torch.ones(1, dtype=torch.float32, device=param.device)
                        qparam = Float8Tensor(
                            shape=param.shape,
                            dtype=torch.bfloat16,
                            data=fp8_data,
                            fp8_scale_inv=scale_inv,
                            fp8_dtype=tex.DType.kFloat8E4M3,
                            quantizer=quantizer,
                        )
                        assert qparam._data is not None, "No data."
                    else:
                        raise ValueError(f"Unsupported quantization type: {qtype}")

                    if replace:
                        qparam.requires_grad = False
                        module._parameters["weight"] = qparam
                    else:
                        qparam = qparam.clone()
                        if not self.store_quantized_params_on_gpu:
                            if qtype == "MXFP4":
                                qparam._rowwise_data = qparam._rowwise_data.cpu().pin_memory()
                                qparam._columnwise_data = qparam._columnwise_data.cpu().pin_memory()
                            elif qtype == "FP8_DS":
                                qparam._data = qparam._data.cpu().pin_memory()
                                if qparam._transpose is not None:
                                    qparam._transpose = qparam._transpose.cpu().pin_memory()
                        quantized_layer_params.append(qparam)
            quantized_params.append(quantized_layer_params)

        return quantized_params

    def _set_quantized_params_cpu(self, model, cpu_params, qtype: str):
        extracted_module = _extract_module(model)
        layers = extracted_module.decoder.layers
        layer_count = len(layers)
        dev = torch.cuda.current_device()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        xfer_stream = torch.cuda.Stream()

        for layer_idx, layer in enumerate(layers):
            if self.first_last_layers_bf16:
                if (
                    layer_idx < self.num_layers_at_start_in_bf16
                    or layer_idx >= layer_count - self.num_layers_at_end_in_bf16
                ):
                    continue

            target_modules = []
            for _name, module in layer.named_modules():
                if not isinstance(module, (te.Linear, te.LayerNormLinear, te.LayerNormMLP)):
                    continue
                if not hasattr(module, "weight"):
                    continue
                target_modules.append(module)

            with torch.no_grad():
                for module in target_modules:
                    module._parameters["weight"] = None

            for module in target_modules:
                with torch.no_grad():
                    weight = cpu_params[layer_idx].pop(0)

                    if not self.store_quantized_params_on_gpu:
                        with torch.cuda.stream(xfer_stream):
                            if qtype == "FP8_DS":
                                weight._data = weight._data.to(dev, non_blocking=True)
                                if weight._transpose is not None:
                                    if _env_enabled("ENABLE_TRANSPOSE_CACHE"):
                                        weight._transpose = weight._transpose.to(dev, non_blocking=True)
                                    else:
                                        weight._transpose = None
                            elif qtype == "MXFP4":
                                weight._rowwise_data = weight._rowwise_data.to(dev, non_blocking=True)
                                weight._columnwise_data = weight._columnwise_data.to(dev, non_blocking=True)
                            else:
                                raise ValueError(f"Unsupported quantization type: {qtype}")

                    weight.requires_grad = False
                    module._parameters["weight"] = weight

            if self.cfg.model.use_transformer_engine_op_fuser:
                layer.mlp._fused_impl = (layer.mlp._make_fused_impl(),)
                layer.self_attention.linear_proj._fused_branches = (
                    layer.self_attention.linear_proj._make_fused_branches()
                )
                layer.self_attention.linear_qkv._fused_branches = (
                    layer.self_attention.linear_qkv._make_fused_branches()
                )

        xfer_stream.synchronize()

    def _pre_ttt_fp8_warmup(self, trainer):
        """Pre-compile FP8 kernels before the timed training window."""
        import megatron.core.fp4_utils

        if not self.fp8_cpu_params:
            return
        if not self.cfg.model.healing_iter or self.cfg.model.healing_iter <= 0:
            return

        _t0 = time.time()

        extracted = _extract_module(trainer.model)
        layers = extracted.decoder.layers
        layer_count = len(layers)
        dev = torch.cuda.current_device()

        saved_weights = []
        xfer_stream = torch.cuda.Stream()
        for layer_idx, layer in enumerate(layers):
            if self.first_last_layers_bf16:
                if (
                    layer_idx < self.num_layers_at_start_in_bf16
                    or layer_idx >= layer_count - self.num_layers_at_end_in_bf16
                ):
                    continue

            param_idx = 0
            for _n, mod in layer.named_modules():
                if not isinstance(mod, (te.Linear, te.LayerNormLinear, te.LayerNormMLP)):
                    continue
                if not hasattr(mod, "weight"):
                    continue

                cpu_fp8 = self.fp8_cpu_params[layer_idx][param_idx]
                with torch.no_grad(), torch.cuda.stream(xfer_stream):
                    gpu_data = cpu_fp8._data.to(dev, non_blocking=True)
                    if cpu_fp8._transpose is not None and _env_enabled("ENABLE_TRANSPOSE_CACHE"):
                        gpu_tp = cpu_fp8._transpose.to(dev, non_blocking=True)
                    else:
                        gpu_tp = None
                    gpu_fp8 = Float8Tensor(
                        shape=cpu_fp8.shape,
                        dtype=cpu_fp8.dtype,
                        data=gpu_data,
                        fp8_scale_inv=cpu_fp8._scale_inv.to(dev, non_blocking=True),
                        fp8_dtype=cpu_fp8._fp8_dtype,
                        data_transpose=gpu_tp,
                        quantizer=self.fp8_quantizer,
                    )
                    gpu_fp8.requires_grad = False

                saved_weights.append((mod, mod.weight))
                mod._parameters["weight"] = gpu_fp8
                param_idx += 1

        xfer_stream.synchronize()

        saved_mxfp4_phase = megatron.core.fp4_utils._mxfp4_phase
        saved_recipe = megatron.core.fp4_utils.get_fp4_recipe
        megatron.core.fp4_utils._mxfp4_phase = False
        megatron.core.fp4_utils.get_fp4_recipe = self.healing_lambda

        if not _env_enabled("ENABLE_TRANSPOSE_CACHE"):
            for m in trainer.model.modules():
                if isinstance(m, (te.Linear, te.LayerNormLinear, te.LayerNormMLP)):
                    m.keep_fp8_weight_transpose_cache = False

        cfg = extracted.config
        saved_recompute_n = cfg.recompute_num_layers
        saved_rg = cfg.recompute_granularity
        saved_rm = cfg.recompute_method
        saved_cgi = cfg.cuda_graph_impl
        saved_ecg = cfg.enable_cuda_graph
        saved_xcg = cfg.external_cuda_graph
        per_rank = cfg.num_layers // max(cfg.pipeline_model_parallel_size, 1)
        # Megatron only uses recompute_num_layers when recompute_granularity == "full"
        # (checkpointed forward in transformer_block). If the user turned off
        # recomputation for CUDA graphs, forcing recompute_num_layers alone does nothing
        # and the forward stores full activations — OOM with FP8 weights resident.
        # For this synthetic warmup step only: disable CUDA graphs (full recompute is
        # incompatible with them in TransformerConfig) and enable full uniform
        # checkpointing. Restore everything before real training.
        cleared_graphs_for_warmup = False
        skip_forward = False
        if saved_rg is None:
            if cfg.cpu_offloading:
                skip_forward = True
            else:
                cleared_graphs_for_warmup = True
                cfg.cuda_graph_impl = "none"
                cfg.enable_cuda_graph = False
                cfg.external_cuda_graph = False
                cfg.recompute_granularity = "full"
                cfg.recompute_method = "uniform"
                cfg.recompute_num_layers = _uniform_recompute_chunk_size(per_rank)
        elif saved_rg == "full":
            cfg.recompute_num_layers = _uniform_recompute_chunk_size(per_rank)
        # selective: leave recompute_num_layers as None (Megatron requirement)

        optimizer = trainer.strategy.optimizers[0]
        try:
            if not skip_forward:
                trainer.model.training_step(trainer.model.get_synthetic_input())
                optimizer.zero_grad(set_to_none=True)
        except Exception as e:
            if trainer.global_rank == 0:
                print(f"[pre_ttt_fp8_warmup] Error (non-fatal): {e}", flush=True)

        megatron.core.fp4_utils._mxfp4_phase = saved_mxfp4_phase
        megatron.core.fp4_utils.get_fp4_recipe = saved_recipe
        cfg.recompute_num_layers = saved_recompute_n
        if cleared_graphs_for_warmup:
            cfg.recompute_granularity = saved_rg
            cfg.recompute_method = saved_rm
            cfg.cuda_graph_impl = saved_cgi
            cfg.enable_cuda_graph = saved_ecg
            cfg.external_cuda_graph = saved_xcg

        for mod, orig_weight in saved_weights:
            mod._parameters["weight"] = orig_weight

        reset_fp8_state(trainer.lightning_module)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


    def on_train_start(self, trainer, pl_module):

        self._healing_setup()

        if not self.cfg.model.fp4:
            import megatron.core.fp4_utils

            megatron.core.fp4_utils._mxfp4_phase = False
            megatron.core.fp4_utils.get_fp4_recipe = self.healing_lambda

        if self.pre_quantized_model:
            self._pre_quantize_model(trainer, pl_module)
        super().on_train_start(trainer, pl_module)

        torch.cuda.synchronize()

        _t0 = time.time()
        trainer.warmup = True
        try:
            trainer.fit_loop.epoch_loop.val_loop.run()
        except Exception as e:
            if trainer.global_rank == 0:
                print(f"[pre_val_warmup] FAILED: {e}", flush=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # if self.cfg.model.healing_iter and self.cfg.model.healing_iter > 0:
        #     self._pre_ttt_fp8_warmup(trainer)

        self.train_start_time = None

    def on_train_end(self, trainer, pl_module):
        if self.rpd and self.rpd_status == "running":
            self.rpd.stop()
            self.rpd_status = "finished"

        if self.train_block_started:
            self._end_train_block(trainer, self.get_train_step_samples_count(trainer, pl_module))

        if self.train_start_time is not None:
            duration_sec = time.time() - self.train_start_time
            train_batch_size = self.get_train_step_samples_count(trainer, pl_module)
            if (hasattr(trainer, "target_reached") and trainer.target_reached) or self.force_success:
                status = self.mllogger.constants.SUCCESS
            else:
                status = self.mllogger.constants.ABORTED
            self.mllogger.end(
                self.mllogger.constants.RUN_STOP,
                metadata={
                    "step": trainer.global_step,
                    self.mllogger.constants.SAMPLES_COUNT: trainer.global_step * train_batch_size,
                    "status": status,
                    "duration": f"{duration_sec} sec -> {duration_sec / 60.0} minutes",
                },
            )
            trainer.run_stop_logged = True

        super().on_train_end(trainer, pl_module)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.train_start_time is None:
            self.train_start_time = time.time()
            if trainer.global_rank == 0:
                torch.cuda.reset_peak_memory_stats()

        self.throughput_timer.start()

        if self.rpd and trainer.global_step >= TOTAL_WARMUP_STEPS and not self.rpd_status:
            self.rpd.start()
            self.rpd_status = "running"

        if trainer.global_step > TOTAL_WARMUP_STEPS + TOTAL_ACTIVE_STEPS:
            if self.rpd and self.rpd_status == "running":
                self.rpd.stop()
                self.rpd_status = "finished"

        return super().on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def _setup_partial_columnwise_cache(self, trainer):
        """After step 1, compute memory budget and flag modules for persistent columnwise."""
        if self.pre_quantized_model:
            return

        torch.cuda.synchronize()
        peak_res_bytes = torch.cuda.max_memory_reserved()
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        nccl_overhead = 19 * 1024**3
        safety_margin = 5 * 1024**3
        budget_bytes = total_bytes - peak_res_bytes - nccl_overhead - safety_margin

        col_per_module = 0.1125 * 1024**3
        max_modules = max(0, int(budget_bytes / col_per_module))
        max_modules = min(max_modules, 320)

        if max_modules == 0:
            return

        extracted = _extract_module(trainer.model)
        layers = extracted.decoder.layers
        flagged = 0
        for layer in reversed(layers):
            for name, m in layer.named_modules():
                if isinstance(m, (te.Linear, te.LayerNormLinear, te.LayerNormMLP)) and hasattr(
                    m, "weight"
                ):
                    if flagged < max_modules:
                        m._mxfp4_persist_columnwise = True
                        flagged += 1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.throughput_timer.stop()

        if trainer.global_step == 1:
            self._setup_partial_columnwise_cache(trainer)

        if trainer.global_rank == 0:
            if self.enable_perf_debug:
                torch.cuda.synchronize()
                self.iter_time = time.time()

            if trainer.global_step == 1 and self.enable_mem_profile:
                torch.cuda.memory._record_memory_history(max_entries=100000)

            if trainer.global_step == 3 and self.enable_mem_profile:
                torch.cuda.memory._dump_snapshot("/results/lora.pickle")

        if trainer.global_step % self.logging_interval == 0:
            self.mllogger.event(
                key="tracked_stats",
                value={"train_loss": outputs["loss"].item()},
                metadata={
                    self.mllogger.constants.SAMPLES_COUNT: trainer.global_step * self.gbs,
                    "lr": trainer.optimizers[0].param_groups[0]["lr"],
                    "step": trainer.global_step,
                },
            )

        if trainer.global_step + 1 == self.cfg.model.healing_iter:

            torch.distributed.barrier()

            self._reset_full_iteration_cuda_graphs(trainer)

            if self.pre_quantized_model:
                assert self.fp8_cpu_params, "FP8 parameters not found"
                self._set_quantized_params_cpu(trainer.model, self.fp8_cpu_params, self.healing_precision)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            import megatron.core.fp4_utils

            megatron.core.fp4_utils.get_fp4_recipe = self.healing_lambda
            megatron.core.fp4_utils._mxfp4_phase = False

            if not _env_enabled("ENABLE_TRANSPOSE_CACHE"):
                _reset = 0
                for m in trainer.model.modules():
                    if isinstance(m, (te.Linear, te.LayerNormLinear, te.LayerNormMLP)):
                        m.keep_fp8_weight_transpose_cache = False
                        _reset += 1

            extracted = _extract_module(trainer.model)
            if extracted.config.recompute_granularity is not None:
                extracted.config.recompute_granularity = None
                extracted.config.recompute_num_layers = None
                extracted.config.recompute_method = None

            _cleared = 0
            for m in trainer.model.modules():
                if hasattr(m, "_mxfp4_weight_cache"):
                    del m._mxfp4_weight_cache
                    _cleared += 1
                if hasattr(m, "_mxfp4_persist_columnwise"):
                    del m._mxfp4_persist_columnwise

            # Force TE to re-initialize fp8_meta under the new (DelayedScaling/MXFP8)
            # recipe on the next forward. Without this, modules keep their old
            # MXFP4BlockScalingRecipeState while the recipe + weights changed under
            # them, and the next GEMM dereferences scale/amax pointers from the
            # wrong RecipeState type — observed as a HIP "Memory access fault on (nil)".
            #
            # NOTE: TE's set_meta_tensor() detects RecipeState type mismatch on its
            # own and overwrites fp8_meta["scaling_fwd"|"scaling_bwd"]. We must NOT
            # pop those keys here, otherwise its early-return branch (gated by
            # fp8_meta_tensors_initialized=True) raises KeyError: 'scaling_fwd'.
            reset_fp8_state(trainer.lightning_module)

            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            self.mllogger.event(
                key="healing",
                value=trainer.global_step + 1,
                metadata={"samples_count": trainer.global_step * self.gbs},
            )

        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_start(self, trainer, pl_module):
        _is_warmup = getattr(trainer, "warmup", False)
        if not _is_warmup:
            torch.cuda.synchronize()
            import megatron.core.fp4_utils

            if megatron.core.fp4_utils._mxfp4_phase:
                torch.cuda.empty_cache()
            self._log_throughput(trainer.global_step)

        super().on_validation_start(trainer, pl_module)

        if not _is_warmup:
            trainer.val_check_interval = ceil(int(os.environ.get("VAL_CHECK_INTERVAL", "384")) / self.gbs)
            trainer.val_check_batch = ceil(int(os.environ.get("VAL_CHECK_INTERVAL", "384")) / self.gbs)

        if self.cfg.model.eval_cp:
            trainer.model.config.context_parallel_size = self.cfg.model.eval_cp
            if self.cp_group is None:
                self.cp_group = parallel_state.get_context_parallel_group()
                self.cp_global_ranks = parallel_state.get_context_parallel_global_ranks()
                extracted = _extract_module(trainer.model)
                self.cp_stream = extracted.decoder.layers[0].self_attention.core_attention.cp_stream

            parallel_state.get_context_parallel_world_size = get_context_parallel_world_size_override
            for module in trainer.model.modules():
                for index, child in enumerate(module.modules()):
                    if isinstance(child, TEDotProductAttention) and hasattr(
                        child, "set_context_parallel_group"
                    ):
                        child.set_context_parallel_group(
                            cp_group=None,
                            cp_global_ranks=None,
                            cp_stream=None,
                            cp_comm_type=None,
                        )
            _extract_module(trainer.model).rotary_pos_emb.cp_group = None

        if self.cfg.model.val_micro_batch_size:
            reconfigure_num_microbatches_calculator(
                rank=torch.distributed.get_rank(),
                rampup_batch_size=None,
                global_batch_size=self.cfg.model.val_global_batch_size,
                micro_batch_size=self.cfg.model.val_micro_batch_size,
                data_parallel_size=trainer.val_dataloaders.batch_sampler.data_parallel_size,
            )

    def on_validation_end(self, trainer, pl_module):
        if self.cfg.model.eval_cp:
            trainer.model.config.context_parallel_size = self.cfg.model.context_parallel_size
            parallel_state.get_context_parallel_world_size = original_get_context_parallel_world_size
            for module in trainer.model.modules():
                for index, child in enumerate(module.modules()):
                    if isinstance(child, TEDotProductAttention) and hasattr(
                        child, "set_context_parallel_group"
                    ):
                        child.set_context_parallel_group(
                            cp_group=self.cp_group,
                            cp_global_ranks=self.cp_global_ranks,
                            cp_stream=self.cp_stream,
                            cp_comm_type=self.cfg.model.cp_comm_type,
                        )
            _extract_module(trainer.model).rotary_pos_emb.cp_group = self.cp_group

        if self.cfg.model.val_micro_batch_size:
            reconfigure_num_microbatches_calculator(
                rank=torch.distributed.get_rank(),
                rampup_batch_size=None,
                global_batch_size=self.cfg.model.global_batch_size,
                micro_batch_size=self.cfg.model.micro_batch_size,
                data_parallel_size=trainer.train_dataloader.batch_sampler.data_parallel_size,
            )

        super().on_validation_end(trainer, pl_module)

    def warmup(self, trainer, pl_module):
        if self.cfg.model.custom.warmup:
            for m in trainer.model.modules():
                m._mxfp4_cache_disabled = True
                if hasattr(m, "_mxfp4_weight_cache"):
                    del m._mxfp4_weight_cache
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            try:
                run_training_warmup(
                    trainer,
                    self.cfg.model.custom.warmup_train_steps,
                    self.cfg.model.custom.warmup_validation_steps,
                )
                if self.cfg.model.fp8 and self.cfg.model.custom.reset_fp8_stats_after_warmup:
                    reset_fp8_state(pl_module)
            except Exception as e:
                if trainer.global_rank == 0:
                    print(f"[training_warmup] FAILED (non-fatal): {e}", flush=True)
            for m in trainer.model.modules():
                if hasattr(m, "_mxfp4_cache_disabled"):
                    del m._mxfp4_cache_disabled
                if hasattr(m, "_mxfp4_weight_cache"):
                    del m._mxfp4_weight_cache
            torch.cuda.empty_cache()

    def _log_throughput(self, global_step):
        throughput = self.throughput_timer.get_throughput()
        self.mllogger.event(
            key="tracked_stats",
            metadata={"step": global_step * self.gbs},
            value={"throughput": throughput},
        )


class MetricsLogger(MLPerfLogger):
    def __init__(
        self,
        cfg: DictConfig,
        model: pl.LightningModule,
    ):
        super().__init__(CustomCallback, model, "val_loss_sum", 0.925, "min", cfg)
        self.gbs = cfg.model.global_batch_size
        self.mbs = cfg.model.micro_batch_size
        self.cfg = cfg

    def __deepcopy__(self, memo):
        output = MetricsLogger(self.cfg, self.model)
        if self.trainer is not None:
            output.trainer = self.trainer
        return output

    def compute_hyperparams(
        self, params: dict[str, Any], *args, **kwargs
    ) -> dict[str, Any]:
        return {
            self.mllogger.constants.GLOBAL_BATCH_SIZE: self.gbs,
            self.mllogger.constants.TRAIN_SAMPLES: np.load(
                f"{self.cfg.data_root}/train.npy", allow_pickle=True
            ).shape[0],
            self.mllogger.constants.EVAL_SAMPLES: np.load(
                f"{self.cfg.data_root}/validation.npy", allow_pickle=True
            ).shape[0],
            self.mllogger.constants.GRADIENT_ACCUMULATION_STEPS: int(os.getenv("MINIBS", "1"))
            // self.mbs,
        }

    def compute_validation_metric(self, metrics: dict[str, float]) -> float:
        return metrics["val_loss_sum"] / metrics["val_loss_count"]
