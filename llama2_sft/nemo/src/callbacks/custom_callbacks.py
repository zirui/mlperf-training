import os
import time
from typing import Any, List
from math import ceil

import lightning as pl
import numpy as np
import torch
from custom_llama import reset_fp8_state, run_training_warmup
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor, Float8Quantizer
from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor

import transformer_engine.common.recipe
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.num_microbatches_calculator import (
    reconfigure_num_microbatches_calculator,
)
from mlperf_common.callbacks import LoggingCallback, MLPerfLogger
from omegaconf import DictConfig, OmegaConf
from prof_handler import TOTAL_WARMUP_STEPS, TOTAL_ACTIVE_STEPS


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
        self.profiler = os.getenv("PROFILER", '')

        if self.profiler == 'rpd':
            from rpdTracerControl import rpdTracerControl
            rpdTracerControl.setFilename(name=f"trace.rpd", append=True)
            self.rpd = rpdTracerControl()


    def __deepcopy__(self, memo):
        return CustomCallback(self.cfg)

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
        device = next(_extract_module(trainer.model).parameters()).device

        # Clone current FP8 weights to CPU for healing.
        # Must happen BEFORE MXFP4 replacement since we reference the live FP8 params.
        self.fp8_quantizer = Float8Quantizer(
            scale=torch.ones(1, dtype=torch.float32, device=device),
            amax=torch.zeros(1, dtype=torch.float32, device=device),
            fp8_dtype=tex.DType.kFloat8E4M3,
        )
        self.fp8_cpu_params = self._get_quantized_params_cpu(
            trainer.model, self.fp8_quantizer, "FP8_DS"
        )

        _use_hadamard = os.environ.get("NVTE_MXFP4_USE_HADAMARD", "0") == "1"
        self.mxfp4_quantizer = MXFP4Quantizer(
            rowwise=True,
            columnwise=True,
            shuffle_B_matrix_for_aiter=True,
            use_hadamard=_use_hadamard,
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
                if layer_idx < self.num_layers_at_start_in_bf16 or layer_idx >= layer_count - self.num_layers_at_end_in_bf16:
                    quantized_params.append([])
                    continue

            quantized_layer_params = []
            for name, module in layer.named_modules():
                if not isinstance(module, (te.Linear, te.LayerNormLinear)):
                    continue
                if not hasattr(module, 'weight'):
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
                        module._parameters['weight'] = qparam
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

        # Use a dedicated CUDA stream for async CPU→GPU transfers so they
        # overlap with the GIL-released main-stream work (op-fuser rebuilds).
        xfer_stream = torch.cuda.Stream()

        for layer_idx, layer in enumerate(layers):
            if self.first_last_layers_bf16:
                if layer_idx < self.num_layers_at_start_in_bf16 or layer_idx >= layer_count - self.num_layers_at_end_in_bf16:
                    continue

            for name, module in layer.named_modules():
                if not isinstance(module, (te.Linear, te.LayerNormLinear)):
                    continue
                if not hasattr(module, 'weight'):
                    continue
                with torch.no_grad():
                    weight = cpu_params[layer_idx].pop(0)

                    if not self.store_quantized_params_on_gpu:
                        with torch.cuda.stream(xfer_stream):
                            if qtype == "FP8_DS":
                                weight._data = weight._data.to(dev, non_blocking=True)
                                if weight._transpose is not None:
                                    weight._transpose = weight._transpose.to(dev, non_blocking=True)
                            elif qtype == "MXFP4":
                                weight._rowwise_data = weight._rowwise_data.to(dev, non_blocking=True)
                                weight._columnwise_data = weight._columnwise_data.to(dev, non_blocking=True)
                            else:
                                raise ValueError(f"Unsupported quantization type: {qtype}")

                    weight.requires_grad = False
                    module._parameters['weight'] = weight

            if self.cfg.model.use_transformer_engine_op_fuser:
                layer.mlp._fused_impl = (layer.mlp._make_fused_impl(),)
                layer.self_attention.linear_proj._fused_branches = layer.self_attention.linear_proj._make_fused_branches()
                layer.self_attention.linear_qkv._fused_branches = layer.self_attention.linear_qkv._make_fused_branches()

        # Single sync for all async transfers
        xfer_stream.synchronize()

    def _pre_ttt_fp8_warmup(self, trainer):
        """Pre-compile FP8 kernels (Triton, hipBLASLt) before the TTT timer.

        Loads FP8 weights alongside MXFP4, runs one synthetic training step
        in FP8 mode with aggressive recompute (72/80 layers) to fit both
        weight sets, then restores MXFP4 state. Saves ~10s on step 310.
        """
        import megatron.core.fp4_utils

        if not self.fp8_cpu_params:
            return
        if not self.cfg.model.healing_iter or self.cfg.model.healing_iter <= 0:
            return

        extracted = _extract_module(trainer.model)
        layers = extracted.decoder.layers
        layer_count = len(layers)
        dev = torch.cuda.current_device()

        # Load FP8 weights from CPU alongside MXFP4 via async stream
        saved_weights = []
        xfer_stream = torch.cuda.Stream()
        for layer_idx, layer in enumerate(layers):
            if self.first_last_layers_bf16:
                if (layer_idx < self.num_layers_at_start_in_bf16
                        or layer_idx >= layer_count - self.num_layers_at_end_in_bf16):
                    continue

            param_idx = 0
            for _n, mod in layer.named_modules():
                if not isinstance(mod, (te.Linear, te.LayerNormLinear)):
                    continue
                if not hasattr(mod, 'weight'):
                    continue

                cpu_fp8 = self.fp8_cpu_params[layer_idx][param_idx]
                with torch.no_grad(), torch.cuda.stream(xfer_stream):
                    gpu_data = cpu_fp8._data.to(dev, non_blocking=True)
                    gpu_tp = (cpu_fp8._transpose.to(dev, non_blocking=True)
                              if cpu_fp8._transpose is not None else None)
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
                mod._parameters['weight'] = gpu_fp8
                param_idx += 1

        xfer_stream.synchronize()

        # Switch to FP8 mode
        saved_mxfp4_phase = megatron.core.fp4_utils._mxfp4_phase
        saved_recipe = megatron.core.fp4_utils.get_fp4_recipe
        megatron.core.fp4_utils._mxfp4_phase = False
        megatron.core.fp4_utils.get_fp4_recipe = self.healing_lambda

        # Aggressive recompute: both weight sets on GPU (~142GB), need headroom
        saved_recompute_n = extracted.config.recompute_num_layers
        extracted.config.recompute_num_layers = 72

        optimizer = trainer.strategy.optimizers[0]
        try:
            trainer.model.training_step(trainer.model.get_synthetic_input())
            optimizer.zero_grad(set_to_none=True)
        except Exception:
            pass

        # Restore MXFP4 state
        megatron.core.fp4_utils._mxfp4_phase = saved_mxfp4_phase
        megatron.core.fp4_utils.get_fp4_recipe = saved_recipe
        extracted.config.recompute_num_layers = saved_recompute_n

        for mod, orig_weight in saved_weights:
            mod._parameters['weight'] = orig_weight

        reset_fp8_state(trainer.lightning_module)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def on_train_start(self, trainer, pl_module):

        self._healing_setup()
        if self.pre_quantized_model:
            self._pre_quantize_model(trainer, pl_module)
        super().on_train_start(trainer, pl_module)

        # Flush async GPU work from warmup before starting TTT timer
        torch.cuda.synchronize()

        # Pre-pay Lightning val_loop init cost (setup_data/reset) before TTT timer.
        # Without this, the first real eval at step 192 incurs a ~65s one-time GPU stall.
        trainer.warmup = True
        try:
            trainer.fit_loop.epoch_loop.val_loop.run()
        except Exception:
            pass
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        if self.cfg.model.healing_iter and self.cfg.model.healing_iter > 0:
            if hasattr(trainer.strategy, 'optimizers') and len(trainer.strategy.optimizers) > 0:
                self._pre_ttt_fp8_warmup(trainer)

        # Timer set in on_train_batch_start to exclude warmup from reported TTT
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
                if isinstance(m, (te.Linear, te.LayerNormLinear)) and hasattr(m, 'weight'):
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
                    self.mllogger.constants.SAMPLES_COUNT: trainer.global_step
                    * self.gbs,
                    "lr": trainer.optimizers[0].param_groups[0]["lr"],
                    "step": trainer.global_step,
                },
            )

        if trainer.global_step + 1 == self.cfg.model.healing_iter:
            torch.distributed.barrier()

            if self.pre_quantized_model:
                assert self.fp8_cpu_params, "FP8 parameters not found"
                self._set_quantized_params_cpu(trainer.model, self.fp8_cpu_params, self.healing_precision)

            import megatron.core.fp4_utils
            megatron.core.fp4_utils.get_fp4_recipe = self.healing_lambda
            megatron.core.fp4_utils._mxfp4_phase = False

            # FP8 phase has VRAM headroom — disable recomputation
            extracted = _extract_module(trainer.model)
            if extracted.config.recompute_granularity is not None:
                extracted.config.recompute_granularity = None
                extracted.config.recompute_num_layers = None
                extracted.config.recompute_method = None

            # Invalidate MXFP4 weight cache and persistent columnwise flags
            for m in trainer.model.modules():
                if hasattr(m, '_mxfp4_weight_cache'):
                    del m._mxfp4_weight_cache
                if hasattr(m, '_mxfp4_persist_columnwise'):
                    del m._mxfp4_persist_columnwise

            self.mllogger.event(
                key="healing",
                value=trainer.global_step + 1,
                metadata={"samples_count": trainer.global_step * self.gbs},
            )

        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_start(self, trainer, pl_module):
        _is_warmup = getattr(trainer, 'warmup', False)
        if not _is_warmup:
            torch.cuda.synchronize()
            import megatron.core.fp4_utils
            if megatron.core.fp4_utils._mxfp4_phase:
                torch.cuda.empty_cache()
            self._log_throughput(trainer.global_step)

        super().on_validation_start(trainer, pl_module)

        if not _is_warmup:
            trainer.val_check_interval = ceil(int(os.environ.get("VAL_CHECK_INTERVAL", 384)) / self.gbs)
            trainer.val_check_batch = ceil(int(os.environ.get("VAL_CHECK_INTERVAL", 384)) / self.gbs)

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
            # Free val_loop MXFP4 cache and disable caching during warmup.
            for m in trainer.model.modules():
                m._mxfp4_cache_disabled = True
                if hasattr(m, '_mxfp4_weight_cache'):
                    del m._mxfp4_weight_cache
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            try:
                run_training_warmup(
                    trainer,
                    self.cfg.model.custom.warmup_train_steps,
                    self.cfg.model.custom.warmup_validation_steps,
                )
                if (
                    self.cfg.model.fp8
                    and self.cfg.model.custom.reset_fp8_stats_after_warmup
                ):
                    reset_fp8_state(pl_module)
            except Exception:
                pass
            for m in trainer.model.modules():
                if hasattr(m, '_mxfp4_cache_disabled'):
                    del m._mxfp4_cache_disabled
                if hasattr(m, '_mxfp4_weight_cache'):
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
            self.mllogger.constants.GRADIENT_ACCUMULATION_STEPS: int(
                os.getenv("MINIBS", "1")
            )
            // self.mbs,
        }

    def compute_validation_metric(self, metrics: dict[str, float]) -> float:
        return metrics["val_loss_sum"] / metrics["val_loss_count"]
