# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
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

import os
import time
from typing import Any
from pprint import pformat

import lightning as pl
import torch
from warmup import run_training_warmup, reset_fp8_state

from mlperf_common.callbacks import LoggingCallback, MLPerfLogger
from omegaconf import DictConfig, OmegaConf
from flux_logging import get_rank_zero_logger

rank = int(os.environ.get("RANK", "0"))
logger = get_rank_zero_logger(__name__, rank)

class ThroughputTimer:
    def __init__(self, gbs):
        self.start_time = None
        self.gbs = gbs

    def start(self):
        if self.start_time is None:
            self.start_time = time.time()

    def compute_throughput(self, global_steps):
        if self.start_time is None:
            raise ValueError("Timer has not been started.")
        elapsed_time = time.time() - self.start_time
        samples = global_steps * self.gbs
        throughput = samples / elapsed_time
        return throughput


class CustomCallback(LoggingCallback):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.gbs = cfg.data.global_batch_size
        self.iter_time = time.time()
        self.enable_perf_debug = int(os.environ.get("ENABLE_PERF_DEBUG", "0"))
        self.enable_mem_profile = int(os.environ.get("ENABLE_MEMORY_PROFILE", "0"))
        if self.enable_mem_profile:
            torch.cuda.memory._record_memory_history(max_entries=100000)
        self.log_every_n_steps = cfg.trainer.log_every_n_steps

        self.profiler = os.getenv("PROFILER", "")
        self.profiler_warmup_steps = int(os.getenv("PROF_WARMUP_STEPS", "0"))
        self.profiler_active_steps = int(os.getenv("PROF_ACTIVE_STEPS", "0"))
        self.rpd = None
        self.rpd_running = False

        if self.profiler == "rpd":
            logger.info("Using RPD profiler")
            logger.info(f"RPD settings: TOTAL_WARMUP_STEPS: {self.profiler_warmup_steps}, TOTAL_ACTIVE_STEPS: {self.profiler_active_steps}")
            from rpdTracerControl import rpdTracerControl
            rpdTracerControl.setFilename('trace.rpd', append=True)
            self.rpd = rpdTracerControl()

        self.timer_obj = ThroughputTimer(self.gbs)

        # Timer set in on_train_batch_start to exclude warmup from reported TTT
        self.train_start_time = None

    def __deepcopy__(self, memo):
        return CustomCallback(self.cfg)

    def get_train_step_samples_count(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> int:
        return self.gbs

    def warmup(self, trainer, pl_module):
        if self.cfg.warmup.enabled:
            run_training_warmup(
                trainer,
                self.cfg.warmup.train_steps,
                self.cfg.warmup.validation_steps,
            )
            if (
                self.cfg.plugins.fp8 is not None
                and self.cfg.warmup.reset_fp8_stats_after_warmup
            ):
                reset_fp8_state(
                    pl_module,
                    reset_fp8_meta_tensors=self.cfg.plugins.fp8_recipe == "delayed",
                )

    def teardown(self, trainer, pl_module, stage):
        # Release CUDA graph to avoid NCCL hang on exit
        module = trainer.model.module.module.module.module
        layers = module.double_blocks + module.single_blocks
        for layer in layers:
            if hasattr(layer, "cudagraph_manager"):
                layer.cudagraph_manager = None

        try:
            from megatron.core.full_cuda_graph import FullCudaGraphWrapper
            FullCudaGraphWrapper.cuda_graph = None
        except ModuleNotFoundError:
            logger.warning("megatron.core.full_cuda_graph cannot be found")

        super().teardown(trainer, pl_module, stage)

    def on_train_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            logger.info(f"Optimized config:\n{OmegaConf.to_yaml(self.cfg)}")
            logger.info(f"\nMCore config:\n{pformat(trainer.model.config)}")
        super().on_train_start(trainer, pl_module)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.train_start_time is None:
            self.train_start_time = time.time()
        self.timer_obj.start()
        if self.profiler == 'rpd':
            if self.rpd and not self.rpd_running and trainer.global_step >= self.profiler_warmup_steps:
                logger.info("Starting RPD profiler")
                self.rpd.start()
                self.rpd.rangePush('python', 'Training', '')
                self.rpd_running = True

            if self.rpd_running and trainer.global_step > self.profiler_warmup_steps + self.profiler_active_steps:
                logger.info("Stopping RPD profiler")
                self.rpd.rangePop()
                self.rpd.stop()
                self.rpd = None
                self.rpd_running = False

        return super().on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_rank == 0:
            if self.enable_perf_debug:
                torch.cuda.synchronize()
                print(f"iter {trainer.global_step} : {time.time() - self.iter_time}")
                torch.cuda.synchronize()
                self.iter_time = time.time()

            if trainer.global_step == 1 and self.enable_mem_profile:
                torch.cuda.memory._record_memory_history(max_entries=100000)

            if trainer.global_step == 3 and self.enable_mem_profile:
                torch.cuda.memory._dump_snapshot("/results/flux.pickle")
                print("memory profile written to /results/flux.pickle")

        if trainer.global_step % self.log_every_n_steps == 0:
            logger.info(f"train_loss: {outputs['loss'].item()}, lr: {trainer.optimizers[0].param_groups[0]['lr']}, step: {trainer.global_step}, samples_count: {trainer.global_step * self.gbs}")

    def on_train_end(self, trainer, pl_module):
        if self.rpd_running:
            self.rpd.rangePop()
            self.rpd.stop()
            self.rpd = None
            self.rpd_running = False

        super().on_train_end(trainer, pl_module)

        if self.train_start_time is not None and trainer.global_rank == 0:
            duration_minutes = (time.time() - self.train_start_time) / 60.0
            self.mllogger.event(
                key="time_to_train_minutes",
                value=duration_minutes,
            )
            if duration_minutes > 0 and trainer.global_step > 0:
                samples = trainer.global_step * self.gbs
                self.mllogger.event(
                    key="throughput_samples_per_second",
                    value=samples / (duration_minutes * 60.0),
                    metadata={"samples_count": samples},
                )

    def on_validation_end(self, trainer, pl_module):
        result = super().on_validation_end(trainer, pl_module)
        if trainer.global_step > 0:
            self.log_throughput(trainer.global_step)
        return result

    def log_throughput(self, global_step):
        throughput = self.timer_obj.compute_throughput(global_step)
        logger.info(f"Throughput: {throughput}, step: {global_step}, samples_count: {global_step * self.gbs}")

    # Overriding the base class methods to avoid logging to MLPerf logger
    def log_custom_timedelta(self, value_key, step: int = 0):
        return

    def _log_train_step_time(
        self,
        trainer_step: int,
        train_batch_size: int,
    ) -> None:
        return


class MetricsLogger(MLPerfLogger):
    def __init__(
        self,
        cfg: DictConfig,
        model: pl.LightningModule,
    ):
        super().__init__(
            CustomCallback, model, "val_loss_sum", cfg.target_accuracy, "min", cfg
        )
        self.gbs = cfg.data.global_batch_size
        self.mbs = cfg.data.micro_batch_size
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
            self.mllogger.constants.TRAIN_SAMPLES: 1099776,  # TODO: can probably read from dataset index
            self.mllogger.constants.EVAL_SAMPLES: 29696,  # TODO: can probably read from dataset index
            "evaluation_frequency": self.cfg.trainer.val_check_interval,
            self.mllogger.constants.GRADIENT_ACCUMULATION_STEPS: max(
                int(os.getenv("MINIBS", "1")) // self.mbs, 1
            ),
            self.mllogger.constants.OPT_NAME: self.mllogger.constants.ADAMW,
            self.mllogger.constants.OPT_LR_WARMUP_STEPS: (
                self.cfg.optim.lr_scheduler.warmup_steps
                if "lr_scheduler" in self.cfg.optim
                else 0
            ),
            self.mllogger.constants.OPT_ADAMW_BETA_1: self.cfg.optim.config.adam_beta1,
            self.mllogger.constants.OPT_ADAMW_BETA_2: self.cfg.optim.config.adam_beta2,
            self.mllogger.constants.OPT_ADAMW_EPSILON: self.cfg.optim.config.adam_eps,
            self.mllogger.constants.OPT_ADAMW_WEIGHT_DECAY: self.cfg.optim.config.weight_decay,
            self.mllogger.constants.OPT_BASE_LR: self.cfg.optim.config.lr,
            self.mllogger.constants.OPT_GRADIENT_CLIP_NORM: self.cfg.optim.config.clip_grad,
            self.mllogger.constants.TENSOR_PARALLELISM: self.cfg.strategy.tensor_model_parallel_size,
            self.mllogger.constants.PIPELINE_PARALLELISM: self.cfg.strategy.pipeline_model_parallel_size,
            self.mllogger.constants.CONTEXT_PARALLELISM: self.cfg.strategy.context_parallel_size,
            self.mllogger.constants.EXPERT_PARALLELISM: 1,
            self.mllogger.constants.MICRO_BATCH_SIZE: self.mbs,
            self.mllogger.constants.CONFIG_FILENAME: os.getenv("CONFIG_FILENAME", ""),
            "lowest_numerical_precision_in_linear": os.getenv("MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR", ""),
        }

    def compute_validation_metric(self, metrics: dict[str, float]) -> float:
        return metrics["val_loss_sum"] / metrics["val_loss_count"]
