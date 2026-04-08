# Copyright (c) 2025, NVIDIA CORPORATION & AMD. All rights reserved.
#
# MLPerf logging callbacks for Flux training.
# Adapted from flux1/nemo/src/custom_callbacks.py.
# Uses mlperf_logging.mllog directly -- no NeMo, PyTorch Lightning, or
# mlperf_common dependency (which requires pytorch_lightning).

import logging
import os
import time

from mlperf_logging import mllog

logger = logging.getLogger(__name__)

_mllogger = mllog.get_mllogger()
_constants = mllog.constants


class ThroughputTimer:
    """Wall-clock throughput tracker for Megatron hook-based training.

    Instead of per-iteration start/stop (which requires Lightning-style
    callbacks), this tracks:
      - training_start_time: wall-clock when training began
      - eval_cumulative_secs: total seconds spent inside evaluation
      - consumed_samples: updated by the training_log hook each iteration

    throughput        = consumed_samples / (wall_clock - eval_time)
    combined_throughput = consumed_samples / wall_clock
    """

    def __init__(self, gbs: int):
        self.gbs = gbs
        self.training_start_time: float | None = None
        self.eval_cumulative_secs: float = 0.0
        self._eval_enter_time: float | None = None
        self.consumed_samples: int = 0

    def mark_training_start(self):
        if self.training_start_time is None:
            self.training_start_time = time.time()

    def update_samples(self, iteration: int):
        self.consumed_samples = iteration * self.gbs

    def pause_for_eval(self):
        self._eval_enter_time = time.time()

    def resume_after_eval(self):
        if self._eval_enter_time is not None:
            self.eval_cumulative_secs += time.time() - self._eval_enter_time
            self._eval_enter_time = None

    def compute_throughput(self):
        """Samples/sec excluding evaluation time."""
        if self.training_start_time is None:
            return 0.0
        wall = time.time() - self.training_start_time
        training_secs = wall - self.eval_cumulative_secs
        if training_secs <= 0:
            return 0.0
        return self.consumed_samples / training_secs

    def compute_combined_throughput(self):
        """Samples/sec including evaluation time."""
        if self.training_start_time is None:
            return 0.0
        wall = time.time() - self.training_start_time
        if wall <= 0:
            return 0.0
        return self.consumed_samples / wall


def _log_event(key, value=None, metadata=None):
    _mllogger.event(key=key, value=value, metadata=metadata)


def _log_start(key, value=None, metadata=None):
    _mllogger.start(key=key, value=value, metadata=metadata)


def _log_end(key, value=None, metadata=None):
    _mllogger.end(key=key, value=value, metadata=metadata)


class FluxMLPerfLogger:
    """
    Standalone MLPerf logger for Flux training.

    Uses mlperf_logging.mllog directly (no mlperf_common, no PyTorch Lightning).
    """

    def __init__(
        self,
        global_batch_size: int,
        micro_batch_size: int,
        target_accuracy: float = 0.586,
        log_every_n_steps: int = 10,
    ):
        self.gbs = global_batch_size
        self.mbs = micro_batch_size
        self.target_accuracy = target_accuracy
        self.log_every_n_steps = log_every_n_steps
        self.timer = ThroughputTimer(global_batch_size)
        self._converged = False

        self.profiler = os.getenv("PROFILER", "")
        self.profiler_warmup_steps = int(os.getenv("PROF_WARMUP_STEPS", "0"))
        self.profiler_active_steps = int(os.getenv("PROF_ACTIVE_STEPS", "0"))
        self.rpd = None
        self.rpd_running = False

        if self.profiler == "rpd":
            try:
                from rpdTracerControl import rpdTracerControl
                rpdTracerControl.setFilename("trace.rpd", append=True)
                self.rpd = rpdTracerControl()
                logger.info("RPD profiler initialized")
            except ImportError:
                logger.warning("rpdTracerControl not available")

    def log_init(self, num_nodes: int, seed: int):
        if int(os.environ.get("RANK", "0")) == 0:
            _log_start(key=_constants.INIT_START)
            _log_event(key=_constants.SUBMISSION_BENCHMARK, value="flux1")
            _log_event(key=_constants.SUBMISSION_ORG,
                       value=os.environ.get("MLLOG_SUBMISSION_ORG", "AMD"))
            _log_event(key=_constants.SUBMISSION_DIVISION,
                       value=os.environ.get("MLLOG_SUBMISSION_DIVISION", "closed"))
            _log_event(key=_constants.SUBMISSION_PLATFORM,
                       value=os.environ.get("MLLOG_SUBMISSION_PLATFORM", "MI355X"))
            _log_event(key=_constants.SUBMISSION_STATUS, value="onprem")
            _log_event(key="target_accuracy", value=self.target_accuracy)
            _log_event(key=_constants.SEED, value=seed)

    def log_hyperparams(self, config):
        """Log MLPerf-required hyperparameters."""
        if int(os.environ.get("RANK", "0")) != 0:
            return
        _log_event(key=_constants.GLOBAL_BATCH_SIZE, value=self.gbs)
        _log_event(key=_constants.TRAIN_SAMPLES, value=1099776)
        _log_event(key=_constants.EVAL_SAMPLES, value=29696)
        _log_event(
            key=_constants.GRADIENT_ACCUMULATION_STEPS,
            value=max(int(os.getenv("MINIBS", "1")) // self.mbs, 1),
        )
        _log_event(key=_constants.OPT_NAME, value="adamw")

    def log_init_stop_run_start(self):
        """Log INIT_STOP and RUN_START back-to-back.

        Called once from the training_log hook on the first iteration,
        which is the earliest point after Megatron has finished internal
        initialization (model built, data loaders created, optimizer set up).
        """
        if int(os.environ.get("RANK", "0")) == 0:
            _log_end(key=_constants.INIT_STOP)
            _log_start(key=_constants.RUN_START)

    def on_train_batch_end(self, global_step: int, loss: float, lr: float):
        self.timer.mark_training_start()
        self.timer.update_samples(global_step)

        self._handle_profiler(global_step)

        if int(os.environ.get("RANK", "0")) != 0:
            return
        if global_step % self.log_every_n_steps == 0:
            _log_event(
                key="tracked_stats",
                value={"train_loss": loss},
                metadata={
                    "samples_count": global_step * self.gbs,
                    "lr": lr,
                    "step": global_step,
                },
            )

    def on_validation_start(self, global_step: int):
        self.timer.update_samples(global_step)
        self.timer.pause_for_eval()

        if global_step > 0 and int(os.environ.get("RANK", "0")) == 0:
            throughput = self.timer.compute_throughput()
            _log_event(
                key="throughput",
                value=throughput,
                metadata={
                    "samples_count": global_step * self.gbs,
                    "step": global_step,
                },
            )

    def on_validation_end(self, global_step: int, val_loss: float):
        self.timer.resume_after_eval()

        if int(os.environ.get("RANK", "0")) == 0:
            _log_event(
                key="eval_accuracy",
                value=val_loss,
                metadata={
                    "samples_count": global_step * self.gbs,
                    "step": global_step,
                },
            )
            combined_throughput = self.timer.compute_combined_throughput()
            _log_event(
                key="combined_throughput",
                value=combined_throughput,
                metadata={
                    "samples_count": global_step * self.gbs,
                    "step": global_step,
                },
            )

    def _handle_profiler(self, global_step: int):
        if self.profiler != "rpd":
            return
        if self.rpd and not self.rpd_running and global_step >= self.profiler_warmup_steps:
            logger.info("Starting RPD profiler")
            self.rpd.start()
            self.rpd.rangePush("python", "Training", "")
            self.rpd_running = True
        if self.rpd_running and global_step > self.profiler_warmup_steps + self.profiler_active_steps:
            logger.info("Stopping RPD profiler")
            self.rpd.rangePop()
            self.rpd.stop()
            self.rpd = None
            self.rpd_running = False

    @property
    def converged(self):
        return self._converged

    def log_run_stop(self, success: bool, global_step: int):
        if success:
            self._converged = True
        if int(os.environ.get("RANK", "0")) == 0:
            status = "success" if success else "aborted"
            _log_end(
                key=_constants.RUN_STOP,
                value=status,
                metadata={
                    "samples_count": global_step * self.gbs,
                    "step": global_step,
                    "status": status,
                },
            )

    def teardown(self):
        if self.rpd_running and self.rpd:
            self.rpd.rangePop()
            self.rpd.stop()
            self.rpd = None
            self.rpd_running = False
