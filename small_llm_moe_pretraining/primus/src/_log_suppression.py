# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Reduce framework chatter; launchers retain and filter full stdout/stderr."""

import logging
import os

VERBOSE_LOGS = os.environ.get("MLPERF_VERBOSE_LOGS", "0") == "1"
QUIET_LOGGER_NAMES = (
    "aiter", "megatron", "megatron.core", "megatron.core.utils",
    "megatron.core.rerun_state_machine", "primus", "primus_turbo",
    "transformer_engine", "torch.distributed", "transformers",
    "huggingface_hub", "filelock", "urllib3",
)


def reapply_quiet_logger_levels() -> None:
    """Preserve warnings and errors; leave verbose mode untouched."""
    if not VERBOSE_LOGS:
        for name in QUIET_LOGGER_NAMES:
            logging.getLogger(name).setLevel(logging.WARNING)


def install() -> None:
    if VERBOSE_LOGS:
        return
    for name, value in {
        "AITER_LOG_LEVEL": "WARNING",
        "AITER_LOG_MORE": "0",
        "AITER_LOG_TUNED_CONFIG": "0",
        "TRANSFORMERS_VERBOSITY": "warning",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    }.items():
        os.environ.setdefault(name, value)
    reapply_quiet_logger_levels()


install()
