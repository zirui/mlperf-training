import torch
import os
import pathlib
from dataclasses import dataclass
from lightning.pytorch.profilers import PyTorchProfiler
from torch.profiler import ProfilerActivity


TORCHPROF_OUTPUT_DIR = os.getenv("TORCHPROF_OUTPUT_DIR", "/results/artifacts")

TORCHPROF_OUTPUT = os.getenv("TORCHPROF_OUTPUT", "csv_handler")
TORCHPROF_VERBOSE = os.getenv("TORCHPROF_VERBOSE", 0)
TORCHPROF_MAXROWS = os.getenv("TORCHPROF_MAXROWS", 100)
TORCHPROF_PROFILE_MEMORY = bool(os.getenv("TORCHPROF_PROFILE_MEMORY", 0))
TORCHPROF_WITH_STACK = bool(os.getenv("TORCHPROF_WITH_STACK", 0))
TORCHPROF_RECORD_SHAPES = bool(os.getenv("TORCHPROF_RECORD_SHAPES", 0))
TORCHPROF_WITH_FLOPS = bool(os.getenv("TORCHPROF_WITH_FLOPS", 0))
PROF_WARMUP_STEPS = int(os.getenv("PROF_WARMUP_STEPS", 3))
PROF_ACTIVE_STEPS = int(os.getenv("PROF_ACTIVE_STEPS", 2))
PROF_REPITIONS = int(os.getenv("PROF_REPITIONS", 1))


@dataclass
class TorchProfConfig:
    skip_first = 1
    wait = 0
    warmup = PROF_WARMUP_STEPS
    active = PROF_ACTIVE_STEPS
    repeat = PROF_REPITIONS


TOTAL_WARMUP_STEPS = TorchProfConfig.skip_first + \
    TorchProfConfig.wait + \
    TorchProfConfig.warmup

TOTAL_ACTIVE_STEPS = TorchProfConfig.active

def trace_handler(prof):
    save_path = f"{TORCHPROF_OUTPUT_DIR}/key_avg_{prof.step_num}_{torch.distributed.get_rank()}.txt"
    print(f"Saving torchprof results at: {save_path}")
    with open(save_path, 'w') as f:
        output = prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=TORCHPROF_MAXROWS)
        f.write(output)
        if TORCHPROF_VERBOSE:
            print(output)

    prof.export_chrome_trace(f"{TORCHPROF_OUTPUT_DIR}/trace_{prof.step_num}_{torch.distributed.get_rank()}.json")


def _get_torchprof():
    if TORCHPROF_OUTPUT == "csv_handler":
        output_handler = trace_handler
    elif TORCHPROF_OUTPUT == "tensorboard":
        output_handler = torch.profiler.tensorboard_trace_handler(TORCHPROF_OUTPUT_DIR)
    else:
        raise ValueError("Invalid Output Handler for TorchProf.")
    pathlib.Path(TORCHPROF_OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

    profiler = PyTorchProfiler(
        dirpath=TORCHPROF_OUTPUT_DIR,
        filename="torchprof",
        group_by_input_shapes=True,
        emit_nvtx=False,
        export_to_chrome=True,
        row_limit=-1,
        record_module_names=True,
        **
        {
            'profile_memory' : TORCHPROF_PROFILE_MEMORY,
            'with_flops': TORCHPROF_WITH_FLOPS,
            'with_stack' : TORCHPROF_WITH_STACK,
            'record_shapes' : TORCHPROF_RECORD_SHAPES,
            'activities' : [ProfilerActivity.CPU, ProfilerActivity.CUDA],
            'schedule' : torch.profiler.schedule(
                    skip_first=TorchProfConfig.skip_first,
                    wait=TorchProfConfig.wait,
                    warmup=TorchProfConfig.warmup,
                    active=TorchProfConfig.active,
                    repeat=TorchProfConfig.repeat
                ),
            'on_trace_ready': output_handler,
        },
    )

    return profiler

def get_profiler():
    if os.getenv("PROFILER", '') == 'torchprof':
        return _get_torchprof()
    return None
