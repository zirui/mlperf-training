# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

# pyre-ignore-all-errors

import functools
import os
import sys

import torch


class _PlainFuncWrapper:
    """Thin wrapper around a plain function that provides no-op register_fake
    and register_kernel methods, mirroring the CustomOpDef API so that
    downstream @func.register_fake / func.register_kernel("cpu") calls
    don't break when the function is not wrapped as a custom op."""

    def __init__(self, func):
        self._func = func
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def register_fake(self, fake_func):
        return fake_func

    def register_kernel(self, device):
        def inner(func):
            return func

        return inner


def maybe_register_custom_op(op_name, mutates_args):
    """
    Conditionally registers a function as a torch custom op.

    When AOTI_LOWER is set in the environment, the function is returned
    unwrapped so that torch.export / Dynamo can trace through the plain
    Python implementation instead of treating the custom op as opaque.
    """

    def decorator(func):
        if os.environ.get("AOTI_LOWER"):
            return _PlainFuncWrapper(func)
        return torch.library.custom_op(op_name, func, mutates_args=mutates_args)

    return decorator


def is_sm100_plus() -> bool:
    """
    Check if this is a Blackwell Datacenter GPU.
    These are between 100 and 103 for B200-GB300.
    """
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major == 10 and (props.minor >= 0 and props.minor <= 3)


def is_sm90() -> bool:
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major == 9 and props.minor == 0


def is_sm90_plus() -> bool:
    return is_sm100_plus() or is_sm90()


@functools.lru_cache(maxsize=None)
def is_amd_mi350() -> bool:
    """Detect an AMD Instinct MI350-class GPU (gfx950) running under ROCm.

    MI350 benefits from the same multi-row, separated-RNG layer-norm-mul-dropout
    path as Blackwell datacenter parts (sm_100), so it is gated together with
    is_sm100_plus() at the kernel dispatch sites.
    """
    if not torch.cuda.is_available():
        return False
    if getattr(torch.version, "hip", None) is None:
        return False
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName or ""
    except (AssertionError, RuntimeError, AttributeError):
        return False
    return "gfx950" in arch


def use_separated_rng_ln_mul_dropout() -> bool:
    """Hardware that should use the autotuned, multi-row ``_ln_mul_dropout_fwd_rng``
    kernel with a precomputed dropout mask instead of the legacy single-row,
    fused-RNG ``_ln_mul_dropout_fwd`` kernel.

    Blackwell datacenter GPUs (sm_100-103) and AMD MI350 (gfx950) both prefer the
    separated-RNG path: it batches rows per program and lets the backward reuse the
    same mask, rather than launching one program per row.
    """
    return is_sm100_plus() or is_amd_mi350()


def arm_print_all_ranks(msg: str) -> None:
    """Print a one-time arming or decline line on EVERY rank, tagged ' [rank N]'.

    Used by the LayerNorm-family kernels (D256 v3, D512 dx-sum v3, LN*U*dropout
    fwd v3 and maskgen, LPT EC reduce) and by the L11 fused out-proj dgrad +
    LN*U*dropout backward. Multi-process training logs occasionally lose a
    line, and a missing rank-0 line reads as a false 'not armed'. One tagged
    copy per rank makes losing every copy negligible and lets arming be
    checked rank by rank. The text before the tag is unchanged, so marker
    greps still match. Logging only: no computed value changes, and each
    caller still prints once per process.

    One write per line. A launcher that filters the merged output line by line
    (e.g. a sed that drops hipBLASLt's untuned-shape warnings) sees those
    warnings from every rank as unbuffered std::cerr tokens, one write per
    `<<`, so a token without its newline is often pending. A line printed right
    then shares one physical line with the token, and the filter deletes the
    whole line. So the line is written as ONE os.write
    of "\n<msg> [rank N]\n". The leading newline ends any pending token, so the
    message gets a physical line of its own (a write of at most PIPE_BUF bytes
    to a pipe is atomic), and an empty line is the only side effect. Python's
    buffered stdout is flushed first, so the line keeps its place among the
    process's own prints. Without a file descriptor (pytest capsys, a
    replaced sys.stdout) or if the write fails before any byte, it falls back
    to print(). DLRM_HSTU_FP8_ARM_PRINT_ATOMIC=0 restores the plain
    print() (read per call). Logging only.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = str(torch.distributed.get_rank())
    else:
        # RANK is not always exported to the spawned trainer processes.
        rank = os.environ.get("RANK", "0") or "0"
    line = f"{msg} [rank {rank}]"
    if os.environ.get("DLRM_HSTU_FP8_ARM_PRINT_ATOMIC", "1") != "0":
        try:
            sys.stdout.flush()
            fd = sys.stdout.fileno()
            data = memoryview(("\n" + line + "\n").encode("utf-8", "backslashreplace"))
            written = os.write(fd, data)
        except Exception:
            written = 0
        if written:
            # A pipe write of <= PIPE_BUF bytes is all-or-nothing; finish a
            # short write (a longer line or a regular file) rather than repeat it.
            while written < len(data):
                written += os.write(fd, data[written:])
            return
    print(line, flush=True)


def copy_if_different_ptr(dst: torch.Tensor, src: torch.Tensor) -> None:
    if torch.compiler.is_compiling():
        # .data_ptr() will break PT2
        dst.copy_(src)
    else:
        if dst.data_ptr() != src.data_ptr():
            dst.copy_(src)
