"""Triton autotune pinning helper.

A handful of Triton kernels in this directory have two stable autotune
equilibria on MI350X gfx950 at the yambda shape, and autotuner measurement
noise decides between them per cold start. The pinned config list makes
every cold start pick the same config deterministically.

Set `TRITON_FULL_AUTOTUNE=1` to bypass the pin and re-enable the full
autotune search (useful when validating a new shape, GPU, or Triton version
before re-capturing winners).
"""

import os
from typing import Callable, List

import triton


def pinned_or_full(
    pinned: List[triton.Config],
    full_configs_fn: Callable[[], List[triton.Config]],
) -> List[triton.Config]:
    if os.environ.get("TRITON_FULL_AUTOTUNE", "0") == "1":
        return full_configs_fn()
    return pinned
