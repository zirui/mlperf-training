# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Triton autotune pinning helper.

Kernels that call `pinned_or_full` use the listed config. Set
`TRITON_FULL_AUTOTUNE=1` to search the full config list instead.
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
