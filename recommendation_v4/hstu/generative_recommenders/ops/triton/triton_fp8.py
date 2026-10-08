# Copyright (c) Meta Platforms, Inc. and affiliates.
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

# pyre-unsafe

"""gfx950 FP8 helpers for the fused HSTU training path.

The public helpers in this module consume activations that were already
quantized by a producer epilogue. They intentionally do not provide a
boundary-cast convenience path: Plan 01 proved that such a path is slower than
BF16 at the production shapes.
"""

from __future__ import annotations

import collections
import functools
import json
import logging
import math
import os
import sys
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.autograd.profiler import record_function


logger: logging.Logger = logging.getLogger(__name__)

_FP8_E4M3: Optional[torch.dtype] = getattr(torch, "float8_e4m3fn", None)
_FP8_MAX: float = 448.0

# The master switch. Nothing here gates on it -- _env_bootstrap has already
# expanded it into the individual knobs below by the time this imports -- but it
# is logged, so a run's own log says whether the arm was selected as a unit or
# assembled knob by knob.
_HSTU_FP8_FWD: bool = os.environ.get("DLRM_HSTU_FP8_FWD", "0") == "1"
_HSTU_FP8_GEMM: bool = os.environ.get("DLRM_HSTU_FP8_GEMM", "0") == "1"
_HSTU_FP8_FAST_ACCUM: bool = (
    os.environ.get("DLRM_HSTU_FP8_GEMM_FASTACC", "0") == "1"
)
_HSTU_FP8_FAIL_CLOSED: bool = (
    os.environ.get("DLRM_HSTU_FP8_FAIL_CLOSED", "0") == "1"
)
_HSTU_FP8_HOST_SCALE_VALUES: bool = (
    _HSTU_FP8_GEMM
    and os.environ.get("DLRM_HSTU_FP8_HOST_SCALE_VALUES", "0") == "1"
)
_HSTU_FP8_FUSE_LN: bool = (
    _HSTU_FP8_GEMM
    and os.environ.get("DLRM_HSTU_FP8_FUSE_LN", "0") == "1"
)
_HSTU_FP8_FUSE_OUTLN: bool = (
    _HSTU_FP8_GEMM
    and os.environ.get("DLRM_HSTU_FP8_FUSE_OUTLN", "0") == "1"
)
_HSTU_FP8_FUSE_QKV: bool = (
    _HSTU_FP8_FUSE_LN
    and os.environ.get("DLRM_HSTU_FP8_FUSE_QKV", "0") == "1"
)
_HSTU_FP8_ATTENTION: bool = (
    _HSTU_FP8_FUSE_QKV
    and os.environ.get("DLRM_HSTU_FP8_ATTN", "0") == "1"
)
_HSTU_FP8_BACKWARD: bool = (
    _HSTU_FP8_GEMM
    and os.environ.get("DLRM_HSTU_FP8_BWD", "0") == "1"
)
_HSTU_FP8_FUSE_SILU: bool = (
    _HSTU_FP8_GEMM
    and os.environ.get("DLRM_HSTU_FP8_FUSE_SILU", "0") == "1"
)
_HSTU_FP8_FUSE_RESIDUAL: bool = (
    _HSTU_FP8_GEMM
    and os.environ.get("DLRM_HSTU_FP8_FUSE_RESIDUAL", "0") == "1"
)
_HSTU_FP8_FUSE_RESIDUAL_ALGORITHM: int = int(
    os.environ.get("DLRM_HSTU_FP8_FUSE_RESIDUAL_ALGORITHM", "1")
)
_HSTU_FP8_FUSE_RESIDUAL_WORKSPACE_BYTES: int = 64 * 1024 * 1024
_HSTU_FP8_ASCALE_MARGIN: float = float(
    os.environ.get("DLRM_HSTU_FP8_GEMM_ASCALE_MARGIN", "1.5")
)
_HSTU_FP8_CALIBRATION_STEPS: int = max(
    1, int(os.environ.get("DLRM_HSTU_FP8_CALIBRATION_STEPS", "4"))
)
_HSTU_FP8_SCALE_MODE: str = os.environ.get(
    "DLRM_HSTU_FP8_SCALE_MODE", "frozen"
)
_HSTU_FP8_TELEMETRY_INTERVAL: int = max(
    0, int(os.environ.get("DLRM_HSTU_FP8_TELEMETRY_INTERVAL", "0"))
)


@dataclass
class _WeightCacheEntry:
    ref: weakref.ReferenceType[torch.Tensor]
    version: int
    data_ptr: int
    shape: torch.Size
    quantized_column_major: torch.Tensor
    scale: torch.Tensor


@dataclass
class _PackedWeightCacheEntry:
    ref: weakref.ReferenceType[torch.Tensor]
    version: int
    data_ptr: int
    packed: torch.Tensor


_WeightLayout = Tuple[Tuple[int, int], ...]
_weight_cache: Dict[Tuple[object, ...], _WeightCacheEntry] = {}
_packed_weight_cache: Dict[Tuple[object, ...], _PackedWeightCacheEntry] = {}
_cache_hits: int = 0
_cache_misses: int = 0
_fp8_one_by_device: Dict[int, torch.Tensor] = {}
_scale_value_cache: Dict[Tuple[object, ...], float] = {}
_inverse_scale_cache: Dict[
    Tuple[object, ...], Tuple[float, torch.Tensor]
] = {}
_folded_input_scale_cache: Dict[Tuple[object, ...], torch.Tensor] = {}
_scaled_output_bias_cache: Dict[Tuple[object, ...], torch.Tensor] = {}
_fp8_output_scale_mode_by_device: Dict[int, str] = {}
_telemetry_calls: Dict[str, int] = {}
_telemetry_last: Dict[str, dict[str, float | int | bool]] = {}
_capability_sites: set[str] = set()
_resolved_logged: bool = False
_capability_contracts: Dict[str, dict[str, object]] = {}
_residual_workspaces: Dict[int, torch.Tensor] = {}


def hstu_fp8_gemm_enabled() -> bool:
    return _HSTU_FP8_GEMM


def hstu_fp8_fail_closed() -> bool:
    return _HSTU_FP8_FAIL_CLOSED


def hstu_fp8_fuse_ln_enabled() -> bool:
    return _HSTU_FP8_FUSE_LN


def hstu_fp8_fuse_outln_enabled() -> bool:
    return _HSTU_FP8_FUSE_OUTLN


def hstu_fp8_fuse_qkv_enabled() -> bool:
    return _HSTU_FP8_FUSE_QKV


def hstu_fp8_attention_enabled() -> bool:
    return _HSTU_FP8_ATTENTION


def refuse_fp8_with_flydsl_attention_forward() -> None:
    """No-op. Kept so old call sites and tests still import a named gate.

    ``DLRM_HSTU_FP8_FWD=1`` plus ``HSTU_ATTN_FWD=flydsl`` (the gin default)
    selects the FlyDSL e4m3 attention forward (fused-silu epilogue included).
    ``HSTU_ATTN_FWD=triton`` is the opt-out to the Triton e4m3 forward.
    Backward selection (``HSTU_ATTN_BWD``) is independent and unchanged.
    """
    return


def hstu_fp8_backward_enabled() -> bool:
    return _HSTU_FP8_BACKWARD


def hstu_fp8_fuse_silu_enabled() -> bool:
    return _HSTU_FP8_FUSE_SILU


def hstu_fp8_fuse_residual_enabled() -> bool:
    return _HSTU_FP8_FUSE_RESIDUAL


def hstu_fp8_scale_mode() -> str:
    return _HSTU_FP8_SCALE_MODE


def fp8_e4m3_dtype() -> torch.dtype:
    if _FP8_E4M3 is None:
        raise RuntimeError("torch.float8_e4m3fn is unavailable")
    return _FP8_E4M3


def fp8_scale_state_size() -> int:
    """Scale, completed-update count, and bounded amax history."""

    return 2 + _HSTU_FP8_CALIBRATION_STEPS


def fp8_scale_tensor(scale_state: torch.Tensor) -> torch.Tensor:
    if scale_state.numel() < 1:
        raise ValueError("FP8 scale state cannot be empty")
    return scale_state.reshape(-1)[:1].reshape(1, 1)


def fp8_supported(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    *,
    input_columns: Optional[int] = None,
) -> bool:
    if (
        not _HSTU_FP8_GEMM
        or _FP8_E4M3 is None
        or torch.version.hip is None
        or not hasattr(torch, "_scaled_mm")
        or not x.is_cuda
        or x.dim() != 2
    ):
        return False
    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    arch = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    if arch != "gfx950":
        return False
    try:
        _fp8_output_scale_mode(x.device)
    except Exception:
        return False
    if weight is None:
        return x.shape[1] % 16 == 0
    k = x.shape[1] if input_columns is None else input_columns
    return (
        weight.is_cuda
        and weight.dim() == 2
        and k == weight.shape[0]
        and k % 16 == 0
        and weight.shape[1] % 16 == 0
    )


def calibrated_scale(
    value: torch.Tensor,
    *,
    margin: Optional[float] = None,
    fp8_max: float = _FP8_MAX,
) -> torch.Tensor:
    """Return a scalar FP32 dequantization scale.

    This helper is for bounded calibration/diagnostics. It must not be called
    on every steady-state HSTU invocation.
    """

    scale_margin = _HSTU_FP8_ASCALE_MARGIN if margin is None else margin
    # Reduce in the producer dtype and cast only the scalar. Casting a
    # production [M, D] activation to FP32 here would add a multi-GiB warmup
    # allocation without improving an amax statistic.
    amax = value.detach().abs().amax().float().clamp_min(1.0e-12)
    return (amax * scale_margin / fp8_max).reshape(1, 1)


def _scale_key(scale: torch.Tensor) -> Tuple[object, ...]:
    device_index = scale.device.index
    if device_index is None:
        device_index = -1
    return (
        id(scale),
        device_index,
        scale.untyped_storage().data_ptr(),
        scale.storage_offset(),
        int(getattr(scale, "_version", 0)),
    )


def activation_scale_value(scale: Optional[torch.Tensor]) -> Optional[float]:
    """Return a cached positive scale without a steady-state D2H sync."""

    if scale is None or scale.numel() < 1:
        return None
    key = _scale_key(scale)
    cached = _scale_value_cache.get(key)
    if cached is not None:
        return cached
    value = float(fp8_scale_tensor(scale).detach().item())
    if not math.isfinite(value) or value <= 0.0:
        return None
    _scale_value_cache[key] = value
    return value


def _host_scale_cache_key(
    kind: str,
    device: torch.device,
    *values: float,
) -> Tuple[object, ...]:
    device_index = device.index
    if device_index is None:
        device_index = -1
    stream_id = -1
    if device.type == "cuda":
        stream_id = int(torch.cuda.current_stream(device).cuda_stream)
    return (kind, device.type, device_index, stream_id, *values)


def _trusted_host_scale_values(
    scale_a: torch.Tensor,
    output_scale: torch.Tensor,
    scale_a_value: Optional[float],
    output_scale_value: Optional[float],
) -> Optional[Tuple[float, float]]:
    if not _HSTU_FP8_HOST_SCALE_VALUES:
        return None
    valid = (
        scale_a.device == output_scale.device
        and isinstance(scale_a_value, (int, float))
        and not isinstance(scale_a_value, bool)
        and isinstance(output_scale_value, (int, float))
        and not isinstance(output_scale_value, bool)
    )
    if valid:
        scale_a_host = float(scale_a_value)
        output_scale_host = float(output_scale_value)
        valid = (
            math.isfinite(scale_a_host)
            and scale_a_host > 0.0
            and math.isfinite(output_scale_host)
            and output_scale_host > 0.0
        )
    if valid:
        return (scale_a_host, output_scale_host)
    if _HSTU_FP8_FAIL_CLOSED:
        raise RuntimeError(
            "DLRM_HSTU_FP8_HOST_SCALE_VALUES requires finite positive "
            "host values for scale_a and output_scale on the same device"
        )
    return None


def _fp8_output_scale_mode(device: torch.device) -> str:
    """Probe whether this backend multiplies or divides by scale_result."""

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    cached = _fp8_output_scale_mode_by_device.get(device_index)
    if cached is not None:
        return cached
    dtype = fp8_e4m3_dtype()
    a = torch.ones((16, 16), dtype=torch.bfloat16, device=device).to(dtype)
    b = torch.eye(16, dtype=torch.bfloat16, device=device).to(dtype).t()
    one = torch.ones((1, 1), dtype=torch.float32, device=device)
    two = torch.full((1, 1), 2.0, dtype=torch.float32, device=device)
    observed = float(
        _scaled_mm(
            a,
            b,
            one,
            one,
            out_dtype=dtype,
            scale_result=two,
        )[0, 0]
        .float()
        .item()
    )
    if math.isclose(observed, 2.0, rel_tol=0.0, abs_tol=0.01):
        mode = "multiply"
    elif math.isclose(observed, 0.5, rel_tol=0.0, abs_tol=0.01):
        mode = "divide"
    elif math.isclose(observed, 1.0, rel_tol=0.0, abs_tol=0.01):
        scale_a_observed = float(
            _scaled_mm(
                a,
                b,
                two,
                one,
                out_dtype=dtype,
            )[0, 0]
            .float()
            .item()
        )
        if not math.isclose(
            scale_a_observed, 2.0, rel_tol=0.0, abs_tol=0.01
        ):
            raise RuntimeError(
                "torch._scaled_mm ignored both FP8 scale_result and scale_a: "
                f"observed {observed} and {scale_a_observed}"
            )
        mode = "input_scale"
    else:
        raise RuntimeError(
            "torch._scaled_mm ignored or misapplied FP8 scale_result: "
            f"expected 2.0 or 0.5, observed {observed}"
        )
    _fp8_output_scale_mode_by_device[device_index] = mode
    return mode


def inverse_scale_tensor(
    scale: torch.Tensor,
    *,
    value: Optional[float] = None,
) -> torch.Tensor:
    """Return the backend-specific result scalar for dequant scale ``scale``."""

    if value is None:
        value = activation_scale_value(scale)
        key = _scale_key(scale)
    else:
        key = _host_scale_cache_key("inverse", scale.device, value)
    if value is None or not math.isfinite(value) or value <= 0.0:
        raise RuntimeError("inverse requested before FP8 scale calibration")
    cached = _inverse_scale_cache.get(key)
    if cached is not None and cached[0] == value:
        return cached[1]
    mode = _fp8_output_scale_mode(scale.device)
    if mode == "multiply":
        result_value = 1.0 / value
    elif mode == "divide":
        result_value = value
    else:
        raise RuntimeError(
            "scale_result is unavailable; fold the output scale into scale_a"
        )
    inverse = torch.full(
        (1, 1),
        result_value,
        dtype=torch.float32,
        device=scale.device,
    )
    _inverse_scale_cache[key] = (value, inverse)
    return inverse


def _fold_output_scale_into_input(
    scale_a: torch.Tensor,
    output_scale: torch.Tensor,
    *,
    scale_a_value: Optional[float] = None,
    output_scale_value: Optional[float] = None,
) -> torch.Tensor:
    if scale_a_value is None or output_scale_value is None:
        key = (*_scale_key(scale_a), *_scale_key(output_scale))
        scale_a_value = activation_scale_value(scale_a)
        output_scale_value = activation_scale_value(output_scale)
    else:
        key = _host_scale_cache_key(
            "folded",
            scale_a.device,
            scale_a_value,
            output_scale_value,
        )
    cached = _folded_input_scale_cache.get(key)
    if cached is not None:
        return cached
    if (
        scale_a_value is None
        or output_scale_value is None
        or not math.isfinite(scale_a_value)
        or scale_a_value <= 0.0
        or not math.isfinite(output_scale_value)
        or output_scale_value <= 0.0
    ):
        raise RuntimeError("FP8 output requested before scale calibration")
    folded = torch.full(
        (1, 1),
        scale_a_value / output_scale_value,
        dtype=torch.float32,
        device=scale_a.device,
    )
    _folded_input_scale_cache[key] = folded
    return folded


def _scale_bias_for_fp8_output(
    bias: torch.Tensor,
    output_scale: torch.Tensor,
    *,
    output_scale_value: Optional[float] = None,
) -> torch.Tensor:
    if output_scale_value is None:
        key = (*_scale_key(bias), *_scale_key(output_scale))
        output_scale_value = activation_scale_value(output_scale)
    else:
        key = (
            *_scale_key(bias),
            *_host_scale_cache_key(
                "scaled_bias",
                bias.device,
                output_scale_value,
            ),
        )
    cached = _scaled_output_bias_cache.get(key)
    if cached is not None:
        return cached
    if (
        output_scale_value is None
        or not math.isfinite(output_scale_value)
        or output_scale_value <= 0.0
    ):
        raise RuntimeError("FP8 bias requested before output scale calibration")
    scaled = (bias.detach().float() / output_scale_value).to(torch.bfloat16)
    _scaled_output_bias_cache[key] = scaled
    return scaled


def calibrate_scale_buffer(
    scale_buffer: torch.Tensor,
    value: torch.Tensor,
    *,
    margin: Optional[float] = None,
    synchronize: bool = True,
) -> float:
    """Update bounded warmup state and publish a frozen scale when complete."""

    if _HSTU_FP8_SCALE_MODE != "frozen":
        raise RuntimeError(
            f"unsupported Plan 08 FP8 scale mode: {_HSTU_FP8_SCALE_MODE}"
        )
    scale_margin = _HSTU_FP8_ASCALE_MARGIN if margin is None else margin
    amax = (
        value.detach().abs().amax().float()
        if value.numel()
        else torch.zeros((), dtype=torch.float32, device=scale_buffer.device)
    )
    if (
        synchronize
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    ):
        torch.distributed.all_reduce(amax, op=torch.distributed.ReduceOp.MAX)
    if float(amax.item()) <= 0.0:
        existing = activation_scale_value(scale_buffer)
        return 0.0 if existing is None else existing
    amax.clamp_min_(1.0e-12)

    flat_state = scale_buffer.reshape(-1)
    if flat_state.numel() == 1:
        scale = amax * scale_margin / _FP8_MAX
        flat_state[:1].copy_(scale.reshape(1))
        result = float(scale.item())
    else:
        if flat_state.numel() < 3:
            raise ValueError(
                "FP8 scale state must contain scale, count, and amax history"
            )
        history = flat_state[2:]
        completed = min(int(flat_state[1].item()), history.numel())
        if completed < history.numel():
            history[completed].copy_(amax)
            completed += 1
            flat_state[1].fill_(completed)
        if completed < history.numel():
            flat_state[0].zero_()
            _scale_value_cache.pop(_scale_key(scale_buffer), None)
            return 0.0
        scale = history[:completed].amax() * scale_margin / _FP8_MAX
        flat_state[0].copy_(scale)
        result = float(scale.item())
    _scale_value_cache[_scale_key(scale_buffer)] = result
    return result


def clear_fp8_scale_runtime_cache() -> None:
    _scale_value_cache.clear()
    _inverse_scale_cache.clear()
    _folded_input_scale_cache.clear()
    _scaled_output_bias_cache.clear()


def quantize_with_scale(
    value: torch.Tensor,
    scale: torch.Tensor,
    *,
    column_major: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    target_dtype = fp8_e4m3_dtype() if dtype is None else dtype
    max_value = torch.finfo(target_dtype).max
    if column_major:
        transposed = value.detach().t().contiguous()
        return (
            (transposed / scale)
            .clamp(-max_value, max_value)
            .to(target_dtype)
            .t()
        )
    return (
        (value.detach().contiguous() / scale)
        .clamp(-max_value, max_value)
        .to(target_dtype)
    )


def _normalize_layout(
    weight: torch.Tensor,
    layout: Optional[_WeightLayout],
) -> _WeightLayout:
    if layout is None:
        return ((0, weight.shape[1]),)
    if not layout:
        raise ValueError("FP8 weight layout cannot be empty")
    for start, end in layout:
        if start < 0 or end <= start or end > weight.shape[1]:
            raise ValueError(
                f"invalid FP8 weight column range {(start, end)} for "
                f"shape {tuple(weight.shape)}"
            )
    return layout


def _weight_key(
    weight: torch.Tensor,
    layout: _WeightLayout,
) -> Tuple[object, ...]:
    device_index = weight.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return (
        id(weight),
        device_index,
        tuple(weight.shape),
        tuple(weight.stride()),
        layout,
    )


def _weight_version(weight: torch.Tensor) -> int:
    return int(getattr(weight, "_version", 0))


def _pack_weight_layout(weight: torch.Tensor, layout: _WeightLayout) -> torch.Tensor:
    if layout == ((0, weight.shape[1]),):
        return weight
    if len(layout) == 1:
        start, end = layout[0]
        return weight[:, start:end]
    packed = torch.empty(
        (weight.shape[0], sum(end - start for start, end in layout)),
        dtype=weight.dtype,
        device=weight.device,
    )
    destination = 0
    for start, end in layout:
        width = end - start
        packed[:, destination : destination + width].copy_(weight[:, start:end])
        destination += width
    return packed


def _get_weight_shadow(
    weight: torch.Tensor,
    layout: Optional[_WeightLayout] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a version-correct column-major E4M3 shadow and its scale."""

    global _cache_hits, _cache_misses

    if _FP8_E4M3 is None:
        raise RuntimeError("E4M3 weight shadow requested without E4M3 support")
    normalized_layout = _normalize_layout(weight, layout)
    key = _weight_key(weight, normalized_layout)
    version = _weight_version(weight)
    entry = _weight_cache.get(key)
    if (
        entry is not None
        and entry.ref() is weight
        and entry.version == version
        and entry.data_ptr == weight.data_ptr()
        and entry.shape == weight.shape
    ):
        _cache_hits += 1
        return entry.quantized_column_major, entry.scale

    _cache_misses += 1
    with record_function("## hstu_fp8_weight_shadow_refresh ##"):
        packed_weight = _pack_weight_layout(weight, normalized_layout)
        scale = calibrated_scale(packed_weight, margin=1.0)
        quantized = quantize_with_scale(
            packed_weight,
            scale,
            column_major=True,
        )
    _weight_cache[key] = _WeightCacheEntry(
        ref=weakref.ref(weight),
        version=version,
        data_ptr=weight.data_ptr(),
        shape=weight.shape,
        quantized_column_major=quantized,
        scale=scale,
    )
    return quantized, scale


def packed_weight_for_layout(
    weight: torch.Tensor,
    layout: _WeightLayout,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return one version-correct packed training weight without repeated cat."""

    normalized_layout = _normalize_layout(weight, layout)
    key = (*_weight_key(weight, normalized_layout), dtype)
    version = _weight_version(weight)
    entry = _packed_weight_cache.get(key)
    if (
        entry is not None
        and entry.ref() is weight
        and entry.version == version
        and entry.data_ptr == weight.data_ptr()
    ):
        return entry.packed
    packed = _pack_weight_layout(weight, normalized_layout).to(dtype)
    _packed_weight_cache[key] = _PackedWeightCacheEntry(
        ref=weakref.ref(weight),
        version=version,
        data_ptr=weight.data_ptr(),
        packed=packed,
    )
    return packed


def clear_fp8_weight_cache() -> None:
    global _cache_hits, _cache_misses
    _weight_cache.clear()
    _packed_weight_cache.clear()
    _fp8_one_by_device.clear()
    _scale_value_cache.clear()
    _inverse_scale_cache.clear()
    _folded_input_scale_cache.clear()
    _scaled_output_bias_cache.clear()
    _fp8_output_scale_mode_by_device.clear()
    _telemetry_calls.clear()
    _telemetry_last.clear()
    _capability_sites.clear()
    _capability_contracts.clear()
    _cache_hits = 0
    _cache_misses = 0


def fp8_weight_cache_stats() -> dict[str, int]:
    stale_keys = [key for key, entry in _weight_cache.items() if entry.ref() is None]
    for key in stale_keys:
        _weight_cache.pop(key, None)
    stale_packed_keys = [
        key for key, entry in _packed_weight_cache.items() if entry.ref() is None
    ]
    for key in stale_packed_keys:
        _packed_weight_cache.pop(key, None)
    return {
        "entries": len(_weight_cache),
        "packed_entries": len(_packed_weight_cache),
        "hits": _cache_hits,
        "misses": _cache_misses,
    }


def fp8_tensor_telemetry(
    value: torch.Tensor,
    scale: torch.Tensor,
) -> dict[str, float | int | bool]:
    """Synchronizing diagnostic for gated saturation/underflow checks."""

    if value.dtype not in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e5m2fnuz", None),
    ):
        raise TypeError(f"expected an FP8 tensor, got {value.dtype}")
    total = value.numel()
    if total == 0:
        return {
            "numel": 0,
            "sampled_numel": 0,
            "finite": True,
            "saturation_fraction": 0.0,
            "zero_fraction": 0.0,
            "scale": float(scale.item()),
        }
    # Telemetry is deliberately bounded: materializing a production QKV tensor
    # as FP32 can require tens of GiB. An evenly spaced million-value sample is
    # enough to fail closed on material saturation/underflow without restoring
    # the cast tax that the fused path removes.
    max_samples = 1 << 20
    sample_stride = max(1, math.ceil(total / max_samples))
    while math.gcd(sample_stride, 6144) != 1:
        sample_stride += 1
    sampled = value.reshape(-1)[::sample_stride][:max_samples]
    as_float = sampled.float()
    max_value = torch.finfo(value.dtype).max
    return {
        "numel": total,
        "sampled_numel": sampled.numel(),
        "finite": bool(torch.isfinite(as_float).all().item()),
        "saturation_fraction": float(
            (as_float.abs() >= max_value).float().mean().item()
        ),
        "zero_fraction": float((as_float == 0).float().mean().item()),
        "scale": float(scale.item()),
    }


def _capability_rank_info() -> Tuple[int, int, int]:
    """Return global rank, local rank, and world size for capability evidence."""

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = int(torch.distributed.get_rank())
        world_size = int(torch.distributed.get_world_size())
    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            str(torch.cuda.current_device() if torch.cuda.is_available() else rank),
        )
    )
    return rank, local_rank, world_size


def _write_fp8_runtime_report() -> None:
    capability_dir = os.environ.get("DLRM_HSTU_FP8_CAPABILITY_DIR")
    if not capability_dir:
        return
    rank, local_rank, world_size = _capability_rank_info()
    record = {
        "schema": "hstu-fp8-capability-v4",
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "pid": os.getpid(),
        "torch_hip": torch.version.hip,
        "device": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else None
        ),
        "fail_closed": _HSTU_FP8_FAIL_CLOSED,
        "fast_accum": _HSTU_FP8_FAST_ACCUM,
        "runtime_flags": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("DLRM_HSTU_FP8_")
            and key != "DLRM_HSTU_FP8_CAPABILITY_DIR"
        },
        "sites": sorted(_capability_sites),
        "contracts": _capability_contracts,
        "telemetry": fp8_runtime_telemetry(),
    }
    output_dir = Path(capability_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"rank{rank}.json"
    temporary = output_dir / f".rank{rank}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)


def maybe_record_fp8_telemetry(
    site: str,
    value: torch.Tensor,
    scale: torch.Tensor,
    *,
    cadence_divisor: int = 1,
) -> None:
    """Sample producer output only at an explicitly requested cadence."""

    if _HSTU_FP8_TELEMETRY_INTERVAL == 0:
        return
    calls = _telemetry_calls.get(site, 0) + 1
    _telemetry_calls[site] = calls
    interval = max(1, _HSTU_FP8_TELEMETRY_INTERVAL // cadence_divisor)
    if calls % interval != 0:
        return
    record = fp8_tensor_telemetry(value, scale)
    record["calls"] = calls
    record["samples"] = 1
    previous = _telemetry_last.get(site)
    if previous is not None:
        record["finite"] = bool(previous.get("finite")) and bool(record["finite"])
        record["sampled_numel"] = min(
            int(previous.get("sampled_numel", 0)),
            int(record["sampled_numel"]),
        )
        record["saturation_fraction"] = max(
            float(previous.get("saturation_fraction", 1.0)),
            float(record["saturation_fraction"]),
        )
        record["zero_fraction"] = max(
            float(previous.get("zero_fraction", 1.0)),
            float(record["zero_fraction"]),
        )
        record["samples"] = int(previous.get("samples", 1)) + 1
    _telemetry_last[site] = record
    _write_fp8_runtime_report()


def fp8_runtime_telemetry() -> dict[str, dict[str, float | int | bool]]:
    return {site: dict(record) for site, record in _telemetry_last.items()}


def _log_fp8_resolved_once() -> None:
    """Say once, per rank, what the FP8 knobs RESOLVED to.

    They are chained -- ATTN requires FUSE_QKV, which requires FUSE_LN, which
    requires GEMM -- so a submit line naming a stage does not mean the stage is
    on, and the arm that runs can be a quieter one than the arm that was asked
    for. Only the resolved values tell those apart, so those are what is logged.
    """

    global _resolved_logged
    if _resolved_logged:
        return
    _resolved_logged = True
    logger.info(
        "[hstu-fp8] resolved: fwd=%d gemm=%d fuse_ln=%d fuse_outln=%d "
        "fuse_qkv=%d attn=%d bwd=%d fuse_silu=%d fuse_residual=%d "
        "scale_mode=%s calibration_steps=%d host_scale_values=%d "
        "fail_closed=%d fast_accum=%d",
        _HSTU_FP8_FWD,
        _HSTU_FP8_GEMM,
        _HSTU_FP8_FUSE_LN,
        _HSTU_FP8_FUSE_OUTLN,
        _HSTU_FP8_FUSE_QKV,
        _HSTU_FP8_ATTENTION,
        _HSTU_FP8_BACKWARD,
        _HSTU_FP8_FUSE_SILU,
        _HSTU_FP8_FUSE_RESIDUAL,
        _HSTU_FP8_SCALE_MODE,
        _HSTU_FP8_CALIBRATION_STEPS,
        _HSTU_FP8_HOST_SCALE_VALUES,
        _HSTU_FP8_FAIL_CLOSED,
        _HSTU_FP8_FAST_ACCUM,
    )


def record_fp8_capability(site: str) -> None:
    """Note that a fused production site ran, once per site per rank.

    The capability FILE is what a certification run collects, and it is still
    written only when asked for. The LOG LINE is for every other run: until it
    existed, a production FP8 run left no record that it was FP8, so a BF16 rate
    reported under an FP8 label was indistinguishable from the real thing.
    """

    if site in _capability_sites:
        return
    _capability_sites.add(site)
    _log_fp8_resolved_once()
    logger.info("[hstu-fp8] engaged: %s", site)
    if not os.environ.get("DLRM_HSTU_FP8_CAPABILITY_DIR"):
        return
    _write_fp8_runtime_report()


def record_fp8_contract(site: str, **tensors: torch.Tensor) -> None:
    """Persist actual production dtypes/shapes for a fused-site proof."""

    capability_dir = os.environ.get("DLRM_HSTU_FP8_CAPABILITY_DIR")
    if not capability_dir:
        return
    contract = {
        name: {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        for name, tensor in sorted(tensors.items())
    }
    existing = _capability_contracts.get(site)
    if existing is not None:
        if {
            name: value["dtype"] for name, value in existing.items()
        } != {name: value["dtype"] for name, value in contract.items()}:
            raise RuntimeError(f"FP8 dtype contract changed at {site}")
        return
    _capability_contracts[site] = contract
    _write_fp8_runtime_report()


_HSTU_FP8_QUIET_HEURISTIC: bool = (
    os.environ.get("DLRM_HSTU_FP8_QUIET_HEURISTIC", "1") == "1"
)
_QUIET_SEEN_MAX: int = 256
_quiet_seen: "collections.OrderedDict[Tuple, None]" = collections.OrderedDict()
_quiet_lock = threading.Lock()
_devnull_fd: Optional[int] = None


def _unwrap_scaled_mm(value):
    return value[0] if isinstance(value, tuple) else value


def _scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    *,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
    scale_result: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    kwargs = {
        "scale_a": scale_a,
        "scale_b": scale_b,
        "out_dtype": out_dtype,
        "use_fast_accum": _HSTU_FP8_FAST_ACCUM,
    }
    if bias is not None:
        kwargs["bias"] = bias
    if scale_result is not None:
        kwargs["scale_result"] = scale_result
    try:
        return _unwrap_scaled_mm(_scaled_mm_call(a, b, kwargs))
    except TypeError:
        # Some ROCm PyTorch builds do not expose use_fast_accum. This is an API
        # compatibility retry, not a BF16 fallback.
        kwargs.pop("use_fast_accum", None)
        return _unwrap_scaled_mm(_scaled_mm_call(a, b, kwargs))


def _heuristic_key(a: torch.Tensor, b: torch.Tensor, kwargs: Dict) -> Tuple:
    """Everything hipBLASLt's heuristic query can see, minus the data."""
    bias = kwargs.get("bias")
    return (
        a.device.index,
        tuple(a.shape), a.stride(), a.dtype,
        tuple(b.shape), b.stride(), b.dtype,
        kwargs["out_dtype"],
        None if bias is None else bias.dtype,
        tuple(kwargs["scale_a"].shape),
        tuple(kwargs["scale_b"].shape),
        "scale_result" in kwargs,
        kwargs.get("use_fast_accum"),
    )


def _scaled_mm_call(a: torch.Tensor, b: torch.Tensor, kwargs: Dict):
    """torch._scaled_mm, with hipBLASLt's heuristic chatter kept off fd 2.

    The first FP8 GEMM at a new shape makes hipBLASLt's analytical heuristic
    print "Warning: Latency not found for MI_M=16, MI_N=16, MI_K=128,
    mi_input_type=BFloat8Float8_fnuz. Returning latency value of 32 (really
    slow)." about 170 times, through an unbuffered std::cerr that issues one
    write(2) per `<<` token (the torn "16ITER35PROBE" fragments in the log show
    it). No environment variable gates it in hipBLASLt 1.2 / ROCm 7.2.
    Standalone, those writes cost ~0.2 ms. In the trainer, eight ranks share a
    single stderr pipe into launch_slurm.sh's sed|tee, and the same query costs
    8 ms of on-CPU time (median of 1703 calls, zero context switches). M is the
    jagged token count and changes every step, so every step pays it several
    times on the main thread while the GPU queue is shallow.

    Only the chatter changes. The GEMM, the heuristic and the kernel it picks
    are untouched, so the output is bit-identical. Redirection applies only to
    a shape key not seen recently (repeat shapes hit hipBLASLt's cache and
    print nothing). A failed quiet call is re-issued with stderr attached so
    any real diagnostic still reaches the log. DLRM_HSTU_FP8_QUIET_HEURISTIC=0
    restores the old behaviour.
    """
    if not _HSTU_FP8_QUIET_HEURISTIC or torch.compiler.is_compiling():
        return torch._scaled_mm(a, b, **kwargs)
    key = _heuristic_key(a, b, kwargs)
    if key in _quiet_seen:
        _quiet_seen.move_to_end(key)
        return torch._scaled_mm(a, b, **kwargs)
    try:
        output = _scaled_mm_stderr_silenced(a, b, kwargs)
    except TypeError:
        raise
    except Exception:
        return torch._scaled_mm(a, b, **kwargs)
    _quiet_seen[key] = None
    if len(_quiet_seen) > _QUIET_SEEN_MAX:
        _quiet_seen.popitem(last=False)
    return output


def _scaled_mm_stderr_silenced(a: torch.Tensor, b: torch.Tensor, kwargs: Dict):
    global _devnull_fd
    # fd 2 is process-wide. The lock stops two threads from interleaving their
    # save/restore, which could otherwise leave fd 2 pointing at /dev/null.
    # Anything another thread writes to stderr inside this window (~1 ms) is
    # lost, which is the accepted cost.
    with _quiet_lock:
        if _devnull_fd is None:
            _devnull_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            sys.stderr.flush()
        except Exception:
            pass
        saved = os.dup(2)
        try:
            os.dup2(_devnull_fd, 2)
            try:
                return torch._scaled_mm(a, b, **kwargs)
            finally:
                os.dup2(saved, 2)
        finally:
            os.close(saved)


@functools.lru_cache(maxsize=1)
def _residual_fused_extension():
    source = Path(__file__).with_name("hipblaslt_scaled_addmm.cpp")
    build_directory = Path("/tmp/dlrm_hstu_fp8_scaled_addmm")
    build_directory.mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.cpp_extension import load

        return load(
            name="dlrm_hstu_fp8_scaled_addmm",
            sources=[str(source)],
            extra_include_paths=["/opt/rocm/include"],
            extra_cflags=["-O3"],
            extra_ldflags=[
                "-L/opt/rocm/lib",
                "-lhipblaslt",
                "-lamdhip64",
            ],
            build_directory=str(build_directory),
            verbose=False,
        )
    except Exception as error:
        raise RuntimeError(
            "requested FP8 residual-fused hipBLASLt path is unavailable"
        ) from error


def prebuild_extensions() -> bool:
    """Build the runtime C++ extensions this configuration will load.

    Called by train_ranker during initialization (before MLPerf INIT_STOP /
    RUN_START, gated by DLRM_HSTU_FP8_PREBUILD) so the one-time
    torch.utils.cpp_extension build of the residual-fused hipBLASLt path is
    not paid inside the timed region on the first FP8 step. Same module, same
    build flags and the same lru_cache entry the lazy path returns, so the
    compiled code is identical. Returns True if something was built/loaded.
    """
    if _HSTU_FP8_FUSE_RESIDUAL:
        _residual_fused_extension()
        return True
    return False


def _residual_workspace(device: torch.device) -> torch.Tensor:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    workspace = _residual_workspaces.get(device_index)
    if workspace is None:
        workspace = torch.empty(
            _HSTU_FP8_FUSE_RESIDUAL_WORKSPACE_BYTES,
            dtype=torch.uint8,
            device=device,
        )
        _residual_workspaces[device_index] = workspace
    return workspace


def scaled_addmm_fp8_preq(
    activation_fp8: torch.Tensor,
    scale_a: torch.Tensor,
    weight: torch.Tensor,
    bias_or_residual: Optional[torch.Tensor],
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    weight_layout: Optional[_WeightLayout] = None,
) -> torch.Tensor:
    """FP8 GEMM consuming an activation emitted by a producer epilogue."""

    normalized_layout = _normalize_layout(weight, weight_layout)
    output_columns = sum(end - start for start, end in normalized_layout)
    if not fp8_supported(activation_fp8, weight) or output_columns % 16 != 0:
        raise RuntimeError(
            "unsupported fused FP8 GEMM shape/backend: "
            f"x={tuple(activation_fp8.shape)} w={tuple(weight.shape)} "
            f"layout={normalized_layout} hip={torch.version.hip}"
        )
    if activation_fp8.dtype != fp8_e4m3_dtype():
        raise TypeError(f"expected E4M3 activation, got {activation_fp8.dtype}")
    if bias_or_residual is not None:
        if bias_or_residual.dim() == 1:
            if bias_or_residual.numel() != output_columns:
                raise ValueError(
                    f"expected bias [{output_columns}], got "
                    f"{tuple(bias_or_residual.shape)}"
                )
        elif tuple(bias_or_residual.shape) != (
            activation_fp8.shape[0],
            output_columns,
        ):
            raise ValueError(
                "expected residual "
                f"[{activation_fp8.shape[0]}, {output_columns}], got "
                f"{tuple(bias_or_residual.shape)}"
            )
    if activation_fp8.shape[0] == 0:
        return torch.empty(
            (0, output_columns),
            dtype=out_dtype,
            device=activation_fp8.device,
        )

    maybe_record_fp8_telemetry(
        f"addmm:{id(weight)}:{tuple(weight.shape)}:{normalized_layout}",
        activation_fp8,
        scale_a,
        cadence_divisor=64 if len(normalized_layout) > 1 else 1,
    )
    weight_fp8, scale_b = _get_weight_shadow(weight, normalized_layout)
    bias = None
    residual = None
    if bias_or_residual is not None:
        if bias_or_residual.dim() == 1:
            bias = bias_or_residual.to(out_dtype)
        else:
            residual = bias_or_residual
    if residual is not None and _HSTU_FP8_FUSE_RESIDUAL:
        if (
            out_dtype != torch.bfloat16
            or residual.dtype != torch.bfloat16
            or not activation_fp8.is_contiguous()
            or not residual.is_contiguous()
            or weight_fp8.stride() != (1, weight_fp8.shape[0])
        ):
            raise RuntimeError(
                "requested FP8 residual fusion requires contiguous E4M3 "
                "activation, column-major E4M3 weight, and contiguous BF16 residual"
            )
        return _residual_fused_extension().scaled_addmm(
            activation_fp8,
            weight_fp8,
            residual,
            scale_a,
            scale_b,
            _residual_workspace(activation_fp8.device),
            _HSTU_FP8_FUSE_RESIDUAL_ALGORITHM,
        )
    output = _scaled_mm(
        activation_fp8,
        weight_fp8,
        scale_a,
        scale_b,
        out_dtype=out_dtype,
        bias=bias,
    )
    return output if residual is None else output + residual


def scaled_mm_fp8_out_preq(
    activation_fp8: torch.Tensor,
    scale_a: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    weight_layout: Optional[_WeightLayout] = None,
    output_scale: Optional[torch.Tensor] = None,
    scale_a_value: Optional[float] = None,
    output_scale_value: Optional[float] = None,
) -> torch.Tensor:
    """Emit a scaled E4M3 projection for a downstream FP8 consumer."""

    normalized_layout = _normalize_layout(weight, weight_layout)
    output_columns = sum(end - start for start, end in normalized_layout)
    if not fp8_supported(activation_fp8, weight) or output_columns % 16 != 0:
        raise RuntimeError(
            "unsupported FP8-output GEMM shape/backend: "
            f"x={tuple(activation_fp8.shape)} w={tuple(weight.shape)} "
            f"layout={normalized_layout}"
        )
    if activation_fp8.dtype != fp8_e4m3_dtype():
        raise TypeError(f"expected E4M3 activation, got {activation_fp8.dtype}")
    if bias is not None and (bias.dim() != 1 or bias.numel() != output_columns):
        raise ValueError(
            f"expected bias [{output_columns}], got {tuple(bias.shape)}"
        )
    if activation_fp8.shape[0] == 0:
        return torch.empty(
            (0, output_columns),
            dtype=fp8_e4m3_dtype(),
            device=activation_fp8.device,
        )
    maybe_record_fp8_telemetry(
        f"mm_out:{id(weight)}:{tuple(weight.shape)}:{normalized_layout}",
        activation_fp8,
        scale_a,
        cadence_divisor=64 if len(normalized_layout) > 1 else 1,
    )
    weight_fp8, scale_b = _get_weight_shadow(weight, normalized_layout)
    device_index = activation_fp8.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    output_scale_mode = _fp8_output_scale_mode(activation_fp8.device)
    effective_scale_a = scale_a
    effective_bias = None if bias is None else bias.to(torch.bfloat16)
    host_scale_values = (
        None
        if output_scale is None
        else _trusted_host_scale_values(
            scale_a,
            output_scale,
            scale_a_value,
            output_scale_value,
        )
    )
    if output_scale is None:
        if output_scale_mode == "input_scale":
            scale_result = None
        else:
            scale_result = _fp8_one_by_device.get(device_index)
            if scale_result is None:
                scale_result = torch.ones(
                    (1, 1),
                    dtype=torch.float32,
                    device=activation_fp8.device,
                )
                _fp8_one_by_device[device_index] = scale_result
    elif output_scale_mode == "input_scale":
        effective_scale_a = _fold_output_scale_into_input(
            scale_a,
            output_scale,
            scale_a_value=(
                None if host_scale_values is None else host_scale_values[0]
            ),
            output_scale_value=(
                None if host_scale_values is None else host_scale_values[1]
            ),
        )
        effective_bias = (
            None
            if bias is None
            else _scale_bias_for_fp8_output(
                bias,
                output_scale,
                output_scale_value=(
                    None if host_scale_values is None else host_scale_values[1]
                ),
            )
        )
        scale_result = None
    else:
        scale_result = inverse_scale_tensor(
            output_scale,
            value=(
                None if host_scale_values is None else host_scale_values[1]
            ),
        )
    output = _scaled_mm(
        activation_fp8,
        weight_fp8,
        effective_scale_a,
        scale_b,
        out_dtype=fp8_e4m3_dtype(),
        bias=effective_bias,
        scale_result=scale_result,
    )
    if output_scale is not None:
        maybe_record_fp8_telemetry(
            f"qkv_output:{id(weight)}:{tuple(weight.shape)}:{normalized_layout}",
            output,
            fp8_scale_tensor(output_scale),
        )
    return output


def scaled_mm_fp8_preq(
    a_fp8: torch.Tensor,
    scale_a: torch.Tensor,
    b_fp8_column_major: torch.Tensor,
    scale_b: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Low-level prequantized GEMM used by selective manual backward."""

    if a_fp8.shape[0] == 0:
        return torch.empty(
            (0, b_fp8_column_major.shape[1]),
            dtype=out_dtype,
            device=a_fp8.device,
        )
    return _scaled_mm(
        a_fp8,
        b_fp8_column_major,
        scale_a,
        scale_b,
        out_dtype=out_dtype,
    )
