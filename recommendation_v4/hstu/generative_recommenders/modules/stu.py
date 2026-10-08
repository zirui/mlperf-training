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

# pyre-strict
import abc
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from generative_recommenders.common import (
    fx_unwrap_optional_tensor,
    HammerKernel,
    HammerModule,
)
from generative_recommenders.ops.hstu_attention import delta_hstu_mha
from generative_recommenders.ops.hstu_compute import (
    hstu_compute_output,
    hstu_compute_uqvk,
    hstu_preprocess_and_attention,
    hstu_preprocess_and_attention_targets_only,
)
from generative_recommenders.ops.jagged_tensors import concat_2D_jagged, split_2D_jagged
from generative_recommenders.ops.triton.triton_fp8 import (
    fp8_scale_state_size,
    hstu_fp8_attention_enabled,
    hstu_fp8_fail_closed,
    hstu_fp8_fuse_ln_enabled,
    hstu_fp8_fuse_outln_enabled,
    hstu_fp8_fuse_qkv_enabled,
    hstu_fp8_fuse_residual_enabled,
    hstu_fp8_fuse_silu_enabled,
    record_fp8_capability,
    record_fp8_contract,
)
from torch.autograd.profiler import record_function

# Targets-only e4m3 VK save (store instead of recompute); gin default on
# (yambda_5b.gin item (15)), absent or "0" = recompute in the targets-only
# backward. Read once at import, after apply_env_bootstrap.
_SAVE_TARGETS_VK: bool = os.environ.get("DLRM_HSTU_FP8_SAVE_TARGETS_VK", "0") != "0"


try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass


class STU(HammerModule, abc.ABC):
    @torch.jit.unused
    def targets_only_unsupported_reason(self) -> Optional[str]:
        """Why this module cannot run the targets-only path, or None if it can.

        A reason rather than a bool because the caller raises with it: the knob
        is opt-in, so a config that can never satisfy it is a mistake to
        surface, and "unsupported" alone does not say which of nine conditions
        to go fix.
        """
        return f"{type(self).__name__} has no targets-only path"

    @torch.jit.unused
    def supports_targets_only(self) -> bool:
        return self.targets_only_unsupported_reason() is None

    def cached_forward(
        self,
        delta_x: torch.Tensor,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @torch.jit.unused
    def forward_targets_only(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pass


@dataclass
class STULayerConfig:
    embedding_dim: int
    num_heads: int
    hidden_dim: int
    attention_dim: int
    output_dropout_ratio: float = 0.3
    causal: bool = True
    target_aware: bool = True
    max_attn_len: Optional[int] = None
    attn_alpha: Optional[float] = None
    use_group_norm: bool = False
    recompute_normed_x: bool = True
    recompute_uvqk: bool = True
    recompute_y: bool = True
    sort_by_length: bool = True
    contextual_seq_len: int = 0


@torch.fx.wrap
def _update_kv_cache(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    max_kv_caching_len: int,
    kv_caching_lengths: Optional[torch.Tensor],
    orig_k_cache: Optional[torch.Tensor],
    orig_v_cache: Optional[torch.Tensor],
    orig_max_kv_caching_len: int,
    orig_kv_caching_offsets: Optional[torch.Tensor],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int, Optional[torch.Tensor]]:
    if kv_caching_lengths is not None:
        kv_caching_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            kv_caching_lengths
        )
        delta_offsets = seq_offsets - kv_caching_offsets
        k_cache, _ = split_2D_jagged(
            max_seq_len=max_seq_len,
            values=fx_unwrap_optional_tensor(k).flatten(1, 2),
            max_len_left=None,
            max_len_right=None,
            offsets_left=kv_caching_offsets,
            offsets_right=delta_offsets,
        )
        v_cache, _ = split_2D_jagged(
            max_seq_len=max_seq_len,
            values=fx_unwrap_optional_tensor(v).flatten(1, 2),
            max_len_left=None,
            max_len_right=None,
            offsets_left=kv_caching_offsets,
            offsets_right=delta_offsets,
        )
        if max_kv_caching_len == 0:
            max_kv_caching_len = int(kv_caching_lengths.max().item())
        return (
            k_cache,
            v_cache,
            max_kv_caching_len,
            kv_caching_offsets,
        )
    else:
        return (
            orig_k_cache,
            orig_v_cache,
            orig_max_kv_caching_len,
            orig_kv_caching_offsets,
        )


@torch.fx.wrap
def _construct_full_kv(
    delta_k: torch.Tensor,
    delta_v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    max_kv_caching_len: int,
    kv_caching_offsets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    L, _ = delta_k.shape
    B = kv_caching_offsets.shape[0] - 1
    delta_size = L // B
    full_k = concat_2D_jagged(
        max_seq_len=max_kv_caching_len + delta_size,
        values_left=k_cache,
        values_right=delta_k,
        max_len_left=max_kv_caching_len,
        max_len_right=delta_size,
        offsets_left=kv_caching_offsets,
        offsets_right=None,
    )
    full_v = concat_2D_jagged(
        max_seq_len=max_kv_caching_len + delta_size,
        values_left=v_cache,
        values_right=delta_v,
        max_len_left=max_kv_caching_len,
        max_len_right=delta_size,
        offsets_left=kv_caching_offsets,
        offsets_right=None,
    )
    full_kv_caching_offsets = kv_caching_offsets + delta_size * torch.arange(
        B + 1, device=delta_k.device
    )
    return (
        full_k,
        full_v,
        max_kv_caching_len + delta_size,
        full_kv_caching_offsets,
    )


class STULayer(STU):
    max_kv_caching_len: int
    k_cache: Optional[torch.Tensor]
    v_cache: Optional[torch.Tensor]
    kv_caching_offsets: Optional[torch.Tensor]
    _fp8_input_scale: torch.Tensor
    _fp8_qkv_scale: torch.Tensor
    _fp8_output_scale: torch.Tensor

    def __init__(
        self,
        config: STULayerConfig,
        is_inference: bool = False,
    ) -> None:
        super().__init__(
            is_inference=is_inference,
        )
        self.reset_kv_cache()
        self._num_heads: int = config.num_heads
        self._embedding_dim: int = config.embedding_dim
        self._hidden_dim: int = config.hidden_dim
        self._attention_dim: int = config.attention_dim
        self._output_dropout_ratio: float = config.output_dropout_ratio
        self._target_aware: bool = config.target_aware
        self._causal: bool = config.causal
        self._max_attn_len: int = config.max_attn_len or 0
        self._attn_alpha: float = config.attn_alpha or 1.0 / (self._attention_dim**0.5)
        self._use_group_norm: bool = config.use_group_norm
        self._recompute_normed_x: bool = config.recompute_normed_x
        self._recompute_uvqk: bool = config.recompute_uvqk
        self._recompute_y: bool = config.recompute_y
        self._sort_by_length: bool = config.sort_by_length
        self._contextual_seq_len: int = config.contextual_seq_len
        self._stack_layer_index: int = -1

        self._uvqk_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(
                (
                    self._embedding_dim,
                    (self._hidden_dim * 2 + self._attention_dim * 2) * self._num_heads,
                )
            ),
        )
        torch.nn.init.xavier_uniform_(self._uvqk_weight)
        self._uvqk_beta: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros(
                (self._hidden_dim * 2 + self._attention_dim * 2) * self._num_heads,
            ),
        )
        self._input_norm_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.ones((self._embedding_dim,)),
        )
        self._input_norm_bias: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros((self._embedding_dim,)),
        )
        self._output_weight = torch.nn.Parameter(
            torch.empty(
                (
                    self._hidden_dim * self._num_heads * 3,
                    self._embedding_dim,
                )
            ),
        )
        torch.nn.init.xavier_uniform_(self._output_weight)
        output_norm_shape: int = (
            self._hidden_dim * self._num_heads
            if not self._use_group_norm
            else self._num_heads
        )
        self._output_norm_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.ones((output_norm_shape,)),
        )
        self._output_norm_bias: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros((output_norm_shape,)),
        )
        # Each checkpointed state contains the published scale, completed
        # warmup count, and bounded cross-rank amax history. A zero scale keeps
        # the path in BF16 calibration until the history is complete.
        self.register_buffer(
            "_fp8_input_scale",
            torch.zeros((fp8_scale_state_size(),), dtype=torch.float32),
        )
        self.register_buffer(
            "_fp8_output_scale",
            torch.zeros((fp8_scale_state_size(),), dtype=torch.float32),
        )
        self.register_buffer(
            "_fp8_qkv_scale",
            torch.zeros((fp8_scale_state_size(),), dtype=torch.float32),
        )

    # pyre-ignore[14]
    def _apply(self, fn, recurse: bool = True) -> "STULayer":  # pyre-ignore[2]
        result = super()._apply(fn, recurse=recurse)
        # Module.to(dtype=bf16) should cast model parameters but scaled-mm
        # requires its dequantization scales to remain FP32.
        self._buffers["_fp8_input_scale"] = self._fp8_input_scale.float()
        self._buffers["_fp8_qkv_scale"] = self._fp8_qkv_scale.float()
        self._buffers["_fp8_output_scale"] = self._fp8_output_scale.float()
        return result

    def reset_kv_cache(self) -> None:
        self.k_cache = None
        self.v_cache = None
        self.kv_caching_offsets = None
        self.max_kv_caching_len = 0

    @torch.jit.unused
    def _allow_saved_fp8_uvqk(self) -> bool:
        raw_layers = os.environ.get("DLRM_HSTU_FP8_SAVE_UVQK_LAYERS")
        if raw_layers is None:
            return True
        if self._stack_layer_index < 0:
            raise RuntimeError(
                "DLRM_HSTU_FP8_SAVE_UVQK_LAYERS requires STULayer ownership "
                "by an STUStack"
            )
        try:
            selected_layers = {
                int(value.strip())
                for value in raw_layers.split(",")
                if value.strip()
            }
        except ValueError as error:
            raise RuntimeError(
                "DLRM_HSTU_FP8_SAVE_UVQK_LAYERS must be comma-separated "
                "zero-based layer indices"
            ) from error
        if any(layer < 0 for layer in selected_layers):
            raise RuntimeError(
                "DLRM_HSTU_FP8_SAVE_UVQK_LAYERS cannot contain negative indices"
            )
        return self._stack_layer_index in selected_layers

    def update_kv_cache(
        self,
        max_seq_len: int,
        seq_offsets: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        max_kv_caching_len: int,
        kv_caching_lengths: Optional[torch.Tensor],
    ) -> None:
        self.k_cache, self.v_cache, self.max_kv_caching_len, self.kv_caching_offsets = (
            _update_kv_cache(
                max_seq_len=max_seq_len,
                seq_offsets=seq_offsets,
                k=k,
                v=v,
                max_kv_caching_len=max_kv_caching_len,
                kv_caching_lengths=kv_caching_lengths,
                orig_k_cache=self.k_cache,
                orig_v_cache=self.v_cache,
                orig_max_kv_caching_len=self.max_kv_caching_len,
                orig_kv_caching_offsets=self.kv_caching_offsets,
            )
        )

    def construct_full_kv(
        self,
        delta_k: torch.Tensor,
        delta_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        return _construct_full_kv(
            delta_k=delta_k,
            delta_v=delta_v,
            k_cache=fx_unwrap_optional_tensor(self.k_cache),
            v_cache=fx_unwrap_optional_tensor(self.v_cache),
            max_kv_caching_len=self.max_kv_caching_len,
            kv_caching_offsets=self.kv_caching_offsets,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hammer_kernel = self.hammer_kernel()
        fp8_input_requested = hstu_fp8_fuse_ln_enabled()
        fp8_input = (
            not torch.jit.is_scripting()
            and hammer_kernel == HammerKernel.TRITON
            and fp8_input_requested
            and self._recompute_normed_x
            and kv_caching_lengths is None
        )
        if (
            fp8_input_requested
            and not fp8_input
            and hstu_fp8_fail_closed()
        ):
            raise RuntimeError(
                "Plan 08 input FP8 requested for an unsupported "
                "backend/recompute/prefill contract"
            )
        fp8_output_requested = hstu_fp8_fuse_outln_enabled()
        fp8_output = (
            not torch.jit.is_scripting()
            and hammer_kernel == HammerKernel.TRITON
            and fp8_output_requested
            and not self._use_group_norm
            and self._recompute_y
        )
        if (
            fp8_output_requested
            and not fp8_output
            and hstu_fp8_fail_closed()
        ):
            raise RuntimeError(
                "Plan 08 output FP8 requested for an unsupported "
                "backend/group-norm/recompute contract"
            )
        fuse_silu_requested = hstu_fp8_fuse_silu_enabled()
        fuse_silu = (
            not torch.jit.is_scripting()
            and hammer_kernel == HammerKernel.TRITON
            and kv_caching_lengths is None
            and not self._use_group_norm
        )
        fuse_silu = fuse_silu_requested and fuse_silu
        if fuse_silu_requested and not fuse_silu and hstu_fp8_fail_closed():
            raise RuntimeError(
                "Plan 08 SiLU fusion requested for an unsupported "
                "backend/group-norm/prefill contract"
            )
        fuse_residual_requested = hstu_fp8_fuse_residual_enabled()
        fuse_residual = (
            fuse_residual_requested
            and fp8_input
            and hammer_kernel == HammerKernel.TRITON
            and kv_caching_lengths is None
        )
        if (
            fuse_residual_requested
            and not fuse_residual
            and hstu_fp8_fail_closed()
        ):
            raise RuntimeError(
                "Plan 08 residual-gradient fusion requested for an unsupported "
                "backend/input/prefill contract"
            )
        with record_function("## stu_preprocess_and_attention ##"):
            allow_save_fp8_uvqk = True
            if not torch.jit.is_scripting():
                allow_save_fp8_uvqk = self._allow_saved_fp8_uvqk()
            preprocess_outputs = hstu_preprocess_and_attention(
                x=x,
                norm_weight=self._input_norm_weight.to(x.dtype),
                norm_bias=self._input_norm_bias.to(x.dtype),
                norm_eps=1e-6,
                num_heads=self._num_heads,
                attn_dim=self._attention_dim,
                hidden_dim=self._hidden_dim,
                uvqk_weight=(
                    self._uvqk_weight
                    if fp8_input
                    else self._uvqk_weight.to(x.dtype)
                ),
                uvqk_bias=(
                    self._uvqk_beta if fp8_input else self._uvqk_beta.to(x.dtype)
                ),
                max_seq_len=max_seq_len,
                seq_offsets=x_offsets,
                attn_alpha=self._attn_alpha,
                causal=self._causal,
                num_targets=num_targets if self._target_aware else None,
                max_attn_len=self._max_attn_len,
                contextual_seq_len=self._contextual_seq_len,
                recompute_uvqk_in_backward=self._recompute_uvqk,
                recompute_normed_x_in_backward=self._recompute_normed_x,
                sort_by_length=self._sort_by_length,
                prefill=kv_caching_lengths is not None,
                kernel=hammer_kernel,
                fp8_input_scale=self._fp8_input_scale if fp8_input else None,
                fp8_qkv_scale=(
                    self._fp8_qkv_scale
                    if fp8_input and hstu_fp8_attention_enabled()
                    else None
                ),
                defer_silu=fuse_silu,
                allow_save_fp8_uvqk=allow_save_fp8_uvqk,
                return_residual_proxy=fuse_residual,
            )
            if fuse_residual:
                u, attn_output, k, v, residual_x = preprocess_outputs
            else:
                u, attn_output, k, v = preprocess_outputs
                residual_x = x

        self.update_kv_cache(
            max_seq_len=max_seq_len,
            seq_offsets=x_offsets,
            k=k,
            v=v,
            max_kv_caching_len=max_kv_caching_len,
            kv_caching_lengths=kv_caching_lengths,
        )

        with record_function("## stu_compute_output ##"):
            return hstu_compute_output(
                attn=attn_output,
                u=u,
                x=residual_x,
                norm_weight=self._output_norm_weight.to(x.dtype),
                norm_bias=self._output_norm_bias.to(x.dtype),
                norm_eps=1e-6,
                dropout_ratio=self._output_dropout_ratio,
                output_weight=(
                    self._output_weight
                    if fp8_output
                    else self._output_weight.to(x.dtype)
                ),
                group_norm=self._use_group_norm,
                num_heads=self._num_heads,
                linear_dim=self._hidden_dim,
                concat_u=True,
                concat_x=True,
                mul_u_activation_type="silu" if fuse_silu else "none",
                training=self.training,
                kernel=self.hammer_kernel(),
                recompute_y_in_backward=self._recompute_y,
                fp8_output_scale=self._fp8_output_scale if fp8_output else None,
            )

    @torch.jit.unused
    def targets_only_unsupported_reason(self) -> Optional[str]:
        if self._is_inference:
            return "layer is in inference mode"
        if self.hammer_kernel() != HammerKernel.TRITON:
            return f"hammer kernel is {self.hammer_kernel()}, not TRITON"
        if self._use_group_norm:
            return "group norm couples the candidate row to the rows dropped"
        if not self._target_aware:
            return "attention is not target-aware, so targets are not a suffix"
        if not self._causal:
            return "attention is not causal, so the candidate row is not last"
        if self._max_attn_len != 0:
            return f"max_attn_len={self._max_attn_len} windows attention"
        if self._contextual_seq_len != 0:
            return f"contextual_seq_len={self._contextual_seq_len} prefixes the sequence"
        if not self._recompute_normed_x:
            return "recompute_normed_x is off"
        if not self._recompute_uvqk:
            return "recompute_uvqk is off"
        return None

    @torch.jit.unused
    def forward_targets_only(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Exact final-layer training path for one suffix target per sequence."""
        if not self.supports_targets_only():
            raise RuntimeError("unsupported HSTU targets-only activation contract")

        candidate_rows = x_offsets[1:] - 1
        fp8_targets_requested = (
            hstu_fp8_fuse_qkv_enabled()
            and hstu_fp8_attention_enabled()
        )
        fp8_targets_vk = fp8_targets_requested
        fuse_residual = hstu_fp8_fuse_residual_enabled()
        with record_function("## stu_targets_only_preprocess_and_attention ##"):
            # Targets-only e4m3 VK save (store instead of recompute in the
            # targets-only backward); gin item (15), see _SAVE_TARGETS_VK.
            allow_save_fp8_vk = _SAVE_TARGETS_VK
            if allow_save_fp8_vk and not torch.jit.is_scripting():
                allow_save_fp8_vk = self._allow_saved_fp8_uvqk()
            preprocess_outputs = hstu_preprocess_and_attention_targets_only(
                x=x,
                norm_weight=self._input_norm_weight.to(x.dtype),
                norm_bias=self._input_norm_bias.to(x.dtype),
                norm_eps=1e-6,
                num_heads=self._num_heads,
                attn_dim=self._attention_dim,
                hidden_dim=self._hidden_dim,
                uvqk_weight=(
                    self._uvqk_weight
                    if fp8_targets_vk
                    else self._uvqk_weight.to(x.dtype)
                ),
                uvqk_bias=(
                    self._uvqk_beta
                    if fp8_targets_vk
                    else self._uvqk_beta.to(x.dtype)
                ),
                max_seq_len=max_seq_len,
                seq_offsets=x_offsets,
                attn_alpha=self._attn_alpha,
                num_targets=num_targets,
                kernel=HammerKernel.TRITON,
                fp8_input_scale=(
                    self._fp8_input_scale if fp8_targets_vk else None
                ),
                fp8_qkv_scale=(
                    self._fp8_qkv_scale if fp8_targets_vk else None
                ),
                return_residual_proxy=fuse_residual,
                allow_save_fp8_vk=allow_save_fp8_vk,
            )
            if fuse_residual:
                u, attn_output, residual_x = preprocess_outputs
            else:
                u, attn_output = preprocess_outputs
                residual_x = x
        candidate_x = torch.index_select(residual_x, 0, candidate_rows)
        with record_function("## stu_targets_only_compute_output ##"):
            output = hstu_compute_output(
                attn=attn_output,
                u=u,
                x=candidate_x,
                norm_weight=self._output_norm_weight.to(x.dtype),
                norm_bias=self._output_norm_bias.to(x.dtype),
                norm_eps=1e-6,
                dropout_ratio=self._output_dropout_ratio,
                output_weight=self._output_weight.to(x.dtype),
                group_norm=False,
                num_heads=self._num_heads,
                linear_dim=self._hidden_dim,
                concat_u=True,
                concat_x=True,
                mul_u_activation_type="none",
                training=self.training,
                kernel=HammerKernel.TRITON,
                recompute_y_in_backward=self._recompute_y,
                rng_row_indices=candidate_rows,
            )
        if hstu_fp8_fuse_outln_enabled():
            record_fp8_capability("phi3_targets_output_banked_bf16")
            record_fp8_contract(
                "phi3_targets_output_banked_bf16",
                final_output=output,
            )
        return output

    def cached_forward(
        self,
        delta_x: torch.Tensor,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with record_function("## stu_compute_uqvk ##"):
            delta_u, delta_q, delta_k, delta_v = hstu_compute_uqvk(
                x=delta_x,
                norm_weight=self._input_norm_weight.to(delta_x.dtype),
                norm_bias=self._input_norm_bias.to(delta_x.dtype),
                norm_eps=1e-6,
                num_heads=self._num_heads,
                attn_dim=self._attention_dim,
                hidden_dim=self._hidden_dim,
                uvqk_weight=self._uvqk_weight.to(delta_x.dtype),
                uvqk_bias=self._uvqk_beta.to(delta_x.dtype),
                kernel=self.hammer_kernel(),
            )
        k, v, max_seq_len, seq_offsets = self.construct_full_kv(
            delta_k=delta_k.flatten(1, 2),
            delta_v=delta_v.flatten(1, 2),
        )
        self.update_kv_cache(
            max_seq_len=max_seq_len,
            seq_offsets=seq_offsets,
            k=k,
            v=v,
            max_kv_caching_len=max_kv_caching_len,
            kv_caching_lengths=kv_caching_lengths,
        )
        k = k.view(-1, self._num_heads, self._attention_dim)
        v = v.view(-1, self._num_heads, self._hidden_dim)
        with record_function("## delta_hstu_mha ##"):
            delta_attn_output = delta_hstu_mha(
                max_seq_len=max_seq_len,
                alpha=self._attn_alpha,
                delta_q=delta_q,
                k=k,
                v=v,
                seq_offsets=seq_offsets,
                num_targets=num_targets if self._target_aware else None,
                max_attn_len=self._max_attn_len,
                contextual_seq_len=self._contextual_seq_len,
                kernel=self.hammer_kernel(),
            ).view(-1, self._hidden_dim * self._num_heads)
        with record_function("## stu_compute_output ##"):
            return hstu_compute_output(
                attn=delta_attn_output,
                u=delta_u,
                x=delta_x,
                norm_weight=self._output_norm_weight.to(delta_x.dtype),
                norm_bias=self._output_norm_bias.to(delta_x.dtype),
                norm_eps=1e-6,
                dropout_ratio=self._output_dropout_ratio,
                output_weight=self._output_weight.to(delta_x.dtype),
                group_norm=self._use_group_norm,
                num_heads=self._num_heads,
                linear_dim=self._hidden_dim,
                concat_u=True,
                concat_x=True,
                mul_u_activation_type="none",
                training=self.training,
                kernel=self.hammer_kernel(),
                recompute_y_in_backward=self._recompute_y,
            )


class STUStack(STU):
    def __init__(
        self,
        stu_list: List[STU],
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        for layer_index, layer in enumerate(stu_list):
            if isinstance(layer, STULayer):
                layer._stack_layer_index = layer_index
        self._stu_layers: torch.nn.ModuleList = torch.nn.ModuleList(modules=stu_list)

    @torch.jit.unused
    def targets_only_unsupported_reason(self) -> Optional[str]:
        if len(self._stu_layers) == 0:
            return "stack has no STU layers"
        reason = self._stu_layers[
            len(self._stu_layers) - 1
        ].targets_only_unsupported_reason()  # pyre-ignore [29]
        return None if reason is None else f"last STU layer: {reason}"

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self._stu_layers:
            x = layer(
                x=x,
                x_lengths=x_lengths,
                x_offsets=x_offsets,
                max_seq_len=max_seq_len,
                num_targets=num_targets,
                max_kv_caching_len=max_kv_caching_len,
                kv_caching_lengths=kv_caching_lengths,
            )
        return x

    @torch.jit.unused
    def forward_targets_only(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
    ) -> torch.Tensor:
        num_layers = len(self._stu_layers)
        if num_layers == 0:
            raise RuntimeError("targets-only requires at least one STU layer")
        for layer_index, layer in enumerate(self._stu_layers):
            if layer_index == num_layers - 1:
                return layer.forward_targets_only(  # pyre-ignore [29]
                    x=x,
                    x_lengths=x_lengths,
                    x_offsets=x_offsets,
                    max_seq_len=max_seq_len,
                    num_targets=num_targets,
                )
            x = layer(
                x=x,
                x_lengths=x_lengths,
                x_offsets=x_offsets,
                max_seq_len=max_seq_len,
                num_targets=num_targets,
            )
        raise AssertionError("unreachable")

    def cached_forward(
        self,
        delta_x: torch.Tensor,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self._stu_layers:
            delta_x = layer.cached_forward(  # pyre-ignore [29]
                delta_x=delta_x,
                num_targets=num_targets,
                max_kv_caching_len=max_kv_caching_len,
                kv_caching_lengths=kv_caching_lengths,
            )
        return delta_x
