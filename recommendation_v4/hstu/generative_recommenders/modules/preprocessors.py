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
from math import sqrt
from typing import Dict, List, Optional, Tuple

import torch
from generative_recommenders.common import (
    fx_unwrap_optional_tensor,
    HammerModule,
    init_mlp_weights_optional_bias,
    jagged_to_padded_dense,
)
from generative_recommenders.modules.action_encoder import ActionEncoder
from torch.fx._symbolic_trace import is_fx_tracing
from generative_recommenders.ops.blas_solution_pins import install_for_tokens
from generative_recommenders.ops.fused_linear_norm import FusedLinearNormMLP
from generative_recommenders.ops.jagged_tensors import concat_2D_jagged
from generative_recommenders.ops.layer_norm import LayerNorm, SwishLayerNorm


class InputPreprocessor(HammerModule):
    """An abstract class for pre-processing sequence embeddings before HSTU layers."""

    @abc.abstractmethod
    def forward(
        self,
        max_uih_len: int,
        max_targets: int,
        total_uih_len: int,
        total_targets: int,
        seq_lengths: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[
        int,
        int,
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        """
        Args:
            max_uih_len: int
            max_targets: int
            total_uih_len: int
            total_targets: int
            seq_lengths: (B,)
            seq_embeddings: (L, D)
            seq_timestamps: (B, N)
            num_targets: (B,) Optional.
            seq_payloads: str-keyed tensors. Implementation specific.

        Returns:
            (max_seq_len, total_uih_len, total_targets, lengths, offsets, timestamps, embeddings, num_targets, payloads) updated based on input preprocessor.
        """
        pass

    def interleave_targets(self) -> bool:
        return False


def get_contextual_input_embeddings(
    seq_lengths: torch.Tensor,
    seq_payloads: Dict[str, torch.Tensor],
    contextual_feature_to_max_length: Dict[str, int],
    contextual_feature_to_min_uih_length: Dict[str, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    padded_values: List[torch.Tensor] = []
    for key, max_len in contextual_feature_to_max_length.items():
        v = torch.flatten(
            jagged_to_padded_dense(
                values=seq_payloads[key].to(dtype),
                offsets=[seq_payloads[key + "_offsets"]],
                max_lengths=[max_len],
                padding_value=0.0,
            ),
            1,
            2,
        )
        min_uih_length = contextual_feature_to_min_uih_length.get(key, 0)
        if min_uih_length > 0:
            v = v * (seq_lengths.view(-1, 1) >= min_uih_length)
        padded_values.append(v)
    return torch.cat(padded_values, dim=1)


@torch.jit.ignore
def _catfused_base(parts: List[torch.Tensor]) -> Optional[torch.Tensor]:
    """The [N, k*W] tensor whose column blocks are exactly `parts`, or None.

    Set by dlrm_v4/train/ec_bf16_output.py (DLRM_HSTU_FP8_EC_CAT_FUSE) on the
    column views of one interleaved expand; torch.cat(parts, dim=1) would
    rebuild that tensor byte for byte.
    """
    if is_fx_tracing() or len(parts) < 2:
        return None
    big = None
    for q, t in enumerate(parts):
        if isinstance(t, torch.fx.Proxy):
            return None
        m = getattr(t, "_dlrmv4_catfused", None)
        if m is None or m[1] != q or m[2] != len(parts):
            return None
        if big is None:
            big = m[0]
        elif m[0] is not big:
            return None
    return big


class ContextualPreprocessor(InputPreprocessor):
    def __init__(
        self,
        input_embedding_dim: int,
        hidden_dim: int,
        output_embedding_dim: int,
        contextual_feature_to_max_length: Dict[str, int],
        contextual_feature_to_min_uih_length: Dict[str, int],
        action_embedding_dim: int = 8,
        action_feature_name: str = "",
        action_weights: Optional[List[int]] = None,
        additional_embedding_features: List[str] = [],
        action_embedding_init_std: float = 0.1,
        is_inference: bool = True,
        fuse_linear_norm: bool = False,
        fuse_ln_residual: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        # F1: fold each Linear's bias gradient into the following norm's
        # backward, deleting the standalone aten::sum over dx. Off by default
        # until an e2e ladder has measured it. See ops/fused_linear_norm.py.
        self._fuse_linear_norm: bool = fuse_linear_norm
        # F2 rides INSIDE F1's container -- it is the same norm kernel gaining
        # an epilogue -- so it cannot be armed on its own. Asking for it
        # without F1 is a request this code cannot honour, and honouring it
        # silently at False would produce an arm that measures nothing while
        # the log says it is on. Raise instead.
        if fuse_ln_residual and not fuse_linear_norm:
            raise ValueError(
                "fuse_ln_residual=True requires fuse_linear_norm=True: the "
                "residual is folded into the fused container's norm epilogue, "
                "which only exists on the F1 path"
            )
        self._fuse_ln_residual: bool = fuse_ln_residual
        self._output_embedding_dim: int = output_embedding_dim
        self._input_embedding_dim: int = input_embedding_dim
        self._hidden_dim: int = hidden_dim
        self._contextual_feature_to_max_length: Dict[str, int] = (
            contextual_feature_to_max_length
        )
        self._max_contextual_seq_len: int = sum(
            contextual_feature_to_max_length.values()
        )
        self._contextual_feature_to_min_uih_length: Dict[str, int] = (
            contextual_feature_to_min_uih_length
        )
        if self._max_contextual_seq_len > 0:
            std = 1.0 * sqrt(
                2.0 / float(input_embedding_dim + self._output_embedding_dim)
            )
            self._batched_contextual_linear_weights: torch.nn.Parameter = (
                torch.nn.Parameter(
                    torch.empty(
                        (
                            self._max_contextual_seq_len,
                            input_embedding_dim,
                            self._output_embedding_dim,
                        )
                    ).normal_(0.0, std)
                )
            )
            self._batched_contextual_linear_bias: torch.nn.Parameter = (
                torch.nn.Parameter(
                    torch.empty(
                        (self._max_contextual_seq_len, self._output_embedding_dim)
                    ).fill_(0.0)
                )
            )
        self._content_embedding_mlp: torch.nn.Module = self._build_mlp(
            self._input_embedding_dim
        )
        self._additional_embedding_features: List[str] = additional_embedding_features
        self._additional_embedding_mlp: torch.nn.Module = self._build_mlp(
            self._input_embedding_dim * len(additional_embedding_features)
        )
        self._action_feature_name: str = action_feature_name
        self._action_weights: Optional[List[int]] = action_weights
        if self._action_weights is not None:
            self._action_encoder: ActionEncoder = ActionEncoder(
                action_feature_name=action_feature_name,
                action_weights=self._action_weights,
                action_embedding_dim=action_embedding_dim,
                embedding_init_std=action_embedding_init_std,
                is_inference=is_inference,
            )
            self._action_embedding_mlp: torch.nn.Module = self._build_mlp(
                self._action_encoder.output_embedding_dim
            )

    def _build_mlp(self, in_features: int) -> torch.nn.Module:
        """Linear -> SwishLayerNorm -> Linear -> LayerNorm.

        The two containers are interchangeable by construction:
        FusedLinearNormMLP is an nn.Sequential SUBCLASS holding the same
        children in the same order, so state_dict keys ("0.weight", "1.weight",
        ...) are identical either way and a checkpoint crosses the F1 flag in
        both directions. Only the backward differs -- see
        ops/fused_linear_norm.py.
        """
        cls = FusedLinearNormMLP if self._fuse_linear_norm else torch.nn.Sequential
        return cls(
            torch.nn.Linear(
                in_features=in_features,
                out_features=self._hidden_dim,
            ),
            SwishLayerNorm(self._hidden_dim),
            torch.nn.Linear(
                in_features=self._hidden_dim,
                out_features=self._output_embedding_dim,
            ),
            LayerNorm(self._output_embedding_dim),
        ).apply(init_mlp_weights_optional_bias)

    def _mlp_add(
        self,
        mlp: torch.nn.Module,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """`residual + mlp(x)`, folded into the MLP's last norm when enabled.

        The fused path adds the residual inside the last norm. Otherwise this
        is `residual + mlp(x)`.
        """
        if self._fuse_ln_residual:
            return mlp(x, residual=residual)
        return residual + mlp(x)

    def forward(  # noqa C901
        self,
        max_uih_len: int,
        max_targets: int,
        total_uih_len: int,
        total_targets: int,
        seq_lengths: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[
        int,
        int,
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        # The preprocessor MLP GEMM pins (ops/blas_solution_pins.py, preproc_*)
        # are keyed on this row count. _item_forward normally registers the same
        # count first; registering here too keeps the pins reachable without
        # depending on that. No-op after the first call per count. Under the
        # pipeline's fx trace the shape is a Proxy, hence the guard.
        num_rows = seq_embeddings.shape[0]
        if isinstance(num_rows, int):
            install_for_tokens(num_rows)
        output_seq_embeddings = self._content_embedding_mlp(seq_embeddings)
        if len(self._additional_embedding_features) > 0:
            _parts = [
                seq_payloads[feature]
                for feature in self._additional_embedding_features
            ]
            _fused = None
            if not torch.jit.is_scripting():
                _fused = _catfused_base(_parts)
            if _fused is not None:
                additional_embeddings = _fused
            else:
                additional_embeddings = torch.cat(_parts, dim=1)
            output_seq_embeddings = self._mlp_add(
                self._additional_embedding_mlp,
                additional_embeddings,
                output_seq_embeddings,
            )
        max_seq_len = max_uih_len + max_targets
        target_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(num_targets)
        seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(seq_lengths)
        uih_offsets = seq_offsets - target_offsets
        if self._action_weights is not None:
            action_embeddings = self._action_encoder(
                max_uih_len=max_uih_len,
                max_targets=max_targets,
                uih_offsets=uih_offsets,
                target_offsets=target_offsets,
                seq_embeddings=seq_embeddings,
                seq_payloads=seq_payloads,
            )
            output_seq_embeddings = self._mlp_add(
                self._action_embedding_mlp,
                action_embeddings,
                output_seq_embeddings,
            )

        output_max_seq_len = max_seq_len
        output_total_uih_len = total_uih_len
        output_total_targets = total_targets
        output_seq_lengths = seq_lengths
        output_num_targets = num_targets
        output_seq_timestamps = seq_timestamps
        output_seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            output_seq_lengths
        )
        # concat contextual embeddings
        if self._max_contextual_seq_len > 0:
            contextual_input_embeddings = get_contextual_input_embeddings(
                seq_lengths=seq_lengths,
                seq_payloads=seq_payloads,
                contextual_feature_to_max_length=self._contextual_feature_to_max_length,
                contextual_feature_to_min_uih_length=self._contextual_feature_to_min_uih_length,
                dtype=seq_embeddings.dtype,
            )
            contextual_embeddings = torch.baddbmm(
                self._batched_contextual_linear_bias.view(
                    -1, 1, self._output_embedding_dim
                ).to(contextual_input_embeddings.dtype),
                contextual_input_embeddings.view(
                    -1, self._max_contextual_seq_len, self._input_embedding_dim
                ).transpose(0, 1),
                self._batched_contextual_linear_weights.to(
                    contextual_input_embeddings.dtype
                ),
            ).transpose(0, 1)
            output_seq_embeddings = concat_2D_jagged(
                max_seq_len=self._max_contextual_seq_len + output_max_seq_len,
                values_left=fx_unwrap_optional_tensor(contextual_embeddings).reshape(
                    -1, self._output_embedding_dim
                ),
                values_right=output_seq_embeddings,
                max_len_left=self._max_contextual_seq_len,
                max_len_right=output_max_seq_len,
                offsets_left=None,
                offsets_right=output_seq_offsets,
                kernel=self.hammer_kernel(),
            )
            output_seq_timestamps = concat_2D_jagged(
                max_seq_len=self._max_contextual_seq_len + output_max_seq_len,
                values_left=torch.zeros(
                    (output_seq_lengths.size(0) * self._max_contextual_seq_len, 1),
                    dtype=output_seq_timestamps.dtype,
                    device=output_seq_timestamps.device,
                ),
                values_right=output_seq_timestamps.unsqueeze(-1),
                max_len_left=self._max_contextual_seq_len,
                max_len_right=output_max_seq_len,
                offsets_left=None,
                offsets_right=output_seq_offsets,
                kernel=self.hammer_kernel(),
            ).squeeze(-1)
            output_max_seq_len = output_max_seq_len + self._max_contextual_seq_len
            output_seq_lengths = output_seq_lengths + self._max_contextual_seq_len
            output_seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                output_seq_lengths
            )
            output_total_uih_len = (
                output_total_uih_len
                + self._max_contextual_seq_len * output_seq_lengths.size(0)
            )

        return (
            output_max_seq_len,
            output_total_uih_len,
            output_total_targets,
            output_seq_lengths,
            output_seq_offsets,
            output_seq_timestamps,
            output_seq_embeddings,
            output_num_targets,
            seq_payloads,
        )
