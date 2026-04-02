#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_HOME=/hf_home

readonly node_rank="${SLURM_NODEID:-0}"
readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-0}}}"

if [[ -z "${SEED}" ]]; then
    echo "SEED is not set!"
    exit 1
fi

declare -a CMD

# Check if we're in a multi-node SLURM job
if [[ -n "${SLURM_NNODES}" ]] && [[ "${SLURM_NNODES}" -gt 1 ]]; then
    echo "Multi-node SLURM detected: ${SLURM_NNODES} nodes"
    echo "Node rank: ${SLURM_NODEID}, Master: ${MASTER_ADDR}:${MASTER_PORT}"

    # Use torchrun for multi-node with proper coordination
    CMD=( 'torchrun' \
        "--nnodes=${SLURM_NNODES}" \
        "--nproc_per_node=${DGXNGPU}" \
        "--node_rank=${SLURM_NODEID}" \
        "--master_addr=${MASTER_ADDR}" \
        "--master_port=${MASTER_PORT:-29501}" \
        "--rdzv_backend=c10d" \
        "--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT:-29501}" )

elif [[ "${LOCAL_WORLD_SIZE}" -gt 1 ]]; then
    # Mode 1: Slurm launched a task for each GPU and set some envvars
    CMD=( 'python' '-u')
else
    # Single node interactive run
    CMD=( 'torchrun' "--nproc_per_node=${DGXNGPU}" )
fi

# Assert $RANDOM is usable
if [[ -z "${RANDOM}" ]]; then
    echo "RANDOM is not set!" >&2
    exit 1
fi

if [[ "${node_rank}" -eq 0 && "${local_rank}" -eq 0 ]]; then
    echo "SEED=${SEED}"
fi

${CMD[@]} src/train.py model=${MODEL} data=${DATA} ${EXTRA_ARGS}

exit 0
