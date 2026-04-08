#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION & AMD. All rights reserved.
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

set -e

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHDYNAMO_DISABLE=1

readonly node_rank="${SLURM_NODEID:-0}"
readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-0}}}"

if [[ -z "${SEED}" ]]; then
    echo "SEED is not set!"
    exit 1
fi

# Create results directory
mkdir -p /results

cd /workspace/code

echo "============================================"
echo "MLPerf Flux1 Training (Primus, NeMo-free)"
echo "============================================"
echo "Config: ${EXP:-N/A}"
echo "Data:   ${DATA_PATH:-/dataset/energon}"
echo "GPUs:   ${GPUS_PER_NODE:-${DGXNGPU:-8}}"
echo "Nodes:  ${NNODES:-${DGXNNODES:-1}}"
echo "Seed:   ${SEED}"
echo "============================================"

# Start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

declare -a CMD

# Check if we're in a multi-node SLURM job
if [[ -n "${SLURM_NNODES}" ]] && [[ "${SLURM_NNODES}" -gt 1 ]]; then
    echo "Multi-node SLURM detected: ${SLURM_NNODES} nodes"
    echo "Node rank: ${SLURM_NODEID}, Master: ${MASTER_ADDR}:${MASTER_PORT}"

    CMD=( 'torchrun' \
        "--nnodes=${SLURM_NNODES}" \
        "--nproc_per_node=${DGXNGPU}" \
        "--node_rank=${SLURM_NODEID}" \
        "--master_addr=${MASTER_ADDR}" \
        "--master_port=${MASTER_PORT:-29501}" \
        "--rdzv_backend=c10d" \
        "--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT:-29501}" )

elif [[ "${LOCAL_WORLD_SIZE:-0}" -gt 1 ]]; then
    CMD=( 'python' '-u')
else
    CMD=( 'torchrun' "--nproc_per_node=${GPUS_PER_NODE:-${DGXNGPU:-8}}" )
fi

if [[ "${node_rank}" -eq 0 && "${local_rank}" -eq 0 ]]; then
    echo "SEED=${SEED}"
fi

# All training params (optimizer, LR, batch sizes, etc.) are loaded from
# the YAML config file pointed to by EXP. No CLI args needed here.
${CMD[@]} src/train.py ${EXTRA_ARGS}

ret_code=$?

# End timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# Report result
result=$(( end - start ))
result_name="FLUX1"
echo "RESULT,$result_name,,$result,AMD,$start_fmt"

if [[ $ret_code != 0 ]]; then
    echo "Training failed with exit code: $ret_code"
    exit $ret_code
fi

exit 0
