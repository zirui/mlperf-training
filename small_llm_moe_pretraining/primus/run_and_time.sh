#!/bin/bash

# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
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

if [[ -z "${SEED}" ]]; then
    echo "SEED is not set!" >&2
    exit 1
fi

mkdir -p /results/artifacts

cd /workspace/code
export MLPERF_VERBOSE_LOGS="${MLPERF_VERBOSE_LOGS:-0}"

# Under a multi-node scheduler wrapper, inherit rendezvous + node sizing from
# SLURM so the same benchmark config scales without edits. Single-node jobs
# fall through to the config defaults.
if [[ -n "${SLURM_NNODES:-}" && "${SLURM_NNODES}" -gt 1 ]]; then
    NNODES="${SLURM_NNODES}"
    NODE_RANK="${SLURM_NODEID:-0}"
fi

# Repeated seeds must not overwrite diagnostics from an earlier trial.
if [[ -z "${TRAIN_LOG_FILE:-}" ]]; then
    TRAIN_LOG_FILE=$(mktemp --suffix=.full.log "/results/artifacts/train.${SEED}.node${NODE_RANK:-0}.XXXXXX")
    # Let the host artifact collector read logs created by a root-run container.
    chmod 644 "${TRAIN_LOG_FILE}"
fi

echo "============================================"
echo "MLPerf GPT-OSS-20B Training"
echo "============================================"
echo "Config: ${EXP}"
echo "Data:   ${DATA_PATH}"
echo "GPUs:   ${GPUS_PER_NODE}"
echo "Nodes:  ${NNODES}"
echo "Rank:   ${NODE_RANK}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Turbo attention: ${PRIMUS_USE_TURBO_ATTENTION:-true}"
echo "============================================"

# Start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# Filter at the launcher boundary: primus-cli does not import the legacy
# src/train.py quiet-mode hooks. Keep all stdout/stderr in the diagnostic log,
# while streaming MLPerf records to the console and submission run log.
set +e
"${PRIMUS_PATH}/primus-cli" direct -- \
    train pretrain \
    --config "${EXP}" \
    2>&1 | tee "${TRAIN_LOG_FILE}" | \
    awk -v verbose="${MLPERF_VERBOSE_LOGS}" 'verbose == "1" || /:::MLLOG/ { print; fflush(); }'
pipeline_status=("${PIPESTATUS[@]}")
ret_code=${pipeline_status[0]}
set -e

# End timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

if [[ $ret_code != 0 ]]; then
    echo "Training failed with exit code: $ret_code. Full log: ${TRAIN_LOG_FILE}" >&2
    if [[ "${MLPERF_VERBOSE_LOGS}" != 1 ]]; then
        tail -n 100 "${TRAIN_LOG_FILE}" >&2 || true
    fi
    exit "$ret_code"
fi

# Do not report success if saving or streaming the training log failed.
for status in "${pipeline_status[@]:1}"; do
    if [[ $status != 0 ]]; then
        echo "Training log pipeline failed with exit code: $status" >&2
        exit "$status"
    fi
done

# Report result only after training and logging succeeded.
result=$(( end - start ))
result_name="GPT_OSS_20B"
echo "RESULT,$result_name,,$result,AMD,$start_fmt"

exit 0
