#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

set -e

if [[ -z "${SEED}" ]]; then
    echo "SEED is not set!" >&2
    exit 1
fi

mkdir -p /results/artifacts

cd /workspace/code
export MLPERF_VERBOSE_LOGS="${MLPERF_VERBOSE_LOGS:-0}"

# Repeated seeds must not overwrite diagnostics from an earlier trial.
if [[ -z "${TRAIN_LOG_FILE:-}" ]]; then
    TRAIN_LOG_FILE=$(mktemp --suffix=.full.log "/results/artifacts/train.${SEED}.node${NODE_RANK:-0}.XXXXXX")
    chmod 644 "${TRAIN_LOG_FILE}"
fi
export MLPERF_LOG_PATH="${MLPERF_LOG_PATH:-${TRAIN_LOG_FILE%.full.log}.mlperf.log}"

# Start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# Drop hipBLASLt's per-shape lines. Keep sed block-buffered; line buffering stalls the trainer.
set +e
python3 -m generative_recommenders.dlrm_v4.train.train_ranker \
    --dataset yambda-5b \
    --mode streaming-train-eval \
    2>&1 | sed -e '/Latency not found for/d' -e '/Returning latency value of/d' | \
    tee "${TRAIN_LOG_FILE}" | \
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

for status in "${pipeline_status[@]:1}"; do
    if [[ $status != 0 ]]; then
        echo "Training log pipeline failed with exit code: $status" >&2
        exit "$status"
    fi
done

result=$(( end - start ))
result_name="DLRMV4"
echo "RESULT,$result_name,,$result,AMD,$start_fmt"

exit 0
