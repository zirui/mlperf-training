#!/bin/bash

set -e

# Create results directory
mkdir -p /results

cd /workspace/code
TRAIN_LOG_FILE="${TRAIN_LOG_FILE:-train.mlperfpretrain.exp.log}"
: "${PRIMUS_PATH:=/workspace/Primus}"

if [[ -n "${SLURM_NNODES:-}" && "${SLURM_NNODES}" -gt 1 ]]; then
    NNODES="${SLURM_NNODES}"
    NODE_RANK="${SLURM_NODEID:-0}"
fi

echo "============================================"
echo "MLPerf LLama3.1 8B Training"
echo "============================================"
echo "Config: ${EXP}"
echo "Data:   ${DATA_PATH}"
echo "GPUs:   ${GPUS_PER_NODE}"
echo "Nodes:  ${NNODES}"
echo "Rank:   ${NODE_RANK}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "ENABLE_PRIMUS_TURBO=${ENABLE_PRIMUS_TURBO:-}"
echo "PRIMUS_USE_TURBO_ATTENTION=${PRIMUS_USE_TURBO_ATTENTION:-}"
echo "PRIMUS_USE_TURBO_GEMM=${PRIMUS_USE_TURBO_GEMM:-}"
echo "PRIMUS_ATTENTION_BACKEND=${PRIMUS_ATTENTION_BACKEND:-}"
echo "PRIMUS_TURBO_ATTN_BACKEND=${PRIMUS_TURBO_ATTN_BACKEND:-}"
echo "PRIMUS_TURBO_AUTO_TUNE=${PRIMUS_TURBO_AUTO_TUNE:-}"
echo "PRIMUS_TURBO_GEMM_BACKEND=${PRIMUS_TURBO_GEMM_BACKEND:-}"
echo "PRIMUS_TURBO_FUSED_DENSE_MLP=${PRIMUS_TURBO_FUSED_DENSE_MLP:-}"
echo "PRIMUS_USE_TURBO_RMS_NORM=${PRIMUS_USE_TURBO_RMS_NORM:-}"
echo "PRIMUS_FUSED_RESIDUAL_NORM_V2=${PRIMUS_FUSED_RESIDUAL_NORM_V2:-}"
echo "PRIMUS_FUSED_RMSNORM_MXFP4=${PRIMUS_FUSED_RMSNORM_MXFP4:-}"
echo "APPLY_SKIP_Y_STE=${APPLY_SKIP_Y_STE:-}"
echo "PRIMUS_EVAL_ITERS=${PRIMUS_EVAL_ITERS:-}"
echo "PRIMUS_ATTN_SOFTMAX_FP32=${PRIMUS_ATTN_SOFTMAX_FP32:-}"
echo "PRIMUS_ACCUM_GRADS_FP32=${PRIMUS_ACCUM_GRADS_FP32:-}"
echo "============================================"


python3 - <<'PY'
try:
    from primus_turbo._version import __commit_id__, __version__
    print(f"primus-turbo {__version__} commit={__commit_id__}")
except Exception as exc:
    print(f"primus-turbo version unavailable: {exc}")
PY

# Start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# Launch through Primus CLI. The YAML stage mlperf_pretrain owns MLPerf
# logging (via AMD-AGI/mlperf-common).
set +e
"${PRIMUS_PATH}/primus-cli" direct -- \
    train pretrain \
    --config "${EXP}" \
    2>&1 | tee "${TRAIN_LOG_FILE}"
ret_code=${PIPESTATUS[0]}
set -e

# End timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# Report result
result=$(( end - start ))
result_name="LLAMA3.1_8B"
echo "RESULT,$result_name,,$result,AMD,$start_fmt"

if [[ $ret_code != 0 ]]; then
    echo "Training failed with exit code: $ret_code"
    exit $ret_code
fi

exit 0
