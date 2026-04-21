#!/bin/bash
# =============================================================================
# MLPerf LLama3.1 8B Configuration for MI355X (1 node, 8 GPUs)
# =============================================================================

# -----------------------------------------------------------------------------
# System Configuration
# -----------------------------------------------------------------------------
export DGXSYSTEM=MI355X_1x8x1
export GPUS_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29502

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
export PRIMUS_PATH=/workspace/Primus
export PRIMUS_MLPERF=1
export PYTHONPATH="${PRIMUS_PATH}:${PRIMUS_PATH}/third_party/Megatron-LM:${PYTHONPATH}"
export EXP=/workspace/code/conf/llama3.1_8B-pretrain-fp8.yaml
export DATA_PATH=/data
export MODEL=/model

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
export PRIMUS_MICRO_BATCH_SIZE=2
export PRIMUS_GLOBAL_BATCH_SIZE=32
export PRIMUS_LR=8e-4
export PRIMUS_MIN_LR=8e-5
export PRIMUS_TRAIN_ITERS=1200000
export PRIMUS_LR_WARMUP_ITERS=128
export PRIMUS_LR_DECAY_ITERS=$((PRIMUS_TRAIN_ITERS-PRIMUS_LR_WARMUP_ITERS))
# export SEED=31952

# Evaluation frequency (sample-based, adjusts automatically with GBS)
export EVAL_SAMPLES_INTERVAL=12288
export PRIMUS_EVAL_INTERVAL=$((EVAL_SAMPLES_INTERVAL / PRIMUS_GLOBAL_BATCH_SIZE))

# -----------------------------------------------------------------------------
# Primus Configuration
# -----------------------------------------------------------------------------
export PRIMUS_APPLY_ROPE_FUSION=True
export PRIMUS_FP8_RECIPE=hybrid

# -----------------------------------------------------------------------------
# ROCm / System Runtime
# -----------------------------------------------------------------------------
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export HSA_ENABLE_INTERRUPT=0
export GPU_MAX_HW_QUEUES=2
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_CHECKS_DISABLE=1

# -----------------------------------------------------------------------------
# NVTE — FP8 & Fused Attention
# -----------------------------------------------------------------------------
export NVTE_FUSED_ATTN=1
export NVTE_FUSED_ATTN_CK=1
export NVTE_FUSED_ATTN_AOTRITON=1
export NVTE_CK_USES_FWD_V3=1
export NVTE_CK_USES_BWD_V3=1
export NVTE_CK_IS_V3_ATOMIC_FP32=0
export USE_TE_SWIGLU=1

export NVTE_ASYNC_AMAX_REDUCTION=1
export NVTE_DP_AMAX_REDUCE_INTERVAL=0

# -----------------------------------------------------------------------------
# MLPerf Logging
# -----------------------------------------------------------------------------
export ENABLE_MLLOG=1
export MLLOG_OUTPUT_FILE=/results/mlperf_output.log
export MLLOG_TRAIN_LOSS_LOG_FREQ=100
export MLLOG_TARGET_EVAL_LOSS=3.3
export MLLOG_SUBMISSION_BENCHMARK=llama31_8b
export MLLOG_SUBMISSION_DIVISION=closed
export MLLOG_SUBMISSION_ORG=AMD
export MLLOG_SUBMISSION_PLATFORM=MI355X
