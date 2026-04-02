#!/bin/bash
# =============================================================================
# MLPerf GPT-OSS-20B Configuration for MI355X (1 node, 8 GPUs)
# =============================================================================

# -----------------------------------------------------------------------------
# System Configuration
# -----------------------------------------------------------------------------
export DGXSYSTEM=MI355X_1x8x1
export GPUS_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29501

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
export PRIMUS_PATH=/workspace/Primus
export PYTHONPATH="${PRIMUS_PATH}:${PRIMUS_PATH}/third_party/Megatron-LM:${PYTHONPATH}"
export EXP=/workspace/code/conf/gpt_oss_20B-pretrain-fp8.yaml
export DATA_PATH=/data
export MODEL=/model

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
export PRIMUS_MICRO_BATCH_SIZE=2
export PRIMUS_GLOBAL_BATCH_SIZE=64
export PRIMUS_LR=1.0e-3
export PRIMUS_MIN_LR=1.0e-4 # Set to 10% of max LR
export PRIMUS_TRAIN_ITERS=1200000
export PRIMUS_LR_WARMUP_ITERS=192
export PRIMUS_LR_DECAY_ITERS=$((PRIMUS_TRAIN_ITERS-PRIMUS_LR_WARMUP_ITERS))
# export SEED=30279

# Evaluation frequency (sample-based, adjusts automatically with GBS)
export EVAL_SAMPLES_INTERVAL=12288   # Evaluate every 12,288 samples
export PRIMUS_EVAL_INTERVAL=$((EVAL_SAMPLES_INTERVAL / PRIMUS_GLOBAL_BATCH_SIZE))  # Auto-computed

# -----------------------------------------------------------------------------
# Optimizations
# -----------------------------------------------------------------------------
export PRIMUS_APPLY_ROPE_FUSION=True
export PRIMUS_FP8_RECIPE=hybrid
export HIP_FORCE_DEV_KERNARG=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export TORCH_NCCL_HIGH_PRIORITY=1
export ENABLE_NUMA_BINDING=1
export HSA_KERNARG_POOL_SIZE=12582912

export PRIMUS_TP=1
export PRIMUS_PP=1
export PRIMUS_EP=8

# -----------------------------------------------------------------------------
# MLPerf Logging
# -----------------------------------------------------------------------------
export MLLOG_TRAIN_LOSS_LOG_FREQ=32
export MLLOG_TARGET_EVAL_LOSS=3.34
export MLLOG_OUTPUT_FILE=/results/mlperf_output.log
export MLLOG_SAVE_TO_FILE=0
export MLLOG_SUBMISSION_BENCHMARK=gpt_oss_20b
export MLLOG_SUBMISSION_DIVISION=closed
export MLLOG_SUBMISSION_ORG=AMD
export MLLOG_SUBMISSION_PLATFORM=MI355X

# -----------------------------------------------------------------------------
# TE Configuration
# -----------------------------------------------------------------------------
export NVTE_ROCM_ENABLE_MXFP8=0
export NVTE_USE_CAST_TRANSPOSE_TRITON=0
export NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=1
export NVTE_CK_IS_V3_ATOMIC_FP32=0
export NVTE_CK_USES_FWD_V3=1
export NVTE_CK_USES_BWD_V3=1
export NVTE_CK_V3_BF16_CVT=0              # 0=RTNE (default on gfx950), 1=RTNA, 2=RTZ (1,2 deprecated on gfx950)
export NVTE_FMHA_BACKEND_DENSE_FWD=ck_v3  # Dense fwd: ck_v3 | ck | aotriton
export NVTE_FMHA_BACKEND_DENSE_BWD=ck_v3  # Dense bwd: ck_v3 | ck | aotriton
export NVTE_FMHA_BACKEND_SWA_FWD=ck       # SWA fwd:   ck | aotriton (no ck_v3 support)
export NVTE_FMHA_BACKEND_SWA_BWD=ck       # SWA bwd:   ck | aotriton (no ck_v3 support)
