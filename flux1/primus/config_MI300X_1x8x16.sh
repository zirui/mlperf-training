#!/bin/bash
# =============================================================================
# MLPerf Flux1 Configuration for MI300X (1 node, 8 GPUs)
# =============================================================================

# Load default settings
source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# -----------------------------------------------------------------------------
# System Configuration
# -----------------------------------------------------------------------------
export DGXSYSTEM=MI300X_1x8x16
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
export EXP=/workspace/code/conf/flux1-pretrain-fp8.yaml
export DATA_PATH=/dataset/energon

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
export MAX_STEPS=${MAX_STEPS:-1000000}
export WARMUP_STEPS=${WARMUP_STEPS:-1600}
export BATCHSIZE=512
export LEARNING_RATE=${LEARNING_RATE:-"0.0002"}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
export MBS=16
export FP8_RECIPE="delayed"

# Flux model architecture
export NUM_JOINT_LAYERS=19
export NUM_SINGLE_LAYERS=38
export HIDDEN_SIZE=3072
export NUM_ATTENTION_HEADS=24
export IN_CHANNELS=64
export CONTEXT_DIM=4096
export MODEL_CHANNELS=256
export VEC_IN_DIM=768
export GUIDANCE_EMBED=False

# FP8 configuration
export FP8=hybrid
export FP8_MARGIN=0
export FP8_AMAX_HISTORY_LEN=1024
export FP8_AMAX_COMPUTE_ALGO=max

# Evaluation
export EVAL_INTERVAL=${EVAL_INTERVAL:-512}

# System parameters
export DGXNNODES=1
export DGXNGPU=8
export SEGMENT=${DGXNNODES:-1}

# -----------------------------------------------------------------------------
# NVTE — FP8 & Cast Transpose
# -----------------------------------------------------------------------------
export NVTE_RS_STRIDED_ATOMIC=2
export NVTE_FP8_DPA_BWD=1
export NVTE_FUSED_ATTN=1
export NVTE_FUSED_ATTN_CK=1
export NVTE_FUSED_ATTN_AOTRITON=1
export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0
export NVTE_USE_HIPBLASLT=1
export NVTE_USE_CAST_TRANSPOSE_TRITON=1
export NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=0
export USE_TE_SWIGLU=1

export NVTE_ROCM_ENABLE_MXFP8=0

# -----------------------------------------------------------------------------
# NVTE — FMHA / CK Backend
# MI300X uses gfx942 — CK v3 not available, use CK (v1/v2) backends
# -----------------------------------------------------------------------------
export NVTE_CK_USES_FWD_V3=0
export NVTE_CK_USES_BWD_V3=0
export NVTE_CK_IS_V3_ATOMIC_FP32=0
export NVTE_FMHA_BACKEND_DENSE_FWD=ck
export NVTE_FMHA_BACKEND_DENSE_BWD=ck

export CK_FUSED_ATTN_LOG_CONFIG=0
export NVTE_LOG_CK_CONFIG=0
export NVTE_LOG_FUSED_ATTN_CONFIG=0
export CHECK_FOR_NAN_IN_GRAD=0

export FUSED_SOFTMAX=0
export RMSNORM_CAST=0
export ENABLE_TRANSPOSE_CACHE=0

export USE_HIPBLASLT=1
export TORCH_BLAS_PREFER_HIPBLASLT=1

# -----------------------------------------------------------------------------
# ROCm / System Runtime
# -----------------------------------------------------------------------------
export HIP_FORCE_DEV_KERNARG=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HSA_KERNARG_POOL_SIZE=12582912
export TORCH_NCCL_HIGH_PRIORITY=1
export ENABLE_NUMA_BINDING=1

# -----------------------------------------------------------------------------
# MLPerf Logging
# -----------------------------------------------------------------------------
export MLLOG_TRAIN_LOSS_LOG_FREQ=32
export MLLOG_TARGET_EVAL_LOSS=${TARGET_ACCURACY}
export MLLOG_OUTPUT_FILE=/results/mlperf_output.log
export MLLOG_SAVE_TO_FILE=0
export MLLOG_SUBMISSION_BENCHMARK=flux1
export MLLOG_SUBMISSION_DIVISION=closed
export MLLOG_SUBMISSION_ORG=AMD
export MLLOG_SUBMISSION_PLATFORM=MI300X

# Walltime
export WALLTIME_RUNANDTIME=200
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

export USE_CUDA_GRAPH=False
