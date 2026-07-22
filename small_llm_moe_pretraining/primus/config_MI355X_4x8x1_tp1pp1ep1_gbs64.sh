#!/bin/bash
# =============================================================================
# MLPerf GPT-OSS-20B Configuration for MI355X (4 node, 8 GPUs)
# =============================================================================

# -----------------------------------------------------------------------------
# System Configuration
# -----------------------------------------------------------------------------
export DGXSYSTEM=MI355X_4x8x1_tp1pp1ep1_gbs64
export GPUS_PER_NODE=8
export NNODES=4
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
export EVAL_ITERS=$((1024 / PRIMUS_GLOBAL_BATCH_SIZE))
export PRIMUS_LR=1.0e-3
export PRIMUS_MIN_LR=1.0e-4
export PRIMUS_TRAIN_ITERS=1200000
export PRIMUS_LR_WARMUP_ITERS=192
export PRIMUS_LR_DECAY_ITERS=$((PRIMUS_TRAIN_ITERS-PRIMUS_LR_WARMUP_ITERS))

export SYNTH_WARMUP_STEPS=10
export EVAL_SAMPLES_INTERVAL=12288
export PRIMUS_EVAL_INTERVAL=$((EVAL_SAMPLES_INTERVAL / PRIMUS_GLOBAL_BATCH_SIZE))

# -----------------------------------------------------------------------------
# Parallelism
# -----------------------------------------------------------------------------
export PRIMUS_TP=1
export PRIMUS_PP=1
export PRIMUS_EP=1

# -----------------------------------------------------------------------------
# Primus Configuration
# -----------------------------------------------------------------------------
export PRIMUS_TURBO_GROUPED_GEMM_BACKEND=TRITON
export PRIMUS_GRAD_REDUCE_IN_BF16=true
export USE_TURBO_RMS_NORM=true
export PRIMUS_FUSED_RESIDUAL_NORM=0
export PRIMUS_MOE_SWIGLU_NOCAT=1

# -----------------------------------------------------------------------------
# ROCm / System Runtime
# -----------------------------------------------------------------------------
export GPU_MAX_HW_QUEUES=2
export HIP_FORCE_DEV_KERNARG=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HSA_KERNARG_POOL_SIZE=12582912
export TORCH_NCCL_HIGH_PRIORITY=1
export ENABLE_NUMA_BINDING=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export HSA_ENABLE_INTERRUPT=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

# -----------------------------------------------------------------------------
# RCCL / NCCL Tuning
# -----------------------------------------------------------------------------
export NCCL_GRAPH_REGISTER=0
export NCCL_LOCAL_REGISTER=0
export NCCL_MIN_P2P_NCHANNELS=32
export NCCL_MIN_CTAS=32
export NCCL_NCHANNELS_PER_NET_PEER=32
export NCCL_NVLS_ENABLE=0
export NCCL_CHECKS_DISABLE=1
export NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
export RCCL_ENABLE_MULTI_STREAM=1
export RCCL_STREAM_PRIORITY=1
# -----------------------------------------------------------------------------
# hipBLASLt
# -----------------------------------------------------------------------------
export USE_HIPBLASLT=1
export TORCH_BLAS_PREFER_HIPBLASLT=1
export HIPBLASLT_TUNING_OVERRIDE_FILE=/workspace/code/tune_gemm_results.txt

# -----------------------------------------------------------------------------
# NVTE — FP8 & Cast Transpose
# -----------------------------------------------------------------------------
export NVTE_ROCM_ENABLE_MXFP8=0
export NVTE_USE_CAST_TRANSPOSE_TRITON=0
export NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=1

# -----------------------------------------------------------------------------
# MoE Token Dispatcher
# -----------------------------------------------------------------------------
export MOE_SKIP_IDENTITY_SORT=1

# -----------------------------------------------------------------------------
# DDP Parameter All-Gather (SDMA)
# -----------------------------------------------------------------------------
export ENABLE_SDMA_ALLGATHER=0

# -----------------------------------------------------------------------------
# NVTE — FMHA / CK Backend
# -----------------------------------------------------------------------------
export NVTE_FLASH_ATTN=0
export NVTE_CK_USES_FWD_V3=1
export NVTE_CK_USES_BWD_V3=1
export NVTE_USE_AITER_ROPE=1
export NVTE_FMHA_USE_BSHD=0
export NVTE_CK_IS_V3_ATOMIC_FP32=0
export NVTE_CK_HOW_V3_BF16_CVT=2
export MLPERF_ENABLE_FWD_ATTN_ASM=1
export FMHA_HD64_ASM_LOG=0

# -----------------------------------------------------------------------------
# NVTE — Debug
# -----------------------------------------------------------------------------
export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0
export NVTE_LOG_FUSED_ATTN_CONFIG=0
export NVTE_LOG_CK_CONFIG=0
export CK_FUSED_ATTN_LOG_CONFIG=0

# -----------------------------------------------------------------------------
# MLPerf Logging
# -----------------------------------------------------------------------------
export LOG_INTERVAL=999999
export MLLOG_TRAIN_LOSS_LOG_FREQ=0
export MLLOG_TARGET_EVAL_LOSS=3.34
export MLLOG_OUTPUT_FILE=/results/mlperf_logging.out
export MLLOG_SAVE_TO_FILE=0
export MLLOG_SUBMISSION_BENCHMARK=gpt_oss_20b
export MLLOG_SUBMISSION_DIVISION=closed
export MLLOG_SUBMISSION_ORG=AMD
export MLLOG_SUBMISSION_PLATFORM=MI355X

export MLLOG_TENSOR_PARALLELISM=1
export MLLOG_PIPELINE_PARALLELISM=1
export MLLOG_CONTEXT_PARALLELISM=1
export MLLOG_EXPERT_PARALLELISM=1
export MLLOG_MICRO_BATCH_SIZE=2
export MLLOG_CONFIG_FILENAME=$(basename "${BASH_SOURCE[0]}")
export MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR='fp8'

export MLLOG_BLOCK_TPUT_LOG=0
export MLPERF_VERBOSE_LOGS=${MLPERF_VERBOSE_LOGS:-0}