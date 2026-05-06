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
export PRIMUS_MICRO_BATCH_SIZE=4
export PRIMUS_GLOBAL_BATCH_SIZE=32
export EVAL_ITERS=$((1024 / PRIMUS_GLOBAL_BATCH_SIZE))  # MLPerf closed: eval_iters * GBS = 1024 eval samples
export PRIMUS_LR=8.0e-4
export PRIMUS_MIN_LR=8.0e-5 # Set to 10% of max LR
export PRIMUS_TRAIN_ITERS=1200000
export PRIMUS_LR_WARMUP_ITERS=128
export PRIMUS_LR_DECAY_ITERS=$((PRIMUS_TRAIN_ITERS-PRIMUS_LR_WARMUP_ITERS)) # 1200000 - 128 = 1199872
# export SEED=30279

# Evaluation frequency (sample-based, adjusts automatically with GBS)
export EVAL_SAMPLES_INTERVAL=12288   # Evaluate every 12,288 samples
export PRIMUS_EVAL_INTERVAL=$((EVAL_SAMPLES_INTERVAL / PRIMUS_GLOBAL_BATCH_SIZE))  # Auto-computed

# -----------------------------------------------------------------------------
# Parallelism
# -----------------------------------------------------------------------------
export PRIMUS_TP=1
export PRIMUS_PP=1
export PRIMUS_EP=1

# -----------------------------------------------------------------------------
# Primus Configuration
# -----------------------------------------------------------------------------
export PRIMUS_APPLY_ROPE_FUSION=True
export PRIMUS_FP8_RECIPE=hybrid
export PRIMUS_TURBO_GROUPED_GEMM_BACKEND=TRITON
export PRIMUS_GRAD_REDUCE_IN_BF16=true
export USE_TURBO_RMS_NORM=true

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

# -----------------------------------------------------------------------------
# NVTE — FP8 & Cast Transpose
# -----------------------------------------------------------------------------
export NVTE_ROCM_ENABLE_MXFP8=0
export NVTE_USE_CAST_TRANSPOSE_TRITON=0
export NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=1

# -----------------------------------------------------------------------------
# NVTE — FMHA / CK Backend
# -----------------------------------------------------------------------------
export NVTE_FLASH_ATTN=0                  # Disable FlashAttention so FusedAttention (CK/ASM) is used
export NVTE_CK_USES_FWD_V3=1              # Globally on; aiter selects v3 vs CK-tile internally
export NVTE_CK_USES_BWD_V3=1              # Globally on; aiter selects v3 vs CK-tile internally
export NVTE_FMHA_USE_BSHD=0               # Native SBHD path (aiter c4b33df0 supports it; skips Megatron's SBHD↔BSHD shim transposes)
export NVTE_CK_IS_V3_ATOMIC_FP32=0        # 0=a16 (atomic fp16), 1=a32 (atomic fp32) 
export NVTE_CK_V3_BF16_CVT=2              # 0=RTNE (default on gfx950), 1=RTNA, 2=RTZ (1,2 deprecated on gfx950)

# -----------------------------------------------------------------------------
# NVTE — Debug (disable for production runs)
# -----------------------------------------------------------------------------
# export NVTE_DEBUG=1
# export NVTE_DEBUG_LEVEL=2
# export NVTE_FMHA_DEBUG=1
# export NVTE_LOG_FUSED_ATTN_CONFIG=1
# export NVTE_LOG_CK_CONFIG=1

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

# -----------------------------------------------------------------------------
# Synthetic Warmup (kernel pre-compilation)
# -----------------------------------------------------------------------------
export SYNTH_WARMUP_STEPS=3

# -----------------------------------------------------------------------------
# MoE Token Dispatcher
# -----------------------------------------------------------------------------
# Skip sort_chunks_by_idxs when the per-local-expert index is an identity
# permutation (fires at EP=1/TP=1). Set to 0 to run the original path; useful
# for A/B measurements. See patches/megatron_moe_skip_identity_sort.patch.
export MOE_SKIP_IDENTITY_SORT=1

# -----------------------------------------------------------------------------
# Run-log verbosity
# -----------------------------------------------------------------------------
# MLPerf run-log verbosity. Default 0 keeps only :::MLLOG + ``run_and_time.sh``
# banners on stdout. Set to 1 to restore the full framework output (Primus
# loguru banners / Megatron / TE / aiter / Gloo / torchrun / hipify / ...)
# when debugging. See src/_log_suppression.py for the full strategy.
export MLPERF_VERBOSE_LOGS=${MLPERF_VERBOSE_LOGS:-0}

# fused rms and swiglu no cat
export PRIMUS_FUSED_RESIDUAL_NORM=1
export PRIMUS_MOE_SWIGLU_NOCAT=1
export MLLOG_BLOCK_TPUT_LOG=0
