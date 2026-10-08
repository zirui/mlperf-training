#!/bin/bash
# =============================================================================
# MLPerf GPT-OSS-20B Configuration for MI355X (1 node, 8 GPUs)
# =============================================================================

# -----------------------------------------------------------------------------
# System Configuration
# -----------------------------------------------------------------------------
export DGXSYSTEM=MI355X_1x8x1_tp1pp1ep1_gbs32
export GPUS_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29501

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
export PRIMUS_PATH=/workspace/Primus
export PYTHONPATH="${PRIMUS_PATH}:${PRIMUS_PATH}/third_party/Megatron-LM:${PYTHONPATH:-}"
export EXP=/workspace/code/conf/gpt_oss_20B-MXFP4-deosc-mlperf-pretrain.yaml
export DATA_PATH=/data
export MODEL=/model

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
export PRIMUS_MICRO_BATCH_SIZE=4
export PRIMUS_GLOBAL_BATCH_SIZE=32
export EVAL_ITERS=$((1024 / PRIMUS_GLOBAL_BATCH_SIZE))  # MLPerf closed: eval_iters * GBS = 1024 eval samples
export PRIMUS_LR=5.7e-4
export PRIMUS_MIN_LR=5.7e-5 # Set to 10% of max LR
export PRIMUS_TRAIN_ITERS=1200000
export PRIMUS_LR_WARMUP_ITERS=117
export PRIMUS_LR_DECAY_ITERS=$((PRIMUS_TRAIN_ITERS-PRIMUS_LR_WARMUP_ITERS)) # 1200000 - 117 = 1199883
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

export PRIMUS_FUSED_ADAM_CLIP=1
export PRIMUS_FUSED_ADAM_MCORE_MASTER=1
export PRIMUS_FUSED_ADAM_BF16_WRITEBACK=1
export PRIMUS_FUSED_ADAM_BF16_GRAD=1
export PRIMUS_STATIC_BF16_LOSS_SCALE_CPU=1
export PRIMUS_USE_PRECISION_AWARE_OPTIMIZER=true
export PRIMUS_EXP_AVG_DTYPE=bf16
export PRIMUS_EXP_AVG_SQ_DTYPE=bf16
export PRIMUS_WEIGHT_DEOSC_FUSION="${PRIMUS_WEIGHT_DEOSC_FUSION:-true}"
export PRIMUS_WEIGHT_DEOSC_LOG_FREQ="${PRIMUS_WEIGHT_DEOSC_LOG_FREQ:-0}"
export PRIMUS_WEIGHT_DEOSC_DIRECT_QDQ="${PRIMUS_WEIGHT_DEOSC_DIRECT_QDQ:-true}"

# -----------------------------------------------------------------------------
# Primus Configuration
# -----------------------------------------------------------------------------
export PRIMUS_USE_TURBO_ATTENTION=true
export PRIMUS_USE_SINK_ATTENTION=true

export PRIMUS_USE_TURBO_GEMM=true
export PRIMUS_TURBO_GEMM_BACKEND="fp8:flydsl,fp4:flydsl,other:hipblaslt"
export PRIMUS_TURBO_GROUPED_GEMM_BACKEND="fp4:flydsl,other:hipblaslt"
export PRIMUS_TURBO_MXFP4_SCALE_ROUNDING="${PRIMUS_TURBO_MXFP4_SCALE_ROUNDING:-2}"
export PRIMUS_TURBO_AUTO_TUNE=0
export PRIMUS_TURBO_FUSED_WGRAD_ACCUM="${PRIMUS_TURBO_FUSED_WGRAD_ACCUM:-1}"
export PRIMUS_TURBO_WGRAD_ACCUM_OVERWRITE_OUT="${PRIMUS_TURBO_WGRAD_ACCUM_OVERWRITE_OUT:-1}"
export PRIMUS_TURBO_GRAD_OWNERSHIP="${PRIMUS_TURBO_GRAD_OWNERSHIP:-1}"
export PRIMUS_NUM_WORKERS="${PRIMUS_NUM_WORKERS:-2}"
export PRIMUS_GRAD_REDUCE_IN_BF16=true
export USE_TURBO_RMS_NORM=true
export PRIMUS_FUSED_QK_RMSNORM_ROPE=1

# -----------------------------------------------------------------------------
# ROCm / System Runtime
# -----------------------------------------------------------------------------
export GPU_MAX_HW_QUEUES=2
export HIP_FORCE_DEV_KERNARG=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HSA_KERNARG_POOL_SIZE=12582912
export TORCH_NCCL_HIGH_PRIORITY=1
export ENABLE_NUMA_BINDING=1
export PYTORCH_ALLOC_CONF=expandable_segments:False
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
export NCCL_MIN_P2P_NCHANNELS=32
export NCCL_MIN_CTAS=32
export NCCL_NCHANNELS_PER_NET_PEER=32
export NCCL_NVLS_ENABLE=0
export NCCL_CHECKS_DISABLE=1
export RCCL_CE_REDUCESCATTER=1
export RCCL_FORCE_CE_REDUCESCATTER=1
export RCCL_CE_AR_STAGING_BYTES=268435456
export RCCL_CE_COOP_LAUNCH=0
export RCCL_CE_REDUCE_PER_CHUNK=1
export RCCL_CE_REDUCE_MAX_BLOCKS=64
export RCCL_DDA_ENABLE=1
export NCCL_CUMEM_ENABLE=1
export NCCL_LOCAL_REGISTER=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,TUNING
export NCCL_DEBUG_FILE=/results/rccl-debug-%h-%p.log

# -----------------------------------------------------------------------------
# hipBLASLt
# -----------------------------------------------------------------------------
export TORCH_BLAS_PREFER_HIPBLASLT=1

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
export MLLOG_MICRO_BATCH_SIZE=4
MLLOG_CONFIG_FILENAME=$(basename "${BASH_SOURCE[0]}")
export MLLOG_CONFIG_FILENAME
export MLLOG_LOWEST_NUMERICAL_PRECISION_IN_LINEAR='mxfp4'
export MLLOG_LOWEST_NUMERICAL_PRECISION_IN_ATTN='bfloat16'
export MLLOG_LOWEST_NUMERICAL_PRECISION_IN_COMM='bfloat16'

# -----------------------------------------------------------------------------
# Synthetic Warmup (kernel pre-compilation)
# -----------------------------------------------------------------------------
export SYNTH_WARMUP_STEPS=1

# -----------------------------------------------------------------------------
# MoE Token Dispatcher
# -----------------------------------------------------------------------------
export MOE_SKIP_IDENTITY_SORT=1

# -----------------------------------------------------------------------------
# DDP Parameter All-Gather (RCCL SDMA)
# -----------------------------------------------------------------------------
export MEGATRON_PARAM_GATHER_BACKEND="${MEGATRON_PARAM_GATHER_BACKEND:-rccl_sdma}"
export MEGATRON_RCCL_SDMA_EAGER_PARAM_BYTES="${MEGATRON_RCCL_SDMA_EAGER_PARAM_BYTES:-40978350080}"

# -----------------------------------------------------------------------------
# Run-log verbosity
# -----------------------------------------------------------------------------
export MLPERF_VERBOSE_LOGS=${MLPERF_VERBOSE_LOGS:-0}

export PRIMUS_FUSED_RESIDUAL_NORM=1
export PRIMUS_FUSED_RESIDUAL_NORM_V2=1
export PRIMUS_FUSED_RMSNORM_MXFP4_QKV=1
export MLLOG_BLOCK_TPUT_LOG=0
