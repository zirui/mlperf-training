#!/bin/bash
# =============================================================================
# MLPerf LLama2 70B LoRA SFT Configuration for MI355X (1 node, 8 GPUs)
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
export PYTHONPATH="${PRIMUS_PATH}:${PRIMUS_PATH}/third_party/Megatron-Bridge:${PYTHONPATH}"
export EXP=/workspace/code/conf/llama2_70b_lora_posttrain.yaml
export DATA_PATH=/data

# Packed data paths (inside container)
export PACKED_TRAIN_DATA_PATH=/data/train.npy
export PACKED_VAL_DATA_PATH=/data/validation.npy
export PACKED_METADATA_PATH=/data/packed_metadata.jsonl

# Pretrained Megatron checkpoint (mounted via MODEL -> /ckpt, same as nemo convention)
export PRETRAINED_CHECKPOINT=/ckpt/megatron_checkpoints/Llama-2-70b-hf/iter_0000000

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
export LR=0.0004
export SEED=1234

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
export OMP_NUM_THREADS=1

# -----------------------------------------------------------------------------
# NVTE — FP8 & Fused Attention
# -----------------------------------------------------------------------------
export NVTE_FUSED_ATTN=1
export NVTE_FUSED_ATTN_CK=1
export NVTE_FUSED_ATTN_AOTRITON=1
export NVTE_CK_USES_FWD_V3=1
export NVTE_CK_USES_BWD_V3=1
export NVTE_CK_IS_V3_ATOMIC_FP32=0
export NVTE_RS_STRIDED_ATOMIC=2
export NVTE_FP8_DPA_BWD=1
export NVTE_USE_HIPBLASLT=1
export NVTE_USE_CAST_TRANSPOSE_TRITON=1
export NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=0
export NVTE_USE_RMSNORM_TRITON=1
export USE_TE_SWIGLU=1

# -----------------------------------------------------------------------------
# Megatron-Bridge
# -----------------------------------------------------------------------------
export MEGATRON_BRIDGE_LOGGING_LEVEL=50
export PYTHONWARNINGS=ignore

# -----------------------------------------------------------------------------
# MLPerf Logging
# -----------------------------------------------------------------------------
export ENABLE_MLLOG=1
export MLLOG_OUTPUT_FILE=/results/mlperf_output.log
export MLLOG_TRAIN_LOSS_LOG_FREQ=10
export MLLOG_TARGET_EVAL_LOSS=0.925
export MLLOG_SUBMISSION_BENCHMARK=llama2_70b_lora
export MLLOG_SUBMISSION_DIVISION=closed
export MLLOG_SUBMISSION_ORG=AMD
export MLLOG_SUBMISSION_PLATFORM=MI355X
