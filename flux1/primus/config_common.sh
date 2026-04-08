export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[1]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export TQDM_DISABLE=True

export MODEL=${MODEL:-schnell}
export DATA=${DATA:-cc12m}
export TARGET_ACCURACY=${TARGET_ACCURACY:-0.586}
export EXP_NAME=${EXP_NAME:-flux-primus-$(date +%y%m%d%H%M%S%N)}

# Validation interval in samples
export VAL_CHECK_INTERVAL=${VAL_CHECK_INTERVAL:-262144}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export GRAD_REDUCE_IN_FP32=${GRAD_REDUCE_IN_FP32:-False}

export WARMUP_ENABLED=${WARMUP_ENABLED:-True}
export WARMUP_TRAIN_STEPS=${WARMUP_TRAIN_STEPS:-2}
export WARMUP_VALIDATION_STEPS=${WARMUP_VALIDATION_STEPS:-2}

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1

# NCCL/RCCL configuration for AMD GPUs
export NCCL_NVLS_ENABLE=0
export NCCL_GRAPH_REGISTER=0
export NCCL_LOCAL_REGISTER=0

# NCCL tuning for multinode performance
export NCCL_MIN_P2P_NCHANNELS=32
export NCCL_MIN_CTAS=32
export NCCL_NCHANNELS_PER_NET_PEER=32

# InfiniBand/RDMA configuration for multinode
export NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7

export MODEL_TFLOP_PER_SAMPLE=20.3
