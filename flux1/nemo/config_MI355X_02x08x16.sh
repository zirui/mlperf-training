# Load default settings
source $(dirname ${BASH_SOURCE[0]})/config_common.sh

export MAX_STEPS=${MAX_STEPS:--1}

# MBS and GBS-dependent hyperparams (BATCHSIZE, LEARNING_RATE, WARMUP_STEPS)
export MBS=${MBS:-32}
export BATCHSIZE=${BATCHSIZE:-512}
source $(dirname ${BASH_SOURCE[0]})/config_gbs${BATCHSIZE}.sh

export FP8_RECIPE="delayed"  # mxfp8 currently not working on AMD GPU
# System parameters
export DGXNNODES=2
export DGXNGPU=8
export SEGMENT=${DGXNNODES:-2}

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

export NVTE_CK_USES_BWD_V3=1
export NVTE_CK_USES_FWD_V3=1
export NVTE_CK_IS_V3_ATOMIC_FP32=0

export CK_FUSED_ATTN_LOG_CONFIG=0
export NVTE_LOG_CK_CONFIG=0
export NVTE_LOG_FUSED_ATTN_CONFIG=0
export CHECK_FOR_NAN_IN_GRAD=0

export FUSED_SOFTMAX=0              # not in TE
export RMSNORM_CAST=0               # not in TE
export ENABLE_TRANSPOSE_CACHE=0     # check

export USE_HIPBLASLT=1
export TORCH_BLAS_PREFER_HIPBLASLT=1

# RCCL-specific optimizations for multinode (NCCL settings are in config_common.sh)
export RCCL_ENABLE_MULTI_STREAM=1
export RCCL_STREAM_PRIORITY=1
#export NCCL_DEBUG=INFO

# Walltime
export WALLTIME_RUNANDTIME=200
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

export CUDA_GRAPH_IMPL=local
export CPU_OFFLOADING=False
export CPU_OFFLOADING_ACTIVATIONS=False
