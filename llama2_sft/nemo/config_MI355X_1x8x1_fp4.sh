#!/bin/bash
export WARMUP=True
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export DGXNNODES=1
export WALLTIME_MINUTES=50
export WALLTIME=$(( (${NEXP:-1} * WALLTIME_MINUTES) ))

export NCCL_MIN_P2P_NCHANNELS=32;
export NCCL_MIN_CTAS=32;
export NCCL_NCHANNELS_PER_NET_PEER=32;
export NCCL_NVLS_ENABLE=0

export TP_COMM_OVERLAP=False 
export MC_TP_OVERLAP_AG=False
export MC_TP_OVERLAP_RS=False
export MC_TP_OVERLAP_RS_DGRAD=False

export CUBLAS_FORCE_XMMA_KERNEL_INIT=DEVICE

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

export LORA_A2A=1
export POSSIBLE_USER_WARNINGS=0
export CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=0

export MAX_STEPS=1
export TP=1
export PP=1
export CP=1
export SP=False
export VBOOST_VALUE=1
export MBS=1
export LR=0.0006
export MINIBS=1
export SKIP_EVALS=3
export VAL_CHECK_INTERVAL=384
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

export FP8_DPA=0    
export FP8=True
export FP8_AMAX_ALGO=most_recent
export FP8_REDUCE_AMAX=False
export FP8_AMAX_HISTORY=4
export FP8_ACTIVATION=True

export FUSED_SOFTMAX=0
export RMSNORM_CAST=0

export PT_TENSOR_VALIDATION=0
export PROFILE_RPD=0

export USE_HIPBLASLT=1
export TORCH_BLAS_PREFER_HIPBLASLT=1

export MLPERF_SUBMISSION_ORG="AMD"
export MLPERF_SUBMISSION_PLATFORM="MI355X"

export NVTE_USE_RMSNORM_TRITON=1
export ENABLE_TRANSPOSE_CACHE=1
export NVIDIA_PYTORCH_VERSION=25.04 
export OMP_NUM_THREADS=1

export LOGGING_INTERVAL=5000
export FP4=True
export FP4_RECIPE=mxfp4
export NVTE_MXFP4_USE_HADAMARD=1
export MAX_STEPS=550
export NEXP=1

export LOAD_CKPT=True
export HEALING_PRECISION=FP8_DS

# Healing: switch from MXFP4 to FP8 at this step (0 = no healing, pure MXFP4)
export HEALING_ITER=340
export PRE_QUANTIZED_MODEL=True

# Activation recompute during MXFP4 phase (disabled at healing for FP8)
export RECOMPUTE_GRANULARITY=full
export RECOMPUTE_METHOD=block
export RECOMPUTE_NUM_LAYERS=8

export FP8_ACT=0

export AITER_CONFIG_GEMM_A4W4=/workspace/code/a4w4_tuned_gemms.csv
export AITER_LOG_TUNED_CONFIG=0

# FP4 GEMM tuning: log kernel selection (set 1 for debug, 0 for perf runs)
export NVTE_FP4_LOG_GEMM_SHAPES=0

export AITER_LOG_LEVEL=ERROR
export AITER_LOG_MORE=0

# MLPerf run-log verbosity. Default 0 keeps only :::MLLOG + ``run_and_time.sh``
# banners on stdout. Set to 1 to restore the full framework output (NeMo /
# Lightning / TE / aiter / Gloo / hipify / ... banners) when debugging.
export MLPERF_VERBOSE_LOGS=${MLPERF_VERBOSE_LOGS:-0}
