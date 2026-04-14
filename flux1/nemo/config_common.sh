export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[1]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export HYDRA_FULL_ERROR=1
export TQDM_DISABLE=True
export DISABLE_PRINT=${DISABLE_PRINT:-True} # filter out nemo prints. Set to False to debug.

export MODEL=${MODEL:-schnell}
export DATA=${DATA:-cc12m}
export TARGET_ACCURACY=${TARGET_ACCURACY:-0.586}
export EXP_NAME=${EXP_NAME:-flux-train-$(date +%y%m%d%H%M%S%N)}
# Validation interval in samples
export VAL_CHECK_INTERVAL=${VAL_CHECK_INTERVAL:-262144}

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export GRAD_REDUCE_IN_FP32=${GRAD_REDUCE_IN_FP32:-False}

export WARMUP_ENABLED=${WARMUP_ENABLED:-True}
export WARMUP_TRAIN_STEPS=${WARMUP_TRAIN_STEPS:-2}
export WARMUP_VALIDATION_STEPS=${WARMUP_VALIDATION_STEPS:-2}

export HF_HUB_OFFLINE=1 # disable network requests for HF transfers

# NCCL/RCCL configuration for AMD GPUs
export NCCL_NVLS_ENABLE=0  # NVLS is NVIDIA-only, disable for AMD
export NCCL_GRAPH_REGISTER=0
export NCCL_LOCAL_REGISTER=0

# NCCL tuning for multinode performance
export NCCL_MIN_P2P_NCHANNELS=32
export NCCL_MIN_CTAS=32
export NCCL_NCHANNELS_PER_NET_PEER=32

# InfiniBand/RDMA configuration for multinode
# Diagnostic checks run automatically in run_with_docker_slurm.sh - check logs for:
#   - rdma link show output (lists HCAs if available)
#   - ibstat output (shows IB device status)
#   - /dev/infiniband/ directory check
#
# If InfiniBand HCAs are found in diagnostics (e.g., mlx5_0, bnxt_re0):
#   1. Set NCCL_IB_DISABLE=0 to enable InfiniBand
#   2. Set NCCL_IB_HCA to comma-separated list of HCA names
#   3. Uncomment NCCL_IB_GID_INDEX
#
# Example if mlx5_0, mlx5_1 are found:
#   export NCCL_IB_DISABLE=0
#   export NCCL_IB_HCA=mlx5_0,mlx5_1
#   export NCCL_IB_GID_INDEX=0
#
# Enable InfiniBand/RDMA
# export NCCL_IB_DISABLE=0  # Enable InfiniBand support
export NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7  # Ionic RDMA HCAs
# export NCCL_IB_GID_INDEX=0  # Use first GID index
# export NCCL_IB_TIMEOUT=22  # IB timeout in seconds
# export NCCL_IB_RETRY_CNT=7  # Number of retries for IB operations
# export NCCL_IB_SL=0  # Service level for IB
# export NCCL_IB_TC=41  # Traffic class for IB (default)

# For Ethernet-based multinode (when InfiniBand is not available):
# Ensure backend NICs are used instead of front-end NIC (enp81s0f1)
# Check available interfaces with: ip link show (in container or host)
# Then override NCCL_SOCKET_IFNAME in config file or run_with_docker_slurm.sh
# Example: export NCCL_SOCKET_IFNAME=enp81s0f0  # Use backend NIC instead of front-end

export MODEL_TFLOP_PER_SAMPLE=20.3
