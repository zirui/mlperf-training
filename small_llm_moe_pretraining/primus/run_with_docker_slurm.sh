set -euxo pipefail

# ======= Change to model dir  =======
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd $SCRIPT_DIR

# ======= Vars without defaults =======
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"
: "${DATADIR:?DATADIR not set}"
: "${LOGDIR:?LOGDIR not set}"

# ======= Vars with defaults ========
: "${NEXP:=1}"
: "${CLEAR_CACHES:=1}"
: "${CONT_NAME:=mlperf_gpt_oss_20b}"
: "${LOG_FREQ:=0}"
: "${ROOT_SEED:=${ROOT_SEED}}"
: "${NGPU:=1}"

# ======= Experiment variables with defaults =======
## Profiler
: "${PROFILER:=""}"
: "${TORCHPROF_OUTPUT_DIR:=""}"

: "${PROF_WARMUP_STEPS:=5}"
: "${PROF_ACTIVE_STEPS:=2}"
## rocprofv3
: "${ROCPROF_PROFILER:='none'}"
: "${ROCPROF:='/opt/rocm/bin/rocprofv3'}"
## hipblaslt
: "${HIPBLASLT_LOG:=0}"
: "${GEMM_OFFLINE_TUNING:=0}"
: "${GEMM_USE_TUNING_RESULTS:=0}"
## optuna
: "${RUN_OPTUNA:=0}"
: "${OPTUNA_CONFIG:=''}"
## memory profiling
: "${ENABLE_MEMORY_PROFILING:=0}"
## power profiling
: "${POWER_PROFILING:=0}"

: "${ENABLE_TRAINING_METRICS_COLLECTION:=0}"

echo "ROOT SEED IS: $ROOT_SEED"

# ======== Get Environment Variables from config ========
readonly _config_file="./config_${DGXSYSTEM}.sh"
readonly _cont_name="${CONT_NAME}"

echo "Running on host: $HOSTNAME"
mkdir -p "${LOGDIR}" && chmod 777 "${LOGDIR}"
_cont_mounts=(
    "--volume=${DATADIR}:/data"
    "--volume=${LOGDIR}:/results"
    "--volume=$(pwd):/workspace/code"
    "--volume=$(pwd)/../../utilities:/workspace/utilities"
)

echo "Stopping all existing containers ..."
docker ps -q | xargs -r docker stop

# Get list of envvars to pass to docker
mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(DGXSYSTEM)

# ========= Get sbatch environment variables =========
# Extract only the export statements from sbatch.sh and source them
readonly _sbatch_file="./sbatch.sh"
mapfile -t _sbatch_env < <(
    # Create a temp script with only export statements
    grep '^export ' "${_sbatch_file}" > /tmp/sbatch_exports.sh
    env -i bash -c ". /tmp/sbatch_exports.sh && compgen -e" | grep -E -v '^(PWD|SHLVL)'
    rm -f /tmp/sbatch_exports.sh
)

# Combine and convert to Docker format
mapfile -t _all_env < <(
    printf '%s\n' "${_config_env[@]}" "${_sbatch_env[@]}" |
    while read -r v; do echo "--env=$v"; done
)

# =============== Cleanup function for Docker container ================
cleanup_docker() {
    docker container rm -f "${_cont_name}" || true
}
cleanup_docker
trap 'set -eux; cleanup_docker' EXIT

docker run --rm --init --detach \
      --net=host --uts=host \
      --ipc=host \
      --device /dev/dri \
      --device /dev/kfd \
      --device /dev/infiniband \
      --group-add video \
      --cap-add SYS_PTRACE \
      --cap-add IPC_LOCK \
      --privileged \
      --ulimit memlock=-1:-1 \
      --security-opt seccomp=unconfined \
      --name="${_cont_name}" "${_cont_mounts[@]}" \
      -e IMAGE_NAME="${CONT}" \
      "${CONT}" sleep infinity

# Make sure container has time to finish initialization
sleep 5

if [[ $RUNTIME_TUNABLES == 1 ]]; then
  echo "Setting runtime tunables"
  bash runtime_tunables.sh
fi

docker exec "${_cont_name}" true

# =============== Check for InfiniBand/RDMA devices in container ============
echo "=== Checking for InfiniBand/RDMA devices in container ==="
docker exec "${_cont_name}" bash -c "rdma link show 2>&1 || echo 'rdma link show: No InfiniBand devices found'" || true
docker exec "${_cont_name}" bash -c "ibstat 2>&1 || echo 'ibstat: Command not found or no InfiniBand devices'" || true
docker exec "${_cont_name}" bash -c "ls -la /dev/infiniband/ 2>&1 || echo 'No /dev/infiniband directory found'" || true
docker exec "${_cont_name}" bash -c "ip link show | grep -i ib || echo 'No InfiniBand interfaces found'" || true

# =============== Set up environment variables for NCCL and GLOO ============
export NCCL_SOCKET_IFNAME=$(route | grep '^default' | grep -o '[^ ]*$')
export GLOO_SOCKET_IFNAME=$(route | grep '^default' | grep -o '[^ ]*$')
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29501

echo "=== Node $(hostname) starting ==="
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# =============== Run experiments  ================
for _experiment_index in $(seq 0 $((NEXP-1))); do
{
  echo "Beginning trial ${_experiment_index} of ${NEXP}"
  if [[ $CLEAR_CACHES == 1 ]]; then
    bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"
  fi

  export SEED=$(($ROOT_SEED - 1 + 10#$_experiment_index))
  _all_env+=(--env=SEED=$SEED)
  echo 'launching experiment using:' ${_all_env[@]} ${_cont_name} ./run_and_time.sh
  docker exec \
    -e MASTER_ADDR=$MASTER_ADDR \
    -e MASTER_PORT=$MASTER_PORT \
    -e SLURM_NNODES=$SLURM_NNODES \
    -e SLURM_NODEID=$SLURM_NODEID \
    -e NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME \
    -e GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME \
    ${_all_env[@]} \
    ${_cont_name} \
    bash ./run_and_time.sh
}
done