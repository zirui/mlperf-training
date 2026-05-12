set -euxo pipefail

# Use sudo for docker if current user doesn't have direct access
if docker info >/dev/null 2>&1; then
    DOCKER="docker"
else
    echo "Docker not accessible directly, using sudo docker"
    DOCKER="sudo docker"
fi

# ======= Change to model dir  =======
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd $SCRIPT_DIR

# ======= Vars without defaults =======
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"
: "${DATADIR:?DATADIR not set}"
: "${LOGDIR:?LOGDIR not set}"
: "${ROOT_SEED:?ROOT_SEED not set}"

# ======= Vars with defaults ========
: "${NEXP:=10}"
: "${CLEAR_CACHES:=1}"
: "${CONT_NAME:=mlperf_flux1}"
: "${LOG_FREQ:=0}"
: "${NGPU:=1}"
: "${RUNTIME_TUNABLES:=1}"

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

# ======== Get Environment Variables from config ========
readonly _config_file="./config_${DGXSYSTEM}.sh"
readonly _cont_name="${CONT_NAME}"

echo "Running on host: $HOSTNAME"
mkdir -p "${LOGDIR}" && chmod 777 "${LOGDIR}"
_cont_mounts=(
    "--volume=${DATADIR}:/dataset"
    "--volume=${LOGDIR}:/results"
    "--volume=$(pwd):/workspace/code"
    "--volume=$(pwd)/../../utilities:/workspace/utilities"
)

echo "Stopping all existing containers ..."
${DOCKER} ps -q | xargs -r ${DOCKER} stop

# ======= Pre-pull image on this node =======
echo "[$(hostname)] pulling ${CONT}"
for _attempt in 1 2 3; do
    if ${DOCKER} pull "${CONT}"; then
        break
    fi
    if [[ ${_attempt} -eq 3 ]]; then
        echo "[$(hostname)] ERROR: failed to pull ${CONT} after 3 attempts" >&2
        exit 1
    fi
    echo "[$(hostname)] pull attempt ${_attempt} failed, retrying in 10s..." >&2
    sleep 10
done

# Get list of envvars to pass to docker
mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(DGXSYSTEM)

# Optional: extra exports from ./sbatch.sh if available
_sbatch_env=()
readonly _sbatch_file="./sbatch.sh"
if [[ -f "${_sbatch_file}" ]]; then
    _sbatch_exports="$(mktemp)"
    grep '^export ' "${_sbatch_file}" > "${_sbatch_exports}" || true
    mapfile -t _sbatch_env < <(
        env -i bash -c ". ${_sbatch_exports} && compgen -e" | grep -E -v '^(PWD|SHLVL)'
    )
    rm -f "${_sbatch_exports}"
fi

_all_env=()
for v in "${_config_env[@]}" "${_sbatch_env[@]}"; do
    _all_env+=("--env=$v=${!v}")
done

# =============== Cleanup function for Docker container ================
cleanup_docker() {
    ${DOCKER} container rm -f "${_cont_name}" || true
}
cleanup_docker
trap 'set -eux; cleanup_docker' EXIT

# ensure gpus are idle
bash "${SCRIPT_DIR}/scripts/check_gpus_idle.sh"

${DOCKER} run --rm --init --detach \
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
      --ulimit nofile=1048576:1048576 \
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

${DOCKER} exec "${_cont_name}" true

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
for _experiment_index in $(seq 1 $NEXP); do
{
  echo "Beginning trial ${_experiment_index} of ${NEXP}"
  if [[ $CLEAR_CACHES == 1 ]]; then
    bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"
    ${DOCKER} exec ${_cont_name} python -c "
from mlperf_logging import mllog
mllogger = mllog.get_mllogger()
mllogger.event(key=mllog.constants.CACHE_CLEAR, value=True)"
  fi

  export SEED=$(( ROOT_SEED + 10#${SLURM_ARRAY_TASK_ID:-0} * 100 + 10#${_experiment_index} ))
  echo 'launching experiment using:' ${_all_env[@]} --env=SEED=$SEED ${_cont_name} ./run_and_time.sh
  ${DOCKER} exec \
    -e MASTER_ADDR=$MASTER_ADDR \
    -e MASTER_PORT=$MASTER_PORT \
    -e SLURM_NNODES=$SLURM_NNODES \
    -e SLURM_NODEID=$SLURM_NODEID \
    -e NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME \
    -e GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME \
    ${_all_env[@]} \
    --env=SEED=$SEED \
    ${_cont_name} \
    bash ./run_and_time.sh
}
done
