#!/bin/bash

# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set -euxo pipefail

if docker info >/dev/null 2>&1; then
    DOCKER="docker"
else
    echo "Docker not accessible directly, using sudo docker"
    DOCKER="sudo docker"
fi

if [[ "${DOCKER}" == "sudo docker" ]] \
   && [[ -s "${HOME}/.docker/config.json" ]] \
   && ! sudo -n test -s /root/.docker/config.json 2>/dev/null; then
    echo "[$(hostname)] bootstrapping /root/.docker/config.json from ${HOME}/.docker/config.json"
    sudo -n install -d -m 700 /root/.docker 2>/dev/null || true
    cat "${HOME}/.docker/config.json" | sudo -n tee /root/.docker/config.json >/dev/null 2>&1 || true
    sudo -n chmod 600 /root/.docker/config.json 2>/dev/null || true
fi

# ======= Change to model dir  =======
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

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
: "${ROOT_SEED:=${RANDOM}}"
: "${NGPU:=1}"
: "${RUNTIME_TUNABLES:=0}"
: "${HF_TOKEN:=""}"

# ======= Experiment variables with defaults =======
## Profiler
: "${PROFILER:=""}"
: "${TORCHPROF_OUTPUT_DIR:=""}"
: "${PROF_WARMUP_STEPS:=5}"
: "${PROF_ACTIVE_STEPS:=2}"
## hipblaslt
: "${HIPBLASLT_LOG:=0}"
: "${GEMM_OFFLINE_TUNING:=0}"
: "${GEMM_USE_TUNING_RESULTS:=0}"
## memory profiling
: "${ENABLE_MEMORY_PROFILING:=0}"
## training metrics
: "${ENABLE_TRAINING_METRICS_COLLECTION:=0}"

echo "ROOT SEED IS: $ROOT_SEED"

# ======== Get Environment Variables from config ========
readonly _config_file="./config_${DGXSYSTEM}.sh"
if [[ ! -f "${_config_file}" ]]; then
    echo "ERROR: config file ${_config_file} not found in ${SCRIPT_DIR}"
    echo "DGXSYSTEM='${DGXSYSTEM}' must match a config_*.sh file basename."
    exit 1
fi
readonly _cont_name="${CONT_NAME}"

echo "Running on host: $HOSTNAME"
mkdir -p "${LOGDIR}" && chmod 777 "${LOGDIR}"

_cont_mounts=(
    "--volume=${DATADIR}:/data"
    "--volume=${LOGDIR}:/results"
    "--volume=$(pwd):/workspace/code"
    "--volume=$(pwd)/../../utilities:/workspace/utilities"
)
if [[ -n "${MODELDIR:-}" && -d "${MODELDIR}" ]]; then
    _cont_mounts+=("--volume=${MODELDIR}:/model")
fi

echo "Stopping all existing containers ..."
${DOCKER} ps -q | xargs -r ${DOCKER} stop

# ======= Pre-pull image on this node, with retries for flaky registries =======
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

# Capture env vars defined by the system config (forward to docker exec).
mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(DGXSYSTEM)
_config_env+=(HF_TOKEN)

# ========= Get sbatch environment variables (optional) =========
readonly _sbatch_file="./sbatch.sh"
_sbatch_env=()
if [[ -f "${_sbatch_file}" ]]; then
    _sbatch_exports="$(mktemp)"
    grep '^export ' "${_sbatch_file}" > "${_sbatch_exports}" || true
    mapfile -t _sbatch_env < <(
        env -i bash -c ". ${_sbatch_exports} && compgen -e" 2>/dev/null | grep -E -v '^(PWD|SHLVL)'
    )
    rm -f "${_sbatch_exports}"
fi

_all_env=()
for v in "${_config_env[@]}" "${_sbatch_env[@]}"; do
    _all_env+=("--env=$v=${!v-}")
done

# =============== Cleanup function for Docker container ================
cleanup_docker() {
    ${DOCKER} container rm -f "${_cont_name}" || true
}
cleanup_docker
trap 'set -eux; cleanup_docker' EXIT

bash "${SCRIPT_DIR}/scripts/check_gpus_idle.sh"

${DOCKER} run --rm --init --detach \
    --net=host --uts=host --ipc=host \
    --device /dev/dri --device /dev/kfd --device /dev/infiniband \
    --cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN --cap-add=IPC_LOCK \
    --security-opt=seccomp=unconfined \
    --group-add video \
    --privileged \
    --ulimit memlock=-1:-1 \
    --ulimit nofile=1048576:1048576 \
    --name="${_cont_name}" "${_cont_mounts[@]}" \
    -e IMAGE_NAME="${CONT}" \
    "${CONT}" sleep infinity

# Make sure container has time to finish initialization
sleep 5

if [[ "${RUNTIME_TUNABLES}" == 1 ]] && [[ -f runtime_tunables.sh ]]; then
    echo "Setting runtime tunables"
    bash runtime_tunables.sh
fi

${DOCKER} exec "${_cont_name}" true

# =============== AINIC ABI mismatch fail-fast guard (multi-node only) ===========
: "${AINIC_REQUIRE:=1}"
if [[ "${SLURM_NNODES:-1}" -gt 1 && "${AINIC_REQUIRE}" == "1" ]]; then
    _ainic_check=$(${DOCKER} exec "${_cont_name}" bash -lc 'ibv_devices 2>&1' || true)
    if grep -q "does not support the kernel ABI" <<<"${_ainic_check}"; then
        echo "ERROR: AINIC userspace/kernel ABI mismatch detected on $(hostname)." >&2
        echo "       libibverbs cannot open any ionic_* HCA; RCCL would fall" >&2
        echo "       back to TCP. Rebuild the dev image with a libionic whose" >&2
        echo "       supported ABI matches the host ionic kernel module." >&2
        echo "       Set AINIC_REQUIRE=0 to force-launch on socket fall-back." >&2
        echo "       ibv_devices output was:" >&2
        sed 's/^/         /' >&2 <<<"${_ainic_check}"
        exit 42
    fi
    if ! grep -qE '^[[:space:]]+ionic_[0-9]' <<<"${_ainic_check}"; then
        echo "ERROR: No ionic_* HCAs visible inside the container on $(hostname)." >&2
        echo "       Either the AINIC bundle did not install correctly, or" >&2
        echo "       /dev/infiniband was not passed through to the container." >&2
        echo "       Set AINIC_REQUIRE=0 to force-launch on socket fall-back." >&2
        echo "       ibv_devices output was:" >&2
        sed 's/^/         /' >&2 <<<"${_ainic_check}"
        exit 42
    fi
    echo "[ainic] OK: $(grep -cE '^[[:space:]]+ionic_[0-9]' <<<"${_ainic_check}") ionic_* HCAs visible to libibverbs."
fi

# =============== Set up environment variables for NCCL/GLOO and rendezvous ======
: "${NCCL_SOCKET_IFNAME:=$(route | awk '/^default/ {print $NF; exit}')}"
: "${GLOO_SOCKET_IFNAME:=$(route | awk '/^default/ {print $NF; exit}')}"
export NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME

if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
else
    export MASTER_ADDR=${MASTER_ADDR:-localhost}
fi
export MASTER_PORT=${MASTER_PORT:-29501}

echo "=== Node $(hostname) starting ==="
echo "SLURM_PROCID:  ${SLURM_PROCID:-N/A}"
echo "SLURM_NODEID:  ${SLURM_NODEID:-N/A}"
echo "SLURM_NNODES:  ${SLURM_NNODES:-N/A}"
echo "Master:        ${MASTER_ADDR}:${MASTER_PORT}"
echo "NCCL/GLOO IF:  ${NCCL_SOCKET_IFNAME}"

# =============== Run experiments  ================
for _experiment_index in $(seq 0 $((NEXP - 1))); do
{
    echo "Beginning trial ${_experiment_index} of ${NEXP}"
    if [[ "${CLEAR_CACHES}" == 1 ]]; then
        bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"
        ${DOCKER} exec "${_cont_name}" python -c "
from mlperf_logging import mllog
mllogger = mllog.get_mllogger()
mllogger.event(key=mllog.constants.CACHE_CLEAR, value=True)"
    fi

    export SEED=$(( ROOT_SEED + 10#${SLURM_ARRAY_TASK_ID:-0} * 100 + 10#${_experiment_index} ))
    _run_env=("${_all_env[@]}" --env=SEED="${SEED}")

    echo "launching experiment using: ${_run_env[*]} ${_cont_name} bash /workspace/code/run_and_time.sh"
    ${DOCKER} exec \
        "${_run_env[@]}" \
        -e MASTER_ADDR="${MASTER_ADDR}" \
        -e MASTER_PORT="${MASTER_PORT}" \
        -e SLURM_NNODES="${SLURM_NNODES:-}" \
        -e SLURM_NODEID="${SLURM_NODEID:-}" \
        -e SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST:-}" \
        -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
        -e GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}" \
        "${_cont_name}" \
        bash /workspace/code/run_and_time.sh
}
done
