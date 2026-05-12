#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

set -euxo pipefail

# Use sudo for docker if current user doesn't have direct access. The launcher
# runs srun under the cluster user; on some hosts that user can't talk to the
# Docker daemon directly. Mirrors flux1/nemo/run_with_docker_slurm.sh.
if docker info >/dev/null 2>&1; then
    DOCKER="docker"
else
    echo "Docker not accessible directly, using sudo docker"
    DOCKER="sudo docker"
fi

# When falling back to sudo docker, root often has no credentials for private
# registry images (rocm/mlperf:* etc). The user's ~/.docker/config.json is on
# NFS so it's visible from every compute node, but mode 600 + NFS root_squash
# means root can't read it directly. Bootstrap a copy into /root/.docker once
# per node, piped via the user's read perms, so subsequent pulls authenticate.
# Idempotent: skips if root already has creds. Silent on failure (best-effort).
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
: "${CONT_NAME:=mlperf_small_llm_pretraining}"
: "${LOG_FREQ:=0}"
: "${ROOT_SEED:=${RANDOM}}"
: "${NGPU:=1}"
: "${RUNTIME_TUNABLES:=0}"
: "${HF_TOKEN:=""}"

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
    mapfile -t _sbatch_env < <(
        grep '^export ' "${_sbatch_file}" > /tmp/sbatch_exports.sh || true
        env -i bash -c ". /tmp/sbatch_exports.sh && compgen -e" 2>/dev/null | grep -E -v '^(PWD|SHLVL)'
        rm -f /tmp/sbatch_exports.sh
    )
fi

# Combine and convert to Docker --env flags with explicit values.
# sudo strips the environment, so --env=NAME alone won't work — we have to
# materialize the value here. SEED is added per-trial in the experiment loop.
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

${DOCKER} run --rm --init --detach \
    --net=host --uts=host --ipc=host \
    --device /dev/dri --device /dev/kfd --device /dev/infiniband \
    --cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN --cap-add=IPC_LOCK \
    --security-opt=seccomp=unconfined \
    --group-add video \
    --privileged \
    --ulimit memlock=-1:-1 \
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
# Prevent the silent failure mode where libibverbs cannot open any ionic_* HCA
# due to a kernel-vs-userspace ABI mismatch. RCCL then falls back to TCP on
# the management NIC (fenic) and "trains" at <0.1 step/min, getting cancelled
# by SLURM with no obvious cause.
# Set AINIC_REQUIRE=0 in the env to skip this check
: "${AINIC_REQUIRE:=1}"
if [[ "${SLURM_NNODES:-1}" -gt 1 && "${AINIC_REQUIRE}" == "1" ]]; then
    _ainic_check=$(${DOCKER} exec "${_cont_name}" bash -lc 'ibv_devices 2>&1' || true)
    if grep -q "does not support the kernel ABI" <<<"${_ainic_check}"; then
        echo "ERROR: AINIC userspace/kernel ABI mismatch detected on $(hostname)." >&2
        echo "       libibverbs cannot open any ionic_* HCA, RCCL will silently" >&2
        echo "       fall back to TCP on the management NIC. Rebuild the dev" >&2
        echo "       image with a libionic whose supported ABI matches the host ionic kernel module " >&2
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
# NCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME pick the default-route interface so
# inter-node bootstrap works on hosts where NIC names vary. Honor any value
# already set by config_*.sh (sourced by sbatch.sh.j2 before this launcher runs)
# so the analysis-doc §7.2 socket-only stop-gap can pin RCCL to the high-speed
# benic*p1 ports without editing this script.
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
    fi

    export SEED=$((ROOT_SEED + _experiment_index))
    _run_env=("${_all_env[@]}" --env=SEED="${SEED}")

    echo "launching experiment using: ${_run_env[*]} ${_cont_name} bash /workspace/code/run_and_time.sh"
    # NOTE: -e flags MUST come AFTER ${_run_env[@]}. _all_env is built from
    # `compgen -e` after sourcing the system config, which exports
    # MASTER_ADDR=localhost (a single-node default that this script overrides
    # via scontrol on multi-node SLURM). Docker uses last-wins semantics for
    # repeated env vars, so listing the explicit -e overrides last guarantees
    # the SLURM-derived MASTER_ADDR / MASTER_PORT win and inter-node c10d
    # rendezvous can complete (otherwise rank 1 connects to localhost:29501
    # and times out).
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


