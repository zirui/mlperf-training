#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

set -euo pipefail

# Use sudo for docker if current user doesn't have direct access. The launcher
# runs srun under the cluster user; on some hosts that user can't talk to the
# Docker daemon directly. Mirrors small_llm_pretraining/primus/run_with_docker_slurm.sh.
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

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"
: "${DATADIR:?DATADIR not set}"
: "${LOGDIR:?LOGDIR not set}"

: "${NEXP:=1}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${CHECK_COMPLIANCE:=1}"
: "${MLPERF_RULESET:=6.1.0}"
: "${CONT_NAME:=mlperf_dlrmv4}"
: "${ROOT_SEED:=${RANDOM}}"
: "${RUNTIME_TUNABLES:=1}"
: "${DLRM_PREFER_LOCAL_DATA:=1}"
: "${DLRM_LOCAL_DATADIR:=/data/mlperf_dlrmv4}"

readonly _config_file="./config_${DGXSYSTEM}.sh"
if [[ ! -f "${_config_file}" ]]; then
    echo "ERROR: config file ${_config_file} not found in ${SCRIPT_DIR}"
    echo "DGXSYSTEM='${DGXSYSTEM}' must match a config_*.sh file basename."
    exit 1
fi
readonly _cont_name="${CONT_NAME}"

echo "Running on host: ${HOSTNAME}"
echo "ROOT SEED IS: ${ROOT_SEED}"

# Stage the NFS dataset onto local disk before the container, outside the timed region, unless this node already has it.
_nfs_data="${DATADIR}"
if [[ "${DLRM_PREFER_LOCAL_DATA}" == 1 && "${_nfs_data}" != "${DLRM_LOCAL_DATADIR}" ]]; then
    if ! grep -qxF "source=${_nfs_data}" "${DLRM_LOCAL_DATADIR}/.stage_complete" 2>/dev/null; then
        echo "[dlrmv4] ${HOSTNAME}: staging ${_nfs_data} -> ${DLRM_LOCAL_DATADIR}"
        DATADIR="${_nfs_data}" DLRM_LOCAL_DATADIR="${DLRM_LOCAL_DATADIR}" \
            bash "${SCRIPT_DIR}/stage_dataset_local.sh"
    fi
    if grep -qxF "source=${_nfs_data}" "${DLRM_LOCAL_DATADIR}/.stage_complete" 2>/dev/null; then
        echo "[dlrmv4] ${HOSTNAME}: using node-local dataset ${DLRM_LOCAL_DATADIR} (staged from ${_nfs_data})"
        export DATADIR="${DLRM_LOCAL_DATADIR}"
    else
        echo "[dlrmv4] ${HOSTNAME}: local stage of ${_nfs_data} did not complete; refusing to train from NFS." >&2
        exit 1
    fi
fi

mkdir -p "${LOGDIR}" "${LOGDIR}/artifacts" && chmod 777 "${LOGDIR}" || true

_cont_mounts=(
    "--volume=${DATADIR}:/data"
    "--volume=${LOGDIR}:/results"
    "--volume=$(pwd):/workspace/code"
)

echo "Stopping all existing containers ..."
${DOCKER} ps -q | xargs -r ${DOCKER} stop

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

mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(DATADIR)
_config_env+=(DGXSYSTEM)
_config_env+=(LOGDIR)

_all_env=()
for v in "${_config_env[@]}"; do
    _all_env+=("--env=$v=${!v-}")
done

cleanup_docker() {
    ${DOCKER} container rm -f "${_cont_name}" || true
}
cleanup_docker
trap 'cleanup_docker' EXIT

${DOCKER} run --rm --init --detach \
    --net=host --uts=host --ipc=host --shm-size=64g \
    --device /dev/dri --device /dev/kfd --device /dev/infiniband \
    --cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN --cap-add=IPC_LOCK \
    --security-opt=seccomp=unconfined \
    --group-add video \
    --privileged \
    --ulimit memlock=-1:-1 \
    --ulimit stack=67108864:67108864 \
    --ulimit nofile=1048576:1048576 \
    --name="${_cont_name}" "${_cont_mounts[@]}" \
    -e IMAGE_NAME="${CONT}" \
    "${CONT}" sleep infinity

sleep 5

if [[ "${RUNTIME_TUNABLES}" == 1 ]] && [[ -f runtime_tunables.sh ]]; then
    echo "Setting runtime tunables"
    bash runtime_tunables.sh
fi

${DOCKER} exec "${_cont_name}" true

# Multi-node RCCL needs the ionic HCAs visible; otherwise it falls back to the management NIC and the job hangs.
: "${AINIC_REQUIRE:=1}"
if [[ "${SLURM_NNODES:-1}" -gt 1 && "${AINIC_REQUIRE}" == "1" ]]; then
    _ainic_check=$(${DOCKER} exec "${_cont_name}" bash -lc 'ibv_devices 2>&1' || true)
    if grep -q "does not support the kernel ABI" <<<"${_ainic_check}"; then
        echo "ERROR: AINIC userspace/kernel ABI mismatch on $(hostname)." >&2
        echo "       Set AINIC_REQUIRE=0 to force-launch on socket fall-back." >&2
        exit 42
    fi
    if ! grep -qE '^[[:space:]]+ionic_[0-9]' <<<"${_ainic_check}"; then
        echo "ERROR: No ionic_* HCAs visible inside the container on $(hostname)." >&2
        echo "       Set AINIC_REQUIRE=0 to force-launch on socket fall-back." >&2
        exit 42
    fi
    echo "[ainic] OK: $(grep -cE '^[[:space:]]+ionic_[0-9]' <<<"${_ainic_check}") ionic_* HCAs visible."
fi

_default_if="$(ip route show default 2>/dev/null | awk '{print $5; exit}' || true)"
# config_MI355X_1x8x1.sh pins NCCL to lo for a standalone container whose
# master is 127.0.0.1. This launcher publishes the node hostname instead, so
# loopback cannot carry the rendezvous.
if [[ -n "${SLURM_JOB_ID:-}" && -n "${_default_if}" ]]; then
    if [[ -z "${NCCL_SOCKET_IFNAME:-}" || "${NCCL_SOCKET_IFNAME}" == "lo" ]]; then
        NCCL_SOCKET_IFNAME="${_default_if}"
    fi
fi
: "${NCCL_SOCKET_IFNAME:=${_default_if}}"
: "${GLOO_SOCKET_IFNAME:=${NCCL_SOCKET_IFNAME}}"
export NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME

export NODE_RANK="${SLURM_NODEID:-0}"

if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
    export MASTER_ADDR
else
    export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
fi
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    # Rank 0 scans for a free port and publishes it; the other nodes must use that same port.
    _port_file="${LOGDIR}/.master_port_${SLURM_JOB_ID}"
    if [[ "${NODE_RANK}" == "0" ]]; then
        MASTER_PORT="$(python3 - <<'PY'
import os, socket
job = int(os.environ["SLURM_JOB_ID"])
start = job % 20000
for offset in range(64):
    port = 20000 + ((start + offset) % 20000)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError:
        sock.close()
        continue
    sock.close()
    print(port)
    break
else:
    raise SystemExit("no free MASTER_PORT")
PY
)"
        printf '%s\n' "${MASTER_PORT}" > "${_port_file}.tmp"
        mv "${_port_file}.tmp" "${_port_file}"
    else
        for _wait in $(seq 1 60); do
            [[ -s "${_port_file}" ]] && break
            sleep 1
        done
        if [[ ! -s "${_port_file}" ]]; then
            echo "ERROR: rank 0 did not publish ${_port_file}" >&2
            exit 1
        fi
        MASTER_PORT="$(cat "${_port_file}")"
    fi
    if [[ -z "${MASTER_PORT}" ]]; then
        echo "ERROR: empty MASTER_PORT" >&2
        exit 1
    fi
    export MASTER_PORT
else
    export MASTER_PORT=${MASTER_PORT:-29500}
fi

echo "=== Node $(hostname) starting ==="
echo "SLURM_NODEID:  ${SLURM_NODEID:-N/A}"
echo "SLURM_NNODES:  ${SLURM_NNODES:-N/A}"
echo "NODE_RANK:     ${NODE_RANK}"
echo "Master:        ${MASTER_ADDR}:${MASTER_PORT}"
echo "NCCL/GLOO IF:  ${NCCL_SOCKET_IFNAME}"
echo "DATADIR:       ${DATADIR}"

for _experiment_index in $(seq 1 "${NEXP}"); do
    echo "Beginning trial ${_experiment_index} of ${NEXP}"
    if [[ "${CLEAR_CACHES}" == 1 ]]; then
        bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3" || true
    fi

    export SEED=$((ROOT_SEED + _experiment_index - 1))
    _logfile="${LOGDIR}/${DATESTAMP}_${HOSTNAME}_${_experiment_index}.log"
    # Config capture runs in a clean shell, so it records NODE_RANK=0 and
    # MASTER_ADDR=127.0.0.1. Docker keeps the last value, so the Slurm values
    # are passed after the config env.
    set +e
    ${DOCKER} exec \
        "${_all_env[@]}" \
        --env=SEED="${SEED}" \
        -e MASTER_ADDR="${MASTER_ADDR}" \
        -e MASTER_PORT="${MASTER_PORT}" \
        -e NODE_RANK="${NODE_RANK}" \
        -e SLURM_NNODES="${SLURM_NNODES:-}" \
        -e SLURM_NODEID="${SLURM_NODEID:-}" \
        -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
        -e GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}" \
        "${_cont_name}" \
        bash /workspace/code/run_and_time.sh \
        | tee "${_logfile}"
    _train_rc=${PIPESTATUS[0]}
    set -e
    if [[ ${_train_rc} -ne 0 ]]; then
        echo "Training failed on $(hostname) with exit code ${_train_rc}" >&2
        exit "${_train_rc}"
    fi

    if [[ "${CHECK_COMPLIANCE}" == 1 && "${NODE_RANK}" == 0 ]]; then
        ${DOCKER} exec "${_cont_name}" \
            python3 -m mlperf_logging.compliance_checker --usage training \
            --ruleset "${MLPERF_RULESET}" \
            --log_output "/results/compliance_${DATESTAMP}.out" \
            "/results/${DATESTAMP}_${HOSTNAME}_${_experiment_index}.log" \
            || echo "WARNING: Compliance check failed for experiment ${_experiment_index} (non-blocking)"
    fi
done
