#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

set -euo pipefail
if [[ "${MLPERF_VERBOSE_LOGS:-0}" == 1 ]]; then
    set -x
fi

# Change directory to the model directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"
: "${DATADIR:?DATADIR not set}"
: "${LOGDIR:?LOGDIR not set}"

# Vars with defaults
: "${NEXP:=1}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${CHECK_COMPLIANCE:=1}"
: "${MLPERF_RULESET:=6.1.0}"
: "${CONT_NAME:=mlperf_dlrmv4}"

# Other vars
readonly _config_file="config_${DGXSYSTEM}.sh"
echo "CONFIG FILE: ${_config_file}"
readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly _cont_name="${CONT_NAME}"
_cont_mounts=("--volume=${DATADIR}:/data" "--volume=$(pwd):/workspace/code" "--volume=${LOGDIR}:/results")

# Setup directories
mkdir -p "${LOGDIR}"
mkdir -p "${LOGDIR}/artifacts/"

# Get list of envvars to pass to docker
mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(DATADIR)
_config_env+=(DGXSYSTEM)
_config_env+=(LOGDIR)
_config_env+=(SEED)

if [[ "${MLPERF_VERBOSE_LOGS:-0}" == 1 ]]; then
    echo "${_config_env[@]}"
fi
mapfile -t _config_env < <(for v in "${_config_env[@]}"; do echo "--env=$v"; done)
_base_config_env=("${_config_env[@]}")

# Cleanup container
cleanup_docker() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${_cont_name}$"; then
        docker container rm -f "${_cont_name}" || true
    else
        echo "Container ${_cont_name} does not exist. Skipping removal."
    fi
}
cleanup_docker
trap cleanup_docker EXIT

# Setup container (AMD/ROCm)
docker run --rm --init --detach \
    --privileged \
    --net=host --uts=host --ipc=host --shm-size=64g \
    --device /dev/dri --device /dev/kfd \
    --cap-add=IPC_LOCK \
    --security-opt=seccomp=unconfined \
    --group-add video \
    --ulimit memlock=-1:-1 \
    --ulimit stack=67108864:67108864 \
    --ulimit nofile=1048576:1048576 \
    --name="${_cont_name}" "${_cont_mounts[@]}" \
    -e IMAGE_NAME="${CONT}" \
    "${CONT}" sleep infinity

# Make sure container has time to finish initialization
sleep 5

bash runtime_tunables.sh
docker exec "${_cont_name}" true

# Run experiments
for _experiment_index in $(seq 1 "${NEXP}"); do
  (
    echo "Beginning trial ${_experiment_index} of ${NEXP}"
    if [[ $CLEAR_CACHES == 1 ]]; then
      bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"
    fi
    # Use existing SEED if set; otherwise use a new RANDOM value
    _run_config_env=("${_base_config_env[@]}" --env=SEED="${SEED:-$RANDOM}")
    docker exec "${_run_config_env[@]}" "${_cont_name}" bash /workspace/code/run_and_time.sh
  ) | grep --line-buffered -v "connected peer ranks" | tee "${_logfile_base}_${_experiment_index}.log"

  if [ "${CHECK_COMPLIANCE}" -eq 1 ]; then
      docker exec "${_cont_name}" \
           python3 -m mlperf_logging.compliance_checker --usage training \
           --ruleset "${MLPERF_RULESET}"                                 \
           --log_output "/results/compliance_${DATESTAMP}.out"           \
           "/results/${DATESTAMP}_${_experiment_index}.log" \
      || echo "WARNING: Compliance check failed for experiment ${_experiment_index} (non-blocking)"
  fi

done
