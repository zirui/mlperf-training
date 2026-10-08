#!/bin/bash

set -euxo pipefail

# Change directory to the primus directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"
: "${DATADIR:?DATADIR not set}"
: "${MODELDIR:?MODELDIR not set}"
: "${LOGDIR:?LOGDIR not set}"

# Vars with defaults
: "${NEXP:=1}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${CHECK_COMPLIANCE:=1}"
: "${MLPERF_RULESET:=6.1.0}"

: "${CONT_NAME:=mlperf_small_llm_pretraining}"
: "${NGPU:=1}"
: "${LOG_FREQ:=0}"
: "${HF_TOKEN:=""}"

# Other vars
readonly _config_file="config_${DGXSYSTEM}.sh"
echo "CONFIG FILE: ${_config_file}"

_apply_config_defaults() {
    local kv key
    while IFS= read -r -d '' kv; do
        key="${kv%%=*}"
        case "${key}" in PWD | OLDPWD | SHLVL | _) continue ;; esac
        if [[ -z "${!key:-}" ]]; then
            export "${key}=${kv#*=}"
        fi
    done < <(env -i bash -c ". ${_config_file} && env -0")
}
set +x
_apply_config_defaults
set -x
: "${EXP:?EXP is empty after loading ${_config_file}}"
echo "EXP: ${EXP}"

readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly _cont_name="${CONT_NAME}"
_cont_mounts=("--volume=${DATADIR}:/data" "--volume=${MODELDIR}:/model" "--volume=$(pwd):/workspace/code" "--volume=${LOGDIR}:/results")


# Setup directories
mkdir -p "${LOGDIR}"
mkdir -p "${LOGDIR}/artifacts/"

# Get list of envvars to pass to docker
mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(DATADIR)
_config_env+=(MODELDIR)
_config_env+=(MODEL)
_config_env+=(DGXSYSTEM)
_config_env+=(PROFILER)
_config_env+=(LOGDIR)
_config_env+=(HIPBLASLT_LOG)
_config_env+=(GEMM_OFFLINE_TUNING)
_config_env+=(GEMM_USE_TUNING_RESULTS)
_config_env+=(HF_TOKEN)
_config_env+=(SEED)
_config_env+=(HIPBLASLT_LOG_MASK)
_config_env+=(HIPBLASLT_LOG_FILE)
_config_env+=(ENABLE_PRIMUS_TURBO)
_config_env+=(PRIMUS_USE_TURBO_ATTENTION)
_config_env+=(PRIMUS_USE_TURBO_GEMM)
_config_env+=(PRIMUS_ATTENTION_BACKEND)
_config_env+=(PRIMUS_TURBO_ATTN_BACKEND)
_config_env+=(PRIMUS_TURBO_AUTO_TUNE)
_config_env+=(PRIMUS_TURBO_GEMM_BACKEND)
_config_env+=(PRIMUS_TURBO_FUSED_DENSE_MLP)
_config_env+=(PRIMUS_TURBO_DENSE_MLP_FUSE_QUANT)
_config_env+=(PRIMUS_TURBO_DENSE_MLP_FUSE_DGLU)
_config_env+=(PRIMUS_USE_TURBO_RMS_NORM)
_config_env+=(PRIMUS_FUSED_RESIDUAL_NORM)
_config_env+=(PRIMUS_FUSED_RESIDUAL_NORM_V2)
_config_env+=(PRIMUS_FUSED_RMSNORM_MXFP4)
_config_env+=(APPLY_SKIP_Y_STE)
_config_env+=(PRIMUS_TRAIN_ITERS)
_config_env+=(PRIMUS_EVAL_INTERVAL)
_config_env+=(PRIMUS_EVAL_ITERS)
_config_env+=(PRIMUS_ATTN_SOFTMAX_FP32)
_config_env+=(PRIMUS_ACCUM_GRADS_FP32)
_config_env+=(PRIMUS_PROFILE)
_config_env+=(PRIMUS_PROFILE_STEP_START)
_config_env+=(PRIMUS_PROFILE_STEP_END)
_config_env+=(PRIMUS_STDERR_SINK_LEVEL)
_config_env+=(PRIMUS_LOG_INTERVAL)

echo "${_config_env[@]}"
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
trap 'set -eux; cleanup_docker' EXIT

# Setup container
docker run --rm --init --detach \
    --net=host --uts=host --ipc=host \
    --device /dev/dri --device /dev/kfd --device=/dev/infiniband \
    --cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN \
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
    echo "launching experiment using: ${_run_config_env[*]} ${_cont_name} /workspace/code/run_and_time.sh"
    docker exec "${_run_config_env[@]}" "${_cont_name}" bash /workspace/code/run_and_time.sh
  ) | grep --line-buffered -v "connected peer ranks" | tee "${_logfile_base}_${_experiment_index}.log"

  if [ "${CHECK_COMPLIANCE}" -eq 1 ]; then
      docker exec "${_run_config_env[@]}" "${_cont_name}"  \
           python3 -m mlperf_logging.compliance_checker --usage training \
           --ruleset "${MLPERF_RULESET}"                                 \
           --log_output "/results/compliance_${DATESTAMP}.out"           \
           "/results/${DATESTAMP}_${_experiment_index}.log" \
      || echo "WARNING: Compliance check failed for experiment ${_experiment_index} (non-blocking)"
  fi

done
