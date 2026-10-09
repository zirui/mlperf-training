#!/bin/bash
# Multi-node MLPerf Flux1 (Primus) launcher. Submit with: sbatch job.sh
#
# run_with_docker_slurm.sh runs once per node (ntasks-per-node=1), starts the
# container there, and derives the rendezvous address from the allocation.
# DGXSYSTEM must name an existing config_<DGXSYSTEM>.sh in this directory.

#SBATCH --gres=gpu:8
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
# #SBATCH --nodelist=<>             # uncomment to pin specific nodes
#SBATCH --time=60:00:00
#SBATCH --job-name=flux1_primus_mlperf
#SBATCH --output=flux1_primus_%j.out
#SBATCH --error=flux1_primus_%j.err
#SBATCH --partition=YOUR_PARTITION

set -x

# Fill these in before submitting.
export CONT="${CONT:?set CONT to the Primus flux1 image}"
export DATADIR="${DATADIR:-/path/to/dataset}"
export LOGDIR="${LOGDIR:-/path/to/results}"
# Must match --nodes above.
export DGXSYSTEM=${DGXSYSTEM:-MI355X_4x8x32}
export NEXP=1

# Drawn HERE, before srun, so all nodes inherit one value -- the fallback inside
# run_with_docker_slurm.sh runs once per node and would hand each a different root.
# A constant here was reaching every trial of a submission as the same ten seeds.
export ROOT_SEED=${ROOT_SEED:-$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')}

# Reading the dataset over shared storage costs ~10% throughput. Files already staged are skipped.
if [[ "${FLUX_STAGE_LOCAL:-1}" == 1 ]]; then
    if ! srun bash scripts/stage_dataset_local.sh; then
        echo "Node-local dataset staging failed; set FLUX_STAGE_LOCAL=0 to train from DATADIR." >&2
        exit 1
    fi
fi

srun bash run_with_docker_slurm.sh
