#!/bin/bash
# Submit from this directory: sbatch job.sh
# Set --partition, CONT, DATADIR, and LOGDIR. CONT must be an image every node can pull.
# --nodes and DGXSYSTEM are a pair:
#   1 node: --nodes=1 and DGXSYSTEM=MI355X_1x8x1 (MI350X: MI350X_1x8x1)
#   2 node: --nodes=2 and DGXSYSTEM=MI355X_2x8x1 (MI350X: MI350X_2x8x1)
#   4 node: --nodes=4 and DGXSYSTEM=MI355X_4x8x1 (MI350X: MI350X_4x8x1)
# NEXP=10 is the submission. NEXP=1 is one trial.

#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=60:00:00
#SBATCH --job-name=dlrmv4
#SBATCH --output=dlrmv4_%j.out
#SBATCH --error=dlrmv4_%j.err
#SBATCH --partition=<YOUR_PARTITION>

cd "$(dirname "$(readlink -f "$0")")"

export CONT="<ROCM_MLPERF_CONTAINER>"
export DATADIR=/path/to/dataset
export LOGDIR=/path/to/results
export DGXSYSTEM=MI355X_1x8x1
export NEXP=10
export ROOT_SEED=${ROOT_SEED:-$RANDOM}

srun --ntasks-per-node=1 bash run_with_docker_slurm.sh
