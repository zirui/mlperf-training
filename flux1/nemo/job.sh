#!/bin/bash

#SBATCH --gres=gpu:8
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
# #SBATCH --nodelist=<>    		# uncomment to use specific nodes
#SBATCH --time=60:00:00
#SBATCH --job-name=flux1_mlperf
#SBATCH --output=flux1_%j.out
#SBATCH --error=flux1_%j.err
#SBATCH --partition=<YOUR_PARTITION>

set -x

export CONT=<ROCM_MLPERF_CONTAINER>
export DATADIR=/path/to/dataset
export LOGDIR=/path/to/results
export DGXSYSTEM=MI355X_02x08x64	# update appropriate node config
export NEXP=10				        # number of experiments

export ROOT_SEED=${ROOT_SEED:-$RANDOM}

srun bash run_with_docker_slurm.sh
