#!/bin/bash

#SBATCH --gres=gpu:8
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
# #SBATCH --nodelist=<>    		# uncomment to use specific nodes
#SBATCH --time=60:00:00
#SBATCH --job-name=gpt_oss_20b_mlperf
#SBATCH --output=gpt_oss_20b_%j.out
#SBATCH --error=gpt_oss_20b_%j.err
#SBATCH --partition=<YOUR_PARTITION>

set -x

export CONT="<ROCM_MLPERF_CONTAINER>"
export DATADIR=/path/to/dataset
export MODELDIR=/path/to/tokenizer
export LOGDIR=/path/to/results
export DGXSYSTEM=MI355X_2x8x1_tp1pp1ep1_gbs32	# update appropriate node config
export NEXP=10				        # number of experiments

export ROOT_SEED=${ROOT_SEED:-$RANDOM}

srun bash run_with_docker_slurm.sh
