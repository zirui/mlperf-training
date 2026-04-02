#!/bin/bash

#SBATCH --gres=gpu:8
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=tw011,tw046
#SBATCH --time=10:00:00
#SBATCH --job-name=flux1_2nodes
#SBATCH --output=flux1_2nodes%j.out
#SBATCH --error=flux1_2nodes_%j.err
#SBATCH --partition=mlperf
#SBATCH --comment=flux1_2nodes

set -x

srun bash run_with_docker_slurm.sh
