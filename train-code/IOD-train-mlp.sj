#!/bin/bash
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH -c 32
#SBATCH -t 12:00:00
#SBATCH -J IOD-train-mlp
#SBATCH -o IOD-train-mlp.%j.out
#SBATCH -e IOD-train-mlp.%j.err
#SBATCH --gpu-bind=none
#SBATCH -A m2621



module load tensorflow
module list

set -x
export MPICH_GPU_SUPPORT_ENABLED=0
srun -l -u python ./IOD-train-mlp.py


