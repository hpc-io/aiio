#!/bin/bash
#SBATCH -C gpu
#SBATCH -G 1
##SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1


##SBATCH --qos=debug
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 20:10:00
##SBATCH -t 00:10:00
#SBATCH -J xgb
#SBATCH -o IOD-train-xgb.%j.out
#SBATCH -e IOD-train-xgb.%j.err
#SBATCH -A m1248

##SBATCH --constraint=haswell


module load cgpu

module load python3/3.9-anaconda-2021.11
#module load tensorflow
module list

set -x
#export MPICH_GPU_SUPPORT_ENABLED=0
srun -l -u python ./IOD-train-xgb.py



