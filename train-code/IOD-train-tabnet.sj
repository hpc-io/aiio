#!/bin/bash
##SBATCH -C cpu
##SBATCH --qos=debug
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 10:10:00
#SBATCH -J tabnet
#SBATCH -o IOD-train-tabnet.%j.out
#SBATCH -e IOD-train-tabnet.%j.err
#SBATCH -A m1248
#SBATCH --constraint=haswell


module load python3/3.9-anaconda-2021.11
#module load tensorflow
module list

set -x
#export MPICH_GPU_SUPPORT_ENABLED=0
srun -l -u python ./IOD-train-tabnet.py


