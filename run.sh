#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=gpulab02
#SBATCH --qos=gpulab02
#SBATCH -J MFAC
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=11910718@mail.sustech.edu.cn

source activate edge
python run_mfac.py