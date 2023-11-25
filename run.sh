#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=gpulab02
#SBATCH --qos=gpulab02
#SBATCH -J mfq10
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xxx@gmail.com

source activate edge
python run_comp_2.py
