#!/bin/bash
#SBATCH --job-name=hs_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --account=geog_geors
#SBATCH --partition=c_foss_gpu
#SBATCH --qos=gpu
#SBATCH --mem-per-cpu=5G
#SBATCH --gres=gpu:4
#SBATCH --time=96:00:00
#SBATCH --output=/fossfs/xiaozhen/HydroState/logs/%x-%j.out

set -euo pipefail
source /home/xiaoz/miniconda/etc/profile.d/conda.sh
conda activate hydrostate
cd /home/xiaoz/HydroState
mkdir -p /fossfs/xiaozhen/HydroState/logs

export OMP_NUM_THREADS=8
torchrun --nproc_per_node=4 tools/train.py configs/hydrostate_v1.py --launcher pytorch
