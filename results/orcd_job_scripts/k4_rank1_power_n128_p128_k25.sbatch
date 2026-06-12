#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=results/logs/k4_rank1_power_n128_p128_k25_%j.out

set -euo pipefail
cd /home/kirilb/data/empirical_regularities
source .venv/bin/activate
export PYTHONPATH=/home/kirilb/data/empirical_regularities

OUT=results/k4_rank1_power_n128_p128_L4_k25_true2p30
TRUE_MEAN=results/k4_rank_truncation_n128_p128_L4_k25_true2p30/true_mean.pt

python experiments/run_k4_rank1_power_sweep.py \
  --device cuda \
  --cumulant-dtype float32 \
  --n 128 \
  --p 128 \
  --depth 4 \
  --k-min 1 \
  --k-max 25 \
  --true-samples 1073741824 \
  --batch-size 8192 \
  --cumulant-batch-size 8192 \
  --power-iterations 8 \
  --ica-seed 0 \
  --mlp-seed 0 \
  --sample-fourth-eig-tol 1e-6 \
  --true-mean-path "${TRUE_MEAN}" \
  --output-dir "${OUT}"
