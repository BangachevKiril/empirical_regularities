#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=results/logs/k4_cp4_approx_compare_%j.out

set -euo pipefail
cd /home/kirilb/data/empirical_regularities
source .venv/bin/activate
export PYTHONPATH=/home/kirilb/data/empirical_regularities

python experiments/compare_k4_cp4_approx.py \
  --device cuda \
  --dtype float32 \
  --n 32 \
  --p 64 \
  --depth 2 \
  --samples 1024 \
  --seed 123 \
  --gram-chunk-size 2048 \
  --out-dir results/k4_cp4_approx_comparison_n32_p64_L2_m1024
