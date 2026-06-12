#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=results/logs/k4_rank0_rank1_compare_n64_p64_k25_%j.out

set -euo pipefail
cd /home/kirilb/data/empirical_regularities
source .venv/bin/activate
export PYTHONPATH=/home/kirilb/data/empirical_regularities

RANK0=results/k4_rank0_n64_p64_L4_k25_true2p30
RANK1=results/k4_rank1_power_n64_p64_L4_k25_true2p30
SAMPLING=results/unknown_a_k4_direct_flops_n64_L4_p64_k25_true2p30
PLOTS=results/k4_rank0_rank1_comparison_n64_p64_L4_k25_true2p30/plots

python -m experiments.run_k4_rank0_explicit_sweep \
  --device cuda \
  --cumulant-dtype float32 \
  --n 64 \
  --p 64 \
  --depth 4 \
  --k-min 1 \
  --k-max 25 \
  --true-samples 1073741824 \
  --batch-size 8192 \
  --cumulant-batch-size 8192 \
  --ica-seed 0 \
  --mlp-seed 0 \
  --true-mean-path "${RANK1}/true_mean.pt" \
  --output-dir "${RANK0}"

python experiments/plot_k4_rank0_rank1_comparison.py \
  --sampling-dir "${SAMPLING}" \
  --rank0-dir "${RANK0}" \
  --rank1-dir "${RANK1}" \
  --output-dir "${PLOTS}" \
  --n 64 \
  --p 64 \
  --depth 4 \
  --k-min 1 \
  --k-max 25 \
  --true-samples 1073741824
