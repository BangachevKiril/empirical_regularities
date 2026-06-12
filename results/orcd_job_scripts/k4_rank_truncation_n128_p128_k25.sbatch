#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --output=results/logs/k4_rank_truncation_n128_p128_k25_%j.out

set -euo pipefail
cd /home/kirilb/data/empirical_regularities
source .venv/bin/activate
export PYTHONPATH=/home/kirilb/data/empirical_regularities

OUT=results/k4_rank_truncation_n128_p128_L4_k25_true2p30

python experiments/run_k4_rank_truncation_sweep.py \
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
  --ica-seed 0 \
  --mlp-seed 0 \
  --sample-fourth-eig-tol 1e-6 \
  --output-dir "${OUT}"

python experiments/plot_k4_structured_cp4_sweep.py \
  --result-dir results/unknown_a_k4_direct_flops_n128_L4_p128_k27_true2p30 \
  --truncation-dir "${OUT}" \
  --output-dir "${OUT}/plots" \
  --n 128 \
  --p 128 \
  --depth 4 \
  --k-min 1 \
  --k-max 25 \
  --true-samples 1073741824
