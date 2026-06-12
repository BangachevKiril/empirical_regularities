#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=results/logs/k4_rank1_power_compare_n64_p64_k25_%j.out

set -euo pipefail
cd /home/kirilb/data/empirical_regularities
source .venv/bin/activate
export PYTHONPATH=/home/kirilb/data/empirical_regularities

BASE=results/unknown_a_k4_direct_flops_n64_L4_p64_k25_true2p30
RANK1=results/k4_rank1_power_n64_p64_L4_k25_true2p30
PLOTS=results/k4_rank1_power_comparison_n64_p64_L4_k25_true2p30/plots
NO_BASE=/tmp/nonexistent_unknown_a_k4_n64_p64_base.csv

python -m experiments.mlp_mean_concentration \
  --device cuda \
  --n 64 \
  --depth 4 \
  --p 64 \
  --input-distribution ica \
  --parameter-estimation unknown \
  --ica-seed 0 \
  --mlp-seed 0 \
  --true-samples 1073741824 \
  --batch-size 8192 \
  --k-min 1 \
  --k-max 25 \
  --runs 1 \
  --cumulant-orders 4 \
  --cumulant-factor \
  --fixed-cumulant-methods none \
  --sample-cumulant-k-min 1 \
  --sample-cumulant-k-max 0 \
  --sample-cumulant-estimator unknown_a \
  --unknown-a-k4-factorization direct \
  --skip-flop-counts \
  --output-dir "${BASE}"

python -m experiments.extend_unknown_a_k4_direct_streamed \
  --device cuda \
  --n 64 \
  --depth 4 \
  --p 64 \
  --k-min 1 \
  --k-max 25 \
  --true-samples 1073741824 \
  --batch-size 8192 \
  --cumulant-batch-size 8192 \
  --base-cumulant-csv "${NO_BASE}" \
  --output-dir "${BASE}"

python experiments/run_k4_rank1_power_sweep.py \
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
  --power-iterations 8 \
  --ica-seed 0 \
  --mlp-seed 0 \
  --sample-fourth-eig-tol 1e-6 \
  --output-dir "${RANK1}"

python experiments/plot_k4_rank1_power_comparison.py \
  --result-dir "${BASE}" \
  --rank1-dir "${RANK1}" \
  --output-dir "${PLOTS}" \
  --n 64 \
  --p 64 \
  --depth 4 \
  --k-min 1 \
  --k-max 25 \
  --true-samples 1073741824
