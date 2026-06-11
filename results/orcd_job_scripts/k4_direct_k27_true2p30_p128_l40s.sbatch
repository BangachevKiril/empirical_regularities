#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=12:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --output=results/logs/k4_direct_k27_true2p30_p128_l40s_%j.out

set -euo pipefail
cd /home/kirilb/data/empirical_regularities
source .venv/bin/activate

OUT=results/unknown_a_k4_direct_flops_n128_L4_p128_k27_true2p30
NO_BASE=/tmp/nonexistent_unknown_a_k4_base.csv

python -m experiments.mlp_mean_concentration \
  --device cuda \
  --n 128 \
  --depth 4 \
  --p 128 \
  --input-distribution ica \
  --parameter-estimation unknown \
  --ica-seed 0 \
  --mlp-seed 0 \
  --true-samples 1073741824 \
  --batch-size 8192 \
  --k-min 1 \
  --k-max 27 \
  --runs 1 \
  --cumulant-orders 4 \
  --cumulant-factor \
  --fixed-cumulant-methods none \
  --sample-cumulant-k-min 1 \
  --sample-cumulant-k-max 0 \
  --sample-cumulant-estimator unknown_a \
  --unknown-a-k4-factorization direct \
  --skip-flop-counts \
  --output-dir "${OUT}"

python -m experiments.extend_unknown_a_k4_direct_streamed \
  --device cuda \
  --n 128 \
  --depth 4 \
  --p 128 \
  --k-min 1 \
  --k-max 27 \
  --true-samples 1073741824 \
  --batch-size 8192 \
  --cumulant-batch-size 8192 \
  --base-cumulant-csv "${NO_BASE}" \
  --output-dir "${OUT}"

python -m experiments.plot_unknown_a_k4_factorization \
  --hide-old \
  --flop-mode no-data-k4-prop \
  --n 128 \
  --depth 4 \
  --result-dir "${OUT}" \
  --output-dir "${OUT}/plots_no_data_k4prop" \
  --title-prefix "Unknown-A ICA direct-Z K4 (n=128, p=128, L=4, truth=2^30)"
