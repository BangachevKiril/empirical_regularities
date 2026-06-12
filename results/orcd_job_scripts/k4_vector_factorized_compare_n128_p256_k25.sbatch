#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=results/logs/k4_vector_factorized_compare_n128_p256_k25_%j.out

set -euo pipefail
cd /home/kirilb/data/empirical_regularities
source .venv/bin/activate
export PYTHONPATH=/home/kirilb/data/empirical_regularities

BASE=results/unknown_a_k4_direct_flops_n128_L4_p256_k27_true2p30
VECTOR=results/k4_vector_factorized_n128_p256_L4_k25_true2p30
PLOTS=results/k4_vector_factorized_comparison_n128_p256_L4_k25_true2p30/plots

python -m experiments.run_k4_rank0_explicit_sweep \
  --device cuda \
  --cumulant-dtype float32 \
  --n 128 \
  --p 256 \
  --depth 4 \
  --k-min 1 \
  --k-max 25 \
  --true-samples 1073741824 \
  --batch-size 8192 \
  --cumulant-batch-size 8192 \
  --ica-seed 0 \
  --mlp-seed 0 \
  --output-dir "${VECTOR}"

python -m experiments.plot_k4_vector_factorized_comparison \
  --result-dir "${BASE}" \
  --vector-dir "${VECTOR}" \
  --output-dir "${PLOTS}" \
  --n 128 \
  --p 256 \
  --depth 4 \
  --k-min 1 \
  --k-max 25 \
  --true-samples 1073741824
