# Empirical Regularities

Initial project structure for inference models and data generators.

## ORCD environment

From `/home/kirilb/data/empirical_regularities` on ORCD:

```bash
module load deprecated-modules
module load gcc/12.2.0-x86_64
module load python/3.10.8-x86_64
module load cuda/12.4.0
source .venv/bin/activate
```

The environment was created with PyTorch CUDA wheels:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Smoke test

```bash
python -m inference_models.smoke_test_relu_mlp
python -m data_generators.smoke_test_ica
```

The default MLP initialization uses Gaussian weights with variance `2 / n`.

## ICA generator

```python
from data_generators import ICADataGenerator

generator = ICADataGenerator(n=128, seed=1234, p=32, device="cuda")
dataset = generator.sample(m=1024, seed_=0)
```

## MLP mean concentration experiment

```bash
srun --immediate=180 -p mit_normal_gpu --gres=gpu:l40s:1 --time=00:15:00 --mem=16G --cpus-per-task=2 \
  python -m experiments.mlp_mean_concentration \
  --device cuda --n 128 --depth 8 --p 32 --ica-seed 0 --mlp-seed 0 \
  --true-samples 1000000 --batch-size 4096 --k-min 1 --k-max 16 --runs 10 \
  --output-dir results/mlp_mean_concentration
```

## MLP effective rank experiment

```bash
srun --immediate=180 -p mit_normal_gpu --gres=gpu:l40s:1 --time=00:10:00 --mem=16G --cpus-per-task=2 \
  python -m experiments.mlp_effective_rank \
  --device cuda --n 128 --depth 8 --p 32 --ica-seed 0 --sample-seed 0 --mlp-seed 0 \
  --samples 8192 --output-dir results/mlp_effective_rank
```

On a GPU node:

```bash
srun --immediate=60 -p mit_normal_gpu --gres=gpu:l40s:1 --time=00:05:00 --mem=8G --cpus-per-task=2 python -m inference_models.smoke_test_relu_mlp
```

Expected GPU output includes `device: cuda`, `cuda_available: True`, and
`output_device: cuda:0`.
