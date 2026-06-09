# Empirical Regularities

Initial project structure for inference models and data generators.

## ORCD environment

From `~/empirical_regularities` on ORCD:

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
```

On a GPU node:

```bash
srun --immediate=60 -p mit_normal_gpu --gres=gpu:l40s:1 --time=00:05:00 --mem=8G --cpus-per-task=2 python -m inference_models.smoke_test_relu_mlp
```

Expected GPU output includes `device: cuda`, `cuda_available: True`, and
`output_device: cuda:0`.
