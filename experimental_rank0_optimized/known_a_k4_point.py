from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch

from cumulant_propagation._arc_mlp_kprop.flop_utils import NamedFlopCounter
from data_generators import make_data_generator
from data_generators.ica import known_parameter_estimation as ica_known_estimation
from experiments.mlp_mean_concentration import (
    _cumulant_dtype,
    _sync_if_cuda,
    cumulant_propagation_mean,
)
from experiments.run_k4_rank_truncation_sweep import _make_args
from inference_models import DeepReLUMLP


def _canonical_device(device: torch.device | str) -> torch.device:
    return torch.empty((), device=device).device


def _write_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "sample_k",
        "sample_count",
        "squared_error",
        "log_squared_error",
        "elapsed_seconds",
        "warmup_seconds",
        "initialization_flops",
        "propagation_flops",
        "total_flops",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--p", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--cumulant-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--ica-seed", type=int, default=0)
    parser.add_argument("--mlp-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--true-mean-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = _canonical_device(args.device)
    dtype = _cumulant_dtype(args.cumulant_dtype)

    torch.manual_seed(args.mlp_seed)
    model = DeepReLUMLP(n=args.n, L=args.depth, device=device, dtype=torch.float32)
    model.eval()
    data_generator = make_data_generator(args=_make_args(args), device=device, dtype=torch.float32)
    true_mean = torch.load(args.true_mean_path, map_location=device).to(device=device, dtype=torch.float64)

    def build_cumulants() -> dict[int, object]:
        return ica_known_estimation.input_cumulants(
            data_generator=data_generator,
            k_max=4,
            device=device,
            dtype=dtype,
        )

    _sync_if_cuda(device)
    warmup_start = time.time()
    _ = cumulant_propagation_mean(
        model=model,
        cumulants=build_cumulants(),
        cumulant_k_max=4,
        factor=True,
        device=device,
        dtype=dtype,
    )
    _sync_if_cuda(device)
    warmup_seconds = time.time() - warmup_start

    _sync_if_cuda(device)
    start = time.time()
    mean = cumulant_propagation_mean(
        model=model,
        cumulants=build_cumulants(),
        cumulant_k_max=4,
        factor=True,
        device=device,
        dtype=dtype,
    )
    _sync_if_cuda(device)
    elapsed_seconds = time.time() - start
    squared_error = torch.sum((mean - true_mean) ** 2).item()

    with NamedFlopCounter() as counter:
        _ = cumulant_propagation_mean(
            model=model,
            cumulants=build_cumulants(),
            cumulant_k_max=4,
            factor=True,
            device=device,
            dtype=dtype,
        )
    _sync_if_cuda(device)
    propagation_flops = counter.total()
    initialization_flops = ica_known_estimation.initialization_flops(
        n=args.n,
        p=args.p,
        k_max=4,
    )
    total_flops = initialization_flops + propagation_flops

    _write_row(
        args.output_dir / "known_a_k4_point.csv",
        {
            "method": "known_a_k4",
            "sample_k": 0,
            # Log-scale plotting stand-in for "uses zero cumulant-estimation samples".
            "sample_count": 1,
            "squared_error": squared_error,
            "log_squared_error": math.log(squared_error),
            "elapsed_seconds": elapsed_seconds,
            "warmup_seconds": warmup_seconds,
            "initialization_flops": initialization_flops,
            "propagation_flops": propagation_flops,
            "total_flops": total_flops,
        },
    )
    print(
        "known A K=4 "
        f"log_sq_error={math.log(squared_error): .6f} "
        f"flops={total_flops} "
        f"warmup={warmup_seconds:.2f}s run={elapsed_seconds:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
