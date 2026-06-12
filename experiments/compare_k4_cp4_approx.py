from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from cumulant_propagation import propagate_cumulants
from cumulant_propagation._arc_mlp_kprop.flop_utils import NamedFlopCounter
from data_generators import make_data_generator
from data_generators.ica import (
    unknown_parameter_estimation_direct_k4 as baseline_k4,
)
from data_generators.ica import (
    unknown_parameter_estimation_direct_k4_cp4 as cp4_k4,
)
from experimental_cumulant_propagation.structured_tensor4 import StructuredTensor4
from inference_models import DeepReLUMLP


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mean_from_tower(tower: dict[int, object]) -> torch.Tensor:
    mean = tower[1]
    return mean.core if hasattr(mean, "core") else mean


def _run_method(
    *,
    method: str,
    estimator,
    model: DeepReLUMLP,
    data_generator,
    m: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    eig_tol: float,
    gram_chunk_size: int,
) -> dict[str, object]:
    _sync(device)
    start = time.time()
    cumulants, rank4 = estimator.input_cumulants(
        data_generator=data_generator,
        m=m,
        seed=seed,
        k_max=4,
        device=device,
        dtype=dtype,
        eig_tol=eig_tol,
        gram_chunk_size=gram_chunk_size,
    )
    _sync(device)
    init_seconds = time.time() - start

    k4_input = cumulants[4]
    input_cp_rank = k4_input.cp_rank if isinstance(k4_input, StructuredTensor4) else 0
    input_pair_rank = k4_input.pair_rank if isinstance(k4_input, StructuredTensor4) else rank4

    counter = NamedFlopCounter()
    _sync(device)
    prop_start = time.time()
    with counter:
        propagated = propagate_cumulants(
            model,
            cumulants,
            k_max=4,
            kind="simple",
            factor=True,
            use_avg_metric=True,
            device=device,
            dtype=dtype,
        )
    _sync(device)
    prop_seconds = time.time() - prop_start
    mean = _mean_from_tower(propagated).detach().to(torch.float64).cpu()
    return {
        "method": method,
        "rank4": rank4,
        "input_cp_rank": input_cp_rank,
        "input_pair_rank": input_pair_rank,
        "init_seconds": init_seconds,
        "propagation_seconds": prop_seconds,
        "propagation_flops": int(counter.total()),
        "mean": mean,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline direct-Z K4 with CP4-backed K4.")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--p", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/k4_cp4_approx_comparison"))
    parser.add_argument("--gram-chunk-size", type=int, default=2048)
    parser.add_argument("--eig-tol", type=float, default=1e-8)
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.manual_seed(args.seed)
    data_generator = make_data_generator(
        args=SimpleNamespace(
            input_distribution="ica",
            n=args.n,
            p=args.p,
            ica_seed=args.seed,
        ),
        device=device,
        dtype=dtype,
    )
    torch.manual_seed(args.seed + 17)
    model = DeepReLUMLP(n=args.n, L=args.depth, device=device, dtype=dtype)
    model.eval()

    if not args.no_warmup:
        _run_method(
            method="warmup",
            estimator=baseline_k4,
            model=model,
            data_generator=data_generator,
            m=max(1, min(args.samples, 8)),
            seed=args.seed + 99,
            device=device,
            dtype=dtype,
            eig_tol=args.eig_tol,
            gram_chunk_size=args.gram_chunk_size,
        )

    baseline = _run_method(
        method="baseline_direct_z",
        estimator=baseline_k4,
        model=model,
        data_generator=data_generator,
        m=args.samples,
        seed=args.seed + 101,
        device=device,
        dtype=dtype,
        eig_tol=args.eig_tol,
        gram_chunk_size=args.gram_chunk_size,
    )
    cp4 = _run_method(
        method="structured_cp4",
        estimator=cp4_k4,
        model=model,
        data_generator=data_generator,
        m=args.samples,
        seed=args.seed + 101,
        device=device,
        dtype=dtype,
        eig_tol=args.eig_tol,
        gram_chunk_size=args.gram_chunk_size,
    )

    diff = cp4["mean"] - baseline["mean"]
    baseline_norm = torch.linalg.vector_norm(baseline["mean"]).item()
    diff_norm = torch.linalg.vector_norm(diff).item()
    max_abs = torch.max(torch.abs(diff)).item()
    relative = diff_norm / max(baseline_norm, 1e-30)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "comparison.csv"
    rows = []
    for result in (baseline, cp4):
        rows.append(
            {
                "method": result["method"],
                "n": args.n,
                "p": args.p,
                "depth": args.depth,
                "samples": args.samples,
                "rank4": result["rank4"],
                "input_cp_rank": result["input_cp_rank"],
                "input_pair_rank": result["input_pair_rank"],
                "init_seconds": result["init_seconds"],
                "propagation_seconds": result["propagation_seconds"],
                "propagation_flops": result["propagation_flops"],
                "l2_diff_vs_baseline": 0.0 if result["method"] == "baseline_direct_z" else diff_norm,
                "relative_l2_diff_vs_baseline": 0.0 if result["method"] == "baseline_direct_z" else relative,
                "max_abs_diff_vs_baseline": 0.0 if result["method"] == "baseline_direct_z" else max_abs,
            }
        )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {csv_path}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
