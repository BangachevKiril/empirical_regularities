from __future__ import annotations

import argparse
import csv
import math
import time
from collections import defaultdict
from functools import cache
from pathlib import Path

import torch

from cumulant_propagation import propagate_cumulants
from cumulant_propagation._arc_mlp_kprop.diagslice import eval_part
from cumulant_propagation._arc_mlp_kprop.factor_k4 import FactoredTensor4
from cumulant_propagation._arc_mlp_kprop.flop_utils import flop_name, slice_factor
from cumulant_propagation._arc_mlp_kprop.kprop_harmonic import (
    get_all_terms_iso,
    multiply_wicks,
)
from cumulant_propagation._arc_mlp_kprop.partitions import check_vec_partition
from cumulant_propagation._arc_mlp_kprop.tensor_utils import symmetrize
from cumulant_propagation._arc_mlp_kprop.wick import relu_wick_coef
from data_generators import make_data_generator
from experiments.mlp_mean_concentration import (
    _cumulant_dtype,
    _sync_if_cuda,
    stream_mlp_mean,
)
from experiments.run_k4_rank_truncation_sweep import (
    _make_args,
    _stream_covariance as _stream_covariance_direct,
)
from inference_models import DeepReLUMLP


def _canonical_device(device: torch.device | str) -> torch.device:
    """Resolve aliases such as cuda to the concrete tensor device cuda:0."""
    return torch.empty((), device=device).device


def _read_done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {int(row["sample_k"]) for row in csv.DictReader(handle)}


def _append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "rank_label",
        "sample_k",
        "sample_count",
        "squared_error",
        "log_squared_error",
        "elapsed_seconds",
        "covariance_seconds",
        "propagation_seconds",
        "actual_rank",
    ]
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def rank0_input_cumulants_optimized(
    *,
    n: int,
    p: int,
    sample_count: int,
    covariance: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, object]:
    eye = torch.eye(n, device=device, dtype=dtype)
    actual_device = eye.device
    m = int(sample_count)

    second_a = float(p * (p - 1)) / float(m + p - 1)
    second_b = float(m) / float(m + p - 1)
    second = covariance.mul(second_b)
    second.diagonal().add_(second_a)
    cumulants: dict[int, object] = {
        1: torch.zeros(n, device=actual_device, dtype=dtype),
        2: second,
    }

    lambda2 = float(m) / float(m + p - 1)
    gamma = float(m) / float(p**3 + (m - 1) * (3 * p - 2))
    beta = lambda2 - float(p) * gamma
    alpha = float(p) - 2.0 * float(p) * lambda2 + float(p * p) * gamma

    # Coalesced analytic K4 rank-0 correction:
    #   Sym((-6 alpha I) x I + (-12 beta C) x I)
    # = Sym((-6 alpha I - 12 beta C) x I).
    analytic_left = covariance.mul(-12.0 * beta)
    analytic_left.diagonal().add_(-6.0 * alpha)
    cumulants[4] = FactoredTensor4(
        n=n,
        factors=(analytic_left[:, :, None], eye[:, :, None]),
        device=actual_device,
        dtype=dtype,
        assume_symmetric=True,
    )
    return cumulants


def _stream_covariance_rank0(
    data_generator,
    *,
    total_samples: int,
    seed: int,
    batch_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Stream ICA covariance through source Gram when the FLOP model prefers it."""
    if not hasattr(data_generator, "A") or not hasattr(data_generator, "p"):
        return _stream_covariance_direct(
            data_generator,
            total_samples=total_samples,
            seed=seed,
            batch_size=batch_size,
            dtype=dtype,
        )

    n = int(data_generator.n)
    p = int(data_generator.p)
    direct_cost = int(total_samples) * (p * n + n * n)
    source_cost = int(total_samples) * p * p + n * p * p + n * n * p
    if source_cost >= direct_cost:
        return _stream_covariance_direct(
            data_generator,
            total_samples=total_samples,
            seed=seed,
            batch_size=batch_size,
            dtype=dtype,
        )

    device = data_generator.device
    generator = data_generator._make_generator(seed)
    source_gram = torch.zeros((p, p), device=device, dtype=dtype)
    samples_seen = 0
    with torch.inference_mode():
        while samples_seen < total_samples:
            current_batch = min(batch_size, total_samples - samples_seen)
            signs = torch.randint(
                low=0,
                high=2,
                size=(current_batch, p),
                generator=generator,
                device=device,
            )
            sources = signs.to(dtype=dtype).mul_(2).sub_(1)
            source_gram.add_(sources.T @ sources)
            samples_seen += current_batch

    source_cov = source_gram / float(total_samples)
    A = data_generator.A.to(dtype=dtype)
    covariance = (A @ source_cov) @ A.T
    return 0.5 * (covariance + covariance.T)


def _final_relu_mean_only(preactivation_tower: dict[int, object]) -> torch.Tensor:
    wk = preactivation_tower
    mean = wk[1].core
    var = wk[2].core.diag()
    n = int(wk[1].n)

    @cache
    @flop_name("get_wick_coef")
    def get_wick_coef(k: int, p: int) -> torch.Tensor:
        return relu_wick_coef(mean=mean, var=var, k=k, p=p)

    pK_slices = defaultdict(lambda: 0.0)
    terms_iso = [
        (int_part, vec_part, count)
        for int_part, vec_part_dict in get_all_terms_iso(k_max=4, d_max=4).items()
        for vec_part, count in vec_part_dict.items()
        if len(int_part) <= 1
        and int_part != (1, 1, 1, 1)
        and (int_part, set(vec_part)) != ((2, 1, 1, 1), {(1, 1, 1, 1)})
        and int_part != (2, 2, 1, 1)
    ]
    for int_part, vec_part, count in terms_iso:
        with flop_name("nonlin_sum", factor=slice_factor(int_part, n=n)):
            term = eval_part(wk, vec_part, len(int_part), output_zero_repeated=True)
            if term is None:
                continue
            pK_slices[int_part] += count * multiply_wicks(
                term,
                check_vec_partition(vec_part, len(int_part)),
                p=int_part,
                wick_lookup=get_wick_coef,
            )

    for int_part in pK_slices:
        pK_slices[int_part] = symmetrize(pK_slices[int_part], vec=int_part)
    core = pK_slices.get((1,))
    if core is None:
        core = torch.zeros_like(mean)
    return core


def optimized_cumulant_propagation_mean(
    *,
    model: DeepReLUMLP,
    cumulants: dict[int, object],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    final_layer = len(tuple(model.weights)) - 1
    preactivation_tower = propagate_cumulants(
        model,
        cumulants,
        k_max=4,
        factor=True,
        return_tensors=False,
        up_to_layer=f"pre{final_layer}",
        output_d_max=4,
        device=device,
        dtype=dtype,
    )
    mean = _final_relu_mean_only(preactivation_tower)
    return mean.to(dtype=torch.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--p", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=25)
    parser.add_argument("--true-samples", type=int, default=2**30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--cumulant-batch-size", type=int, default=8192)
    parser.add_argument("--cumulant-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--ica-seed", type=int, default=0)
    parser.add_argument("--mlp-seed", type=int, default=0)
    parser.add_argument("--true-seed-base", type=int, default=10_000)
    parser.add_argument("--cumulant-sample-seed-base", type=int, default=2_000_000_000)
    parser.add_argument("--seed-stride", type=int, default=10_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--true-mean-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "rank0_optimized_results.csv"
    done = _read_done(csv_path)

    device = _canonical_device(args.device)
    dtype = _cumulant_dtype(args.cumulant_dtype)
    torch.manual_seed(args.mlp_seed)
    model = DeepReLUMLP(n=args.n, L=args.depth, device=device, dtype=torch.float32)
    model.eval()
    data_generator = make_data_generator(args=_make_args(args), device=device, dtype=torch.float32)

    if args.true_mean_path is not None and args.true_mean_path.exists():
        true_mean = torch.load(args.true_mean_path, map_location=device).to(device=device, dtype=torch.float64)
        print(f"loaded true mean from {args.true_mean_path}", flush=True)
    else:
        true_mean_result = stream_mlp_mean(
            model=model,
            data_generator=data_generator,
            total_samples=args.true_samples,
            batch_size=args.batch_size,
            seed_base=args.true_seed_base,
        )
        true_mean = true_mean_result.mean
        torch.save(true_mean.detach().cpu(), args.output_dir / "true_mean.pt")
        print(
            f"computed true mean from {args.true_samples} samples "
            f"(forward={true_mean_result.forward_seconds:.2f}s)",
            flush=True,
        )

    print(f"optimized rank-0 sweep n={args.n} p={args.p} L={args.depth} k={args.k_min}..{args.k_max}", flush=True)
    for sample_k in range(args.k_min, args.k_max + 1):
        if sample_k in done:
            continue
        sample_count = 2**sample_k
        seed = args.cumulant_sample_seed_base + sample_k * args.seed_stride
        _sync_if_cuda(device)
        start = time.time()
        cov_start = time.time()
        covariance = _stream_covariance_rank0(
            data_generator,
            total_samples=sample_count,
            seed=seed,
            batch_size=args.cumulant_batch_size,
            dtype=dtype,
        )
        _sync_if_cuda(device)
        covariance_seconds = time.time() - cov_start

        cumulants = rank0_input_cumulants_optimized(
            n=args.n,
            p=args.p,
            sample_count=sample_count,
            covariance=covariance,
            device=device,
            dtype=dtype,
        )
        prop_start = time.time()
        mean = optimized_cumulant_propagation_mean(
            model=model,
            cumulants=cumulants,
            device=device,
            dtype=dtype,
        )
        _sync_if_cuda(device)
        propagation_seconds = time.time() - prop_start
        squared_error = torch.sum((mean - true_mean) ** 2).item()
        elapsed_seconds = time.time() - start
        _append_row(
            csv_path,
            {
                "method": "structured_cp4_rank0_optimized",
                "rank_label": "r=0",
                "sample_k": sample_k,
                "sample_count": sample_count,
                "squared_error": squared_error,
                "log_squared_error": math.log(squared_error),
                "elapsed_seconds": elapsed_seconds,
                "covariance_seconds": covariance_seconds,
                "propagation_seconds": propagation_seconds,
                "actual_rank": 0,
            },
        )
        print(
            f"k={sample_k} m={sample_count} log_sq_error={math.log(squared_error): .6f} "
            f"cov={covariance_seconds:.2f}s prop={propagation_seconds:.2f}s",
            flush=True,
        )
        del covariance, cumulants, mean
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
