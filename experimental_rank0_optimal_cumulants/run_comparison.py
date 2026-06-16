from __future__ import annotations

import argparse
import csv
import math
import time
from collections import defaultdict
from functools import cache
from pathlib import Path

import torch

import experiments.plot_k4_rank1_power_comparison as plot_base
from cumulant_propagation import propagate_cumulants
from cumulant_propagation._arc_mlp_kprop.diagslice import eval_part
from cumulant_propagation._arc_mlp_kprop.flop_utils import (
    NamedFlopCounter,
    flop_name,
    slice_factor,
)
from cumulant_propagation._arc_mlp_kprop.kprop_harmonic import (
    get_all_terms_iso,
    multiply_wicks,
)
from cumulant_propagation._arc_mlp_kprop.partitions import check_vec_partition
from cumulant_propagation._arc_mlp_kprop.tensor_utils import symmetrize
from cumulant_propagation._arc_mlp_kprop.wick import relu_wick_coef
from data_generators import make_data_generator
from experimental_rank0_optimal_cumulants.initialization import rank0_input_cumulants
from experimental_rank0_optimal_cumulants.reduced_k4 import install, vec_under_degree
from experimental_rank0_optimized.rank0_optimized_sweep import (
    _canonical_device,
    _stream_covariance_rank0,
)
from experiments.mlp_mean_concentration import _cumulant_dtype, _sync_if_cuda
from experiments.run_k4_rank_truncation_sweep import _make_args
from inference_models import DeepReLUMLP


def _read_done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {int(row["sample_k"]) for row in csv.DictReader(handle)}


def _append_row(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _rank0_covariance_flops(*, n: int, p: int, sample_count: int) -> int:
    n_i = int(n)
    p_i = int(p)
    m_i = int(sample_count)
    direct = 2 * m_i * p_i * n_i + 2 * m_i * n_i * n_i
    source = 2 * m_i * p_i * p_i + 2 * n_i * p_i * p_i + 2 * n_i * n_i * p_i
    return min(direct, source)


def _initialization_flops(*, n: int) -> int:
    n_i = int(n)
    return 2 * n_i * n_i + 2 * n_i


def _final_relu_mean_only_reduced(preactivation_tower: dict[int, object]) -> torch.Tensor:
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
        and vec_under_degree(vec_part) <= 4
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


def reduced_cumulant_propagation_mean(
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
    return _final_relu_mean_only_reduced(preactivation_tower).to(dtype=torch.float64)


def _load_true_mean(path: Path, *, device: torch.device) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"true mean not found: {path}")
    true_mean = torch.load(path, map_location=device).to(device=device, dtype=torch.float64)
    print(f"loaded true mean from {path}", flush=True)
    return true_mean


def _run_new_method(
    *,
    args: argparse.Namespace,
    model: DeepReLUMLP,
    data_generator,
    true_mean: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    csv_path: Path,
) -> None:
    fieldnames = [
        "method",
        "sample_k",
        "sample_count",
        "squared_error",
        "log_squared_error",
        "flops",
        "propagation_flops",
        "elapsed_seconds",
        "covariance_seconds",
        "propagation_seconds",
    ]
    done = _read_done(csv_path)
    propagation_flops: int | None = None

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

        cumulants = rank0_input_cumulants(
            n=args.n,
            p=args.p,
            sample_count=sample_count,
            covariance=covariance,
            device=device,
            dtype=dtype,
        )
        prop_start = time.time()
        if propagation_flops is None:
            with NamedFlopCounter() as counter:
                mean = reduced_cumulant_propagation_mean(
                    model=model,
                    cumulants=cumulants,
                    device=device,
                    dtype=dtype,
                )
            propagation_flops = int(counter.total())
        else:
            mean = reduced_cumulant_propagation_mean(
                model=model,
                cumulants=cumulants,
                device=device,
                dtype=dtype,
            )
        _sync_if_cuda(device)
        propagation_seconds = time.time() - prop_start
        squared_error = torch.sum((mean - true_mean) ** 2).item()
        elapsed_seconds = time.time() - start
        flops = (
            _rank0_covariance_flops(n=args.n, p=args.p, sample_count=sample_count)
            + _initialization_flops(n=args.n)
            + int(propagation_flops)
        )
        _append_row(
            csv_path,
            fieldnames,
            {
                "method": "rank0_optimal_sum_le_k",
                "sample_k": sample_k,
                "sample_count": sample_count,
                "squared_error": squared_error,
                "log_squared_error": math.log(squared_error),
                "flops": flops,
                "propagation_flops": propagation_flops,
                "elapsed_seconds": elapsed_seconds,
                "covariance_seconds": covariance_seconds,
                "propagation_seconds": propagation_seconds,
            },
        )
        print(
            f"rank0-optimal k={sample_k} m={sample_count} "
            f"log_sq_error={math.log(squared_error): .6f} "
            f"cov={covariance_seconds:.2f}s prop={propagation_seconds:.2f}s",
            flush=True,
        )
        del covariance, cumulants, mean
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _read_existing_sampling(path: Path, *, k_min: int, k_max: int) -> list[plot_base.Point]:
    points: list[plot_base.Point] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            sample_k = int(row["k"])
            if k_min <= sample_k <= k_max:
                points.append(
                    plot_base.Point(
                        "sampling",
                        sample_k,
                        float(row["m"]),
                        float(row["forward_flops"]),
                        float(row["mean_squared_error"]),
                    )
                )
    return points


def _read_existing_k4(path: Path, *, k_min: int, k_max: int) -> list[plot_base.Point]:
    points: list[plot_base.Point] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            sample_k = int(row["sample_k"])
            if k_min <= sample_k <= k_max:
                points.append(
                    plot_base.Point(
                        "current K=4",
                        sample_k,
                        float(row["sample_count"]),
                        float(row["flops"]),
                        float(row["squared_error"]),
                    )
                )
    return points


def _read_new(path: Path, *, k_min: int, k_max: int) -> list[plot_base.Point]:
    points: list[plot_base.Point] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            sample_k = int(row["sample_k"])
            if k_min <= sample_k <= k_max:
                points.append(
                    plot_base.Point(
                        "rank0 optimal",
                        sample_k,
                        float(row["sample_count"]),
                        float(row["flops"]),
                        float(row["squared_error"]),
                    )
                )
    return points


def _plot(args: argparse.Namespace, *, new_csv: Path) -> None:
    plot_base.COLORS = {
        "sampling": "#1f77b4",
        "current K=4": "#9467bd",
        "rank0 optimal": "#2ca02c",
    }
    plot_base.SERIES_ORDER = ["sampling", "current K=4", "rank0 optimal"]
    points = []
    points.extend(_read_existing_sampling(args.sampling_csv, k_min=args.k_min, k_max=args.k_max))
    points.extend(_read_existing_k4(args.current_k4_csv, k_min=args.k_min, k_max=args.k_max))
    points.extend(_read_new(new_csv, k_min=args.k_min, k_max=args.k_max))
    points = sorted(points, key=lambda point: (plot_base.SERIES_ORDER.index(point.label), point.sample_k))

    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_base._write_points(points, plot_dir / "plot_points.csv")
    subtitle = (
        f"ICA unknown A, n={args.n}, p={args.p}, L={args.depth}, "
        f"m=2^{args.k_min}..2^{args.k_max}, truth=2^30"
    )
    plot_base._draw_svg(
        output=plot_dir / "flops_vs_error.svg",
        title="Error vs FLOPs",
        subtitle=subtitle,
        xlabel="FLOPs (log scale; p-dependent data generation excluded)",
        ylabel="Squared error",
        points=points,
        x_value=lambda point: point.flops,
        y_value=lambda point: point.error,
        x_tick_kind="flops",
        y_tick_kind="error",
    )
    plot_base._draw_svg(
        output=plot_dir / "samples_vs_error.svg",
        title="Error vs Samples",
        subtitle=subtitle,
        xlabel="Samples m",
        ylabel="Squared error",
        points=points,
        x_value=lambda point: point.samples,
        y_value=lambda point: point.error,
        x_tick_kind="samples",
        y_tick_kind="error",
    )
    plot_base._draw_svg(
        output=plot_dir / "samples_vs_flops.svg",
        title="FLOPs vs Samples",
        subtitle=subtitle,
        xlabel="Samples m",
        ylabel="FLOPs",
        points=points,
        x_value=lambda point: point.samples,
        y_value=lambda point: point.flops,
        x_tick_kind="samples",
        y_tick_kind="flops",
    )
    print(f"wrote plots to {plot_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--p", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--cumulant-batch-size", type=int, default=65536)
    parser.add_argument("--cumulant-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--ica-seed", type=int, default=0)
    parser.add_argument("--mlp-seed", type=int, default=0)
    parser.add_argument("--cumulant-sample-seed-base", type=int, default=2_000_000_000)
    parser.add_argument("--seed-stride", type=int, default=10_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--true-mean-path", type=Path, required=True)
    parser.add_argument("--sampling-csv", type=Path, required=True)
    parser.add_argument("--current-k4-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    install()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = _canonical_device(args.device)
    dtype = _cumulant_dtype(args.cumulant_dtype)
    torch.manual_seed(args.mlp_seed)
    model = DeepReLUMLP(n=args.n, L=args.depth, device=device, dtype=torch.float32)
    model.eval()
    data_generator = make_data_generator(args=_make_args(args), device=device, dtype=torch.float32)
    true_mean = _load_true_mean(args.true_mean_path, device=device)

    new_csv = args.output_dir / "rank0_optimal_results.csv"
    _run_new_method(
        args=args,
        model=model,
        data_generator=data_generator,
        true_mean=true_mean,
        device=device,
        dtype=dtype,
        csv_path=new_csv,
    )
    _plot(args, new_csv=new_csv)


if __name__ == "__main__":
    main()
