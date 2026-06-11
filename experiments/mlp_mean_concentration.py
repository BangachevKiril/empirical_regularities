from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from cumulant_propagation import propagate_cumulants
from cumulant_propagation._arc_mlp_kprop.flop_utils import NamedFlopCounter
from data_generators import (
    PROCESSES,
    DataGenerator,
    estimation_module,
    make_data_generator,
)
from data_generators.gaussian import known_parameter_estimation as gaussian_known_estimation
from data_generators.gaussian_lowrank import (
    known_parameter_estimation as gaussian_lowrank_known_estimation,
)
from data_generators.ica import known_parameter_estimation as ica_known_estimation
from data_generators.ica import unknown_parameter_estimation as ica_unknown_estimation
from inference_models import DeepReLUMLP


@dataclass(frozen=True)
class RunResult:
    k: int
    m: int
    run: int
    seed_base: int
    squared_error: float
    elapsed_seconds: float
    forward_seconds: float
    forward_flops: int

    @property
    def log_squared_error(self) -> float:
        return math.log(self.squared_error)


@dataclass(frozen=True)
class SummaryResult:
    k: int
    m: int
    runs: int
    mean_squared_error: float
    std_squared_error: float
    mean_log_squared_error: float
    std_log_squared_error: float
    mean_elapsed_seconds: float
    std_elapsed_seconds: float
    mean_forward_seconds: float
    std_forward_seconds: float
    forward_flops: int
    mean_forward_flops_per_second: float


@dataclass(frozen=True)
class MeanStreamResult:
    mean: torch.Tensor
    elapsed_seconds: float
    forward_seconds: float
    forward_flops: int


@dataclass(frozen=True)
class CumulantResult:
    method: str
    cumulant_k_max: int
    sample_k: int | None
    sample_count: int | None
    squared_error: float
    elapsed_seconds: float
    warmup_seconds: float
    rank4: int | None = None
    initialization_flops: int | None = None
    propagation_flops: int | None = None

    @property
    def log_squared_error(self) -> float:
        return math.log(self.squared_error)

    @property
    def propagation_label(self) -> str:
        return _propagation_label(self.cumulant_k_max)

    @property
    def display_method(self) -> str:
        return self.method.replace("_", " ")

    @property
    def label(self) -> str:
        return f"{self.display_method} {self.propagation_label}"

    @property
    def total_flops(self) -> int | None:
        if self.initialization_flops is None or self.propagation_flops is None:
            return None
        return self.initialization_flops + self.propagation_flops


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _propagation_label(cumulant_k_max: int) -> str:
    if cumulant_k_max == 1:
        return "mean"
    if cumulant_k_max == 2:
        return "covariance"
    return f"K={cumulant_k_max}"


def _mlp_forward_flops(*, n: int, depth: int, sample_count: int) -> int:
    return int(sample_count) * int(depth) * (2 * int(n) * int(n) + int(n))


def _ica_sample_flops(*, n: int, p: int, sample_count: int) -> int:
    return ica_known_estimation.sample_flops(n=n, p=p, sample_count=sample_count)


def _k2_estimator_flops(*, n: int, sample_count: int) -> int:
    return ica_known_estimation.k2_estimator_flops(n=n, sample_count=sample_count)


_prior_initialization_flops = ica_known_estimation.prior_initialization_flops
_known_distribution_initialization_flops = ica_known_estimation.initialization_flops
_known_lowrank_gaussian_initialization_flops = (
    gaussian_lowrank_known_estimation.initialization_flops
)
_gaussian_initialization_flops = gaussian_known_estimation.initialization_flops
_sample_initialization_flops = ica_known_estimation.sample_initialization_flops
_unknown_a_tau_initialization_flops = ica_unknown_estimation.tau_initialization_flops
_unknown_a_k4_estimator_flops = ica_unknown_estimation.k4_estimator_flops
_unknown_a_initialization_flops = ica_unknown_estimation.initialization_flops


def prior_input_cumulants(
    *,
    n: int,
    p: int,
    k_max: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, object]:
    return ica_known_estimation.prior_input_cumulants(
        n=n,
        p=p,
        k_max=k_max,
        device=device,
        dtype=dtype,
    )


def gaussian_input_cumulants(
    *,
    n: int,
    p: int | float,
    k_max: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    data_generator = SimpleNamespace(n=int(n), p=p)
    return gaussian_known_estimation.input_cumulants(
        data_generator=data_generator,
        k_max=k_max,
        device=device,
        dtype=dtype,
    )


known_distribution_scalar_fourth_core = ica_known_estimation.scalar_fourth_core
known_distribution_input_cumulants = ica_known_estimation.input_cumulants
known_lowrank_gaussian_input_cumulants = gaussian_lowrank_known_estimation.input_cumulants
sample_average_input_cumulants = ica_known_estimation.sample_average_input_cumulants
unknown_a_tau_raw_estimate = ica_unknown_estimation.tau_raw_estimate
unknown_a_input_cumulants = ica_unknown_estimation.input_cumulants


def stream_mlp_mean(
    *,
    model: DeepReLUMLP,
    data_generator: DataGenerator,
    total_samples: int,
    batch_size: int,
    seed_base: int,
) -> MeanStreamResult:
    output_sum = torch.zeros(
        data_generator.n,
        device=data_generator.device,
        dtype=torch.float64,
    )
    samples_seen = 0
    batch_index = 0
    forward_seconds = 0.0
    stream_start = time.time()

    with torch.inference_mode():
        while samples_seen < total_samples:
            current_batch = min(batch_size, total_samples - samples_seen)
            samples = data_generator.sample(current_batch, seed_=seed_base + batch_index)
            _sync_if_cuda(data_generator.device)
            forward_start = time.time()
            outputs = model(samples.T.contiguous())
            _sync_if_cuda(data_generator.device)
            forward_seconds += time.time() - forward_start
            output_sum += outputs.to(dtype=torch.float64).sum(dim=1)
            samples_seen += current_batch
            batch_index += 1

    _sync_if_cuda(data_generator.device)
    return MeanStreamResult(
        mean=output_sum / total_samples,
        elapsed_seconds=time.time() - stream_start,
        forward_seconds=forward_seconds,
        forward_flops=_mlp_forward_flops(
            n=data_generator.n,
            depth=model.L,
            sample_count=total_samples,
        ),
    )


def _cumulant_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported cumulant dtype: {name}.")


def _parse_cumulant_orders(value: str) -> list[int]:
    orders = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not orders:
        raise argparse.ArgumentTypeError("must provide at least one cumulant order")
    if any(order < 1 for order in orders):
        raise argparse.ArgumentTypeError("cumulant orders must be at least 1")
    return orders


def _parse_fixed_cumulant_methods(value: str) -> list[str]:
    methods = [part.strip() for part in value.split(",") if part.strip()]
    if methods == ["none"]:
        return []
    allowed = {"prior", "known_distribution"}
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unsupported fixed cumulant method(s): {', '.join(unknown)}"
        )
    return methods


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cumulant_propagation_mean(
    *,
    model: DeepReLUMLP,
    cumulants: dict[int, object],
    cumulant_k_max: int,
    factor: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    propagated = propagate_cumulants(
        model,
        cumulants,
        k_max=cumulant_k_max,
        factor=factor,
        return_tensors=False,
        device=device,
        dtype=dtype,
    )
    mean = propagated[1]
    if hasattr(mean, "to_tensor"):
        mean = mean.to_tensor()
    return mean.to(dtype=torch.float64)


def _run_cumulant_once(
    *,
    args: argparse.Namespace,
    model: DeepReLUMLP,
    true_mean: torch.Tensor,
    cumulant_k_max: int,
    device: torch.device,
    dtype: torch.dtype,
    build_cumulants,
) -> tuple[float, int | None]:
    cumulants, rank4 = build_cumulants()
    mean = cumulant_propagation_mean(
        model=model,
        cumulants=cumulants,
        cumulant_k_max=cumulant_k_max,
        factor=args.cumulant_factor,
        device=device,
        dtype=dtype,
    )
    _sync_if_cuda(device)
    squared_error = torch.sum((mean - true_mean) ** 2).item()
    del cumulants, mean
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return squared_error, rank4


def _count_cumulant_propagation_flops(
    *,
    args: argparse.Namespace,
    model: DeepReLUMLP,
    cumulant_k_max: int,
    device: torch.device,
    dtype: torch.dtype,
    build_cumulants,
) -> tuple[int | None, int | None]:
    if args.skip_flop_counts:
        return None, None
    cumulants, rank4 = build_cumulants()
    try:
        with NamedFlopCounter() as counter:
            _ = cumulant_propagation_mean(
                model=model,
                cumulants=cumulants,
                cumulant_k_max=cumulant_k_max,
                factor=args.cumulant_factor,
                device=device,
                dtype=dtype,
            )
        _sync_if_cuda(device)
        return counter.total(), rank4
    finally:
        del cumulants
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _propagation_flop_cache_key(
    *,
    method: str,
    cumulant_k_max: int,
    rank4: int | None,
    factor: bool,
    dtype: torch.dtype,
) -> tuple[int, bool, str, int | str]:
    # The propagation FLOP counter is far slower than the propagation itself for
    # high-rank sample K4 inputs. We count each sampled order once and keep the
    # sample-count/rank dependence in the initialization FLOP estimate.
    if method in {"sample_avg", "gaussian_exact"}:
        rank_or_method: int | str = method
    else:
        rank_or_method = rank4 if rank4 is not None else method
    return cumulant_k_max, factor, str(dtype), rank_or_method


def _timed_cumulant_result(
    *,
    args: argparse.Namespace,
    model: DeepReLUMLP,
    true_mean: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    method: str,
    cumulant_k_max: int,
    sample_k: int | None,
    sample_count: int | None,
    build_cumulants,
    estimate_initialization_flops,
    propagation_flop_cache: dict[tuple[int, bool, str, int | str], tuple[int | None, int | None]]
    | None = None,
) -> CumulantResult:
    _sync_if_cuda(device)
    warmup_start = time.time()
    _run_cumulant_once(
        args=args,
        model=model,
        true_mean=true_mean,
        cumulant_k_max=cumulant_k_max,
        device=device,
        dtype=dtype,
        build_cumulants=build_cumulants,
    )
    _sync_if_cuda(device)
    warmup_seconds = time.time() - warmup_start

    _sync_if_cuda(device)
    start = time.time()
    squared_error, rank4 = _run_cumulant_once(
        args=args,
        model=model,
        true_mean=true_mean,
        cumulant_k_max=cumulant_k_max,
        device=device,
        dtype=dtype,
        build_cumulants=build_cumulants,
    )
    _sync_if_cuda(device)
    elapsed = time.time() - start
    cache_key = _propagation_flop_cache_key(
        method=method,
        cumulant_k_max=cumulant_k_max,
        rank4=rank4,
        factor=args.cumulant_factor,
        dtype=dtype,
    )
    if propagation_flop_cache is not None and cache_key in propagation_flop_cache:
        propagation_flops, counted_rank4 = propagation_flop_cache[cache_key]
    else:
        propagation_flops, counted_rank4 = _count_cumulant_propagation_flops(
            args=args,
            model=model,
            cumulant_k_max=cumulant_k_max,
            device=device,
            dtype=dtype,
            build_cumulants=build_cumulants,
        )
        if propagation_flop_cache is not None:
            propagation_flop_cache[cache_key] = (propagation_flops, counted_rank4)
    if rank4 is None:
        rank4 = counted_rank4
    initialization_flops = estimate_initialization_flops(rank4)

    return CumulantResult(
        method=method,
        cumulant_k_max=cumulant_k_max,
        sample_k=sample_k,
        sample_count=sample_count,
        squared_error=squared_error,
        elapsed_seconds=elapsed,
        warmup_seconds=warmup_seconds,
        rank4=rank4,
        initialization_flops=initialization_flops,
        propagation_flops=propagation_flops,
    )


def run_cumulant_experiments(
    *,
    args: argparse.Namespace,
    model: DeepReLUMLP,
    data_generator: DataGenerator,
    true_mean: torch.Tensor,
    device: torch.device,
) -> list[CumulantResult]:
    dtype = _cumulant_dtype(args.cumulant_dtype)
    results: list[CumulantResult] = []
    propagation_flop_cache: dict[
        tuple[int, bool, str, int | str], tuple[int | None, int | None]
    ] = {}
    known_estimation = estimation_module(
        data_generation=args.input_distribution,
        parameter_estimation="known",
    )
    unknown_estimation = estimation_module(
        data_generation=args.input_distribution,
        parameter_estimation="unknown",
    )

    if args.parameter_estimation == "unknown":
        builders = []
    elif args.parameter_estimation == "auto" and args.input_distribution == "ica":
        available_builders = {
            "prior": lambda k_max: ica_known_estimation.prior_input_cumulants(
                n=args.n,
                p=args.p,
                k_max=k_max,
                device=device,
                dtype=dtype,
            ),
            "known_distribution": lambda k_max: known_estimation.input_cumulants(
                data_generator=data_generator,
                k_max=k_max,
                device=device,
                dtype=dtype,
            ),
        }
        builders = [
            (method, available_builders[method])
            for method in args.fixed_cumulant_methods
        ]
    else:
        builders = [
            (
                "known_distribution",
                lambda k_max, module=known_estimation: module.input_cumulants(
                    data_generator=data_generator,
                    k_max=k_max,
                    device=device,
                    dtype=dtype,
                ),
            )
        ]

    def sample_cumulant_jobs(
        *,
        cumulant_k_max: int,
        sample_k: int,
        sample_count: int,
        seed: int,
    ):
        if args.parameter_estimation == "known":
            return
        if args.parameter_estimation == "unknown":
            method_name = "unknown_a" if args.input_distribution == "ica" else "unknown_parameter"
            yield (
                method_name,
                lambda method_k=cumulant_k_max, count=sample_count, seed_=seed, module=unknown_estimation: module.input_cumulants(
                    data_generator=data_generator,
                    m=count,
                    seed=seed_,
                    k_max=method_k,
                    device=device,
                    dtype=dtype,
                    eig_tol=args.sample_fourth_eig_tol,
                    gram_chunk_size=args.unknown_a_gram_chunk_size,
                ),
                lambda rank4, method_k=cumulant_k_max, count=sample_count, module=unknown_estimation: module.initialization_flops(
                    n=args.n,
                    p=args.p,
                    sample_count=count,
                    k_max=method_k,
                    rank4=rank4,
                ),
            )
            return

        if args.input_distribution == "gaussian":
            yield (
                "gaussian_exact",
                lambda method_k=cumulant_k_max, module=known_estimation: (
                    module.input_cumulants(
                        data_generator=data_generator,
                        k_max=method_k,
                        device=device,
                        dtype=dtype,
                    ),
                    None,
                ),
                lambda rank4, method_k=cumulant_k_max, module=known_estimation: module.initialization_flops(
                    n=args.n,
                    p=args.p,
                    k_max=method_k,
                    rank4=rank4,
                ),
            )
            return
        if args.input_distribution in {"gaussian_lowrank", "subspace_gaussian"}:
            return

        if args.sample_cumulant_estimator in {"source_compressed", "both"}:
            yield (
                "sample_avg",
                lambda method_k=cumulant_k_max, count=sample_count, seed_=seed: ica_known_estimation.sample_average_input_cumulants(
                    data_generator=data_generator,
                    m=count,
                    seed=seed_,
                    k_max=method_k,
                    device=device,
                    dtype=dtype,
                    eig_tol=args.sample_fourth_eig_tol,
                ),
                lambda rank4, method_k=cumulant_k_max, count=sample_count: ica_known_estimation.sample_initialization_flops(
                    n=args.n,
                    p=args.p,
                    sample_count=count,
                    k_max=method_k,
                    rank4=rank4,
                ),
            )
        if args.sample_cumulant_estimator in {"unknown_a", "both"}:
            yield (
                "unknown_a",
                lambda method_k=cumulant_k_max, count=sample_count, seed_=seed, module=unknown_estimation: module.input_cumulants(
                    data_generator=data_generator,
                    m=count,
                    seed=seed_,
                    k_max=method_k,
                    device=device,
                    dtype=dtype,
                    eig_tol=args.sample_fourth_eig_tol,
                    gram_chunk_size=args.unknown_a_gram_chunk_size,
                ),
                lambda rank4, method_k=cumulant_k_max, count=sample_count, module=unknown_estimation: module.initialization_flops(
                    n=args.n,
                    p=args.p,
                    sample_count=count,
                    k_max=method_k,
                    rank4=rank4,
                ),
            )

    def fixed_initialization_flops(
        *,
        method_name: str,
        method_k: int,
        rank4: int | None,
    ) -> int:
        if method_name == "prior":
            return ica_known_estimation.prior_initialization_flops(
                n=args.n,
                p=args.p,
                k_max=method_k,
                rank4=rank4,
            )
        return known_estimation.initialization_flops(
            n=args.n,
            p=args.p,
            k_max=method_k,
            rank4=rank4,
        )

    for cumulant_k_max in args.cumulant_orders:
        for method, build_cumulants in builders:
            result = _timed_cumulant_result(
                args=args,
                model=model,
                true_mean=true_mean,
                device=device,
                dtype=dtype,
                method=method,
                cumulant_k_max=cumulant_k_max,
                sample_k=None,
                sample_count=None,
                build_cumulants=lambda method_k=cumulant_k_max, builder=build_cumulants: (
                    builder(method_k),
                    None,
                ),
                propagation_flop_cache=propagation_flop_cache,
                estimate_initialization_flops=(
                    lambda rank4, method_name=method, method_k=cumulant_k_max: fixed_initialization_flops(
                        method_name=method_name,
                        method_k=method_k,
                        rank4=rank4,
                    )
                ),
            )
            results.append(result)
            print(
                f"{result.label} log_sq_error={result.log_squared_error: .6f} "
                f"warmup={result.warmup_seconds:.2f}s run={result.elapsed_seconds:.2f}s",
                flush=True,
            )

    for cumulant_k_max in args.cumulant_orders:
        for sample_k in range(args.sample_cumulant_k_min, args.sample_cumulant_k_max + 1):
            sample_count = 2**sample_k
            seed = args.cumulant_sample_seed_base + sample_k * args.seed_stride
            for method, build_cumulants, estimate_initialization_flops in sample_cumulant_jobs(
                cumulant_k_max=cumulant_k_max,
                sample_k=sample_k,
                sample_count=sample_count,
                seed=seed,
            ):
                result = _timed_cumulant_result(
                    args=args,
                    model=model,
                    true_mean=true_mean,
                    device=device,
                    dtype=dtype,
                    method=method,
                    cumulant_k_max=cumulant_k_max,
                    sample_k=sample_k,
                    sample_count=sample_count,
                    build_cumulants=build_cumulants,
                    propagation_flop_cache=propagation_flop_cache,
                    estimate_initialization_flops=estimate_initialization_flops,
                )
                results.append(result)
                rank_text = "" if result.rank4 is None else f" rank4={result.rank4}"
                print(
                    f"{result.label} m=2^{sample_k} log_sq_error={result.log_squared_error: .6f} "
                    f"warmup={result.warmup_seconds:.2f}s run={result.elapsed_seconds:.2f}s{rank_text}",
                    flush=True,
                )
    return results


def summarize_results(run_results: list[RunResult]) -> list[SummaryResult]:
    by_k: dict[int, list[RunResult]] = {}
    for result in run_results:
        by_k.setdefault(result.k, []).append(result)

    summaries: list[SummaryResult] = []
    for k in sorted(by_k):
        results = by_k[k]
        squared_errors = [result.squared_error for result in results]
        log_squared_errors = [result.log_squared_error for result in results]
        elapsed_seconds = [result.elapsed_seconds for result in results]
        forward_seconds = [result.forward_seconds for result in results]
        forward_flops = results[0].forward_flops
        mean_forward_seconds = _mean(forward_seconds)
        summaries.append(
            SummaryResult(
                k=k,
                m=results[0].m,
                runs=len(results),
                mean_squared_error=_mean(squared_errors),
                std_squared_error=_sample_std(squared_errors),
                mean_log_squared_error=_mean(log_squared_errors),
                std_log_squared_error=_sample_std(log_squared_errors),
                mean_elapsed_seconds=_mean(elapsed_seconds),
                std_elapsed_seconds=_sample_std(elapsed_seconds),
                mean_forward_seconds=mean_forward_seconds,
                std_forward_seconds=_sample_std(forward_seconds),
                forward_flops=forward_flops,
                mean_forward_flops_per_second=forward_flops / mean_forward_seconds
                if mean_forward_seconds > 0.0
                else 0.0,
            )
        )
    return summaries


def run_experiment(
    args: argparse.Namespace,
) -> tuple[list[RunResult], list[SummaryResult], list[CumulantResult]]:
    device = torch.device(args.device)
    torch.manual_seed(args.mlp_seed)

    model = DeepReLUMLP(
        n=args.n,
        L=args.depth,
        device=device,
        dtype=torch.float32,
    )
    model.eval()

    data_generator = make_data_generator(
        args=args,
        device=device,
        dtype=torch.float32,
    )

    start = time.time()
    true_mean_result = stream_mlp_mean(
        model=model,
        data_generator=data_generator,
        total_samples=args.true_samples,
        batch_size=args.batch_size,
        seed_base=args.true_seed_base,
    )
    true_mean = true_mean_result.mean
    print(
        f"computed true mean from {args.true_samples} samples "
        f"in {time.time() - start:.2f}s "
        f"(forward={true_mean_result.forward_seconds:.2f}s)",
        flush=True,
    )

    cumulant_results = run_cumulant_experiments(
        args=args,
        model=model,
        data_generator=data_generator,
        true_mean=true_mean,
        device=device,
    )

    run_results: list[RunResult] = []
    for k in range(args.k_min, args.k_max + 1):
        m = 2**k
        for run in range(args.runs):
            seed_base = (
                args.estimate_seed_base
                + k * args.seed_stride
                + run * args.run_seed_stride
            )
            mean_estimate_result = stream_mlp_mean(
                model=model,
                data_generator=data_generator,
                total_samples=m,
                batch_size=args.batch_size,
                seed_base=seed_base,
            )
            mean_estimate = mean_estimate_result.mean
            squared_error = torch.sum((mean_estimate - true_mean) ** 2).item()
            run_results.append(
                RunResult(
                    k=k,
                    m=m,
                    run=run,
                    seed_base=seed_base,
                    squared_error=squared_error,
                    elapsed_seconds=mean_estimate_result.elapsed_seconds,
                    forward_seconds=mean_estimate_result.forward_seconds,
                    forward_flops=mean_estimate_result.forward_flops,
                )
            )
        summary = summarize_results([result for result in run_results if result.k == k])[0]
        print(
            f"k={k:2d} m={m:6d} "
            f"mean_log_sq_error={summary.mean_log_squared_error: .6f} "
            f"std={summary.std_log_squared_error: .6f} "
            f"forward={summary.mean_forward_seconds:.4f}s",
            flush=True,
        )

    return run_results, summarize_results(run_results), cumulant_results


def write_run_csv(results: list[RunResult], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "k",
                "m",
                "run",
                "seed_base",
                "squared_error",
                "log_squared_error",
                "elapsed_seconds",
                "forward_seconds",
                "non_forward_seconds",
                "forward_flops",
                "samples_per_forward_second",
                "forward_flops_per_second",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "k": result.k,
                    "m": result.m,
                    "run": result.run,
                    "seed_base": result.seed_base,
                    "squared_error": result.squared_error,
                    "log_squared_error": result.log_squared_error,
                    "elapsed_seconds": result.elapsed_seconds,
                    "forward_seconds": result.forward_seconds,
                    "non_forward_seconds": result.elapsed_seconds - result.forward_seconds,
                    "forward_flops": result.forward_flops,
                    "samples_per_forward_second": result.m / result.forward_seconds
                    if result.forward_seconds > 0.0
                    else "",
                    "forward_flops_per_second": result.forward_flops / result.forward_seconds
                    if result.forward_seconds > 0.0
                    else "",
                }
            )


def write_summary_csv(results: list[SummaryResult], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "k",
                "m",
                "runs",
                "mean_squared_error",
                "std_squared_error",
                "mean_log_squared_error",
                "std_log_squared_error",
                "mean_elapsed_seconds",
                "std_elapsed_seconds",
                "mean_forward_seconds",
                "std_forward_seconds",
                "forward_flops",
                "mean_samples_per_forward_second",
                "mean_forward_flops_per_second",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "k": result.k,
                    "m": result.m,
                    "runs": result.runs,
                    "mean_squared_error": result.mean_squared_error,
                    "std_squared_error": result.std_squared_error,
                    "mean_log_squared_error": result.mean_log_squared_error,
                    "std_log_squared_error": result.std_log_squared_error,
                    "mean_elapsed_seconds": result.mean_elapsed_seconds,
                    "std_elapsed_seconds": result.std_elapsed_seconds,
                    "mean_forward_seconds": result.mean_forward_seconds,
                    "std_forward_seconds": result.std_forward_seconds,
                    "forward_flops": result.forward_flops,
                    "mean_samples_per_forward_second": result.m / result.mean_forward_seconds
                    if result.mean_forward_seconds > 0.0
                    else "",
                    "mean_forward_flops_per_second": result.mean_forward_flops_per_second,
                }
            )


def write_sampling_timing_csv(results: list[RunResult], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "k",
                "m",
                "run",
                "seed_base",
                "elapsed_seconds",
                "forward_seconds",
                "non_forward_seconds",
                "forward_flops",
                "samples_per_forward_second",
                "forward_flops_per_second",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "k": result.k,
                    "m": result.m,
                    "run": result.run,
                    "seed_base": result.seed_base,
                    "elapsed_seconds": result.elapsed_seconds,
                    "forward_seconds": result.forward_seconds,
                    "non_forward_seconds": result.elapsed_seconds - result.forward_seconds,
                    "forward_flops": result.forward_flops,
                    "samples_per_forward_second": result.m / result.forward_seconds
                    if result.forward_seconds > 0.0
                    else "",
                    "forward_flops_per_second": result.forward_flops / result.forward_seconds
                    if result.forward_seconds > 0.0
                    else "",
                }
            )


def write_cumulant_csv(results: list[CumulantResult], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "propagation",
                "cumulant_k_max",
                "sample_k",
                "sample_count",
                "squared_error",
                "log_squared_error",
                "elapsed_seconds",
                "warmup_seconds",
                "initialization_flops",
                "propagation_flops",
                "total_flops",
                "flops_per_second",
                "rank4",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "method": result.method,
                    "propagation": result.propagation_label,
                    "cumulant_k_max": result.cumulant_k_max,
                    "sample_k": "" if result.sample_k is None else result.sample_k,
                    "sample_count": "" if result.sample_count is None else result.sample_count,
                    "squared_error": result.squared_error,
                    "log_squared_error": result.log_squared_error,
                    "elapsed_seconds": result.elapsed_seconds,
                    "warmup_seconds": result.warmup_seconds,
                    "initialization_flops": ""
                    if result.initialization_flops is None
                    else result.initialization_flops,
                    "propagation_flops": ""
                    if result.propagation_flops is None
                    else result.propagation_flops,
                    "total_flops": "" if result.total_flops is None else result.total_flops,
                    "flops_per_second": ""
                    if result.total_flops is None or result.elapsed_seconds <= 0.0
                    else result.total_flops / result.elapsed_seconds,
                    "rank4": "" if result.rank4 is None else result.rank4,
                }
            )


def _nice_ticks(min_value: float, max_value: float, count: int) -> list[float]:
    if math.isclose(min_value, max_value):
        return [min_value]
    step = (max_value - min_value) / max(count - 1, 1)
    return [min_value + i * step for i in range(count)]


def write_svg(
    results: list[SummaryResult],
    svg_path: Path,
    *,
    cumulant_results: list[CumulantResult] | None = None,
) -> None:
    svg_path.parent.mkdir(parents=True, exist_ok=True)

    xs = [float(result.k) for result in results]
    ys = [result.mean_log_squared_error for result in results]
    lower_ys = [
        result.mean_log_squared_error - result.std_log_squared_error
        for result in results
    ]
    upper_ys = [
        result.mean_log_squared_error + result.std_log_squared_error
        for result in results
    ]
    x_min, x_max = min(xs), max(xs)
    comparison_ys = [*lower_ys, *upper_ys]
    if cumulant_results is not None:
        comparison_ys.extend(result.log_squared_error for result in cumulant_results)
    y_min, y_max = min(comparison_ys), max(comparison_ys)
    y_padding = 0.08 * max(y_max - y_min, 1.0)
    y_min -= y_padding
    y_max += y_padding

    width, height = 960, 640
    margin_left, margin_right = 96, 36
    margin_top, margin_bottom = 68, 92
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def sx(x: float) -> float:
        if math.isclose(x_min, x_max):
            return margin_left + plot_width / 2
        return margin_left + (x - x_min) / (x_max - x_min) * plot_width

    def sy(y: float) -> float:
        if math.isclose(y_min, y_max):
            return margin_top + plot_height / 2
        return margin_top + (y_max - y) / (y_max - y_min) * plot_height

    points = [(sx(x), sy(y)) for x, y in zip(xs, ys)]
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    upper_points = [(sx(x), sy(y)) for x, y in zip(xs, upper_ys)]
    lower_points = [(sx(x), sy(y)) for x, y in zip(reversed(xs), reversed(lower_ys))]
    shade_points = " ".join(
        f"{x:.2f},{y:.2f}" for x, y in [*upper_points, *lower_points]
    )

    x_ticks = [float(k) for k in range(int(x_min), int(x_max) + 1)]
    y_ticks = _nice_ticks(y_min, y_max, 6)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="640" viewBox="0 0 960 640">',
        "<style>",
        "text { font-family: Arial, Helvetica, sans-serif; fill: #172033; }",
        ".title { font-size: 25px; font-weight: 700; }",
        ".label { font-size: 16px; font-weight: 600; }",
        ".tick { font-size: 13px; fill: #526071; }",
        ".grid { stroke: #d8dee8; stroke-width: 1; }",
        ".axis { stroke: #172033; stroke-width: 1.4; }",
        ".shade { fill: #1f77b4; opacity: 0.18; }",
        ".line { fill: none; stroke: #1f77b4; stroke-width: 3; }",
        ".point { fill: #d62728; stroke: white; stroke-width: 1.5; }",
        ".legend { font-size: 14px; fill: #263447; }",
        "</style>",
        '<rect width="960" height="640" fill="#ffffff"/>',
        '<text class="title" x="480" y="38" text-anchor="middle">MLP Output Mean Concentration</text>',
        f'<line class="axis" x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}"/>',
        f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}"/>',
    ]

    for tick in x_ticks:
        x = sx(tick)
        parts.extend(
            [
                f'<line class="grid" x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}"/>',
                f'<line class="axis" x1="{x:.2f}" y1="{height - margin_bottom}" x2="{x:.2f}" y2="{height - margin_bottom + 6}"/>',
                f'<text class="tick" x="{x:.2f}" y="{height - margin_bottom + 25}" text-anchor="middle">{int(tick)}</text>',
            ]
        )

    for tick in y_ticks:
        y = sy(tick)
        parts.extend(
            [
                f'<line class="grid" x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}"/>',
                f'<line class="axis" x1="{margin_left - 6}" y1="{y:.2f}" x2="{margin_left}" y2="{y:.2f}"/>',
                f'<text class="tick" x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end">{tick:.2f}</text>',
            ]
        )

    parts.append(f'<polygon class="shade" points="{shade_points}"/>')
    parts.append(f'<polyline class="line" points="{polyline}"/>')
    for x, y in points:
        parts.append(f'<circle class="point" cx="{x:.2f}" cy="{y:.2f}" r="4.5"/>')
    legend_x, legend_y = 612, 76
    parts.extend(
        [
            f'<line class="line" x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 42}" y2="{legend_y}"/>',
            f'<text class="legend" x="{legend_x + 52}" y="{legend_y + 5}">sampling mean</text>',
        ]
    )
    if cumulant_results is not None:
        colors = ["#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#7f7f7f"]
        warmup_total = sum(result.warmup_seconds for result in cumulant_results)
        run_total = sum(result.elapsed_seconds for result in cumulant_results)
        parts.extend(
            [
                '<rect x="112" y="76" width="248" height="62" rx="4" fill="#ffffff" stroke="#b9c3d3" stroke-width="1"/>',
                '<text class="legend" x="124" y="98">Cumulant timing totals</text>',
                f'<text class="tick" x="124" y="118">warmup: {warmup_total:.2f}s</text>',
                f'<text class="tick" x="238" y="118">timed: {run_total:.2f}s</text>',
            ]
        )
        fixed_results = [result for result in cumulant_results if result.sample_k is None]
        sample_results = [result for result in cumulant_results if result.sample_k is not None]
        for index, result in enumerate(fixed_results):
            color = colors[index % len(colors)]
            y = sy(result.log_squared_error)
            legend_line_y = legend_y + 24 * (index + 1)
            parts.extend(
                [
                    f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" style="fill:none;stroke:{color};stroke-width:3;stroke-dasharray:10 7"/>',
                    f'<line x1="{legend_x}" y1="{legend_line_y}" x2="{legend_x + 42}" y2="{legend_line_y}" style="fill:none;stroke:{color};stroke-width:3;stroke-dasharray:10 7"/>',
                    f'<text class="legend" x="{legend_x + 52}" y="{legend_line_y + 5}">{result.label}</text>',
                ]
            )
        sample_groups: dict[int, list[CumulantResult]] = {}
        for result in sample_results:
            sample_groups.setdefault(result.cumulant_k_max, []).append(result)
        sample_start = len(fixed_results)
        for offset, cumulant_k_max in enumerate(sorted(sample_groups)):
            color = colors[(sample_start + offset) % len(colors)]
            group = sorted(sample_groups[cumulant_k_max], key=lambda result: result.sample_k or 0)
            group_points = [
                (sx(float(result.sample_k)), sy(result.log_squared_error))
                for result in group
                if result.sample_k is not None
            ]
            polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in group_points)
            legend_line_y = legend_y + 24 * (sample_start + offset + 1)
            parts.extend(
                [
                    f'<polyline points="{polyline}" style="fill:none;stroke:{color};stroke-width:3"/>',
                    f'<line x1="{legend_x}" y1="{legend_line_y}" x2="{legend_x + 42}" y2="{legend_line_y}" style="fill:none;stroke:{color};stroke-width:3"/>',
                    f'<text class="legend" x="{legend_x + 52}" y="{legend_line_y + 5}">sample avg {_propagation_label(cumulant_k_max)}</text>',
                ]
            )
            for x, y in group_points:
                parts.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.8" style="fill:{color};stroke:white;stroke-width:1.2"/>'
                )

    parts.extend(
        [
            '<text class="label" x="480" y="604" text-anchor="middle">k, where m = 2^k</text>',
            '<text class="label" transform="translate(28 320) rotate(-90)" text-anchor="middle">log squared error</text>',
            '<text class="tick" x="480" y="630" text-anchor="middle">natural logs; shaded band is +/- one standard deviation across sampling runs</text>',
            "</svg>",
        ]
    )
    svg_path.write_text("\n".join(parts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--p", type=int, default=32)
    parser.add_argument(
        "--input-distribution",
        choices=sorted(PROCESSES),
        default="ica",
    )
    parser.add_argument(
        "--parameter-estimation",
        choices=["auto", "known", "unknown"],
        default="auto",
        help="Estimator family for input cumulants. auto preserves legacy behavior; "
        "known uses the known_parameter_estimation module; unknown uses the "
        "unknown_parameter_estimation module with sample-dependent cumulants.",
    )
    parser.add_argument("--ica-seed", type=int, default=0)
    parser.add_argument("--gaussian-seed", type=int, default=0)
    parser.add_argument("--lowrank-seed", type=int, default=0)
    parser.add_argument("--subspace-seed", type=int, default=0)
    parser.add_argument("--mlp-seed", type=int, default=0)
    parser.add_argument("--true-samples", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=16)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--cumulant-orders",
        type=_parse_cumulant_orders,
        default=[1, 2, 3, 4],
        help="Comma-separated propagation budgets. K=1 is mean propagation, "
        "K=2 is covariance propagation, and higher K are cumulant propagation. "
        "Input odd cumulants are assumed zero in this ICA experiment.",
    )
    parser.add_argument("--cumulant-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--cumulant-factor", action="store_true")
    parser.add_argument(
        "--fixed-cumulant-methods",
        type=_parse_fixed_cumulant_methods,
        default=["prior", "known_distribution"],
        help="Comma-separated fixed ICA cumulant baselines: prior, known_distribution, or none.",
    )
    parser.add_argument("--sample-cumulant-k-min", type=int, default=1)
    parser.add_argument("--sample-cumulant-k-max", type=int, default=15)
    parser.add_argument(
        "--sample-cumulant-estimator",
        choices=["source_compressed", "unknown_a", "both"],
        default="source_compressed",
        help="ICA sample cumulant estimator. source_compressed keeps the legacy path that uses A/sources for compact K4 storage; unknown_a uses observations only.",
    )
    parser.add_argument("--cumulant-sample-seed-base", type=int, default=2_000_000_000)
    parser.add_argument("--sample-fourth-eig-tol", type=float, default=1e-6)
    parser.add_argument(
        "--unknown-a-gram-chunk-size",
        type=int,
        default=2048,
        help="Compatibility option for unknown-A estimators; current implementation avoids dense sample-Gram chunks.",
    )
    parser.add_argument(
        "--skip-flop-counts",
        action="store_true",
        help="Skip ARC propagation FLOP counting. CSV FLOP fields for cumulant propagation will be blank.",
    )
    parser.add_argument("--true-seed-base", type=int, default=10_000)
    parser.add_argument("--estimate-seed-base", type=int, default=1_000_000)
    parser.add_argument("--seed-stride", type=int, default=10_000)
    parser.add_argument("--run-seed-stride", type=int, default=1_000_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/mlp_mean_concentration"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={args.device}", flush=True)
    print(f"input_distribution={args.input_distribution}", flush=True)
    print(f"parameter_estimation={args.parameter_estimation}", flush=True)
    print(f"n={args.n} depth={args.depth} p={args.p}", flush=True)
    print(f"true_samples={args.true_samples} batch_size={args.batch_size}", flush=True)
    print(f"runs_per_k={args.runs}", flush=True)
    print(f"fixed_cumulant_methods={args.fixed_cumulant_methods}", flush=True)
    print(f"sample_cumulant_estimator={args.sample_cumulant_estimator}", flush=True)

    run_results, summary_results, cumulant_results = run_experiment(args)
    run_csv_path = args.output_dir / "run_results.csv"
    summary_csv_path = args.output_dir / "results.csv"
    sampling_timing_csv_path = args.output_dir / "sampling_timing.csv"
    cumulant_csv_path = args.output_dir / "cumulant_results.csv"
    svg_path = args.output_dir / "plot_log_error_vs_k.svg"
    write_run_csv(run_results, run_csv_path)
    write_summary_csv(summary_results, summary_csv_path)
    write_sampling_timing_csv(run_results, sampling_timing_csv_path)
    write_cumulant_csv(cumulant_results, cumulant_csv_path)
    write_svg(summary_results, svg_path, cumulant_results=cumulant_results)

    print(f"wrote {run_csv_path}", flush=True)
    print(f"wrote {summary_csv_path}", flush=True)
    print(f"wrote {sampling_timing_csv_path}", flush=True)
    print(f"wrote {cumulant_csv_path}", flush=True)
    print(f"wrote {svg_path}", flush=True)


if __name__ == "__main__":
    main()
