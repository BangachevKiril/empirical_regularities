"""CIFAR-10 image-data MLP mean-estimation experiment.

This is the CIFAR analogue of ``run_relu_mean_prop_experiment.py``:

* load the CIFAR-10 train split as RGB image vectors in ``R^(3 * 32^2)``;
* optionally apply one fixed Gaussian random projection to dimension ``n``;
* build a random Kaiming ReLU MLP of width ``n`` and depth ``L``;
* use the exact empirical CIFAR split mean after the MLP as the truth;
* compare direct sampling, K=1 mean propagation, and custom cov-prop.
"""

from __future__ import annotations

import sys

if sys.path:
    sys.path.pop(0)

import argparse
import json
import math
import os
import pickle
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import torch

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(PACKAGE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from simple_CP_empirical_prop.cov_prop import custom_cov_prop_stages
from simple_CP_empirical_prop.mean_prop import mean_prop_stages
from simple_CP_empirical_prop.run_relu_flop_experiment import (
    baseline_forward_flops,
    build_kaiming_relu_linears,
    rms_error,
    save_json,
)


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_DIRNAME = "cifar-10-batches-py"


def _svg_star(cx: float, cy: float, radius: float, color: str) -> str:
    points: list[str] = []
    inner = radius * 0.45
    for idx in range(10):
        angle = -math.pi / 2.0 + idx * math.pi / 5.0
        r = radius if idx % 2 == 0 else inner
        points.append(f"{cx + r * math.cos(angle):.2f},{cy + r * math.sin(angle):.2f}")
    return (
        f'<polygon points="{" ".join(points)}" fill="{color}" '
        f'stroke="{color}" stroke-width="1.2" />'
    )


def ensure_cifar10(root: Path, *, download: bool) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    dataset_dir = root / CIFAR10_DIRNAME
    if dataset_dir.exists():
        return dataset_dir
    if not download:
        raise FileNotFoundError(
            f"CIFAR-10 not found at {dataset_dir}; rerun with --download-cifar"
        )
    archive = root / "cifar-10-python.tar.gz"
    print(f"[cifar] downloading {CIFAR10_URL}", flush=True)
    urllib.request.urlretrieve(CIFAR10_URL, archive)
    print(f"[cifar] extracting {archive}", flush=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"download did not produce {dataset_dir}")
    return dataset_dir


def _load_cifar_batch(path: Path) -> torch.Tensor:
    with path.open("rb") as handle:
        batch = pickle.load(handle, encoding="latin1")
    data = torch.as_tensor(batch["data"], dtype=torch.float32)
    # CIFAR stores channel-major rows. This gives RGB PNG-equivalent HWC order.
    images = data.reshape(-1, 3, 32, 32).permute(0, 2, 3, 1).contiguous()
    return images.reshape(images.shape[0], -1) / 255.0


def load_cifar10_vectors(
    root: Path,
    *,
    split: str,
    download: bool,
) -> torch.Tensor:
    dataset_dir = ensure_cifar10(root, download=download)
    split_name = split.lower()
    if split_name == "train":
        batches = [_load_cifar_batch(dataset_dir / f"data_batch_{idx}") for idx in range(1, 6)]
        return torch.cat(batches, dim=0)
    if split_name in {"train_available", "available"}:
        paths = sorted(dataset_dir.glob("data_batch_*"))
        if not paths:
            raise FileNotFoundError(f"no data_batch_* files found in {dataset_dir}")
        print(
            "[cifar] using available train batches: "
            + ", ".join(path.name for path in paths),
            flush=True,
        )
        return torch.cat([_load_cifar_batch(path) for path in paths], dim=0)
    if split_name in {"test", "val", "validation"}:
        return _load_cifar_batch(dataset_dir / "test_batch")
    raise ValueError('split must be "train", "train_available", or "test"')


@torch.no_grad()
def project_pool(
    raw_vectors_cpu: torch.Tensor,
    projection: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    outs: list[torch.Tensor] = []
    for start in range(0, raw_vectors_cpu.shape[0], batch_size):
        batch = raw_vectors_cpu[start : start + batch_size].to(
            device=projection.device,
            dtype=projection.dtype,
        )
        outs.append(batch @ projection.transpose(0, 1))
    return torch.cat(outs, dim=0)


@torch.no_grad()
def forward_relu_means_from_pool(
    linears: list[torch.nn.Linear],
    pool: torch.Tensor,
    *,
    batch_size: int,
) -> list[torch.Tensor]:
    sums = [torch.zeros(linear.weight.shape[0], device=pool.device, dtype=torch.float64) for linear in linears]
    total = pool.shape[0]
    for start in range(0, total, batch_size):
        x = pool[start : start + batch_size]
        for layer, linear in enumerate(linears):
            x = torch.relu(linear(x))
            sums[layer] += x.to(torch.float64).sum(dim=0)
    return [(value / total).cpu() for value in sums]


@torch.no_grad()
def sample_pool_relu_means(
    linears: list[torch.nn.Linear],
    pool: torch.Tensor,
    *,
    sample_count: int,
    batch_size: int,
    seed: int,
) -> list[torch.Tensor]:
    gen = torch.Generator(device=pool.device).manual_seed(seed)
    sums = [torch.zeros(linear.weight.shape[0], device=pool.device, dtype=torch.float64) for linear in linears]
    done = 0
    start_time = time.time()
    while done < sample_count:
        current = min(batch_size, sample_count - done)
        indices = torch.randint(
            low=0,
            high=pool.shape[0],
            size=(current,),
            generator=gen,
            device=pool.device,
        )
        x = pool.index_select(0, indices)
        for layer, linear in enumerate(linears):
            x = torch.relu(linear(x))
            sums[layer] += x.to(torch.float64).sum(dim=0)
        done += current
        if done == sample_count or done % (batch_size * 8) == 0:
            elapsed = time.time() - start_time
            print(f"[sampling] {done}/{sample_count} samples in {elapsed:.1f}s", flush=True)
    return [(value / sample_count).cpu() for value in sums]


def sample_projected_pool(
    pool: torch.Tensor,
    *,
    sample_count: int,
    seed: int,
) -> torch.Tensor:
    gen = torch.Generator(device=pool.device).manual_seed(seed)
    indices = torch.randint(
        low=0,
        high=pool.shape[0],
        size=(sample_count,),
        generator=gen,
        device=pool.device,
    )
    return pool.index_select(0, indices).contiguous()


def write_svg_plot(path: Path, payload: dict[str, Any]) -> None:
    series = {
        "sampling": [
            (row["flops"], row["last_layer_rms_error"]) for row in payload["baseline"]
        ],
        "mean_prop K=1": [
            (row["flops"], row["last_layer_rms_error"]) for row in payload["mean_prop"]
        ],
        "cov_prop": [
            (row["flops"], row["last_layer_rms_error"]) for row in payload["cov_prop"]
        ],
    }
    points = [pt for pts in series.values() for pt in pts if pt[0] > 0 and pt[1] > 0]
    if not points:
        return
    xmin = min(math.log10(x) for x, _ in points)
    xmax = max(math.log10(x) for x, _ in points)
    ymin = min(math.log10(y) for _, y in points)
    ymax = max(math.log10(y) for _, y in points)
    xpad = 0.08 * (xmax - xmin if xmax > xmin else 1.0)
    ypad = 0.10 * (ymax - ymin if ymax > ymin else 1.0)
    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad

    width, height = 980, 640
    left, right, top, bottom = 95, 35, 35, 85
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(x: float) -> float:
        return left + (math.log10(x) - xmin) / (xmax - xmin) * plot_w

    def sy(y: float) -> float:
        return top + (ymax - math.log10(y)) / (ymax - ymin) * plot_h

    colors = {"sampling": "#111111", "mean_prop K=1": "#d62728", "cov_prop": "#1f77b4"}
    lines: list[str] = []
    markers: list[str] = []
    for name, pts in series.items():
        pts = sorted(pts)
        if not pts:
            continue
        coords = [(sx(x), sy(y)) for x, y in pts]
        point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
        lines.append(
            f'<polyline points="{point_text}" fill="none" stroke="{colors[name]}" stroke-width="2.5" />'
        )
        if name == "cov_prop":
            for row in sorted(payload["cov_prop"], key=lambda r: r["flops"]):
                x = sx(row["flops"])
                y = sy(row["last_layer_rms_error"])
                if row.get("mode") == "cp":
                    markers.append(_svg_star(x, y, 6.5, colors[name]))
                else:
                    markers.append(
                        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.4" fill="{colors[name]}" />'
                    )
        else:
            for x, y in coords:
                markers.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="{colors[name]}" />'
                )

    ticks: list[str] = []
    for val in [10**p for p in range(math.floor(xmin), math.ceil(xmax) + 1)]:
        x = sx(val)
        ticks.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#e4e4e4" />')
        ticks.append(f'<text x="{x:.2f}" y="{top + plot_h + 28}" text-anchor="middle">1e{int(round(math.log10(val)))}</text>')
    for val in [10**p for p in range(math.floor(ymin), math.ceil(ymax) + 1)]:
        y = sy(val)
        ticks.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e4e4e4" />')
        ticks.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">1e{int(round(math.log10(val)))}</text>')

    projection_label = (
        "no projection"
        if payload["config"].get("projection") == "none"
        else "random projection"
    )
    title = (
        f"CIFAR-10 {projection_label} n={payload['dataset']['projected_dimension']}, "
        f"ReLU MLP L={payload['config']['layers']}"
    )
    legend = """
<line x1="740" y1="62" x2="780" y2="62" stroke="#111111" stroke-width="3" />
<text x="790" y="67" font-size="14">sampling</text>
<line x1="740" y1="88" x2="780" y2="88" stroke="#d62728" stroke-width="3" />
<text x="790" y="93" font-size="14">mean_prop K=1</text>
<line x1="740" y1="114" x2="780" y2="114" stroke="#1f77b4" stroke-width="3" />
<polygon points="756.00,107.50 758.19,111.99 763.14,112.69 759.57,116.18 760.41,121.10 756.00,118.78 751.59,121.10 752.43,116.18 748.86,112.69 753.81,111.99" fill="#1f77b4" stroke="#1f77b4" stroke-width="1.2" />
<circle cx="777" cy="114" r="4.4" fill="#1f77b4" />
<text x="790" y="119" font-size="14">cov_prop (star = CP)</text>
"""
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<style>text {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #222; }}</style>
<text x="{left}" y="24" font-size="20" font-weight="700">{title}</text>
{''.join(ticks)}
<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#222" stroke-width="1.2" />
{''.join(lines)}
{''.join(markers)}
<text x="{left + plot_w / 2}" y="{height - 28}" font-size="16" text-anchor="middle">analytic FLOPs (log scale)</text>
<text x="24" y="{top + plot_h / 2}" font-size="16" transform="rotate(-90 24 {top + plot_h / 2})" text-anchor="middle">RMS error vs exact CIFAR-train truth (log scale)</text>
{legend}
</svg>
"""
    path.write_text(svg)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "cifar10_projected_mlp_results.json"
    svg_path = output_dir / "cifar10_projected_mlp.svg"
    projection_path = output_dir / "cifar_projection_W.pt"
    weights_path = output_dir / "mlp_weights.pt"

    print(f"[setup] device={device} dtype={dtype}", flush=True)
    print(f"[cifar] loading split={args.cifar_split}", flush=True)
    raw_vectors = load_cifar10_vectors(
        Path(args.cifar_root),
        split=args.cifar_split,
        download=args.download_cifar,
    )
    input_dim = raw_vectors.shape[1]
    if args.projection == "random":
        if args.n < 1:
            raise ValueError("random projection requires positive n")
        projection_gen = torch.Generator(device=device).manual_seed(args.projection_seed)
        projection = torch.randn(
            args.n,
            input_dim,
            generator=projection_gen,
            device=device,
            dtype=dtype,
        ) / math.sqrt(input_dim)
        torch.save(projection.detach().cpu(), projection_path)
        projected_pool = project_pool(
            raw_vectors,
            projection,
            batch_size=args.projection_batch_size,
        )
        recorded_projection_path: str | None = str(projection_path)
        projection_entry_variance: float | None = 1.0 / input_dim
    elif args.projection == "none":
        if args.n != input_dim:
            raise ValueError(
                f'projection="none" requires n={input_dim} for CIFAR-10 RGB vectors'
            )
        projected_pool = raw_vectors.to(device=device, dtype=dtype).contiguous()
        recorded_projection_path = None
        projection_entry_variance = None
    else:
        raise ValueError('projection must be "random" or "none"')
    del raw_vectors
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model_width = int(projected_pool.shape[1])

    linears = build_kaiming_relu_linears(
        n=model_width,
        layers=args.layers,
        device=device,
        dtype=dtype,
        seed=args.network_seed,
    )
    torch.save(
        [linear.weight.detach().cpu() for linear in linears],
        weights_path,
    )
    stages = [(linear, "relu") for linear in linears]
    payload: dict[str, Any] = {
        "config": vars(args),
        "device": str(device),
        "dtype": str(dtype),
        "baseline": [],
        "mean_prop": [],
        "cov_prop": [],
        "dataset": {
            "name": "CIFAR-10",
            "split": args.cifar_split,
            "pool_size": int(projected_pool.shape[0]),
            "raw_dimension": int(input_dim),
            "projection": args.projection,
            "projected_dimension": model_width,
            "projection_entry_variance": projection_entry_variance,
            "projection_path": recorded_projection_path,
            "mlp_weights_path": str(weights_path),
            "truth": "exact empirical mean over CIFAR-10 split after projection mode",
        },
        "notes": [
            "Sampling baselines sample CIFAR images with replacement from the finite split.",
            "Mean_prop and cov_prop initialize from CIFAR samples with replacement after the configured projection mode.",
            "FLOP x-axis counts estimator/network propagation, matching previous experiments; common CIFAR loading/projection preprocessing is recorded separately.",
        ],
    }

    print("[truth] exact empirical CIFAR split", flush=True)
    true_means = forward_relu_means_from_pool(
        linears,
        projected_pool,
        batch_size=args.forward_batch_size,
    )
    true_last = true_means[-1]
    payload["true_last_mean_first8"] = [float(x) for x in true_last[:8]]
    payload["true_layer_mean_norms"] = [float(mean.norm().item()) for mean in true_means]
    save_json(json_path, payload)

    for power in args.sample_powers:
        count = 2**power
        print(f"[sampling] m=2^{power}", flush=True)
        sample_means = sample_pool_relu_means(
            linears,
            projected_pool,
            sample_count=count,
            batch_size=min(2 ** args.forward_batch_power, count),
            seed=args.baseline_seed + power,
        )
        payload["baseline"].append(
            {
                "sample_power": power,
                "sample_count": count,
                "flops": baseline_forward_flops(
                    sample_count=count,
                    n=model_width,
                    layers=args.layers,
                ),
                "last_layer_rms_error": rms_error(sample_means[-1], true_last),
            }
        )

        print(f"[mean_prop] M=2^{power}", flush=True)
        samples = sample_projected_pool(
            projected_pool,
            sample_count=count,
            seed=args.mean_prop_seed + power,
        )
        with torch.no_grad():
            result = mean_prop_stages(stages, samples, variance_min=args.variance_min)
        payload["mean_prop"].append(
            {
                "sample_power": power,
                "sample_count": count,
                "flops": result.diagnostics.total_analytic_flops,
                "flop_breakdown": result.diagnostics.analytic_flops_by_stage,
                "last_layer_rms_error": rms_error(result.mean.detach().cpu(), true_last),
            }
        )
        del samples, result
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"[cov_prop] M=2^{power}", flush=True)
        cov_samples = sample_projected_pool(
            projected_pool,
            sample_count=count,
            seed=args.cov_prop_seed + power,
        )
        start = time.time()
        with torch.no_grad():
            cov_result = custom_cov_prop_stages(
                stages,
                cov_samples,
                rank=count,
                relu_degree_cap=args.cov_prop_relu_degree_cap,
                cp_seed=args.cov_prop_algorithm_seed + power,
                quadrature_degree=args.cov_prop_quadrature_degree,
                variance_min=args.variance_min,
            )
        elapsed = time.time() - start
        payload["cov_prop"].append(
            {
                "sample_power": power,
                "sample_count": count,
                "rank": count,
                "mode": cov_result.diagnostics.mode,
                "used_cp": cov_result.diagnostics.mode == "cp",
                "relu_degree_cap": args.cov_prop_relu_degree_cap,
                "quadrature_degree": args.cov_prop_quadrature_degree,
                "flops": cov_result.diagnostics.total_analytic_flops,
                "flop_breakdown": cov_result.diagnostics.analytic_flops_by_stage,
                "last_layer_rms_error": rms_error(cov_result.mean.detach().cpu(), true_last),
                "elapsed_seconds": elapsed,
            }
        )
        del cov_samples, cov_result
        if device.type == "cuda":
            torch.cuda.empty_cache()

        save_json(json_path, payload)
        write_svg_plot(svg_path, payload)

    save_json(json_path, payload)
    write_svg_plot(svg_path, payload)
    print(f"[done] wrote {json_path}", flush=True)
    print(f"[done] wrote {svg_path}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--layers", type=int, default=64)
    parser.add_argument("--projection", choices=["random", "none"], default="random")
    parser.add_argument("--sample-powers", type=int, nargs="+", default=[3, 6, 9, 12, 15, 18])
    parser.add_argument("--cifar-root", type=str, default="/home/shadeform/datasets")
    parser.add_argument("--cifar-split", type=str, default="train")
    parser.add_argument("--download-cifar", action="store_true", default=True)
    parser.add_argument("--no-download-cifar", dest="download_cifar", action="store_false")
    parser.add_argument("--projection-seed", type=int, default=80_000)
    parser.add_argument("--network-seed", type=int, default=123)
    parser.add_argument("--baseline-seed", type=int, default=20_000)
    parser.add_argument("--mean-prop-seed", type=int, default=50_000)
    parser.add_argument("--cov-prop-seed", type=int, default=60_000)
    parser.add_argument("--cov-prop-algorithm-seed", type=int, default=70_000)
    parser.add_argument("--cov-prop-relu-degree-cap", type=int, default=2)
    parser.add_argument("--cov-prop-quadrature-degree", type=int, default=40)
    parser.add_argument("--variance-min", type=float, default=1e-10)
    parser.add_argument("--projection-batch-size", type=int, default=8192)
    parser.add_argument("--forward-batch-power", type=int, default=15)
    parser.add_argument("--forward-batch-size", type=int, default=8192)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(PACKAGE_DIR) / "experiment_outputs" / "cifar10_projected_mlp_n256_l64"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
