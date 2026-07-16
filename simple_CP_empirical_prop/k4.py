"""Hybrid direct-rank K=4 ordinary cumulant propagation.

This module implements the reference K=4 path described by the retained-diagram
bundle. The state is hybrid: the mean is explicit, covariance is dense exactly
when ``M >= n`` and CP otherwise, while third and fourth cumulants are always
rank-``M`` CP tensors.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

import torch
from numpy.polynomial import Polynomial
from torch import Tensor

from .compression import compress_cp_sources
from .config import MeanVariance
from .hermite import ActivationWickSpec, activation_spec_from_name, polynomial_wick_spec
from .initialization import initialize_cumulant_cp
from .rng import RNGStreams
from .types import CPTensor, WeightedCPSource, zero_cp


DiagramInput = (
    str
    | Path
    | Sequence["RetainedDiagramSpec"]
    | Mapping[int, Sequence["RetainedDiagramSpec"]]
)

K4_FLOP_CONVENTION = (
    "Approximate analytic FLOPs for K=4 hybrid CP arithmetic. Counts explicit "
    "mean/covariance matvec and matmul work, CP factor matmuls, diagram factor "
    "products, explicit covariance materialization, CP compression arithmetic, "
    "and final mean reads. Excludes RNG, indexing, eigensolver internals, "
    "allocation, memory movement, and Python overhead."
)


def _fraction_changed(before: Tensor, after: Tensor) -> float:
    if before.numel() == 0:
        return 0.0
    return float((before != after).to(torch.float64).mean().item())


@dataclass(frozen=True)
class K4OrdinaryCPConfig:
    variance_min: float = 1e-10
    variance_max: float | None = None
    mean_abs_clip: float | None = None
    activate_final: bool = True
    reference_two_stage_nonlinear: bool = True
    explicit_stats_sample_count: int | None = None
    use_all_samples_for_explicit_stats: bool = False
    covariance_eig_tol: float = 0.0
    record_intermediate_states: bool = False
    validate_every_operation: bool = True
    polynomial_by_layer: tuple[Polynomial, ...] | None = None
    allow_nonpolynomial: bool = False
    hermite_degree_cap: int | None = None

    def __post_init__(self) -> None:
        if self.variance_min <= 0:
            raise ValueError("variance_min must be positive")
        if self.variance_max is not None and self.variance_max <= self.variance_min:
            raise ValueError("variance_max must exceed variance_min")
        if self.explicit_stats_sample_count is not None and self.explicit_stats_sample_count < 1:
            raise ValueError("explicit_stats_sample_count must be positive")
        if self.hermite_degree_cap is not None and self.hermite_degree_cap < 0:
            raise ValueError("hermite_degree_cap must be nonnegative")


@dataclass(frozen=True)
class RetainedDiagramSpec:
    diagram_id: str
    orbit_id: str
    output_order: int
    hermite_degrees: tuple[int, ...]
    blocks: tuple[tuple[int, ...], ...]
    coefficient: Fraction
    block_orders: tuple[int, ...]
    owners_by_block: tuple[tuple[int, ...], ...]
    source_notation: str
    notation: str
    table_keep_policy: str


@dataclass(frozen=True)
class CPBlockSource:
    order: int
    factors: tuple[Tensor, ...]
    label: str

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError("order must be positive")
        if len(self.factors) != self.order:
            raise ValueError("number of factors must equal order")
        first = self.factors[0]
        if first.ndim != 2:
            raise ValueError("factors must have shape [width, rank]")
        width, rank = first.shape
        if rank < 1:
            raise ValueError("source rank must be positive")
        for factor in self.factors:
            if tuple(factor.shape) != (width, rank):
                raise ValueError("all source factors must have identical [width, rank]")
            if factor.device != first.device or factor.dtype != first.dtype:
                raise ValueError("all source factors must share device and dtype")

    @property
    def width(self) -> int:
        return int(self.factors[0].shape[0])

    @property
    def rank(self) -> int:
        return int(self.factors[0].shape[1])

    @property
    def device(self) -> torch.device:
        return self.factors[0].device

    @property
    def dtype(self) -> torch.dtype:
        return self.factors[0].dtype

    def gather_columns(self, indices: Tensor) -> tuple[Tensor, ...]:
        idx = indices.to(self.device)
        return tuple(factor.index_select(1, idx) for factor in self.factors)


@dataclass(frozen=True)
class K4CumulantState:
    mean: Tensor
    covariance_dense: Tensor | None
    covariance_cp: CPTensor | None
    third: CPTensor
    fourth: CPTensor

    @property
    def width(self) -> int:
        return int(self.mean.shape[0])

    @property
    def covariance_mode(self) -> Literal["dense", "cp"]:
        return "dense" if self.covariance_dense is not None else "cp"


@dataclass
class K4OrdinaryCPDiagnostics:
    seed: int
    sample_count: int
    input_width: int
    rank: int
    covariance_mode: str
    activate_final: bool
    catalog_name: str = ""
    catalog_path: str | None = None
    catalog_hash: str = ""
    orbit_count: int | None = None
    labelled_count_by_order: dict[int, int] = field(default_factory=dict)
    evaluated_diagram_ids_by_layer: dict[int, list[str]] = field(default_factory=dict)
    zero_wick_diagram_ids_by_layer: dict[int, list[str]] = field(default_factory=dict)
    diagrams_by_layer_and_order: dict[tuple[int, int], int] = field(default_factory=dict)
    initialization_sources_by_order: dict[int, int] = field(default_factory=dict)
    unused_samples_by_order: dict[int, int] = field(default_factory=dict)
    mean_clip_fraction_by_layer: list[float] = field(default_factory=list)
    variance_clip_fraction_by_layer: list[float] = field(default_factory=list)
    raw_variance_min_by_layer: list[float] = field(default_factory=list)
    raw_variance_max_by_layer: list[float] = field(default_factory=list)
    covariance_symmetry_error_by_layer: list[float] = field(default_factory=list)
    total_analytic_flops: float = 0.0
    analytic_flops_by_stage: dict[str, float] = field(default_factory=dict)
    flop_count_convention: str = K4_FLOP_CONVENTION
    notes: list[str] = field(default_factory=list)

    def add_flops(self, label: str, flops: float) -> None:
        value = float(flops)
        self.analytic_flops_by_stage[label] = self.analytic_flops_by_stage.get(label, 0.0) + value
        self.total_analytic_flops += value


@dataclass
class K4OrdinaryCPResult:
    mean: Tensor
    final_state: K4CumulantState
    pre_states: list[K4CumulantState] | None
    act_states: list[K4CumulantState] | None
    diagnostics: K4OrdinaryCPDiagnostics


def _owners(block: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(vertex for vertex, multiplicity in enumerate(block) for _ in range(multiplicity))


def _canonical_blocks(blocks: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(tuple(int(x) for x in block) for block in blocks))


def _permute_block(block: tuple[int, ...], permutation: tuple[int, ...]) -> tuple[int, ...]:
    out = [0] * len(block)
    for old_vertex, multiplicity in enumerate(block):
        out[permutation[old_vertex]] = multiplicity
    return tuple(out)


def _expand_blocks(
    output_order: int,
    blocks: tuple[tuple[int, ...], ...],
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    members = {
        _canonical_blocks(tuple(_permute_block(block, permutation) for block in blocks))
        for permutation in itertools.permutations(range(output_order))
    }
    return tuple(sorted(members))


def _hermite_degrees(
    output_order: int,
    blocks: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    return tuple(sum(block[vertex] for block in blocks) for vertex in range(output_order))


def _notation(blocks: tuple[tuple[int, ...], ...]) -> str:
    if not blocks:
        return "empty"
    return " . ".join(
        "".join(str(vertex + 1) * multiplicity for vertex, multiplicity in enumerate(block))
        for block in blocks
    )


def _diagram_sort_key(spec: RetainedDiagramSpec) -> tuple[Any, ...]:
    return (
        spec.output_order,
        spec.orbit_id,
        spec.hermite_degrees,
        len(spec.blocks),
        spec.blocks,
        spec.diagram_id,
    )


def _spec_from_payload(row: Mapping[str, Any]) -> RetainedDiagramSpec:
    output_order = int(row["output_order"])
    blocks = _canonical_blocks(row["blocks"])
    coefficient = Fraction(
        int(row["coefficient"]["numerator"]),
        int(row["coefficient"]["denominator"]),
    )
    spec = RetainedDiagramSpec(
        diagram_id=str(row.get("id", row.get("diagram_id", ""))),
        orbit_id=str(row.get("orbit_id", row.get("id", ""))),
        output_order=output_order,
        hermite_degrees=tuple(int(x) for x in row.get("hermite_degrees", _hermite_degrees(output_order, blocks))),
        blocks=blocks,
        coefficient=coefficient,
        block_orders=tuple(int(x) for x in row.get("block_orders", tuple(sum(block) for block in blocks))),
        owners_by_block=tuple(tuple(int(v) for v in owners) for owners in row.get("owners_by_block", tuple(_owners(block) for block in blocks))),
        source_notation=str(row.get("source_notation", row.get("notation", ""))),
        notation=str(row.get("notation", _notation(blocks))),
        table_keep_policy=str(row.get("table_keep_policy", "custom")),
    )
    validate_retained_diagram(spec)
    return spec


def validate_retained_diagram(spec: RetainedDiagramSpec) -> None:
    if not 1 <= spec.output_order <= 4:
        raise ValueError("output_order must be in 1..4")
    if len(spec.hermite_degrees) != spec.output_order:
        raise ValueError(f"{spec.diagram_id}: hermite_degrees length mismatch")
    if spec.coefficient.denominator <= 0:
        raise ValueError(f"{spec.diagram_id}: coefficient denominator must be positive")
    if len(spec.blocks) != len(spec.block_orders) or len(spec.blocks) != len(spec.owners_by_block):
        raise ValueError(f"{spec.diagram_id}: block metadata length mismatch")
    expected_degrees = _hermite_degrees(spec.output_order, spec.blocks)
    if spec.hermite_degrees != expected_degrees:
        raise ValueError(f"{spec.diagram_id}: hermite_degrees do not match blocks")
    for idx, block in enumerate(spec.blocks):
        if len(block) != spec.output_order:
            raise ValueError(f"{spec.diagram_id}: block dimension mismatch")
        order = sum(block)
        if not 1 <= order <= 4:
            raise ValueError(f"{spec.diagram_id}: block order must be in 1..4")
        if spec.block_orders[idx] != order:
            raise ValueError(f"{spec.diagram_id}: block_orders do not match blocks")
        if spec.owners_by_block[idx] != _owners(block):
            raise ValueError(f"{spec.diagram_id}: owners_by_block do not match blocks")


def _load_json_from_path_or_resource(path: str | Path | None) -> tuple[dict[str, Any], str | None]:
    if path is None:
        resource = files("simple_CP_empirical_prop.diagram_lists").joinpath("k4_default_orbits.json")
        with resource.open("r", encoding="utf-8") as handle:
            return json.load(handle), str(resource)
    p = Path(path)
    with p.open("r", encoding="utf-8") as handle:
        return json.load(handle), str(p)


def _catalog_hash(specs: Sequence[RetainedDiagramSpec]) -> str:
    payload = [
        {
            "id": spec.diagram_id,
            "orbit_id": spec.orbit_id,
            "output_order": spec.output_order,
            "hermite_degrees": spec.hermite_degrees,
            "blocks": spec.blocks,
            "coefficient": [spec.coefficient.numerator, spec.coefficient.denominator],
        }
        for spec in sorted(specs, key=_diagram_sort_key)
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_k4_retained_diagrams(
    retained_diagrams: DiagramInput | None = None,
) -> tuple[dict[int, tuple[RetainedDiagramSpec, ...]], dict[str, Any]]:
    """Load K=4 retained diagrams and normalize to labelled specs by order."""
    source_path: str | None = None
    catalog_name = "python_supplied"
    orbit_count: int | None = None

    if retained_diagrams is None or isinstance(retained_diagrams, (str, Path)):
        payload, source_path = _load_json_from_path_or_resource(retained_diagrams)
        catalog_name = str(payload.get("name", "unnamed"))
        kind = payload.get("kind")
        specs: list[RetainedDiagramSpec] = []
        if payload.get("K") != 4:
            raise ValueError("expected a K=4 catalog")
        if kind == "orbit_catalog":
            orbits = payload.get("orbits", [])
            orbit_count = len(orbits)
            for orbit in orbits:
                if not orbit.get("enabled_by_default", True):
                    continue
                output_order = int(orbit["output_order"])
                representative = _canonical_blocks(orbit["blocks"])
                members = _expand_blocks(output_order, representative)
                expected = int(orbit.get("orbit_size", len(members)))
                if len(members) != expected:
                    raise ValueError(f"orbit-size mismatch for {orbit['id']}")
                for member_index, blocks in enumerate(members, start=1):
                    row = dict(orbit)
                    row["id"] = f"{orbit['id']}_member_{member_index:02d}"
                    row["orbit_id"] = orbit["id"]
                    row["blocks"] = blocks
                    row["source_notation"] = orbit["notation"]
                    row["notation"] = _notation(blocks)
                    row["hermite_degrees"] = _hermite_degrees(output_order, blocks)
                    row["block_orders"] = tuple(sum(block) for block in blocks)
                    row["owners_by_block"] = tuple(_owners(block) for block in blocks)
                    specs.append(_spec_from_payload(row))
        elif kind == "labelled_catalog":
            rows = payload.get("diagrams", [])
            orbit_count = len({str(row.get("orbit_id", row.get("id", ""))) for row in rows})
            specs = [_spec_from_payload(row) for row in rows if row.get("enabled_by_default", True)]
        else:
            raise ValueError("expected an orbit_catalog or labelled_catalog")
    elif isinstance(retained_diagrams, Mapping):
        specs = [
            spec
            for order in range(1, 5)
            for spec in retained_diagrams.get(order, ())
        ]
    else:
        specs = list(retained_diagrams)

    normalized: dict[int, tuple[RetainedDiagramSpec, ...]] = {}
    for order in range(1, 5):
        items = tuple(sorted((spec for spec in specs if spec.output_order == order), key=_diagram_sort_key))
        for spec in items:
            validate_retained_diagram(spec)
        normalized[order] = items

    all_specs = [spec for items in normalized.values() for spec in items]
    meta = {
        "catalog_name": catalog_name,
        "catalog_path": source_path,
        "orbit_count": orbit_count,
        "labelled_count_by_order": {order: len(normalized[order]) for order in range(1, 5)},
        "catalog_hash": _catalog_hash(all_specs),
    }
    return normalized, meta


def validate_k4_state(state: K4CumulantState, *, M: int) -> None:
    if state.mean.ndim != 1:
        raise ValueError("mean must have shape [n]")
    n = state.width
    if (state.covariance_dense is None) == (state.covariance_cp is None):
        raise ValueError("exactly one covariance representation must be present")
    if M >= n:
        if state.covariance_dense is None:
            raise ValueError("M >= n requires dense covariance")
        if tuple(state.covariance_dense.shape) != (n, n):
            raise ValueError("dense covariance must have shape [n, n]")
    else:
        if state.covariance_cp is None:
            raise ValueError("M < n requires CP covariance")
        state.covariance_cp.validate(expected_order=2, expected_rank=M, expected_width=n)
    state.third.validate(expected_order=3, expected_rank=M, expected_width=n)
    state.fourth.validate(expected_order=4, expected_rank=M, expected_width=n)


def _select_stats_subset(samples: Tensor, *, M: int, config: K4OrdinaryCPConfig, seed: int) -> Tensor:
    m = samples.shape[0]
    if config.use_all_samples_for_explicit_stats:
        count = m
    elif config.explicit_stats_sample_count is None:
        count = min(m, M)
    else:
        count = min(m, config.explicit_stats_sample_count)
    if count < 1:
        raise ValueError("at least one statistics sample is required")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(m, generator=gen, device="cpu")[:count].to(samples.device)
    return samples.index_select(0, indices)


def initialize_k4_state(
    samples: Tensor,
    *,
    M: int,
    config: K4OrdinaryCPConfig,
    seed: int,
    diagnostics: K4OrdinaryCPDiagnostics | None = None,
) -> K4CumulantState:
    if samples.ndim != 2:
        raise ValueError("samples must have shape [m, n]")
    if samples.shape[0] < 1:
        raise ValueError("at least one sample is required")
    if samples.shape[1] < 1:
        raise ValueError("width must be positive")
    if not isinstance(M, int) or M < 1:
        raise ValueError("rank M must be a positive integer")
    m, n = samples.shape
    subset = _select_stats_subset(samples, M=M, config=config, seed=seed)
    mean = subset.mean(dim=0)
    covariance_dense: Tensor | None = None
    covariance_cp: CPTensor | None = None
    if M >= n:
        if subset.shape[0] < 2:
            raise ValueError("dense covariance initialization requires at least two stats samples")
        centered = subset - mean
        covariance_dense = centered.transpose(0, 1) @ centered / (subset.shape[0] - 1)
        covariance_dense = 0.5 * (covariance_dense + covariance_dense.transpose(0, 1))
        if diagnostics is not None:
            diagnostics.add_flops(
                "initialization_explicit_mean_covariance",
                float(subset.shape[0] * n + n * n * (2 * subset.shape[0] - 1) + 2 * n * n),
            )
    streams = RNGStreams(seed, device="cpu")
    orders = [3, 4]
    if M < n:
        orders.insert(0, 2)
    cps: dict[int, CPTensor] = {}
    for order in orders:
        if m // order < 1:
            raise ValueError(f"need at least {order} samples to initialize order {order}")
        cp = initialize_cumulant_cp(
            samples,
            order=order,
            rank=M,
            group_generator=streams.generator("k4-init-group", order),
            compression_generator=streams.generator("k4-init-compress", order),
            shuffle=True,
        )
        cps[order] = cp
        if diagnostics is not None:
            diagnostics.initialization_sources_by_order[order] = int(m // order)
            diagnostics.unused_samples_by_order[order] = int(m - (m // order) * order)
            diagnostics.add_flops(f"initialization_cp_order_{order}", float(order * n * M))
    if M < n:
        covariance_cp = cps[2]
    state = K4CumulantState(
        mean=mean,
        covariance_dense=covariance_dense,
        covariance_cp=covariance_cp,
        third=cps[3],
        fourth=cps[4],
    )
    validate_k4_state(state, M=M)
    return state


def exact_covariance_factor_view(covariance: Tensor, *, eig_tol: float = 0.0) -> CPBlockSource:
    cov = 0.5 * (covariance + covariance.transpose(0, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    keep = eigenvalues.abs() > eig_tol
    eigenvalues = eigenvalues[keep]
    eigenvectors = eigenvectors[:, keep]
    rank = int(eigenvalues.numel())
    if rank == 0:
        zeros = cov.new_zeros((cov.shape[0], 1))
        return CPBlockSource(2, (zeros, zeros), "dense_covariance_zero")
    scale = torch.sqrt(torch.as_tensor(float(rank), device=cov.device, dtype=cov.dtype) * eigenvalues.abs())
    left = eigenvectors * scale[None, :]
    right = eigenvectors * (eigenvalues.sign() * scale)[None, :]
    return CPBlockSource(2, (left, right), "dense_covariance_eigh")


def _cp_source(cp: CPTensor, label: str) -> CPBlockSource:
    return CPBlockSource(cp.order, cp.factors, label)


def source_for_order(
    state: K4CumulantState,
    order: int,
    *,
    covariance_source: CPBlockSource | None = None,
) -> CPBlockSource:
    if order == 2:
        if state.covariance_dense is not None:
            if covariance_source is None:
                covariance_source = exact_covariance_factor_view(state.covariance_dense)
            return covariance_source
        if state.covariance_cp is None:
            raise ValueError("CP covariance source missing")
        return _cp_source(state.covariance_cp, "covariance_cp")
    if order == 3:
        return _cp_source(state.third, "third_cp")
    if order == 4:
        return _cp_source(state.fourth, "fourth_cp")
    raise KeyError(f"unsupported block order {order}")


def linear_k4_state(
    state: K4CumulantState,
    linear_or_weight: torch.nn.Linear | Tensor,
    bias: Tensor | None = None,
    *,
    M: int,
) -> K4CumulantState:
    if isinstance(linear_or_weight, torch.nn.Linear):
        weight = linear_or_weight.weight
        layer_bias = linear_or_weight.bias
    else:
        weight = linear_or_weight
        layer_bias = bias
    if weight.ndim != 2 or weight.shape[1] != state.width:
        raise ValueError("weight must have shape [output_width, input_width]")
    mean = weight @ state.mean
    if layer_bias is not None:
        mean = mean + layer_bias
    covariance_dense: Tensor | None = None
    covariance_cp: CPTensor | None = None
    if state.covariance_dense is not None:
        covariance_dense = weight @ state.covariance_dense @ weight.transpose(0, 1)
        covariance_dense = 0.5 * (covariance_dense + covariance_dense.transpose(0, 1))
    else:
        if state.covariance_cp is None:
            raise ValueError("CP covariance source missing")
        covariance_cp = CPTensor(tuple(weight @ factor for factor in state.covariance_cp.factors))
    out = K4CumulantState(
        mean=mean,
        covariance_dense=covariance_dense,
        covariance_cp=covariance_cp,
        third=CPTensor(tuple(weight @ factor for factor in state.third.factors)),
        fourth=CPTensor(tuple(weight @ factor for factor in state.fourth.factors)),
    )
    validate_k4_state(out, M=M)
    return out


def _linear_k4_flops(state: K4CumulantState, *, output_width: int, input_width: int, M: int, has_bias: bool) -> float:
    mean = output_width * (2 * input_width - 1)
    if has_bias:
        mean += output_width
    if state.covariance_dense is not None:
        covariance = output_width * input_width * (2 * input_width - 1)
        covariance += output_width * output_width * (2 * input_width - 1)
        covariance += 2 * output_width * output_width
    else:
        covariance = 2 * output_width * M * (2 * input_width - 1)
    third = 3 * output_width * M * (2 * input_width - 1)
    fourth = 4 * output_width * M * (2 * input_width - 1)
    return float(mean + covariance + third + fourth)


def mean_variance_from_k4_state(state: K4CumulantState, config: K4OrdinaryCPConfig) -> MeanVariance:
    raw_mean = state.mean
    if state.covariance_dense is not None:
        raw_variance = state.covariance_dense.diagonal()
    elif state.covariance_cp is not None:
        raw_variance = state.covariance_cp.diagonal_order2()
    else:
        raise ValueError("missing covariance representation")
    mean = raw_mean
    if config.mean_abs_clip is not None:
        mean = mean.clamp(-config.mean_abs_clip, config.mean_abs_clip)
    variance = raw_variance.clamp_min(config.variance_min)
    if config.variance_max is not None:
        variance = variance.clamp_max(config.variance_max)
    return MeanVariance(
        raw_mean=raw_mean,
        raw_variance=raw_variance,
        mean=mean,
        variance=variance,
        sigma=variance.sqrt(),
        mean_clip_fraction=_fraction_changed(raw_mean, mean),
        variance_clip_fraction=_fraction_changed(raw_variance, variance),
    )


def _required_degrees(catalog_by_order: Mapping[int, Sequence[RetainedDiagramSpec]]) -> tuple[int, ...]:
    degrees = {degree for specs in catalog_by_order.values() for spec in specs for degree in spec.hermite_degrees}
    return tuple(sorted(degrees))


def wick_vectors_for_catalog(
    activation: ActivationWickSpec,
    mean: Tensor,
    variance: Tensor,
    catalog_by_order: Mapping[int, Sequence[RetainedDiagramSpec]],
) -> dict[int, Tensor]:
    return {
        degree: activation.coefficient_fn(mean, variance, degree)
        for degree in _required_degrees(catalog_by_order)
    }


def _coefficient_tensor(coefficient: Fraction, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.as_tensor(
        float(coefficient.numerator) / float(coefficient.denominator),
        device=device,
        dtype=dtype,
    )


def _zero_wick(spec: RetainedDiagramSpec, vectors: Mapping[int, Tensor]) -> bool:
    return any(torch.count_nonzero(vectors[degree]).item() == 0 for degree in spec.hermite_degrees)


def sample_labelled_diagram_columns(
    spec: RetainedDiagramSpec,
    state: K4CumulantState,
    wick_vectors_by_degree: Mapping[int, Tensor],
    *,
    M: int,
    generator: torch.Generator,
    covariance_source: CPBlockSource | None,
) -> tuple[Tensor, ...]:
    n = state.width
    out = [
        wick_vectors_by_degree[degree][:, None].expand(n, M).clone()
        for degree in spec.hermite_degrees
    ]
    for block, owners_for_block in zip(spec.blocks, spec.owners_by_block):
        block_order = sum(block)
        source = source_for_order(
            state,
            block_order,
            covariance_source=covariance_source,
        )
        indices_cpu = torch.randint(
            low=0,
            high=source.rank,
            size=(M,),
            generator=generator,
            device="cpu",
        )
        indices = indices_cpu.to(source.device)
        for mode, owner in enumerate(owners_for_block):
            selected = source.factors[mode].index_select(1, indices)
            out[owner] = out[owner] * selected
    return tuple(out)


def _sample_diagram_cp(
    spec: RetainedDiagramSpec,
    state: K4CumulantState,
    wick_vectors_by_degree: Mapping[int, Tensor],
    *,
    M: int,
    generator: torch.Generator,
    covariance_source: CPBlockSource | None,
) -> CPTensor:
    return CPTensor(
        sample_labelled_diagram_columns(
            spec,
            state,
            wick_vectors_by_degree,
            M=M,
            generator=generator,
            covariance_source=covariance_source,
        )
    )


def _nonlinear_cp_order(
    order: int,
    specs: Sequence[RetainedDiagramSpec],
    state: K4CumulantState,
    wick_vectors_by_degree: Mapping[int, Tensor],
    *,
    M: int,
    streams: RNGStreams,
    layer_index: int,
    covariance_source: CPBlockSource | None,
    zero_wick_ids: list[str],
    evaluated_ids: list[str],
) -> CPTensor:
    weighted: list[WeightedCPSource] = []
    for spec_index, spec in enumerate(specs):
        if _zero_wick(spec, wick_vectors_by_degree):
            zero_wick_ids.append(spec.diagram_id)
            continue
        cp = _sample_diagram_cp(
            spec,
            state,
            wick_vectors_by_degree,
            M=M,
            generator=streams.generator("k4-diagram", layer_index, order, spec_index),
            covariance_source=covariance_source,
        )
        weighted.append(WeightedCPSource(spec.coefficient, cp))
        evaluated_ids.append(spec.diagram_id)
    if not weighted:
        return zero_cp(
            width=state.width,
            order=order,
            rank=M,
            device=state.mean.device,
            dtype=state.mean.dtype,
        )
    return compress_cp_sources(
        weighted,
        rank=M,
        generator=streams.generator("k4-final-compress", layer_index, order),
    )


def _diagram_leg_flops(spec: RetainedDiagramSpec, *, width: int, rank: int) -> float:
    return float(width * rank * sum(spec.block_orders))


def nonlinear_k4_state(
    state: K4CumulantState,
    mean_variance: MeanVariance,
    activation: ActivationWickSpec,
    *,
    catalog_by_order: Mapping[int, Sequence[RetainedDiagramSpec]],
    M: int,
    streams: RNGStreams,
    layer_index: int,
    config: K4OrdinaryCPConfig,
    diagnostics: K4OrdinaryCPDiagnostics | None = None,
) -> K4CumulantState:
    vectors = wick_vectors_for_catalog(
        activation,
        mean_variance.mean,
        mean_variance.variance,
        catalog_by_order,
    )
    covariance_source = (
        exact_covariance_factor_view(state.covariance_dense, eig_tol=config.covariance_eig_tol)
        if state.covariance_dense is not None
        else None
    )
    zero_wick_ids: list[str] = []
    evaluated_ids: list[str] = []

    new_mean = torch.zeros_like(state.mean)
    for spec_index, spec in enumerate(catalog_by_order[1]):
        if _zero_wick(spec, vectors):
            zero_wick_ids.append(spec.diagram_id)
            continue
        factors = sample_labelled_diagram_columns(
            spec,
            state,
            vectors,
            M=M,
            generator=streams.generator("k4-diagram", layer_index, 1, spec_index),
            covariance_source=covariance_source,
        )
        new_mean = new_mean + _coefficient_tensor(spec.coefficient, device=new_mean.device, dtype=new_mean.dtype) * factors[0].mean(dim=1)
        evaluated_ids.append(spec.diagram_id)

    covariance_dense: Tensor | None = None
    covariance_cp: CPTensor | None = None
    if state.covariance_dense is not None:
        covariance_dense = torch.zeros(
            state.width,
            state.width,
            device=state.mean.device,
            dtype=state.mean.dtype,
        )
        for spec_index, spec in enumerate(catalog_by_order[2]):
            if _zero_wick(spec, vectors):
                zero_wick_ids.append(spec.diagram_id)
                continue
            left, right = sample_labelled_diagram_columns(
                spec,
                state,
                vectors,
                M=M,
                generator=streams.generator("k4-diagram", layer_index, 2, spec_index),
                covariance_source=covariance_source,
            )
            coef = _coefficient_tensor(spec.coefficient, device=state.mean.device, dtype=state.mean.dtype)
            covariance_dense = covariance_dense + coef * (left @ right.transpose(0, 1) / M)
            evaluated_ids.append(spec.diagram_id)
        covariance_dense = 0.5 * (covariance_dense + covariance_dense.transpose(0, 1))
    else:
        covariance_cp = _nonlinear_cp_order(
            2,
            catalog_by_order[2],
            state,
            vectors,
            M=M,
            streams=streams,
            layer_index=layer_index,
            covariance_source=covariance_source,
            zero_wick_ids=zero_wick_ids,
            evaluated_ids=evaluated_ids,
        )

    third = _nonlinear_cp_order(
        3,
        catalog_by_order[3],
        state,
        vectors,
        M=M,
        streams=streams,
        layer_index=layer_index,
        covariance_source=covariance_source,
        zero_wick_ids=zero_wick_ids,
        evaluated_ids=evaluated_ids,
    )
    fourth = _nonlinear_cp_order(
        4,
        catalog_by_order[4],
        state,
        vectors,
        M=M,
        streams=streams,
        layer_index=layer_index,
        covariance_source=covariance_source,
        zero_wick_ids=zero_wick_ids,
        evaluated_ids=evaluated_ids,
    )
    if diagnostics is not None:
        diagnostics.evaluated_diagram_ids_by_layer[layer_index] = evaluated_ids
        diagnostics.zero_wick_diagram_ids_by_layer[layer_index] = zero_wick_ids
        for order in range(1, 5):
            diagnostics.diagrams_by_layer_and_order[(layer_index, order)] = len(catalog_by_order[order])
            leg_work = sum(
                _diagram_leg_flops(spec, width=state.width, rank=M)
                for spec in catalog_by_order[order]
            )
            diagnostics.add_flops(f"layer_{layer_index}_nonlinear_order_{order}_diagrams", leg_work)
        if state.covariance_dense is not None:
            diagnostics.add_flops(
                f"layer_{layer_index}_nonlinear_order_2_explicit_covariance",
                float(len(catalog_by_order[2]) * state.width * state.width * (2 * M - 1)),
            )
        else:
            for order in (2, 3, 4):
                diagnostics.add_flops(
                    f"layer_{layer_index}_nonlinear_order_{order}_compression",
                    float(len(catalog_by_order[order]) * state.width * M),
                )
    out = K4CumulantState(
        mean=new_mean,
        covariance_dense=covariance_dense,
        covariance_cp=covariance_cp,
        third=third,
        fourth=fourth,
    )
    validate_k4_state(out, M=M)
    return out


def _record_moments(
    diagnostics: K4OrdinaryCPDiagnostics,
    state: K4CumulantState,
    config: K4OrdinaryCPConfig,
) -> MeanVariance:
    mean_var = mean_variance_from_k4_state(state, config)
    diagnostics.mean_clip_fraction_by_layer.append(mean_var.mean_clip_fraction)
    diagnostics.variance_clip_fraction_by_layer.append(mean_var.variance_clip_fraction)
    diagnostics.raw_variance_min_by_layer.append(float(mean_var.raw_variance.min().item()))
    diagnostics.raw_variance_max_by_layer.append(float(mean_var.raw_variance.max().item()))
    if state.covariance_dense is not None:
        denom = state.covariance_dense.abs().max().clamp_min(config.variance_min)
        sym = (state.covariance_dense - state.covariance_dense.transpose(0, 1)).abs().max() / denom
        diagnostics.covariance_symmetry_error_by_layer.append(float(sym.item()))
    else:
        diagnostics.covariance_symmetry_error_by_layer.append(float("nan"))
    return mean_var


def propagate_k4_stages(
    stages: Sequence[tuple[torch.nn.Linear, ActivationWickSpec | None]],
    samples: Tensor,
    *,
    rank: int,
    retained_diagrams: DiagramInput | None = None,
    config: K4OrdinaryCPConfig | None = None,
    seed: int = 0,
    return_all: bool = False,
) -> K4OrdinaryCPResult:
    if samples.ndim != 2:
        raise ValueError("samples must have shape [m, n]")
    if samples.shape[0] < 1:
        raise ValueError("at least one sample is required")
    if samples.shape[1] < 1:
        raise ValueError("width must be positive")
    if not isinstance(rank, int) or rank < 1:
        raise ValueError("rank M must be a positive integer")
    if config is None:
        config = K4OrdinaryCPConfig()
    catalog_by_order, catalog_meta = load_k4_retained_diagrams(retained_diagrams)
    diagnostics = K4OrdinaryCPDiagnostics(
        seed=seed,
        sample_count=int(samples.shape[0]),
        input_width=int(samples.shape[1]),
        rank=rank,
        covariance_mode="dense" if rank >= samples.shape[1] else "cp",
        activate_final=config.activate_final,
        catalog_name=str(catalog_meta["catalog_name"]),
        catalog_path=catalog_meta["catalog_path"],
        catalog_hash=str(catalog_meta["catalog_hash"]),
        orbit_count=catalog_meta["orbit_count"],
        labelled_count_by_order=dict(catalog_meta["labelled_count_by_order"]),
        notes=[
            "K=4 hybrid path uses explicit mean and dense covariance iff M >= n.",
            "Runtime diagrams come from the supplied retained catalog.",
            "Nonlinear CP outputs use the reference two-stage compression path.",
        ],
    )
    state = initialize_k4_state(
        samples,
        M=rank,
        config=config,
        seed=seed,
        diagnostics=diagnostics,
    )
    streams = RNGStreams(seed, device="cpu")
    pre_states: list[K4CumulantState] | None = [] if return_all else None
    act_states: list[K4CumulantState] | None = [state] if return_all else None
    for layer_index, (linear, activation) in enumerate(stages):
        diagnostics.add_flops(
            f"layer_{layer_index}_linear",
            _linear_k4_flops(
                state,
                output_width=int(linear.weight.shape[0]),
                input_width=int(linear.weight.shape[1]),
                M=rank,
                has_bias=linear.bias is not None,
            ),
        )
        pre_state = linear_k4_state(state, linear, M=rank)
        if pre_states is not None:
            pre_states.append(pre_state)
        if activation is None:
            state = pre_state
            if act_states is not None:
                act_states.append(state)
            continue
        mean_var = _record_moments(diagnostics, pre_state, config)
        state = nonlinear_k4_state(
            pre_state,
            mean_var,
            activation,
            catalog_by_order=catalog_by_order,
            M=rank,
            streams=streams,
            layer_index=layer_index,
            config=config,
            diagnostics=diagnostics,
        )
        if act_states is not None:
            act_states.append(state)
    diagnostics.add_flops("final_mean", float(state.width))
    return K4OrdinaryCPResult(
        mean=state.mean,
        final_state=state,
        pre_states=pre_states,
        act_states=act_states,
        diagnostics=diagnostics,
    )


def _linears_and_names_from_mlp(mlp: object) -> tuple[list[torch.nn.Linear], list[str]]:
    if hasattr(mlp, "Ws"):
        linears = list(getattr(mlp, "Ws"))
        names_obj = getattr(mlp, "nonlin_names", [])
        if isinstance(names_obj, str):
            names = [names_obj] * len(linears)
        else:
            names = list(names_obj)
        return linears, names
    if isinstance(mlp, torch.nn.Sequential):
        linears: list[torch.nn.Linear] = []
        names: list[str] = []
        pending_linear = False
        for module in mlp:
            if isinstance(module, torch.nn.Linear):
                linears.append(module)
                pending_linear = True
            elif pending_linear:
                if isinstance(module, torch.nn.ReLU):
                    names.append("relu")
                else:
                    raise NotImplementedError(f"unsupported sequential activation {module.__class__.__name__}")
                pending_linear = False
            else:
                raise NotImplementedError(f"unsupported module {module.__class__.__name__}")
        return linears, names
    raise TypeError("mlp must expose Ws or be a torch.nn.Sequential")


def _activation_for_layer(
    layer_index: int,
    names: Sequence[str],
    config: K4OrdinaryCPConfig,
) -> ActivationWickSpec:
    if config.polynomial_by_layer is not None:
        if layer_index >= len(config.polynomial_by_layer):
            raise ValueError("not enough polynomials for activated layers")
        poly = config.polynomial_by_layer[layer_index]
        if not isinstance(poly, Polynomial):
            poly = Polynomial(poly)
        return polynomial_wick_spec(poly, name=f"poly_layer_{layer_index}")
    name = names[layer_index] if layer_index < len(names) else "square"
    return activation_spec_from_name(
        name,
        allow_nonpolynomial=config.allow_nonpolynomial,
        hermite_degree_cap=config.hermite_degree_cap,
    )


def ordinary_cp_mlp_k4(
    mlp: object,
    samples: Tensor,
    *,
    rank: int,
    retained_diagrams: DiagramInput | None = None,
    config: K4OrdinaryCPConfig | None = None,
    seed: int = 0,
    activate_final: bool | None = None,
    return_all: bool = False,
) -> K4OrdinaryCPResult:
    if config is None:
        config = K4OrdinaryCPConfig()
    if activate_final is not None:
        config = K4OrdinaryCPConfig(
            variance_min=config.variance_min,
            variance_max=config.variance_max,
            mean_abs_clip=config.mean_abs_clip,
            activate_final=activate_final,
            reference_two_stage_nonlinear=config.reference_two_stage_nonlinear,
            explicit_stats_sample_count=config.explicit_stats_sample_count,
            use_all_samples_for_explicit_stats=config.use_all_samples_for_explicit_stats,
            covariance_eig_tol=config.covariance_eig_tol,
            record_intermediate_states=config.record_intermediate_states,
            validate_every_operation=config.validate_every_operation,
            polynomial_by_layer=config.polynomial_by_layer,
            allow_nonpolynomial=config.allow_nonpolynomial,
            hermite_degree_cap=config.hermite_degree_cap,
        )
    linears, names = _linears_and_names_from_mlp(mlp)
    if not linears:
        raise ValueError("mlp must contain at least one linear layer")
    stages: list[tuple[torch.nn.Linear, ActivationWickSpec | None]] = []
    for layer_index, linear in enumerate(linears):
        is_last = layer_index == len(linears) - 1
        activation = None
        if config.activate_final or not is_last:
            activation = _activation_for_layer(layer_index, names, config)
        stages.append((linear, activation))
    return propagate_k4_stages(
        stages,
        samples,
        rank=rank,
        retained_diagrams=retained_diagrams,
        config=config,
        seed=seed,
        return_all=return_all,
    )


def exact_dense_diagram(
    spec: RetainedDiagramSpec,
    dense_cumulants: Mapping[int, Tensor],
    wick_vectors_by_degree: Mapping[int, Tensor],
) -> Tensor:
    """Dense oracle for one labelled diagram on tiny widths."""
    output_order = spec.output_order
    width = int(next(iter(wick_vectors_by_degree.values())).shape[0])
    out = torch.zeros((width,) * output_order, device=next(iter(wick_vectors_by_degree.values())).device, dtype=next(iter(wick_vectors_by_degree.values())).dtype)
    for visible in itertools.product(range(width), repeat=output_order):
        value = _coefficient_tensor(spec.coefficient, device=out.device, dtype=out.dtype)
        for vertex, degree in enumerate(spec.hermite_degrees):
            value = value * wick_vectors_by_degree[degree][visible[vertex]]
        for block, owners_for_block in zip(spec.blocks, spec.owners_by_block):
            tensor = dense_cumulants[sum(block)]
            idx = tuple(visible[owner] for owner in owners_for_block)
            value = value * tensor[idx]
        out[visible] = value
    return out
