from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from cumulant_propagation._arc_mlp_kprop.harmonic import HTensor
from cumulant_propagation._arc_mlp_kprop.kprop_harmonic import (
    AUGMENT,
    BASE,
    OLD,
    SIMPLE,
    Kind,
    get_d_max,
    mlp_kprop,
)


@dataclass(frozen=True)
class _WeightOnlyLayer:
    weight: Tensor
    bias: None = None


@dataclass(frozen=True)
class _DeepReLUMLPAdapter:
    Ws: tuple[_WeightOnlyLayer, ...]
    nonlins: tuple[object, ...]
    nonlin_names: tuple[str, ...]
    init_scale: tuple[float | Tensor, ...]
    layernorm: bool = False


_KIND_BY_NAME = {
    "old": OLD,
    "simple": SIMPLE,
    "augment": AUGMENT,
    "base": BASE,
}


def _normalize_kind(kind: Kind | str) -> Kind:
    if isinstance(kind, Kind):
        return kind
    try:
        return _KIND_BY_NAME[kind.lower()]
    except KeyError as exc:
        valid = ", ".join(sorted(_KIND_BY_NAME))
        raise ValueError(f"kind must be one of {valid}; got {kind!r}") from exc


def _model_weights(model: Any) -> tuple[Tensor, ...]:
    if not hasattr(model, "weights"):
        raise TypeError("model must expose a .weights iterable like DeepReLUMLP.")
    weights = tuple(model.weights)
    if not weights:
        raise ValueError("model.weights must contain at least one weight matrix.")
    for index, weight in enumerate(weights):
        if weight.ndim != 2:
            raise ValueError(
                f"model.weights[{index}] must be a matrix; got {tuple(weight.shape)}."
            )
    return weights


def _weight_init_scale(model: Any, weight: Tensor) -> float | Tensor:
    if hasattr(model, "init_std"):
        return float(weight.shape[1]) * float(model.init_std) ** 2
    return weight.detach().pow(2).sum(dim=1).mean()


def _adapt_model(
    model: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> _DeepReLUMLPAdapter:
    weights = tuple(
        weight.detach().to(device=device, dtype=dtype) for weight in _model_weights(model)
    )
    if any(weight.shape[0] != weight.shape[1] for weight in weights):
        raise ValueError("DeepReLUMLP cumulant propagation currently expects square weight matrices.")
    n = weights[0].shape[0]
    if any(weight.shape != (n, n) for weight in weights):
        raise ValueError("all DeepReLUMLP weights must have the same square shape.")

    init_scale: list[float | Tensor] = []
    for weight in weights:
        scale = _weight_init_scale(model, weight)
        if isinstance(scale, Tensor):
            scale = scale.to(device=device, dtype=dtype)
        init_scale.append(scale)

    return _DeepReLUMLPAdapter(
        Ws=tuple(_WeightOnlyLayer(weight=weight) for weight in weights),
        nonlins=tuple(object() for _ in weights),
        nonlin_names=tuple("relu" for _ in weights),
        init_scale=tuple(init_scale),
    )


def _check_and_cast_cumulants(
    cumulants: Mapping[int, Tensor | HTensor],
    *,
    n: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, Tensor | HTensor]:
    if not cumulants:
        raise ValueError("cumulants must be a non-empty mapping from order to tensor.")
    if 1 not in cumulants or 2 not in cumulants:
        raise ValueError(
            "ReLU cumulant propagation requires caller-supplied first and second cumulants "
            "so the Wick expansion can use the initialized mean and variance."
        )

    cast: dict[int, Tensor | HTensor] = {}
    for order, value in cumulants.items():
        if not isinstance(order, int) or order <= 0:
            raise ValueError(f"cumulant orders must be positive integers; got {order!r}.")
        expected_shape = (n,) * order
        if isinstance(value, HTensor):
            if value.n != n:
                raise ValueError(f"K[{order}] has width {value.n}; expected {n}.")
            cast[order] = value.to(device=device, dtype=dtype)
            continue
        if hasattr(value, "contract_W") and hasattr(value, "n"):
            if value.n != n:
                raise ValueError(f"K[{order}] has width {value.n}; expected {n}.")
            if getattr(value, "d", order) != order:
                raise ValueError(f"K[{order}] reports order {value.d}; expected {order}.")
            cast[order] = value
            continue
        if not isinstance(value, Tensor):
            raise TypeError(f"K[{order}] must be a torch.Tensor or HTensor; got {type(value)!r}.")
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"K[{order}] must have shape {expected_shape}; got {tuple(value.shape)}.")
        cast[order] = value.to(device=device, dtype=dtype)
    return cast


def _as_tensor(value: Tensor | HTensor) -> Tensor:
    if isinstance(value, HTensor):
        return value.to_tensor()
    return value


def tower_to_tensors(tower: Mapping[int, Tensor | HTensor]) -> dict[int, Tensor]:
    """Convert one cumulant tower from harmonic tensors to ordinary PyTorch tensors."""
    return {order: _as_tensor(value) for order, value in tower.items()}


def cumulants_to_tensors(
    cumulants: Mapping[int, Tensor | HTensor] | Mapping[str, Mapping[int, Tensor | HTensor]],
) -> dict[int, Tensor] | dict[str, dict[int, Tensor]]:
    """Convert a final tower or an ``output_all=True`` layer mapping to PyTorch tensors."""
    if not cumulants:
        return {}
    first_value = next(iter(cumulants.values()))
    if isinstance(first_value, Mapping):
        return {
            layer: tower_to_tensors(tower) for layer, tower in cumulants.items()
        }  # type: ignore[arg-type]
    return tower_to_tensors(cumulants)  # type: ignore[arg-type]


def standard_gaussian_cumulants(
    n: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> dict[int, Tensor]:
    """Convenience initializer; propagation itself does not assume this form."""
    factory_kwargs = {"device": device, "dtype": dtype}
    return {
        1: torch.zeros(n, **factory_kwargs),
        2: torch.eye(n, **factory_kwargs),
    }


@torch.no_grad()
def propagate_cumulants(
    model: Any,
    cumulants: Mapping[int, Tensor | HTensor],
    *,
    k_max: int,
    kind: Kind | str = SIMPLE,
    use_avg_metric: bool = False,
    factor: bool = False,
    use_pK: bool = True,
    output_all: bool = False,
    up_to_layer: str | None = None,
    output_d_max: int | None = None,
    return_tensors: bool = False,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Mapping[int, Tensor | HTensor] | Mapping[str, Mapping[int, Tensor | HTensor]]:
    """Propagate arbitrary initialized cumulants through this repo's ``DeepReLUMLP``.

    ``cumulants`` is the initialization: provide any tensors you want for orders
    ``1, 2, ...``. Missing higher-order cumulants are treated as absent/zero by
    the ARC propagation routine; orders 1 and 2 are required because ReLU Wick
    coefficients are expanded around the initialized mean and variance.

    This repository's MLP applies ReLU after every weight matrix, so the final
    output layer is ``act{L-1}``, unlike the upstream ARC convention where the
    final linear layer has no activation.
    """
    if k_max <= 0:
        raise ValueError("k_max must be positive.")

    weights = _model_weights(model)
    default_device = weights[0].device
    default_dtype = weights[0].dtype
    resolved_device = torch.device(device) if device is not None else default_device
    resolved_dtype = dtype or default_dtype

    adapted_model = _adapt_model(model, device=resolved_device, dtype=resolved_dtype)
    n = adapted_model.Ws[0].weight.shape[0]
    cast_cumulants = _check_and_cast_cumulants(
        cumulants,
        n=n,
        device=resolved_device,
        dtype=resolved_dtype,
    )

    normalized_kind = _normalize_kind(kind)
    if output_d_max is None:
        output_d_max = get_d_max(k_max, normalized_kind)

    propagated = mlp_kprop(
        adapted_model,
        cast_cumulants,
        k_max=k_max,
        output_all=output_all,
        kind=normalized_kind,
        use_avg_metric=use_avg_metric,
        factor=factor,
        use_pK=use_pK,
        up_to_layer=up_to_layer,
        output_d_max=output_d_max,
    )

    if return_tensors:
        return cumulants_to_tensors(propagated)  # type: ignore[arg-type]
    return propagated


def _load_model(
    path: Path,
    *,
    n: int | None,
    depth: int | None,
    device: torch.device,
    allow_pickled_model: bool,
) -> Any:
    loaded = torch.load(path, map_location=device, weights_only=not allow_pickled_model)
    if hasattr(loaded, "weights"):
        return loaded
    if not isinstance(loaded, Mapping):
        raise TypeError("model file must contain a DeepReLUMLP object or a state_dict mapping.")
    if n is None or depth is None:
        raise ValueError("--n and --depth are required when --model-path is a state_dict.")
    from inference_models import DeepReLUMLP

    model = DeepReLUMLP(n=n, L=depth, device=device)
    model.load_state_dict(loaded)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Propagate cumulants through DeepReLUMLP.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cumulants-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--k-max", type=int, required=True)
    parser.add_argument("--kind", choices=sorted(_KIND_BY_NAME), default="simple")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-avg-metric", action="store_true")
    parser.add_argument("--output-all", action="store_true")
    parser.add_argument("--up-to-layer", default=None)
    parser.add_argument("--keep-harmonic", action="store_true")
    parser.add_argument(
        "--allow-pickled-model",
        action="store_true",
        help="Allow torch.load to unpickle a full model object instead of only loading tensors.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model = _load_model(
        args.model_path,
        n=args.n,
        depth=args.depth,
        device=device,
        allow_pickled_model=args.allow_pickled_model,
    )
    cumulants = torch.load(args.cumulants_path, map_location=device)

    output = propagate_cumulants(
        model,
        cumulants,
        k_max=args.k_max,
        kind=args.kind,
        use_avg_metric=args.use_avg_metric,
        output_all=args.output_all,
        up_to_layer=args.up_to_layer,
        return_tensors=not args.keep_harmonic,
        device=device,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output_path)


if __name__ == "__main__":
    main()
