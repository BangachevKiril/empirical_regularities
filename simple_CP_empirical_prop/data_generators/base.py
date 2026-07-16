"""Shared return type for dataset generators."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return repr(value)


@dataclass
class GeneratedDataset:
    """Samples plus metadata/tensors that define the generating distribution."""

    name: str
    samples: Tensor
    metadata: dict[str, Any] = field(default_factory=dict)
    tensors: dict[str, Tensor] = field(default_factory=dict)

    def save(self, output_dir: str | Path) -> None:
        """Persist samples, recorded tensors, and a JSON metadata summary."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.samples.detach().cpu(), path / "samples.pt")
        torch.save(
            {key: value.detach().cpu() for key, value in self.tensors.items()},
            path / "recorded_tensors.pt",
        )
        tensor_shapes = {
            key: list(value.shape) for key, value in self.tensors.items()
        }
        payload = {
            "name": self.name,
            "sample_shape": list(self.samples.shape),
            "metadata": _jsonable(self.metadata),
            "recorded_tensor_shapes": tensor_shapes,
        }
        (path / "metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
