"""Reference loader for JSON-retained diagram catalogs.

Codex should move the public dataclasses/functions into the package's normal
modules (for example ``diagrams.py`` or ``diagram_io.py``).  This file is
standalone so the data bundle can be validated before the repository patch is
applied.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from fractions import Fraction
from importlib.resources import files
from pathlib import Path
from typing import Iterable, Mapping, Sequence


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


def _canonical_blocks(blocks: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(tuple(int(x) for x in block) for block in blocks))


def _owners(block: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        vertex
        for vertex, multiplicity in enumerate(block)
        for _ in range(multiplicity)
    )


def _permute_block(
    block: tuple[int, ...],
    permutation: tuple[int, ...],
) -> tuple[int, ...]:
    out = [0] * len(block)
    for old_vertex, multiplicity in enumerate(block):
        out[permutation[old_vertex]] = multiplicity
    return tuple(out)


def _expand_blocks(
    output_order: int,
    blocks: tuple[tuple[int, ...], ...],
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    members = {
        _canonical_blocks(
            _permute_block(block, permutation)
            for block in blocks
        )
        for permutation in itertools.permutations(range(output_order))
    }
    return tuple(sorted(members))


def _hermite_degrees(
    output_order: int,
    blocks: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    return tuple(
        sum(block[vertex] for block in blocks)
        for vertex in range(output_order)
    )


def _notation(blocks: tuple[tuple[int, ...], ...]) -> str:
    if not blocks:
        return "∅"
    tokens = []
    for block in blocks:
        tokens.append(
            "".join(
                str(vertex + 1) * multiplicity
                for vertex, multiplicity in enumerate(block)
            )
        )
    return " · ".join(tokens)


def load_orbit_catalog(path: str | Path | None = None) -> tuple[RetainedDiagramSpec, ...]:
    """Load and expand the K=4 canonical orbit catalog.

    ``None`` loads ``diagram_lists/k4_default_orbits.json`` from package data.
    Every orbit member keeps the same vector-partition coefficient.  The
    function never multiplies a representative by ``orbit_size``.
    """

    if path is None:
        resource = files("simple_CP_empirical_prop.diagram_lists").joinpath(
            "k4_default_orbits.json"
        )
        with resource.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

    if payload.get("K") != 4 or payload.get("kind") != "orbit_catalog":
        raise ValueError("expected a K=4 orbit catalog")

    out: list[RetainedDiagramSpec] = []
    for orbit in payload["orbits"]:
        if not orbit.get("enabled_by_default", True):
            continue
        output_order = int(orbit["output_order"])
        representative = _canonical_blocks(orbit["blocks"])
        members = _expand_blocks(output_order, representative)
        expected_orbit_size = int(orbit.get("orbit_size", len(members)))
        if len(members) != expected_orbit_size:
            raise ValueError(
                f"orbit-size mismatch for {orbit['id']}: "
                f"computed {len(members)}, manifest {expected_orbit_size}"
            )
        coefficient = Fraction(
            int(orbit["coefficient"]["numerator"]),
            int(orbit["coefficient"]["denominator"]),
        )
        for member_index, blocks in enumerate(members, start=1):
            out.append(
                RetainedDiagramSpec(
                    diagram_id=f"{orbit['id']}_member_{member_index:02d}",
                    orbit_id=str(orbit["id"]),
                    output_order=output_order,
                    hermite_degrees=_hermite_degrees(output_order, blocks),
                    blocks=blocks,
                    coefficient=coefficient,
                    block_orders=tuple(sum(block) for block in blocks),
                    owners_by_block=tuple(_owners(block) for block in blocks),
                    source_notation=str(orbit["notation"]),
                    notation=_notation(blocks),
                    table_keep_policy=str(orbit["table_keep_policy"]),
                )
            )

    out.sort(
        key=lambda spec: (
            spec.output_order,
            spec.orbit_id,
            spec.hermite_degrees,
            spec.blocks,
        )
    )
    return tuple(out)


def group_by_output_order(
    specs: Sequence[RetainedDiagramSpec],
) -> Mapping[int, tuple[RetainedDiagramSpec, ...]]:
    return {
        output_order: tuple(
            spec for spec in specs if spec.output_order == output_order
        )
        for output_order in range(1, 5)
    }


if __name__ == "__main__":
    # Running this module from the source tree requires the repository parent
    # to be on PYTHONPATH.
    specs = load_orbit_catalog()
    grouped = group_by_output_order(specs)
    print({order: len(items) for order, items in grouped.items()})
