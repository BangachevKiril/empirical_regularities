from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def canonical_blocks(blocks: list[list[int]] | tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(tuple(int(x) for x in block) for block in blocks))


def permute_block(block: tuple[int, ...], permutation: tuple[int, ...]) -> tuple[int, ...]:
    out = [0] * len(block)
    for old_vertex, count in enumerate(block):
        out[permutation[old_vertex]] = count
    return tuple(out)


def expand_orbit(output_order: int, blocks: tuple[tuple[int, ...], ...]) -> tuple[tuple[tuple[int, ...], ...], ...]:
    return tuple(
        sorted(
            {
                canonical_blocks(
                    tuple(permute_block(block, permutation) for block in blocks)
                )
                for permutation in itertools.permutations(range(output_order))
            }
        )
    )


def expected_coefficient(blocks: tuple[tuple[int, ...], ...]) -> Fraction:
    denominator = 1
    for multiplicity in Counter(blocks).values():
        denominator *= math.factorial(multiplicity)
    for block in blocks:
        for count in block:
            denominator *= math.factorial(count)
    return Fraction(1, denominator)


def block_is_even(block: tuple[int, ...]) -> bool:
    return all(count % 2 == 0 for count in block)


def paper_score(blocks: tuple[tuple[int, ...], ...]) -> int:
    return sum(1 + int(block_is_even(block)) - sum(block) for block in blocks)


def is_mixed(blocks: tuple[tuple[int, ...], ...]) -> bool:
    for block in blocks:
        if sum(block) <= 2 and sum(count > 0 for count in block) <= 1:
            return False
    return True


def is_connected(output_order: int, blocks: tuple[tuple[int, ...], ...]) -> bool:
    if output_order == 1:
        return True
    if not blocks:
        return False
    adjacency = [set() for _ in range(output_order)]
    present = set()
    for block in blocks:
        support = [vertex for vertex, count in enumerate(block) if count]
        present.update(support)
        for vertex in support:
            adjacency[vertex].update(other for other in support if other != vertex)
    if present != set(range(output_order)):
        return False
    seen = {0}
    stack = [0]
    while stack:
        vertex = stack.pop()
        for other in adjacency[vertex]:
            if other not in seen:
                seen.add(other)
                stack.append(other)
    return len(seen) == output_order


def load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def validate() -> None:
    orbit_doc = load_json(HERE / 'k4_default_orbits.json')
    labelled_doc = load_json(HERE / 'k4_default_labelled.json')
    orbits = orbit_doc['orbits']
    labelled = labelled_doc['diagrams']

    assert orbit_doc['K'] == 4
    assert labelled_doc['K'] == 4
    assert len(orbits) == 26
    assert len(labelled) == 82

    orbit_counts: dict[int, int] = defaultdict(int)
    labelled_counts: dict[int, int] = defaultdict(int)
    expanded_keys = set()

    for orbit in orbits:
        output_order = int(orbit['output_order'])
        blocks = canonical_blocks(orbit['blocks'])
        assert all(len(block) == output_order for block in blocks)
        assert all(1 <= sum(block) <= 4 for block in blocks)
        assert is_mixed(blocks)
        assert is_connected(output_order, blocks)
        assert paper_score(blocks) >= -3
        assert orbit['paper_cost'] == 1 - paper_score(blocks)

        coefficient = Fraction(
            int(orbit['coefficient']['numerator']),
            int(orbit['coefficient']['denominator']),
        )
        assert coefficient == expected_coefficient(blocks)

        members = expand_orbit(output_order, blocks)
        assert len(members) == int(orbit['orbit_size'])
        manifest_members = tuple(
            sorted(canonical_blocks(member) for member in orbit['members'])
        )
        assert members == manifest_members
        orbit_counts[output_order] += 1

        for member in members:
            expanded_keys.add(
                (
                    output_order,
                    member,
                    coefficient.numerator,
                    coefficient.denominator,
                )
            )

    labelled_keys = set()
    for diagram in labelled:
        output_order = int(diagram['output_order'])
        blocks = canonical_blocks(diagram['blocks'])
        coefficient = Fraction(
            int(diagram['coefficient']['numerator']),
            int(diagram['coefficient']['denominator']),
        )
        labelled_keys.add(
            (
                output_order,
                blocks,
                coefficient.numerator,
                coefficient.denominator,
            )
        )
        labelled_counts[output_order] += 1

    assert expanded_keys == labelled_keys
    assert [orbit_counts[r] for r in range(1, 5)] == [3, 10, 9, 4]
    assert [labelled_counts[r] for r in range(1, 5)] == [3, 15, 35, 29]

    only_unfactorized = {
        orbit['notation']
        for orbit in orbits
        if orbit['table_keep_policy'] == 'only_unfactorized'
    }
    assert only_unfactorized == {'123 · 12', '12 · 234'}

    print('catalog validation passed')
    print('orbit counts:   ', {r: orbit_counts[r] for r in range(1, 5)})
    print('labelled counts:', {r: labelled_counts[r] for r in range(1, 5)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.parse_args()
    validate()
