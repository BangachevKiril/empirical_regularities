"""Small combinatorial utilities for ordinary cumulants and diagrams."""

from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache
from fractions import Fraction
from itertools import product


def falling_factorial(n: int, k: int) -> int:
    out = 1
    for j in range(k):
        out *= n - j
    return out


@lru_cache(maxsize=None)
def set_partitions(n: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Return deterministic set partitions of {0, ..., n - 1}."""
    if n < 0:
        raise ValueError("n must be nonnegative")
    if n == 0:
        return ((),)
    parts: list[tuple[tuple[int, ...], ...]] = []
    for prev in set_partitions(n - 1):
        parts.append(tuple(sorted((*prev, (n - 1,)), key=lambda b: (len(b), b))))
        for block_index, block in enumerate(prev):
            new_blocks = list(prev)
            new_blocks[block_index] = tuple(sorted((*block, n - 1)))
            parts.append(tuple(sorted(new_blocks, key=lambda b: (len(b), b))))
    unique = sorted(set(parts), key=lambda p: (len(p), p))
    return tuple(unique)


def nonzero_vectors_leq(k_vec: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    ranges = [range(v + 1) for v in k_vec]
    vectors = [v for v in product(*ranges) if any(v)]
    return tuple(sorted(vectors, key=lambda v: (sum(v), v)))


@lru_cache(maxsize=None)
def vector_partitions(k_vec: tuple[int, ...]) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Return multisets of nonzero vectors that sum to k_vec."""
    if any(k < 0 for k in k_vec):
        raise ValueError("k_vec entries must be nonnegative")
    if not any(k_vec):
        return ((),)
    blocks = nonzero_vectors_leq(k_vec)

    def rec(remaining: tuple[int, ...], min_index: int) -> list[tuple[tuple[int, ...], ...]]:
        if not any(remaining):
            return [()]
        out: list[tuple[tuple[int, ...], ...]] = []
        for idx in range(min_index, len(blocks)):
            block = blocks[idx]
            if all(block[j] <= remaining[j] for j in range(len(k_vec))):
                new_remaining = tuple(remaining[j] - block[j] for j in range(len(k_vec)))
                for suffix in rec(new_remaining, idx):
                    out.append((block, *suffix))
        return out

    return tuple(rec(k_vec, 0))


def block_support(block: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(i for i, count in enumerate(block) if count)


def is_connected(part: tuple[tuple[int, ...], ...], *, d: int) -> bool:
    """Connectivity of the visible vertices induced by block supports."""
    if d == 1:
        return True
    if not part:
        return False
    parent = list(range(d))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    seen: set[int] = set()
    for block in part:
        support = block_support(block)
        seen.update(support)
        for vertex in support[1:]:
            union(support[0], vertex)
    if seen != set(range(d)):
        return False
    root = find(0)
    return all(find(v) == root for v in range(d))


def is_mixed(part: tuple[tuple[int, ...], ...], *, m: int = 2) -> bool:
    """Return true when every block touches at most m visible vertices."""
    return all(len(block_support(block)) <= m for block in part)


def vec_part_coef(
    part: tuple[tuple[int, ...], ...], *, divide_fac: bool = False
) -> int | Fraction:
    """Vector-partition fiber count.

    With divide_fac=False this returns c_vec, the number of labeled diagrams
    represented by the vector partition. With divide_fac=True it returns
    c_vec / prod_v k_v!.
    """
    if not part:
        return Fraction(1, 1) if divide_fac else 1
    d = len(part[0])
    k_vec = tuple(sum(block[v] for block in part) for v in range(d))
    numerator = math.prod(math.factorial(k) for k in k_vec)
    multiplicity = Counter(part)
    denominator = math.prod(math.factorial(count) for count in multiplicity.values())
    for block in part:
        denominator *= math.prod(math.factorial(x) for x in block)
    if divide_fac:
        return Fraction(1, denominator)
    return numerator // denominator


def vector_partition_weight(part: tuple[tuple[int, ...], ...]) -> Fraction:
    """Return 1 / (prod_u nu(u)! prod_{u in nu} prod_v u_v!)."""
    multiplicity = Counter(part)
    denominator = math.prod(math.factorial(count) for count in multiplicity.values())
    for block in part:
        denominator *= math.prod(math.factorial(x) for x in block)
    return Fraction(1, denominator)
