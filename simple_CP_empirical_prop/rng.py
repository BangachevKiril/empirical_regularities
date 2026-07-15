"""Deterministic tagged RNG streams."""

from __future__ import annotations

import hashlib

import torch


def derive_seed(base_seed: int, *tags: object) -> int:
    """Derive a stable 63-bit seed from a base seed and structural tags."""
    payload = repr((int(base_seed), *tags)).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


class RNGStreams:
    """Factory for deterministic torch.Generator objects keyed by tags."""

    def __init__(self, base_seed: int, *, device: str | torch.device = "cpu") -> None:
        self.base_seed = int(base_seed)
        self.device = torch.device(device)

    def generator(self, *tags: object) -> torch.Generator:
        gen = torch.Generator(device=self.device)
        gen.manual_seed(derive_seed(self.base_seed, *tags))
        return gen
