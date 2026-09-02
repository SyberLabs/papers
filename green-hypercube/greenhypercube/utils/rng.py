"""Seeded random number generation.

Every stochastic component draws from an explicit ``numpy.random.Generator`` so
that an entire experiment is reproducible from a single integer seed. Child
generators are spawned deterministically so that, e.g., per-replicate streams do
not interfere with one another.
"""

from __future__ import annotations

import numpy as np

RNG = np.random.Generator


def make_rng(seed: int) -> RNG:
    """Create a PCG64-backed generator from an integer seed."""
    return np.random.default_rng(seed)


def spawn(rng: RNG, n: int) -> list[RNG]:
    """Spawn ``n`` independent child generators from ``rng``.

    Uses the parent's bit generator ``SeedSequence`` so children are
    statistically independent and reproducible.
    """
    seed_seq = rng.bit_generator.seed_seq  # type: ignore[attr-defined]
    return [np.random.default_rng(s) for s in seed_seq.spawn(n)]
