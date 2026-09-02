"""Phylogenetic generalization.

Hypothesis: useful properties cluster in clades, so once a useful plant is
found, its phylogenetic neighbors are promising. The score of a species is its
maximum phylogenetic similarity to any known hit. Before the first hit, the
strategy explores at random.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import ScoreStrategy
from ..hypercube import Manifold
from ..utils.rng import RNG

if TYPE_CHECKING:
    from ..simulation.environment import Environment


class PhylogeneticSearch(ScoreStrategy):
    def __init__(self, manifold: Manifold, rng: RNG, scale: float | None = None, **params):
        super().__init__(manifold, rng, **params)
        # Precompute phylogenetic similarity once; reused across the episode.
        self.sim = manifold.phylo_similarity(scale=scale)

    def has_signal(self) -> bool:
        return len(self.hits) > 0

    def score(self, env: Environment) -> np.ndarray:
        hit_idx = np.array(self.hits, dtype=int)
        # Max similarity to any known hit.
        return self.sim[:, hit_idx].max(axis=1)
