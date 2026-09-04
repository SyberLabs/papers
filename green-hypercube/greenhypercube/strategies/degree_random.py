"""Topology-weighted random search (M2 ladder rung 4).

Proposes untested species with probability proportional to co-occurrence +
animal-association graph degree: the structural bias toward well-connected
nodes, without sensory features or reward-linked hit propagation. Beating this
baseline is required before attributing strategy advantage to reward-linked
coupling; failing to beat it implicates pool geometry and base-rate heterogeneity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import ScoreStrategy

if TYPE_CHECKING:
    from ..simulation.environment import Environment


class DegreeRandomSearch(ScoreStrategy):
    def __init__(self, manifold, rng, **params):
        super().__init__(manifold, rng, **params)
        eco_d = np.fromiter(
            (manifold.eco.degree(i) for i in range(manifold.n)), dtype=np.float64
        )
        bio_d = np.fromiter(
            (manifold.bio.degree(i) for i in range(manifold.n)), dtype=np.float64
        )
        self.weights = eco_d + bio_d + 1.0

    def has_signal(self) -> bool:
        return True  # fixed topology weights; never uniform fallback

    def score(self, env: Environment) -> np.ndarray:
        return self.weights.copy()
