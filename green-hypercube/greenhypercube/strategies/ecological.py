"""Ecological association learning.

Hypothesis: useful plants are revealed by ecological context -- they grow
alongside other useful plants (shared habitat) and share animal associates
(an animal eating a plant is a cue to its edibility/activity). The score of a
species is its summed association strength, through the co-occurrence and
animal-association graphs, to currently known hits. A light sensory prior breaks
the cold-start before the first hit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import ScoreStrategy
from ..hypercube import Manifold
from ..utils.rng import RNG

if TYPE_CHECKING:
    from ..simulation.environment import Environment


class EcologicalSearch(ScoreStrategy):
    def __init__(
        self,
        manifold: Manifold,
        rng: RNG,
        eco_weight: float = 1.0,
        bio_weight: float = 1.0,
        sensory_prior: float = 0.25,
        **params,
    ):
        super().__init__(manifold, rng, **params)
        self.A = eco_weight * manifold.eco_adj + bio_weight * manifold.bio_adj
        sal = manifold.sensory_salience.astype(np.float64)
        self.prior = sensory_prior * (sal / (sal.max() or 1.0))

    def has_signal(self) -> bool:
        # The sensory prior always provides a weak signal.
        return True

    def score(self, env: Environment) -> np.ndarray:
        s = self.prior.copy()
        if self.hits:
            hit_idx = np.array(self.hits, dtype=int)
            s = s + self.A[:, hit_idx].sum(axis=1)
        return s
