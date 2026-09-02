"""Sensory heuristic search.

Hypothesis: plants advertise bioactivity through salient cues -- bitterness
(alkaloids), strong aroma (terpenes/essential oils), pungency (phenols). The
strategy prioritizes species by a fixed sensory-salience prior derived from
their chemistry. It needs no prior hits, modeling attention drawn by the senses,
but is an imperfect predictor by construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import ScoreStrategy
from ..hypercube import Manifold
from ..utils.rng import RNG

if TYPE_CHECKING:
    from ..simulation.environment import Environment


class SensorySearch(ScoreStrategy):
    def __init__(self, manifold: Manifold, rng: RNG, **params):
        super().__init__(manifold, rng, **params)
        sal = manifold.sensory_salience.astype(np.float64)
        denom = sal.max() if sal.max() > 0 else 1.0
        self._score = sal / denom

    def has_signal(self) -> bool:
        return True  # salience is available from the start

    def score(self, env: Environment) -> np.ndarray:
        return self._score
