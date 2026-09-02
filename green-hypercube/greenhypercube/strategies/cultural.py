"""Culturally mediated attention allocation.

The integrative model. Culture is treated as an adaptive prior that allocates
attention across multiple cue systems at once -- phylogenetic, sensory, and
ecological. Each cue produces a normalized score; the strategy combines them
with weights that are updated online by multiplicative weights: cues that
recently pointed at genuine hits gain influence, cues that misfire lose it. This
captures cumulative cultural selection over *which heuristics to trust*, layered
on top of the heuristics themselves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import ScoreStrategy
from ..hypercube import Manifold
from ..utils.rng import RNG

if TYPE_CHECKING:
    from ..simulation.environment import Environment


class CulturalSearch(ScoreStrategy):
    def __init__(
        self,
        manifold: Manifold,
        rng: RNG,
        learning_rate: float = 0.5,
        phylo_scale: float | None = None,
        **params,
    ):
        super().__init__(manifold, rng, **params)
        self.lr = learning_rate
        self.sim = manifold.phylo_similarity(scale=phylo_scale)
        self.A = manifold.eco_adj + manifold.bio_adj
        sal = manifold.sensory_salience.astype(np.float64)
        self._sensory = sal / (sal.max() or 1.0)
        self.cue_names = ["phylo", "sensory", "ecological"]
        self.weights = np.ones(len(self.cue_names), dtype=np.float64)

    def has_signal(self) -> bool:
        return True  # sensory cue is always informative

    def _cue_scores(self) -> np.ndarray:
        """Return a (n, n_cues) matrix of per-cue normalized desirability."""
        n = self.m.n
        cues = np.zeros((n, 3), dtype=np.float64)
        if self.hits:
            hit_idx = np.array(self.hits, dtype=int)
            phylo = self.sim[:, hit_idx].max(axis=1)
            eco = self.A[:, hit_idx].sum(axis=1)
        else:
            phylo = np.zeros(n)
            eco = np.zeros(n)
        cues[:, 0] = _norm(phylo)
        cues[:, 1] = self._sensory
        cues[:, 2] = _norm(eco)
        return cues

    def score(self, env: Environment) -> np.ndarray:
        cues = self._cue_scores()
        w = self.weights / self.weights.sum()
        return cues @ w

    def observe(self, idx: int, reward: float) -> None:
        # Credit-assign to cues *before* updating the hit set, so we measure how
        # well each cue predicted this just-tested species.
        cues = self._cue_scores()
        cue_vals = cues[idx]
        super().observe(idx, reward)
        # Reward signal in [0,1]; multiplicative-weights update toward cues that
        # ranked this species highly when it paid off (and away when it didn't).
        signal = (reward - self.m.discovery_threshold)
        self.weights *= np.exp(self.lr * signal * cue_vals)
        self.weights = np.clip(self.weights, 1e-3, 1e3)


def _norm(v: np.ndarray) -> np.ndarray:
    m = v.max()
    return v / m if m > 0 else v
