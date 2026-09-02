"""The search environment.

The environment is the only object that can reveal a species' hidden reward, and
only by paying for an "experiment" (the costly action representing actually
preparing and trying a plant). Everything else about a species -- its chemistry,
sensory salience, phylogenetic position, co-occurrence and animal associates --
is an observable cue that strategies read directly from the manifold.

This separation is the heart of the model: discovery is expensive, cues are
cheap, and the question is whether cue-structured search beats blind sampling.
"""

from __future__ import annotations

import numpy as np

from ..hypercube import Manifold
from ..utils.rng import RNG


class Environment:
    def __init__(self, manifold: Manifold, rng: RNG, observation_noise: float = 0.0):
        self.m = manifold
        self.rng = rng
        self.observation_noise = observation_noise
        self.tested = np.zeros(manifold.n, dtype=bool)
        self.observed_reward = np.full(manifold.n, np.nan, dtype=np.float32)
        self.n_tested = 0

    def reset(self) -> None:
        self.tested[:] = False
        self.observed_reward[:] = np.nan
        self.n_tested = 0

    def is_tested(self, idx: int) -> bool:
        return bool(self.tested[idx])

    def untested_indices(self) -> np.ndarray:
        return np.flatnonzero(~self.tested)

    def experiment(self, idx: int) -> float:
        """Pay to reveal the (noisy) reward of species ``idx``."""
        if not self.tested[idx]:
            self.tested[idx] = True
            self.n_tested += 1
        true_r = float(self.m.reward[idx])
        if self.observation_noise > 0:
            obs = true_r + self.rng.normal(0.0, self.observation_noise)
            obs = float(np.clip(obs, 0.0, 1.0))
        else:
            obs = true_r
        self.observed_reward[idx] = obs
        return obs
