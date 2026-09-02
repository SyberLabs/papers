"""Strategy interfaces.

A :class:`Strategy` consumes the (cue-only) view of the manifold plus its own
history of revealed rewards and proposes which untested species to experiment on
next. The engine runs strategies under a fixed experiment budget.

Most strategies are single-agent and expressible as a *scoring rule*: given
current knowledge, assign every species a desirability score; propose the best
untested one (with optional epsilon exploration). :class:`ScoreStrategy`
captures this pattern. Multi-agent strategies (social transmission) override
:meth:`propose_batch` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..hypercube import Manifold
from ..utils.rng import RNG

if TYPE_CHECKING:  # avoid importing the simulation package at runtime (circular)
    from ..simulation.environment import Environment


class Strategy:
    """Base class. Subclasses must implement :meth:`propose_batch`."""

    #: Number of agents acting per round (experiments consumed per round).
    n_agents: int = 1

    def __init__(self, manifold: Manifold, rng: RNG, **params):
        self.m = manifold
        self.rng = rng
        self.params = params

    def reset(self) -> None:  # pragma: no cover - trivial default
        pass

    def propose_batch(self, env: Environment) -> list[int]:
        raise NotImplementedError

    def observe(self, idx: int, reward: float) -> None:  # pragma: no cover - default
        pass


class ScoreStrategy(Strategy):
    """Single-agent strategy defined by a desirability score over species.

    Tracks revealed rewards and the set of "hits" (species whose observed reward
    exceeds the discovery threshold), which most scoring rules build upon.
    """

    def __init__(self, manifold: Manifold, rng: RNG, epsilon: float = 0.05, **params):
        super().__init__(manifold, rng, **params)
        self.epsilon = epsilon
        self.reset()

    def reset(self) -> None:
        self.observed: dict[int, float] = {}
        self.hits: list[int] = []

    def observe(self, idx: int, reward: float) -> None:
        self.observed[idx] = reward
        if reward >= self.m.discovery_threshold:
            self.hits.append(idx)

    # --- scoring rule to be provided by subclasses ---------------------------
    def score(self, env: Environment) -> np.ndarray:
        """Return a length-n desirability vector (higher = test sooner)."""
        raise NotImplementedError

    def has_signal(self) -> bool:
        """Whether the scoring rule has anything to go on yet (else explore)."""
        return True

    def propose_batch(self, env: Environment) -> list[int]:
        untested = env.untested_indices()
        if untested.size == 0:
            return []
        # Epsilon-greedy exploration, and pure exploration before any signal.
        if (not self.has_signal()) or (self.rng.random() < self.epsilon):
            return [int(self.rng.choice(untested))]
        scores = self.score(env)
        masked = np.full(self.m.n, -np.inf, dtype=np.float64)
        masked[untested] = scores[untested]
        # Tiny noise breaks ties without changing ordering meaningfully.
        masked[untested] += self.rng.normal(0, 1e-6, size=untested.size)
        return [int(np.argmax(masked))]
