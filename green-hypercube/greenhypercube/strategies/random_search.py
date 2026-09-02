"""Random search: the unstructured baseline.

Samples untested species uniformly at random. This is the "exhaustive search by
luck" null model against which every structured strategy is measured.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import ScoreStrategy

if TYPE_CHECKING:
    from ..simulation.environment import Environment


class RandomSearch(ScoreStrategy):
    def has_signal(self) -> bool:
        return False  # always explore -> uniform sampling

    def score(self, env: Environment) -> np.ndarray:  # pragma: no cover - unused
        return np.zeros(self.m.n)
