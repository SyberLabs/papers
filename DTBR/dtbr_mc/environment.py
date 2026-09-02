"""Environment (repository / site) sampling.

Nine variables describe a plausible deep-time site configuration, all normalized
to [0, 1]. ``EnvironmentSampler`` supports both random sampling and the fixed
overrides used during controlled parameter sweeps.

Future environment *classes* (e.g. desert burial, deep borehole, surface
monument) can subclass :class:`EnvironmentSampler` and override :meth:`sample`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dtbr_mc.config.schemas import ENVIRONMENT_VARIABLE_NAMES, EnvironmentConfig
from dtbr_mc.distributions import sample_distribution

ENVIRONMENT_VARIABLES: tuple[str, ...] = ENVIRONMENT_VARIABLE_NAMES


class EnvironmentSampler:
    """Samples per-agent environment draws from an :class:`EnvironmentConfig`."""

    def __init__(self, config: EnvironmentConfig):
        self.config = config

    def sample(
        self,
        n: int,
        rng: np.random.Generator,
        overrides: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Return ``n`` environment rows.

        ``overrides`` pins named variables to a constant value across the whole
        population (used to sweep a control while holding everything else fixed).
        """
        overrides = overrides or {}
        cols: dict[str, np.ndarray] = {}
        for name in ENVIRONMENT_VARIABLES:
            if name in overrides:
                cols[name] = np.full(n, float(overrides[name]))
            else:
                cols[name] = sample_distribution(self.config.variables[name], n, rng)
        return pd.DataFrame(cols)[list(ENVIRONMENT_VARIABLES)]
