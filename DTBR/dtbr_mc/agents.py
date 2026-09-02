"""Agent population sampling.

Each agent carries eight traits normalized to [0, 1]. Sampling supports
configurable marginals, Gaussian-copula correlations, and a configurable
upper-tail "explorer" minority.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dtbr_mc.config.schemas import AGENT_VARIABLE_NAMES, AgentConfig
from dtbr_mc.distributions import correlated_uniforms, quantile, sample_distribution

AGENT_VARIABLES: tuple[str, ...] = AGENT_VARIABLE_NAMES


class AgentSampler:
    """Samples agent populations from an :class:`AgentConfig`."""

    def __init__(self, config: AgentConfig):
        self.config = config

    def sample(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        """Return a DataFrame of ``n`` agents (columns = traits + ``is_explorer``)."""
        cols: dict[str, np.ndarray] = {}

        # 1. Independent marginal draws for every variable.
        for name in AGENT_VARIABLES:
            cols[name] = sample_distribution(self.config.variables[name], n, rng)

        # 2. Impose correlations via a Gaussian copula on the involved subset.
        if self.config.correlations:
            corr_vars = sorted(
                {p.var_a for p in self.config.correlations}
                | {p.var_b for p in self.config.correlations}
            )
            unsupported = {
                v
                for v in corr_vars
                if self.config.variables[v].kind in ("mixture",)
            }
            corr_vars = [v for v in corr_vars if v not in unsupported]
            if len(corr_vars) >= 2:
                idx = {v: i for i, v in enumerate(corr_vars)}
                d = len(corr_vars)
                R = np.eye(d)
                for p in self.config.correlations:
                    if p.var_a in idx and p.var_b in idx:
                        R[idx[p.var_a], idx[p.var_b]] = p.rho
                        R[idx[p.var_b], idx[p.var_a]] = p.rho
                u = correlated_uniforms(R, n, rng)
                for v in corr_vars:
                    cols[v] = quantile(self.config.variables[v], u[:, idx[v]])

        # 3. Explorer minority: redraw selected traits from the upper tail.
        ex = self.config.explorer
        n_explorer = int(round(ex.fraction * n))
        is_explorer = np.zeros(n, dtype=bool)
        if n_explorer > 0:
            explorer_idx = rng.choice(n, size=n_explorer, replace=False)
            is_explorer[explorer_idx] = True
            for v in ex.variables:
                if v not in cols:
                    continue
                lo, hi = self.config.variables[v].clip
                floor = lo + ex.lower_quantile * (hi - lo)
                cols[v][explorer_idx] = rng.uniform(floor, hi, size=n_explorer)

        df = pd.DataFrame(cols)[list(AGENT_VARIABLES)]
        df["is_explorer"] = is_explorer
        return df
