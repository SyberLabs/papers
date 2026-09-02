"""Episode results and aggregate metrics.

The primary observable is the *discovery curve*: cumulative count of genuine
useful species found as a function of the number of experiments spent. From the
per-experiment trajectory we derive interpretable summaries:

- discoveries          : useful species found within budget
- audc                 : area under the (normalized) discovery curve in [0,1];
                         higher = found useful plants sooner
- time_to_first        : experiments until the first discovery
- reward_coverage      : fraction of the manifold's total reward mass discovered
- final_regret         : oracle cumulative reward minus achieved, at budget
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class EpisodeResult:
    strategy: str
    order: list[int]                 # species indices in the order tested
    observed: list[float]            # observed reward per experiment
    true_reward: list[float]         # true reward per tested species
    is_useful: list[bool]            # whether each tested species was useful
    n_total_useful: int
    total_reward: float
    meta: dict = field(default_factory=dict)

    # --- derived curves ------------------------------------------------------
    def discovery_curve(self) -> np.ndarray:
        return np.cumsum(np.array(self.is_useful, dtype=float))

    def reward_curve(self) -> np.ndarray:
        return np.cumsum(np.array(self.true_reward, dtype=float))

    def audc(self) -> float:
        curve = self.discovery_curve()
        if self.n_total_useful == 0 or len(curve) == 0:
            return 0.0
        return float(curve.mean() / self.n_total_useful)

    def time_to_first(self) -> int:
        for t, u in enumerate(self.is_useful):
            if u:
                return t + 1
        return len(self.is_useful) + 1  # never found within budget

    def reward_coverage(self) -> float:
        if self.total_reward <= 0:
            return 0.0
        return float(sum(self.true_reward) / self.total_reward)

    def oracle_reward_curve(self) -> np.ndarray:
        budget = len(self.order)
        best = np.sort(np.array(self.true_reward_full))[::-1][:budget]
        return np.cumsum(best)

    # The full reward vector is needed for the oracle baseline.
    true_reward_full: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]

    def final_regret(self) -> float:
        if self.true_reward_full is None:
            return float("nan")
        oracle = self.oracle_reward_curve()
        achieved = self.reward_curve()
        if len(oracle) == 0:
            return 0.0
        return float(oracle[-1] - achieved[-1])

    def summary_row(self) -> dict:
        return {
            "strategy": self.strategy,
            "discoveries": int(self.discovery_curve()[-1]) if len(self.order) else 0,
            "n_total_useful": self.n_total_useful,
            "audc": self.audc(),
            "time_to_first": self.time_to_first(),
            "reward_coverage": self.reward_coverage(),
            "final_regret": self.final_regret(),
            "budget": len(self.order),
            **self.meta,
        }


def summarize(results: list[EpisodeResult]) -> pd.DataFrame:
    """Per-replicate summary table across many episodes."""
    return pd.DataFrame([r.summary_row() for r in results])


def aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std across replicates, grouped by strategy."""
    metrics = ["discoveries", "audc", "time_to_first", "reward_coverage", "final_regret"]
    g = summary.groupby("strategy")
    out = g[metrics].agg(["mean", "std"])
    out.columns = [f"{m}_{stat}" for m, stat in out.columns]
    return out.reset_index().sort_values("audc_mean", ascending=False)
