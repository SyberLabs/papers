"""Outcome metrics and bootstrap confidence intervals.

Point estimates are computed on the full sample. Confidence intervals use a
*Poisson bootstrap* (each agent's resample multiplicity ~ Poisson(1)), which is
a well-known scalable approximation to the n-out-of-n bootstrap and is fully
vectorizable -- essential at 100k agents.

Metric definitions
------------------
expected_harm                  mean( P_encounter * intervention * severity )
encounter_rate                 mean( P_encounter )
intervention_rate              fraction in {INVESTIGATE, EXCAVATE} (disturbance)
excavation_rate                fraction EXCAVATE
avoidance_rate                 fraction AVOID
preservation_rate              fraction PRESERVE
mean_hesitation_proxy          mean( caution )
mystery_to_curiosity_index     corr( mystery, curiosity_drive )  [+ => mystery feeds curiosity]
prestige_inversion_index       corr( phenomenological_caution, intervention )
                               [+ => louder caution markers RAISE intervention: a backfire]
behavioral_degradation_gradient  -slope( expected_harm ~ interpretive_capacity )
                               [+ => harm rises as interpretive capacity falls]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dtbr_mc.config.schemas import DISTURBANCE_STATES
from dtbr_mc.simulation import SimulationResult

# Metrics that are simple per-agent means (bootstrapped as weighted means).
_MEAN_METRICS = (
    "expected_harm",
    "encounter_rate",
    "intervention_rate",
    "excavation_rate",
    "avoidance_rate",
    "preservation_rate",
    "mean_hesitation_proxy",
)
# Metrics that are correlation/slope statistics (need weighted covariances).
_CORR_METRICS = (
    "mystery_to_curiosity_index",
    "prestige_inversion_index",
    "behavioral_degradation_gradient",
)


@dataclass
class _Arrays:
    """Per-agent vectors needed for all metrics."""

    expected_harm: np.ndarray
    p_encounter: np.ndarray
    is_disturbance: np.ndarray
    is_excavate: np.ndarray
    is_avoid: np.ndarray
    is_preserve: np.ndarray
    caution: np.ndarray
    mystery: np.ndarray
    curiosity_drive: np.ndarray
    phen_caution: np.ndarray
    intervention: np.ndarray
    interpretive_capacity: np.ndarray


def _extract(result: SimulationResult) -> _Arrays:
    o = result.outcomes
    return _Arrays(
        expected_harm=result.expected_harm,
        p_encounter=result.behavior.p_encounter,
        is_disturbance=np.isin(o, np.asarray(DISTURBANCE_STATES, dtype=object)).astype(float),
        is_excavate=(o == "EXCAVATE").astype(float),
        is_avoid=(o == "AVOID").astype(float),
        is_preserve=(o == "PRESERVE").astype(float),
        caution=result.behavior.caution,
        mystery=result.behavior.mystery,
        curiosity_drive=result.behavior.curiosity,
        phen_caution=result.environment["phenomenological_caution"].to_numpy(),
        intervention=result.behavior.intervention,
        interpretive_capacity=result.agents["interpretive_capacity"].to_numpy(),
    )


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    sx, sy = x.std(), y.std()
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    vx = x.var()
    if vx < 1e-12:
        return float("nan")
    return float(np.cov(x, y, bias=True)[0, 1] / vx)


def point_metrics(result: SimulationResult) -> dict[str, float]:
    a = _extract(result)
    return {
        "expected_harm": float(a.expected_harm.mean()),
        "encounter_rate": float(a.p_encounter.mean()),
        "intervention_rate": float(a.is_disturbance.mean()),
        "excavation_rate": float(a.is_excavate.mean()),
        "avoidance_rate": float(a.is_avoid.mean()),
        "preservation_rate": float(a.is_preserve.mean()),
        "mean_hesitation_proxy": float(a.caution.mean()),
        "mystery_to_curiosity_index": _corr(a.mystery, a.curiosity_drive),
        "prestige_inversion_index": _corr(a.phen_caution, a.intervention),
        "behavioral_degradation_gradient": -_slope(a.interpretive_capacity, a.expected_harm),
    }


def _weighted_means(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Weighted column means. w: (B, n), X: (n, k) -> (B, k)."""
    W = w.sum(axis=1, keepdims=True)
    return (w @ X) / W


def _bootstrap(
    result: SimulationResult, n_boot: int, ci: float, seed: int
) -> dict[str, tuple[float, float]]:
    a = _extract(result)
    n = a.expected_harm.shape[0]
    rng = np.random.default_rng(seed)

    # Stack the per-agent vectors for the mean metrics into one matrix.
    mean_cols = np.column_stack(
        [
            a.expected_harm,
            a.p_encounter,
            a.is_disturbance,
            a.is_excavate,
            a.is_avoid,
            a.is_preserve,
            a.caution,
        ]
    )

    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    mean_samples: list[np.ndarray] = []
    corr_samples = {k: [] for k in _CORR_METRICS}

    batch = max(1, min(64, n_boot))
    done = 0
    while done < n_boot:
        b = min(batch, n_boot - done)
        w = rng.poisson(1.0, size=(b, n)).astype(float)
        # guard against an all-zero replicate
        w[w.sum(axis=1) == 0, 0] = 1.0
        mean_samples.append(_weighted_means(w, mean_cols))

        W = w.sum(axis=1)
        def wmean(x):
            return (w @ x) / W

        # mystery_to_curiosity_index: corr(mystery, curiosity_drive)
        for name, x, y, kind in (
            ("mystery_to_curiosity_index", a.mystery, a.curiosity_drive, "corr"),
            ("prestige_inversion_index", a.phen_caution, a.intervention, "corr"),
            ("behavioral_degradation_gradient", a.interpretive_capacity, a.expected_harm, "slope"),
        ):
            mx, my = wmean(x), wmean(y)
            mxy = wmean(x * y)
            cov = mxy - mx * my
            vx = wmean(x * x) - mx * mx
            if kind == "corr":
                vy = wmean(y * y) - my * my
                denom = np.sqrt(np.clip(vx * vy, 0, None))
                val = np.where(denom > 1e-12, cov / np.where(denom > 1e-12, denom, 1), np.nan)
            else:  # slope -> degradation gradient is the negative slope
                val = np.where(vx > 1e-12, -cov / np.where(vx > 1e-12, vx, 1), np.nan)
            corr_samples[name].append(val)
        done += b

    mean_arr = np.vstack(mean_samples)  # (n_boot, 7)
    out: dict[str, tuple[float, float]] = {}
    for j, name in enumerate(_MEAN_METRICS):
        col = mean_arr[:, j]
        out[name] = (float(np.percentile(col, lo_q)), float(np.percentile(col, hi_q)))
    for name in _CORR_METRICS:
        col = np.concatenate(corr_samples[name])
        col = col[~np.isnan(col)]
        if col.size == 0:
            out[name] = (float("nan"), float("nan"))
        else:
            out[name] = (float(np.percentile(col, lo_q)), float(np.percentile(col, hi_q)))
    return out


def compute_metrics(
    result: SimulationResult,
    bootstrap_n: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a DataFrame indexed by metric with columns estimate / ci_lo / ci_hi."""
    point = point_metrics(result)
    if bootstrap_n and bootstrap_n > 0:
        cis = _bootstrap(result, bootstrap_n, ci, seed)
    else:
        cis = {k: (float("nan"), float("nan")) for k in point}
    rows = []
    for name, est in point.items():
        lo, hi = cis.get(name, (float("nan"), float("nan")))
        rows.append({"metric": name, "estimate": est, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows).set_index("metric")


__all__ = ["compute_metrics", "point_metrics"]
