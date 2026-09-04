"""Multivariate cue->reward coupling (P2).

Single-channel Spearman coupling can miss signal that integrative strategies
exploit across cues jointly. This module tests whether a **vector** of channel
cues: sensory salience plus phylo/eco/bio neighbourhood fields: jointly
predicts reward beyond coverage and reward-depth confounds.

Statistic: rank-based **partial R²** (incremental variance explained by all
channel cues given optional coverage covariates). Significance via label
permutation on reward (block update: graph-based cue columns follow the permuted
reward, matching the univariate coupling null).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..hypercube import Manifold
from ..utils import get_logger
from ..utils.rng import RNG
from .coupling import _channel_weights
from .residual import depth_correlation, rank_residualize

log = get_logger("validation.multivariate")

CHANNEL_NAMES = ("sensory", "phylo", "eco", "bio")


def build_channel_cues(m: Manifold, reward: np.ndarray) -> np.ndarray:
    """Per-species cue vector aligned with univariate coupling channels.

    Columns: sensory salience, phylo-weighted neighbour reward,
    eco-weighted neighbour reward, bio-weighted neighbour reward.
    """
    reward = np.asarray(reward, dtype=float).ravel()
    salience = np.asarray(m.sensory_salience, dtype=float)
    weights = _channel_weights(m)
    return np.column_stack([
        salience,
        weights["phylo"] @ reward,
        weights["eco"] @ reward,
        weights["bio"] @ reward,
    ])


def _rank_columns(M: np.ndarray) -> list[np.ndarray]:
    """Rank-transform columns with non-zero variance."""
    M = np.asarray(M, dtype=float)
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    return [rankdata(M[:, k]) for k in range(M.shape[1]) if np.ptp(M[:, k]) > 0]


def rank_partial_r2(
    y: np.ndarray,
    X: np.ndarray,
    C: np.ndarray | None = None,
) -> float:
    """Incremental R² of rank(y) explained by rank(X) given rank(C).

    When ``C`` is None this reduces to the usual rank-based R². Values lie in
    [0, 1]; 0 when predictors carry no incremental signal.
    """
    y = np.asarray(y, dtype=float).ravel()
    if y.size < 4 or np.ptp(y) == 0:
        return 0.0
    ry = rankdata(y)
    cols_x = _rank_columns(X)
    if not cols_x:
        return 0.0

    ss_tot = float(np.sum((ry - ry.mean()) ** 2))
    if ss_tot <= 0:
        return 0.0

    if C is None:
        M = np.column_stack([np.ones(y.size)] + cols_x)
        resid = ry - M @ np.linalg.lstsq(M, ry, rcond=None)[0]
        return float(max(0.0, 1.0 - np.sum(resid ** 2) / ss_tot))

    cols_c = _rank_columns(C)
    M_red = np.column_stack([np.ones(y.size)] + cols_c)
    M_full = np.column_stack([np.ones(y.size)] + cols_c + cols_x)
    resid_red = ry - M_red @ np.linalg.lstsq(M_red, ry, rcond=None)[0]
    ss_red = float(np.sum(resid_red ** 2))
    if ss_red <= 0:
        return 0.0
    resid_full = ry - M_full @ np.linalg.lstsq(M_full, ry, rcond=None)[0]
    return float(max(0.0, 1.0 - np.sum(resid_full ** 2) / ss_red))


def _prepare_reward(
    m: Manifold,
    residual_reward: bool,
) -> tuple[np.ndarray, float | None]:
    """Return reward vector for coupling tests; optionally residualize on depth."""
    reward = np.asarray(m.reward, dtype=float)
    rho_pre: float | None = None
    if residual_reward:
        if m.reward_depth is None:
            raise ValueError(
                "residual-reward multivariate coupling requested but "
                "m.reward_depth is None"
            )
        depth = np.asarray(m.reward_depth, dtype=float)
        rho_pre = depth_correlation(reward, depth)
        reward = rank_residualize(reward, depth)
        log.info(
            "reward residualized on depth (pre-residual depth rho=%.3f)", rho_pre
        )
    return reward, rho_pre


def measure_multivariate_coupling(
    m: Manifold,
    rng: RNG,
    n_perm: int = 999,
    control: bool = False,
    residual_reward: bool = False,
) -> pd.DataFrame:
    """Joint coupling of the four channel cues -> reward with permutation null.

    Returns one row with partial R², null band, z-score, p-value, and metadata.
    """
    reward, _ = _prepare_reward(m, residual_reward)
    cov = None
    if control:
        if m.coverage is None:
            raise ValueError(
                "coverage-controlled multivariate coupling requested but "
                "m.coverage is None"
            )
        cov = np.asarray(m.coverage, dtype=float)

    cues = build_channel_cues(m, reward)
    n_pred = cues.shape[1]
    n_eff = m.n
    min_df = (cov.shape[1] if cov is not None else 0) + n_pred + 2
    if n_eff < min_df:
        log.warning(
            "multivariate coupling: n=%d below df guard (%d); returning null stat",
            n_eff, min_df,
        )
        return pd.DataFrame([{
            "method": "partial_r2",
            "channels": ",".join(CHANNEL_NAMES),
            "observed": 0.0,
            "null_mean": 0.0,
            "null_std": 0.0,
            "z": 0.0,
            "p_value": 1.0,
            "n_eff": n_eff,
            "n_predictors": n_pred,
        }])

    def _stat(r: np.ndarray) -> float:
        Z = build_channel_cues(m, r)
        return rank_partial_r2(r, Z, cov)

    observed = _stat(reward)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        null[i] = _stat(reward[rng.permutation(m.n)])

    null_mean, null_std = float(null.mean()), float(null.std(ddof=1))
    z = (observed - null_mean) / null_std if null_std > 0 else 0.0
    p = (1 + int(np.sum(null >= observed))) / (n_perm + 1)

    parts = []
    if residual_reward:
        parts.append("reward-residualized ")
    if control:
        parts.append("coverage-controlled ")
    log.info(
        "measured %smultivariate coupling (partial R²=%.4f, p=%.3f) "
        "over n=%d, %d permutations",
        "".join(parts), observed, p, m.n, n_perm,
    )

    return pd.DataFrame([{
        "method": "partial_r2",
        "channels": ",".join(CHANNEL_NAMES),
        "observed": float(observed),
        "null_mean": null_mean,
        "null_std": null_std,
        "z": float(z),
        "p_value": float(p),
        "n_eff": n_eff,
        "n_predictors": n_pred,
    }])


def measure_depth_stratified_multivariate(
    m: Manifold,
    rng: RNG,
    n_bins: int = 5,
    n_perm: int = 399,
    control: bool = False,
    residual_reward: bool = False,
) -> pd.DataFrame:
    """Per-effort-bin joint partial R² (four channel cues -> reward)."""
    from .controls import effort_index

    reward, _ = _prepare_reward(m, residual_reward)
    depth = effort_index(m)
    if np.ptp(depth) == 0:
        log.warning("depth-stratified multivariate: flat effort surface")
        return pd.DataFrame()

    edges = np.quantile(depth, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    bins = np.digitize(depth, edges[1:-1], right=False)
    vacuous_bins: set[int] = set()
    for b in range(n_bins):
        mask = bins == b
        if int(mask.sum()) >= 12 and np.ptp(reward[mask]) < 1e-6:
            vacuous_bins.add(b)

    cov_full = None
    if control:
        if m.coverage is None:
            raise ValueError("coverage-controlled multivariate requires m.coverage")
        cov_full = np.asarray(m.coverage, dtype=float)

    n_pred = len(CHANNEL_NAMES)
    rows = []
    for b in range(n_bins):
        valid = bins == b
        n_eff = int(valid.sum())
        depth_lo, depth_hi = float(edges[b]), float(edges[b + 1])
        bin_vacuous = b in vacuous_bins
        min_df = (cov_full.shape[1] if cov_full is not None else 0) + n_pred + 2
        if n_eff < min_df or bin_vacuous:
            rows.append({
                "bin": b, "depth_lo": depth_lo, "depth_hi": depth_hi,
                "method": "partial_r2", "observed": 0.0, "null_mean": 0.0,
                "null_std": 0.0, "p_value": 1.0, "n_eff": n_eff,
                "mean_reward": float(reward[valid].mean()) if n_eff else 0.0,
                "informative": False, "vacuous_bin": bin_vacuous,
            })
            continue

        idx = np.where(valid)[0]
        cov = cov_full[idx] if cov_full is not None else None

        def _stat(r_bin: np.ndarray) -> float:
            r_full = reward.copy()
            r_full[idx] = r_bin
            Z = build_channel_cues(m, r_full)[idx]
            return rank_partial_r2(r_full[idx], Z, cov)

        observed = _stat(reward[idx])
        null = np.empty(n_perm, dtype=float)
        for i in range(n_perm):
            perm = reward[idx[rng.permutation(len(idx))]]
            null[i] = _stat(perm)
        p = (1 + int(np.sum(null >= observed))) / (n_perm + 1)
        rows.append({
            "bin": b, "depth_lo": depth_lo, "depth_hi": depth_hi,
            "method": "partial_r2", "observed": float(observed),
            "null_mean": float(null.mean()),
            "null_std": float(null.std(ddof=1)),
            "p_value": float(p), "n_eff": n_eff,
            "mean_reward": float(reward[valid].mean()),
            "informative": True, "vacuous_bin": False,
        })

    out = pd.DataFrame(rows)
    informative = out[out["informative"]]
    sig = informative[informative["p_value"] < 0.05]
    log.info(
        "depth-stratified multivariate: %d/%d informative bins p<0.05 (n_bins=%d)",
        len(sig), len(informative), n_bins,
    )
    return out
