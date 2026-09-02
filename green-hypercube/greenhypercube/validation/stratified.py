"""Depth-stratified coupling (ChEMBL identifiability / over-correction check).

Within narrow effort bands, asks whether cue–reward association survives when
global study depth is held approximately constant. If coupling is null in every
band, depth carries the signal; if it survives at fixed depth, global
residualization may have over-corrected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..hypercube import Manifold
from ..utils import get_logger
from ..utils.rng import RNG
from .controls import effort_index
from .coupling import _channel_weights, _statistic
from .residual import rank_residualize

log = get_logger("validation.stratified")


def measure_depth_stratified_coupling(
    m: Manifold,
    rng: RNG,
    n_bins: int = 5,
    n_perm: int = 399,
    control: bool = False,
    residual_reward: bool = False,
) -> pd.DataFrame:
    """Per-effort-bin coupling for each channel (label permutation within bin)."""
    reward = np.asarray(m.reward, dtype=float)
    if residual_reward and m.reward_depth is not None:
        reward = rank_residualize(reward, np.asarray(m.reward_depth, float))
    depth = effort_index(m)
    if np.ptp(depth) == 0:
        log.warning("depth-stratified coupling: flat effort surface")
        return pd.DataFrame()

    edges = np.quantile(depth, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    bins = np.digitize(depth, edges[1:-1], right=False)
    vacuous_bins: set[int] = set()
    for b in range(n_bins):
        mask = bins == b
        if int(mask.sum()) >= 12 and np.ptp(reward[mask]) < 1e-6:
            vacuous_bins.add(b)
    salience = np.asarray(m.sensory_salience, dtype=float)
    weights = _channel_weights(m)
    cov = np.asarray(m.coverage, dtype=float) if control and m.coverage is not None else None

    rows = []
    for b in range(n_bins):
        valid_bin = bins == b
        n_eff = int(valid_bin.sum())
        depth_lo, depth_hi = float(edges[b]), float(edges[b + 1])
        bin_vacuous = b in vacuous_bins
        if n_eff < 12:
            for channel in weights:
                rows.append({
                    "bin": b, "depth_lo": depth_lo, "depth_hi": depth_hi,
                    "channel": channel, "observed": 0.0, "null_mean": 0.0,
                    "null_std": 0.0, "p_value": 1.0, "n_eff": n_eff,
                    "mean_reward": float(reward[valid_bin].mean()) if n_eff else 0.0,
                    "informative": False,
                    "vacuous_bin": bin_vacuous,
                })
            continue

        for channel, W in weights.items():
            if channel == "sensory":
                valid = valid_bin.copy()
            else:
                valid = valid_bin & (W.sum(axis=1) > 0)
            n_ch = int(valid.sum())
            if n_ch < 12:
                rows.append({
                    "bin": b, "depth_lo": depth_lo, "depth_hi": depth_hi,
                    "channel": channel, "observed": 0.0, "null_mean": 0.0,
                    "null_std": 0.0, "p_value": 1.0, "n_eff": n_ch,
                    "mean_reward": float(reward[valid_bin].mean()),
                    "informative": False,
                    "vacuous_bin": bin_vacuous,
                })
                continue

            observed = _statistic(channel, W, salience, reward, valid, cov)
            null = np.empty(n_perm, dtype=float)
            idx = np.where(valid)[0]
            for i in range(n_perm):
                perm_r = reward.copy()
                perm_r[idx] = reward[idx[rng.permutation(len(idx))]]
                null[i] = _statistic(channel, W, salience, perm_r, valid, cov)
            p = (1 + int(np.sum(np.abs(null) >= abs(observed)))) / (n_perm + 1)
            rows.append({
                "bin": b, "depth_lo": depth_lo, "depth_hi": depth_hi,
                "channel": channel, "observed": float(observed),
                "null_mean": float(null.mean()),
                "null_std": float(null.std(ddof=1)),
                "p_value": float(p), "n_eff": n_ch,
                "mean_reward": float(reward[valid_bin].mean()),
                "informative": not bin_vacuous,
                "vacuous_bin": bin_vacuous,
            })

    out = pd.DataFrame(rows)
    informative = out[out["informative"] & (out["n_eff"] >= 12)]
    sig = informative[informative["p_value"] < 0.05]
    log.info(
        "depth-stratified coupling: %d/%d informative cells p<0.05 "
        "(%d/%d total; n_bins=%d, vacuous_bins=%d)",
        len(sig), len(informative), len(sig), len(out), n_bins, len(vacuous_bins),
    )
    return out
