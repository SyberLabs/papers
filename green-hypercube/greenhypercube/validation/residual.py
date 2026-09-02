"""Reward-side residualization for symmetric confound control (P1).

Cue-side coverage control partials study effort out of *cues* while leaving
reward intact. Reward-side residualization removes the component of reward
predictable from documentation / chemistry-study depth:

    reward* = reward − E[reward | reward_depth]

Coupling measured on ``reward*`` (optionally with cue-side ``--control``) tests
whether cues predict reward beyond what shared study effort on the reward
provenance explains. This is the symmetric counterpart to coverage partialization
required for a publishable Arc A claim (hypothesis H2.8).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def rank_residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Rank-residualize ``y`` after linear removal of rank-transformed ``X``.

    Each column of ``X`` with zero variance is dropped. Returns residuals on the
    original scale of ranked ``y`` (centred, not re-normalized to [0, 1]).
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if y.size < 4 or np.ptp(y) == 0:
        return y - y.mean()
    ry = rankdata(y)
    cols = [rankdata(X[:, k]) for k in range(X.shape[1]) if np.ptp(X[:, k]) > 0]
    if not cols:
        return ry - ry.mean()
    M = np.column_stack([np.ones(y.size)] + cols)
    resid = ry - M @ np.linalg.lstsq(M, ry, rcond=None)[0]
    return resid.astype(float)


def depth_correlation(reward: np.ndarray, depth: np.ndarray) -> float:
    """Spearman-like check: how much reward aligns with depth before residualization."""
    from scipy.stats import spearmanr

    r = np.asarray(reward, dtype=float)
    d = np.asarray(depth, dtype=float)
    if d.ndim == 1:
        d = d.reshape(-1, 1)
    # Use first PC-like summary: mean rank of columns
    summary = np.mean([rankdata(d[:, k]) for k in range(d.shape[1])], axis=0)
    if np.ptp(r) == 0 or np.ptp(summary) == 0:
        return 0.0
    rho, _ = spearmanr(r, summary)
    return 0.0 if np.isnan(rho) else float(rho)
