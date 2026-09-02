"""Tests for reward-side residualization (P1)."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from greenhypercube.validation.residual import depth_correlation, rank_residualize


def test_rank_residualize_removes_depth_correlation():
    rng = np.random.default_rng(0)
    n = 80
    depth = rng.integers(1, 50, size=(n, 3)).astype(float)
    reward = 0.6 * depth[:, 0] + 0.3 * depth[:, 1] + rng.normal(0, 2, n)
    reward = (reward - reward.min()) / (reward.max() - reward.min())
    rho_before, _ = spearmanr(reward, depth.mean(axis=1))
    resid = rank_residualize(reward, depth)
    rho_after, _ = spearmanr(resid, depth.mean(axis=1))
    assert abs(rho_before) > 0.5
    assert abs(rho_after) < 0.15


def test_depth_correlation_detects_alignment():
    depth = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
    reward = np.array([0.1, 0.3, 0.6, 0.9])
    assert depth_correlation(reward, depth) > 0.9
