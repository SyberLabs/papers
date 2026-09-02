"""Tests for multivariate coupling (P2)."""

from __future__ import annotations

import numpy as np

from greenhypercube.config import DataConfig, ManifoldConfig
from greenhypercube.data import ingest
from greenhypercube.hypercube import build_manifold
from greenhypercube.validation.multivariate import (
    measure_multivariate_coupling,
    rank_partial_r2,
)
from greenhypercube.utils.rng import make_rng

N = 90


def _manifold(tmp_path, signal, seed=7):
    data = DataConfig(
        source="sample",
        cache_dir=str(tmp_path / f"mv_{signal}_{seed}"),
        n_species=N,
        reward_density=0.2,
        signal_strength=signal,
    )
    cache = ingest(data, seed=seed, force=True)
    return build_manifold(cache, ManifoldConfig(reward_top_frac=1.0))


def test_rank_partial_r2_increments_with_signal():
    rng = np.random.default_rng(0)
    n = 60
    C = rng.normal(size=(n, 2))
    X = 0.8 * C[:, :1] + rng.normal(scale=0.3, size=(n, 2))
    y = X.sum(axis=1) + rng.normal(scale=0.5, size=n)
    r2_full = rank_partial_r2(y, X, C)
    r2_none = rank_partial_r2(y, X, None)
    assert r2_full > 0.1
    assert r2_none >= r2_full


def test_zero_signal_multivariate_not_significant(tmp_path):
    m = _manifold(tmp_path, signal=0.0)
    tbl = measure_multivariate_coupling(m, make_rng(0), n_perm=399)
    assert tbl.iloc[0]["p_value"] > 0.01


def test_planted_signal_multivariate_exceeds_null(tmp_path):
    lo = measure_multivariate_coupling(_manifold(tmp_path, 0.0), make_rng(1), n_perm=399)
    hi = measure_multivariate_coupling(_manifold(tmp_path, 0.9), make_rng(1), n_perm=399)
    assert hi.iloc[0]["observed"] > lo.iloc[0]["observed"] + 0.05
