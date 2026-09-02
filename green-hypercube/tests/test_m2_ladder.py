"""Tests for M2 ladder nulls and effort stratification."""

from __future__ import annotations

import numpy as np

from greenhypercube.config import DataConfig, ManifoldConfig
from greenhypercube.data import ingest
from greenhypercube.hypercube import build_manifold
from greenhypercube.validation.controls import (
    effort_index,
    permute_features,
    permute_reward_within_effort,
)
from greenhypercube.utils.rng import make_rng

N = 60


def _manifold(tmp_path, signal=0.5, seed=3):
    data = DataConfig(
        source="sample",
        cache_dir=str(tmp_path / "m2"),
        n_species=N,
        reward_density=0.2,
        signal_strength=signal,
    )
    cache = ingest(data, seed=seed, force=True)
    return build_manifold(cache, ManifoldConfig(reward_top_frac=1.0))


def test_effort_stratified_perm_preserves_marginals(tmp_path):
    m = _manifold(tmp_path)
    effort = effort_index(m)
    out = permute_reward_within_effort(m, make_rng(0))
    # effort unchanged per species
    assert np.allclose(effort_index(out), effort)
    # reward multiset preserved globally
    assert np.allclose(np.sort(out.reward), np.sort(m.reward))


def test_feature_perm_destroys_x_reward_link(tmp_path):
    m = _manifold(tmp_path)
    out = permute_features(m, make_rng(1))
    assert not np.allclose(out.sensory_salience, m.sensory_salience)
    assert np.allclose(out.reward, m.reward)
    # shuffled salience should not correlate with original reward as strongly
    before = np.corrcoef(m.sensory_salience, m.reward)[0, 1]
    after = np.corrcoef(out.sensory_salience, m.reward)[0, 1]
    assert abs(after) < abs(before) + 0.05
