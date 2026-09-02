"""Tests for the cue-reward coupling instrument and reward sparsification.

The coupling estimator is the anti-tautology check: it must report ~no coupling
when none was planted (signal_strength=0) and detect coupling when it was. The
sparsifier must impose the requested reward density without disturbing the cues.
"""

from __future__ import annotations

import numpy as np

from greenhypercube.config import DataConfig, ManifoldConfig
from greenhypercube.data import ingest
from greenhypercube.hypercube import build_manifold
from greenhypercube.validation import measure_coupling
from greenhypercube.utils.rng import make_rng

N = 90


def _manifold(tmp_path, signal, top_frac=1.0, seed=7):
    data = DataConfig(
        source="sample",
        cache_dir=str(tmp_path / f"c_{signal}_{top_frac}_{seed}"),
        n_species=N,
        reward_density=0.2,
        signal_strength=signal,
    )
    cache = ingest(data, seed=seed, force=True)
    return build_manifold(cache, ManifoldConfig(reward_top_frac=top_frac))


def _max_observed(tbl):
    return float(tbl["observed"].max())


def test_zero_signal_reads_as_no_coupling(tmp_path):
    m = _manifold(tmp_path, signal=0.0)
    tbl = measure_coupling(m, make_rng(0), n_perm=399)
    # No channel should be strongly significant when nothing was planted.
    assert (tbl["p_value"] < 0.01).sum() == 0, tbl.to_string()


def test_planted_signal_is_detected_and_exceeds_null(tmp_path):
    lo = measure_coupling(_manifold(tmp_path, 0.0), make_rng(1), n_perm=399)
    hi = measure_coupling(_manifold(tmp_path, 0.9), make_rng(1), n_perm=399)
    # Planting strong coupling lifts the strongest channel well above the null case
    assert _max_observed(hi) > _max_observed(lo) + 0.15
    # ...and at least one channel becomes significant with a positive sign.
    sig = hi[(hi["p_value"] < 0.05) & (hi["observed"] > 0)]
    assert len(sig) >= 1, hi.to_string()


def test_sparsify_controls_reward_density(tmp_path):
    full = _manifold(tmp_path, signal=0.7, top_frac=1.0)
    sparse = _manifold(tmp_path, signal=0.7, top_frac=0.1)
    assert int((sparse.reward > 0).sum()) == max(1, round(0.1 * N))
    assert (sparse.reward > 0).sum() < (full.reward > 0).sum()
    # Survivors keep their original (highest) reward values.
    keep = sparse.reward > 0
    assert np.allclose(sparse.reward[keep], full.reward[keep])
