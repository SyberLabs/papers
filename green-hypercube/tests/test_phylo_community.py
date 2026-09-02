"""Tests for P7 phylogenetic community metrics (NRI/NTI, hot-node enrichment)."""

from __future__ import annotations

import numpy as np

from greenhypercube.config import DataConfig, ManifoldConfig
from greenhypercube.data import ingest
from greenhypercube.data import schema
from greenhypercube.hypercube import build_manifold
from greenhypercube.validation.phylo_community import measure_phylo_community, mpd, mntd
from greenhypercube.utils.rng import make_rng

N = 120


def _manifold(tmp_path, signal=0.9, density=0.25, seed=11):
    data = DataConfig(
        source="sample",
        cache_dir=str(tmp_path / f"p7_{signal}_{density}_{seed}"),
        n_species=N,
        reward_density=density,
        signal_strength=signal,
        coupling={"phylo": 1.0},
    )
    cache = ingest(data, seed=seed, force=True)
    return build_manifold(cache, ManifoldConfig(reward_top_frac=1.0)), cache


def test_mpd_mntd_basic():
    D = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]], dtype=float)
    idx = np.array([0, 1])
    assert mpd(D, idx) == 1.0
    assert mntd(D, idx) == 1.0


def test_phylo_clustering_detected_on_planted_signal(tmp_path):
    m, cache = _manifold(tmp_path, signal=0.95, density=0.3)
    newick = cache.read_text(schema.PHYLOGENY_NAME)
    summary, _hot, _fp = measure_phylo_community(
        m, newick, make_rng(0), n_perm=399, min_clade_tips=6, min_genus_tips=4,
    )
    raw = summary[(summary["label"] == "raw") & (summary["null_mode"] == "standard")]
    assert len(raw) == 1
    row = raw.iloc[0]
    # Phylo-planted reward should cluster (positive NRI = lower MPD than null).
    assert row["nri"] > 1.0, row.to_string()
    assert row["n_community"] >= 2


def test_fp_context_accounts_for_multiplicity(tmp_path):
    m, cache = _manifold(tmp_path, signal=0.9)
    newick = cache.read_text(schema.PHYLOGENY_NAME)
    _, _, fp = measure_phylo_community(m, newick, make_rng(2), n_perm=199)
    assert "n_tested" in fp.columns
    assert "e_fp_p05" in fp.columns
    raw_all = fp[(fp["label"] == "raw") & (fp["unit"] == "all")].iloc[0]
    assert raw_all["e_fp_p05"] == raw_all["n_tested"] * 0.05


def test_effort_null_runs_when_covariates_present(tmp_path):
    m, cache = _manifold(tmp_path)
    newick = cache.read_text(schema.PHYLOGENY_NAME)
    summary, _, _ = measure_phylo_community(m, newick, make_rng(1), n_perm=99)
    assert "effort" in set(summary["null_mode"])
    assert "standard" in set(summary["null_mode"])
