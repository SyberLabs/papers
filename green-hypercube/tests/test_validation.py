"""Rigor guarantees: the framework must NOT manufacture an advantage from noise.

Two properties pin this down empirically:

1. With cue-reward coupling set to zero (``signal_strength=0``), no structured
   strategy may beat the random baseline -- the cues carry no information.
2. On a coupled landscape, permuting the reward labels (the strongest null)
   must collapse any structured advantage back to ~0.

CRITICAL METHOD NOTE: these comparisons are averaged over many independently
generated LANDSCAPES (data seeds), not over search RNG on a single landscape. A
near-deterministic strategy yields one fixed ordering per landscape whose AUDC
is a single draw; single-landscape AUDC variance (~0.08) is large enough to fake
an advantage. Averaging over landscapes is the only correct null test -- and is
the whole reason landscape replication exists.
"""

from __future__ import annotations

import numpy as np

from greenhypercube.config import DataConfig, ManifoldConfig, StrategyConfig
from greenhypercube.data import ingest
from greenhypercube.hypercube import build_manifold
from greenhypercube.simulation.engine import run_episode
from greenhypercube.strategies import build_strategy
from greenhypercube.validation import permute_reward
from greenhypercube.utils.rng import make_rng

N = 120
BUDGET = 70
N_LANDSCAPES = 14


def _manifold(tmp_path, signal, seed, density=0.15):
    data = DataConfig(
        source="sample",
        cache_dir=str(tmp_path / f"c_{signal}_{seed}"),
        n_species=N,
        reward_density=density,
        signal_strength=signal,
    )
    cache = ingest(data, seed=seed, force=True)
    return build_manifold(cache, ManifoldConfig())


def _audc(kind, m, seed, params=None):
    spec = StrategyConfig(name=kind, kind=kind, params=params or {})
    strat = build_strategy(spec, m, make_rng(seed))
    strat.name = kind
    return run_episode(m, strat, BUDGET, make_rng(seed + 777)).audc()


def _mean_advantage(tmp_path, signal, kind, params=None, transform=None):
    """Mean (AUDC_strategy - AUDC_random) over independent landscapes."""
    diffs = []
    for k in range(N_LANDSCAPES):
        m = _manifold(tmp_path, signal, seed=100 + k)
        if transform is not None:
            m = transform(m, make_rng(500 + k))
        r = _audc("random", m, seed=k)
        s = _audc(kind, m, seed=k, params=params)
        diffs.append(s - r)
    return float(np.mean(diffs))


def test_zero_signal_gives_parity_with_random(tmp_path):
    # Averaged over landscapes, no structured strategy may beat random at signal 0.
    for kind, params in [
        ("sensory", None),
        ("ecological", None),
        ("cultural", {"learning_rate": 0.6}),
        ("phylogenetic", None),
    ]:
        adv = _mean_advantage(tmp_path, 0.0, kind, params)
        assert adv < 0.04, f"{kind} beat random with zero signal (adv={adv:.3f})"


def test_coupled_landscape_shows_real_advantage(tmp_path):
    adv = _mean_advantage(tmp_path, 0.85, "cultural", {"learning_rate": 0.6})
    assert adv > 0.12, f"expected clear advantage on coupled landscape (adv={adv:.3f})"


def test_permute_reward_collapses_advantage(tmp_path):
    # Same coupled landscapes, but with reward permuted: advantage must vanish.
    adv = _mean_advantage(
        tmp_path, 0.85, "cultural", {"learning_rate": 0.6}, transform=permute_reward
    )
    assert abs(adv) < 0.04, f"advantage survived reward permutation (adv={adv:.3f})"
