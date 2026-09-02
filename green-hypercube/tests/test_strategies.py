"""Tests that every strategy runs and structured search beats random."""

from __future__ import annotations

import numpy as np

from greenhypercube.config import StrategyConfig, SimulationConfig
from greenhypercube.strategies import build_strategy, STRATEGY_KINDS
from greenhypercube.simulation.engine import run_episode
from greenhypercube.utils.rng import make_rng


def _run(kind, manifold, budget=300, params=None, seed=0):
    spec = StrategyConfig(name=kind, kind=kind, params=params or {})
    strat = build_strategy(spec, manifold, make_rng(seed))
    strat.name = kind
    return run_episode(manifold, strat, budget, make_rng(seed + 1))


def test_all_kinds_registered():
    assert set(STRATEGY_KINDS) == {
        "random", "phylogenetic", "sensory", "ecological", "degree_random",
        "social", "cultural"
    }


def test_every_strategy_runs(small_manifold):
    for kind in STRATEGY_KINDS:
        res = _run(kind, small_manifold)
        assert len(res.order) > 0
        assert len(res.order) == len(set(res.order))  # no species tested twice
        assert 0.0 <= res.audc() <= 1.0


def test_structured_beats_random(small_manifold):
    budget = 200
    rng_seeds = range(8)
    rand = np.mean([_run("random", small_manifold, budget, seed=s).audc() for s in rng_seeds])
    cult = np.mean([
        _run("cultural", small_manifold, budget, {"learning_rate": 0.6}, seed=s).audc()
        for s in rng_seeds
    ])
    eco = np.mean([_run("ecological", small_manifold, budget, seed=s).audc() for s in rng_seeds])
    # The integrative and ecological strategies should find useful plants sooner.
    assert cult > rand
    assert eco > rand


def test_social_population_consumes_budget(small_manifold):
    # Budget below the species pool (n=120) so it can be fully spent.
    res = _run("social", small_manifold, budget=96, params={"n_agents": 8})
    assert len(res.order) == 96
