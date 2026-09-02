"""Higher-level studies: landscape replication and the null-control battery.

``run_study`` repeats the whole experiment across independently generated
landscapes (distinct data-generating seeds), so aggregate confidence intervals
reflect uncertainty about the landscape itself -- not merely the search RNG.

``run_controls`` runs every strategy on the real manifold and on each negative
control, returning a tidy per-replicate frame from which we read whether the
structured advantage collapses under the nulls (it must, if the advantage is
real signal rather than leakage).
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..data import ingest
from ..hypercube import build_manifold
from ..strategies import build_strategy
from ..utils import get_logger
from ..utils.rng import make_rng
from ..validation import make_null
from .engine import run_episode, run_experiment

log = get_logger("simulation.study")


def _landscape(cfg: Config, k: int, force: bool = False):
    """Build the manifold for landscape ``k`` (own cache + data seed)."""
    cfg_k = copy.deepcopy(cfg)
    multi = cfg.simulation.n_landscapes > 1
    if multi:
        cfg_k.data.cache_dir = str(Path(cfg.data.cache_dir) / f"landscape_{k}")
    data_seed = cfg.seed + 10007 * k
    cache = ingest(cfg_k.data, data_seed, force=force or multi)
    cfg_k.seed = cfg.seed + k
    return cfg_k, build_manifold(cache, cfg_k.manifold)


def run_study(cfg: Config, force: bool = False) -> dict:
    """Run all strategies across ``n_landscapes`` landscapes; combine results."""
    n_land = max(1, cfg.simulation.n_landscapes)
    if cfg.data.source == "live" and n_land > 1:
        log.warning("landscape replication unsupported for live source; using 1")
        n_land = 1

    all_results = []
    curve_stacks: dict[str, list[np.ndarray]] = {}
    n_useful_vals = []
    for k in range(n_land):
        cfg_k, m = _landscape(cfg, k, force=force)
        exp = run_experiment(cfg_k, m)
        for res in exp["results"]:
            res.meta["landscape"] = k
        all_results.extend(exp["results"])
        for name, curve in exp["curves"].items():
            curve_stacks.setdefault(name, []).append(curve)
        n_useful_vals.append(exp["n_useful"])

    curves = {name: np.mean(np.vstack(st), axis=0) for name, st in curve_stacks.items()}
    return {
        "results": all_results,
        "curves": curves,
        "n_useful": float(np.mean(n_useful_vals)),
        "n_landscapes": n_land,
    }


# Nulls that only permute reward — safe to apply to one cached cue manifold.
REWARD_ONLY_NULLS = frozenset({"permute_reward", "permute_reward_within_effort"})


def run_controls(cfg: Config, conditions: list[str], force: bool = False) -> pd.DataFrame:
    """Run strategies on the real manifold and each null; return per-replicate rows.

    For **live** cached landscapes the cue manifold is built once and reused
    across replicates and reward-only nulls (reward-blind correctness check).
    Feature/graph nulls receive a fresh ``make_null`` copy each time.

    For **sample** data a fresh landscape is generated per replicate so structured
    strategies are averaged over data-generating orderings.

    Columns: condition, strategy, audc, discoveries, replicate.
    """
    sim = cfg.simulation
    rows: list[dict] = []

    if cfg.data.source == "live":
        _, base = _landscape(cfg, 0, force=force)
        for rep in range(sim.n_replicates):
            _run_control_replicate(cfg, base, conditions, rep, rows)
    else:
        for rep in range(sim.n_replicates):
            _, base = _controls_landscape(cfg, rep, force)
            _run_control_replicate(cfg, base, conditions, rep, rows)

    return pd.DataFrame(rows)


def _run_control_replicate(
    cfg: Config,
    base,
    conditions: list[str],
    rep: int,
    rows: list[dict],
) -> None:
    sim = cfg.simulation
    all_conds = ["real"] + conditions
    for ci, condition in enumerate(all_conds):
        if condition == "real":
            m = base
        elif condition in REWARD_ONLY_NULLS:
            m = make_null(condition, base, make_rng(cfg.seed + 7919 * rep + 17 * ci))
        else:
            m = make_null(condition, base, make_rng(cfg.seed + 7919 * rep + 17 * ci))
        for si, spec in enumerate(cfg.strategies):
            build_rng = make_rng(cfg.seed + 31 * rep + si)
            run_rng = make_rng(cfg.seed + 977 * rep + 13 * si + 101 * ci)
            strat = build_strategy(spec, m, build_rng)
            strat.name = spec.name  # type: ignore[attr-defined]
            res = run_episode(m, strat, sim.budget, run_rng, sim.observation_noise)
            rows.append({
                "condition": condition,
                "strategy": spec.name,
                "replicate": rep,
                "audc": res.audc(),
                "discoveries": int(res.discovery_curve()[-1]) if res.order else 0,
            })


def _controls_landscape(cfg: Config, rep: int, force: bool):
    """A per-replicate landscape for the controls battery (sample source)."""
    if cfg.data.source != "sample":
        return _landscape(cfg, 0, force=force and rep == 0)
    cfg_r = copy.deepcopy(cfg)
    cfg_r.data.cache_dir = str(Path(cfg.data.cache_dir) / f"ctrl_{rep}")
    cache = ingest(cfg_r.data, cfg.seed + 9973 * rep, force=True)
    return cfg_r, build_manifold(cache, cfg_r.manifold)
