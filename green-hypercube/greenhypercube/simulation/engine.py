"""Episode and experiment engines.

``run_episode`` plays one strategy against the environment under a fixed
experiment budget. ``run_experiment`` runs the full config -- every strategy
across many seeded replicates -- and returns the per-replicate results plus the
mean discovery curves used for figures. All randomness derives from the config
seed via independent child streams, so results are fully reproducible.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from ..config import Config
from ..hypercube import Manifold
from ..strategies import build_strategy
from ..utils import get_logger
from ..utils.rng import make_rng
from .environment import Environment
from .metrics import EpisodeResult

log = get_logger("simulation.engine")


def run_episode(
    manifold: Manifold,
    strategy,
    budget: int,
    rng,
    observation_noise: float = 0.0,
) -> EpisodeResult:
    env = Environment(manifold, rng, observation_noise)
    strategy.reset()

    order: list[int] = []
    observed: list[float] = []
    true_reward: list[float] = []
    is_useful: list[bool] = []

    budget = min(budget, manifold.n)
    while env.n_tested < budget:
        proposals = strategy.propose_batch(env)
        if not proposals:
            break
        progressed = False
        for idx in proposals:
            if env.n_tested >= budget:
                break
            if env.is_tested(idx):
                continue
            r = env.experiment(idx)
            strategy.observe(idx, r)
            order.append(int(idx))
            observed.append(float(r))
            tr = float(manifold.reward[idx])
            true_reward.append(tr)
            is_useful.append(tr >= manifold.discovery_threshold)
            progressed = True
        if not progressed:
            break

    return EpisodeResult(
        strategy=getattr(strategy, "name", strategy.__class__.__name__),
        order=order,
        observed=observed,
        true_reward=true_reward,
        is_useful=is_useful,
        n_total_useful=manifold.n_useful,
        total_reward=manifold.total_reward,
        true_reward_full=manifold.reward.copy(),
    )


def run_experiment(cfg: Config, manifold: Manifold) -> dict:
    """Run all strategies x replicates; return results and mean curves."""
    sim = cfg.simulation
    seed_seq = np.random.SeedSequence(cfg.seed)
    # One child stream per (strategy, replicate); two sub-streams each (build/run).
    children = seed_seq.spawn(len(cfg.strategies) * sim.n_replicates)

    all_results: list[EpisodeResult] = []
    curves: dict[str, np.ndarray] = {}

    child_iter = iter(children)
    for spec in cfg.strategies:
        per_strategy: list[np.ndarray] = []
        for _ in tqdm(range(sim.n_replicates), desc=spec.name, leave=False):
            sub = next(child_iter).spawn(2)
            build_rng = make_rng(int(sub[0].generate_state(1)[0]))
            run_rng = make_rng(int(sub[1].generate_state(1)[0]))
            strat = build_strategy(spec, manifold, build_rng)
            strat.name = spec.name  # type: ignore[attr-defined]
            res = run_episode(
                manifold, strat, sim.budget, run_rng, sim.observation_noise
            )
            res.meta["kind"] = spec.kind
            all_results.append(res)
            per_strategy.append(_padded_discovery(res, sim.budget))
        curves[spec.name] = np.mean(np.vstack(per_strategy), axis=0)
        log.info(
            "%s: mean discoveries=%.1f / %d useful",
            spec.name, curves[spec.name][-1], manifold.n_useful,
        )

    return {"results": all_results, "curves": curves, "n_useful": manifold.n_useful}


def _padded_discovery(res: EpisodeResult, budget: int) -> np.ndarray:
    """Discovery curve padded/truncated to exactly ``budget`` points."""
    curve = res.discovery_curve()
    out = np.zeros(budget, dtype=float)
    k = min(len(curve), budget)
    out[:k] = curve[:k]
    if k < budget and k > 0:
        out[k:] = curve[k - 1]
    return out
