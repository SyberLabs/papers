"""Null-model transforms and the advantage-collapse analysis.

Each null returns a NEW :class:`Manifold` with one structural relationship
broken:

- ``permute_reward``  : shuffle the reward vector across species (preserves the
  sparse reward marginal, destroys ALL cue-reward coupling). The strongest
  global control: every structured strategy must fall to the random baseline.
- ``rewire_graphs``   : degree-preserving double-edge swaps on the eco/bio graphs
  (preserves each species' connectivity, destroys *which* species are linked).
  Targets ecological-association learning.
- ``shuffle_phylo``   : permute the phylogeny's tip-to-species assignment
  (preserves the tree metric, destroys reward's phylogenetic arrangement).
  Targets phylogenetic generalization.
- ``permute_reward_within_effort`` : shuffle reward within effort strata only
  (M2 rung 2 — preserves reward–effort link, destroys feature/topology coupling).
- ``permute_features`` : shuffle observable cue matrix across species (M2 rung 3).
- ``all``             : all three at once.
"""

from __future__ import annotations

from dataclasses import replace

import networkx as nx
import numpy as np
import pandas as pd

from ..hypercube import Manifold
from ..utils.rng import RNG


def permute_reward(m: Manifold, rng: RNG) -> Manifold:
    perm = rng.permutation(m.n)
    return replace(m, reward=m.reward[perm].copy())


def effort_index(m: Manifold) -> np.ndarray:
    """Composite study-effort surface for stratified permutations and binning.

    Combines cue-side coverage (occ_count, interaction_count, chem_count) and
    reward-side depth (NAEB documentation + has_chem_record when present).
    Used by ``permute_reward_within_effort`` (M2 ladder) and depth-stratified
    coupling — chemistry missingness enters strategy-side effort stratification
    as well as coupling-side residualization.
    """
    parts: list[np.ndarray] = []
    if m.coverage is not None:
        parts.append(np.asarray(m.coverage, dtype=float))
    if m.reward_depth is not None:
        parts.append(np.asarray(m.reward_depth, dtype=float))
    if not parts:
        return np.zeros(m.n, dtype=float)
    return np.log1p(np.sum(np.hstack(parts), axis=1))


def permute_reward_within_effort(m: Manifold, rng: RNG, n_strata: int = 5) -> Manifold:
    """Shuffle reward within effort quantiles (M2 ladder rung 2)."""
    effort = effort_index(m)
    if np.ptp(effort) == 0:
        return permute_reward(m, rng)
    edges = np.quantile(effort, np.linspace(0, 1, n_strata + 1))
    edges[-1] += 1e-9
    bins = np.digitize(effort, edges[1:-1], right=False)
    reward = m.reward.copy()
    for b in range(n_strata):
        idx = np.where(bins == b)[0]
        if len(idx) < 2:
            continue
        perm = rng.permutation(idx)
        reward[idx] = m.reward[perm]
    return replace(m, reward=reward)


def permute_features(m: Manifold, rng: RNG) -> Manifold:
    """Shuffle observable cues across species; reward and graphs fixed (M2 rung 3)."""
    perm = rng.permutation(m.n)
    return replace(
        m,
        X=m.X[perm].copy(),
        sensory_salience=m.sensory_salience[perm].copy(),
    )


def rewire_graphs(m: Manifold, rng: RNG) -> Manifold:
    return replace(m, eco=_rewire(m.eco, rng), bio=_rewire(m.bio, rng))


def shuffle_phylo(m: Manifold, rng: RNG) -> Manifold:
    perm = rng.permutation(m.n)
    D = m.D_phylo[np.ix_(perm, perm)].copy()
    return replace(m, D_phylo=D)


def all_nulls(m: Manifold, rng: RNG) -> Manifold:
    return shuffle_phylo(rewire_graphs(permute_reward(m, rng), rng), rng)


NULLS = {
    "permute_reward": permute_reward,
    "permute_reward_within_effort": permute_reward_within_effort,
    "permute_features": permute_features,
    "rewire_graphs": rewire_graphs,
    "shuffle_phylo": shuffle_phylo,
    "all": all_nulls,
}


def make_null(name: str, m: Manifold, rng: RNG) -> Manifold:
    if name not in NULLS:
        raise ValueError(f"unknown null {name!r}; choose from {sorted(NULLS)}")
    return NULLS[name](m, rng)


def _rewire(g: nx.Graph, rng: RNG) -> nx.Graph:
    h = g.copy()
    n_edges = h.number_of_edges()
    if n_edges < 2:
        return h
    seed = int(rng.integers(0, 2**31 - 1))
    try:
        nx.double_edge_swap(h, nswap=5 * n_edges, max_tries=100 * n_edges, seed=seed)
    except (nx.NetworkXError, nx.NetworkXAlgorithmError):
        pass  # too few swappable edges; leave as-is
    return h


def null_advantage_table(per_replicate: pd.DataFrame) -> pd.DataFrame:
    """Summarize advantage-over-random AUDC per (condition, strategy).

    ``per_replicate`` must have columns: condition, strategy, audc. Returns mean
    advantage = mean(AUDC_strategy) - mean(AUDC_random) within each condition,
    so a healthy result shows large advantage under 'real' and ~0 under nulls.
    """
    rows = []
    for condition, grp in per_replicate.groupby("condition"):
        rand = grp.loc[grp["strategy"] == "random", "audc"]
        rand_mean = float(rand.mean()) if len(rand) else 0.0
        for strategy, sg in grp.groupby("strategy"):
            rows.append(
                {
                    "condition": condition,
                    "strategy": strategy,
                    "audc_mean": float(sg["audc"].mean()),
                    "audc_std": float(sg["audc"].std(ddof=1)) if len(sg) > 1 else 0.0,
                    "advantage_over_random": float(sg["audc"].mean()) - rand_mean,
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["condition", "advantage_over_random"], ascending=[True, False])
