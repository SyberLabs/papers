"""Measure the cue->reward coupling actually present in a manifold.

This is the anti-tautology instrument. Our synthetic study plants a coupling via
``signal_strength`` and then shows structured search exploits it -- which is
AUDIT (a built-in mechanism cannot be evidence for the hypothesis it encodes).
The only thing that can count as evidence is the *measured magnitude* of the
coupling on real data. This module estimates, per cue channel, how strongly a
species' reward is predicted by that channel, with a label-permutation null so
each estimate comes with a calibrated significance.

Channels
--------
- ``sensory``  : Spearman(sensory_salience, reward) -- the chemistry/sensory cue.
  Species without any Duke chemistry record get zero salience (zero-imputed, not
  dropped); n_eff = n. Raw sensory coupling can therefore partly reflect
  missingness-as-signal (absence of chemistry record correlates with absence of
  use, both via study effort). Coverage control partials on chem_count; reward
  residualization includes has_chem_record when present.
- ``phylo``    : Spearman(reward_i, phylo-weighted neighbour reward) -- does
  phylogenetic proximity predict reward? (The premise of phylogenetic search.)
- ``eco``      : same, over the co-occurrence graph neighbours.
- ``bio``      : same, over the animal-association graph neighbours.

The neighbour statistic is a Moran's-I-flavoured autocorrelation: each species'
reward against a (row-normalized, zero-diagonal) weighted average of others'
reward. A positive, significant value means the cue carries exploitable signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from ..hypercube import Manifold
from ..utils import get_logger
from ..utils.rng import RNG
from .residual import depth_correlation, rank_residualize

log = get_logger("validation.coupling")


def _row_normalize(W: np.ndarray) -> np.ndarray:
    """Zero the diagonal and row-normalize a non-negative weight matrix."""
    W = np.array(W, dtype=float, copy=True)
    np.fill_diagonal(W, 0.0)
    s = W.sum(axis=1, keepdims=True)
    nz = s[:, 0] > 0
    W[nz] = W[nz] / s[nz]
    return W


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    rho, _ = spearmanr(a, b)
    return 0.0 if np.isnan(rho) else float(rho)


def _partial_spearman(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> float:
    """Spearman(a, b) controlling for covariates ``C`` (rank-residual partial).

    Rank-transform everything, regress the ranks of ``a`` and ``b`` on the ranks
    of ``C`` (with intercept), and correlate the residuals. This is the
    coverage-controlled coupling: how much cue and reward still co-vary once
    research effort (occurrence/interaction/chemistry counts) is removed.
    """
    if a.size < 4 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    ra, rb = rankdata(a), rankdata(b)
    cols = [rankdata(C[:, k]) for k in range(C.shape[1]) if np.ptp(C[:, k]) > 0]
    M = np.column_stack([np.ones(a.size)] + cols)
    ra_res = ra - M @ np.linalg.lstsq(M, ra, rcond=None)[0]
    rb_res = rb - M @ np.linalg.lstsq(M, rb, rcond=None)[0]
    if np.ptp(ra_res) == 0 or np.ptp(rb_res) == 0:
        return 0.0
    return float(np.corrcoef(ra_res, rb_res)[0, 1])


def _channel_weights(m: Manifold) -> dict[str, np.ndarray | None]:
    """Neighbour weight matrices per channel (None for the direct sensory cue)."""
    return {
        "sensory": None,  # direct feature, handled specially
        "phylo": _row_normalize(np.asarray(m.phylo_similarity(), dtype=float)),
        "eco": _row_normalize(np.asarray(m.eco_adj, dtype=float)),
        "bio": _row_normalize(np.asarray(m.bio_adj, dtype=float)),
    }


def _statistic(channel: str, W: np.ndarray | None, salience: np.ndarray,
               reward: np.ndarray, valid: np.ndarray,
               control: np.ndarray | None = None) -> float:
    """Coupling statistic for one channel given a (possibly permuted) reward.

    When ``control`` (per-species covariates) is given, a coverage-controlled
    partial-rank correlation is used instead of a plain Spearman.
    """
    if channel == "sensory":
        a, b = salience, reward
    else:
        a, b = reward, W @ reward
    a, b = a[valid], b[valid]
    if control is None:
        return _spearman(a, b)
    return _partial_spearman(a, b, control[valid])


def measure_coupling(m: Manifold, rng: RNG, n_perm: int = 999,
                     control: bool = False,
                     residual_reward: bool = False) -> pd.DataFrame:
    """Estimate per-channel cue->reward coupling with a label-permutation null.

    Returns one row per channel with: observed statistic, null mean/std, a
    z-score, a two-sided empirical p-value, and the number of informative species
    (``n_eff``) the statistic is computed over.

    With ``control=True`` the statistic is partialled on the manifold's
    research-coverage covariates (occurrence/interaction/chemistry counts), so a
    surviving coupling cannot be explained by study effort alone. Requires
    ``m.coverage`` to be present.

    With ``residual_reward=True`` the reward is rank-residualized against
    ``m.reward_depth`` before coupling (reward* = reward − E[reward | depth]).
    Simulation reward is unchanged; only the coupling instrument uses reward*.
    """
    reward = np.asarray(m.reward, dtype=float)
    if residual_reward:
        if m.reward_depth is None:
            raise ValueError(
                "residual-reward coupling requested but m.reward_depth is None"
            )
        depth = np.asarray(m.reward_depth, dtype=float)
        rho_pre = depth_correlation(reward, depth)
        reward = rank_residualize(reward, depth)
        log.info("reward residualized on depth (pre-residual depth rho=%.3f)", rho_pre)
    salience = np.asarray(m.sensory_salience, dtype=float)
    weights = _channel_weights(m)
    cov = None
    if control:
        if m.coverage is None:
            raise ValueError("coverage-controlled coupling requested but m.coverage is None")
        cov = np.asarray(m.coverage, dtype=float)

    rows = []
    for channel, W in weights.items():
        if channel == "sensory":
            valid = np.ones(m.n, dtype=bool)
            n_eff = m.n
        else:
            valid = W.sum(axis=1) > 0  # species with at least one neighbour
            n_eff = int(valid.sum())
        if n_eff < 3:
            log.info("coupling[%s]: too few informative species (%d); skipping", channel, n_eff)
            rows.append({"channel": channel, "observed": 0.0, "null_mean": 0.0,
                         "null_std": 0.0, "z": 0.0, "p_value": 1.0, "n_eff": n_eff})
            continue

        observed = _statistic(channel, W, salience, reward, valid, cov)
        null = np.empty(n_perm, dtype=float)
        for i in range(n_perm):
            perm = reward[rng.permutation(m.n)]
            null[i] = _statistic(channel, W, salience, perm, valid, cov)
        null_mean, null_std = float(null.mean()), float(null.std(ddof=1))
        z = (observed - null_mean) / null_std if null_std > 0 else 0.0
        p = (1 + int(np.sum(np.abs(null) >= abs(observed)))) / (n_perm + 1)
        rows.append({"channel": channel, "observed": float(observed),
                     "null_mean": null_mean, "null_std": null_std,
                     "z": float(z), "p_value": float(p), "n_eff": n_eff})

    out = pd.DataFrame(rows)
    parts = []
    if residual_reward:
        parts.append("reward-residualized ")
    if control:
        parts.append("coverage-controlled ")
    log.info("measured %scue-reward coupling over n=%d species, %d permutations",
             "".join(parts), m.n, n_perm)
    return out
