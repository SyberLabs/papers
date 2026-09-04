"""M2 ladder inference: genuine-component equivalence and stratified diagnostics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from ..hypercube import Manifold
from .controls import effort_index

EFFORT_STRAT_NULL = "permute_reward_within_effort"
KEY_STRATEGIES = ("ecological", "cultural", "sensory")

# Pre-specified SESOI for paired genuine-component TOST (AUDC units).
# ~25–50% of observed real structured advantage (0.05–0.10) and ~0.5× random
# replicate SD on full-flora landscapes: below a gain that would alter a
# fixed-budget screening decision.
TOST_DELTA_AUDC = 0.03
TOST_DELTA_RATIONALE = (
    "Pre-specified smallest effect of interest: 0.03 AUDC ≈ one quarter to one "
    "half of real structured advantage over random and on the order of the "
    "random-strategy replicate SD; sub-threshold for fixed-budget screening."
)


def paired_genuine_component(
    per_rep: pd.DataFrame,
    delta: float = TOST_DELTA_AUDC,
    baseline: str = "random",
) -> pd.DataFrame:
    """Paired advantage(real) − advantage(effort-stratified null) with TOST.

    The effort-independent component is ≈0 when this difference is centered on
    zero and passes equivalence at margin ``delta``. Report per strategy using
    paired replicates on the same landscape ordering.
    """
    rows = []
    for strategy in KEY_STRATEGIES:
        real = _adv_series(per_rep, "real", strategy, baseline)
        strat = _adv_series(per_rep, EFFORT_STRAT_NULL, strategy, baseline)
        if real is None or strat is None:
            continue
        diff = real - strat
        n = len(diff)
        mean = float(diff.mean())
        se = float(diff.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
        ci_lo, ci_hi = _ci95(diff)
        t_p = float(stats.ttest_1samp(diff, 0.0).pvalue) if n > 1 else 1.0
        tost_p = _tost_1samp(diff, delta) if n > 1 else 1.0
        rows.append({
            "strategy": strategy,
            "n_pairs": n,
            "genuine_component_mean": mean,
            "genuine_component_se": se,
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "t_test_p_vs_zero": t_p,
            "tost_delta": delta,
            "tost_p_equivalent": tost_p,
            "equivalent_at_delta": tost_p < 0.05,
        })
    return pd.DataFrame(rows)


def _adv_series(
    per_rep: pd.DataFrame, condition: str, strategy: str, baseline: str,
) -> pd.Series | None:
    sub = per_rep[per_rep["condition"] == condition]
    if sub.empty:
        return None
    rand = sub[sub["strategy"] == baseline].set_index("replicate")["audc"]
    strat = sub[sub["strategy"] == strategy].set_index("replicate")["audc"]
    common = rand.index.intersection(strat.index)
    if len(common) == 0:
        return None
    return strat.loc[common] - rand.loc[common]


def _ci95(x: pd.Series) -> tuple[float, float]:
    n = len(x)
    if n < 2:
        m = float(x.mean()) if n else 0.0
        return m, m
    se = x.std(ddof=1) / math.sqrt(n)
    h = se * stats.t.ppf(0.975, n - 1)
    m = float(x.mean())
    return m - h, m + h


def _tost_1samp(diff: pd.Series, delta: float) -> float:
    """Two one-sided tests for |mean| < delta (equivalence)."""
    d = diff.to_numpy(dtype=float)
    n = len(d)
    mean = d.mean()
    se = d.std(ddof=1) / math.sqrt(n)
    if se <= 0:
        return 1.0 if abs(mean) >= delta else 0.0
    df = n - 1
    p_upper = stats.t.cdf((mean - delta) / se, df)
    p_lower = stats.t.cdf((-delta - mean) / se, df)
    return float(max(p_upper, p_lower))


def stratified_bin_diagnostics(
    m: Manifold,
    n_bins: int = 5,
    residual_reward: bool = False,
) -> pd.DataFrame:
    """Within-bin reward variation: distinguishes vacuous vs informative nulls."""
    from .residual import rank_residualize

    reward = np.asarray(m.reward, dtype=float)
    if residual_reward and m.reward_depth is not None:
        reward = rank_residualize(reward, np.asarray(m.reward_depth, float))

    depth = effort_index(m)
    edges = np.quantile(depth, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    bins = np.digitize(depth, edges[1:-1], right=False)
    thr = m.discovery_threshold

    rows = []
    for b in range(n_bins):
        mask = bins == b
        r = reward[mask]
        n = int(mask.sum())
        rows.append({
            "bin": b,
            "depth_lo": float(edges[b]),
            "depth_hi": float(edges[b + 1]),
            "n_species": n,
            "reward_mean": float(r.mean()) if n else 0.0,
            "reward_std": float(r.std(ddof=1)) if n > 1 else 0.0,
            "reward_ptp": float(np.ptp(r)) if n else 0.0,
            "n_useful": int((r >= thr).sum()) if n else 0,
            "frac_useful": float((r >= thr).mean()) if n else 0.0,
            "vacuous": bool(n >= 12 and np.ptp(r) < 1e-6),
        })
    return pd.DataFrame(rows)


def stratified_false_positive_context(n_cells: int, n_sig: int, alpha: float = 0.05) -> dict:
    """Binomial expectation for significant cells under global null."""
    exp = n_cells * alpha
    p_ge = float(stats.binom.sf(n_sig - 1, n_cells, alpha)) if n_cells else 1.0
    return {
        "n_cells": n_cells,
        "n_sig": n_sig,
        "alpha_per_cell": alpha,
        "expected_false_positives": exp,
        "p_at_least_n_sig_under_null": p_ge,
    }


def stratified_reference_grid(
    stratified: pd.DataFrame,
    min_n_eff: int = 12,
) -> set[tuple[int, str]]:
    """Bin×channel keys for the keystone informative grid (from reference layer)."""
    if "channel" not in stratified.columns:
        return set()
    ref = stratified
    if "informative" in ref.columns:
        ref = ref[ref["informative"]]
    ref = ref[ref["n_eff"] >= min_n_eff]
    return {(int(r.bin), str(r.channel)) for r in ref.itertuples()}


def stratified_fp_from_table(
    stratified: pd.DataFrame,
    bin_diag: pd.DataFrame,
    alpha: float = 0.05,
    min_n_eff: int = 12,
    reference_grid: set[tuple[int, str]] | None = None,
) -> dict:
    """FP context on informative bin×channel cells only (excludes vacuous depth bins).

    When ``reference_grid`` is supplied (typically from controlled+residual),
    all layers are scored on the same bin×channel cells so raw vs controlled
    comparisons are apples-to-apples.
    """
    vacuous = set(bin_diag.loc[bin_diag["vacuous"], "bin"].astype(int))
    if reference_grid is not None and "channel" in stratified.columns:
        informative = stratified[
            stratified.apply(
                lambda r: (int(r["bin"]), str(r["channel"])) in reference_grid,
                axis=1,
            )
        ]
    elif reference_grid is not None:
        ref_bins = {b for b, _ in reference_grid}
        informative = stratified[stratified["bin"].isin(ref_bins)]
        if "informative" in informative.columns:
            informative = informative[informative["informative"]]
    elif "informative" in stratified.columns:
        informative = stratified[stratified["informative"]]
    else:
        informative = stratified[
            (~stratified["bin"].isin(vacuous)) & (stratified["n_eff"] >= min_n_eff)
        ]
    if "channel" in informative.columns:
        informative = informative[informative["n_eff"] >= min_n_eff]
    n_cells = len(informative)
    n_sig = int((informative["p_value"] < alpha).sum())
    ctx = stratified_false_positive_context(n_cells, n_sig, alpha)
    ctx["n_cells_total"] = len(stratified)
    ctx["n_vacuous_bins"] = len(vacuous)
    ctx["common_grid"] = reference_grid is not None
    return ctx


def high_documentation_stratified_summary(
    stratified: pd.DataFrame,
    bin_diag: pd.DataFrame,
    *,
    top_bins: tuple[int, ...] = (3, 4),
    channels: tuple[str, ...] = ("eco",),
) -> pd.DataFrame:
    """Eco (or other channels) in high-documentation effort bins: label-noise check."""
    vacuous = set(bin_diag.loc[bin_diag["vacuous"], "bin"].astype(int))
    sub = stratified[
        stratified["bin"].isin(top_bins)
        & (~stratified["bin"].isin(vacuous))
        & (stratified["n_eff"] >= 12)
    ].copy()
    if "channel" in sub.columns:
        sub = sub[sub["channel"].isin(channels)]
    sub["high_documentation"] = True
    return sub.sort_values(["bin"] + (["channel"] if "channel" in sub.columns else [])).reset_index(drop=True)


def high_doc_eco_decomposition(
    controlled: pd.DataFrame,
    controlled_residual: pd.DataFrame,
    multivariate_controlled: pd.DataFrame,
    bin_diag: pd.DataFrame,
    *,
    top_bins: tuple[int, ...] = (3, 4),
) -> pd.DataFrame:
    """High-doc eco split: confound (controlled->+residual) vs redundancy (multivariate).

    - controlled -> controlled+residual (univariate eco): if eco dies, within-bin
      documentation depth carried the signal.
    - multivariate controlled (no reward-residual): if joint partial R² is null,
      eco is redundant with other channels given coverage.
    """
    vacuous = set(bin_diag.loc[bin_diag["vacuous"], "bin"].astype(int))
    rows = []
    for b in top_bins:
        if b in vacuous:
            continue
        eco_c = controlled[
            (controlled["bin"] == b) & (controlled["channel"] == "eco")
        ]
        eco_r = controlled_residual[
            (controlled_residual["bin"] == b) & (controlled_residual["channel"] == "eco")
        ]
        mv = multivariate_controlled[multivariate_controlled["bin"] == b]
        if eco_c.empty or eco_r.empty or mv.empty:
            continue
        rows.append({
            "bin": b,
            "eco_controlled_p": float(eco_c.iloc[0]["p_value"]),
            "eco_controlled_rho": float(eco_c.iloc[0]["observed"]),
            "eco_residual_controlled_p": float(eco_r.iloc[0]["p_value"]),
            "eco_residual_controlled_rho": float(eco_r.iloc[0]["observed"]),
            "multivariate_controlled_p": float(mv.iloc[0]["p_value"]),
            "multivariate_controlled_r2": float(mv.iloc[0]["observed"]),
            "depth_confound": bool(
                eco_c.iloc[0]["p_value"] < 0.05 and eco_r.iloc[0]["p_value"] >= 0.05
            ),
            "multivariate_redundancy": (
                "supported"
                if eco_c.iloc[0]["p_value"] < 0.05 and mv.iloc[0]["p_value"] >= 0.05
                and float(mv.iloc[0]["observed"]) < 0.15
                else "underpowered"
                if eco_c.iloc[0]["p_value"] < 0.05 and mv.iloc[0]["p_value"] >= 0.05
                and float(mv.iloc[0]["observed"]) >= 0.15
                else "n/a"
            ),
        })
    return pd.DataFrame(rows)
