"""M2 ladder orchestration: partition strategy advantage by null type."""

from __future__ import annotations

import copy

import pandas as pd

from ..config import Config, StrategyConfig
from ..data import ingest
from ..hypercube import build_manifold
from ..utils import get_logger
from .stratified import measure_depth_stratified_coupling
from .multivariate import measure_depth_stratified_multivariate
from .m2_analysis import (
    paired_genuine_component,
    stratified_bin_diagnostics,
    stratified_fp_from_table,
    stratified_reference_grid,
    high_documentation_stratified_summary,
    high_doc_eco_decomposition,
    TOST_DELTA_AUDC,
    TOST_DELTA_RATIONALE,
)
from ..utils.rng import make_rng

log = get_logger("validation.m2_ladder")

M2_CONDITIONS = [
    "permute_reward",
    "permute_reward_within_effort",
    "permute_features",
]


def _with_degree_random(cfg: Config) -> Config:
    cfg = copy.deepcopy(cfg)
    if not any(s.name == "degree_random" for s in cfg.strategies):
        cfg.strategies = list(cfg.strategies) + [
            StrategyConfig(name="degree_random", kind="degree_random", params={"epsilon": 0.08}),
        ]
    return cfg


def run_m2_ladder(cfg: Config, force: bool = False) -> dict[str, pd.DataFrame]:
    """Run M2 decomposition controls + depth-stratified coupling on one config."""
    from ..simulation.study import run_controls  # lazy: breaks study↔validation cycle

    cfg_ctrl = _with_degree_random(cfg)
    per_rep = run_controls(cfg_ctrl, M2_CONDITIONS, force=force)
    advantage = _m2_advantage_summary(per_rep)

    cache = ingest(cfg.data, cfg.seed, force=force)
    m = build_manifold(cache, cfg.manifold)
    stratified = measure_depth_stratified_coupling(
        m, make_rng(cfg.seed + 11), n_bins=5, n_perm=399,
    )
    stratified_control = measure_depth_stratified_coupling(
        m, make_rng(cfg.seed + 12), n_bins=5, n_perm=399, control=True,
    )
    stratified_residual = measure_depth_stratified_coupling(
        m, make_rng(cfg.seed + 13), n_bins=5, n_perm=399,
        control=True, residual_reward=True,
    )
    stratified_mv_control = measure_depth_stratified_multivariate(
        m, make_rng(cfg.seed + 14), n_bins=5, n_perm=399, control=True,
    )
    genuine = paired_genuine_component(per_rep)
    bin_diag = stratified_bin_diagnostics(m)
    bin_diag_resid = stratified_bin_diagnostics(m, residual_reward=True)
    ref_grid = stratified_reference_grid(stratified_residual)
    fp_context = pd.DataFrame([
        {**stratified_fp_from_table(stratified, bin_diag, reference_grid=ref_grid),
         "layer": "raw"},
        {**stratified_fp_from_table(stratified_control, bin_diag, reference_grid=ref_grid),
         "layer": "controlled"},
        {**stratified_fp_from_table(stratified_residual, bin_diag, reference_grid=ref_grid),
         "layer": "controlled_residual"},
        {**stratified_fp_from_table(stratified_mv_control, bin_diag, reference_grid=ref_grid),
         "layer": "multivariate_controlled"},
    ])
    high_doc = {
        "raw": high_documentation_stratified_summary(stratified, bin_diag),
        "controlled": high_documentation_stratified_summary(stratified_control, bin_diag),
        "controlled_residual": high_documentation_stratified_summary(
            stratified_residual, bin_diag),
        "multivariate_controlled": high_documentation_stratified_summary(
            stratified_mv_control, bin_diag, channels=()),
    }
    eco_decomp = high_doc_eco_decomposition(
        stratified_control, stratified_residual, stratified_mv_control, bin_diag,
    )

    return {
        "controls_replicates": per_rep,
        "advantage": advantage,
        "genuine_component": genuine,
        "stratified_by_depth": stratified,
        "stratified_controlled": stratified_control,
        "stratified_controlled_residual": stratified_residual,
        "stratified_multivariate_controlled": stratified_mv_control,
        "stratified_bin_diagnostics": bin_diag,
        "stratified_bin_diagnostics_residual": bin_diag_resid,
        "stratified_fp_context": fp_context,
        "high_doc_eco": high_doc,
        "high_doc_eco_decomposition": eco_decomp,
        "tost_delta": TOST_DELTA_AUDC,
        "tost_delta_rationale": TOST_DELTA_RATIONALE,
    }


def _m2_advantage_summary(per_rep: pd.DataFrame) -> pd.DataFrame:
    """Advantage over random and over degree_random, by condition."""
    rows = []
    for condition, grp in per_rep.groupby("condition"):
        rand = grp.loc[grp["strategy"] == "random", "audc"]
        rand_mean = float(rand.mean()) if len(rand) else 0.0
        deg = grp.loc[grp["strategy"] == "degree_random", "audc"]
        deg_mean = float(deg.mean()) if len(deg) else 0.0
        for strategy, sg in grp.groupby("strategy"):
            audc = float(sg["audc"].mean())
            rows.append({
                "condition": condition,
                "strategy": strategy,
                "audc_mean": audc,
                "advantage_over_random": audc - rand_mean,
                "advantage_over_degree_random": audc - deg_mean
                if strategy != "degree_random" else 0.0,
            })
    return pd.DataFrame(rows).sort_values(
        ["condition", "advantage_over_random"], ascending=[True, False]
    )
