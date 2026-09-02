"""Tests for M2 genuine-component inference."""

from __future__ import annotations

import pandas as pd

from greenhypercube.validation.m2_analysis import paired_genuine_component


def test_genuine_component_zero_when_real_equals_strat():
    rows = []
    for rep in range(20):
        for cond in ("real", "permute_reward_within_effort"):
            for strat, audc in (("random", 0.1), ("ecological", 0.25)):
                rows.append({"condition": cond, "strategy": strat, "replicate": rep, "audc": audc})
    g = paired_genuine_component(pd.DataFrame(rows))
    row = g[g["strategy"] == "ecological"].iloc[0]
    assert abs(row["genuine_component_mean"]) < 1e-9
    assert row["tost_p_equivalent"] < 0.05


def test_stratified_fp_excludes_vacuous_bins():
    from greenhypercube.validation.m2_analysis import (
        stratified_fp_from_table,
        stratified_reference_grid,
    )

    bin_diag = pd.DataFrame([
        {"bin": 0, "vacuous": True},
        {"bin": 1, "vacuous": True},
        {"bin": 2, "vacuous": False},
        {"bin": 3, "vacuous": False},
        {"bin": 4, "vacuous": False},
    ])
    stratified = pd.DataFrame([
        {"bin": 0, "channel": "eco", "p_value": 0.01, "n_eff": 69, "informative": False},
        {"bin": 3, "channel": "eco", "p_value": 0.01, "n_eff": 70, "informative": True},
        {"bin": 3, "channel": "sensory", "p_value": 0.5, "n_eff": 70, "informative": True},
        {"bin": 4, "channel": "eco", "p_value": 0.002, "n_eff": 68, "informative": True},
        {"bin": 4, "channel": "sensory", "p_value": 0.9, "n_eff": 68, "informative": True},
    ])
    ref = stratified_reference_grid(stratified)
    ctx = stratified_fp_from_table(stratified, bin_diag, reference_grid=ref)
    assert ctx["n_cells"] == 4
    assert ctx["n_sig"] == 2
    assert ctx["common_grid"] is True
