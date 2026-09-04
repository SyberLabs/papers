"""Test suite for DTBR-MC.

These are correctness and contract tests, not statistical validation of the
science. They check: determinism, that every behavioural quantity stays in
[0,1], sampling shapes, the registry, outcome mapping edges, metric sanity, and
that the H1 verdict machinery can return each possible verdict (so the test is
genuinely falsifiable).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dtbr_mc.agents import AgentSampler
from dtbr_mc.behavior import BEHAVIOR_MODELS, get_model, register_model
from dtbr_mc.config.schemas import (
    AGENT_VARIABLE_NAMES,
    ENVIRONMENT_VARIABLE_NAMES,
    AgentConfig,
    BehaviorWeights,
    EnvironmentConfig,
    ExperimentConfig,
    OutcomeThresholds,
    SimulationConfig,
)
from dtbr_mc.environment import EnvironmentSampler
from dtbr_mc.experiments import run_experiment_001, sensitivity_analysis
from dtbr_mc.metrics import compute_metrics, point_metrics
from dtbr_mc.simulation import Simulator, map_outcomes


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #


def test_agent_sampling_shape_and_columns():
    rng = np.random.default_rng(0)
    df = AgentSampler(AgentConfig()).sample(500, rng)
    assert len(df) == 500
    for name in AGENT_VARIABLE_NAMES:
        assert name in df.columns
    assert "is_explorer" in df.columns


def test_environment_sampling_shape_and_columns():
    rng = np.random.default_rng(0)
    df = EnvironmentSampler(EnvironmentConfig()).sample(500, rng)
    assert len(df) == 500
    for name in ENVIRONMENT_VARIABLE_NAMES:
        assert name in df.columns


def test_sampled_values_in_unit_interval():
    rng = np.random.default_rng(1)
    a = AgentSampler(AgentConfig()).sample(2000, rng)
    e = EnvironmentSampler(EnvironmentConfig()).sample(2000, rng)
    for name in AGENT_VARIABLE_NAMES:
        assert a[name].min() >= 0.0 and a[name].max() <= 1.0
    for name in ENVIRONMENT_VARIABLE_NAMES:
        assert e[name].min() >= 0.0 and e[name].max() <= 1.0


def test_environment_override_pins_constant():
    rng = np.random.default_rng(0)
    e = EnvironmentSampler(EnvironmentConfig()).sample(100, rng, overrides={"marker_clarity": 0.3})
    assert np.allclose(e["marker_clarity"].to_numpy(), 0.3)


def test_explorer_fraction_default():
    rng = np.random.default_rng(0)
    a = AgentSampler(AgentConfig()).sample(10000, rng)
    frac = a["is_explorer"].mean()
    assert 0.02 < frac < 0.08  # ~5% default


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #


def test_simulation_deterministic_same_seed():
    cfg = SimulationConfig(n_agents=3000, seed=42, model="baseline")
    f1 = Simulator(cfg).run().frame()
    f2 = Simulator(cfg).run().frame()
    pd.testing.assert_frame_equal(f1, f2)


def test_simulation_differs_with_seed():
    f1 = Simulator(SimulationConfig(n_agents=3000, seed=1)).run().frame()
    f2 = Simulator(SimulationConfig(n_agents=3000, seed=2)).run().frame()
    assert not np.allclose(f1["intervention"].to_numpy(), f2["intervention"].to_numpy())


# --------------------------------------------------------------------------- #
# behaviour models
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_name", ["baseline", "backfire", "linear"])
def test_behavior_outputs_in_unit_interval(model_name):
    rng = np.random.default_rng(0)
    a = AgentSampler(AgentConfig()).sample(3000, rng)
    e = EnvironmentSampler(EnvironmentConfig()).sample(3000, rng)
    model = get_model(model_name)
    b = model(a, e, BehaviorWeights())
    for arr in [b.p_encounter, b.comprehension, b.curiosity, b.caution, b.drive, b.intervention]:
        arr = np.asarray(arr)
        assert arr.min() >= 0.0 - 1e-9
        assert arr.max() <= 1.0 + 1e-9


def test_registry_contains_models_and_raises_on_unknown():
    assert {"baseline", "backfire", "linear"}.issubset(BEHAVIOR_MODELS.keys())
    with pytest.raises(KeyError):
        get_model("does_not_exist")


def test_register_model_roundtrip():
    @register_model("tmp_identity_model")
    def _m(agents, env, weights):
        return get_model("baseline")(agents, env, weights)

    assert "tmp_identity_model" in BEHAVIOR_MODELS
    del BEHAVIOR_MODELS["tmp_identity_model"]


def test_baseline_caution_is_a_brake():
    """Raising phenomenological_caution must not increase intervention (baseline)."""
    rng = np.random.default_rng(0)
    a = AgentSampler(AgentConfig()).sample(4000, rng)
    e = EnvironmentSampler(EnvironmentConfig()).sample(4000, rng)
    model = get_model("baseline")
    e_lo = e.copy(); e_lo["phenomenological_caution"] = 0.1
    e_hi = e.copy(); e_hi["phenomenological_caution"] = 0.9
    lo = model(a, e_lo, BehaviorWeights()).intervention.mean()
    hi = model(a, e_hi, BehaviorWeights()).intervention.mean()
    assert hi <= lo


# --------------------------------------------------------------------------- #
# outcome mapping
# --------------------------------------------------------------------------- #


def test_outcome_mapping_edges():
    thr = OutcomeThresholds()
    scores = np.array([0.0, 0.24, 0.25, 0.44, 0.45, 0.64, 0.65, 0.84, 0.85, 1.0])
    outs = map_outcomes(scores, thr)
    expected = ["AVOID", "AVOID", "OBSERVE", "OBSERVE", "PRESERVE",
                "PRESERVE", "INVESTIGATE", "INVESTIGATE", "EXCAVATE", "EXCAVATE"]
    assert list(outs) == expected


def test_outcome_labels_and_edges_consistent():
    thr = OutcomeThresholds()
    assert len(thr.labels) == 5
    assert thr.edges == sorted(thr.edges)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def test_metrics_present_and_finite():
    res = Simulator(SimulationConfig(n_agents=4000, seed=0)).run()
    m = compute_metrics(res, bootstrap_n=100, ci=0.95, seed=1)
    required = {
        "expected_harm", "encounter_rate", "intervention_rate", "excavation_rate",
        "avoidance_rate", "preservation_rate", "mean_hesitation_proxy",
        "mystery_to_curiosity_index", "prestige_inversion_index",
        "behavioral_degradation_gradient",
    }
    assert required.issubset(set(m.index))
    assert np.isfinite(m.loc["expected_harm", "estimate"])


def test_rates_are_probabilities():
    res = Simulator(SimulationConfig(n_agents=4000, seed=0)).run()
    pm = point_metrics(res)
    for k in ["encounter_rate", "intervention_rate", "excavation_rate",
              "avoidance_rate", "preservation_rate"]:
        assert 0.0 <= pm[k] <= 1.0


def test_bootstrap_ci_brackets_estimate():
    res = Simulator(SimulationConfig(n_agents=4000, seed=0)).run()
    m = compute_metrics(res, bootstrap_n=300, ci=0.95, seed=1)
    row = m.loc["expected_harm"]
    assert row["ci_lo"] <= row["estimate"] <= row["ci_hi"]


# --------------------------------------------------------------------------- #
# experiment + H1 verdict machinery
# --------------------------------------------------------------------------- #


def test_experiment_runs_and_has_h1():
    cfg = ExperimentConfig(n_agents=2000, seed=0, model="baseline", bootstrap_n=0)
    r = run_experiment_001(cfg, heatmap_n=1500)
    assert not r.heatmaps.empty
    assert not r.h1.table.empty
    assert isinstance(r.h1.verdict, str)


def test_h1_can_be_falsified():
    """With PC's brake disabled and strong backfire, PC should raise intervention,
    yielding an H1 FALSIFIED verdict: proving the test is not rigged."""
    w = BehaviorWeights(caut_phenomenological_caution=0.0, caut_comprehension=0.6,
                        caut_institutional_strength=0.4, backfire_strength=2.0)
    cfg = ExperimentConfig(n_agents=2000, seed=0, model="backfire", bootstrap_n=0, weights=w)
    r = run_experiment_001(cfg, heatmap_n=1500)
    assert "FALSIFIED" in r.h1.verdict


def test_sensitivity_ranks_severity_high():
    s = sensitivity_analysis(model_name="baseline", n_base=512, seed=0)
    ranking = s["ranking"]
    assert "repository_severity" in ranking.index
    # severity is a direct multiplicative factor -> should be a top driver of E[H]
    top3 = set(ranking.head(3).index)
    assert "repository_severity" in top3
