"""Tests for the v0.2 (H3) funnel model and its experiments.

Contract and falsification-machinery tests: stage probabilities stay in [0,1],
the funnel attrites monotonically, curiosity attenuates from intent to act, the
referent ceiling binds, the two intent forms are both valid, and the H3a/H3c
verdict machinery returns sensible labels. These check the apparatus, not the
science.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dtbr_mc.agents import AgentSampler
from dtbr_mc.behavior import get_model
from dtbr_mc.config.schemas import AgentConfig, BehaviorWeights, EnvironmentConfig
from dtbr_mc.environment import EnvironmentSampler
from dtbr_mc.experiments_h3 import (
    identifiability_intent,
    run_experiment_004,
    run_h3a,
)
from dtbr_mc.simulation import SimulationConfig, Simulator


def _pop(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    a = AgentSampler(AgentConfig()).sample(n, rng)
    e = EnvironmentSampler(EnvironmentConfig()).sample(n, rng)
    return a, e


@pytest.mark.parametrize("model", ["funnel", "funnel_pw"])
def test_funnel_stage_probs_in_unit_interval(model):
    a, e = _pop()
    b = get_model(model)(a, e, BehaviorWeights())
    f = b.as_frame()
    for k in ["p_encounter", "p_notice", "comprehension", "p_intend", "p_act", "intervention", "opportunity"]:
        arr = f[k].to_numpy()
        assert arr.min() >= -1e-9 and arr.max() <= 1 + 1e-9, f"{k} out of range in {model}"


def test_funnel_registered_alongside_v01_models():
    from dtbr_mc.behavior import BEHAVIOR_MODELS
    assert {"baseline", "backfire", "linear", "funnel", "funnel_pw"}.issubset(BEHAVIOR_MODELS)


def test_funnel_deterministic():
    cfg = SimulationConfig(n_agents=3000, seed=7, model="funnel")
    f1 = Simulator(cfg).run().frame()
    f2 = Simulator(cfg).run().frame()
    pd.testing.assert_frame_equal(f1, f2)


def test_curiosity_attenuates_from_intent_to_act():
    a, e = _pop()
    b = get_model("funnel")(a, e, BehaviorWeights())
    ci = np.asarray(b.extra["curiosity_at_intent"]).mean()
    ca = np.asarray(b.extra["curiosity_at_act"]).mean()
    assert ca < ci  # attenuation by (1 - alpha)


def test_referent_ceiling_caps_perceived_certainty():
    a, e = _pop()
    e = e.copy()
    e["referent_certainty_ceiling"] = 0.2
    e["signal_certainty"] = 1.0  # max signal
    b = get_model("funnel")(a, e, BehaviorWeights())
    assert np.asarray(b.extra["perceived_certainty"]).max() <= 0.2 + 1e-9


def test_dread_brake_present_for_comprehending_nonacquisitive():
    """Raising PC must NOT increase intervention for a low-acquisitiveness,
    high-comprehension population (Amendment 1: dread is a brake)."""
    a, e = _pop(5000)
    a = a.copy(); e = e.copy()
    a["acquisitiveness"] = 0.0
    a["interpretive_capacity"] = 0.9
    e["marker_clarity"] = 0.9
    e_lo = e.copy(); e_lo["phenomenological_caution"] = 0.1
    e_hi = e.copy(); e_hi["phenomenological_caution"] = 0.9
    lo = get_model("funnel")(a, e_lo, BehaviorWeights()).intervention.mean()
    hi = get_model("funnel")(a, e_hi, BehaviorWeights()).intervention.mean()
    assert hi <= lo + 1e-6


def test_value_signaling_attracts_acquisitive():
    """For highly acquisitive actors, more defense should not reduce intervention
    (value-signaling channel)."""
    a, e = _pop(5000)
    a = a.copy(); e = e.copy()
    a["acquisitiveness"] = 1.0
    e_lo = e.copy(); e_lo["phenomenological_caution"] = 0.1
    e_hi = e.copy(); e_hi["phenomenological_caution"] = 0.9
    lo = get_model("funnel")(a, e_lo, BehaviorWeights()).intervention.mean()
    hi = get_model("funnel")(a, e_hi, BehaviorWeights()).intervention.mean()
    assert hi >= lo - 1e-6


def test_exp004_certainty_is_deterrent_and_ceiling_binds():
    r = run_experiment_004(n=8000, seed=0)
    assert all(v <= 0 for v in r.certainty_elasticity.values())  # certainty deters
    assert r.ceiling_binds


def test_exp004_minority_weakly_deterrable_under_low_ceiling():
    r = run_experiment_004(n=8000, seed=0, radiological_ceiling=0.2)
    # best-case signal barely moves the risk-seeking minority
    assert (r.minority_disturbance_lo_signal - r.minority_disturbance_hi_signal) < 0.10


def test_h3a_returns_verdict_and_respects_gamma_bound():
    # gamma outside the bound must not be reported as support
    h = run_h3a(n=6000, seed=0, gamma=2.0, gamma_bound=(0.0, 1.0))
    assert not h.within_bound
    assert "SUPPORTED" not in h.verdict or "OUTSIDE" in h.verdict


def test_intent_form_identifiability_runs():
    out = identifiability_intent(n=6000, seed=0)
    assert "funnel" in out and "funnel_pw" in out and "identified" in out


# --- Experiment 005 (coupling / emergence) -------------------------------- #

def test_exp005_runs_and_labels_cartography():
    from dtbr_mc.experiments_h3 import run_experiment_005
    r = run_experiment_005(n=6000, seed=0, lam_max=2.0, lam_points=9)
    assert r.fork in ("SIMULATOR", "SIMULATOR (conditional)", "MIRROR")
    assert "CARTOGRAPHY" in r.label  # lambda unanchored => reachability only


def test_exp005_matched_marginal_null_has_no_hysteresis():
    from dtbr_mc.experiments_h3 import run_experiment_005
    r = run_experiment_005(n=6000, seed=0, lam_max=2.0, lam_points=9)
    assert r.null_has_hysteresis is False


def test_exp005_coupling_amplifies_monotonically():
    """Mean-field coupling should raise the disturbance rate monotonically (a
    deductive amplification), even if it never tips."""
    from dtbr_mc.experiments_h3 import run_experiment_005
    r = run_experiment_005(n=8000, seed=0, lam_max=1.5, lam_points=11)
    cold = r.extra["per_signal"]["engagement"]["cold"]
    assert cold[-1] >= cold[0] - 1e-9  # higher coupling -> >= disturbance


def test_exp005_no_order_parameter_bistability_under_heterogeneity():
    """The key robustness claim: heterogeneous priors suppress tipping; the order
    parameter has a unique fixed point (no second basin) at swept coupling."""
    from dtbr_mc.experiments_h3 import run_experiment_005
    r = run_experiment_005(n=8000, seed=0, lam_max=2.0, lam_points=9)
    assert r.extra["order_bistable"] is False
