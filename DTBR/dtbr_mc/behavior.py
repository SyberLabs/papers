"""Behavioral logic.

This module turns agent traits + environment variables into an *intervention
score* in [0, 1], passing through transparent intermediates (encounter,
comprehension, curiosity, caution). Every model is registered by name and is
trivially replaceable; the active model is chosen by ``SimulationConfig.model``.

--------------------------------------------------------------------------------
INTERPRETATION NOTE (read before trusting any number)
--------------------------------------------------------------------------------
The specification wrote the baseline equations with ``*`` between every term,
e.g. ``P_encounter = 0.5*visibility * 0.5*accessibility``. Taken literally this
is the product ``0.25 * visibility * accessibility`` (always <= 0.25), and the
intervention line ``int = cur * 0.5*economic_pressure * 0.3*technical_capability
* caut`` would make caution *increase* intervention -- which is incoherent with
the entire research question ("does increasing caution *reduce* intervention").

Because (a) in four of the five equations the per-term coefficients sum exactly
to 1.0 and (b) the research question requires caution to act as a brake, the
coefficients are read as a **weighted linear combination**, and caution is
applied as a **multiplicative brake** ``intervention = drive * (1 - caution)``.
This is a deliberate, documented choice -- not silent invention. The weights and
the functional form are fully configurable, so the literal reading (or any
other) can be reinstated by registering a different model.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from dtbr_mc.config.schemas import BehaviorWeights


@dataclass
class BehaviorResult:
    """Vectorized behavioral intermediates and the final intervention score.

    All arrays are length ``n`` and clipped to [0, 1]. Intermediates are exposed
    so that experiments and metrics can inspect *why* an outcome occurred.
    """

    p_encounter: np.ndarray
    comprehension: np.ndarray
    curiosity: np.ndarray
    caution: np.ndarray
    drive: np.ndarray
    intervention: np.ndarray
    mystery: np.ndarray  # phenomenological_caution * (1 - comprehension)
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    def as_frame(self) -> pd.DataFrame:
        data = {
            "p_encounter": self.p_encounter,
            "comprehension": self.comprehension,
            "curiosity": self.curiosity,
            "caution": self.caution,
            "drive": self.drive,
            "intervention": self.intervention,
            "mystery": self.mystery,
            **self.extra,
        }
        return pd.DataFrame(data)


BehaviorModel = Callable[[pd.DataFrame, pd.DataFrame, BehaviorWeights], BehaviorResult]
BEHAVIOR_MODELS: dict[str, BehaviorModel] = {}


def register_model(name: str) -> Callable[[BehaviorModel], BehaviorModel]:
    """Register a behavioral model under ``name`` (used by ``SimulationConfig.model``)."""

    def deco(fn: BehaviorModel) -> BehaviorModel:
        if name in BEHAVIOR_MODELS:
            raise ValueError(f"behavior model already registered: {name}")
        BEHAVIOR_MODELS[name] = fn
        return fn

    return deco


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _common(a: pd.DataFrame, e: pd.DataFrame, w: BehaviorWeights):
    """Shared intermediates used by all models."""
    p_encounter = _clip(
        w.enc_visibility * e["visibility"].to_numpy()
        + w.enc_accessibility * e["accessibility"].to_numpy()
    )
    comprehension = _clip(
        w.comp_interpretive_capacity * a["interpretive_capacity"].to_numpy()
        + w.comp_marker_clarity * e["marker_clarity"].to_numpy()
    )
    curiosity_base = _clip(
        w.cur_curiosity * a["curiosity"].to_numpy()
        + w.cur_prestige_risk * e["prestige_risk"].to_numpy()
        + w.cur_artificial_intentionality * e["artificial_intentionality"].to_numpy()
        + w.cur_ritualization_tendency * a["ritualization_tendency"].to_numpy()
    )
    mystery = _clip(e["phenomenological_caution"].to_numpy() * (1.0 - comprehension))
    return p_encounter, comprehension, curiosity_base, mystery


def _caution(a: pd.DataFrame, e: pd.DataFrame, w: BehaviorWeights, comprehension: np.ndarray) -> np.ndarray:
    return _clip(
        w.caut_phenomenological_caution * e["phenomenological_caution"].to_numpy()
        + w.caut_comprehension * comprehension
        + w.caut_institutional_strength * a["institutional_strength"].to_numpy()
    )


def _drive(a: pd.DataFrame, w: BehaviorWeights, curiosity: np.ndarray) -> np.ndarray:
    return _clip(
        w.int_curiosity * curiosity
        + w.int_economic_pressure * a["economic_pressure"].to_numpy()
        + w.int_technical_capability * a["technical_capability"].to_numpy()
    )


@register_model("baseline")
def baseline_model(a: pd.DataFrame, e: pd.DataFrame, w: BehaviorWeights) -> BehaviorResult:
    """Caution acts purely as a brake; no prestige-inversion backfire.

    This is the "intended" reading. Because comprehension is gated 0.7 on
    interpretive capacity, the semantic-clarity channel into caution is throttled
    when capacity is low, whereas phenomenological caution feeds caution directly.
    """
    p_encounter, comprehension, curiosity, mystery = _common(a, e, w)
    caution = _caution(a, e, w, comprehension)
    drive = _drive(a, w, curiosity)
    intervention = _clip(drive * (1.0 - caution))
    return BehaviorResult(p_encounter, comprehension, curiosity, caution, drive, intervention, mystery)


@register_model("backfire")
def backfire_model(a: pd.DataFrame, e: pd.DataFrame, w: BehaviorWeights) -> BehaviorResult:
    """Prestige-inversion: ominous-but-incomprehensible sites inflate curiosity.

    Encodes the *Alternative* hypothesis. Curiosity gains
    ``backfire_strength * prestige_sensitivity * mystery`` where
    ``mystery = phenomenological_caution * (1 - comprehension)``. When capacity
    is high, comprehension kills the mystery term and this reduces to baseline;
    when capacity is low, a louder phenomenological warning can *attract* the
    explorer minority -- the mechanism that can falsify H1.
    """
    p_encounter, comprehension, curiosity_base, mystery = _common(a, e, w)
    curiosity = _clip(
        curiosity_base
        + w.backfire_strength * a["prestige_sensitivity"].to_numpy() * mystery
    )
    caution = _caution(a, e, w, comprehension)
    drive = _drive(a, w, curiosity)
    intervention = _clip(drive * (1.0 - caution))
    return BehaviorResult(p_encounter, comprehension, curiosity, caution, drive, intervention, mystery)


@register_model("linear")
def linear_model(a: pd.DataFrame, e: pd.DataFrame, w: BehaviorWeights) -> BehaviorResult:
    """Robustness variant: additive brake ``intervention = drive - caution``.

    Tests whether conclusions depend on the multiplicative ``(1 - caution)`` form.
    """
    p_encounter, comprehension, curiosity, mystery = _common(a, e, w)
    caution = _caution(a, e, w, comprehension)
    drive = _drive(a, w, curiosity)
    intervention = _clip(drive - caution)
    return BehaviorResult(p_encounter, comprehension, curiosity, caution, drive, intervention, mystery)


# --------------------------------------------------------------------------- #
# v0.2 (H3): the C-HIP staged funnel
#
# encounter -> notice -> comprehend -> appraise(value vs deterrence) -> intend -> act
#
# Forms are pre-registered in SPEC_H3.md (section 5). The model leaves the v0.1
# additive models intact as the null it must beat. It maps its terminal pass-
# probability onto BehaviorResult.intervention (= P(disturb | encountered)) so
# the existing simulation/metrics pipeline is unchanged, and exposes every stage
# quantity in `extra` so the falsification tests can inspect attrition directly.
# --------------------------------------------------------------------------- #


def _link(eu: np.ndarray, theta: np.ndarray, gain: float, form: str) -> np.ndarray:
    """Map expected utility to a pass-probability. Two defensible forms for the
    identifiability protocol (SPEC_H3 section 7)."""
    z = gain * (eu - theta)
    if form == "logistic":
        return 1.0 / (1.0 + np.exp(-z))
    if form == "piecewise":
        # gentler linear ramp through (theta, 0.5), clipped to [0,1]
        return _clip(0.5 + 0.5 * z)
    raise ValueError(f"unknown intent_form: {form!r}")


def _funnel_core(a: pd.DataFrame, e: pd.DataFrame, w: BehaviorWeights, form: str) -> BehaviorResult:
    g = lambda k: e[k].to_numpy()
    h = lambda k: a[k].to_numpy()

    # -- encounter / notice ------------------------------------------------- #
    p_encounter = _clip(w.enc_visibility * g("visibility") + w.enc_accessibility * g("accessibility"))
    conspicuity = g("artificial_intentionality")
    load = 1.0 - g("marker_clarity")  # an illegible marker imposes processing load
    p_notice = _clip(w.notice_base + w.notice_conspicuity * conspicuity - w.notice_load * load)

    # -- comprehend (the ONLY interpretive-capacity-gated stage) ------------ #
    c = _clip(w.comp_interpretive_capacity * h("interpretive_capacity")
              + w.comp_marker_clarity * g("marker_clarity"))

    # -- appraise: perceived VALUE (where both backfire channels live) ------ #
    defense_level = _clip(w.defense_pc * g("phenomenological_caution")
                          + w.defense_artificial * g("artificial_intentionality")
                          + w.defense_prestige_risk * g("prestige_risk"))
    acq = h("acquisitiveness")
    material_value = g("resource_attractiveness") * acq
    value_signaling = w.gamma_value_signaling * defense_level * acq      # channel 2 (looting)
    mystery = _clip(g("phenomenological_caution") * (1.0 - c))
    info_reward = w.delta_info_reward * mystery * h("curiosity")         # channel 3a
    prohibition_salience = _clip(0.5 * g("marker_clarity") + 0.5 * g("phenomenological_caution"))
    reactance_bump = w.rho_reactance * prohibition_salience              # channel 3b (near-universal)
    curiosity_value = info_reward + reactance_bump                       # the attenuating channels
    perceived_value = material_value + value_signaling + curiosity_value

    # -- appraise: perceived DETERRENCE (certainty, NOT severity) ----------- #
    # achievable certainty is capped by the referent: no message exceeds it.
    perceived_certainty = np.minimum(
        g("referent_certainty_ceiling"),
        c * (w.cert_base + w.kappa_signal_certainty * g("signal_certainty")),
    )
    perceived_consequence = c * g("repository_severity")                 # comprehension-gated self-harm
    # AMENDMENT 1: comprehended dread is an affective brake (warning-label anchor).
    # Same cue: dread for those who comprehend, mystery for those who don't.
    hazard_salience = g("phenomenological_caution") * c
    if w.deterrence_form == "product":
        # symmetric Becker risk-neutral expected cost (does NOT assume CAP)
        perceived_deterrence = (w.w_deterrence * perceived_certainty * perceived_consequence
                                + w.dread_weight * hazard_salience)
    elif w.deterrence_form == "certainty_gated":
        # certainty is a necessary gate; severity only modulates (encodes CAP)
        perceived_deterrence = (
            w.w_deterrence * perceived_certainty
            * (w.cap_base + (1.0 - w.cap_base) * perceived_consequence)
            + w.dread_weight * hazard_salience
        )
    else:
        raise ValueError(f"unknown deterrence_form: {w.deterrence_form!r}")

    # -- intent (Becker EU -> link), risk attitude lowers the threshold ----- #
    cost = w.cost_weight * (1.0 - g("accessibility"))
    eu_intent = perceived_value - perceived_deterrence - cost
    theta = w.intent_threshold - w.intent_risk_shift * h("risk_tolerance")
    p_intend = _link(eu_intent, theta, w.intent_gain, form)

    # -- act: curiosity attenuates from intent to action; then coupling ----- #
    # value_at_act discounts the curiosity channels by (1 - alpha): lab curiosity
    # is robust, field behaviour ~null. EU_act <= EU_intent, so link composes to
    # p_act_solo = link(EU_act) directly (see SPEC_H3 5.6 derivation).
    eu_act = perceived_value - (1.0 - w.alpha_attenuation) * curiosity_value - perceived_deterrence - cost
    opportunity = _clip(g("accessibility") * (0.5 + 0.5 * h("technical_capability")))
    p_act_solo = _link(eu_act, theta, w.intent_gain, form) * opportunity

    if w.lambda_coupling != 0.0:
        # First-order mean-field coupling placeholder (social modeling). The
        # graph-based forms (lattice / random graph) belong to Exp 005's
        # identifiability check; this mean-field term is ASSUMPTION-grade.
        neighbor_act_fraction = float(np.mean(p_act_solo))
        p_act = _clip(p_act_solo * (1.0 + w.lambda_coupling * neighbor_act_fraction))
    else:
        p_act = p_act_solo

    # conditional-on-encounter disturbance propensity (notice gates the funnel)
    intervention = _clip(p_notice * p_act)

    extra = {
        "p_notice": p_notice,
        "defense_level": defense_level,
        "material_value": material_value,
        "value_signaling": value_signaling,
        "info_reward": info_reward,
        "reactance_bump": reactance_bump,
        "curiosity_value": curiosity_value,
        "perceived_value": perceived_value,
        "perceived_certainty": perceived_certainty,
        "perceived_consequence": perceived_consequence,
        "perceived_deterrence": perceived_deterrence,
        "hazard_salience": hazard_salience,
        "eu_intent": eu_intent,
        "eu_act": eu_act,
        "theta": theta,
        "p_intend": p_intend,
        "p_act": p_act,
        "opportunity": opportunity,
        # H3b diagnostics: curiosity contribution surviving to each stage
        "curiosity_at_intent": curiosity_value,
        "curiosity_at_act": w.alpha_attenuation * curiosity_value,
    }
    # Map onto the v0.1 BehaviorResult shape (bounded views in the named fields).
    return BehaviorResult(
        p_encounter=p_encounter,
        comprehension=c,
        curiosity=_clip(perceived_value),
        caution=_clip(perceived_deterrence),
        drive=p_intend,
        intervention=intervention,
        mystery=mystery,
        extra=extra,
    )


@register_model("funnel")
def funnel_model(a: pd.DataFrame, e: pd.DataFrame, w: BehaviorWeights) -> BehaviorResult:
    """C-HIP staged funnel with logistic intent (H3 default)."""
    return _funnel_core(a, e, w, form="logistic")


@register_model("funnel_pw")
def funnel_pw_model(a: pd.DataFrame, e: pd.DataFrame, w: BehaviorWeights) -> BehaviorResult:
    """Identifiability twin of ``funnel`` with a piecewise-linear intent form.

    If ``funnel`` and ``funnel_pw`` give observationally indistinguishable
    outputs for a claim, that claim is non-identified w.r.t. the intent form and
    must be reported as such (SPEC_H3 section 7)."""
    return _funnel_core(a, e, w, form="piecewise")


def get_model(name: str) -> BehaviorModel:
    if name not in BEHAVIOR_MODELS:
        raise KeyError(
            f"unknown behavior model {name!r}; available: {sorted(BEHAVIOR_MODELS)}"
        )
    return BEHAVIOR_MODELS[name]


__all__ = [
    "BehaviorResult",
    "BehaviorModel",
    "BEHAVIOR_MODELS",
    "register_model",
    "get_model",
    "baseline_model",
    "backfire_model",
    "linear_model",
    "funnel_model",
    "funnel_pw_model",
]
