"""Configuration schemas for DTBR-MC.

Everything that governs a run is expressed as a (JSON-serializable) pydantic
model so that scenarios are reproducible and nothing is hardcoded inside the
simulation logic. The defaults reproduce the baseline specification.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# --------------------------------------------------------------------------- #
# Distributions
# --------------------------------------------------------------------------- #


class DistributionConfig(BaseModel):
    """A univariate distribution on (a subset of) the unit interval.

    Supports the families required by the specification. ``mixture`` references
    nested ``DistributionConfig`` components with mixing ``weights``.
    """

    kind: Literal["uniform", "normal", "beta", "mixture", "constant"] = "uniform"

    # uniform
    low: float = 0.0
    high: float = 1.0
    # normal (truncated by clip)
    mean: float = 0.5
    std: float = 0.15
    # beta
    a: float = 2.0
    b: float = 2.0
    # constant
    value: float = 0.5
    # mixture
    components: Optional[list["DistributionConfig"]] = None
    weights: Optional[list[float]] = None

    clip: tuple[float, float] = (0.0, 1.0)

    @model_validator(mode="after")
    def _check_mixture(self) -> "DistributionConfig":
        if self.kind == "mixture":
            if not self.components:
                raise ValueError("mixture distribution requires `components`")
            if self.weights is not None and len(self.weights) != len(self.components):
                raise ValueError("`weights` length must match `components`")
        return self


# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #

AGENT_VARIABLE_NAMES: tuple[str, ...] = (
    "curiosity",
    "risk_tolerance",
    "technical_capability",
    "interpretive_capacity",
    "institutional_strength",
    "prestige_sensitivity",
    "economic_pressure",
    "ritualization_tendency",
    # v0.2 (H3): explicit acquisitive motive -- the empirically-located sign-flip
    # moderator (warning-compliance vs grave-robbing). Distinct from economic
    # pressure: the disposition to *extract value*, not merely to be under strain.
    "acquisitiveness",
)


class CorrelationPair(BaseModel):
    """A target Pearson correlation between two agent variables (copula)."""

    var_a: str
    var_b: str
    rho: float = Field(..., ge=-0.999, le=0.999)


class ExplorerConfig(BaseModel):
    """An upper-tail explorer minority.

    A fraction of agents are resampled on selected traits from the upper tail of
    their marginal distribution, modelling the small subgroup that drives most
    high-risk intervention.
    """

    fraction: float = Field(0.05, ge=0.0, le=1.0)
    variables: list[str] = Field(
        default_factory=lambda: [
            "curiosity",
            "risk_tolerance",
            "technical_capability",
            "prestige_sensitivity",
            # v0.2: the explorer minority is the risk-seeking *offender*
            # subpopulation (Becker: certainty-aphorism => intruders are
            # risk-seeking). Load the boost onto acquisitiveness too.
            "acquisitiveness",
        ]
    )
    # explorer trait values are drawn uniformly from [lower_quantile, 1.0]
    lower_quantile: float = Field(0.8, ge=0.0, le=1.0)


class AgentConfig(BaseModel):
    variables: dict[str, DistributionConfig] = Field(default_factory=dict)
    explorer: ExplorerConfig = Field(default_factory=ExplorerConfig)
    correlations: list[CorrelationPair] = Field(default_factory=list)

    @model_validator(mode="after")
    def _fill_defaults(self) -> "AgentConfig":
        for name in AGENT_VARIABLE_NAMES:
            self.variables.setdefault(name, DistributionConfig(kind="uniform"))
        unknown = set(self.variables) - set(AGENT_VARIABLE_NAMES)
        if unknown:
            raise ValueError(f"unknown agent variables: {sorted(unknown)}")
        return self


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #

ENVIRONMENT_VARIABLE_NAMES: tuple[str, ...] = (
    "visibility",
    "accessibility",
    "resource_attractiveness",
    "marker_clarity",
    "phenomenological_caution",
    "artificial_intentionality",
    "prestige_risk",
    "historical_memory",
    "repository_severity",
    # v0.2 (H3): deterrence operates through perceived CERTAINTY of personal
    # consequence, not severity. signal_certainty is the designable marker lever
    # (how strongly the markers assert certain, immediate consequence).
    # referent_certainty_ceiling is a property of the HAZARD: the maximum
    # achievable perceived certainty given that the harm is latent/invisible/
    # delayed. For a radiological referent this ceiling is LOW, which is the
    # binding constraint H3c claims no message can overcome.
    "signal_certainty",
    "referent_certainty_ceiling",
)


class EnvironmentConfig(BaseModel):
    variables: dict[str, DistributionConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _fill_defaults(self) -> "EnvironmentConfig":
        for name in ENVIRONMENT_VARIABLE_NAMES:
            self.variables.setdefault(name, DistributionConfig(kind="uniform"))
        unknown = set(self.variables) - set(ENVIRONMENT_VARIABLE_NAMES)
        if unknown:
            raise ValueError(f"unknown environment variables: {sorted(unknown)}")
        return self


# --------------------------------------------------------------------------- #
# Behavioral weights
# --------------------------------------------------------------------------- #


class BehaviorWeights(BaseModel):
    """Replaceable coefficients for the behavioral equations.

    See ``dtbr_mc/behavior.py`` for the equations and the interpretation note on
    how the (ambiguous) specification was rendered into well-posed weighted
    combinations.
    """

    # Encounter: weighted combination of visibility & accessibility
    enc_visibility: float = 0.5
    enc_accessibility: float = 0.5

    # Comprehension: interpretive_capacity dominates, marker_clarity secondary
    comp_interpretive_capacity: float = 0.7
    comp_marker_clarity: float = 0.3

    # Curiosity (drive toward the object)
    cur_curiosity: float = 0.4
    cur_prestige_risk: float = 0.3
    cur_artificial_intentionality: float = 0.2
    cur_ritualization_tendency: float = 0.1

    # Caution (brake)
    caut_phenomenological_caution: float = 0.5
    caut_comprehension: float = 0.3
    caut_institutional_strength: float = 0.2

    # Intervention drive (push factors, combined then braked by caution)
    int_curiosity: float = 0.5
    int_economic_pressure: float = 0.3
    int_technical_capability: float = 0.2

    # Backfire / prestige-inversion channel (only used by the "backfire" model).
    # Strength with which an ominous-but-incomprehensible site inflates curiosity
    # via prestige sensitivity. 0.0 => no backfire (reduces to baseline curiosity).
    backfire_strength: float = 0.6

    # ----------------------------------------------------------------------- #
    # v0.2 (H3) C-HIP FUNNEL PARAMETERS (used only by the "funnel"/"funnel_pw"
    # models). Forms are pre-registered in SPEC_H3.md; defaults below are the
    # registered starting values. Where a parameter's only license is a bounded
    # or contested effect, the data-bound is noted -- experiments must keep the
    # parameter inside it (the anti-tautology guard).
    # ----------------------------------------------------------------------- #
    # Notice stage (warning lit: ~88% notice baseline)
    notice_base: float = 0.88
    notice_conspicuity: float = 0.10      # marker prominence raises noticing
    notice_load: float = 0.10             # cognitive load lowers noticing
    # Appraisal -- perceived value channels
    gamma_value_signaling: float = 0.5    # defense -> inferred worth (looting). BOUND: [0, 1.0]
    delta_info_reward: float = 0.4        # curiosity-as-reward via mystery
    rho_reactance: float = 0.3            # forbidden-fruit bump (upstream)
    # Appraisal -- perceived deterrence (certainty, NOT severity)
    cert_base: float = 0.1                # floor on perceived certainty pre-signal
    kappa_signal_certainty: float = 0.8   # how marker signal raises certainty
    w_deterrence: float = 1.0             # weight on perceived_certainty*consequence
    # AMENDMENT 1 (post-first-run, exploratory): phenomenological caution must
    # have a BRAKE pathway, not only backfire channels. Comprehended dread
    # (PC * comprehension) raises an affective hazard salience that deters; this
    # is the warning-label anchor (perceived hazard -> avoidance) and is also the
    # one channel that partly bypasses the certainty ceiling (immediate affect,
    # not cognitive risk calc). Uncomprehended PC remains mystery (backfire).
    dread_weight: float = 0.4             # affective brake from comprehended dread
    # Intent stage (Becker EU -> logistic), risk attitude shifts threshold
    intent_gain: float = 6.0              # logistic steepness a
    intent_threshold: float = 0.25        # base EU threshold theta_0
    intent_risk_shift: float = 0.25       # risk_tolerance lowers theta by up to this
    cost_weight: float = 0.2              # k_cost on (1 - accessibility)
    # Act stage: curiosity attenuation + social-modeling coupling
    alpha_attenuation: float = 0.2        # share of curiosity reaching ACTION.
                                          # BOUND small: lab curiosity robust,
                                          # field behavior ~null (media ratings).
    lambda_coupling: float = 0.0          # social modeling at act stage.
                                          # ASSUMPTION (magnitude): sweep, never
                                          # point-estimate. Default 0 = no coupling.
    # defense_level composite weights (what reads as "defended")
    defense_pc: float = 0.4
    defense_artificial: float = 0.3
    defense_prestige_risk: float = 0.3
    # Identifiability: intent functional form ("logistic" | "piecewise").
    # The two registered funnel models pin this; do not rely on a default.
    intent_form: str = "logistic"
    # Identifiability: deterrence functional form. "product" is the symmetric
    # Becker risk-neutral form (certainty*consequence) -- it does NOT bake in the
    # certainty-over-severity aphorism. "certainty_gated" weights certainty as a
    # necessary multiplicative gate with severity only modulating, encoding CAP.
    # H3c is tested for ROBUSTNESS across both; the ceiling result must hold in
    # either (SPEC_H3 7, H3c).
    deterrence_form: str = "product"
    cap_base: float = 0.6   # certainty_gated: share of deterrence independent of severity


# --------------------------------------------------------------------------- #
# Outcome state machine
# --------------------------------------------------------------------------- #


class OutcomeThresholds(BaseModel):
    """Cut points mapping the intervention score to a discrete outcome state.

    Bins (left-closed, right-open except the last): AVOID, OBSERVE, PRESERVE,
    INVESTIGATE, EXCAVATE.
    """

    avoid: float = 0.25
    observe: float = 0.45
    preserve: float = 0.65
    investigate: float = 0.85

    @model_validator(mode="after")
    def _monotone(self) -> "OutcomeThresholds":
        cuts = [self.avoid, self.observe, self.preserve, self.investigate]
        if any(b <= a for a, b in zip(cuts, cuts[1:])):
            raise ValueError("outcome thresholds must be strictly increasing")
        return self

    @property
    def edges(self) -> list[float]:
        return [0.0, self.avoid, self.observe, self.preserve, self.investigate, 1.0 + 1e-9]

    @property
    def labels(self) -> list[str]:
        return ["AVOID", "OBSERVE", "PRESERVE", "INVESTIGATE", "EXCAVATE"]


# Outcome states that constitute *physical disturbance* of the repository.
DISTURBANCE_STATES: tuple[str, ...] = ("INVESTIGATE", "EXCAVATE")


# --------------------------------------------------------------------------- #
# Simulation & Experiment
# --------------------------------------------------------------------------- #


class SimulationConfig(BaseModel):
    n_agents: int = Field(100_000, ge=1)
    seed: int = 0
    model: str = "baseline"
    weights: BehaviorWeights = Field(default_factory=BehaviorWeights)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    thresholds: OutcomeThresholds = Field(default_factory=OutcomeThresholds)


class SweepSpec(BaseModel):
    variable: str
    start: float = 0.0
    stop: float = 1.0
    num: int = Field(25, ge=2)


class ExperimentConfig(BaseModel):
    """Configuration for Experiment 001 and friends."""

    name: str = "experiment_001"
    n_agents: int = Field(100_000, ge=1)
    seed: int = 0
    model: str = "baseline"

    # Variables swept (Experiment 001: marker_clarity, phenomenological_caution,
    # interpretive_capacity). Each becomes a population-level control during the
    # sweep, holding all other random traits fixed (common random numbers).
    sweeps: list[SweepSpec] = Field(
        default_factory=lambda: [
            SweepSpec(variable="marker_clarity", num=25),
            SweepSpec(variable="phenomenological_caution", num=25),
            SweepSpec(variable="interpretive_capacity", num=25),
        ]
    )
    # Discrete interpretive-capacity slices used for 2-D phase heatmaps.
    ic_levels: list[float] = Field(default_factory=lambda: [0.15, 0.5, 0.85])

    bootstrap_n: int = Field(1000, ge=0)
    bootstrap_ci: float = Field(0.95, gt=0.0, lt=1.0)

    # Optional behaviour-coefficient override. When None the model defaults are
    # used; supply a BehaviorWeights to e.g. raise backfire_strength and probe
    # whether the framework can produce an H1 falsification.
    weights: Optional[BehaviorWeights] = None

    output_dir: str = "outputs"


DistributionConfig.model_rebuild()


__all__ = [
    "DistributionConfig",
    "CorrelationPair",
    "ExplorerConfig",
    "AgentConfig",
    "EnvironmentConfig",
    "BehaviorWeights",
    "OutcomeThresholds",
    "DISTURBANCE_STATES",
    "SimulationConfig",
    "SweepSpec",
    "ExperimentConfig",
    "AGENT_VARIABLE_NAMES",
    "ENVIRONMENT_VARIABLE_NAMES",
]
