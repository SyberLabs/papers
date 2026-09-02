"""DTBR-MC: Deep-Time Behavioral Risk Monte Carlo Simulator.

A falsification-oriented Monte Carlo framework for evaluating long-term hazard
communication strategies (inspired by nuclear semiotics) under interpretive
uncertainty.

The objective being minimized is the *Expected Harm*:

    E[H] = P(Encounter) * P(Intervention | Encounter) * Severity

The package is deliberately built to *search for failure* of its own central
hypothesis (H1) rather than to confirm it. See ``README.md`` and
``dtbr_mc/behavior.py`` for the explicit interpretation notes on the baseline
equations supplied in the specification.
"""

from __future__ import annotations

__version__ = "0.1.0"

from dtbr_mc.config import (  # noqa: F401
    AgentConfig,
    BehaviorWeights,
    DistributionConfig,
    EnvironmentConfig,
    ExperimentConfig,
    OutcomeThresholds,
    SimulationConfig,
)
from dtbr_mc.agents import AGENT_VARIABLES, AgentSampler  # noqa: F401
from dtbr_mc.environment import ENVIRONMENT_VARIABLES, EnvironmentSampler  # noqa: F401
from dtbr_mc.behavior import BEHAVIOR_MODELS, BehaviorResult, register_model  # noqa: F401
from dtbr_mc.simulation import SimulationResult, Simulator  # noqa: F401

__all__ = [
    "__version__",
    "AgentConfig",
    "BehaviorWeights",
    "DistributionConfig",
    "EnvironmentConfig",
    "ExperimentConfig",
    "OutcomeThresholds",
    "SimulationConfig",
    "AGENT_VARIABLES",
    "AgentSampler",
    "ENVIRONMENT_VARIABLES",
    "EnvironmentSampler",
    "BEHAVIOR_MODELS",
    "BehaviorResult",
    "register_model",
    "SimulationResult",
    "Simulator",
]
