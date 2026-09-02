"""Historical calibration hooks.

Per the specification we do **not** calibrate automatically. This module only
provides the *interfaces* for loading externally-specified priors and scenarios,
so that a future user can plug in historically-informed distributions without
touching the simulator core.

A :class:`Scenario` bundles an agent prior, an environment prior, a behaviour
model name + weights, and free-form provenance notes. Scenarios are plain JSON
and round-trip through pydantic. The bundled files under ``examples/`` are
illustrative placeholders, NOT empirically calibrated — every value there is a
modelling stipulation and is labelled as such in its ``notes``.

Example
-------
>>> from dtbr_mc.calibration import load_scenario
>>> scn = load_scenario("examples/scenario_low_capacity_future.json")
>>> sim_cfg = scn.to_simulation_config(n_agents=10_000, seed=0)
"""

from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel, Field

from dtbr_mc.config.schemas import (
    AgentConfig,
    BehaviorWeights,
    EnvironmentConfig,
    SimulationConfig,
)


class Scenario(BaseModel):
    """A named, fully-specified set of priors for one simulated world.

    Nothing here is fitted to data. ``notes`` should record the provenance of
    every non-default choice so that calibration claims are auditable.
    """

    name: str
    description: str = ""
    notes: str = ""
    calibrated: bool = Field(
        default=False,
        description="MUST stay False unless the priors were fitted to real data. "
        "The shipped examples are stipulations, not calibrations.",
    )

    agents: AgentConfig = Field(default_factory=AgentConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    model: str = "baseline"
    weights: Optional[BehaviorWeights] = None

    def to_simulation_config(self, n_agents: int = 100_000, seed: int = 0) -> SimulationConfig:
        """Materialise a SimulationConfig from this scenario's priors."""
        update = {
            "n_agents": n_agents,
            "seed": seed,
            "model": self.model,
            "agents": self.agents,
            "environment": self.environment,
        }
        if self.weights is not None:
            update["weights"] = self.weights
        return SimulationConfig(**update)


def load_scenario(path: str) -> Scenario:
    """Load a :class:`Scenario` from a JSON file. Interface only — no fitting."""
    with open(path) as fh:
        data = json.load(fh)
    return Scenario.model_validate(data)


def save_scenario(scenario: Scenario, path: str) -> str:
    """Persist a scenario to JSON (round-trips with :func:`load_scenario`)."""
    with open(path, "w") as fh:
        fh.write(scenario.model_dump_json(indent=2))
    return path


def load_agent_prior(path: str) -> AgentConfig:
    """Load an agent prior (distribution set) from JSON. Interface only."""
    with open(path) as fh:
        return AgentConfig.model_validate(json.load(fh))


def load_environment_prior(path: str) -> EnvironmentConfig:
    """Load an environment prior from JSON. Interface only."""
    with open(path) as fh:
        return EnvironmentConfig.model_validate(json.load(fh))


__all__ = [
    "Scenario",
    "load_scenario",
    "save_scenario",
    "load_agent_prior",
    "load_environment_prior",
]
