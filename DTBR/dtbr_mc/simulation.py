"""Monte Carlo orchestration.

``Simulator`` samples an agent population and a per-agent environment, evaluates
the active behavioral model, maps the intervention score through the outcome
state machine, and computes per-agent expected harm. Sampling and evaluation are
deliberately separated so that controlled sweeps can reuse one population
(common random numbers) and vary only the controls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dtbr_mc.agents import AgentSampler
from dtbr_mc.behavior import BehaviorResult, get_model
from dtbr_mc.config.schemas import OutcomeThresholds, SimulationConfig
from dtbr_mc.environment import EnvironmentSampler


@dataclass
class SimulationResult:
    agents: pd.DataFrame
    environment: pd.DataFrame
    behavior: BehaviorResult
    outcomes: np.ndarray          # dtype object, one label per agent
    expected_harm: np.ndarray     # per-agent E[H] contribution
    thresholds: OutcomeThresholds

    def frame(self) -> pd.DataFrame:
        """Wide DataFrame combining inputs, intermediates, outcomes, and harm."""
        out = pd.concat(
            [self.agents.reset_index(drop=True), self.environment.reset_index(drop=True)],
            axis=1,
        )
        bframe = self.behavior.as_frame().reset_index(drop=True)
        # avoid duplicate "curiosity" column name (agent trait vs curiosity drive)
        bframe = bframe.rename(columns={"curiosity": "curiosity_drive"})
        out = pd.concat([out, bframe], axis=1)
        out["outcome"] = self.outcomes
        out["expected_harm"] = self.expected_harm
        return out


def map_outcomes(intervention: np.ndarray, thresholds: OutcomeThresholds) -> np.ndarray:
    """Map intervention scores to discrete outcome labels (left-closed bins)."""
    inner = [thresholds.avoid, thresholds.observe, thresholds.preserve, thresholds.investigate]
    idx = np.digitize(intervention, inner, right=False)
    labels = np.asarray(thresholds.labels, dtype=object)
    return labels[idx]


class Simulator:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.agent_sampler = AgentSampler(config.agents)
        self.env_sampler = EnvironmentSampler(config.environment)

    # -- sampling -------------------------------------------------------- #

    def sample(
        self, n: int | None = None, seed: int | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Sample an agent population and matching per-agent environment.

        A single seeded ``Generator`` is threaded through both samplers, so the
        same ``seed`` reproduces the same population exactly.
        """
        n = n if n is not None else self.config.n_agents
        seed = seed if seed is not None else self.config.seed
        rng = np.random.default_rng(seed)
        agents = self.agent_sampler.sample(n, rng)
        env = self.env_sampler.sample(n, rng)
        return agents, env

    # -- evaluation (pure) ----------------------------------------------- #

    def evaluate(self, agents: pd.DataFrame, env: pd.DataFrame) -> SimulationResult:
        """Evaluate the active model on a (possibly overridden) population."""
        model = get_model(self.config.model)
        behavior = model(agents, env, self.config.weights)
        outcomes = map_outcomes(behavior.intervention, self.config.thresholds)
        expected_harm = (
            behavior.p_encounter
            * behavior.intervention
            * env["repository_severity"].to_numpy()
        )
        return SimulationResult(
            agents=agents,
            environment=env,
            behavior=behavior,
            outcomes=outcomes,
            expected_harm=expected_harm,
            thresholds=self.config.thresholds,
        )

    # -- convenience ----------------------------------------------------- #

    def run(
        self,
        n: int | None = None,
        seed: int | None = None,
        env_overrides: dict[str, float] | None = None,
        agent_overrides: dict[str, float] | None = None,
    ) -> SimulationResult:
        """Sample, apply scalar overrides (broadcast), and evaluate."""
        agents, env = self.sample(n=n, seed=seed)
        if agent_overrides:
            agents = agents.copy()
            for k, v in agent_overrides.items():
                agents[k] = float(v)
        if env_overrides:
            env = env.copy()
            for k, v in env_overrides.items():
                env[k] = float(v)
        return self.evaluate(agents, env)


__all__ = ["Simulator", "SimulationResult", "map_outcomes"]
