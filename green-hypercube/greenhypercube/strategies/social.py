"""Social transmission networks.

Hypothesis: discovery is distributed. A population of foragers explores in
parallel and shares successes across a social network, so a useful plant found
by one person biases the attention of their network neighbors. Each agent scores
species by ecological/phylogenetic association to the hits it has *heard about*
(its own plus transmitted ones). Transmission is imperfect and structured by the
network topology, so connectivity and population size shape efficiency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from .base import Strategy
from ..hypercube import Manifold
from ..utils.rng import RNG

if TYPE_CHECKING:
    from ..simulation.environment import Environment


class SocialSearch(Strategy):
    def __init__(
        self,
        manifold: Manifold,
        rng: RNG,
        n_agents: int = 8,
        topology: str = "watts_strogatz",
        transmission_prob: float = 0.4,
        epsilon: float = 0.1,
        phylo_weight: float = 0.5,
        sensory_prior: float = 0.25,
        ws_k: int = 4,
        ws_p: float = 0.2,
        **params,
    ):
        super().__init__(manifold, rng, **params)
        self.n_agents = int(n_agents)
        self.transmission_prob = transmission_prob
        self.epsilon = epsilon
        self.phylo_weight = phylo_weight
        self.A = manifold.eco_adj + manifold.bio_adj
        self.sim = manifold.phylo_similarity()
        sal = manifold.sensory_salience.astype(np.float64)
        self.prior = sensory_prior * (sal / (sal.max() or 1.0))
        self.network = self._build_network(topology, ws_k, ws_p)
        self.reset()

    def _build_network(self, topology: str, k: int, p: float) -> nx.Graph:
        seed = int(self.rng.integers(0, 2**31 - 1))
        n = self.n_agents
        if topology == "complete":
            return nx.complete_graph(n)
        if topology == "ring":
            return nx.cycle_graph(n)
        if topology == "star":
            return nx.star_graph(n - 1)
        # default: small-world
        k = min(max(2, k), n - 1)
        if k % 2 == 1:
            k -= 1
        k = max(2, k)
        return nx.watts_strogatz_graph(n, k, p, seed=seed)

    def reset(self) -> None:
        self.known: list[set[int]] = [set() for _ in range(self.n_agents)]
        # Pending transmissions delivered at the start of the next round.
        self._pending: list[set[int]] = [set() for _ in range(self.n_agents)]
        self._agent_of_idx: dict[int, int] = {}

    def _agent_score(self, agent: int) -> np.ndarray:
        s = self.prior.copy()
        known = self.known[agent]
        if known:
            hit_idx = np.array(sorted(known), dtype=int)
            s = s + self.A[:, hit_idx].sum(axis=1)
            s = s + self.phylo_weight * self.sim[:, hit_idx].max(axis=1)
        return s

    def propose_batch(self, env: Environment) -> list[int]:
        # Deliver transmissions queued from the previous round.
        for a in range(self.n_agents):
            self.known[a] |= self._pending[a]
            self._pending[a].clear()

        untested = env.untested_indices()
        if untested.size == 0:
            return []
        chosen: list[int] = []
        chosen_set: set[int] = set()
        avail = np.ones(self.m.n, dtype=bool)
        avail[env.tested] = False

        for a in range(min(self.n_agents, int(avail.sum()))):
            cand = np.flatnonzero(avail)
            if cand.size == 0:
                break
            if self.rng.random() < self.epsilon:
                pick = int(self.rng.choice(cand))
            else:
                scores = self._agent_score(a)
                masked = np.full(self.m.n, -np.inf)
                masked[cand] = scores[cand] + self.rng.normal(0, 1e-6, size=cand.size)
                pick = int(np.argmax(masked))
            chosen.append(pick)
            chosen_set.add(pick)
            avail[pick] = False
            self._agent_of_idx[pick] = a
        return chosen

    def observe(self, idx: int, reward: float) -> None:
        agent = self._agent_of_idx.get(idx, 0)
        if reward >= self.m.discovery_threshold:
            self.known[agent].add(idx)
            # Transmit to network neighbors with the transmission probability.
            for nb in self.network.neighbors(agent):
                if self.rng.random() < self.transmission_prob:
                    self._pending[nb].add(idx)
