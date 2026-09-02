"""The Green Hypercube manifold: a unified, query-ready representation.

A :class:`Manifold` bundles everything a search strategy may consult:

- ``species``        : taxonomy table (index aligned to 0..n-1).
- ``X`` / features   : observable cue matrix (sensory/chemical/spatial).
- ``sensory_salience``: scalar per species summarizing smell/taste/visual cues.
- ``D_phylo``        : (n, n) patristic distance matrix.
- ``eco`` / ``bio``  : co-occurrence and animal-association graphs (+ adjacency).
- ``reward``         : hidden utility in [0, 1] (sparse).

The reward is *hidden*: strategies never read it directly. They observe cues and
only learn a species' reward by paying to "experiment" on it in the environment.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class Manifold:
    species: pd.DataFrame
    X: np.ndarray
    feature_names: list[str]
    sensory_salience: np.ndarray
    D_phylo: np.ndarray
    eco: nx.Graph
    bio: nx.Graph
    reward: np.ndarray
    discovery_threshold: float

    # Dense adjacency (weighted) for fast scoring; built in __post_init__.
    eco_adj: np.ndarray = None  # type: ignore[assignment]
    bio_adj: np.ndarray = None  # type: ignore[assignment]

    # Per-species research-coverage covariates (counts), for confound control.
    # Shape (n, k); None when not computed. Strategies NEVER see these -- they are
    # only used by the coupling instrument to partial out study effort.
    coverage: np.ndarray | None = None
    coverage_names: list[str] | None = None

    # Per-species reward-side documentation depth (for symmetric confound control).
    # Shape (n, k); None when not computed. Never visible to search strategies.
    reward_depth: np.ndarray | None = None
    reward_depth_names: list[str] | None = None

    def __post_init__(self) -> None:
        self.eco_adj = _weighted_adjacency(self.eco, self.n)
        self.bio_adj = _weighted_adjacency(self.bio, self.n)

    @property
    def n(self) -> int:
        return len(self.species)

    @property
    def useful_mask(self) -> np.ndarray:
        """Boolean mask of species counting as genuine discoveries."""
        return self.reward >= self.discovery_threshold

    @property
    def n_useful(self) -> int:
        return int(self.useful_mask.sum())

    @property
    def total_reward(self) -> float:
        return float(self.reward.sum())

    def phylo_similarity(self, scale: float | None = None) -> np.ndarray:
        """Return an (n, n) similarity matrix exp(-D/scale)."""
        if scale is None:
            pos = self.D_phylo[self.D_phylo > 0]
            scale = float(np.median(pos)) if pos.size else 1.0
        return np.exp(-self.D_phylo / max(scale, 1e-6))


def _weighted_adjacency(g: nx.Graph, n: int) -> np.ndarray:
    """Dense weighted adjacency aligned to species_id (0..n-1)."""
    A = np.zeros((n, n), dtype=np.float32)
    for u, v, data in g.edges(data=True):
        w = float(data.get("weight", 1.0))
        A[u, v] = w
        A[v, u] = w
    return A
