"""Build strategy instances from config by ``kind``."""

from __future__ import annotations

from ..config import StrategyConfig
from ..hypercube import Manifold
from ..utils.rng import RNG
from .base import Strategy
from .random_search import RandomSearch
from .phylogenetic import PhylogeneticSearch
from .sensory import SensorySearch
from .ecological import EcologicalSearch
from .degree_random import DegreeRandomSearch
from .social import SocialSearch
from .cultural import CulturalSearch

STRATEGY_KINDS: dict[str, type[Strategy]] = {
    "random": RandomSearch,
    "phylogenetic": PhylogeneticSearch,
    "sensory": SensorySearch,
    "ecological": EcologicalSearch,
    "degree_random": DegreeRandomSearch,
    "social": SocialSearch,
    "cultural": CulturalSearch,
}


def build_strategy(spec: StrategyConfig, manifold: Manifold, rng: RNG) -> Strategy:
    if spec.kind not in STRATEGY_KINDS:
        raise ValueError(
            f"unknown strategy kind {spec.kind!r}; choose from {sorted(STRATEGY_KINDS)}"
        )
    cls = STRATEGY_KINDS[spec.kind]
    return cls(manifold, rng, **spec.params)
