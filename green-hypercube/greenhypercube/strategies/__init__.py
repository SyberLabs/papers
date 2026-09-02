"""Search strategies and a registry to build them from config."""

from __future__ import annotations

from .base import Strategy, ScoreStrategy
from .registry import build_strategy, STRATEGY_KINDS

__all__ = ["Strategy", "ScoreStrategy", "build_strategy", "STRATEGY_KINDS"]
