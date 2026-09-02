"""Hypercube assembly: turn normalized tables into the unified manifold."""

from __future__ import annotations

from .manifold import Manifold
from .builder import build_manifold

__all__ = ["Manifold", "build_manifold"]
