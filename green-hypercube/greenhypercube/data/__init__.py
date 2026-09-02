"""Data layer: adapters that materialize normalized ecological tables.

Every adapter (synthetic ``sample`` or ``live`` GBIF/Duke/GloBI/OpenTree) writes
the same normalized table set defined in :mod:`greenhypercube.data.schema` into a
:class:`~greenhypercube.utils.cache.ParquetCache`. The manifold builder reads only
from the cache, so data acquisition and modeling stay fully decoupled.
"""

from __future__ import annotations

from . import schema
from .ingest import ingest

__all__ = ["schema", "ingest"]
