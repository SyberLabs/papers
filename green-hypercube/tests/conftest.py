"""Shared pytest fixtures: a small, fast manifold built from sample data."""

from __future__ import annotations

import pytest

from greenhypercube.config import Config, DataConfig, ManifoldConfig
from greenhypercube.data import ingest
from greenhypercube.hypercube import build_manifold


@pytest.fixture(scope="session")
def small_manifold(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("cache")
    data = DataConfig(source="sample", cache_dir=str(cache_dir), n_species=120, reward_density=0.15)
    cache = ingest(data, seed=1, force=True)
    return build_manifold(cache, ManifoldConfig())


@pytest.fixture(scope="session")
def small_config(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("cache_cfg")
    return Config(
        seed=3,
        data=DataConfig(source="sample", cache_dir=str(cache_dir), n_species=120, reward_density=0.15),
        manifold=ManifoldConfig(),
    )
