"""Configuration schema for Green Hypercube experiments.

Configs are plain YAML files validated through pydantic models. A single
``Config`` object fully determines a run: where data comes from, how the
manifold is built, which strategies compete, and the simulation budget. This
keeps every experiment reproducible and self-documenting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class RegionConfig(BaseModel):
    """Geographic scope of the study, as a lon/lat bounding box.

    Defaults describe a broad neotropical / Amazonian window.
    """

    name: str = "neotropics"
    min_lon: float = -82.0
    min_lat: float = -20.0
    max_lon: float = -34.0
    max_lat: float = 13.0

    def bbox(self) -> tuple[float, float, float, float]:
        """Return ``(min_lon, min_lat, max_lon, max_lat)`` (GloBI/GBIF order)."""
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)


class DataConfig(BaseModel):
    """Where the ecological data comes from and how it is cached.

    ``source`` is either ``"sample"`` (a fully synthetic, offline snapshot with
    realistic structure) or ``"live"`` (pull from GBIF/Duke/GloBI/OpenTree and
    cache to Parquet). The sample source guarantees the pipeline runs end to end
    without network access.
    """

    source: str = Field(default="sample", pattern="^(sample|live)$")
    cache_dir: str = "data_cache"
    external_dir: str = "data_external"  # manually-downloaded bundles (Duke full, NAEB)
    region: RegionConfig = RegionConfig()

    # Live seeding strategy:
    #   "chemistry" -- seed the species pool from the Dr. Duke's chemistry mirror
    #     (curated, pre-enriched for bioactivity; reward is dense).
    #   "region"    -- seed an UNFILTERED flora from GBIF occurrences in the region,
    #     then bridge to phytochemistry via LOTUS (natural products) and reward via
    #     ChEMBL. Most species have no studied chemistry, so reward is genuinely
    #     sparse -- the honest needle-in-haystack. Cues come from phylogeny,
    #     co-occurrence and animal associations (defined for the whole flora).
    #   "naeb"      -- seed a North American flora from NAEB (documented-use enriched).
    #   "naeb_full" -- Moerman 1979 design: seed an unfiltered GBIF regional flora,
    #     label each species used/unused by NAEB Drug records (reward sparse by
    #     construction). Cues from GBIF/OTL/GloBI + Dr. Duke's offline.
    seed_mode: str = Field(default="chemistry", pattern="^(chemistry|region|naeb|naeb_full)$")

    # Reward provenance for live runs:
    #   "chembl"   -- measured bioactivity potency (chemistry/region modes).
    #   "naeb_use" -- documented medicinal use intensity (naeb mode); independent
    #     of every cue, so it sidesteps the research-coverage confound.
    reward_source: str = Field(default="chembl", pattern="^(chembl|naeb_use)$")

    # Size of the species pool. For the sample source this is generated; for the
    # live source it caps how many resolved taxa we keep.
    n_species: int = 400

    # Fraction of species with a non-zero hidden utility (reward sparsity). Only
    # used by the sample generator; live data derives this from the assay source.
    reward_density: float = 0.12

    # Cue-reward coupling for the SYNTHETIC source only. The reward is drawn
    # through an independent assay channel from a latent z; each cue (sensory,
    # ecological, animal, phylogenetic) reflects z with fidelity
    # ``signal_strength * coupling[channel]``. signal_strength=0 makes every cue
    # statistically independent of the reward -- the built-in null. This makes
    # the synthetic study a sensitivity analysis rather than a foregone result.
    signal_strength: float = 0.7
    coupling: dict[str, float] = Field(default_factory=dict)  # per-channel multipliers
    assay_noise: float = 0.1  # independent measurement noise on the reward

    # Live-source taxon scope. Families to seed the neotropical flora from.
    seed_families: list[str] = Field(default_factory=list)


class ManifoldConfig(BaseModel):
    """Parameters controlling how the unified manifold is assembled."""

    # Number of spatial grid cells used to define co-occurrence "sites".
    n_sites: int = 60

    # Edge thresholds for the ecological / interaction graphs (Jaccard overlap).
    eco_edge_threshold: float = 0.15
    bio_edge_threshold: float = 0.10

    # Reward threshold above which a species counts as a genuine "discovery".
    discovery_threshold: float = 0.05

    # Reward sparsification: keep only the top fraction of species (by reward)
    # as useful, zeroing the rest. 1.0 = no sparsification. This lets us impose a
    # controlled "needle in a haystack" density on top of any landscape (notably
    # the live one, whose curated medicinal flora is otherwise reward-dense) to
    # test whether the structured-search advantage survives as reward gets rare.
    reward_top_frac: float = 1.0


class StrategyConfig(BaseModel):
    """A single strategy entry in an experiment."""

    name: str
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)


class SimulationConfig(BaseModel):
    """Simulation budget and replication settings."""

    budget: int = 800
    n_replicates: int = 20
    n_agents: int = 1
    observation_noise: float = 0.0

    # Number of independently generated landscapes (data-generating seeds). With
    # >1, confidence intervals reflect uncertainty about the landscape itself,
    # not merely the search RNG. Supported by the synthetic source.
    n_landscapes: int = 1


class Config(BaseModel):
    """Top-level experiment configuration."""

    seed: int = 12345
    output_dir: str = "results"
    data: DataConfig = DataConfig()
    manifold: ManifoldConfig = ManifoldConfig()
    simulation: SimulationConfig = SimulationConfig()
    strategies: list[StrategyConfig] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load and validate a config from a YAML file."""
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.model_validate(raw)

    def to_yaml(self, path: str | Path) -> None:
        """Persist the (validated) config to YAML, e.g. alongside results."""
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.model_dump(), fh, sort_keys=False)
