"""Ingestion orchestrator.

Materializes the normalized cache from the configured source. For ``sample`` it
calls the synthetic generator; for ``live`` it seeds species from the Dr. Duke's
chemistry mirror, resolves them against the GBIF backbone + Open Tree, enriches
with GBIF occurrences and GloBI interactions, and derives an independent reward
from ChEMBL measured potency.

Both paths leave the cache in the identical normalized schema, so everything
downstream is source-agnostic.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import DataConfig
from ..utils import ParquetCache, get_logger
from . import sample as sample_mod
from . import schema

log = get_logger("data.ingest")

_REQUIRED = schema.REQUIRED_TABLES


def ingest(cfg: DataConfig, seed: int, force: bool = False) -> ParquetCache:
    """Populate and return the cache for the configured data source."""
    cache = ParquetCache(cfg.cache_dir)

    have_all = all(cache.has(t) for t in _REQUIRED) and cache.has(
        schema.PHYLOGENY_NAME, suffix="nwk"
    )
    if have_all and not force:
        log.info("cache already populated at %s (use --force to rebuild)", cache.dir)
        _ensure_reward_depth(cfg, cache)
        return cache

    if cfg.source == "sample":
        log.info("generating synthetic sample snapshot")
        sample_mod.generate(cfg, cache, seed)
    elif cfg.source == "live":
        _ingest_live(cfg, cache)
    else:  # pragma: no cover - validated by pydantic
        raise ValueError(f"unknown source {cfg.source!r}")

    _validate_cache(cache)
    return cache


def _ensure_reward_depth(cfg: DataConfig, cache: ParquetCache) -> None:
    """Backfill optional reward_depth for caches built before P1."""
    if cfg.seed_mode not in ("naeb", "naeb_full") or cache.has(schema.REWARD_DEPTH_NAME):
        return
    from . import naeb

    species = cache.read_table("species")
    name_to_id = {
        str(n).strip(): int(i)
        for n, i in zip(species["scientific_name"], species["species_id"])
    }
    depth = naeb.documentation_depth(cfg.external_dir, name_to_id)
    cache.write_table(schema.REWARD_DEPTH_NAME, depth)
    log.info("backfilled %s for existing NAEB cache", schema.REWARD_DEPTH_NAME)


def _ingest_live(cfg: DataConfig, cache: ParquetCache) -> None:
    """Run the live data pipeline against real databases.

    Provenance separation (the leakage-resistant design):
      - CUES: chemistry/sensory (Dr. Duke's / LOTUS), spatial co-occurrence
        (GBIF), animal associations (GloBI), phylogeny (Open Tree of Life).
      - REWARD: independently measured bioactivity potency (ChEMBL), keyed by
        compound -- not derived from the cue record nor from documented use.

    Two seeding modes (``cfg.seed_mode``): ``chemistry`` seeds from the curated
    Dr. Duke's mirror (reward-dense); ``region`` seeds an unfiltered GBIF flora
    and bridges chemistry via LOTUS (reward-sparse, the honest haystack).
    """
    if cfg.seed_mode == "region":
        _ingest_live_region(cfg, cache)
    elif cfg.seed_mode == "naeb":
        _ingest_live_naeb(cfg, cache)
    elif cfg.seed_mode == "naeb_full":
        _ingest_live_naeb_full(cfg, cache)
    else:
        _ingest_live_chemistry(cfg, cache)


def _ingest_live_chemistry(cfg: DataConfig, cache: ParquetCache) -> None:
    """Curated path: species pool seeded from the Dr. Duke's chemistry mirror."""
    from . import gbif, duke, globi, opentree, chembl
    from . import taxonomy

    cache_dir = Path(cfg.cache_dir)

    # 1. Seed from Dr. Duke's chemistry mirror (species + compounds).
    mirror = duke.load_mirror(cache_dir)
    names = sorted(mirror["plant_species"].unique().tolist())[: cfg.n_species]

    # 2. Resolve to the GBIF backbone (REST) and keep matched taxa.
    resolved = gbif.resolve_names(names)
    if cfg.seed_families:
        resolved = resolved[resolved["family"].isin(cfg.seed_families)]
    species = taxonomy.assign_species_ids(resolved)

    # 3. Resolve OTT ids; keep taxa we can both key in GBIF and place in OTL.
    species = opentree.resolve_ott_ids(species)
    species = taxonomy.require_resolved(species, need_ott=True)

    name_to_id = {
        str(n).strip(): int(i)
        for n, i in zip(species["scientific_name"], species["species_id"])
    }

    # 4. Phylogeny (induced subtree over resolved species) -- CUE.
    opentree.fetch_induced_subtree(species, cache)

    # 5. Chemistry / sensory salience from Dr. Duke's -- CUE.
    chemicals = duke.chemicals_table(mirror, name_to_id)

    # 6. Spatial co-occurrence (GBIF) and animal associations (GloBI) -- CUES.
    occurrences = gbif.fetch_occurrences(species, cfg)
    interactions = globi.fetch_interactions(species, cfg)

    # 7. Independent reward: measured ChEMBL potency per species.
    bioassay = chembl.fetch_bioassay(species, chemicals, cfg)

    cache.write_table("species", species[schema.TABLES["species"]])
    cache.write_table("occurrences", occurrences)
    cache.write_table("chemicals", chemicals)
    cache.write_table("interactions", interactions)
    cache.write_table("bioassay", bioassay)


def _ingest_live_region(cfg: DataConfig, cache: ParquetCache) -> None:
    """Unfiltered path: seed a regional GBIF flora; bridge chemistry via LOTUS.

    The species are NOT pre-selected for bioactivity, so reward (ChEMBL potency
    of the plant's LOTUS natural products) is sparse and the chemistry/sensory
    cue is largely absent -- making phylogeny, co-occurrence and animal
    association the cues that must do the work. This is the honest test of the
    sparsity gate.
    """
    from . import gbif, duke, globi, opentree, chembl, lotus
    from . import taxonomy

    cache_dir = Path(cfg.cache_dir)

    # 1. Seed an unfiltered flora from GBIF occurrences in the region.
    seed = gbif.seed_species_for_region(cfg, limit=cfg.n_species * 2)
    if cfg.seed_families:
        seed = seed[seed["family"].isin(cfg.seed_families)]
    seed = seed.head(cfg.n_species).reset_index(drop=True)
    species = taxonomy.assign_species_ids(seed)

    # 2. Place in the Open Tree; keep taxa we can resolve.
    species = opentree.resolve_ott_ids(species)
    species = taxonomy.require_resolved(species, need_ott=True)
    name_to_id = {
        str(n).strip(): int(i)
        for n, i in zip(species["scientific_name"], species["species_id"])
    }

    # 3. Phylogeny + spatial co-occurrence + animal associations -- CUES.
    opentree.fetch_induced_subtree(species, cache)
    occurrences = gbif.fetch_occurrences(species, cfg)
    interactions = globi.fetch_interactions(species, cfg)

    # 4. Chemistry where Dr. Duke's happens to cover a species (sparse) -- CUE.
    mirror = duke.load_mirror(cache_dir)
    chemicals = duke.chemicals_table(mirror, name_to_id)

    # 5. Broad species->compound bridge (LOTUS) -> measured reward (ChEMBL).
    compounds = lotus.fetch_compounds(species, cache_dir)
    bioassay = chembl.fetch_bioassay_via_compounds(species, compounds, cfg)

    cache.write_table("species", species[schema.TABLES["species"]])
    cache.write_table("occurrences", occurrences)
    cache.write_table("chemicals", chemicals)
    cache.write_table("interactions", interactions)
    cache.write_table("bioassay", bioassay)


def refresh_reward(cfg: DataConfig) -> ParquetCache:
    """Recompute ONLY the bioassay (reward) table on the existing cached landscape.

    Used to complete a run after a transient ChEMBL outage without reseeding the
    flora or refetching cues (which would change the landscape). Reuses the
    cached species table plus the cached compound bridge (LOTUS for region mode,
    the chemicals table for chemistry mode).
    """
    from . import chembl

    cache = ParquetCache(cfg.cache_dir)
    if not cache.has("species"):
        raise FileNotFoundError(f"no cached landscape at {cfg.cache_dir}; run build first")
    species = cache.read_table("species").sort_values("species_id").reset_index(drop=True)

    if cfg.seed_mode in ("naeb", "naeb_full"):
        from . import naeb as naeb_mod
        name_to_id = {str(n).strip(): int(i)
                      for n, i in zip(species["scientific_name"], species["species_id"])}
        bioassay = naeb_mod.use_reward(cfg.external_dir, name_to_id)
    elif cfg.seed_mode == "region":
        from . import lotus
        compounds = lotus.fetch_compounds(species, cfg.cache_dir)
        bioassay = chembl.fetch_bioassay_via_compounds(species, compounds, cfg)
    else:
        bioassay = chembl.fetch_bioassay(species, cache.read_table("chemicals"), cfg)

    cache.write_table("bioassay", bioassay)
    log.info("reward refreshed: %d/%d species with measured potency",
             int((bioassay["assay_value"] > 0).sum()), len(species))
    return cache


def _ingest_live_naeb(cfg: DataConfig, cache: ParquetCache) -> None:
    """North-American path: NAEB flora, reward = documented medicinal use.

    Reward here is *independent of every cue*: it is how much a plant was
    documented as a Native American medicine (NAEB Drug uses), not its chemistry
    or how well it is studied. Cues are phylogeny (OTL), co-occurrence (GBIF),
    animal association (GloBI) and chemistry/sensory (full Dr. Duke's). This is
    the cleanest live test of whether ecological structure predicts the plants
    people actually adopted -- and, with coverage controls, whether that holds
    once research effort is partialled out.
    """
    from . import gbif, globi, opentree, duke_offline, naeb
    from . import taxonomy

    # 1. Seed a North American flora from NAEB (species with documented use).
    seed = naeb.species_pool(cfg.external_dir, cfg.n_species)
    if cfg.seed_families:
        seed = seed[seed["family"].isin(cfg.seed_families)]

    # 2. Resolve to the GBIF backbone; place in the Open Tree.
    resolved = gbif.resolve_names(seed["scientific_name"].tolist())
    species = taxonomy.assign_species_ids(resolved)
    species = opentree.resolve_ott_ids(species)
    species = taxonomy.require_resolved(species, need_ott=True)
    name_to_id = {
        str(n).strip(): int(i)
        for n, i in zip(species["scientific_name"], species["species_id"])
    }

    # 3. Cues: phylogeny + co-occurrence + animal association + chemistry.
    opentree.fetch_induced_subtree(species, cache)
    occurrences = gbif.fetch_occurrences(species, cfg)
    interactions = globi.fetch_interactions(species, cfg)
    chemicals = duke_offline.chemicals_for_names(cfg.external_dir, name_to_id)

    # 4. Reward: documented medicinal-use intensity (independent of cues).
    bioassay = naeb.use_reward(cfg.external_dir, name_to_id)
    reward_depth = naeb.documentation_depth(cfg.external_dir, name_to_id)

    cache.write_table("species", species[schema.TABLES["species"]])
    cache.write_table("occurrences", occurrences)
    cache.write_table("chemicals", chemicals)
    cache.write_table("interactions", interactions)
    cache.write_table("bioassay", bioassay)
    cache.write_table(schema.REWARD_DEPTH_NAME, reward_depth)


def _ingest_live_naeb_full(cfg: DataConfig, cache: ParquetCache) -> None:
    """Moerman full-flora path: GBIF regional pool + NAEB used/unused labels.

    Species are drawn from GBIF occurrences in the configured region (North
    America for NAEB), *not* pre-filtered to documented NAEB plants. Reward is
    documented medicinal-use intensity for species in NAEB, zero otherwise
    the classic *used vs available* estimand (Moerman 1979).
    """
    from . import gbif, globi, opentree, duke_offline, naeb
    from . import taxonomy

    seed = gbif.seed_species_for_region(cfg, limit=cfg.n_species * 2)
    if cfg.seed_families:
        seed = seed[seed["family"].isin(cfg.seed_families)]
    seed = seed.head(cfg.n_species).reset_index(drop=True)

    species = taxonomy.assign_species_ids(seed)
    species = opentree.resolve_ott_ids(species)
    species = taxonomy.require_resolved(species, need_ott=True)
    name_to_id = {
        str(n).strip(): int(i)
        for n, i in zip(species["scientific_name"], species["species_id"])
    }

    opentree.fetch_induced_subtree(species, cache)
    occurrences = gbif.fetch_occurrences(species, cfg)
    interactions = globi.fetch_interactions(species, cfg, use_bbox=True)
    chemicals = duke_offline.chemicals_for_names(cfg.external_dir, name_to_id)

    bioassay = naeb.use_reward(cfg.external_dir, name_to_id)
    reward_depth = naeb.documentation_depth(cfg.external_dir, name_to_id)
    naeb.log_pool_composition(bioassay, label="Moerman full flora")

    cache.write_table("species", species[schema.TABLES["species"]])
    cache.write_table("occurrences", occurrences)
    cache.write_table("chemicals", chemicals)
    cache.write_table("interactions", interactions)
    cache.write_table("bioassay", bioassay)
    cache.write_table(schema.REWARD_DEPTH_NAME, reward_depth)


def _validate_cache(cache: ParquetCache) -> None:
    for t in _REQUIRED:
        df = cache.read_table(t)
        schema.validate(t, list(df.columns))
    cache.read_text(schema.PHYLOGENY_NAME)  # ensure phylogeny exists
    log.info("cache validated: all required normalized tables present")
