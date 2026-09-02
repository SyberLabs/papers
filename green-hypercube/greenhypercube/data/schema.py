"""Normalized table schema shared by all data adapters.

These small, source-agnostic tables are the contract between the data layer and
the manifold builder. Each maps to one face of the Green Hypercube.

Crucially, the *reward* (``bioassay``) is kept on a separate provenance from the
*cues*. The reward is measured biological potency (synthetic assay, or live
ChEMBL-style assay data), which is determined by chemistry-vs-target rather than
by whether a plant was historically adopted. The cues -- chemistry/sensory,
spatial co-occurrence, animal associations, phylogeny -- are observed
independently. This avoids two leakage traps: (a) predicting bioactivity from
chemistry within a single database, and (b) using documented ethnobotanical use
(itself a product of past discovery) as the search target.

- ``species``        : the points of the space (taxonomy + cross-source keys).
- ``occurrences``    : species x site presence  -> spatial co-occurrence (G_eco).
- ``chemicals``      : species x chemical class  -> sensory salience CUE features.
- ``interactions``   : species x animal associate -> animal-cue graph (G_bio).
- ``bioassay``       : species -> measured potency -> the HIDDEN REWARD (independent).
- ``phylogeny``      : a Newick tree over species -> phylogenetic distance.

Optional (retained for realism / alternative analyses, never used for reward):
- ``bioactivities``  : chemical -> bioactivity label.
- ``uses``           : species -> documented ethnobotanical use (leakage-prone).
"""

from __future__ import annotations

# Recognized chemical classes and a coarse sensory channel each maps to. These
# proxies let "sensory heuristic" search prioritize plants whose chemistry would
# produce salient smell/taste/visual cues.
CHEM_CLASSES: dict[str, str] = {
    "alkaloid": "bitter",
    "terpene": "aromatic",
    "essential_oil": "aromatic",
    "tannin": "astringent",
    "flavonoid": "color",
    "glycoside": "sweet",
    "saponin": "foaming",
    "phenol": "pungent",
}

SENSORY_CHANNELS: list[str] = sorted(set(CHEM_CLASSES.values()))

# Required columns per table (used for light validation).
TABLES: dict[str, list[str]] = {
    "species": ["species_id", "scientific_name", "genus", "family", "order"],
    "occurrences": ["species_id", "site_id"],
    "chemicals": ["species_id", "chemical", "chem_class", "amount"],
    "bioassay": ["species_id", "assay_value"],
    "interactions": ["species_id", "animal_taxon", "interaction_type"],
    # optional, not used for reward:
    "bioactivities": ["chemical", "activity"],
    "uses": ["species_id", "use_category"],
}

# Optional: reward-side documentation depth for symmetric confound control (P1).
# Columns after species_id are source-specific (NAEB: total_uses, n_tribes, ...;
# ChEMBL: chem_count, lotus_count). Not required for simulation.
REWARD_DEPTH_NAME = "reward_depth"

# Tables that must be present for the pipeline to run.
REQUIRED_TABLES: list[str] = [
    "species", "occurrences", "chemicals", "interactions", "bioassay",
]

# The phylogeny is stored as a Newick text file, not a Parquet table.
PHYLOGENY_NAME = "phylogeny"


def validate(name: str, columns: list[str]) -> None:
    """Raise if a materialized table is missing required columns."""
    required = TABLES.get(name)
    if required is None:
        return
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(f"table '{name}' missing required columns: {missing}")
