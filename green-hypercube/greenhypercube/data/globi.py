"""GloBI adapter: plant-animal interaction edges.

Queries the open GloBI REST API for interactions where the plant is the source
taxon, restricted to the study bounding box. The resulting ``interactions``
table (species_id, animal_taxon, interaction_type) drives the animal-cue graph
``G_bio``: plants sharing animal associates are linked.

``requests`` is imported lazily.
"""

from __future__ import annotations

import pandas as pd

from ..config import DataConfig
from ..utils import get_logger

log = get_logger("data.globi")

GLOBI_URL = "https://api.globalbioticinteractions.org/interaction"

# Interaction types where a plant is the source and an animal the target.
# (hasDispersalVectorOf is omitted: the GloBI endpoint currently 500s for it.)
PLANT_SOURCE_TYPES = ["pollinatedBy", "eatenBy", "visitedBy"]


def fetch_interactions(
    species: pd.DataFrame, cfg: DataConfig, use_bbox: bool = False
) -> pd.DataFrame:
    """Fetch plant->animal interactions for each species.

    By default the query is global: taxa seeded from the chemistry source are
    often pantropical, so a regional bounding box would discard most edges. Set
    ``use_bbox=True`` for region-seeded runs where local interactions matter.
    """
    import requests  # lazy

    bbox = None
    if use_bbox:
        min_lon, min_lat, max_lon, max_lat = cfg.region.bbox()
        bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    rows = []
    for _, sp in species.iterrows():
        name = sp.get("accepted_name") or sp.get("scientific_name")
        if not name:
            continue
        for itype in PLANT_SOURCE_TYPES:
            params = {
                "sourceTaxon": str(name),
                "interactionType": itype,
                "type": "json",
                "fields": "target_taxon_name",
            }
            if bbox:
                params["bbox"] = bbox
            try:
                resp = requests.get(GLOBI_URL, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:  # pragma: no cover - network dependent
                log.warning("GloBI query failed for %s/%s: %s", name, itype, exc)
                continue
            for record in data.get("data", []):
                if not record:
                    continue
                animal = str(record[0]).strip()
                if not animal or animal.lower() == "no name":
                    continue
                rows.append(
                    {
                        "species_id": int(sp["species_id"]),
                        "animal_taxon": animal,
                        "interaction_type": itype,
                    }
                )
    df = pd.DataFrame(
        rows, columns=["species_id", "animal_taxon", "interaction_type"]
    ).drop_duplicates()
    log.info("GloBI -> %d interaction rows", len(df))
    return df
