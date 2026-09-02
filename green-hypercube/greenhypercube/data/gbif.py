"""GBIF adapter: taxon resolution + occurrence sampling (REST API).

Uses the open GBIF REST API directly via ``requests`` (no credentials needed for
the ``species/match`` and ``occurrence/search`` endpoints). We call REST rather
than ``pygbif`` because the wrapper's keyword surface drifts across releases; the
HTTP API is stable. Occurrences are reduced to a coarse spatial grid so that
"shared site" co-occurrence is well defined and the graph stays sparse.
"""

from __future__ import annotations

import math
import time

import pandas as pd

from ..config import DataConfig
from ..utils import get_logger

log = get_logger("data.gbif")

GBIF_MATCH = "https://api.gbif.org/v1/species/match"
GBIF_OCC_SEARCH = "https://api.gbif.org/v1/occurrence/search"
GRID_DEG = 1.0  # spatial grid resolution in degrees -> "sites"


def resolve_names(names: list[str]) -> pd.DataFrame:
    """Resolve scientific names to the GBIF backbone via the REST match API.

    Returns a frame with columns: scientific_name (the input name, used as the
    join key downstream), gbif_key, accepted_name, genus, family, order.
    Unmatched names are dropped.
    """
    import requests  # lazy

    sess = requests.Session()
    rows = []
    for name in names:
        try:
            r = sess.get(GBIF_MATCH, params={"name": name, "kingdom": "Plantae"}, timeout=30)
            r.raise_for_status()
            m = r.json()
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("GBIF match failed for %r: %s", name, exc)
            continue
        if not m or m.get("matchType") == "NONE" or "usageKey" not in m:
            continue
        rows.append(
            {
                "scientific_name": name,
                "gbif_key": int(m["usageKey"]),
                "accepted_name": m.get("species", name),
                "genus": m.get("genus", ""),
                "family": m.get("family", ""),
                "order": m.get("order", ""),
            }
        )
    log.info("GBIF resolved %d/%d names", len(rows), len(names))
    return pd.DataFrame(rows)


def seed_species_for_region(cfg: DataConfig, limit: int) -> pd.DataFrame:
    """Discover plant species occurring in the configured region (REST)."""
    import requests  # lazy

    sess = requests.Session()
    min_lon, min_lat, max_lon, max_lat = cfg.region.bbox()
    wkt = (
        f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},"
        f"{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))"
    )
    seen: dict[int, dict] = {}
    offset, page = 0, 300
    while len(seen) < limit and offset < 200_000:
        try:
            r = sess.get(
                GBIF_OCC_SEARCH,
                params={
                    "geometry": wkt, "taxonKey": 6, "hasCoordinate": "true",
                    "limit": page, "offset": offset,
                },
                timeout=60,
            )
            r.raise_for_status()
            resp = r.json()
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("GBIF region seed failed at offset %d: %s", offset, exc)
            break
        results = resp.get("results", [])
        if not results:
            break
        for rec in results:
            key, sciname = rec.get("speciesKey"), rec.get("species")
            if key and sciname and key not in seen:
                seen[key] = {
                    "scientific_name": sciname, "gbif_key": int(key),
                    "accepted_name": sciname, "genus": rec.get("genus", ""),
                    "family": rec.get("family", ""), "order": rec.get("order", ""),
                }
        offset += page
        if resp.get("endOfRecords"):
            break
    df = pd.DataFrame(list(seen.values())).head(limit)
    log.info("GBIF region seed produced %d species", len(df))
    return df


def fetch_occurrences(species: pd.DataFrame, cfg: DataConfig, per_species: int = 100) -> pd.DataFrame:
    """Fetch occurrences for each species and reduce to grid sites (REST).

    ``species`` must contain ``species_id`` and ``gbif_key``. Returns the
    normalized ``occurrences`` table (species_id, site_id). Occurrences are
    global (not region-limited) so co-occurrence is defined even for pantropical
    taxa seeded from the chemistry source.
    """
    import requests  # lazy

    sess = requests.Session()
    rows = []
    for _, sp in species.iterrows():
        key = sp.get("gbif_key")
        if not key:
            continue
        try:
            r = sess.get(
                GBIF_OCC_SEARCH,
                params={"taxonKey": int(key), "hasCoordinate": "true",
                        "limit": min(per_species, 300)},
                timeout=45,
            )
            r.raise_for_status()
            resp = r.json()
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("GBIF occ search failed for key %s: %s", key, exc)
            continue
        for rec in resp.get("results", []):
            lat, lon = rec.get("decimalLatitude"), rec.get("decimalLongitude")
            if lat is None or lon is None:
                continue
            rows.append({"species_id": int(sp["species_id"]), "site_id": _grid_cell(lat, lon)})
        time.sleep(0.02)
    df = pd.DataFrame(rows, columns=["species_id", "site_id"]).drop_duplicates()
    log.info("GBIF occurrences -> %d species-site rows", len(df))
    return df


def _grid_cell(lat: float, lon: float) -> int:
    """Map a coordinate to a stable integer grid-cell id."""
    r = int(math.floor((lat + 90.0) / GRID_DEG))
    c = int(math.floor((lon + 180.0) / GRID_DEG))
    return r * 100_000 + c
