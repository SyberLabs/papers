"""LOTUS adapter: a broad plant -> natural-product bridge via Wikidata.

The Dr. Duke's mirror only covers a few hundred curated medicinal species, far
too few to seed an *unfiltered* regional flora. LOTUS (the open
structure-organism database, mirrored in Wikidata via the "found in taxon"
relation, P703) links hundreds of thousands of natural products to ~40k
organisms. We use it as the species -> compound bridge so an arbitrary regional
flora can be given a phytochemistry, and thence a measured-bioactivity reward
(ChEMBL, keyed by the compound).

Each compound row carries an InChIKey (P235) and an English label, so the reward
adapter can resolve it in ChEMBL by either identifier. Results are cached to
Parquet so the (slow) SPARQL pass runs once. ``requests`` is imported lazily.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from ..utils import get_logger

log = get_logger("data.lotus")

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
_UA = "greenhypercube/0.1 (ethnobotanical search modelling; research use)"


def _query_species(sess, name: str, max_compounds: int) -> list[dict]:
    """Return compound rows (inchikey, compound label) recorded for one taxon."""
    query = (
        "SELECT ?inchikey ?cl WHERE { "
        f'?taxon wdt:P225 "{name}" . '
        "?compound wdt:P703 ?taxon ; wdt:P235 ?inchikey . "
        'OPTIONAL { ?compound rdfs:label ?cl FILTER(lang(?cl)="en") } '
        f"}} LIMIT {max_compounds}"
    )
    for attempt in range(3):
        try:
            r = sess.get(
                WIKIDATA_SPARQL,
                params={"query": query, "format": "json"},
                headers={"User-Agent": _UA, "Accept": "application/sparql-results+json"},
                timeout=60,
            )
            if r.status_code == 429:  # rate limited; back off
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
            out = []
            for b in r.json()["results"]["bindings"]:
                out.append(
                    {
                        "inchikey": b["inchikey"]["value"],
                        "compound": b.get("cl", {}).get("value", ""),
                    }
                )
            return out
        except Exception as exc:  # pragma: no cover - network dependent
            if attempt == 2:
                log.warning("LOTUS query failed for %r: %s", name, exc)
            else:
                time.sleep(1.0 * (attempt + 1))
    return []


def fetch_compounds(
    species: pd.DataFrame, cache_dir: str | Path, max_per_species: int = 12
) -> pd.DataFrame:
    """Map each species to its LOTUS compounds (cached to Parquet).

    ``species`` needs ``species_id`` and ``scientific_name``. Returns a frame
    with columns: species_id, inchikey, compound. Cached at
    ``<cache_dir>/lotus_compounds.parquet`` keyed by the species set.
    """
    import requests  # lazy

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / "lotus_compounds.parquet"
    sid_by_name = {str(n).strip(): int(i)
                   for n, i in zip(species["scientific_name"], species["species_id"])}

    if local.exists():
        cached = pd.read_parquet(local)
        if set(cached["species_id"]).issuperset(set(species["species_id"])):
            log.info("LOTUS: using cached compounds (%d rows)", len(cached))
            return cached[cached["species_id"].isin(species["species_id"])].reset_index(drop=True)

    sess = requests.Session()
    rows = []
    for name, sid in sid_by_name.items():
        for c in _query_species(sess, name, max_per_species):
            rows.append({"species_id": sid, "inchikey": c["inchikey"], "compound": c["compound"]})
        time.sleep(0.05)  # be polite to WDQS
    df = pd.DataFrame(rows, columns=["species_id", "inchikey", "compound"]).drop_duplicates()
    df.to_parquet(local, index=False)
    log.info("LOTUS: %d compound rows for %d/%d species",
             len(df), df["species_id"].nunique() if len(df) else 0, len(species))
    return df
