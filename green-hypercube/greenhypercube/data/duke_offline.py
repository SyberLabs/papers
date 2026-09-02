"""Offline loader for the *full* Dr. Duke's relational CSV distribution.

Unlike :mod:`greenhypercube.data.duke` (which pulls a slim ~299-species CC0
mirror over HTTP), this reads the authoritative ``Duke-Source-CSV`` bundle that
the user downloaded manually (the USDA host blocks unattended downloads). The
full release covers ~2,374 species and ships the relational tables:

- ``FNFTAX.csv``      : taxonomy (FNFNUM <-> scientific name, genus, family).
- ``FARMACY_NEW.csv`` : species (FNFNUM) -> chemical (+ class) occurrences.
- ``ETHNOBOT.csv``    : species -> documented folk use (+ country).

We expose the chemistry as the normalized ``chemicals`` CUE table for any species
pool, and the folk uses as the optional ``uses`` table. Chemistry coverage from
the full release materially reduces the "studied-ness" coverage confound that the
slim mirror exaggerates.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils import get_logger
from . import duke, schema

log = get_logger("data.duke_offline")

_ENC = "latin-1"


def _src(external_dir: str | Path) -> Path:
    return Path(external_dir) / "duke_src"


def load_taxonomy(external_dir: str | Path) -> pd.DataFrame:
    """FNFTAX -> [fnfnum, scientific_name, genus, family] (one row per taxon)."""
    df = pd.read_csv(_src(external_dir) / "FNFTAX.csv", encoding=_ENC, low_memory=False)
    out = pd.DataFrame({
        "fnfnum": pd.to_numeric(df["FNFNUM"], errors="coerce"),
        "scientific_name": df["TAXON"].astype(str).str.strip(),
        "genus": df["GENUS"].astype(str).str.strip(),
        "family": df.get("FAMILY", "").astype(str).str.strip(),
    }).dropna(subset=["fnfnum"])
    out["fnfnum"] = out["fnfnum"].astype(int)
    return out


def chemicals_for_names(external_dir: str | Path, name_to_id: dict[str, int]) -> pd.DataFrame:
    """Build the normalized ``chemicals`` CUE table for a species pool.

    Joins the pool (keyed by scientific_name) to FNFTAX (to recover FNFNUM) and
    then to FARMACY_NEW (the species->chemical occurrence table). Chemical class
    uses Duke's CHEMCLASS when usable, else a keyword guess from the name.
    """
    tax = load_taxonomy(external_dir)
    name_lower_to_fnf = {n.lower(): f for n, f in zip(tax["scientific_name"], tax["fnfnum"])}
    pool_fnf = {}
    for name, sid in name_to_id.items():
        f = name_lower_to_fnf.get(str(name).strip().lower())
        if f is not None:
            pool_fnf[int(f)] = int(sid)
    if not pool_fnf:
        log.warning("Duke(full): no species-pool overlap with FNFTAX")
        return pd.DataFrame(columns=schema.TABLES["chemicals"])

    farm = pd.read_csv(
        _src(external_dir) / "FARMACY_NEW.csv", encoding=_ENC, low_memory=False,
        usecols=["FNFNUM", "CHEM", "CHEMCLASS"],
    )
    farm["FNFNUM"] = pd.to_numeric(farm["FNFNUM"], errors="coerce")
    farm = farm.dropna(subset=["FNFNUM"])
    farm["FNFNUM"] = farm["FNFNUM"].astype(int)
    farm = farm[farm["FNFNUM"].isin(pool_fnf)]

    rows = []
    for _, r in farm.iterrows():
        sid = pool_fnf.get(int(r["FNFNUM"]))
        if sid is None:
            continue
        chem = str(r["CHEM"]).strip()
        if not chem or chem.lower() == "nan":
            continue
        rows.append({"species_id": sid, "chemical": chem,
                     "chem_class": duke._guess_class(chem), "amount": 1.0})
    out = pd.DataFrame(rows, columns=schema.TABLES["chemicals"]).drop_duplicates()
    log.info("Duke(full): %d chemical rows for %d/%d pool species",
             len(out), out["species_id"].nunique() if len(out) else 0, len(name_to_id))
    return out


def uses_for_names(external_dir: str | Path, name_to_id: dict[str, int]) -> pd.DataFrame:
    """Optional ``uses`` table from ETHNOBOT (species -> folk use). Not a reward."""
    eth = pd.read_csv(
        _src(external_dir) / "ETHNOBOT.csv", encoding=_ENC, low_memory=False,
        usecols=["TAXON", "ACTIVITY"],
    )
    rows = []
    for _, r in eth.iterrows():
        sid = name_to_id.get(str(r["TAXON"]).strip())
        if sid is None:
            continue
        rows.append({"species_id": sid, "use_category": str(r["ACTIVITY"]).strip()})
    return pd.DataFrame(rows, columns=schema.TABLES["uses"]).drop_duplicates()
