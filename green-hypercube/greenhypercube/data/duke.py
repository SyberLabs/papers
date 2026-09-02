"""Dr. Duke's Phytochemical & Ethnobotanical Database adapter.

Downloads and parses the bulk ``Duke-Source-CSV.zip`` distribution (USDA Open
Data Catalog) into the normalized ``chemicals``, ``bioactivities`` and ``uses``
tables. The Duke distribution ships several CSVs; we read the ones describing
chemical occurrence in plants, chemical bioactivities, and ethnobotanical uses.

Because the exact filenames in the bundle vary across mirrors, the parser is
defensive: it inspects the archive and matches files by content/columns rather
than hard-coded names, and logs anything it cannot interpret.

``requests`` is imported lazily; nothing here is needed for the sample pipeline.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from ..utils import get_logger
from . import schema

log = get_logger("data.duke")

# The authoritative bulk archive lives on USDA Ag Data Commons
# (doi:10.15482/USDA.ADC/1239279) but that host sits behind a JavaScript/CDN
# challenge that blocks unattended downloads. As a programmatically reachable
# alternative we use a cleaned, PubChem/ChEMBL-cross-referenced mirror of Dr.
# Duke's records published on the Hugging Face Hub (CC0, same source data).
DUKE_BULK_URL = "https://data.nal.usda.gov/sites/default/files/Duke-Source-CSV.zip"
DUKE_HF_PARQUET = (
    "https://huggingface.co/datasets/wirthal1990-tech/USDA-Phytochemical-Database-JSON"
    "/resolve/main/ethno_sample_400.parquet"
)


def load_mirror(cache_dir: Path) -> pd.DataFrame:
    """Load the Dr. Duke's mirror (cached locally after first download).

    Returns a frame with columns: plant_species, chemical, application,
    pubchem_cid, canonical_smiles, chembl_bioactivity_count. ``application`` is
    the documented ethnobotanical use (kept for optional analysis, NOT used for
    reward); the reward is derived independently from measured ChEMBL potency.
    """
    import requests  # lazy

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / "duke_mirror.parquet"
    if not local.exists():
        log.info("downloading Dr. Duke's mirror from Hugging Face")
        r = requests.get(DUKE_HF_PARQUET, headers={"User-Agent": "greenhypercube"}, timeout=120)
        r.raise_for_status()
        local.write_bytes(r.content)
    df = pd.read_parquet(local)
    keep = [c for c in [
        "plant_species", "chemical", "application", "pubchem_cid",
        "canonical_smiles", "chembl_bioactivity_count",
    ] if c in df.columns]
    df = df[keep].dropna(subset=["plant_species", "chemical"]).reset_index(drop=True)
    df["plant_species"] = df["plant_species"].astype(str).str.strip()
    df["chemical"] = df["chemical"].astype(str).str.strip()
    log.info("Duke mirror: %d rows, %d distinct species",
             len(df), df["plant_species"].nunique())
    return df


def chemicals_table(mirror: pd.DataFrame, name_to_id: dict[str, int]) -> pd.DataFrame:
    """Build the normalized ``chemicals`` (CUE) table from the mirror."""
    rows = []
    for _, r in mirror.iterrows():
        sid = name_to_id.get(str(r["plant_species"]).strip())
        if sid is None:
            continue
        chem = str(r["chemical"]).strip()
        rows.append(
            {
                "species_id": sid,
                "chemical": chem,
                "chem_class": _guess_class(chem),
                "amount": 1.0,
            }
        )
    return pd.DataFrame(rows, columns=schema.TABLES["chemicals"]).drop_duplicates()


def download_archive(dest: Path) -> Path:
    """Download the Duke bulk CSV archive to ``dest`` (cached if present)."""
    import requests  # lazy

    dest = Path(dest)
    if dest.exists():
        log.info("Duke archive already present at %s", dest)
        return dest
    log.info("downloading Duke archive from %s", DUKE_BULK_URL)
    resp = requests.get(DUKE_BULK_URL, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def parse_archive(archive_path: Path, species: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Parse the Duke archive into normalized tables keyed to ``species``.

    ``species`` provides the scientific_name -> species_id map used to attach the
    Duke records (matched on lowercased binomial). Returns a dict with keys
    ``chemicals``, ``bioactivities``, ``uses``.
    """
    name_to_id = {
        str(n).strip().lower(): int(i)
        for n, i in zip(species["scientific_name"], species["species_id"])
    }

    frames: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(archive_path) as zf:
        members = {m.lower(): m for m in zf.namelist() if m.lower().endswith(".csv")}
        raw: dict[str, pd.DataFrame] = {}
        for low, member in members.items():
            try:
                with zf.open(member) as fh:
                    raw[low] = pd.read_csv(io.TextIOWrapper(fh, encoding="latin-1"), low_memory=False)
            except Exception as exc:  # pragma: no cover - data dependent
                log.warning("could not read %s: %s", member, exc)

    frames["chemicals"] = _extract_chemicals(raw, name_to_id)
    frames["bioactivities"] = _extract_bioactivities(raw)
    frames["uses"] = _extract_uses(raw, name_to_id)
    return frames


def _find_table(raw: dict[str, pd.DataFrame], needed: set[str]) -> pd.DataFrame | None:
    """Return the first CSV whose (lowercased) columns superset ``needed``."""
    for df in raw.values():
        cols = {c.lower() for c in df.columns}
        if needed.issubset(cols):
            return df
    return None


def _col(df: pd.DataFrame, name: str) -> str:
    for c in df.columns:
        if c.lower() == name:
            return c
    raise KeyError(name)


def _extract_chemicals(raw: dict[str, pd.DataFrame], name_to_id: dict[str, int]) -> pd.DataFrame:
    df = _find_table(raw, {"taxon", "chemical"})
    if df is None:
        log.warning("Duke: no chemical-occurrence table found")
        return pd.DataFrame(columns=schema.TABLES["chemicals"])
    taxon_c, chem_c = _col(df, "taxon"), _col(df, "chemical")
    rows = []
    for _, r in df.iterrows():
        sid = name_to_id.get(str(r[taxon_c]).strip().lower())
        if sid is None:
            continue
        chem = str(r[chem_c]).strip()
        rows.append(
            {
                "species_id": sid,
                "chemical": chem,
                "chem_class": _guess_class(chem),
                "amount": 1.0,
            }
        )
    return pd.DataFrame(rows, columns=schema.TABLES["chemicals"])


def _extract_bioactivities(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = _find_table(raw, {"chemical", "activity"})
    if df is None:
        log.warning("Duke: no bioactivity table found")
        return pd.DataFrame(columns=schema.TABLES["bioactivities"])
    chem_c, act_c = _col(df, "chemical"), _col(df, "activity")
    out = df[[chem_c, act_c]].rename(columns={chem_c: "chemical", act_c: "activity"})
    return out.dropna().astype(str)


def _extract_uses(raw: dict[str, pd.DataFrame], name_to_id: dict[str, int]) -> pd.DataFrame:
    df = _find_table(raw, {"taxon", "activity"})  # ethnobotanical-use table
    if df is None:
        log.warning("Duke: no ethnobotanical-use table found")
        return pd.DataFrame(columns=schema.TABLES["uses"])
    taxon_c, use_c = _col(df, "taxon"), _col(df, "activity")
    rows = []
    for _, r in df.iterrows():
        sid = name_to_id.get(str(r[taxon_c]).strip().lower())
        if sid is None:
            continue
        rows.append({"species_id": sid, "use_category": str(r[use_c]).strip()})
    return pd.DataFrame(rows, columns=schema.TABLES["uses"])


_CLASS_KEYWORDS = {
    "alkaloid": "alkaloid",
    "terpen": "terpene",
    "oil": "essential_oil",
    "tannin": "tannin",
    "flavon": "flavonoid",
    "glycerid": "glycoside",
    "glycos": "glycoside",
    "saponin": "saponin",
    "phenol": "phenol",
}


def _guess_class(chemical: str) -> str:
    low = chemical.lower()
    for kw, cls in _CLASS_KEYWORDS.items():
        if kw in low:
            return cls
    return "phenol"  # default coarse bucket
