"""NAEB adapter: Native American Ethnobotany database (Moerman), offline dump.

This supplies an *independent measure of human use* -- the dependent variable our
hypothesis is actually about -- that is not chemistry or bioactivity. Reward
sourced from NAEB therefore breaks the coverage/provenance loop that ties ChEMBL
reward to research effort: a plant's reward is how much it was *documented as
used* (here, distinct medicinal/"Drug" uses across tribes and sources), entirely
separate from the chemistry, occurrence, and interaction cues.

NAEB is North American, so it must be paired with a North American flora (see
``configs/live_naeb_nam.yaml``); pairing it with a neotropical pool would waste
the signal.

Dump layout: ``<external_dir>/naeb_src/data/naeb_dump/{species,uses,
use_categories}.csv``. ``use_category`` 2 == "Drug".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import get_logger
from . import schema

log = get_logger("data.naeb")

DRUG_CATEGORY = 2  # use_categories.csv: 1 Food, 2 Drug, 3 Other, 4 Fiber, 5 Dye


def _src(external_dir: str | Path) -> Path:
    return Path(external_dir) / "naeb_src" / "data" / "naeb_dump"


def _binomial(name: str) -> str:
    """Extract a Genus species binomial from a NAEB name (drops authorities)."""
    toks = str(name).replace("\xa0", " ").split()
    if len(toks) >= 2 and toks[0][:1].isupper():
        return f"{toks[0]} {toks[1].lower()}"
    return str(name).strip()


def load_species(external_dir: str | Path) -> pd.DataFrame:
    """NAEB species -> [naeb_id, scientific_name, genus, family]."""
    df = pd.read_csv(_src(external_dir) / "species.csv", low_memory=False)
    binom = df["name"].map(_binomial)
    out = pd.DataFrame({
        "naeb_id": df["id"].astype(int),
        "scientific_name": binom,
        "genus": binom.str.split().str[0],
        "family": df.get("family_apg", df.get("family", "")).astype(str).str.strip(),
    })
    return out


def species_pool(external_dir: str | Path, n: int) -> pd.DataFrame:
    """Seed pool: the most-documented NA plants across *all* use categories.

    Seeding by total documented use (food/drug/fiber/dye/other), NOT by medicinal
    use, makes the reward -- medicinal (Drug) use intensity -- a genuinely sparse,
    naturally-embedded minority within a flora of well-attested useful plants.
    The search must locate the medicinal needles among food/fiber/dye species
    using ecological cues, with reward independent of chemistry.
    """
    sp = load_species(external_dir)
    total = _use_counts(external_dir, drug_only=False)  # naeb_id -> total uses
    sp["total_uses"] = sp["naeb_id"].map(total).fillna(0).astype(int)
    sp = sp[sp["total_uses"] > 0].sort_values("total_uses", ascending=False)
    sp = sp.drop_duplicates(subset=["scientific_name"]).head(n).reset_index(drop=True)
    drug = _use_counts(external_dir, drug_only=True)
    n_drug = int(sp["naeb_id"].map(drug).fillna(0).gt(0).sum())
    log.info("NAEB pool: %d documented-useful species (%d with medicinal use)",
             len(sp), n_drug)
    return sp[["scientific_name", "genus", "family"]]


def _use_counts(external_dir: str | Path, drug_only: bool) -> dict[int, int]:
    uses = pd.read_csv(_src(external_dir) / "uses.csv", low_memory=False,
                       usecols=["species", "use_category"])
    if drug_only:
        uses = uses[pd.to_numeric(uses["use_category"], errors="coerce") == DRUG_CATEGORY]
    return uses.groupby("species").size().to_dict()


def documentation_depth(external_dir: str | Path, name_to_id: dict[str, int]) -> pd.DataFrame:
    """Per-species NAEB documentation depth (reward-side confound covariates).

    Columns: species_id, total_uses, n_tribes, n_sources, drug_uses: counts of
    all documented uses, distinct reporting groups, distinct literature sources,
    and Drug-category uses. Used to residualize NAEB reward before coupling tests.
    """
    uses = pd.read_csv(
        _src(external_dir) / "uses.csv", low_memory=False,
        usecols=["species", "tribe", "source", "use_category"],
    )
    sp = load_species(external_dir)
    name_to_naeb: dict[str, int] = {}
    for nid, sci in zip(sp["naeb_id"], sp["scientific_name"]):
        name_to_naeb.setdefault(sci, int(nid))

    depth_by_naeb: dict[int, dict[str, float]] = {}
    for nid, grp in uses.groupby("species"):
        nid = int(nid)
        drug = grp[pd.to_numeric(grp["use_category"], errors="coerce") == DRUG_CATEGORY]
        depth_by_naeb[nid] = {
            "total_uses": float(len(grp)),
            "n_tribes": float(grp["tribe"].nunique()),
            "n_sources": float(grp["source"].nunique()),
            "drug_uses": float(len(drug)),
        }

    rows = []
    for name, sid in name_to_id.items():
        nid = name_to_naeb.get(str(name).strip())
        d = depth_by_naeb.get(nid, {}) if nid is not None else {}
        rows.append({
            "species_id": int(sid),
            "total_uses": d.get("total_uses", 0.0),
            "n_tribes": d.get("n_tribes", 0.0),
            "n_sources": d.get("n_sources", 0.0),
            "drug_uses": d.get("drug_uses", 0.0),
        })
    out = pd.DataFrame(rows)
    log.info("NAEB documentation depth: mean total_uses=%.1f tribes=%.1f sources=%.1f",
             out["total_uses"].mean(), out["n_tribes"].mean(), out["n_sources"].mean())
    return out


def log_pool_composition(bioassay: pd.DataFrame, label: str = "NAEB pool") -> None:
    """Log used vs unused counts for Moerman-style full-flora pools."""
    n = len(bioassay)
    n_used = int((bioassay["assay_value"] > 0).sum())
    n_raw = int((bioassay["raw"] > 0).sum()) if "raw" in bioassay.columns else n_used
    log.info(
        "%s: %d/%d species with NAEB Drug use (%.1f%% unused in pool)",
        label, n_used, n, 100.0 * (n - n_used) / max(n, 1),
    )
    log.info("%s: %d species with any raw Drug record before normalization", label, n_raw)


def use_reward(external_dir: str | Path, name_to_id: dict[str, int]) -> pd.DataFrame:
    """Reward = documented medicinal-use intensity (log-scaled), in [0, 1].

    For each pooled species, count distinct NAEB Drug-use records, log1p-compress
    (so a few heavily-documented species do not dominate), and normalize. Species
    with no documented medicinal use get 0. This is independent of every cue.
    """
    sp = load_species(external_dir)
    counts = _use_counts(external_dir, drug_only=True)
    name_to_count: dict[str, int] = {}
    for nid, sci in zip(sp["naeb_id"], sp["scientific_name"]):
        c = counts.get(int(nid), 0)
        if c > name_to_count.get(sci, 0):
            name_to_count[sci] = c  # dedup binomials -> max count

    rows = []
    for name, sid in name_to_id.items():
        rows.append({"species_id": int(sid),
                     "raw": float(name_to_count.get(str(name).strip(), 0))})
    df = pd.DataFrame(rows)
    raw = np.log1p(df["raw"].to_numpy(dtype=float))
    df["assay_value"] = raw / raw.max() if raw.max() > 0 else raw
    log.info("NAEB reward: %d/%d species with documented medicinal use",
             int((df["assay_value"] > 0).sum()), len(df))
    return df[schema.TABLES["bioassay"]]
