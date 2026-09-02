"""Taxon resolution: join all sources onto a single species table.

Live sources speak different name dialects. We anchor on the GBIF backbone
(accepted name + ``gbif_key``), attach the Open Tree ``ott_id``, and keep only
taxa confidently resolved across the sources we actually need (always GBIF;
OTL when a phylogeny is requested). Drop counts are logged so attrition is
transparent and reproducible.
"""

from __future__ import annotations

import pandas as pd

from ..utils import get_logger

log = get_logger("data.taxonomy")


def assign_species_ids(resolved: pd.DataFrame) -> pd.DataFrame:
    """Assign a dense 0..N-1 ``species_id`` and normalize taxonomy columns."""
    df = resolved.copy().reset_index(drop=True)
    df["species_id"] = range(len(df))
    for col in ("genus", "family", "order"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    if "scientific_name" not in df.columns:
        df["scientific_name"] = df.get("accepted_name", "")
    return df


def require_resolved(
    species: pd.DataFrame, need_ott: bool = True
) -> pd.DataFrame:
    """Drop species lacking the keys required downstream, logging attrition."""
    n0 = len(species)
    df = species.dropna(subset=["gbif_key"])
    n_gbif = len(df)
    if need_ott and "ott_id" in df.columns:
        df = df.dropna(subset=["ott_id"])
    n_final = len(df)
    log.info(
        "taxon resolution: %d seed -> %d with GBIF key -> %d fully resolved "
        "(dropped %d)",
        n0, n_gbif, n_final, n0 - n_final,
    )
    # Reassign dense ids after dropping so downstream arrays stay contiguous.
    return assign_species_ids(df.drop(columns=["species_id"], errors="ignore"))
