"""Open Tree of Life adapter: phylogeny for the resolved species.

Matches species names to Open Tree Taxonomy (OTT) ids via TNRS, then requests
the induced subtree relating them and stores it as Newick. Species that OTL
cannot place are reported back so the builder can drop them consistently.

The ``opentree`` package is imported lazily.
"""

from __future__ import annotations

import pandas as pd

from ..utils import get_logger
from . import schema
from ..utils.cache import ParquetCache

log = get_logger("data.opentree")


def resolve_ott_ids(species: pd.DataFrame) -> pd.DataFrame:
    """Add an ``ott_id`` column to ``species`` via TNRS matching."""
    from opentree import OT  # lazy

    names = species["scientific_name"].astype(str).tolist()
    try:
        result = OT.tnrs_match(names, do_approximate_matching=True)
        matches = result.response_dict["results"]
    except Exception as exc:  # pragma: no cover - network dependent
        log.warning("OTL TNRS failed: %s", exc)
        species = species.copy()
        species["ott_id"] = pd.NA
        return species

    name_to_ott: dict[str, int] = {}
    for entry in matches:
        ms = entry.get("matches", [])
        if ms:
            ott = ms[0]["taxon"]["ott_id"]
            name_to_ott[entry["name"]] = int(ott)
    out = species.copy()
    out["ott_id"] = out["scientific_name"].map(name_to_ott)
    log.info("OTL resolved %d/%d ott ids", out["ott_id"].notna().sum(), len(out))
    return out


def fetch_induced_subtree(species: pd.DataFrame, cache: ParquetCache) -> str:
    """Fetch the induced subtree (Newick) over species with an ``ott_id``.

    Tips are relabeled ``sp{species_id}`` so the builder can map them back. The
    Newick is written to the cache and also returned.
    """
    from opentree import OT  # lazy

    placed = species.dropna(subset=["ott_id"])
    ott_ids = [int(x) for x in placed["ott_id"].tolist()]
    if len(ott_ids) < 2:
        raise ValueError("need at least 2 OTT ids for an induced subtree")

    # Some OTT ids resolve via TNRS but are not present in the synthetic tree
    # (pruned/broken taxa); ignore_unknown_ids drops them server-side instead of
    # failing the whole request. Such species become phylogenetically isolated
    # (fallback distance) in the patristic matrix, which is handled downstream.
    out = OT.synth_induced_tree(
        ott_ids=ott_ids, label_format="id", ignore_unknown_ids=True
    )
    tree = out.tree

    # Relabel tips from ott id to sp{species_id}.
    ott_to_sid = {int(o): int(s) for o, s in zip(placed["ott_id"], placed["species_id"])}
    for leaf in tree.leaf_node_iter():
        label = leaf.taxon.label if leaf.taxon else None
        ott = _parse_ott(label)
        if ott is not None and ott in ott_to_sid:
            leaf.taxon.label = f"sp{ott_to_sid[ott]}"
    newick = tree.as_string(schema="newick")
    cache.write_text(schema.PHYLOGENY_NAME, newick)
    return newick


def _parse_ott(label: str | None) -> int | None:
    if not label:
        return None
    digits = "".join(ch for ch in label if ch.isdigit())
    return int(digits) if digits else None
