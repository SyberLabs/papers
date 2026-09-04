"""Assemble a :class:`Manifold` from the normalized cache tables.

This is where the four data faces fuse into one object:

- chemicals (+ class -> sensory channel) -> observable salience features
- occurrences -> Jaccard co-occurrence graph (G_eco)
- interactions -> Jaccard animal-association graph (G_bio)
- bioassay (independent measured potency) -> hidden reward
- phylogeny (Newick) -> patristic distance matrix

The reward construction is identical for sample and live data: it comes solely
from the ``bioassay`` table (synthetic assay, or live ChEMBL potency), kept
deliberately separate from every cue, then normalized to [0, 1].
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

from ..config import ManifoldConfig
from ..utils import ParquetCache, get_logger
from .manifold import Manifold
from .phylo import patristic_matrix
from ..data import schema

log = get_logger("hypercube.builder")


def build_manifold(cache: ParquetCache, cfg: ManifoldConfig) -> Manifold:
    species = cache.read_table("species").sort_values("species_id").reset_index(drop=True)
    n = len(species)
    sid_index = {int(s): i for i, s in enumerate(species["species_id"])}

    chemicals = cache.read_table("chemicals")
    occurrences = cache.read_table("occurrences")
    interactions = cache.read_table("interactions")
    bioassay = cache.read_table("bioassay")
    newick = cache.read_text(schema.PHYLOGENY_NAME)

    X, feature_names, sensory_salience = _build_features(
        n, sid_index, chemicals, occurrences
    )
    reward = _build_reward(n, sid_index, bioassay)
    reward = _sparsify_reward(reward, cfg.reward_top_frac)
    eco = _cooccurrence_graph(n, sid_index, occurrences, cfg.eco_edge_threshold)
    bio = _association_graph(n, sid_index, interactions, cfg.bio_edge_threshold)
    D_phylo = patristic_matrix(newick, n)
    coverage, coverage_names = _coverage_covariates(
        n, sid_index, occurrences, interactions, chemicals
    )
    reward_depth, reward_depth_names = _build_reward_depth(
        n, sid_index, chemicals, cache
    )

    manifold = Manifold(
        species=species,
        X=X,
        feature_names=feature_names,
        sensory_salience=sensory_salience,
        D_phylo=D_phylo,
        eco=eco,
        bio=bio,
        reward=reward,
        discovery_threshold=cfg.discovery_threshold,
        coverage=coverage,
        coverage_names=coverage_names,
        reward_depth=reward_depth,
        reward_depth_names=reward_depth_names,
    )
    n_chem = int((coverage[:, 2] > 0).sum()) if coverage is not None else 0
    log.info(
        "manifold built: n=%d, features=%d, useful=%d (%.1f%%), "
        "eco_edges=%d, bio_edges=%d, chem_covered=%d (zero-imputed sensory elsewhere)",
        n, X.shape[1], manifold.n_useful, 100 * manifold.useful_mask.mean(),
        eco.number_of_edges(), bio.number_of_edges(), n_chem,
    )
    return manifold


def _build_features(
    n: int,
    sid_index: dict[int, int],
    chemicals: pd.DataFrame,
    occurrences: pd.DataFrame,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Observable cue features: per-channel sensory salience + chem-class counts + range size."""
    channels = schema.SENSORY_CHANNELS
    chem_classes = list(schema.CHEM_CLASSES.keys())

    channel_amt = np.zeros((n, len(channels)), dtype=np.float32)
    class_count = np.zeros((n, len(chem_classes)), dtype=np.float32)

    ch_idx = {c: i for i, c in enumerate(channels)}
    cl_idx = {c: i for i, c in enumerate(chem_classes)}

    for _, r in chemicals.iterrows():
        i = sid_index.get(int(r["species_id"]))
        if i is None:
            continue
        cc = str(r["chem_class"])
        if cc in cl_idx:
            class_count[i, cl_idx[cc]] += 1.0
            channel = schema.CHEM_CLASSES.get(cc)
            if channel in ch_idx:
                channel_amt[i, ch_idx[channel]] += float(r.get("amount", 1.0))

    # Range size (number of distinct sites) as a spatial cue.
    range_size = np.zeros((n, 1), dtype=np.float32)
    if len(occurrences):
        counts = occurrences.groupby("species_id")["site_id"].nunique()
        for sid, c in counts.items():
            i = sid_index.get(int(sid))
            if i is not None:
                range_size[i, 0] = float(c)

    channel_amt = np.log1p(channel_amt)
    range_size = np.log1p(range_size)

    X = np.hstack([channel_amt, class_count, range_size])
    feature_names = (
        [f"sensory:{c}" for c in channels]
        + [f"chemclass:{c}" for c in chem_classes]
        + ["range_size"]
    )

    # Sensory salience = emphasis on the most behaviorally salient channels.
    salient = ["bitter", "aromatic", "pungent"]
    sal_idx = [ch_idx[c] for c in salient if c in ch_idx]
    sensory_salience = channel_amt[:, sal_idx].sum(axis=1).astype(np.float32)
    return X, feature_names, sensory_salience


def _coverage_covariates(
    n: int,
    sid_index: dict[int, int],
    occurrences: pd.DataFrame,
    interactions: pd.DataFrame,
    chemicals: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Per-species research-coverage proxies (raw counts).

    Well-studied plants accumulate more occurrence records, more recorded animal
    interactions and broader documented chemistry -- and are also more likely to
    have a measured reward. These counts let the coupling instrument partial out
    that study effort so an apparent cue->reward coupling can be checked against a
    pure coverage explanation. Returned counts are log1p-compressed.
    """
    names = ["occ_count", "interaction_count", "chem_count"]
    cov = np.zeros((n, len(names)), dtype=float)
    for col, table, key in (
        (0, occurrences, "site_id"),
        (1, interactions, "animal_taxon"),
        (2, chemicals, "chemical"),
    ):
        if len(table):
            counts = table.groupby("species_id")[key].nunique()
            for sid, c in counts.items():
                i = sid_index.get(int(sid))
                if i is not None:
                    cov[i, col] = float(c)
    return np.log1p(cov), names


def _build_reward_depth(
    n: int,
    sid_index: dict[int, int],
    chemicals: pd.DataFrame,
    cache: ParquetCache,
) -> tuple[np.ndarray | None, list[str] | None]:
    """Reward-side documentation / study-depth covariates (log1p counts).

    NAEB landscapes store a ``reward_depth`` table at ingest (total uses, tribes,
    sources). ChEMBL landscapes derive depth from chemistry-bridge breadth (Duke +
    LOTUS compound counts). Used only by the coupling instrument for symmetric
    confound control: never by search strategies.
    """
    if cache.has(schema.REWARD_DEPTH_NAME):
        df = cache.read_table(schema.REWARD_DEPTH_NAME).sort_values("species_id")
        cols = [c for c in df.columns if c != "species_id"]
        depth = np.zeros((n, len(cols)), dtype=float)
        for _, r in df.iterrows():
            i = sid_index.get(int(r["species_id"]))
            if i is not None:
                depth[i] = [float(r[c]) for c in cols]
        # Chemistry-coverage indicator: any Duke record (not depth among covered).
        chem_cov = np.zeros((n, 1), dtype=float)
        if len(chemicals):
            for sid in chemicals["species_id"].unique():
                i = sid_index.get(int(sid))
                if i is not None:
                    chem_cov[i, 0] = 1.0
        depth = np.hstack([depth, chem_cov])
        cols = cols + ["has_chem_record"]
        return np.log1p(depth), cols

    # ChEMBL / chemistry paths: compound-bridge breadth proxies study effort on reward.
    chem_n: dict[int, float] = {}
    if len(chemicals):
        for sid, c in chemicals.groupby("species_id")["chemical"].nunique().items():
            chem_n[int(sid)] = float(c)
    lotus_n: dict[int, float] = {}
    lotus_path = cache.dir / "lotus_compounds.parquet"
    if lotus_path.exists():
        lc = pd.read_parquet(lotus_path)
        if "inchikey" in lc.columns:
            for sid, c in lc.groupby("species_id")["inchikey"].nunique().items():
                lotus_n[int(sid)] = float(c)
    if not chem_n and not lotus_n:
        return None, None
    depth = np.zeros((n, 2), dtype=float)
    for sid, i in sid_index.items():
        depth[i, 0] = chem_n.get(sid, 0.0)
        depth[i, 1] = lotus_n.get(sid, 0.0)
    return np.log1p(depth), ["chem_bridge_count", "lotus_compound_count"]


def _build_reward(
    n: int,
    sid_index: dict[int, int],
    bioassay: pd.DataFrame,
) -> np.ndarray:
    """Hidden reward = independently measured bioactivity potency in [0, 1].

    Reward provenance is deliberately separate from every cue: it comes from the
    ``bioassay`` table (synthetic assay, or live ChEMBL potency), never from the
    chemistry/sensory cue, the ecological/animal graphs, or documented uses.
    """
    reward = np.zeros(n, dtype=np.float32)
    if len(bioassay):
        for sid, val in zip(bioassay["species_id"], bioassay["assay_value"]):
            i = sid_index.get(int(sid))
            if i is not None:
                reward[i] = float(val)
    m = reward.max()
    if m > 0:
        reward = reward / m
    return reward.astype(np.float32)


def _sparsify_reward(reward: np.ndarray, top_frac: float) -> np.ndarray:
    """Keep only the top ``top_frac`` of species by reward; zero the rest.

    Imposes a controlled reward density (the "needle in a haystack" knob) without
    touching the cues, so the cue-reward coupling is preserved among survivors
    while the landscape is made arbitrarily sparse.
    """
    if top_frac >= 1.0:
        return reward
    n = len(reward)
    keep = max(1, int(round(float(np.clip(top_frac, 0.0, 1.0)) * n)))
    out = np.zeros_like(reward)
    top_idx = np.argsort(reward)[-keep:]
    out[top_idx] = reward[top_idx]
    return out.astype(reward.dtype)


def _cooccurrence_graph(
    n: int, sid_index: dict[int, int], occurrences: pd.DataFrame, threshold: float
) -> nx.Graph:
    """Link species whose site sets overlap (Jaccard) above ``threshold``."""
    g = nx.Graph()
    g.add_nodes_from(range(n))
    if not len(occurrences):
        return g
    sets: dict[int, set[int]] = {}
    for sid, grp in occurrences.groupby("species_id"):
        i = sid_index.get(int(sid))
        if i is not None:
            sets[i] = set(int(s) for s in grp["site_id"])
    # Invert to candidate pairs sharing at least one site (keeps it sparse).
    site_to_species: dict[int, list[int]] = {}
    for i, ss in sets.items():
        for s in ss:
            site_to_species.setdefault(s, []).append(i)
    candidates: set[tuple[int, int]] = set()
    for members in site_to_species.values():
        for a, b in combinations(sorted(set(members)), 2):
            candidates.add((a, b))
    for a, b in candidates:
        j = _jaccard(sets[a], sets[b])
        if j >= threshold:
            g.add_edge(a, b, weight=j)
    return g


def _association_graph(
    n: int, sid_index: dict[int, int], interactions: pd.DataFrame, threshold: float
) -> nx.Graph:
    """Link species sharing animal associates (Jaccard) above ``threshold``."""
    g = nx.Graph()
    g.add_nodes_from(range(n))
    if not len(interactions):
        return g
    sets: dict[int, set[str]] = {}
    for sid, grp in interactions.groupby("species_id"):
        i = sid_index.get(int(sid))
        if i is not None:
            sets[i] = set(str(a) for a in grp["animal_taxon"])
    animal_to_species: dict[str, list[int]] = {}
    for i, aset in sets.items():
        for a in aset:
            animal_to_species.setdefault(a, []).append(i)
    candidates: set[tuple[int, int]] = set()
    for members in animal_to_species.values():
        for a, b in combinations(sorted(set(members)), 2):
            candidates.add((a, b))
    for a, b in candidates:
        j = _jaccard(sets[a], sets[b])
        if j >= threshold:
            g.add_edge(a, b, weight=j)
    return g


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)
