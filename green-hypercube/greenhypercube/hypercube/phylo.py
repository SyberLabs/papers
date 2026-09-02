"""Lightweight Newick parsing and patristic distance computation.

We avoid heavy phylogenetics dependencies. A small recursive-descent parser
turns a Newick string into a weighted tree (networkx graph); pairwise patristic
distances among the ``sp{species_id}`` tips are then computed with Dijkstra.

For the few-hundred-tip trees used here, an all-pairs computation is cheap and
gives a clean (n, n) distance matrix aligned to ``species_id`` order.
"""

from __future__ import annotations

import re

import networkx as nx
import numpy as np

from ..utils import get_logger

log = get_logger("hypercube.phylo")


class _Node:
    __slots__ = ("name", "length", "children")

    def __init__(self) -> None:
        self.name: str | None = None
        self.length: float = 1.0
        self.children: list["_Node"] = []


def parse_newick(text: str) -> _Node:
    """Parse a Newick string into a tree of ``_Node``."""
    text = text.strip()
    if not text.endswith(";"):
        text += ";"
    pos = 0

    def parse_node() -> _Node:
        nonlocal pos
        node = _Node()
        if text[pos] == "(":
            pos += 1  # consume '('
            while True:
                node.children.append(parse_node())
                if text[pos] == ",":
                    pos += 1
                    continue
                if text[pos] == ")":
                    pos += 1
                    break
        # read label
        m = re.match(r"[^,():;]*", text[pos:])
        label = m.group(0)
        pos += len(label)
        # read branch length
        if pos < len(text) and text[pos] == ":":
            pos += 1
            lm = re.match(r"[0-9eE.+-]+", text[pos:])
            if lm:
                node.length = float(lm.group(0))
                pos += len(lm.group(0))
        node.name = label or None
        return node

    root = parse_node()
    return root


def _to_graph(root: _Node) -> tuple[nx.Graph, list[str]]:
    """Convert the parsed tree to a weighted graph; return graph + tip labels."""
    g = nx.Graph()
    tips: list[str] = []
    counter = {"i": 0}

    def add(node: _Node, parent_id: int | None) -> None:
        if node.name and not node.children:
            node_id = node.name  # use tip label as node id (unique)
            tips.append(node.name)
        else:
            node_id = f"__internal_{counter['i']}"
            counter["i"] += 1
        g.add_node(node_id)
        if parent_id is not None:
            g.add_edge(parent_id, node_id, weight=max(node.length, 1e-6))
        for ch in node.children:
            add(ch, node_id)

    add(root, None)
    return g, tips


def species_in_tree_mask(newick: str, n_species: int) -> np.ndarray:
    """Boolean mask: species_id ``i`` is a labelled tip ``sp{i}`` in ``newick``."""
    root = parse_newick(newick)
    g, tips = _to_graph(root)
    tip_set = set(tips)
    mask = np.zeros(n_species, dtype=bool)
    for i in range(n_species):
        if f"sp{i}" in tip_set:
            mask[i] = True
    return mask


def clade_species_map(newick: str, n_species: int) -> list[tuple[str, list[int]]]:
    """Map each internal clade to descendant species ids (tip labels ``sp{id}``)."""
    root = parse_newick(newick)
    clades: list[tuple[str, list[int]]] = []
    counter = {"i": 0}

    def walk(node: _Node, path: str) -> list[int]:
        if node.name and not node.children:
            try:
                sid = int(node.name[2:]) if node.name.startswith("sp") else -1
            except ValueError:
                sid = -1
            return [sid] if 0 <= sid < n_species else []

        ids: list[int] = []
        for j, ch in enumerate(node.children):
            ids.extend(walk(ch, f"{path}_{j}"))
        if len(ids) >= 2:
            name = node.name or f"__internal_{counter['i']}"
            counter["i"] += 1
            clades.append((name, ids))
        return ids

    walk(root, "root")
    return clades


def patristic_matrix(newick: str, n_species: int) -> np.ndarray:
    """Return an (n, n) patristic distance matrix indexed by species_id.

    Tips must be labeled ``sp{species_id}``. Species absent from the tree get a
    large fallback distance so they are treated as phylogenetically isolated.
    """
    root = parse_newick(newick)
    g, tips = _to_graph(root)

    tip_set = set(tips)
    present = [f"sp{i}" for i in range(n_species) if f"sp{i}" in tip_set]
    log.info("phylogeny: %d/%d species placed in tree", len(present), n_species)

    # All-pairs shortest path lengths among present tips.
    lengths = dict(nx.all_pairs_dijkstra_path_length(g, weight="weight"))

    finite = [
        lengths[a][b]
        for a in present
        for b in present
        if a in lengths and b in lengths[a] and a != b
    ]
    fallback = float(np.percentile(finite, 95)) * 1.5 if finite else 10.0

    D = np.full((n_species, n_species), fallback, dtype=np.float32)
    np.fill_diagonal(D, 0.0)
    for i in range(n_species):
        a = f"sp{i}"
        if a not in lengths:
            continue
        la = lengths[a]
        for j in range(n_species):
            b = f"sp{j}"
            if b in la:
                D[i, j] = la[b]
    return D
