"""Tests for Newick parsing and patristic distances."""

from __future__ import annotations

import numpy as np

from greenhypercube.hypercube.phylo import parse_newick, patristic_matrix


def test_parse_simple_newick():
    root = parse_newick("(sp0:1.0,sp1:1.0)root;")
    assert len(root.children) == 2
    names = {c.name for c in root.children}
    assert names == {"sp0", "sp1"}


def test_patristic_matrix_structure():
    # sp0 and sp1 are siblings; sp2 is an outgroup -> d(0,1) < d(0,2).
    newick = "((sp0:1.0,sp1:1.0)g0:1.0,sp2:2.0)root;"
    D = patristic_matrix(newick, n_species=3)
    assert D.shape == (3, 3)
    assert np.allclose(np.diag(D), 0.0)
    assert D[0, 1] < D[0, 2]
    assert D[0, 1] == D[1, 0]  # symmetric


def test_missing_tips_get_fallback():
    newick = "(sp0:1.0,sp1:1.0)root;"
    D = patristic_matrix(newick, n_species=4)  # sp2, sp3 absent
    assert D.shape == (4, 4)
    assert D[2, 3] > 0  # fallback distance applied
