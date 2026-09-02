"""Validation: negative-control null models for the search benchmark.

These transforms destroy a specific kind of cue-reward structure while
preserving marginal distributions. Under them, a strategy that genuinely
exploits that structure must collapse toward the random baseline; if it does
not, the apparent advantage is an artifact (leakage or a metric quirk) rather
than real predictive signal.
"""

from __future__ import annotations

from .controls import (
    NULLS,
    make_null,
    null_advantage_table,
    permute_reward,
    rewire_graphs,
    shuffle_phylo,
    all_nulls,
)
from .coupling import measure_coupling
from .multivariate import measure_multivariate_coupling
from .phylo_community import measure_phylo_community
from .m2_matched import run_matched_genuine_batch, summarize_matched_genuine

__all__ = [
    "NULLS",
    "make_null",
    "null_advantage_table",
    "permute_reward",
    "rewire_graphs",
    "shuffle_phylo",
    "all_nulls",
    "measure_coupling",
    "measure_multivariate_coupling",
    "measure_phylo_community",
    "run_matched_genuine_batch",
    "summarize_matched_genuine",
]
