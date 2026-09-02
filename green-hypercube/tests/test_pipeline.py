"""End-to-end pipeline tests on the sample data source."""

from __future__ import annotations

import numpy as np

from greenhypercube.data import schema


def test_manifold_shapes(small_manifold):
    m = small_manifold
    assert m.n == 120
    assert m.X.shape == (120, len(m.feature_names))
    assert m.D_phylo.shape == (120, 120)
    assert m.reward.shape == (120,)
    assert 0.0 <= m.reward.min() <= m.reward.max() <= 1.0


def test_reward_is_sparse_but_present(small_manifold):
    m = small_manifold
    assert m.n_useful > 0
    # Most species should be non-useful (sparse-reward haystack).
    assert m.useful_mask.mean() < 0.5


def test_features_align_with_schema(small_manifold):
    m = small_manifold
    expected = (
        len(schema.SENSORY_CHANNELS) + len(schema.CHEM_CLASSES) + 1
    )
    assert m.X.shape[1] == expected


def test_useful_species_have_higher_salience(small_manifold):
    # Sensory salience should, on average, be higher for useful species.
    m = small_manifold
    useful = m.sensory_salience[m.useful_mask].mean()
    other = m.sensory_salience[~m.useful_mask].mean()
    assert useful > other
