"""Synthetic sample data source with EXPLICIT, tunable cue-reward coupling.

This generator exists to make the synthetic study a controlled sensitivity
analysis rather than a foregone conclusion. The design separates a latent
bioactivity variable ``z`` (which drives the reward through an independent assay
channel) from the cues, and couples each cue to ``z`` with a fidelity we set
explicitly:

    fidelity(channel) = clip(signal_strength * coupling[channel], 0, 1)

- ``signal_strength = 0``  -> every cue is statistically independent of the
  reward. This is the built-in null: structured search MUST then match random.
- ``signal_strength = 1``  -> cues are strong (but still noisy) reflections of z.

The reward (``bioassay`` table) is the latent passed through an independent
measurement channel with its own noise; it is never a copy of any cue, and it is
not derived from documented ethnobotanical use. Common cause (z) between
chemistry and bioactivity is intentional and realistic -- the point is that its
strength is a knob we can turn to zero and verify the null.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DataConfig
from ..utils import ParquetCache, get_logger
from ..utils.rng import make_rng
from . import schema

log = get_logger("data.sample")

_FAMILIES = [
    "Apocynaceae", "Solanaceae", "Fabaceae", "Rubiaceae", "Myristicaceae",
    "Malpighiaceae", "Euphorbiaceae", "Arecaceae", "Moraceae", "Annonaceae",
    "Piperaceae", "Lauraceae", "Melastomataceae", "Bignoniaceae", "Sapindaceae",
    "Asteraceae", "Lamiaceae", "Cucurbitaceae", "Bromeliaceae", "Poaceae",
]

# Chemical classes split into "salient" (drive the sensory cue) and noise.
_SALIENT_CLASSES = [c for c, ch in schema.CHEM_CLASSES.items() if ch in ("bitter", "aromatic", "pungent")]
_NOISE_CLASSES = [c for c in schema.CHEM_CLASSES if c not in _SALIENT_CLASSES]


def _standardize(v: np.ndarray) -> np.ndarray:
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else v - v.mean()


def _sigmoid(v: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-v))


def generate(cfg: DataConfig, cache: ParquetCache, seed: int) -> None:
    """Generate the full normalized snapshot into ``cache``."""
    rng = make_rng(seed)
    n = cfg.n_species

    s = float(np.clip(cfg.signal_strength, 0.0, 1.0))
    mult = cfg.coupling or {}

    def cpl(name: str) -> float:
        return float(np.clip(s * float(mult.get(name, 1.0)), 0.0, 1.0))

    c_phylo, c_sens, c_eco, c_bio = cpl("phylo"), cpl("sensory"), cpl("eco"), cpl("bio")

    # --- taxonomy ------------------------------------------------------------
    n_families = min(len(_FAMILIES), max(6, n // 25))
    families = _FAMILIES[:n_families]
    fam_weights = rng.dirichlet(np.ones(n_families) * 1.5)
    fam_assign = rng.choice(n_families, size=n, p=fam_weights)

    species_rows = []
    for i in range(n):
        fam = families[fam_assign[i]]
        genus = f"{fam[:4]}genus{int(fam_assign[i] * 1000 + (i % 7)) % 50}"
        species_rows.append(
            {
                "species_id": i,
                "scientific_name": f"{genus} sp{i}",
                "genus": genus,
                "family": fam,
                "order": f"Order_{fam_assign[i] % 6}",
            }
        )
    species = pd.DataFrame(species_rows)
    genus_codes, _ = pd.factorize(species["genus"])

    # --- latent bioactivity z, phylo-clustered with fidelity c_phylo ---------
    fam_base = rng.normal(0, 1, size=n_families)[fam_assign]
    genus_base = rng.normal(0, 1, size=genus_codes.max() + 1)[genus_codes]
    clustered = _standardize(0.7 * fam_base + 0.5 * genus_base)
    z = np.sqrt(c_phylo) * clustered + np.sqrt(1.0 - c_phylo) * rng.normal(0, 1, size=n)
    z = _standardize(z)

    # --- reward: independent assay channel on z ------------------------------
    assay_latent = z + cfg.assay_noise * rng.normal(0, 1, size=n)
    target = int(max(1, round(cfg.reward_density * n)))
    active_idx = np.argsort(assay_latent)[-target:]
    assay_value = np.zeros(n, dtype=float)
    la = assay_latent[active_idx]
    rng_la = (la - la.min()) / (la.max() - la.min() + 1e-9)
    assay_value[active_idx] = 0.4 + 0.6 * rng_la
    is_active = np.zeros(n, dtype=bool)
    is_active[active_idx] = True
    log.info(
        "sample: n=%d signal=%.2f (phylo=%.2f sens=%.2f eco=%.2f bio=%.2f) actives=%d (%.1f%%)",
        n, s, c_phylo, c_sens, c_eco, c_bio, target, 100 * target / n,
    )

    # --- sensory cue: chemistry encodes a z-coupled salience -----------------
    s_sens = _standardize(np.sqrt(c_sens) * z + np.sqrt(1.0 - c_sens) * rng.normal(0, 1, size=n))
    sens_level = _sigmoid(1.8 * s_sens)
    chem_rows = []
    for i in range(n):
        k_sal = int(np.clip(rng.poisson(0.3 + 3.0 * sens_level[i]), 0, len(_SALIENT_CLASSES) * 2))
        for _ in range(k_sal):
            cc = str(rng.choice(_SALIENT_CLASSES))
            chem_rows.append(
                {
                    "species_id": i,
                    "chemical": f"{cc}_{rng.integers(0, 500)}",
                    "chem_class": cc,
                    "amount": float(np.round(rng.gamma(2.0, 0.5 + sens_level[i]), 3)),
                }
            )
        # Noise chemistry unrelated to z (populates non-salient features).
        k_noise = int(rng.poisson(0.9))
        for _ in range(k_noise):
            cc = str(rng.choice(_NOISE_CLASSES))
            chem_rows.append(
                {
                    "species_id": i,
                    "chemical": f"{cc}_{rng.integers(0, 500)}",
                    "chem_class": cc,
                    "amount": float(np.round(rng.gamma(2.0, 1.0), 3)),
                }
            )
    chemicals = pd.DataFrame(chem_rows, columns=schema.TABLES["chemicals"])

    # --- ecological cue: habitat (site) membership coupled to z via c_eco ----
    eco_level = _sigmoid(1.8 * _standardize(
        np.sqrt(c_eco) * z + np.sqrt(1.0 - c_eco) * rng.normal(0, 1, size=n)
    ))
    n_sites = 60
    rich_sites = set(rng.choice(n_sites, size=max(4, n_sites // 6), replace=False).tolist())
    occ_rows = []
    for i in range(n):
        n_occ = int(rng.integers(3, 10))
        w = np.ones(n_sites)
        for st in rich_sites:
            w[st] += 8.0 * eco_level[i]
        w = w / w.sum()
        for st in set(rng.choice(n_sites, size=n_occ, replace=True, p=w).tolist()):
            occ_rows.append({"species_id": i, "site_id": int(st)})
    occurrences = pd.DataFrame(occ_rows, columns=schema.TABLES["occurrences"])

    # --- animal cue: shared associates coupled to z via c_bio ----------------
    bio_level = _sigmoid(1.8 * _standardize(
        np.sqrt(c_bio) * z + np.sqrt(1.0 - c_bio) * rng.normal(0, 1, size=n)
    ))
    n_animals = 80
    pref_animals = set(rng.choice(n_animals, size=max(5, n_animals // 8), replace=False).tolist())
    itypes = ["pollinatedBy", "eatenBy", "visitedBy"]
    inter_rows = []
    for i in range(n):
        n_int = int(rng.integers(0, 6))
        w = np.ones(n_animals)
        for a in pref_animals:
            w[a] += 7.0 * bio_level[i]
        w = w / w.sum()
        for a in set(rng.choice(n_animals, size=n_int, replace=True, p=w).tolist()):
            inter_rows.append(
                {
                    "species_id": i,
                    "animal_taxon": f"Animal_{int(a)}",
                    "interaction_type": str(rng.choice(itypes)),
                }
            )
    interactions = pd.DataFrame(inter_rows, columns=schema.TABLES["interactions"])

    bioassay = pd.DataFrame({"species_id": np.arange(n), "assay_value": assay_value})

    newick = _taxonomy_newick(species)

    cache.write_table("species", species)
    cache.write_table("occurrences", occurrences)
    cache.write_table("chemicals", chemicals)
    cache.write_table("interactions", interactions)
    cache.write_table("bioassay", bioassay)
    cache.write_text(schema.PHYLOGENY_NAME, newick)


def _taxonomy_newick(species: pd.DataFrame) -> str:
    """Build a Newick string from the taxonomic hierarchy with branch lengths."""

    def fmt_species(sid: int) -> str:
        return f"sp{sid}:1.0"

    order_parts = []
    for order, ofam in species.groupby("order"):
        fam_parts = []
        for fam, fgen in ofam.groupby("family"):
            gen_parts = []
            for genus, gsp in fgen.groupby("genus"):
                sp_parts = [fmt_species(int(x)) for x in gsp["species_id"]]
                gen_parts.append(f"({','.join(sp_parts)}){_safe(genus)}:1.0")
            fam_parts.append(f"({','.join(gen_parts)}){_safe(fam)}:1.0")
        order_parts.append(f"({','.join(fam_parts)}){_safe(order)}:1.0")
    return f"({','.join(order_parts)})root;"


def _safe(label: str) -> str:
    for ch in "(),:; ":
        label = label.replace(ch, "_")
    return label
