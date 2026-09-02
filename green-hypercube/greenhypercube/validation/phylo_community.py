"""Saslis-style phylogenetic community metrics (P7).

Complements pairwise cue->reward coupling (H2.1/H2.2) with clade-level estimands:

- **NRI / NTI** — net relatedness / nearest-taxon indices from MPD and MNTD
  (Webb et al. 2002; Milliken et al. 2021 sign convention).
- **Hot-node / taxon enrichment** — internal clades and genera with more used
  species than label-permutation null.

Label variants:

- ``raw`` — ``reward >= discovery_threshold`` (documented use / active assay).
- ``residual`` — rank-residualized reward on ``reward_depth`` > 0 (documentation
  control when depth covariates exist).
- ``effort_null`` — same raw labels, but NRI/NTI null draws match the effort
  distribution of the observed community (stratified sampling).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..hypercube import Manifold
from ..hypercube.phylo import clade_species_map, species_in_tree_mask
from ..utils import get_logger
from ..utils.rng import RNG
from .controls import effort_index
from .residual import rank_residualize

log = get_logger("validation.phylo_community")

NRI_THRESHOLD = 1.96


@dataclass(frozen=True)
class CommunityResult:
    label: str
    null_mode: str
    n_community: int
    n_pool: int
    mpd: float
    mntd: float
    nri: float
    nti: float
    nri_p: float
    nti_p: float
    mpd_null_mean: float
    mntd_null_mean: float


def mpd(D: np.ndarray, indices: np.ndarray) -> float:
    """Mean pairwise patristic distance among ``indices``."""
    if indices.size < 2:
        return 0.0
    sub = D[np.ix_(indices, indices)]
    iu = np.triu_indices(indices.size, k=1)
    vals = sub[iu]
    finite = vals[np.isfinite(vals)]
    return float(finite.mean()) if finite.size else 0.0


def mntd(D: np.ndarray, indices: np.ndarray) -> float:
    """Mean nearest-taxon distance within ``indices``."""
    if indices.size < 2:
        return 0.0
    sub = D[np.ix_(indices, indices)].astype(float, copy=True)
    np.fill_diagonal(sub, np.inf)
    nearest = sub.min(axis=1)
    finite = nearest[np.isfinite(nearest)]
    return float(finite.mean()) if finite.size else 0.0


def _nri_nti(
    D: np.ndarray,
    community: np.ndarray,
    pool: np.ndarray,
    rng: RNG,
    n_perm: int,
    null_mode: str,
    effort: np.ndarray | None,
) -> CommunityResult:
    """Compute NRI/NTI for one community vs null samples from ``pool``."""
    n_comm = int(community.size)
    if n_comm < 2 or pool.size < n_comm:
        return CommunityResult(
            label="", null_mode=null_mode, n_community=n_comm, n_pool=int(pool.size),
            mpd=0.0, mntd=0.0, nri=0.0, nti=0.0, nri_p=1.0, nti_p=1.0,
            mpd_null_mean=0.0, mntd_null_mean=0.0,
        )

    mpd_obs = mpd(D, community)
    mntd_obs = mntd(D, community)

    mpd_null = np.empty(n_perm, dtype=float)
    mntd_null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        if null_mode == "effort" and effort is not None and n_comm >= 2:
            sample = _effort_matched_sample(pool, community, effort, rng)
        else:
            sample = rng.choice(pool, size=n_comm, replace=False)
        mpd_null[i] = mpd(D, sample)
        mntd_null[i] = mntd(D, sample)

    mpd_mu, mpd_sd = float(mpd_null.mean()), float(mpd_null.std(ddof=1))
    mntd_mu, mntd_sd = float(mntd_null.mean()), float(mntd_null.std(ddof=1))

    nri = -(mpd_obs - mpd_mu) / mpd_sd if mpd_sd > 0 else 0.0
    nti = -(mntd_obs - mntd_mu) / mntd_sd if mntd_sd > 0 else 0.0

    # Two-sided permutation *rank* p-values (NOT normal z→p conversion).
    # Count null draws at least as extreme as observed MPD/MNTD deviation.
    nri_p = (1 + int(np.sum(np.abs(mpd_null - mpd_mu) >= abs(mpd_obs - mpd_mu)))) / (n_perm + 1)
    nti_p = (1 + int(np.sum(np.abs(mntd_null - mntd_mu) >= abs(mntd_obs - mntd_mu)))) / (n_perm + 1)

    return CommunityResult(
        label="", null_mode=null_mode, n_community=n_comm, n_pool=int(pool.size),
        mpd=mpd_obs, mntd=mntd_obs, nri=float(nri), nti=float(nti),
        nri_p=float(nri_p), nti_p=float(nti_p),
        mpd_null_mean=mpd_mu, mntd_null_mean=mntd_mu,
    )


def _effort_matched_sample(
    pool: np.ndarray,
    community: np.ndarray,
    effort: np.ndarray,
    rng: RNG,
    n_strata: int = 5,
) -> np.ndarray:
    """Draw ``len(community)`` species from ``pool`` matching community effort bins."""
    comm_eff = effort[community]
    pool_eff = effort[pool]
    if np.ptp(comm_eff) == 0 or np.ptp(pool_eff) == 0:
        return rng.choice(pool, size=community.size, replace=False)

    edges = np.quantile(comm_eff, np.linspace(0, 1, n_strata + 1))
    edges[-1] += 1e-9
    comm_bins = np.digitize(comm_eff, edges[1:-1], right=False)
    picked: list[int] = []
    for b in range(n_strata):
        need = int((comm_bins == b).sum())
        if need == 0:
            continue
        pool_bins = np.digitize(pool_eff, edges[1:-1], right=False)
        candidates = pool[(pool_bins == b)]
        if candidates.size == 0:
            candidates = pool
        replace = candidates.size < need
        picked.extend(rng.choice(candidates, size=need, replace=replace).tolist())
    if len(picked) < community.size:
        extra = community.size - len(picked)
        picked.extend(rng.choice(pool, size=extra, replace=False).tolist())
    return np.array(picked[: community.size], dtype=int)


def _label_variants(m: Manifold) -> dict[str, np.ndarray]:
    """Binary community labels under raw and documentation-residual variants."""
    raw = np.asarray(m.useful_mask, dtype=bool)
    out: dict[str, np.ndarray] = {"raw": raw}
    if m.reward_depth is not None:
        resid = rank_residualize(np.asarray(m.reward, dtype=float), m.reward_depth)
        out["residual"] = resid > 0.0
    return out


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR q-values for a vector of p-values."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = np.empty(m, dtype=float)
    ranked[order] = np.minimum.accumulate(
        (p[order] * m / (np.arange(m) + 1))[::-1]
    )[::-1]
    return np.clip(ranked, 0.0, 1.0)


def _hot_node_fp_context(hot: pd.DataFrame) -> pd.DataFrame:
    """Keystone-style FP accounting for clade/genus enrichment tests."""
    if hot.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for label, grp in hot.groupby("label"):
        for unit in ("all", "clade", "genus"):
            sub = grp if unit == "all" else grp[grp["unit"] == unit]
            if sub.empty:
                continue
            n = len(sub)
            sig = sub[(sub["p_value"] < 0.05) & sub["enriched"]]
            fdr_sig = sig[sig["q_value"] < 0.05] if "q_value" in sig.columns else sig.iloc[0:0]
            rows.append({
                "label": label,
                "unit": unit,
                "n_tested": n,
                "n_sig_p05_enriched": int(len(sig)),
                "e_fp_p05": float(n * 0.05),
                "n_fdr_005_enriched": int(len(fdr_sig)),
            })
    return pd.DataFrame(rows)


def measure_phylo_community(
    m: Manifold,
    newick: str,
    rng: RNG,
    n_perm: int = 999,
    min_clade_tips: int = 4,
    min_genus_tips: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run NRI/NTI and hot-node enrichment for Saslis-style estimands (P7).

    Returns ``(summary, hot_nodes, fp_context)`` DataFrames.

    NRI/NTI ``*_p`` columns are two-sided **permutation rank** p-values
    (``(1 + #{null ≥ observed}) / (n_perm + 1)``), not normal conversions of
    the reported z-like NRI/NTI indices.
    """
    D = np.asarray(m.D_phylo, dtype=float)
    in_tree = species_in_tree_mask(newick, m.n)
    pool = np.where(in_tree)[0]
    effort = effort_index(m) if m.coverage is not None or m.reward_depth is not None else None

    summary_rows: list[dict] = []
    hot_rows: list[dict] = []

    for label_name, labels in _label_variants(m).items():
        community = np.where(labels & in_tree)[0]
        log.info(
            "phylo_community[%s]: n_used=%d n_pool=%d",
            label_name, community.size, pool.size,
        )

        for null_mode in ("standard", "effort"):
            if null_mode == "effort" and effort is None:
                continue
            res = _nri_nti(
                D, community, pool, rng, n_perm,
                null_mode=null_mode,
                effort=effort if null_mode == "effort" else None,
            )
            summary_rows.append({
                "label": label_name,
                "null_mode": null_mode,
                "n_community": res.n_community,
                "n_pool": res.n_pool,
                "mpd": res.mpd,
                "mntd": res.mntd,
                "nri": res.nri,
                "nti": res.nti,
                "nri_p": res.nri_p,
                "nti_p": res.nti_p,
                "p_method": "permutation_rank",
                "nri_sig": abs(res.nri) >= NRI_THRESHOLD,
                "nti_sig": abs(res.nti) >= NRI_THRESHOLD,
                "mpd_null_mean": res.mpd_null_mean,
                "mntd_null_mean": res.mntd_null_mean,
            })

        hot_rows.extend(
            _hot_node_enrichment(
                m, labels, in_tree, newick, rng, n_perm,
                label_name, min_clade_tips, min_genus_tips,
            )
        )

    summary = pd.DataFrame(summary_rows)
    hot = pd.DataFrame(hot_rows)
    if not hot.empty:
        hot["q_value"] = np.nan
        for label, grp in hot.groupby("label"):
            idx = grp.index
            hot.loc[idx, "q_value"] = _bh_fdr(grp["p_value"].values)
        hot = hot.sort_values(["label", "p_value", "n_used"], ascending=[True, True, False])
    fp_context = _hot_node_fp_context(hot)
    return summary, hot, fp_context


def _hot_node_enrichment(
    m: Manifold,
    labels: np.ndarray,
    in_tree: np.ndarray,
    newick: str,
    rng: RNG,
    n_perm: int,
    label_name: str,
    min_clade_tips: int,
    min_genus_tips: int,
) -> list[dict]:
    """Permutation enrichment for internal clades and genera."""
    rows: list[dict] = []
    n_used_total = int(labels.sum())
    if n_used_total < 1:
        return rows

    pool = np.where(in_tree)[0]
    label_arr = np.asarray(labels, dtype=bool)

    for clade_name, species_ids in clade_species_map(newick, m.n):
        ids = np.array([i for i in species_ids if in_tree[i]], dtype=int)
        if ids.size < min_clade_tips:
            continue
        obs = int(label_arr[ids].sum())
        null = np.empty(n_perm, dtype=int)
        for i in range(n_perm):
            shuffled = label_arr[rng.permutation(m.n)]
            null[i] = int(shuffled[ids].sum())
        p = (1 + int(np.sum(null >= obs))) / (n_perm + 1)  # one-sided rank p (enrichment)
        rows.append({
            "label": label_name,
            "unit": "clade",
            "name": clade_name,
            "n_tips": int(ids.size),
            "n_used": obs,
            "frac_used": obs / ids.size,
            "null_mean": float(null.mean()),
            "p_value": float(p),
            "enriched": obs > null.mean(),
        })

    genus_groups = m.species.groupby("genus")["species_id"].apply(list)
    for genus, sid_list in genus_groups.items():
        ids = np.array([int(i) for i in sid_list if in_tree[int(i)]], dtype=int)
        if ids.size < min_genus_tips:
            continue
        obs = int(label_arr[ids].sum())
        null = np.empty(n_perm, dtype=int)
        for i in range(n_perm):
            shuffled = label_arr[rng.permutation(m.n)]
            null[i] = int(shuffled[ids].sum())
        p = (1 + int(np.sum(null >= obs))) / (n_perm + 1)  # one-sided rank p (enrichment)
        rows.append({
            "label": label_name,
            "unit": "genus",
            "name": str(genus),
            "n_tips": int(ids.size),
            "n_used": obs,
            "frac_used": obs / ids.size,
            "null_mean": float(null.mean()),
            "p_value": float(p),
            "enriched": obs > null.mean(),
        })

    return rows
