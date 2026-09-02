"""Distribution sampling utilities.

All variables live on (a clipped subset of) the unit interval. This module
provides marginal samplers, quantile functions (for copula transforms), and a
Gaussian-copula routine for correlated sampling.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from dtbr_mc.config.schemas import DistributionConfig


def sample_distribution(cfg: DistributionConfig, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` independent samples from ``cfg`` and clip to ``cfg.clip``."""
    lo, hi = cfg.clip
    if cfg.kind == "uniform":
        x = rng.uniform(cfg.low, cfg.high, size=n)
    elif cfg.kind == "normal":
        x = rng.normal(cfg.mean, cfg.std, size=n)
    elif cfg.kind == "beta":
        x = rng.beta(cfg.a, cfg.b, size=n)
    elif cfg.kind == "constant":
        x = np.full(n, cfg.value, dtype=float)
    elif cfg.kind == "mixture":
        comps = cfg.components or []
        w = np.asarray(cfg.weights, dtype=float) if cfg.weights else np.ones(len(comps))
        w = w / w.sum()
        idx = rng.choice(len(comps), size=n, p=w)
        x = np.empty(n, dtype=float)
        for k, comp in enumerate(comps):
            mask = idx == k
            if mask.any():
                x[mask] = sample_distribution(comp, int(mask.sum()), rng)
    else:  # pragma: no cover - guarded by pydantic Literal
        raise ValueError(f"unknown distribution kind: {cfg.kind}")
    return np.clip(x, lo, hi)


def quantile(cfg: DistributionConfig, u: np.ndarray) -> np.ndarray:
    """Inverse CDF of the marginal at probabilities ``u`` (for copula transforms).

    Only the smooth families (uniform/normal/beta) have a usable closed-form ppf
    here. ``constant`` collapses to its value; ``mixture`` is not supported under
    correlation (callers fall back to independent sampling).
    """
    lo, hi = cfg.clip
    if cfg.kind == "uniform":
        x = stats.uniform(loc=cfg.low, scale=max(cfg.high - cfg.low, 1e-12)).ppf(u)
    elif cfg.kind == "normal":
        x = stats.norm(loc=cfg.mean, scale=max(cfg.std, 1e-12)).ppf(u)
    elif cfg.kind == "beta":
        x = stats.beta(cfg.a, cfg.b).ppf(u)
    elif cfg.kind == "constant":
        x = np.full_like(u, cfg.value, dtype=float)
    else:
        raise ValueError(f"quantile not supported for kind={cfg.kind}")
    return np.clip(x, lo, hi)


def correlated_uniforms(
    corr: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Gaussian-copula uniforms with target correlation ``corr`` (d x d, PSD).

    Returns an ``(n, d)`` array of U(0,1) variables whose rank dependence matches
    the supplied correlation matrix (up to the usual copula approximation).
    """
    d = corr.shape[0]
    # Nearest-PSD safety net: clip negative eigenvalues, then re-symmetrise.
    vals, vecs = np.linalg.eigh(corr)
    vals = np.clip(vals, 1e-8, None)
    psd = vecs @ np.diag(vals) @ vecs.T
    dinv = np.diag(1.0 / np.sqrt(np.diag(psd)))
    psd = dinv @ psd @ dinv  # rescale to unit diagonal
    chol = np.linalg.cholesky(psd)
    z = rng.standard_normal(size=(n, d)) @ chol.T
    return stats.norm.cdf(z)
