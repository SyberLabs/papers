"""
grokking_scaling_theory.survival_fit
------------------------------------
Censoring-aware scaling-law fits via a log-normal Accelerated Failure Time
(AFT) model, i.e. Tobit regression in log space.

Motivation
==========
`fit_competition` currently drops censored runs (runs that hit the training
budget without grokking). A censored run is not missing data: it is the
one-sided observation ``tau > T_budget``. Excluding it biases fitted
exponents wherever censoring correlates with the covariates -- which it
does here, since the only censored run (p=31) sits at the small-p edge of
the transition zone, exactly where the inverted-scaling claim is decided.

Model
=====
    log tau_i = x_i . theta + sigma * eps_i,     eps_i ~ N(0, 1)

Observed runs contribute the normal density; censored runs contribute the
survival probability::

    L_i = phi((log tau_i - x_i.theta) / sigma) / sigma        (event)
    L_i = 1 - Phi((log T_i  - x_i.theta) / sigma)             (censored)

Designs provided
================
- ``free``:        log tau = c0 + a*log p + q*log log p - beta*log wd
- ``fixed_p2``:    log tau = c0 + 2*log p + q*log log p - beta*log wd
                   (the repository's preferred fixed-burden form)
- ``power_only``:  log tau = c0 + a*log p - beta*log wd
                   (transition-zone form; no log-correction term)

Conventions match `fit_competition`: everything is fit in natural-log space
and reported alongside leave-one-out-style comparisons where meaningful.

Run selection
=============
- Events:   ``dataset.fit_runs()`` (grokked, include_in_scaling_fit=True).
- Censored: runs with no observed grokking epoch whose notes explicitly
  mark censoring (e.g. the p=31 row). Trace-only rows excluded from the
  fit for curation reasons are *not* silently treated as censored.

Usage
=====
    python -m grokking_scaling_theory.survival_fit

or::

    from grokking_scaling_theory.survival_fit import censoring_impact_report
    report = censoring_impact_report()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, stats

from grokking_scaling_theory.scaling_study import GrokkingRun, ScalingDataset

DEFAULT_RUN_TABLE = (
    Path(__file__).resolve().parents[2] / "data" / "empirical_scaling_runs.csv"
)

# --------------------------------------------------------------------------
# Design matrices
# --------------------------------------------------------------------------

DesignFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _design_free(p: np.ndarray, wd: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.ones_like(p), np.log(p), np.log(np.log(p)), -np.log(wd)]
    )


def _design_fixed_p2(p: np.ndarray, wd: np.ndarray) -> np.ndarray:
    # 2*log p enters through the offset; free terms are log-correction + wd.
    return np.column_stack([np.ones_like(p), np.log(np.log(p)), -np.log(wd)])


def _design_power_only(p: np.ndarray, wd: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(p), np.log(p), -np.log(wd)])


DESIGNS: Dict[str, Tuple[DesignFn, List[str], Callable[[np.ndarray], np.ndarray]]] = {
    "free": (_design_free, ["log_C", "a", "q", "beta"], lambda p: np.zeros_like(p)),
    "fixed_p2": (
        _design_fixed_p2,
        ["log_C", "q", "beta"],
        lambda p: 2.0 * np.log(p),
    ),
    "power_only": (
        _design_power_only,
        ["log_C", "a", "beta"],
        lambda p: np.zeros_like(p),
    ),
}

# NOTE on signs: the classical repository form is
#   tau = C * p^a / ((log p)^q * wd^beta)
# so in log space q enters with a *negative* sign on log log p and beta with
# a negative sign on log wd. The designs above absorb those signs so that the
# reported q and beta are directly comparable to COLLAPSE_ANALYSIS_RESULTS.md
# (positive q means log correction speeds grokking; positive beta means
# weight decay shortens tau).


def _apply_sign_conventions(name: str, params: np.ndarray) -> np.ndarray:
    """Flip fitted coefficients into the repository's (q, beta > 0) convention."""
    out = params.copy()
    if name == "free":
        out[2] = -params[2]  # q
    if name == "fixed_p2":
        out[1] = -params[1]  # q
    return out


def _sign_vector(name: str, n_params: int) -> np.ndarray:
    signs = np.ones(n_params)
    if name == "free":
        signs[2] = -1.0
    if name == "fixed_p2":
        signs[1] = -1.0
    return signs


# --------------------------------------------------------------------------
# Run selection
# --------------------------------------------------------------------------


def select_runs(
    dataset: ScalingDataset,
    modulus_range: Optional[Tuple[int, int]] = None,
    architectures: Optional[Sequence[str]] = None,
) -> Tuple[List[GrokkingRun], List[GrokkingRun]]:
    """Return (event_runs, censored_runs) for survival fitting.

    Censored runs must be explicitly marked (notes contain 'censored'),
    which excludes trace-only rows held out for curation reasons.
    """

    def in_scope(run: GrokkingRun) -> bool:
        if modulus_range is not None:
            lo, hi = modulus_range
            if not (lo <= run.condition.modulus <= hi):
                return False
        if architectures is not None:
            if run.condition.architecture not in architectures:
                return False
        return True

    def is_effectively_censored(run: GrokkingRun) -> bool:
        # The loader's is_censored property only fires when grokking_epoch is
        # missing. Rows like p=31 record grokking_epoch == max_epochs with an
        # explicit CENSORED note; both patterns are right-censoring.
        # Only explicit markers count: published anchor rows encode
        # max_epochs == grokking_epoch (budget unknown), so a >= max_epochs
        # heuristic would misclassify them as censored.
        if "censored" in (run.notes or "").lower():
            return True
        return run.grokking_epoch is None

    events = [
        run
        for run in dataset.fit_runs()
        if in_scope(run) and not is_effectively_censored(run)
    ]
    censored = [
        run
        for run in dataset.runs
        if in_scope(run)
        and is_effectively_censored(run)
        and run.max_epochs > 0
        and run.include_in_scaling_fit is False  # curation flag doubles as marker
        and "trace" not in (run.source or "").lower()  # skip curation-pending traces
    ]
    return events, censored


# --------------------------------------------------------------------------
# AFT model
# --------------------------------------------------------------------------


@dataclass
class AFTResult:
    """Fit summary for one log-normal AFT model."""

    design: str
    param_names: List[str]
    params: Dict[str, float]
    std_errors: Dict[str, float]
    sigma: float
    log_likelihood: float
    n_events: int
    n_censored: int
    converged: bool
    notes: str = ""
    _raw: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    _keep: List[int] = field(default_factory=list, repr=False)

    def predict_tau(self, modulus: float, weight_decay: float) -> float:
        design_fn, names, offset_fn = DESIGNS[self.design]
        p = np.array([float(modulus)])
        wd = np.array([float(weight_decay)])
        X = design_fn(p, wd)
        keep = self._keep or list(range(len(names)))
        mu = offset_fn(p) + X[:, keep] @ self._raw[: len(keep)]
        return float(np.exp(mu[0]))


def _neg_log_likelihood(
    raw: np.ndarray,
    X: np.ndarray,
    offset: np.ndarray,
    log_t: np.ndarray,
    event: np.ndarray,
) -> float:
    k = X.shape[1]
    theta = raw[:k]
    log_sigma = raw[k]
    sigma = np.exp(log_sigma)

    mu = offset + X @ theta
    z = (log_t - mu) / sigma

    ll = 0.0
    if np.any(event):
        ll += np.sum(stats.norm.logpdf(z[event]) - log_sigma)
    if np.any(~event):
        ll += np.sum(stats.norm.logsf(z[~event]))
    return -ll


def fit_aft(
    events: Sequence[GrokkingRun],
    censored: Sequence[GrokkingRun] = (),
    design: str = "free",
) -> AFTResult:
    """Fit a log-normal AFT model with right-censoring."""
    if design not in DESIGNS:
        raise ValueError(f"Unknown design '{design}'. Options: {sorted(DESIGNS)}")
    if len(events) < 3:
        raise ValueError("At least three observed (event) runs are required.")

    design_fn, names, offset_fn = DESIGNS[design]

    runs = list(events) + list(censored)
    p = np.array([r.condition.modulus for r in runs], dtype=float)
    wd = np.array([r.condition.weight_decay for r in runs], dtype=float)
    times = np.array(
        [
            float(r.grokking_epoch) if r.grokking_epoch is not None else float(r.max_epochs)
            for r in runs
        ]
    )
    event = np.array([r.grokking_epoch is not None for r in runs], dtype=bool)

    X = design_fn(p, wd)
    offset = offset_fn(p)
    log_t = np.log(times)

    # Prune zero-variance covariates (e.g. a wd column when every run shares
    # one weight decay) so the likelihood stays identifiable. Pruned
    # parameters are reported as 0.0 with a note.
    keep = [0] + [
        j for j in range(1, X.shape[1]) if float(np.ptp(X[:, j])) > 1e-12
    ]
    pruned = [names[j] for j in range(X.shape[1]) if j not in keep]
    X = X[:, keep]
    names_kept = [names[j] for j in keep]

    # OLS-on-events initialization
    theta0, *_ = np.linalg.lstsq(X[event], log_t[event] - offset[event], rcond=None)
    resid = log_t[event] - offset[event] - X[event] @ theta0
    sigma0 = max(float(np.std(resid)), 1e-2)
    raw0 = np.concatenate([theta0, [np.log(sigma0)]])

    res = optimize.minimize(
        _neg_log_likelihood,
        raw0,
        args=(X, offset, log_t, event),
        method="Nelder-Mead",
        options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-10},
    )
    nm_fun = float(res.fun)
    res = optimize.minimize(
        _neg_log_likelihood,
        res.x,
        args=(X, offset, log_t, event),
        method="BFGS",
        options={"maxiter": 5000},
    )

    raw = res.x
    k = X.shape[1]
    sigma = float(np.exp(raw[k]))

    # Numerical Hessian for standard errors.
    std = np.full(k + 1, np.nan)
    try:
        hess = _numerical_hessian(
            lambda v: _neg_log_likelihood(v, X, offset, log_t, event), raw
        )
        cov = np.linalg.inv(hess)
        diag = np.diag(cov)
        std = np.sqrt(np.where(diag > 0, diag, np.nan))
    except np.linalg.LinAlgError:
        pass

    # Map kept coefficients back to the full parameter list; pruned entries
    # are reported as 0.0 with NaN standard errors. Sign conventions (q, beta
    # positive in the tau = C p^a / (log^q p * wd^beta) form) apply to the
    # full-vector positions.
    full_signs = _sign_vector(design, len(names))
    params_full = {n: 0.0 for n in names}
    se_full = {n: float("nan") for n in names}
    for slot, (j, coef, se) in enumerate(zip(keep, raw[:k], std[:k])):
        params_full[names[j]] = float(full_signs[j] * coef)
        se_full[names[j]] = float(se)

    prune_note = (
        f" Pruned zero-variance covariates: {pruned}." if pruned else ""
    )

    return AFTResult(
        design=design,
        param_names=names,
        params=params_full,
        std_errors=se_full,
        sigma=sigma,
        log_likelihood=-float(res.fun),
        n_events=int(np.sum(event)),
        n_censored=int(np.sum(~event)),
        converged=bool(res.success) or float(res.fun) <= nm_fun + 1e-6,
        _raw=raw,
        notes=(
            "Reported q and beta follow the tau = C * p^a / ((log p)^q * wd^beta) "
            "convention; raw regression signs are handled internally." + prune_note
        ),
        _keep=list(keep),
    )


def _numerical_hessian(fn: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-4) -> np.ndarray:
    n = len(x)
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = h
            ej[j] = h
            fpp = fn(x + ei + ej)
            fpm = fn(x + ei - ej)
            fmp = fn(x - ei + ej)
            fmm = fn(x - ei - ej)
            hess[i, j] = hess[j, i] = (fpp - fpm - fmp + fmm) / (4 * h * h)
    return hess


# --------------------------------------------------------------------------
# Censoring impact report
# --------------------------------------------------------------------------


def censoring_impact_report(
    csv_path: Path = DEFAULT_RUN_TABLE,
    design: str = "power_only",
    modulus_range: Tuple[int, int] = (2, 80),
) -> Dict[str, object]:
    """Fit the transition-zone law with and without the censored run(s).

    Returns both fits so the exponent shift attributable to censoring-aware
    treatment is explicit and reportable.
    """
    dataset = ScalingDataset.from_run_table_csv(csv_path)
    events, censored = select_runs(dataset, modulus_range=modulus_range)

    fit_excluding = fit_aft(events, censored=(), design=design)
    fit_including = fit_aft(events, censored=censored, design=design)

    return {
        "design": design,
        "modulus_range": modulus_range,
        "n_events": len(events),
        "n_censored": len(censored),
        "censored_runs": [
            {
                "modulus": r.condition.modulus,
                "budget_epochs": r.max_epochs,
                "interpretation": f"tau > {r.max_epochs}",
            }
            for r in censored
        ],
        "fit_excluding_censored": fit_excluding,
        "fit_including_censored": fit_including,
    }


def format_report(report: Dict[str, object]) -> str:
    """Human-readable summary of a censoring impact report."""
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append(" CENSORING-AWARE SCALING FIT (log-normal AFT / Tobit in log space)")
    lines.append("=" * 78)
    lines.append(
        f" design={report['design']}  modulus_range={report['modulus_range']}  "
        f"events={report['n_events']}  censored={report['n_censored']}"
    )
    for item in report["censored_runs"]:
        lines.append(
            f"   censored: p={item['modulus']}  ({item['interpretation']})"
        )
    for key, label in [
        ("fit_excluding_censored", "EXCLUDING censored (status quo)"),
        ("fit_including_censored", "INCLUDING censored (AFT)"),
    ]:
        fit: AFTResult = report[key]  # type: ignore[assignment]
        lines.append("-" * 78)
        lines.append(f" {label}")
        for name in fit.param_names:
            se = fit.std_errors.get(name, float("nan"))
            lines.append(f"   {name:8s} = {fit.params[name]:8.3f}  (se {se:.3f})")
        lines.append(
            f"   sigma    = {fit.sigma:8.3f}   logL = {fit.log_likelihood:8.3f}"
            f"   converged={fit.converged}"
        )
    lines.append("=" * 78)
    return "\n".join(lines)


def main() -> None:
    report = censoring_impact_report()
    print(format_report(report))


if __name__ == "__main__":
    main()
