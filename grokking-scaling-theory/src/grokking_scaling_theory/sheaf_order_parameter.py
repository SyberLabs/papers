"""
grokking_scaling_theory.sheaf_order_parameter
---------------------------------------------
Categorical order parameters for grokking (Phase 2, Variant A and B).

Pre-registration: experiments/PHASE2_CATEGORICAL_ORDER_PARAMETERS.md.
This module implements the measurement pipeline and the three validation
gates that must pass on synthetic data BEFORE any confirmatory run:

  Gate 1: D_sheaf_B shows a transition on synthetic grokking dynamics,
          and its logistic half-maximum epoch recovers the planted
          transition epoch within 10%.
  Gate 2: label-shuffled evaluation shows no alignment transition
          (intra-class edge fraction stays at chance ~ 1/p).
  Gate 3: global representation collapse is flagged degenerate rather
          than reported as D_sheaf_B -> 1 (no false positive).

Construction (Variant B, label-free):
  At each checkpoint t, build a k-nearest-neighbor graph on the hidden
  representations phi(x, t) (labels touch nothing), with identity
  restriction maps in the prototype (learned Hansen-Ghrist restrictions
  are a later ablation). The Dirichlet energy per edge is contrasted
  against an edge-count-matched random graph on the same points:

      D_sheaf_B(t) = 1 - E_knn(t) / E_rand(t)

  Collapse guard: if total representation variance < tol, the checkpoint
  is flagged degenerate and D is NaN (both energies vanish together, so
  the ratio is uninformative, not evidence of gluing).

Spectral observables (Variant B): with identity restrictions the sheaf
Laplacian factorizes as L_graph (x) I_d, so all spectral content lives in
the scalar graph Laplacian L_graph(t). The pre-registered "gap opens"
event is operationalized as the p-th eigengap of L_graph(t):

      gap_p(t) = lambda_{p+1}(t) - lambda_p(t)

together with the near-kernel dimension (count of eigenvalues below
tol_eig * lambda_max), which approaches p as the representation graph
separates into p rule-class components. See amendment 2 in the
pre-registration for this sharpening.

Variant A (diagnostic only): fixed graph built from class labels at t0
(k random within-class neighbors), identity restrictions, energy of the
evolving section. Honestly characterized as a spectral refinement of the
variance order parameter D_var; it carries no confirmatory weight.

Expected real-trace format (.npz):
  epochs  : int array [T]
  hidden  : float array [T, N, d]   per-example hidden states
  classes : int array [N]           rule class c(x), e.g. (a+b) mod p
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize

# ---------------------------------------------------------------------------
# Trace container
# ---------------------------------------------------------------------------


@dataclass
class TraceArrays:
    """Per-example hidden-state trace for one run."""

    epochs: np.ndarray  # [T]
    hidden: np.ndarray  # [T, N, d]
    classes: np.ndarray  # [N]

    def __post_init__(self) -> None:
        T, N, _ = self.hidden.shape
        if len(self.epochs) != T:
            raise ValueError("epochs length must match hidden's first axis")
        if len(self.classes) != N:
            raise ValueError("classes length must match hidden's second axis")

    @property
    def num_classes(self) -> int:
        return int(len(np.unique(self.classes)))


def load_trace_npz(path: str | Path) -> TraceArrays:
    """Load a per-example trace saved as .npz with keys epochs/hidden/classes.

    Hidden states saved as float16 by the training harness are upcast to
    float64 here, so all downstream PCA/energy math runs in full precision.
    """
    data = np.load(Path(path))
    return TraceArrays(
        epochs=np.asarray(data["epochs"], dtype=int),
        hidden=np.asarray(data["hidden"], dtype=float),
        classes=np.asarray(data["classes"], dtype=int),
    )


def project_pca(
    trace: TraceArrays,
    d: int = 64,
    fit_epoch: Optional[int] = None,
) -> TraceArrays:
    """Project hidden states to ``d`` dims via PCA (PHASE2 4.4 scale control).

    The pre-registration fixes two things that this function honors exactly:

    1. The PCA basis is fit at a *single* reference checkpoint (the grok
       epoch, or the checkpoint nearest to ``fit_epoch``; if None, the last
       checkpoint is used as a proxy for the post-grok representation).
    2. That *fixed* basis is applied to every checkpoint, so cross-time
       comparison of the projected representations stays valid (a per-epoch
       refit would rotate the subspace and make the D_sheaf_B time series
       incomparable across epochs).

    Centering uses the fit-checkpoint mean, also held fixed across time.
    Returns a new TraceArrays with hidden of shape [T, N, d']; d' = min(d,
    original width, N). The original trace is not mutated.
    """
    T, N, width = trace.hidden.shape
    d_eff = int(min(d, width, N))

    if fit_epoch is None:
        fit_idx = T - 1
    else:
        fit_idx = int(np.argmin(np.abs(trace.epochs - fit_epoch)))

    ref = trace.hidden[fit_idx].astype(np.float64)  # [N, width]
    mean = ref.mean(axis=0, keepdims=True)
    # Right singular vectors of the centered reference are the PCA axes.
    _, _, vt = np.linalg.svd(ref - mean, full_matrices=False)
    basis = vt[:d_eff].T  # [width, d_eff], fixed across time

    projected = np.empty((T, N, d_eff), dtype=np.float64)
    for t in range(T):
        projected[t] = (trace.hidden[t].astype(np.float64) - mean) @ basis

    return TraceArrays(
        epochs=trace.epochs.copy(),
        hidden=projected,
        classes=trace.classes.copy(),
    )


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def knn_edges(X: np.ndarray, k: int, chunk: int = 512) -> np.ndarray:
    """Symmetrized k-nearest-neighbor edge list [(u, v), ...] with u < v.

    Brute-force distances in chunks; adequate at pre-registered scale
    (N <= 2000). No labels are used.
    """
    N = X.shape[0]
    if k >= N:
        raise ValueError("k must be smaller than the number of points")
    sq_norms = np.sum(X * X, axis=1)
    edges = set()
    for start in range(0, N, chunk):
        stop = min(start + chunk, N)
        block = X[start:stop]
        d2 = (
            sq_norms[start:stop, None]
            - 2.0 * block @ X.T
            + sq_norms[None, :]
        )
        for row_offset in range(stop - start):
            u = start + row_offset
            d2[row_offset, u] = np.inf
            nbrs = np.argpartition(d2[row_offset], k)[:k]
            for v in nbrs:
                edges.add((min(u, int(v)), max(u, int(v))))
    return np.array(sorted(edges), dtype=int)


def within_class_edges(classes: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Variant A graph: k random same-class neighbors per node, fixed once.

    Uses labels only (no representations), so the Variant A Laplacian is
    constant over training by construction.
    """
    edges = set()
    for c in np.unique(classes):
        members = np.flatnonzero(classes == c)
        if len(members) < 2:
            continue
        kk = min(k, len(members) - 1)
        for u in members:
            others = members[members != u]
            for v in rng.choice(others, size=kk, replace=False):
                edges.add((min(int(u), int(v)), max(int(u), int(v))))
    return np.array(sorted(edges), dtype=int)


def graph_laplacian_dense(edges: np.ndarray, N: int) -> np.ndarray:
    """Dense combinatorial graph Laplacian (prototype scale)."""
    L = np.zeros((N, N))
    for u, v in edges:
        L[u, u] += 1.0
        L[v, v] += 1.0
        L[u, v] -= 1.0
        L[v, u] -= 1.0
    return L


# ---------------------------------------------------------------------------
# Energies and order parameters
# ---------------------------------------------------------------------------

COLLAPSE_TOL = 1e-8


def edge_energy(X: np.ndarray, edges: np.ndarray) -> float:
    """Mean squared representation difference across the given edges."""
    if len(edges) == 0:
        return float("nan")
    diffs = X[edges[:, 0]] - X[edges[:, 1]]
    return float(np.mean(np.sum(diffs * diffs, axis=1)))


def random_edge_energy(
    X: np.ndarray, n_edges: int, rng: np.random.Generator
) -> float:
    """Energy on an edge-count-matched random graph (contrast normalizer)."""
    N = X.shape[0]
    u = rng.integers(0, N, size=n_edges)
    v = rng.integers(0, N, size=n_edges)
    mask = u != v
    diffs = X[u[mask]] - X[v[mask]]
    return float(np.mean(np.sum(diffs * diffs, axis=1)))


@dataclass
class CheckpointResult:
    """All Variant B observables at one checkpoint."""

    epoch: int
    d_sheaf_b: float
    degenerate: bool
    intra_class_fraction: float
    intra_class_fraction_shuffled: float
    near_kernel_dim: Optional[int] = None
    eigengap_p: Optional[float] = None


def variant_b_checkpoint(
    X: np.ndarray,
    classes: np.ndarray,
    epoch: int,
    k: int,
    rng: np.random.Generator,
    shuffled_classes: np.ndarray,
    compute_spectrum: bool = False,
    tol_eig: float = 1e-6,
) -> CheckpointResult:
    """Compute Variant B observables for one checkpoint (labels used only
    for the intra-class evaluation statistics, never in construction)."""
    total_var = float(np.var(X))
    if total_var < COLLAPSE_TOL:
        return CheckpointResult(
            epoch=epoch,
            d_sheaf_b=float("nan"),
            degenerate=True,
            intra_class_fraction=float("nan"),
            intra_class_fraction_shuffled=float("nan"),
        )

    edges = knn_edges(X, k)
    e_knn = edge_energy(X, edges)
    e_rand = random_edge_energy(X, max(len(edges), 1000), rng)
    if e_rand < COLLAPSE_TOL:
        return CheckpointResult(
            epoch=epoch,
            d_sheaf_b=float("nan"),
            degenerate=True,
            intra_class_fraction=float("nan"),
            intra_class_fraction_shuffled=float("nan"),
        )

    d_b = 1.0 - e_knn / e_rand
    intra = float(np.mean(classes[edges[:, 0]] == classes[edges[:, 1]]))
    intra_shuf = float(
        np.mean(shuffled_classes[edges[:, 0]] == shuffled_classes[edges[:, 1]])
    )

    near_kernel = None
    gap_p = None
    if compute_spectrum:
        L = graph_laplacian_dense(edges, X.shape[0])
        evals = np.linalg.eigvalsh(L)
        lam_max = max(float(evals[-1]), 1e-12)
        near_kernel = int(np.sum(evals < tol_eig * lam_max + 1e-12))
        p = len(np.unique(classes))
        if len(evals) > p:
            gap_p = float(evals[p] - evals[p - 1])

    return CheckpointResult(
        epoch=epoch,
        d_sheaf_b=float(d_b),
        degenerate=False,
        intra_class_fraction=intra,
        intra_class_fraction_shuffled=intra_shuf,
        near_kernel_dim=near_kernel,
        eigengap_p=gap_p,
    )


def variant_a_series(trace: TraceArrays, k: int, seed: int = 0) -> np.ndarray:
    """Variant A (diagnostic): energy of the evolving section on a fixed,
    label-built graph. Returns D_A(t) = 1 - E(t)/E(t_0)."""
    rng = np.random.default_rng(seed)
    edges = within_class_edges(trace.classes, k, rng)
    e0 = edge_energy(trace.hidden[0], edges)
    if not np.isfinite(e0) or e0 < COLLAPSE_TOL:
        return np.full(len(trace.epochs), np.nan)
    return np.array(
        [1.0 - edge_energy(trace.hidden[t], edges) / e0 for t in range(len(trace.epochs))]
    )


def compute_order_parameters(
    trace: TraceArrays,
    k: int = 10,
    seed: int = 0,
    compute_spectrum: bool = False,
    spectrum_stride: int = 1,
) -> Dict[str, np.ndarray]:
    """Full Variant B series (plus Variant A diagnostic) for one trace."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(trace.classes)

    results: List[CheckpointResult] = []
    for t in range(len(trace.epochs)):
        spectral = compute_spectrum and (t % spectrum_stride == 0)
        results.append(
            variant_b_checkpoint(
                trace.hidden[t],
                trace.classes,
                int(trace.epochs[t]),
                k,
                rng,
                shuffled,
                compute_spectrum=spectral,
            )
        )

    return {
        "epochs": np.array([r.epoch for r in results]),
        "d_sheaf_b": np.array([r.d_sheaf_b for r in results]),
        "degenerate": np.array([r.degenerate for r in results]),
        "intra_class_fraction": np.array([r.intra_class_fraction for r in results]),
        "intra_class_fraction_shuffled": np.array(
            [r.intra_class_fraction_shuffled for r in results]
        ),
        "near_kernel_dim": np.array(
            [np.nan if r.near_kernel_dim is None else r.near_kernel_dim for r in results]
        ),
        "eigengap_p": np.array(
            [np.nan if r.eigengap_p is None else r.eigengap_p for r in results]
        ),
        "d_variant_a": variant_a_series(trace, k, seed),
    }


# ---------------------------------------------------------------------------
# Pre-registered timing metric
# ---------------------------------------------------------------------------


def _logistic4(t: np.ndarray, lo: float, hi: float, t_half: float, width: float) -> np.ndarray:
    return lo + (hi - lo) / (1.0 + np.exp(-(t - t_half) / max(width, 1e-9)))


def transition_details(
    epochs: np.ndarray, values: np.ndarray, r2_gate: float = 0.8
) -> Tuple[Optional[float], float, float]:
    """Half-maximum epoch of a 4-parameter logistic fit (pre-registered
    timing metric), with fit diagnostics.

    Returns (t_half or None if unclassifiable, R^2, amplitude), where
    amplitude = |hi - lo| of the fitted logistic. Amplitude is reported
    alongside every fit (amendment 3): the R^2 gate alone cannot distinguish
    a genuine transition from a well-fit small-amplitude drift, so amplitude
    must be visible in all reported results. It is NOT used as a gate for
    the original 18-run sweep (that would be post hoc); a pre-registered
    amplitude gate applies only to arms declared after amendment 3.
    """
    mask = np.isfinite(values)
    t = np.asarray(epochs, dtype=float)[mask]
    y = np.asarray(values, dtype=float)[mask]
    if len(t) < 6:
        return None, float("nan"), float("nan")
    span = t[-1] - t[0]
    p0 = [float(np.min(y)), float(np.max(y)), float(np.median(t)), span / 10.0]
    try:
        popt, _ = optimize.curve_fit(_logistic4, t, y, p0=p0, maxfev=20000)
    except RuntimeError:
        return None, float("nan"), float("nan")
    pred = _logistic4(t, *popt)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    amplitude = float(abs(popt[1] - popt[0]))
    if r2 < r2_gate:
        return None, r2, amplitude
    return float(popt[2]), r2, amplitude


def transition_epoch(
    epochs: np.ndarray, values: np.ndarray, r2_gate: float = 0.8
) -> Tuple[Optional[float], float]:
    """Backward-compatible wrapper around ``transition_details`` returning
    (t_half or None if unclassifiable, R^2)."""
    t_half, r2, _ = transition_details(epochs, values, r2_gate=r2_gate)
    return t_half, r2


# ---------------------------------------------------------------------------
# Synthetic validation (the three gates)
# ---------------------------------------------------------------------------


def synthetic_grokking_trace(
    p: int = 29,
    per_class: int = 20,
    d: int = 32,
    T: int = 60,
    epoch_step: int = 20,
    t_star_frac: float = 0.5,
    width_frac: float = 0.08,
    noise: float = 0.05,
    seed: int = 0,
    collapse: bool = False,
) -> Tuple[TraceArrays, float]:
    """Planted-transition trajectory: example-specific representations
    interpolating to rule-class Fourier prototypes on a logistic schedule.

    Returns (trace, planted_transition_epoch). With collapse=True, all
    representations converge to a single point instead (Gate 3 control).
    """
    rng = np.random.default_rng(seed)
    N = p * per_class
    classes = np.repeat(np.arange(p), per_class)
    epochs = np.arange(T) * epoch_step
    t_star = float(epochs[-1]) * t_star_frac
    width = float(epochs[-1]) * width_frac

    mem = rng.normal(size=(N, d))
    mem /= np.linalg.norm(mem, axis=1, keepdims=True)

    proto = np.zeros((p, d))
    for f in range(1, 3):
        proto[:, 2 * (f - 1)] = np.cos(2 * np.pi * f * np.arange(p) / p)
        proto[:, 2 * (f - 1) + 1] = np.sin(2 * np.pi * f * np.arange(p) / p)
    rule = proto[classes]

    hidden = np.zeros((T, N, d))
    for i, ep in enumerate(epochs):
        s = 1.0 / (1.0 + np.exp(-(ep - t_star) / width))
        if collapse:
            target = (1.0 - s) * mem  # everything shrinks to the origin
        else:
            target = (1.0 - s) * mem + s * rule
        hidden[i] = target + noise * rng.normal(size=(N, d))
        if collapse and s > 0.95:
            hidden[i] = 1e-6 * rng.normal(size=(N, d))
    return TraceArrays(epochs=epochs, hidden=hidden, classes=classes), t_star


def run_synthetic_validation(seed: int = 0, verbose: bool = True) -> Dict[str, object]:
    """Run Gates 1-3. Returns a report dict; raises AssertionError on failure."""
    report: Dict[str, object] = {}

    # --- Gate 1 + 2: planted transition ---
    trace, t_star = synthetic_grokking_trace(seed=seed)
    series = compute_order_parameters(
        trace, k=10, seed=seed, compute_spectrum=True, spectrum_stride=5
    )
    t_half, r2 = transition_epoch(series["epochs"], series["d_sheaf_b"])
    assert t_half is not None, "Gate 1 FAIL: logistic fit unclassifiable"
    rel_err = abs(t_half - t_star) / t_star
    assert rel_err <= 0.10, f"Gate 1 FAIL: timing error {rel_err:.1%} > 10%"
    report["gate1"] = {
        "planted_t_star": t_star,
        "recovered_t_half": t_half,
        "relative_error": rel_err,
        "r2": r2,
    }

    p = trace.num_classes
    intra_end = float(series["intra_class_fraction"][-1])
    shuf = series["intra_class_fraction_shuffled"]
    shuf_max = float(np.nanmax(shuf))
    chance = 1.0 / p
    assert intra_end > 0.9, f"Gate 2 FAIL: true intra-class fraction {intra_end:.2f}"
    assert shuf_max < 3 * chance, (
        f"Gate 2 FAIL: shuffled alignment {shuf_max:.3f} exceeds 3x chance {chance:.3f}"
    )
    report["gate2"] = {
        "intra_class_final": intra_end,
        "shuffled_max": shuf_max,
        "chance": chance,
    }

    nk = series["near_kernel_dim"]
    nk_final = int(np.nanmax(nk))
    report["spectral"] = {
        "near_kernel_final": nk_final,
        "num_classes": p,
        "eigengap_p_final": float(np.nanmax(series["eigengap_p"])),
    }

    # --- Gate 3: collapse must be flagged, not celebrated ---
    ctrace, _ = synthetic_grokking_trace(seed=seed, collapse=True)
    cseries = compute_order_parameters(ctrace, k=10, seed=seed)
    late = slice(int(0.8 * len(ctrace.epochs)), None)
    assert bool(np.any(cseries["degenerate"][late])), (
        "Gate 3 FAIL: collapse not flagged degenerate"
    )
    assert not bool(
        np.any(np.nan_to_num(cseries["d_sheaf_b"][late], nan=0.0) > 0.99)
    ), "Gate 3 FAIL: collapse reported as near-perfect gluing"
    report["gate3"] = {
        "degenerate_flagged": True,
        "late_d_values": [
            None if not np.isfinite(v) else round(float(v), 3)
            for v in cseries["d_sheaf_b"][late]
        ],
    }

    if verbose:
        print("=" * 74)
        print(" SHEAF ORDER PARAMETER: SYNTHETIC VALIDATION GATES")
        print("=" * 74)
        g1 = report["gate1"]
        print(
            f" Gate 1 PASS  planted t*={g1['planted_t_star']:.0f}"
            f"  recovered t_half={g1['recovered_t_half']:.0f}"
            f"  err={g1['relative_error']:.1%}  R^2={g1['r2']:.3f}"
        )
        g2 = report["gate2"]
        print(
            f" Gate 2 PASS  intra-class final={g2['intra_class_final']:.3f}"
            f"  shuffled max={g2['shuffled_max']:.3f} (chance={g2['chance']:.3f})"
        )
        sp = report["spectral"]
        print(
            f" Spectral     near-kernel dim -> {sp['near_kernel_final']}"
            f" (p={sp['num_classes']})  eigengap_p={sp['eigengap_p_final']:.3f}"
        )
        print(" Gate 3 PASS  collapse flagged degenerate; no false gluing")
        print("=" * 74)
    return report


if __name__ == "__main__":
    run_synthetic_validation()
