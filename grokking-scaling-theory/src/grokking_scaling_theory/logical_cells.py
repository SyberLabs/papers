"""
grokking_scaling_theory.logical_cells
-------------------------------------
The decidability order parameter D_logic (Phase 2, logical leg).

Ports the *Logical Information Cells* measurement (Belfiore, Bennequin &
Giraud, arXiv 2108.04751) to modular addition, under the operationalization
fixed by PHASE2 amendment 4 (experiments/PHASE2_CATEGORICAL_ORDER_PARAMETERS.md):

- Proposition family (prime p): low-order Fourier sign propositions
  cos(2 pi k c / p) > 0 and sin(2 pi k c / p) > 0 for k = 1..4, plus the
  quadratic-residue proposition c in QR(p) (c = 0 excluded from QR).
- Logical unit: per-checkpoint standardized activity, ternary states with
  bins (-1/3, +1/3); purity >= 0.80 in one outer bin on the P side AND
  >= 0.80 in the opposite outer bin on the not-P side (two-sided
  informativeness, amendment 4 item 3).
- Conclusive group: <= 3 informative units whose majority vote attains
  balanced accuracy >= 0.90 for P vs not-P (greedy: best single, else best
  triple). A proposition is covered iff a conclusive group exists.
- Decidability: an example is decidable iff the intersection of its
  covered-proposition verdicts is the singleton {c(x)}. Singleton-but-wrong
  is NOT decidable.
- Ceiling normalization: D_logic is reported raw and divided by the
  family's theoretical ceiling (fraction of examples whose class signature
  is unique under the full family) so levels are comparable across p.
  Timing uses the normalized series.

Validation gates (amendment 4 item 7) run via ``__main__`` and must pass
before any real trace is analyzed:

  L1: planted decidable code -> timing recovered within 10%, D_logic -> ~1.
  L2: label shuffle -> D_logic ~ 0, no transition.
  L3: Fourier-only planted units -> Fourier props covered, QR uncovered.

Usage
=====
    python -m grokking_scaling_theory.logical_cells      # gates
    from grokking_scaling_theory.logical_cells import d_logic_series
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from grokking_scaling_theory.sheaf_order_parameter import (
    TraceArrays,
    transition_epoch,
)

PURITY = 0.80          # section 2.3, fixed pre-data
BIN_LO, BIN_HI = -1.0 / 3.0, 1.0 / 3.0
BALANCED_ACC_TARGET = 0.90  # amendment 4 item 4 (primary)
MAX_GROUP = 3          # section 4.1


# ---------------------------------------------------------------------------
# Proposition family (amendment 4 item 1) and ceiling (item 2)
# ---------------------------------------------------------------------------


def proposition_family(p: int, k_max: int = 4) -> Dict[str, np.ndarray]:
    """Fixed proposition family: boolean masks over classes 0..p-1."""
    c = np.arange(p)
    props: Dict[str, np.ndarray] = {}
    for k in range(1, k_max + 1):
        props[f"cos{k}+"] = np.cos(2.0 * np.pi * k * c / p) > 0
        props[f"sin{k}+"] = np.sin(2.0 * np.pi * k * c / p) > 0
    qr = np.zeros(p, dtype=bool)
    qr[np.unique((np.arange(1, p) ** 2) % p)] = True
    qr[0] = False  # 0 is neither QR nor QNR
    props["QR"] = qr
    return props


def family_ceiling(props: Dict[str, np.ndarray], classes: np.ndarray) -> float:
    """Fraction of examples whose class has a UNIQUE proposition signature.

    Computed from the family alone (no activations): this is the maximum
    decidable fraction any measurement could achieve, and the normalizer
    for D_logic (amendment 4 item 2).
    """
    sig = np.stack([m for m in props.values()], axis=1)  # [p, n_props]
    # unique signature <=> no other class shares the full row
    _, inverse, counts = np.unique(
        sig, axis=0, return_inverse=True, return_counts=True
    )
    unique_class = counts[inverse] == 1  # [p]
    return float(np.mean(unique_class[classes]))


# ---------------------------------------------------------------------------
# Per-checkpoint measurement
# ---------------------------------------------------------------------------


def unit_states(X: np.ndarray) -> np.ndarray:
    """Ternary unit states in {-1, 0, +1} after per-unit standardization."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    z = (X - mu) / np.where(sd < 1e-12, 1.0, sd)
    states = np.zeros_like(z, dtype=np.int8)
    states[z > BIN_HI] = 1
    states[z < BIN_LO] = -1
    # dead units (sd ~ 0) stay 0 everywhere -> abstain
    states[:, (sd < 1e-12).ravel()] = 0
    return states


@dataclass
class CheckpointLogic:
    """D_logic observables at one checkpoint."""

    epoch: int
    d_logic_raw: float
    d_logic: float  # ceiling-normalized
    covered: Dict[str, bool]
    n_informative_units: int
    # Diagnostic (no threshold involved): the best two-sided purity any
    # (unit, proposition) pair attains, i.e. max over pairs of
    # min(P-side concentration in bin b, not-P-side concentration in -b).
    # Distinguishes "criterion barely missed" (~0.75) from "no logical
    # cells at all" (~0.5) when coverage is zero.
    max_two_sided_purity: float = 0.0


def _informative_units(
    states: np.ndarray, member: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Units informative for one proposition (amendment 4 item 3).

    ``member``: boolean example mask (P side). Returns (unit_indices, signs,
    best_two_sided_purity) where sign b means: state == b asserts P,
    state == -b asserts not-P. Two-sided rule: >= PURITY of P-side states in
    bin b AND >= PURITY of not-P-side states in bin -b. The returned best
    purity is a threshold-free diagnostic (max over units of the two-sided
    minimum), reported even when no unit passes.
    """
    P, nP = states[member], states[~member]
    if len(P) == 0 or len(nP) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), 0.0
    fp_pos = (P == 1).mean(axis=0)
    fp_neg = (P == -1).mean(axis=0)
    fn_pos = (nP == 1).mean(axis=0)
    fn_neg = (nP == -1).mean(axis=0)
    two_sided = np.maximum(
        np.minimum(fp_pos, fn_neg),  # b = +1
        np.minimum(fp_neg, fn_pos),  # b = -1
    )
    plus = (fp_pos >= PURITY) & (fn_neg >= PURITY)   # b = +1
    minus = (fp_neg >= PURITY) & (fn_pos >= PURITY)  # b = -1
    idx = np.flatnonzero(plus | minus)
    signs = np.where(plus[idx], 1, -1)
    return idx, signs, float(two_sided.max()) if two_sided.size else 0.0


def _majority_verdict(
    states: np.ndarray, idx: np.ndarray, signs: np.ndarray
) -> np.ndarray:
    """Per-example verdict of a unit group: +1 asserts P, -1 asserts not-P,
    0 abstains (tie or all-zero states). Vote of non-zero aligned states."""
    aligned = states[:, idx] * signs[None, :]  # +1 votes P, -1 votes not-P
    votes = aligned.sum(axis=1)
    return np.sign(votes).astype(np.int8)


def _balanced_accuracy(verdict: np.ndarray, member: np.ndarray) -> float:
    tp = float(np.mean(verdict[member] == 1)) if member.any() else 0.0
    tn = float(np.mean(verdict[~member] == -1)) if (~member).any() else 0.0
    return 0.5 * (tp + tn)


def conclusive_group(
    states: np.ndarray,
    member: np.ndarray,
    idx: np.ndarray,
    signs: np.ndarray,
    target: float = BALANCED_ACC_TARGET,
    max_group: int = MAX_GROUP,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Greedy conclusive group (amendment 4 item 4): best single informative
    unit, else the top-``max_group`` units by individual balanced accuracy
    as a majority-vote group. None if the target is unattainable."""
    if len(idx) == 0:
        return None
    accs = np.array(
        [
            _balanced_accuracy(
                _majority_verdict(states, idx[j : j + 1], signs[j : j + 1]), member
            )
            for j in range(len(idx))
        ]
    )
    best = int(np.argmax(accs))
    if accs[best] >= target:
        return idx[best : best + 1], signs[best : best + 1]
    order = np.argsort(-accs)[:max_group]
    gi, gs = idx[order], signs[order]
    if _balanced_accuracy(_majority_verdict(states, gi, gs), member) >= target:
        return gi, gs
    return None


def checkpoint_logic(
    X: np.ndarray,
    classes: np.ndarray,
    epoch: int,
    props: Dict[str, np.ndarray],
    ceiling: float,
    target: float = BALANCED_ACC_TARGET,
) -> CheckpointLogic:
    """Full D_logic measurement at one checkpoint."""
    states = unit_states(X)
    N, p = len(classes), len(next(iter(props.values())))

    covered: Dict[str, bool] = {}
    n_info = 0
    max_purity = 0.0
    # candidate class sets per example start as all classes
    candidates = np.ones((N, p), dtype=bool)

    for name, mask in props.items():
        member = mask[classes]  # boolean over examples
        idx, signs, best_purity = _informative_units(states, member)
        max_purity = max(max_purity, best_purity)
        n_info += len(idx)
        group = conclusive_group(states, member, idx, signs, target=target)
        covered[name] = group is not None
        if group is None:
            continue
        verdict = _majority_verdict(states, *group)
        # intersect candidate sets: verdict +1 keeps P-classes, -1 keeps rest
        candidates[verdict == 1] &= mask[None, :]
        candidates[verdict == -1] &= ~mask[None, :]

    n_candidates = candidates.sum(axis=1)
    singleton = n_candidates == 1
    correct = np.zeros(N, dtype=bool)
    if singleton.any():
        picked = np.argmax(candidates[singleton], axis=1)
        correct[singleton] = picked == classes[singleton]
    d_raw = float(np.mean(singleton & correct))
    return CheckpointLogic(
        epoch=epoch,
        d_logic_raw=d_raw,
        d_logic=d_raw / max(ceiling, 1e-12),
        covered=covered,
        n_informative_units=int(n_info),
        max_two_sided_purity=max_purity,
    )


# ---------------------------------------------------------------------------
# Trace-level series
# ---------------------------------------------------------------------------


def d_logic_series(
    trace: TraceArrays,
    k_max: int = 4,
    target: float = BALANCED_ACC_TARGET,
) -> Dict[str, np.ndarray]:
    """D_logic over all checkpoints of one trace (amendment 4 primary
    measurement). Returns epochs, raw and normalized series, ceiling, and
    per-leg coverage counts (Fourier vs QR) for the split report."""
    p = int(trace.classes.max()) + 1
    props = proposition_family(p, k_max=k_max)
    ceiling = family_ceiling(props, trace.classes)

    results = [
        checkpoint_logic(
            trace.hidden[t], trace.classes, int(trace.epochs[t]), props, ceiling,
            target=target,
        )
        for t in range(len(trace.epochs))
    ]
    fourier_names = [n for n in props if n != "QR"]
    return {
        "epochs": np.array([r.epoch for r in results]),
        "d_logic": np.array([r.d_logic for r in results]),
        "d_logic_raw": np.array([r.d_logic_raw for r in results]),
        "ceiling": np.array([ceiling]),
        "fourier_covered": np.array(
            [sum(r.covered[n] for n in fourier_names) for r in results]
        ),
        "qr_covered": np.array([int(r.covered["QR"]) for r in results]),
        "informative_units": np.array([r.n_informative_units for r in results]),
        "max_two_sided_purity": np.array(
            [r.max_two_sided_purity for r in results]
        ),
    }


# ---------------------------------------------------------------------------
# Fourier concentration (amendment 5, D3)
# ---------------------------------------------------------------------------


def fourier_concentration(X: np.ndarray, classes: np.ndarray) -> Dict[str, float]:
    """Class-profile Fourier concentration of a layer (amendment 5, D3).

    For each unit, form the class-conditional mean activation profile
    m(c) = E[phi_a(x) | c(x) = c] and measure how concentrated its non-DC
    Fourier power is: concentration = max_k P_k / sum_k P_k over harmonics
    k >= 1. Returns the mean and 90th percentile over units with
    non-trivial profile variance.

    Known limitation (stated in amendment 5): a sign-quantized sinusoid
    keeps ~0.81 base-harmonic concentration, so this separates
    class-structured from unstructured units, not sine from square. The
    (concentration, purity) pair is the discriminator.
    """
    X = np.asarray(X, dtype=np.float64)
    p = int(classes.max()) + 1
    prof = np.zeros((p, X.shape[1]))
    for c in range(p):
        m = classes == c
        if m.any():
            prof[c] = X[m].mean(axis=0)
    F = np.fft.rfft(prof, axis=0)
    power = np.abs(F[1:]) ** 2  # drop DC
    total = power.sum(axis=0)
    conc = np.where(total > 1e-12, power.max(axis=0) / np.maximum(total, 1e-12), 0.0)
    var_ok = prof.var(axis=0) > 1e-9
    vals = conc[var_ok]
    if len(vals) == 0:
        return {"mean": 0.0, "p90": 0.0, "n_units": 0}
    return {
        "mean": float(vals.mean()),
        "p90": float(np.percentile(vals, 90)),
        "n_units": int(var_ok.sum()),
    }


# ---------------------------------------------------------------------------
# Validation gates (amendment 4 item 7)
# ---------------------------------------------------------------------------


def _planted_logic_trace(
    p: int = 29,
    per_class: int = 12,
    d: int = 64,
    T: int = 50,
    epoch_step: int = 20,
    t_star_frac: float = 0.5,
    ramp_frac: float = 0.06,
    noise: float = 0.15,
    seed: int = 0,
    fourier_only: bool = False,
    units_per_prop: int = 3,
) -> Tuple[TraceArrays, float]:
    """Trace whose units come to encode the proposition family on a planted
    fast-onset schedule: s = 0 before t_star, linear ramp to 1 over
    ramp_frac of the span. Returns (trace, true half-maximum of the
    schedule = t_star + ramp/2).

    Design note: D_logic is a THRESHOLDED detector (units count only once
    purity crosses 0.80), so a slow sigmoid schedule makes it fire at the
    detectability threshold (s ~ 0.1 at these amplitudes), far from the
    schedule midpoint — that is a property of any threshold detector, not a
    timing error. The gate therefore plants a fast onset, and the truth the
    fit must recover is the schedule's own half-maximum. With
    fourier_only=True the QR proposition is never encoded (Gate L3)."""
    rng = np.random.default_rng(seed)
    classes = np.repeat(np.arange(p), per_class)
    N = len(classes)
    epochs = np.arange(T) * epoch_step
    span = float(epochs[-1])
    t_star = span * t_star_frac
    ramp = span * ramp_frac
    t_true = t_star + ramp / 2.0  # half-maximum of the planted schedule

    props = proposition_family(p)
    names = [n for n in props if not (fourier_only and n == "QR")]
    hidden = np.zeros((T, N, d))
    unit = 0
    encodings = []
    for name in names:
        member = props[name][classes].astype(float) * 2.0 - 1.0  # +/-1
        for _ in range(units_per_prop):
            encodings.append((unit % d, member))
            unit += 1
    for i, ep in enumerate(epochs):
        s = float(np.clip((ep - t_star) / max(ramp, 1e-9), 0.0, 1.0))
        hidden[i] = noise * rng.normal(size=(N, d))
        for u, member in encodings:
            hidden[i][:, u] += s * 3.0 * member
    return TraceArrays(epochs=epochs, hidden=hidden, classes=classes), t_true


def run_validation_gates(seed: int = 0, verbose: bool = True) -> Dict[str, object]:
    """Gates L1-L3. Raises AssertionError on failure."""
    report: Dict[str, object] = {}

    # --- L1: planted decidable code ---
    trace, t_star = _planted_logic_trace(seed=seed)
    series = d_logic_series(trace)
    t_half, r2 = transition_epoch(series["epochs"], series["d_logic"])
    assert t_half is not None, "Gate L1 FAIL: unclassifiable"
    rel = abs(t_half - t_star) / t_star
    assert rel <= 0.10, f"Gate L1 FAIL: timing error {rel:.1%}"
    final = float(series["d_logic"][-1])
    assert final >= 0.9, f"Gate L1 FAIL: normalized D_logic ends at {final:.2f}"
    report["L1"] = {"t_star": t_star, "t_half": t_half, "rel_err": rel,
                    "final": final, "ceiling": float(series["ceiling"][0])}

    # --- L2: label shuffle ---
    rng = np.random.default_rng(seed)
    shuffled = TraceArrays(
        epochs=trace.epochs.copy(),
        hidden=trace.hidden.copy(),
        classes=rng.permutation(trace.classes),
    )
    s2 = d_logic_series(shuffled)
    peak = float(np.nanmax(s2["d_logic"]))
    assert peak < 0.05, f"Gate L2 FAIL: shuffled D_logic peaks at {peak:.3f}"
    report["L2"] = {"shuffled_peak": peak}

    # --- L3: Fourier-only encoding -> QR uncovered, split measurable ---
    ftrace, _ = _planted_logic_trace(seed=seed, fourier_only=True)
    s3 = d_logic_series(ftrace)
    assert int(s3["fourier_covered"][-1]) >= 6, (
        f"Gate L3 FAIL: only {s3['fourier_covered'][-1]} Fourier props covered"
    )
    assert int(s3["qr_covered"][-1]) == 0, "Gate L3 FAIL: QR spuriously covered"
    report["L3"] = {
        "fourier_covered_final": int(s3["fourier_covered"][-1]),
        "qr_covered_final": int(s3["qr_covered"][-1]),
    }

    # --- F1 (amendment 5): fourier_concentration sanity ---
    p_f1 = 59
    rng_f1 = np.random.default_rng(seed)
    classes_f1 = np.repeat(np.arange(p_f1), 30)
    n_f1 = len(classes_f1)
    # planted sinusoids: every unit is cos(2 pi k c / p) + small noise
    Xs = np.stack(
        [
            np.cos(2.0 * np.pi * k * classes_f1 / p_f1)
            + 0.02 * rng_f1.normal(size=n_f1)
            for k in range(1, 9)
        ],
        axis=1,
    )
    cs = fourier_concentration(Xs, classes_f1)
    assert cs["mean"] > 0.9, f"Gate F1 FAIL: sinusoid concentration {cs['mean']:.2f}"
    # unstructured random units
    Xr = rng_f1.normal(size=(n_f1, 64))
    cr = fourier_concentration(Xr, classes_f1)
    assert cr["mean"] < 0.2, f"Gate F1 FAIL: random concentration {cr['mean']:.2f}"
    report["F1"] = {"sinusoid_mean": cs["mean"], "random_mean": cr["mean"]}

    if verbose:
        print("=" * 74)
        print(" LOGICAL CELLS (D_logic): VALIDATION GATES (amendment 4)")
        print("=" * 74)
        g = report["L1"]
        print(
            f" Gate L1 PASS  planted t*={g['t_star']:.0f} recovered {g['t_half']:.0f}"
            f" err={g['rel_err']:.1%}  final D_logic={g['final']:.2f}"
            f" (ceiling {g['ceiling']:.2f})"
        )
        print(f" Gate L2 PASS  shuffled peak = {report['L2']['shuffled_peak']:.3f}")
        print(
            f" Gate L3 PASS  fourier covered = {report['L3']['fourier_covered_final']}/8,"
            f" QR covered = {report['L3']['qr_covered_final']} (split measurable)"
        )
        f1 = report["F1"]
        print(
            f" Gate F1 PASS  fourier_concentration: sinusoid {f1['sinusoid_mean']:.2f}"
            f" (> 0.9), random {f1['random_mean']:.2f} (< 0.2)"
        )
        print("=" * 74)
    return report


if __name__ == "__main__":
    run_validation_gates()
