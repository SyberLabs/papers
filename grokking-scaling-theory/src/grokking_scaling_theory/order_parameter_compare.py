"""
grokking_scaling_theory.order_parameter_compare
------------------------------------------------
Phase 2 order-parameter comparison and hypothesis evaluation.

Implements the comparison layer pre-registered in
``experiments/PHASE2_CATEGORICAL_ORDER_PARAMETERS.md`` (section 4.1,
``order_parameter_compare.py``): it aligns the candidate order parameters on
one epoch axis, computes each one's transition epoch via the pre-registered
timing metric, and evaluates the hypotheses under the pre-registered decision
rules -- WITHOUT introducing any new tuning surface.

What is computed here
=====================
- ``D_var(t)``  (PHASE2 2.1, the phase-1 incumbent baseline):
      D_var(t) = 1 - E_c[ Var(phi | c) ] / Var(phi)
  This is the object H0 / H1 are defined *relative to*, so it must be
  measured, not assumed.
- ``D_sheaf_B(t)``  raw and PCA-64 (PHASE2 4.4), from
  ``sheaf_order_parameter.compute_order_parameters`` / ``project_pca``.
- ``lambda spectral event``  (H4): the p-th eigengap series of the graph
  Laplacian, from the sheaf module's ``compute_spectrum`` path.

Timing metric and window (fixed, PHASE2 3)
==========================================
- Transition epoch = half-maximum of a 4-parameter logistic fit
  (``sheaf_order_parameter.transition_epoch``); R^2 < 0.8 => unclassifiable.
- "Times grokking" = transition epoch within W = +/- 10% of the val-accuracy
  grok epoch tau, per run.
- Primary estimand = median |t_transition - tau| / tau per order parameter
  per family, with a seed-level bootstrap 95% CI.

Explicit scope limits (reported, not hidden)
============================================
- ``D_logic`` (the logical/decidability leg, PHASE2 2.3) is NOT implemented in
  this module; it lives in a separate ``logical_cells.py`` that does not yet
  exist. Hypotheses H2 and H3 depend on D_logic and are therefore reported as
  ``pending (D_logic not implemented)``, never adjudicated from partial data.
- tau here is the val-accuracy grok epoch recorded by the training harness
  (``trace_*.json``), i.e. the same tau the pre-registration uses.

Usage
=====
    python -m grokking_scaling_theory.order_parameter_compare \\
        --trace_dir data/phase2_traces \\
        --out analysis/ORDER_PARAMETER_RESULTS.md
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from grokking_scaling_theory.sheaf_order_parameter import (
    TraceArrays,
    load_trace_npz,
    project_pca,
    compute_order_parameters,
    transition_details,
)
from grokking_scaling_theory.logical_cells import d_logic_series, fourier_concentration

W_FRAC = 0.10  # +/- 10% window, PHASE2 section 3 (fixed)
R2_GATE = 0.8  # logistic-fit gate, PHASE2 section 3 (fixed)


# ---------------------------------------------------------------------------
# D_var (phase-1 incumbent baseline, PHASE2 2.1)
# ---------------------------------------------------------------------------


def d_var_series(trace: TraceArrays) -> np.ndarray:
    """Variance order parameter D_var(t) = 1 - E_c[Var(phi|c)] / Var(phi).

    Uses total (trace-of-covariance) variance, i.e. summed over feature dims,
    which is the natural multivariate generalization of the scalar definition
    and is invariant to how the within-class means are distributed.
    """
    T = trace.hidden.shape[0]
    classes = trace.classes
    uniq = np.unique(classes)
    out = np.empty(T)
    for t in range(T):
        X = trace.hidden[t].astype(np.float64)  # [N, d]
        total = float(np.sum(np.var(X, axis=0)))  # tr(Cov) over all examples
        if total < 1e-12:
            out[t] = np.nan
            continue
        within = 0.0
        for c in uniq:
            Xc = X[classes == c]
            if len(Xc) < 2:
                continue
            within += float(np.sum(np.var(Xc, axis=0))) * (len(Xc) / len(classes))
        out[t] = 1.0 - within / total
    return out


# ---------------------------------------------------------------------------
# Per-run evaluation
# ---------------------------------------------------------------------------


@dataclass
class OrderParamTiming:
    name: str
    t_transition: Optional[float]
    r2: float
    amplitude: float  # |hi - lo| of the fitted logistic (amendment 3: always reported)
    within_W: Optional[bool]  # None if unclassifiable
    rel_error: Optional[float]  # |t - tau| / tau, None if unclassifiable
    gated: bool = False  # True if degenerate-gated by amendment 4 item 6


@dataclass
class RunEvaluation:
    arch: str
    modulus: int
    seed: int
    tau: int
    timings: Dict[str, OrderParamTiming]
    # Amendment 3, item 1: tau re-derived at the 95% threshold from the
    # stored val-accuracy series (None if series absent or never reaches 95).
    tau95: Optional[int] = None
    # Amendment 4: D_logic family ceiling and final proposition coverage
    # (Fourier legs out of 8, QR out of 1) for the Fourier-vs-logical split.
    d_logic_ceiling: float = float("nan")
    fourier_covered_final: int = -1
    qr_covered_final: int = -1
    max_purity_final: float = float("nan")
    # Amendment 5: canonical-layer fourier concentration (all traces, for
    # D3), any-checkpoint coverage flag, and per-aux-layer diagnostics for
    # multi-layer (depth-arm) traces.
    fourier_conc_mean_final: float = float("nan")
    covered_any_canonical: bool = False
    aux_layers: Dict[str, Dict[str, object]] = field(default_factory=dict)


def _timing(name: str, epochs: np.ndarray, values: np.ndarray, tau: int) -> OrderParamTiming:
    th, r2, amp = transition_details(epochs, values, r2_gate=R2_GATE)
    if th is None:
        return OrderParamTiming(name, None, r2, amp, None, None)
    rel = abs(th - tau) / tau
    return OrderParamTiming(name, float(th), r2, amp, bool(rel <= W_FRAC), float(rel))


def _timing_gated(
    name: str, epochs: np.ndarray, values: np.ndarray, tau: int
) -> OrderParamTiming:
    """Timing with the amendment-4 item-6 fit sanity gate, applied to fits
    from that amendment onward (every D_logic fit). A fit is
    degenerate-unclassifiable if amplitude > 1.5x the observed series range,
    amplitude < 0.05, or t_half outside the observed epoch span. The gate is
    NOT applied to the four pre-amendment order parameters, whose published
    verdicts would otherwise change retroactively."""
    th, r2, amp = transition_details(epochs, values, r2_gate=R2_GATE)
    if th is None:
        return OrderParamTiming(name, None, r2, amp, None, None)
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    span_ok = float(np.min(epochs)) <= th <= float(np.max(epochs))
    obs_range = float(np.max(finite) - np.min(finite)) if len(finite) else float("nan")
    degenerate = (
        not span_ok
        or not np.isfinite(amp)
        or amp < 0.05
        or (np.isfinite(obs_range) and amp > 1.5 * obs_range)
    )
    if degenerate:
        return OrderParamTiming(name, None, r2, amp, None, None, gated=True)
    rel = abs(th - tau) / tau
    return OrderParamTiming(name, float(th), r2, amp, bool(rel <= W_FRAC), float(rel))


def _aux_diagnostics(
    npz_path: Path, trace: TraceArrays, tau: Optional[int]
) -> Dict[str, Dict[str, object]]:
    """Per-aux-layer cell diagnostics for multi-layer (depth-arm) traces.

    Timing is computed only when tau exists; censored runs report
    dlogic_status = 'no-tau' but keep the structural diagnostics
    (amendment 5 item 2)."""
    aux_layers: Dict[str, Dict[str, object]] = {}
    raw = np.load(npz_path)
    if "epochs_aux" not in raw:
        return aux_layers
    epochs_aux = np.asarray(raw["epochs_aux"], dtype=int)
    aux_keys = sorted(
        (k for k in raw.files if k.startswith("hidden_l")),
        key=lambda k: int(k.rsplit("hidden_l", 1)[1]),
    )
    for key in aux_keys:
        aux_trace = TraceArrays(
            epochs=epochs_aux,
            hidden=np.asarray(raw[key], dtype=float),
            classes=trace.classes.copy(),
        )
        als = d_logic_series(aux_trace)
        if tau is not None:
            atiming = _timing_gated(
                f"D_logic_{key}", als["epochs"], als["d_logic"], tau
            )
            status = (
                "classified" if atiming.t_transition is not None
                else ("degen-gated" if atiming.gated else "unclassif")
            )
        else:
            status = "no-tau"
        afc = fourier_concentration(aux_trace.hidden[-1], trace.classes)
        aux_layers[key] = {
            "max_purity_final": float(als["max_two_sided_purity"][-1]),
            "covered_final": int(als["fourier_covered"][-1] + als["qr_covered"][-1]),
            "covered_any": bool(
                np.any(als["fourier_covered"] + als["qr_covered"] > 0)
            ),
            "dlogic_status": status,
            "fourier_conc_mean": float(afc["mean"]),
        }
    return aux_layers


def _tau_at_threshold(npz_path: Path, threshold: float) -> Optional[int]:
    """First checkpoint epoch with val accuracy >= threshold, from the stored
    series (amendment 3, item 2). None if the series is absent or the
    threshold is never reached."""
    data = np.load(npz_path)
    if "val_acc" not in data:
        return None
    val = np.asarray(data["val_acc"], dtype=float)
    epochs = np.asarray(data["epochs"], dtype=int)
    hits = np.flatnonzero(val >= threshold)
    return int(epochs[hits[0]]) if len(hits) else None


def evaluate_trace(
    npz_path: Path, k: int = 10, seed: int = 0, spectrum_stride: int = 5
) -> RunEvaluation:
    """Compute D_var, D_sheaf_B (raw + PCA-64) and the spectral event for one
    trace, and time each against tau."""
    meta = json.loads(npz_path.with_suffix(".json").read_text())
    cfg, res = meta["config"], meta["result"]
    tau = res["grokking_epoch"]
    arch, p, run_seed = cfg["arch"], cfg["modulus"], cfg["seed"]
    tau95 = _tau_at_threshold(npz_path, 95.0)

    trace = load_trace_npz(npz_path)
    timings: Dict[str, OrderParamTiming] = {}

    # Cell-formation diagnostics are computed for ALL runs, censored
    # included (amendment 5 item 2: cells may form without deployment).
    ls = d_logic_series(trace)
    fc = fourier_concentration(trace.hidden[-1], trace.classes)
    covered_any = bool(np.any(ls["fourier_covered"] + ls["qr_covered"] > 0))
    aux_layers = _aux_diagnostics(npz_path, trace, tau)

    if tau is None:
        # Censored run: no grok epoch to time against; timings stay empty
        # but the structural diagnostics above are reported.
        return RunEvaluation(
            arch, p, run_seed, tau=-1, timings=timings,
            d_logic_ceiling=float(ls["ceiling"][0]),
            fourier_covered_final=int(ls["fourier_covered"][-1]),
            qr_covered_final=int(ls["qr_covered"][-1]),
            max_purity_final=float(ls["max_two_sided_purity"][-1]),
            fourier_conc_mean_final=float(fc["mean"]),
            covered_any_canonical=covered_any,
            aux_layers=aux_layers,
        )

    # D_var baseline
    dvar = d_var_series(trace)
    timings["D_var"] = _timing("D_var", trace.epochs, dvar, tau)

    # D_sheaf_B raw
    s_raw = compute_order_parameters(
        trace, k=k, seed=seed, compute_spectrum=True, spectrum_stride=spectrum_stride
    )
    timings["D_sheaf_B_raw"] = _timing(
        "D_sheaf_B_raw", s_raw["epochs"], s_raw["d_sheaf_b"], tau
    )
    # H4 spectral event: eigengap_p opening (use as a rising series toward its
    # own transition; where absent, unclassifiable).
    timings["lambda_gap_raw"] = _timing(
        "lambda_gap_raw", s_raw["epochs"], s_raw["eigengap_p"], tau
    )

    # D_sheaf_B PCA-64 (basis fit at grok epoch, PHASE2 4.4)
    trace_p = project_pca(trace, d=64, fit_epoch=tau)
    s_pca = compute_order_parameters(trace_p, k=k, seed=seed)
    timings["D_sheaf_B_pca64"] = _timing(
        "D_sheaf_B_pca64", s_pca["epochs"], s_pca["d_sheaf_b"], tau
    )

    # D_logic (amendment 4): ceiling-normalized decidability, sanity-gated fit.
    timings["D_logic"] = _timing_gated("D_logic", ls["epochs"], ls["d_logic"], tau)

    return RunEvaluation(
        arch, p, run_seed, tau=tau, timings=timings, tau95=tau95,
        d_logic_ceiling=float(ls["ceiling"][0]),
        fourier_covered_final=int(ls["fourier_covered"][-1]),
        qr_covered_final=int(ls["qr_covered"][-1]),
        max_purity_final=float(ls["max_two_sided_purity"][-1]),
        fourier_conc_mean_final=float(fc["mean"]),
        covered_any_canonical=covered_any,
        aux_layers=aux_layers,
    )


# ---------------------------------------------------------------------------
# Family aggregation + bootstrap
# ---------------------------------------------------------------------------


def _bootstrap_median_diff(
    a: List[float], b: List[float], n_boot: int = 5000, seed: int = 0
) -> Tuple[float, float, float]:
    """(median(a) - median(b), bootstrap 95% CI of that difference)."""
    rng = np.random.default_rng(seed)
    da, db = np.array(a, dtype=float), np.array(b, dtype=float)
    diffs = [
        float(
            np.median(rng.choice(da, len(da), replace=True))
            - np.median(rng.choice(db, len(db), replace=True))
        )
        for _ in range(n_boot)
    ]
    return (
        float(np.median(da) - np.median(db)),
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
    )


def _bootstrap_median_ci(
    values: List[float], n_boot: int = 5000, seed: int = 0
) -> Tuple[float, float, float]:
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    meds = [
        float(np.median(rng.choice(arr, size=len(arr), replace=True)))
        for _ in range(n_boot)
    ]
    return (float(np.median(arr)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))


@dataclass
class FamilySummary:
    arch: str
    op_name: str
    n_total: int
    n_classified: int
    n_within_W: int
    median_rel_error: float
    ci_lo: float
    ci_hi: float


def summarize(
    evaluations: List[RunEvaluation],
    tau_of=None,
) -> List[FamilySummary]:
    """Family-level summaries of the primary estimand.

    ``tau_of`` selects the grok epoch per run: default is the primary tau
    (90% threshold). Passing ``lambda e: e.tau95`` produces the pre-declared
    95% sensitivity pass -- relative errors are recomputed from each order
    parameter's (threshold-independent) transition epoch, so the same
    logistic fits serve both thresholds.
    """
    if tau_of is None:
        tau_of = lambda e: e.tau  # noqa: E731
    op_names = ["D_var", "D_sheaf_B_raw", "D_sheaf_B_pca64", "lambda_gap_raw", "D_logic"]
    archs = sorted({e.arch for e in evaluations if e.tau != -1})
    summaries: List[FamilySummary] = []
    for arch in archs:
        fam = [e for e in evaluations if e.arch == arch and e.tau != -1]
        for op in op_names:
            rels: List[float] = []
            within = 0
            for e in fam:
                tm = e.timings.get(op)
                tau = tau_of(e)
                if tm is None or tm.t_transition is None or tau is None or tau <= 0:
                    continue
                rel = abs(tm.t_transition - tau) / tau
                rels.append(rel)
                within += int(rel <= W_FRAC)
            med, lo, hi = _bootstrap_median_ci(rels)
            summaries.append(
                FamilySummary(
                    arch=arch,
                    op_name=op,
                    n_total=len(fam),
                    n_classified=len(rels),
                    n_within_W=within,
                    median_rel_error=med,
                    ci_lo=lo,
                    ci_hi=hi,
                )
            )
    return summaries


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_markdown(
    evaluations: List[RunEvaluation], summaries: List[FamilySummary]
) -> str:
    lines: List[str] = []
    lines.append("# Phase 2 Order-Parameter Results")
    lines.append("")
    lines.append(
        "Generated by `grokking_scaling_theory.order_parameter_compare`. "
        "Decision rules and thresholds are fixed by "
        "`experiments/PHASE2_CATEGORICAL_ORDER_PARAMETERS.md` section 3 "
        f"(W = +/- {int(W_FRAC*100)}% of tau; logistic R^2 gate = {R2_GATE}). "
        "No thresholds were chosen after seeing these data."
    )
    lines.append("")

    censored = [e for e in evaluations if e.tau == -1]
    lines.append(
        f"Runs analyzed: {len(evaluations)} "
        f"({len(evaluations) - len(censored)} grokked, {len(censored)} censored)."
    )
    lines.append("")

    # --- per-family summary ---
    lines.append("## Family summary (primary estimand)")
    lines.append("")
    lines.append(
        "Median relative timing error `|t_transition - tau| / tau` per order "
        "parameter per family, with seed-level bootstrap 95% CI. "
        "`within_W` counts runs whose transition lands inside the +/-10% window."
    )
    lines.append("")
    lines.append("| arch | order parameter | classified/total | within_W | median rel.err | 95% CI |")
    lines.append("|---|---|---:|---:|---:|---|")
    for s in summaries:
        med = "n/a" if s.median_rel_error != s.median_rel_error else f"{s.median_rel_error:.3f}"
        ci = (
            "n/a"
            if s.ci_lo != s.ci_lo
            else f"[{s.ci_lo:.3f}, {s.ci_hi:.3f}]"
        )
        lines.append(
            f"| {s.arch} | {s.op_name} | {s.n_classified}/{s.n_total} "
            f"| {s.n_within_W}/{s.n_total} | {med} | {ci} |"
        )
    lines.append("")

    # --- 95% sensitivity pass (amendment 3, item 1) ---
    if any(e.tau95 is not None for e in evaluations):
        lines.append("## Threshold sensitivity (tau at 95% val accuracy)")
        lines.append("")
        lines.append(
            "Same transition epochs, re-timed against tau_95 derived from the "
            "stored val-accuracy series. Pre-declared in amendment 3; verdicts "
            "are rendered by the primary (90%) pass above."
        )
        lines.append("")
        lines.append("| arch | order parameter | classified/total | within_W | median rel.err | 95% CI |")
        lines.append("|---|---|---:|---:|---:|---|")
        for s in summarize(evaluations, tau_of=lambda e: e.tau95):
            med = "n/a" if s.median_rel_error != s.median_rel_error else f"{s.median_rel_error:.3f}"
            ci = "n/a" if s.ci_lo != s.ci_lo else f"[{s.ci_lo:.3f}, {s.ci_hi:.3f}]"
            lines.append(
                f"| {s.arch} | {s.op_name} | {s.n_classified}/{s.n_total} "
                f"| {s.n_within_W}/{s.n_total} | {med} | {ci} |"
            )
        lines.append("")

    # --- per-run detail ---
    lines.append("## Per-run detail")
    lines.append("")
    lines.append("| arch | p | seed | tau | order parameter | t_transition | R^2 | amplitude | within W? | rel.err |")
    lines.append("|---|---:|---:|---:|---|---:|---:|---:|:---:|---:|")
    for e in sorted(evaluations, key=lambda r: (r.arch, r.modulus, r.seed)):
        if e.tau == -1:
            lines.append(f"| {e.arch} | {e.modulus} | {e.seed} | CENSORED | - | - | - | - | - | - |")
            continue
        for op, tm in e.timings.items():
            if tm.t_transition is None:
                tt = "degen-gated" if tm.gated else "unclassif"
            else:
                tt = f"{tm.t_transition:.0f}"
            amp = "-" if tm.amplitude != tm.amplitude else f"{tm.amplitude:.3f}"
            wW = "-" if tm.within_W is None else ("yes" if tm.within_W else "no")
            re_ = "-" if tm.rel_error is None else f"{tm.rel_error:.3f}"
            lines.append(
                f"| {e.arch} | {e.modulus} | {e.seed} | {e.tau} | {op} "
                f"| {tt} | {tm.r2:.3f} | {amp} | {wW} | {re_} |"
            )
    lines.append("")

    # --- D_logic coverage split (amendment 4: Fourier-vs-logical) ---
    if any(e.fourier_covered_final >= 0 for e in evaluations):
        lines.append("## D_logic proposition coverage (Fourier-vs-logical split)")
        lines.append("")
        lines.append(
            "Final-checkpoint conclusive-group coverage per family: Fourier "
            "sign propositions (of 8) vs the quadratic-residue proposition "
            "(of 1), with the family decidability ceiling."
        )
        lines.append("")
        lines.append("| arch | mean Fourier covered /8 | mean QR covered /1 | mean ceiling | mean max 2-sided purity (final) |")
        lines.append("|---|---:|---:|---:|---:|")
        for arch in sorted({e.arch for e in evaluations if e.tau != -1}):
            fam = [e for e in evaluations if e.arch == arch and e.fourier_covered_final >= 0]
            if not fam:
                continue
            fc = float(np.mean([e.fourier_covered_final for e in fam]))
            qc = float(np.mean([e.qr_covered_final for e in fam]))
            ce = float(np.mean([e.d_logic_ceiling for e in fam]))
            mp = float(np.mean([e.max_purity_final for e in fam]))
            lines.append(f"| {arch} | {fc:.1f} | {qc:.1f} | {ce:.2f} | {mp:.2f} |")
        lines.append("")
        lines.append(
            "The purity column is the threshold-free diagnostic: how close "
            "the best (unit, proposition) pair comes to the 0.80 logical-cell "
            "criterion. ~0.5 means no unit is anywhere near a logical cell; "
            "~0.75 means the criterion is barely missed."
        )
        lines.append("")

    # --- Depth arm (amendment 5) ---
    deep = [e for e in evaluations if e.aux_layers]
    if deep:
        lines.append("## Depth arm (amendment 5)")
        lines.append("")
        lines.append(
            "Per-run, per-layer cell diagnostics for the multi-layer family. "
            "`purity` = max two-sided purity at the final checkpoint; "
            "`cov` = propositions covered (final / any checkpoint); "
            "`conc` = mean class-profile Fourier concentration."
        )
        lines.append("")
        lines.append("| p | seed | tau | layer | purity | cov final/any | conc | D_logic |")
        lines.append("|---:|---:|---:|---|---:|:---:|---:|---|")
        for e in sorted(deep, key=lambda r: (r.modulus, r.seed)):
            tau_s = "CENSORED" if e.tau == -1 else str(e.tau)
            rows = list(e.aux_layers.items()) + [
                (
                    "final",
                    {
                        "max_purity_final": e.max_purity_final,
                        "covered_final": e.fourier_covered_final + e.qr_covered_final,
                        "covered_any": e.covered_any_canonical,
                        "dlogic_status": (
                            "no-tau" if e.tau == -1
                            else "classified"
                            if "D_logic" in e.timings
                            and e.timings["D_logic"].t_transition is not None
                            else (
                                "degen-gated"
                                if "D_logic" in e.timings and e.timings["D_logic"].gated
                                else "unclassif"
                            )
                        ),
                        "fourier_conc_mean": e.fourier_conc_mean_final,
                    },
                )
            ]
            for lname, d in rows:
                lines.append(
                    f"| {e.modulus} | {e.seed} | {tau_s} | {lname} "
                    f"| {d['max_purity_final']:.2f} "
                    f"| {d['covered_final']}/{'y' if d['covered_any'] else 'n'} "
                    f"| {d['fourier_conc_mean']:.2f} | {d['dlogic_status']} |"
                )
        lines.append("")

        # D1: runs with >= 1 covered proposition in ANY layer at any checkpoint.
        d1_runs = sum(
            1
            for e in deep
            if e.covered_any_canonical
            or any(d["covered_any"] for d in e.aux_layers.values())
        )
        d1 = d1_runs >= 5
        lines.append(
            f"- **D1 (cell formation)**: {d1_runs}/{len(deep)} runs with >= 1 "
            f"covered proposition in any layer => "
            f"{'POSITIVE' if d1 else 'NEGATIVE'} (threshold 5/9)."
        )
        # D2: only meaningful if coverage exists.
        if d1_runs > 0:
            d2_ok = sum(
                1
                for e in deep
                if (
                    e.covered_any_canonical
                    or any(d["covered_any"] for d in e.aux_layers.values())
                )
                and (
                    ("D_logic" in e.timings and e.timings["D_logic"].t_transition is not None)
                    or any(
                        d["dlogic_status"] == "classified"
                        for d in e.aux_layers.values()
                    )
                )
            )
            lines.append(
                f"- **D2 (D_logic functions)**: {d2_ok}/{d1_runs} covering runs "
                f"with a classifiable D_logic."
            )
        else:
            lines.append(
                "- **D2 (D_logic functions)**: moot (no coverage anywhere; D1 "
                "negative)."
            )
        # D3: final-layer concentration, deep vs pooled shallow. Primary
        # comparison uses GROKKED deep runs only: an un-grokked network has
        # no solution, and its (low) concentration would fake "vanishing".
        # All-deep is reported as secondary.
        shallow_conc = [
            e.fourier_conc_mean_final for e in evaluations
            if not e.aux_layers and e.tau != -1
            and np.isfinite(e.fourier_conc_mean_final)
        ]
        deep_grokked = [
            e.fourier_conc_mean_final for e in deep
            if e.tau != -1 and np.isfinite(e.fourier_conc_mean_final)
        ]
        deep_all = [
            e.fourier_conc_mean_final for e in deep
            if np.isfinite(e.fourier_conc_mean_final)
        ]
        if deep_grokked and shallow_conc:
            med, lo, hi = _bootstrap_median_diff(deep_grokked, shallow_conc)
            d3 = hi < 0
            lines.append(
                f"- **D3 (Fourier vanishing, grokked deep runs only, "
                f"n={len(deep_grokked)})**: final-layer concentration, deep "
                f"median {np.median(deep_grokked):.2f} vs shallow pooled "
                f"median {np.median(shallow_conc):.2f}; difference {med:+.2f} "
                f"[{lo:+.2f}, {hi:+.2f}] => "
                f"{'SUPPORTED' if d3 else 'not supported'} (CI must exclude 0 "
                f"from above)."
            )
        elif shallow_conc:
            lines.append(
                "- **D3 (Fourier vanishing)**: unresolvable — no grokked deep "
                "runs to compare (all censored)."
            )
        if deep_all and shallow_conc and len(deep_all) != len(deep_grokked):
            med, lo, hi = _bootstrap_median_diff(deep_all, shallow_conc)
            lines.append(
                f"  - secondary (all deep runs incl. censored, "
                f"n={len(deep_all)}): difference {med:+.2f} [{lo:+.2f}, {hi:+.2f}]."
            )
        lines.append(
            "- **D4 (cascade at depth)**: see the family summary and per-run "
            "tables (the `deep` family rows), reported without a support "
            "threshold."
        )
        lines.append("")

    # --- hypothesis read-out (mechanical, per the fixed rules) ---
    lines.append("## Hypotheses (per pre-registered decision rules)")
    lines.append("")
    lines.append(_hypothesis_readout(summaries, evaluations))
    lines.append("")
    lines.append("## Scope limits")
    lines.append("")
    lines.append(
        "- **D_logic** is computed under the amendment-4 operationalization "
        "(Fourier-sign k=1..4 + QR proposition family, two-sided "
        "informativeness at purity 0.80, conclusive groups <= 3 units at "
        "balanced accuracy >= 0.90, ceiling normalization; validation gates "
        "L1-L3 passed pre-data). Its timing fits carry the amendment-4 "
        "sanity gate; the four pre-amendment order parameters do not "
        "(non-retroactivity).\n"
        "- **H4** is evaluated via the p-th eigengap series (`lambda_gap_raw`); "
        "a flat or non-transitioning gap is reported as unclassifiable, not as "
        "evidence against H4.\n"
        "- Single wd (=1.0), 3 seeds/cell; this is a mechanism-isolation study, "
        "not a universality proof, exactly as PHASE2 5 scopes it."
    )
    return "\n".join(lines)


def _family(summaries: List[FamilySummary], arch: str, op: str) -> Optional[FamilySummary]:
    for s in summaries:
        if s.arch == arch and s.op_name == op:
            return s
    return None


def _paired_timing(
    evaluations: List[RunEvaluation], arch: str, op_a: str, op_b: str
) -> Tuple[List[float], List[float]]:
    """(signed diffs t_a - t_b, |diff|/tau) over runs where both classify."""
    diffs: List[float] = []
    rels: List[float] = []
    for e in evaluations:
        if e.arch != arch or e.tau == -1:
            continue
        ta = e.timings.get(op_a)
        tb = e.timings.get(op_b)
        if (
            ta is not None and tb is not None
            and ta.t_transition is not None and tb.t_transition is not None
        ):
            d = ta.t_transition - tb.t_transition
            diffs.append(d)
            rels.append(abs(d) / e.tau)
    return diffs, rels


def _hypothesis_readout(
    summaries: List[FamilySummary], evaluations: List[RunEvaluation]
) -> str:
    archs = sorted({s.arch for s in summaries})
    out: List[str] = []

    # H1: D_sheaf_B times grokking in BOTH families. The REGISTERED family
    # rule (PHASE2 section 3) is: median timing error within W. That rule and
    # only that rule adjudicates H1. An earlier draft of this module also
    # required a majority of runs individually within W; that was an
    # unregistered strengthening and is now reported separately as a
    # secondary robustness check, never as the verdict.
    def times_grokking(arch: str, op: str) -> Optional[bool]:
        s = _family(summaries, arch, op)
        if s is None or s.n_total == 0:
            return None
        if s.n_classified == 0:
            return False
        return s.median_rel_error <= W_FRAC  # registered rule: median only

    def majority_within(arch: str, op: str) -> Optional[bool]:
        # Secondary (unregistered, stricter) check: majority of ALL runs in
        # the family land individually within W. Reported for transparency.
        s = _family(summaries, arch, op)
        if s is None or s.n_total == 0:
            return None
        return s.n_within_W > s.n_total / 2

    if {"mlp", "residual"}.issubset(set(archs)):
        for op in ["D_sheaf_B_raw", "D_sheaf_B_pca64"]:
            mlp = times_grokking("mlp", op)
            res = times_grokking("residual", op)
            dvar_mlp = times_grokking("mlp", "D_var")
            dvar_res = times_grokking("residual", "D_var")
            verdict = (
                "H1 SUPPORTED" if (mlp and res)
                else "H1 not supported (this order parameter)"
            )
            out.append(
                f"- **{op}** (registered rule: median rel.err <= {W_FRAC}): "
                f"MLP={mlp}, residual={res}. D_var -- MLP={dvar_mlp}, "
                f"residual={dvar_res}.  => {verdict}."
            )
            out.append(
                f"  - secondary (unregistered, stricter) majority-within-W: "
                f"MLP={majority_within('mlp', op)}, "
                f"residual={majority_within('residual', op)}."
            )
    else:
        out.append(
            "- H1 requires both MLP and residual families; not both present."
        )

    # H2/H3 (amendment 4, item 5) — adjudicated now that D_logic exists.
    has_dlogic = any("D_logic" in e.timings for e in evaluations)
    if not has_dlogic:
        out.append("- **H2, H3: pending** (D_logic not implemented).")
    else:
        # H2 leg 1: registered family rule on D_logic.
        leg1 = {a: times_grokking(a, "D_logic") for a in archs}
        leg1_str = ", ".join(f"{a}={v}" for a, v in leg1.items())
        # H2 leg 2: residual paired diffs t_Dlogic - t_Dsheaf_raw, median > 0
        # with seed-level bootstrap 95% CI excluding 0.
        diffs, _ = _paired_timing(evaluations, "residual", "D_logic", "D_sheaf_B_raw")
        if diffs:
            med, lo, hi = _bootstrap_median_ci(diffs)
            leg2 = (med > 0) and (lo > 0)
            out.append(
                f"- **H2 leg 1** (D_logic times grokking, registered family "
                f"rule): {leg1_str}."
            )
            out.append(
                f"- **H2 leg 2** (residual: D_logic later than D_sheaf_B_raw): "
                f"n={len(diffs)} paired runs, median diff = {med:+.0f} epochs, "
                f"bootstrap 95% CI [{lo:+.0f}, {hi:+.0f}] => "
                f"{'SUPPORTED' if leg2 else 'not supported'}."
            )
            h2 = bool(leg1.get("mlp")) and bool(leg1.get("residual")) and leg2
            out.append(f"- **H2 verdict**: {'SUPPORTED' if h2 else 'NOT SUPPORTED'}.")
        else:
            out.append(
                f"- **H2 leg 1**: {leg1_str}. **H2 leg 2**: no residual run "
                f"has both D_logic and D_sheaf_B_raw classifiable => H2 "
                f"unresolvable on these data (reported, not adjudicated)."
            )
        # H3: median |t_Dlogic - t_Dsheaf_raw| / tau <= W per family.
        for a in archs:
            _, rels = _paired_timing(evaluations, a, "D_logic", "D_sheaf_B_raw")
            if rels:
                med, lo, hi = _bootstrap_median_ci(rels)
                out.append(
                    f"- **H3 ({a})**: n={len(rels)} paired, median "
                    f"|t_Dlogic - t_Dsheaf|/tau = {med:.3f} "
                    f"[{lo:.3f}, {hi:.3f}] => "
                    f"{'agree (within W)' if med <= W_FRAC else 'disagree'}."
                )
            else:
                out.append(
                    f"- **H3 ({a})**: no runs with both legs classifiable."
                )
        # Amendment-4 gate accounting for D_logic fits.
        n_gated = sum(
            1 for e in evaluations
            if "D_logic" in e.timings and e.timings["D_logic"].gated
        )
        out.append(
            f"- D_logic fit sanity gate (amendment 4 item 6): "
            f"{n_gated} fit(s) degenerate-gated."
        )
    # H4
    for arch in archs:
        s = _family(summaries, arch, "lambda_gap_raw")
        if s and s.n_classified > 0:
            out.append(
                f"- **H4 ({arch})**: spectral eigengap classified in "
                f"{s.n_classified}/{s.n_total} runs, "
                f"{s.n_within_W}/{s.n_total} within W "
                f"(median rel.err {s.median_rel_error:.3f})."
            )
        else:
            out.append(
                f"- **H4 ({arch})**: spectral eigengap unclassifiable on all "
                f"runs (no clean gap-opening transition); not evidence against "
                f"H4, reported as null."
            )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(
    trace_dir: Path,
    out_path: Optional[Path] = None,
    spectrum_stride: int = 5,
) -> str:
    npzs = sorted(trace_dir.glob("trace_*.npz"))
    if not npzs:
        raise SystemExit(f"No trace_*.npz found in {trace_dir}")
    evaluations = []
    for i, p in enumerate(npzs, 1):
        print(f"[{i}/{len(npzs)}] evaluating {p.name}", flush=True)
        evaluations.append(evaluate_trace(p, spectrum_stride=spectrum_stride))
    summaries = summarize(evaluations)
    md = render_markdown(evaluations, summaries)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md)
    return md


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace_dir", type=str, default="data/phase2_traces")
    parser.add_argument("--out", type=str, default="analysis/ORDER_PARAMETER_RESULTS.md")
    parser.add_argument(
        "--spectrum_stride", type=int, default=5,
        help="compute the (expensive) spectral observables every Nth checkpoint",
    )
    args = parser.parse_args()
    md = run(
        Path(args.trace_dir),
        Path(args.out) if args.out else None,
        spectrum_stride=args.spectrum_stride,
    )
    print(md)


if __name__ == "__main__":
    main()
