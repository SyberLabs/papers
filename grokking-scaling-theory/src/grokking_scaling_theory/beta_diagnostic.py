"""
grokking_scaling_theory.beta_diagnostic
---------------------------------------
The weight-decay exponent as a bottleneck diagnostic.

Claim under test
================
In the (M, R, D) effective theory, if the microscopic memorization dynamics
are ``dM/dt = -a1 * wd * M`` (i.e. beta = 1 microscopically), then the
*measured* local exponent

    beta_eff(p, wd, arch) = - d log(tau) / d log(wd)

is an emergent property of the coupled stopping time and identifies which
latent clock binds:

    beta_eff -> 1   cleanup-limited (M-gated: rule structure forms early,
                    deployment waits on memorization decay)
    beta_eff -> 0   formation-limited (R-gated: cleanup finishes early,
                    grokking waits on rule formation)
    0 < beta_eff < 1  crossover regime (clocks comparable)

Predictions:
    P1  Families whose traces show early-R / late-D should measure
        beta_eff ~ 1.  (Residual family.)
    P2  Formation-limited families should show beta_eff decreasing with p
        inside the Fourier regime, since tau_R ~ p^2/log(p)^2 grows.
    P3  Discriminator vs. a genuinely fractional mechanism (heterogeneous
        dissipation spectrum): the crossover account predicts beta_eff
        drifts across a wide wd ladder; a true fractional exponent
        predicts beta_eff stable across decades of wd.

This module computes beta_eff from the trace-backed scaling table and
emits a summary suitable for the analysis directory.

Usage
=====
    python -m grokking_scaling_theory.beta_diagnostic
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

DEFAULT_TABLE = (
    Path(__file__).resolve().parents[2] / "analysis" / "expanded_scaling_runs.csv"
)


@dataclass(frozen=True)
class BetaEstimate:
    """One local or ladder-level beta estimate."""

    group: str
    architecture: str
    modulus: int
    wd_low: float
    wd_high: float
    beta: float
    kind: str  # "pairwise" or "ladder"
    n_points: int


def load_ladders(
    csv_path: Path = DEFAULT_TABLE, measured_only: bool = True
) -> Dict[Tuple[str, str, int], Dict[float, float]]:
    """Group grokking times into (group, architecture, modulus) -> {wd: tau}."""
    ladders: Dict[Tuple[str, str, int], Dict[float, float]] = {}
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            if measured_only and row.get("measured", "True").strip().lower() != "true":
                continue
            try:
                tau = float(row["grokking_epoch"])
                wd = float(row["weight_decay"])
                p = int(row["modulus"])
            except (KeyError, ValueError):
                continue
            key = (row.get("group", "unknown"), row.get("architecture", "unknown"), p)
            ladders.setdefault(key, {})[wd] = tau
    return ladders


def pairwise_betas(ladders: Dict[Tuple[str, str, int], Dict[float, float]]) -> List[BetaEstimate]:
    """Local slopes between adjacent weight-decay points at fixed modulus."""
    estimates: List[BetaEstimate] = []
    for (group, arch, p), ladder in sorted(ladders.items()):
        wds = sorted(ladder)
        for w1, w2 in zip(wds, wds[1:]):
            beta = -np.log(ladder[w2] / ladder[w1]) / np.log(w2 / w1)
            estimates.append(
                BetaEstimate(group, arch, p, w1, w2, float(beta), "pairwise", 2)
            )
    return estimates


def ladder_betas(ladders: Dict[Tuple[str, str, int], Dict[float, float]]) -> List[BetaEstimate]:
    """Least-squares beta over full wd ladders with >= 3 points."""
    estimates: List[BetaEstimate] = []
    for (group, arch, p), ladder in sorted(ladders.items()):
        if len(ladder) < 3:
            continue
        wds = np.array(sorted(ladder))
        taus = np.array([ladder[w] for w in wds])
        X = np.column_stack([np.ones_like(wds), -np.log(wds)])
        coef, *_ = np.linalg.lstsq(X, np.log(taus), rcond=None)
        estimates.append(
            BetaEstimate(
                group, arch, p, float(wds[0]), float(wds[-1]),
                float(coef[1]), "ladder", len(wds),
            )
        )
    return estimates


def architecture_summary(estimates: List[BetaEstimate]) -> Dict[str, Dict[str, float]]:
    """Mean/std of pairwise beta per architecture, plus the p-trend slope."""
    summary: Dict[str, Dict[str, float]] = {}
    by_arch: Dict[str, List[BetaEstimate]] = {}
    for est in estimates:
        if est.kind == "pairwise":
            by_arch.setdefault(est.architecture, []).append(est)
    for arch, ests in sorted(by_arch.items()):
        betas = np.array([e.beta for e in ests])
        moduli = np.array([e.modulus for e in ests], dtype=float)
        entry = {
            "n": float(len(ests)),
            "beta_mean": float(np.mean(betas)),
            "beta_std": float(np.std(betas)),
        }
        if len(set(moduli)) >= 2:
            slope = np.polyfit(np.log(moduli), betas, 1)[0]
            entry["dbeta_dlogp"] = float(slope)
        summary[arch] = entry
    return summary


def classify(beta_mean: float, beta_std: float) -> str:
    """Coarse bottleneck classification with an honest uncertainty gate."""
    if beta_std > 0.15:
        return "indeterminate (spread too large)"
    if beta_mean >= 0.95:
        return "cleanup-limited (M-gated)"
    if beta_mean <= 0.25:
        return "formation-limited (R-gated)"
    return "crossover (clocks comparable)"


def format_report(
    estimates: List[BetaEstimate], summary: Dict[str, Dict[str, float]]
) -> str:
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append(" WEIGHT-DECAY EXPONENT AS BOTTLENECK DIAGNOSTIC")
    lines.append("   beta_eff = -d log(tau) / d log(wd), trace-backed points only")
    lines.append("=" * 78)
    lines.append(f" {'group':16s} {'arch':9s} {'p':>4s} {'wd range':>12s} {'beta':>7s} {'kind':>9s}")
    for est in estimates:
        lines.append(
            f" {est.group:16s} {est.architecture:9s} {est.modulus:4d}"
            f" {est.wd_low:5.2f}->{est.wd_high:4.2f} {est.beta:7.3f} {est.kind:>9s}"
        )
    lines.append("-" * 78)
    for arch, entry in summary.items():
        cls = classify(entry["beta_mean"], entry["beta_std"])
        trend = (
            f"  d(beta)/d(log p) = {entry['dbeta_dlogp']:+.3f}"
            if "dbeta_dlogp" in entry
            else ""
        )
        lines.append(
            f" {arch:9s} beta = {entry['beta_mean']:.3f} +/- {entry['beta_std']:.3f}"
            f" (n={int(entry['n'])})  -> {cls}{trend}"
        )
    lines.append("=" * 78)
    return "\n".join(lines)


def main(csv_path: Optional[Path] = None) -> None:
    ladders = load_ladders(csv_path or DEFAULT_TABLE)
    estimates = pairwise_betas(ladders) + ladder_betas(ladders)
    summary = architecture_summary(estimates)
    print(format_report(estimates, summary))


if __name__ == "__main__":
    main()
