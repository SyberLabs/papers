"""
Log-Exponent Collapse Analysis for Grokking Scaling Law

This script discriminates between competing scaling hypotheses:
    tau ~ p^2 / log(p)^q
for q in {0, 1, 1.5, 2, 2.5, 3}.

The key figure shows:
1. Coefficient of variation (CV) as a function of q
2. Collapsed data at the optimal q value
3. Residual structure to check for systematic deviations

A successful collapse at q=2 supports the log-squared mechanism from:
    - Harmonic mode accumulation: H(p) ~ log(p)
    - Marginal coordination susceptibility: chi(p) ~ log(p)
    - Combined: gamma_R(p) ~ log(p)^2 / p^2

Usage:
    python log_exponent_collapse.py [--output-dir figures/]
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import sys

import numpy as np

# Optional matplotlib import for environments without display
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Text output only.")


@dataclass
class ScalingPoint:
    """Single empirical (p, wd, tau) measurement."""
    modulus: int
    weight_decay: float
    grokking_epoch: int
    source: str = "unknown"

    def __post_init__(self):
        if self.modulus < 2:
            raise ValueError(f"Invalid modulus: {self.modulus}")
        if self.weight_decay <= 0:
            raise ValueError(f"Invalid weight_decay: {self.weight_decay}")
        if self.grokking_epoch <= 0:
            raise ValueError(f"Invalid grokking_epoch: {self.grokking_epoch}")


def load_scaling_data(csv_path: Path) -> List[ScalingPoint]:
    """Load empirical scaling data from run table CSV."""
    points = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows not included in scaling fit
            include = row.get("include_in_scaling_fit", "True")
            if include.lower() in ("false", "0", "no"):
                continue

            # Skip censored runs
            is_censored = row.get("is_censored", "False")
            if is_censored.lower() in ("true", "1", "yes"):
                continue

            # Parse required fields
            try:
                grokking_epoch = row.get("grokking_epoch", "")
                if not grokking_epoch or grokking_epoch == "":
                    continue

                points.append(ScalingPoint(
                    modulus=int(float(row["modulus"])),
                    weight_decay=float(row["weight_decay"]),
                    grokking_epoch=int(float(grokking_epoch)),
                    source=row.get("source", "unknown"),
                ))
            except (ValueError, KeyError) as e:
                print(f"Skipping row: {e}")
                continue

    return points


def compute_normalized_tau(
    point: ScalingPoint,
    q: float,
    beta: float = 0.65,
) -> float:
    """
    Compute the normalized grokking time under the scaling hypothesis.

    If the true law is tau = C * p^2 / (log(p)^q * wd^beta),
    then the normalized quantity
        tau_norm = tau * log(p)^q * wd^beta / p^2
    should equal C (constant across all points).
    """
    p = point.modulus
    wd = point.weight_decay
    tau = point.grokking_epoch

    log_p = np.log(p)
    normalized = tau * (log_p ** q) * (wd ** beta) / (p ** 2)

    return normalized


def compute_collapse_metric(
    points: List[ScalingPoint],
    q: float,
    beta: float = 0.65,
) -> Tuple[float, float, float]:
    """
    Compute collapse quality metrics for a given q value.

    Returns:
        cv: Coefficient of variation (std/mean) - lower is better
        mean_C: Estimated calibration constant
        std_C: Standard deviation of calibration constant estimates
    """
    normalized = [compute_normalized_tau(p, q, beta) for p in points]

    mean_C = np.mean(normalized)
    std_C = np.std(normalized)
    cv = std_C / mean_C if mean_C > 0 else float('inf')

    return cv, mean_C, std_C


def scan_q_values(
    points: List[ScalingPoint],
    q_range: Tuple[float, float] = (0.0, 3.0),
    n_points: int = 61,
    beta: float = 0.65,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Scan q values and find the optimal collapse.

    Returns:
        q_values: Array of q values tested
        cv_values: Array of CV values at each q
        q_best: Optimal q value (minimum CV)
        cv_best: CV at optimal q
    """
    q_values = np.linspace(q_range[0], q_range[1], n_points)
    cv_values = np.array([compute_collapse_metric(points, q, beta)[0] for q in q_values])

    best_idx = np.argmin(cv_values)
    q_best = q_values[best_idx]
    cv_best = cv_values[best_idx]

    return q_values, cv_values, q_best, cv_best


def compute_prediction_errors(
    points: List[ScalingPoint],
    q: float,
    beta: float = 0.65,
) -> List[Tuple[ScalingPoint, float, float, float]]:
    """
    Compute prediction errors for each point at given q.

    Returns list of (point, tau_predicted, tau_observed, relative_error).
    """
    # First compute the calibration constant
    _, C, _ = compute_collapse_metric(points, q, beta)

    results = []
    for point in points:
        p = point.modulus
        wd = point.weight_decay
        tau_obs = point.grokking_epoch

        # Predicted: tau = C * p^2 / (log(p)^q * wd^beta)
        tau_pred = C * (p ** 2) / ((np.log(p) ** q) * (wd ** beta))

        rel_error = (tau_pred - tau_obs) / tau_obs

        results.append((point, tau_pred, tau_obs, rel_error))

    return results


def print_analysis_report(
    points: List[ScalingPoint],
    q_values: np.ndarray,
    cv_values: np.ndarray,
    q_best: float,
    beta: float = 0.65,
):
    """Print detailed analysis report to stdout."""
    print("=" * 80)
    print(" LOG-EXPONENT COLLAPSE ANALYSIS")
    print("=" * 80)
    print(f"\nDataset: {len(points)} points included in scaling fit")
    print(f"Fixed beta (weight decay exponent): {beta}")

    print("\n" + "-" * 80)
    print(" Q-VALUE SCAN")
    print("-" * 80)
    print(f"{'q':<8} {'CV':<12} {'Quality':<20}")
    print("-" * 40)

    for q_test in [0.0, 1.0, 1.5, 2.0, 2.5, 3.0]:
        cv, _, _ = compute_collapse_metric(points, q_test, beta)
        quality = "*** BEST ***" if abs(q_test - q_best) < 0.1 else ""
        print(f"{q_test:<8.1f} {cv:<12.4f} {quality}")

    print("-" * 40)
    print(f"\nOptimal q = {q_best:.2f} (CV = {cv_values.min():.4f})")

    # Detailed results at optimal q
    print("\n" + "-" * 80)
    print(f" PREDICTIONS AT q = {q_best:.1f}")
    print("-" * 80)

    errors = compute_prediction_errors(points, q_best, beta)
    _, C, _ = compute_collapse_metric(points, q_best, beta)

    print(f"\nCalibration constant C = {C:.2f}")
    print(f"\nScaling law: tau = {C:.1f} * p^2 / (log(p)^{q_best:.1f} * wd^{beta})")

    print(f"\n{'p':<8} {'wd':<8} {'tau_obs':<12} {'tau_pred':<12} {'error':<10} {'source':<15}")
    print("-" * 70)

    rel_errors = []
    for point, tau_pred, tau_obs, rel_err in errors:
        print(f"{point.modulus:<8} {point.weight_decay:<8.2f} {tau_obs:<12} "
              f"{tau_pred:<12.0f} {rel_err*100:>+8.1f}% {point.source:<15}")
        rel_errors.append(abs(rel_err))

    print("-" * 70)
    print(f"Mean absolute error: {np.mean(rel_errors)*100:.1f}%")
    print(f"Max absolute error:  {np.max(rel_errors)*100:.1f}%")

    # Hypothesis comparison
    print("\n" + "-" * 80)
    print(" HYPOTHESIS COMPARISON")
    print("-" * 80)

    hypotheses = [
        (0.0, "tau ~ p^2 / wd^beta (no log correction)"),
        (1.0, "tau ~ p^2 / (log(p) * wd^beta)"),
        (1.5, "tau ~ p^2 / (log(p)^1.5 * wd^beta)"),
        (2.0, "tau ~ p^2 / (log(p)^2 * wd^beta)"),
    ]

    print(f"\n{'q':<6} {'CV':<10} {'Mean Err':<12} {'Hypothesis':<40}")
    print("-" * 70)

    for q_test, hypothesis in hypotheses:
        cv, _, _ = compute_collapse_metric(points, q_test, beta)
        errs = compute_prediction_errors(points, q_test, beta)
        mean_err = np.mean([abs(e[3]) for e in errs]) * 100
        marker = " <-- BEST" if abs(q_test - q_best) < 0.25 else ""
        print(f"{q_test:<6.1f} {cv:<10.4f} {mean_err:<12.1f}% {hypothesis}{marker}")

    print("=" * 80)


def create_publication_figure(
    points: List[ScalingPoint],
    q_values: np.ndarray,
    cv_values: np.ndarray,
    q_best: float,
    beta: float = 0.65,
    output_path: Optional[Path] = None,
):
    """Create publication-ready collapse analysis figure."""
    if not HAS_MATPLOTLIB:
        print("Skipping figure generation (matplotlib not available)")
        return

    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)

    # Panel 1: CV(q) curve
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(q_values, cv_values, 'b-', linewidth=2)
    ax1.axvline(x=q_best, color='r', linestyle='--', linewidth=1.5,
                label=f'Optimal q = {q_best:.2f}')

    # Mark key q values
    for q_mark in [0, 1, 1.5, 2]:
        idx = np.argmin(np.abs(q_values - q_mark))
        ax1.plot(q_mark, cv_values[idx], 'ko', markersize=8)
        ax1.annotate(f'q={q_mark}', (q_mark, cv_values[idx]),
                     textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax1.set_xlabel('Log exponent q', fontsize=12)
    ax1.set_ylabel('Coefficient of Variation', fontsize=12)
    ax1.set_title('(a) Collapse Quality vs Log Exponent', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 3)

    # Panel 2: Collapsed data at optimal q
    ax2 = fig.add_subplot(gs[1])

    _, C, _ = compute_collapse_metric(points, q_best, beta)

    # Group by weight decay for coloring
    wd_values = sorted(set(p.weight_decay for p in points))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(wd_values)))
    wd_colors = {wd: colors[i] for i, wd in enumerate(wd_values)}

    for point in points:
        normalized = compute_normalized_tau(point, q_best, beta)
        ax2.scatter(point.modulus, normalized,
                   c=[wd_colors[point.weight_decay]],
                   s=100, edgecolors='black', linewidth=1)

    ax2.axhline(y=C, color='r', linestyle='-', linewidth=2,
                label=f'C = {C:.1f}')
    ax2.axhline(y=C*1.2, color='r', linestyle=':', alpha=0.5)
    ax2.axhline(y=C*0.8, color='r', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Modulus p', fontsize=12)
    ax2.set_ylabel(f'$\\tau \\cdot \\log(p)^{{{q_best:.1f}}} \\cdot wd^{{{beta}}} / p^2$', fontsize=12)
    ax2.set_title(f'(b) Data Collapse at q = {q_best:.1f}', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Add colorbar for weight decay
    sm = plt.cm.ScalarMappable(cmap='viridis',
                                norm=plt.Normalize(vmin=min(wd_values), vmax=max(wd_values)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, shrink=0.8)
    cbar.set_label('Weight Decay', fontsize=10)

    # Panel 3: Residuals
    ax3 = fig.add_subplot(gs[2])

    errors = compute_prediction_errors(points, q_best, beta)

    for point, tau_pred, tau_obs, rel_err in errors:
        ax3.scatter(point.modulus, rel_err * 100,
                   c=[wd_colors[point.weight_decay]],
                   s=100, edgecolors='black', linewidth=1)

    ax3.axhline(y=0, color='k', linestyle='-', linewidth=1)
    ax3.axhline(y=20, color='r', linestyle=':', alpha=0.5)
    ax3.axhline(y=-20, color='r', linestyle=':', alpha=0.5)

    ax3.set_xlabel('Modulus p', fontsize=12)
    ax3.set_ylabel('Relative Error (%)', fontsize=12)
    ax3.set_title('(c) Prediction Residuals', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(
        f'Log-Exponent Discrimination: $\\tau = C \\cdot p^2 / (\\log p)^q \\cdot wd^{{\\beta}}$',
        fontsize=14, y=1.02
    )

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")

    plt.show()


def create_modulus_scaling_figure(
    points: List[ScalingPoint],
    q_best: float,
    beta: float = 0.65,
    output_path: Optional[Path] = None,
):
    """Create figure showing tau vs p^2/log(p)^q scaling."""
    if not HAS_MATPLOTLIB:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Filter to wd=1.0 points for clean modulus scaling
    wd1_points = [p for p in points if abs(p.weight_decay - 1.0) < 0.01]

    if len(wd1_points) < 2:
        print("Not enough wd=1.0 points for modulus scaling figure")
        return

    moduli = np.array([p.modulus for p in wd1_points])
    taus = np.array([p.grokking_epoch for p in wd1_points])

    # Panel 1: Raw tau vs p
    ax1.scatter(moduli, taus, s=100, c='blue', edgecolors='black', linewidth=1)

    # Fit lines for different q
    p_fit = np.linspace(min(moduli)*0.9, max(moduli)*1.1, 100)

    for q_test, color, style in [(0, 'gray', ':'), (1, 'orange', '--'), (2, 'red', '-')]:
        _, C, _ = compute_collapse_metric(wd1_points, q_test, beta)
        tau_fit = C * (p_fit ** 2) / (np.log(p_fit) ** q_test)
        ax1.plot(p_fit, tau_fit, color=color, linestyle=style, linewidth=2,
                label=f'q={q_test}: $\\tau \\sim p^2/\\log(p)^{q_test}$')

    ax1.set_xlabel('Modulus p', fontsize=12)
    ax1.set_ylabel('Grokking Time $\\tau$', fontsize=12)
    ax1.set_title('(a) Grokking Time vs Modulus (wd=1.0)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Log-log plot
    ax2.scatter(np.log(moduli), np.log(taus), s=100, c='blue', edgecolors='black', linewidth=1)

    log_p = np.log(p_fit)
    for q_test, color, style in [(0, 'gray', ':'), (1, 'orange', '--'), (2, 'red', '-')]:
        _, C, _ = compute_collapse_metric(wd1_points, q_test, beta)
        log_tau_fit = np.log(C) + 2*log_p - q_test*np.log(log_p)
        ax2.plot(log_p, log_tau_fit, color=color, linestyle=style, linewidth=2,
                label=f'q={q_test}')

    ax2.set_xlabel('$\\log(p)$', fontsize=12)
    ax2.set_ylabel('$\\log(\\tau)$', fontsize=12)
    ax2.set_title('(b) Log-Log Scaling', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Modulus Scaling at Fixed Weight Decay', fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Log-exponent collapse analysis")
    parser.add_argument("--data", type=Path,
                       default=Path(__file__).parent.parent / "data" / "empirical_scaling_runs.csv",
                       help="Path to empirical scaling runs CSV")
    parser.add_argument("--output-dir", type=Path,
                       default=Path(__file__).parent / "figures",
                       help="Output directory for figures")
    parser.add_argument("--beta", type=float, default=0.65,
                       help="Fixed weight decay exponent")
    parser.add_argument("--no-figures", action="store_true",
                       help="Skip figure generation")

    args = parser.parse_args()

    # Load data
    print(f"Loading data from: {args.data}")
    points = load_scaling_data(args.data)

    if len(points) < 3:
        print(f"Error: Need at least 3 points for analysis, got {len(points)}")
        sys.exit(1)

    print(f"Loaded {len(points)} points for analysis")

    # Run q-value scan
    q_values, cv_values, q_best, cv_best = scan_q_values(points, beta=args.beta)

    # Print report
    print_analysis_report(points, q_values, cv_values, q_best, beta=args.beta)

    # Generate figures
    if not args.no_figures and HAS_MATPLOTLIB:
        create_publication_figure(
            points, q_values, cv_values, q_best,
            beta=args.beta,
            output_path=args.output_dir / "log_exponent_collapse.png"
        )

        create_modulus_scaling_figure(
            points, q_best,
            beta=args.beta,
            output_path=args.output_dir / "modulus_scaling.png"
        )

    # Return results for programmatic use
    return {
        "q_best": q_best,
        "cv_best": cv_best,
        "n_points": len(points),
        "beta": args.beta,
    }


if __name__ == "__main__":
    main()
