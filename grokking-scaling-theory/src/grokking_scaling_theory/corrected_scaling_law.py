"""
isomorphic.derived.corrected_scaling_law
----------------------------------------
Structured evaluation of candidate grokking scaling laws.

This module now uses the repository's run-table dataset format rather than
hard-coded tuples. The preferred fixed-exponent law remains available:

    tau = C * p^2 / (log(p)^k * wd^beta)

but it is evaluated alongside competing models rather than in isolation.

Current interpretation:
    this law is a candidate effective summary of hidden dynamics involving
    memorization burden, rule formation, and deployment, not yet a final
    first-principles derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from grokking_scaling_theory.fit_competition import compare_models
from grokking_scaling_theory.scaling_study import ScalingDataset


DEFAULT_RUN_TABLE = Path(__file__).resolve().parents[2] / "data" / "empirical_scaling_runs.csv"


@dataclass
class CorrectedScalingLaw:
    """
    Fixed-exponent candidate scaling law.

    tau = C * p^2 / (log(p)^k * wd^beta)
    """

    k: float = 1.5
    beta: float = 0.65
    C: float = 1.0

    def predict(self, p: int, wd: float, lr: float = 1e-3) -> float:
        """Predict grokking epoch for one condition."""
        del lr  # learning-rate dependence is not modeled in this law yet
        return self.C * p**2 / (np.log(p) ** self.k * wd**self.beta)

    def fit_constant(self, dataset: ScalingDataset) -> float:
        """Fit the calibration constant C in log space on observed runs."""
        observed = dataset.fit_runs()
        if not observed:
            raise ValueError("Cannot fit corrected scaling law without observed runs.")

        log_empirical = np.array([np.log(run.grokking_epoch) for run in observed], dtype=float)
        log_predicted_c1 = np.array([
            np.log(self.predict(run.condition.modulus, run.condition.weight_decay, run.condition.learning_rate))
            for run in observed
        ])

        log_C = np.mean(log_empirical - log_predicted_c1)
        self.C = float(np.exp(log_C))
        return self.C

    def evaluate(self, dataset: ScalingDataset) -> Dict[str, object]:
        """Evaluate the fixed-exponent law against the observed dataset."""
        observed = dataset.fit_runs()
        if not observed:
            raise ValueError("Cannot evaluate corrected scaling law without observed runs.")

        self.fit_constant(dataset)

        results = []
        for run in observed:
            tau_emp = float(run.grokking_epoch)
            tau_pred = self.predict(
                run.condition.modulus,
                run.condition.weight_decay,
                run.condition.learning_rate,
            )
            rel_error = abs(tau_pred - tau_emp) / tau_emp * 100.0
            results.append({
                "p": run.condition.modulus,
                "wd": run.condition.weight_decay,
                "optimizer": run.condition.optimizer,
                "architecture": run.condition.architecture,
                "tau_empirical": tau_emp,
                "tau_predicted": tau_pred,
                "error_pct": rel_error,
                "source": run.source,
                "notes": run.notes,
            })

        errors = np.array([result["error_pct"] for result in results], dtype=float)

        return {
            "C": self.C,
            "k": self.k,
            "beta": self.beta,
            "mean_error": float(np.mean(errors)),
            "max_error": float(np.max(errors)),
            "std_error": float(np.std(errors)),
            "num_points": len(results),
            "results": results,
        }


def load_default_dataset(csv_path: Path = DEFAULT_RUN_TABLE) -> ScalingDataset:
    """Load the default empirical run table."""
    return ScalingDataset.from_run_table_csv(csv_path)


def summarize_dataset(dataset: ScalingDataset) -> Dict[str, object]:
    """Print and return a compact dataset summary."""
    summary = dataset.summary()
    print("=" * 90)
    print(" STRUCTURED EMPIRICAL DATASET")
    print("=" * 90)
    print(f"Runs:         {summary['num_runs']}")
    print(f"Observed:     {summary['num_observed']}")
    print(f"Censored:     {summary['num_censored']}")
    print(f"Used in fit:  {summary['num_fit_runs']}")
    print(f"Moduli:       {summary['moduli']}")
    print(f"Weight decays:{summary['weight_decays']}")
    print(f"Optimizers:   {summary['optimizers']}")
    print(f"Architectures:{summary['architectures']}")
    print("=" * 90)
    return summary


def report_diagnostic_coverage(dataset: ScalingDataset) -> List[Dict[str, object]]:
    """Print a compact summary of runs carrying diagnostic traces."""
    summaries = dataset.diagnostic_summaries()
    print("\n" + "=" * 90)
    print(" DIAGNOSTIC TRACE COVERAGE")
    print("=" * 90)
    if not summaries:
        print("No trace-backed runs are currently attached to the run table.")
        print("-" * 90)
        return summaries

    print(
        f"{'p':<8} {'wd':<8} {'grok':<8} {'peak_l2':<12} {'cleanup':<10} "
        f"{'best_val':<10} {'peak_vel':<10} {'alg_mass':<10} {'source':<15}"
    )
    print("-" * 90)
    for summary in summaries:
        peak_l2 = summary.get("peak_l2_norm")
        cleanup = summary.get("cleanup_onset_epoch")
        best_val = summary.get("best_val_acc")
        peak_vel = summary.get("peak_val_acc_velocity")
        alg_mass = summary.get("peak_algorithmic_mode_mass")
        print(
            f"{summary['modulus']:<8} {summary['weight_decay']:<8} "
            f"{str(summary.get('grokking_epoch')):<8} "
            f"{(f'{peak_l2:.2f}' if peak_l2 is not None else 'n/a'):<12} "
            f"{str(cleanup) if cleanup is not None else 'n/a':<10} "
            f"{(f'{best_val:.2f}' if best_val is not None else 'n/a'):<10} "
            f"{(f'{peak_vel:.4f}' if peak_vel is not None else 'n/a'):<10} "
            f"{(f'{alg_mass:.3f}' if alg_mass is not None else 'n/a'):<10} "
            f"{summary.get('source', 'unknown'):<15}"
        )
    print("-" * 90)
    return summaries


def report_local_trace_family(dataset: ScalingDataset) -> List[Dict[str, object]]:
    """Report local trace-backed runs ordered by weight decay."""
    summaries = sorted(
        [
            summary for summary in dataset.diagnostic_summaries()
            if str(summary.get("source", "")).startswith("local_trace")
        ],
        key=lambda summary: summary["weight_decay"],
    )

    print("\n" + "=" * 90)
    print(" LOCAL TRACE FAMILY")
    print("=" * 90)
    if not summaries:
        print("No local trace family is currently attached.")
        print("-" * 90)
        return summaries

    print(
        f"{'wd':<8} {'grok':<8} {'cleanup':<10} {'peak_vel':<10} "
        f"{'best_val':<10} {'final_mem':<10} {'source':<18}"
    )
    print("-" * 90)
    for summary in summaries:
        peak_vel = summary.get("peak_val_acc_velocity")
        best_val = summary.get("best_val_acc")
        final_mem = summary.get("final_memorization_burden")
        cleanup = summary.get("cleanup_onset_epoch")
        print(
            f"{summary['weight_decay']:<8} "
            f"{str(summary.get('grokking_epoch')):<8} "
            f"{str(cleanup) if cleanup is not None else 'n/a':<10} "
            f"{(f'{peak_vel:.4f}' if peak_vel is not None else 'n/a'):<10} "
            f"{(f'{best_val:.2f}' if best_val is not None else 'n/a'):<10} "
            f"{(f'{final_mem:.3f}' if final_mem is not None else 'n/a'):<10} "
            f"{summary.get('source', 'unknown'):<18}"
        )
    print("-" * 90)
    return summaries


def report_fit_competition(dataset: ScalingDataset) -> List[Dict[str, object]]:
    """Run default fit competition and print the ranked results."""
    ranked = compare_models(dataset)

    print("\n" + "=" * 90)
    print(" FIT COMPETITION")
    print("=" * 90)
    print(f"{'Model':<24} {'LOO RMSE(log)':<16} {'In-sample RMSE(log)':<20} {'Max |log err|':<14}")
    print("-" * 90)
    for result in ranked:
        print(
            f"{result.name:<24} "
            f"{result.loo_rmse_log:<16.4f} "
            f"{result.in_sample_rmse_log:<20.4f} "
            f"{result.max_abs_log_error:<14.4f}"
        )
    print("-" * 90)

    return [
        {
            "name": result.name,
            "loo_rmse_log": result.loo_rmse_log,
            "in_sample_rmse_log": result.in_sample_rmse_log,
            "max_abs_log_error": result.max_abs_log_error,
            "coefficients": result.coefficients,
        }
        for result in ranked
    ]


def test_corrected_law(dataset: ScalingDataset | None = None) -> Dict[str, object]:
    """Evaluate the preferred fixed-exponent law on the structured dataset."""
    dataset = dataset or load_default_dataset()

    print("\n" + "=" * 90)
    print(" FIXED-EXPONENT CANDIDATE LAW")
    print("=" * 90)
    print("tau = C * p^2 / (log(p)^k * wd^beta)")
    print("Current fixed exponents under test: k = 1.5, beta = 0.65")

    law = CorrectedScalingLaw(k=1.5, beta=0.65)
    evaluation = law.evaluate(dataset)

    print("-" * 90)
    print(f"Fitted constant C = {evaluation['C']:.2f}")
    print("-" * 90)
    print(f"{'p':<8} {'wd':<8} {'tau_emp':<12} {'tau_pred':<12} {'error':<10} {'source':<15}")
    print("-" * 90)
    for result in evaluation["results"]:
        print(
            f"{result['p']:<8} {result['wd']:<8} {result['tau_empirical']:<12.0f} "
            f"{result['tau_predicted']:<12.0f} {result['error_pct']:<10.1f}% {result['source']:<15}"
        )
    print("-" * 90)
    print(f"Mean error: {evaluation['mean_error']:.1f}%")
    print(f"Max error:  {evaluation['max_error']:.1f}%")
    print(f"Std error:  {evaluation['std_error']:.1f}%")
    print("-" * 90)

    return evaluation


def test_exponent_sensitivity(dataset: ScalingDataset | None = None) -> List[Dict[str, float]]:
    """Grid-check sensitivity to k and beta on the structured dataset."""
    dataset = dataset or load_default_dataset()

    print("\n" + "=" * 90)
    print(" EXPONENT SENSITIVITY ANALYSIS")
    print("=" * 90)
    print(f"{'k':<8} {'beta':<8} {'Mean Err':<12} {'Max Err':<12}")
    print("-" * 45)

    grid_results: List[Dict[str, float]] = []
    best_error = float("inf")
    best_pair = (1.5, 0.65)

    for k in [1.0, 1.25, 1.5, 1.75, 2.0]:
        for beta in [0.5, 0.6, 0.65, 0.7, 0.8]:
            law = CorrectedScalingLaw(k=k, beta=beta)
            evaluation = law.evaluate(dataset)
            print(f"{k:<8} {beta:<8} {evaluation['mean_error']:<12.1f}% {evaluation['max_error']:<12.1f}%")
            grid_results.append({
                "k": k,
                "beta": beta,
                "mean_error": evaluation["mean_error"],
                "max_error": evaluation["max_error"],
            })
            if evaluation["mean_error"] < best_error:
                best_error = evaluation["mean_error"]
                best_pair = (k, beta)

    print("-" * 45)
    print(f"Best grid point: k = {best_pair[0]}, beta = {best_pair[1]}, mean error = {best_error:.1f}%")
    return grid_results


def generate_predictions(dataset: ScalingDataset | None = None) -> Dict[str, object]:
    """Generate candidate predictions after fitting the fixed-exponent law."""
    dataset = dataset or load_default_dataset()
    law = CorrectedScalingLaw(k=1.5, beta=0.65)
    law.evaluate(dataset)

    print("\n" + "=" * 90)
    print(" CANDIDATE PREDICTIONS")
    print("=" * 90)
    print(f"Using fitted constant C = {law.C:.2f}")

    predictions = {
        "wd_1": {p: law.predict(p, 1.0) for p in [17, 23, 37, 53, 67, 83, 127, 151]},
        "p_97": {wd: law.predict(97, wd) for wd in [2.0, 0.5, 0.2, 0.05, 0.005]},
    }

    print("\nPredictions for wd=1.0:")
    for p, tau in predictions["wd_1"].items():
        print(f"  p={p:<4} -> tau ~ {tau:.0f}")

    print("\nPredictions for p=97:")
    for wd, tau in predictions["p_97"].items():
        print(f"  wd={wd:<5} -> tau ~ {tau:.0f}")

    return predictions


def main() -> Dict[str, object]:
    """Run the structured scaling-law analysis workflow."""
    dataset = load_default_dataset()
    dataset_summary = summarize_dataset(dataset)
    diagnostic_summaries = report_diagnostic_coverage(dataset)
    local_trace_family = report_local_trace_family(dataset)
    fit_results = report_fit_competition(dataset)
    corrected_evaluation = test_corrected_law(dataset)
    sensitivity = test_exponent_sensitivity(dataset)
    predictions = generate_predictions(dataset)

    print("\n" + "=" * 90)
    print(" WORKFLOW STATUS")
    print("=" * 90)
    print("The structured run-table path is now the default analysis route.")
    print("The fixed-exponent law is reported as a candidate model alongside fit competition.")
    print("Use the run table to add measured runs without editing Python source.")
    print("=" * 90)

    return {
        "dataset_summary": dataset_summary,
        "diagnostic_summaries": diagnostic_summaries,
        "local_trace_family": local_trace_family,
        "fit_competition": fit_results,
        "corrected_law": corrected_evaluation,
        "sensitivity": sensitivity,
        "predictions": predictions,
    }


if __name__ == "__main__":
    main()
