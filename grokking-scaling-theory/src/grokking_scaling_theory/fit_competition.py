"""
isomorphic.derived.fit_competition
----------------------------------
Fit competition for candidate grokking scaling laws.

This module compares multiple asymptotic forms on a common footing using the
observed portion of a ScalingDataset. The current implementation uses
log-space least squares and leave-one-out validation so it remains lightweight.

Important:
- censored runs are tracked but not yet handled with survival-analysis methods
- this module is meant to replace "fit only the preferred law" workflows
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from grokking_scaling_theory.scaling_study import ScalingDataset


ArrayTransform = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
ArrayOffset = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class CandidateModel:
    """Definition of one log-linear candidate scaling law."""

    name: str
    feature_names: List[str]
    offset: ArrayOffset
    transform: ArrayTransform


@dataclass
class FitResult:
    """Fit summary for one candidate model."""

    name: str
    offset: ArrayOffset
    transform: ArrayTransform
    feature_names: List[str]
    coefficients: Dict[str, float]
    in_sample_rmse_log: float
    loo_rmse_log: float
    max_abs_log_error: float
    num_points: int

    def predict(self, modulus: float, weight_decay: float, learning_rate: float = 1e-3) -> float:
        """Predict tau for one configuration."""
        p = np.array([modulus], dtype=float)
        wd = np.array([weight_decay], dtype=float)
        lr = np.array([learning_rate], dtype=float)
        X = self.transform(p, wd, lr)
        offset = self.offset(p, wd, lr)
        coeffs = np.array([self.coefficients["intercept"]])
        coeffs = np.concatenate([
            coeffs,
            np.array([self.coefficients[name] for name in self.feature_names], dtype=float),
        ])
        log_tau = float(offset[0] + X[0] @ coeffs)
        return float(np.exp(log_tau))


def safe_log(values: np.ndarray) -> np.ndarray:
    """Numerically safe natural log for positive arrays."""
    values = np.asarray(values, dtype=float)
    if np.any(values <= 0):
        raise ValueError("All inputs must be positive for log-space fitting.")
    return np.log(values)


def design_matrix(model: CandidateModel, p: np.ndarray, wd: np.ndarray, lr: np.ndarray) -> np.ndarray:
    """Construct the log-linear design matrix for a candidate model."""
    features = model.transform(p, wd, lr)
    if features.ndim != 2:
        raise ValueError("CandidateModel transform must return a 2D design matrix.")
    return features


def zero_offset(p: np.ndarray, wd: np.ndarray, lr: np.ndarray) -> np.ndarray:
    """Default offset for free-form models."""
    return np.zeros_like(p, dtype=float)


def fit_candidate(
    dataset: ScalingDataset,
    model: CandidateModel,
) -> FitResult:
    """Fit one candidate model using observed grokking runs only."""
    observed = dataset.fit_runs()
    if len(observed) < 3:
        raise ValueError("At least three observed runs are required for fit competition.")

    p = np.array([run.condition.modulus for run in observed], dtype=float)
    wd = np.array([run.condition.weight_decay for run in observed], dtype=float)
    lr = np.array([run.condition.learning_rate for run in observed], dtype=float)
    tau = np.array([run.grokking_epoch for run in observed], dtype=float)

    X = design_matrix(model, p, wd, lr)
    y = safe_log(tau) - model.offset(p, wd, lr)
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coeffs
    residuals = y_hat - y

    loo_errors = []
    for holdout_idx in range(len(observed)):
        mask = np.ones(len(observed), dtype=bool)
        mask[holdout_idx] = False
        coeffs_loo, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
        pred = float(X[holdout_idx] @ coeffs_loo)
        loo_errors.append(pred - y[holdout_idx])

    coefficients = {"intercept": float(coeffs[0])}
    for index, feature_name in enumerate(model.feature_names, start=1):
        coefficients[feature_name] = float(coeffs[index])

    return FitResult(
        name=model.name,
        offset=model.offset,
        transform=model.transform,
        feature_names=model.feature_names,
        coefficients=coefficients,
        in_sample_rmse_log=float(np.sqrt(np.mean(residuals**2))),
        loo_rmse_log=float(np.sqrt(np.mean(np.square(loo_errors)))),
        max_abs_log_error=float(np.max(np.abs(residuals))),
        num_points=len(observed),
    )


def compare_models(dataset: ScalingDataset) -> List[FitResult]:
    """Fit and rank all default candidate models by leave-one-out error."""
    results = [fit_candidate(dataset, model) for model in default_candidate_models()]
    return sorted(results, key=lambda result: (result.loo_rmse_log, result.in_sample_rmse_log))


def default_candidate_models() -> List[CandidateModel]:
    """Default set of competing laws for Phase 1 scaling analysis."""
    return [
        CandidateModel(
            name="power_law_p_only",
            feature_names=["log_p"],
            offset=zero_offset,
            transform=lambda p, wd, lr: np.column_stack([
                np.ones_like(p),
                safe_log(p),
            ]),
        ),
        CandidateModel(
            name="mean_field_p2_over_wd",
            feature_names=[],
            offset=lambda p, wd, lr: 2.0 * safe_log(p) - safe_log(wd),
            transform=lambda p, wd, lr: np.column_stack([
                np.ones_like(p),
            ]),
        ),
        CandidateModel(
            name="p2_over_logp",
            feature_names=[],
            offset=lambda p, wd, lr: 2.0 * safe_log(p) - safe_log(safe_log(p)) - safe_log(wd),
            transform=lambda p, wd, lr: np.column_stack([
                np.ones_like(p),
            ]),
        ),
        CandidateModel(
            name="p2_over_logp2",
            feature_names=[],
            offset=lambda p, wd, lr: 2.0 * safe_log(p) - 2.0 * safe_log(safe_log(p)) - safe_log(wd),
            transform=lambda p, wd, lr: np.column_stack([
                np.ones_like(p),
            ]),
        ),
        CandidateModel(
            name="flexible_logk_wdbeta",
            feature_names=["neg_log_log_p", "neg_log_wd"],
            offset=lambda p, wd, lr: 2.0 * safe_log(p),
            transform=lambda p, wd, lr: np.column_stack([
                np.ones_like(p),
                -safe_log(safe_log(p)),
                -safe_log(wd),
            ]),
        ),
    ]


def empirical_dataset_from_tuples(rows: List[tuple]) -> ScalingDataset:
    """
    Convert legacy tuple data into a ScalingDataset.

    Expected tuple format:
        (modulus, weight_decay, learning_rate, grokking_epoch, source)
    """
    from grokking_scaling_theory.scaling_study import ExperimentCondition, GrokkingRun

    runs = []
    for seed, row in enumerate(rows):
        modulus, weight_decay, learning_rate, grokking_epoch, source = row
        condition = ExperimentCondition(
            modulus=modulus,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
        )
        runs.append(
            GrokkingRun(
                condition=condition,
                seed=seed,
                max_epochs=int(grokking_epoch),
                source=source,
                observed_grokking_epoch=int(grokking_epoch),
            )
        )
    return ScalingDataset.from_runs(runs)
