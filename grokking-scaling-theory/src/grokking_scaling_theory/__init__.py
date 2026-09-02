"""Standalone package for grokking scaling theory and RG-inspired analysis."""

from grokking_scaling_theory.corrected_scaling_law import CorrectedScalingLaw, load_default_dataset
from grokking_scaling_theory.fit_competition import compare_models
from grokking_scaling_theory.scaling_study import ScalingDataset, ExperimentCondition, GrokkingRun

# Fable DTE expansion (2026-07): censoring-aware survival fits, the weight-decay
# bottleneck diagnostic, and the Phase 2 categorical (sheaf) order parameter.
from grokking_scaling_theory.survival_fit import (
    censoring_impact_report,
    fit_aft,
    AFTResult,
)
from grokking_scaling_theory.beta_diagnostic import (
    load_ladders,
    pairwise_betas,
    architecture_summary,
    BetaEstimate,
)
from grokking_scaling_theory.logical_cells import (
    d_logic_series,
    proposition_family,
    family_ceiling,
    fourier_concentration,
    run_validation_gates,
)
from grokking_scaling_theory.sheaf_order_parameter import (
    TraceArrays,
    load_trace_npz,
    project_pca,
    compute_order_parameters,
    transition_epoch,
    transition_details,
    run_synthetic_validation,
)

__all__ = [
    "CorrectedScalingLaw",
    "load_default_dataset",
    "compare_models",
    "ScalingDataset",
    "ExperimentCondition",
    "GrokkingRun",
    # survival_fit
    "censoring_impact_report",
    "fit_aft",
    "AFTResult",
    # beta_diagnostic
    "load_ladders",
    "pairwise_betas",
    "architecture_summary",
    "BetaEstimate",
    # logical_cells
    "d_logic_series",
    "proposition_family",
    "family_ceiling",
    "fourier_concentration",
    "run_validation_gates",
    # sheaf_order_parameter
    "TraceArrays",
    "load_trace_npz",
    "project_pca",
    "compute_order_parameters",
    "transition_epoch",
    "run_synthetic_validation",
]
