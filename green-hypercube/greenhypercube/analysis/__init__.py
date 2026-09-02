"""Analysis and figure generation."""

from __future__ import annotations

from .figures import (
    plot_discovery_curves,
    plot_summary_bars,
    plot_sensitivity,
    plot_null_controls,
    plot_coupling,
    plot_multivariate_coupling,
    plot_m2_advantage,
    plot_matched_density_advantage,
)

__all__ = [
    "plot_discovery_curves",
    "plot_summary_bars",
    "plot_sensitivity",
    "plot_null_controls",
    "plot_coupling",
    "plot_multivariate_coupling",
    "plot_m2_advantage",
    "plot_matched_density_advantage",
]
