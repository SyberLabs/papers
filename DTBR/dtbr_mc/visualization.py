"""Plotting for Experiment 001 and the sensitivity analysis.

All functions take tidy DataFrames produced by :mod:`dtbr_mc.experiments` and
write PNG files to a directory. Matplotlib + seaborn only; no display backend is
required (``Agg`` is forced) so the module runs headless.

The plots are deliberately plain: they are evidence, not decoration. Each one is
designed to make a specific claim checkable by eye -- in particular whether the
two policy levers (Semantic Clarity vs Phenomenological Caution) cross over as
interpretive capacity changes, which is the crux of H1.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

if TYPE_CHECKING:
    from dtbr_mc.experiments import Experiment001Result

sns.set_theme(style="whitegrid", context="notebook")

# A single metric name used throughout for the "harm-ish" surface on heatmaps.
_HEAT_METRIC = "disturbance_rate"


def _save(fig: plt.Figure, path: str) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Heatmaps: marker_clarity x phenomenological_caution at each IC level
# --------------------------------------------------------------------------- #


def plot_heatmaps(heatmaps: pd.DataFrame, outdir: str, metric: str = _HEAT_METRIC) -> list[str]:
    """One heatmap per interpretive-capacity slice (SC on x, PC on y)."""
    paths: list[str] = []
    ic_levels = sorted(heatmaps["ic_level"].unique())
    vmin = float(heatmaps[metric].min())
    vmax = float(heatmaps[metric].max())
    if vmax - vmin < 1e-12:
        vmax = vmin + 1e-12
    for ic in ic_levels:
        sub = heatmaps[heatmaps["ic_level"] == ic]
        grid = sub.pivot(index="phen_caution", columns="marker_clarity", values=metric)
        grid = grid.sort_index(ascending=False)  # PC high at top
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        sns.heatmap(
            grid, ax=ax, cmap="rocket_r", vmin=vmin, vmax=vmax,
            cbar_kws={"label": metric.replace("_", " ")},
        )
        ax.set_xlabel("Semantic Clarity  (marker_clarity)")
        ax.set_ylabel("Phenomenological Caution")
        ax.set_title(f"{metric.replace('_', ' ').title()}  |  interpretive_capacity = {ic:g}")
        # thin the tick labels
        ax.set_xticks(ax.get_xticks()[::4])
        ax.set_xticklabels([f"{float(t.get_text()):.2f}" for t in ax.get_xticklabels()[::1]][: len(ax.get_xticks())], rotation=45)
        ax.set_yticks(ax.get_yticks()[::4])
        p = os.path.join(outdir, f"heatmap_{metric}_ic{ic:g}.png")
        paths.append(_save(fig, p))
    return paths


# --------------------------------------------------------------------------- #
# Phase diagram: which lever is the stronger local brake on intervention?
# --------------------------------------------------------------------------- #


def plot_phase_diagram(heatmaps: pd.DataFrame, outdir: str) -> str:
    """Map regions where PC vs SC is the locally stronger brake on intervention.

    At each (SC, PC) cell and IC slice we estimate the local gradient of
    mean_intervention along each lever (finite differences over the grid). The
    region is shaded by which lever has the more negative (more suppressing)
    gradient. This is the spatial picture behind the H1 crossover test.
    """
    ic_levels = sorted(heatmaps["ic_level"].unique())
    n = len(ic_levels)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.6), squeeze=False)
    for j, ic in enumerate(ic_levels):
        sub = heatmaps[heatmaps["ic_level"] == ic]
        piv = sub.pivot(index="phen_caution", columns="marker_clarity", values="mean_intervention")
        piv = piv.sort_index()
        pc_axis = piv.index.to_numpy()
        sc_axis = piv.columns.to_numpy()
        Z = piv.to_numpy()
        # gradient along SC (columns -> axis=1) and PC (rows -> axis=0)
        dSC = np.gradient(Z, sc_axis, axis=1)
        dPC = np.gradient(Z, pc_axis, axis=0)
        # PC is the stronger brake where its (negative) slope is below SC's.
        pc_stronger = (dPC < dSC).astype(float)
        ax = axes[0, j]
        ax.pcolormesh(sc_axis, pc_axis, pc_stronger, cmap="coolwarm", vmin=0, vmax=1, shading="auto")
        ax.set_title(f"IC = {ic:g}")
        ax.set_xlabel("Semantic Clarity")
        if j == 0:
            ax.set_ylabel("Phenomenological Caution")
    # legend
    from matplotlib.patches import Patch

    handles = [
        Patch(color=plt.cm.coolwarm(0.95), label="PC stronger brake"),
        Patch(color=plt.cm.coolwarm(0.05), label="SC stronger brake"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Phase diagram: locally stronger brake on intervention", y=1.02)
    return _save(fig, os.path.join(outdir, "phase_diagram.png"))


# --------------------------------------------------------------------------- #
# Outcome distribution: stacked area over each swept lever
# --------------------------------------------------------------------------- #


def plot_outcome_distributions(outcome_distribution: pd.DataFrame, outdir: str) -> list[str]:
    """Stacked outcome composition as each swept variable moves 0 -> 1."""
    order = ["AVOID", "OBSERVE", "PRESERVE", "INVESTIGATE", "EXCAVATE"]
    palette = sns.color_palette("viridis", n_colors=len(order))
    paths: list[str] = []
    for var in outcome_distribution["variable"].unique():
        sub = outcome_distribution[outcome_distribution["variable"] == var]
        wide = sub.pivot(index="value", columns="outcome", values="fraction").fillna(0.0)
        wide = wide.reindex(columns=[c for c in order if c in wide.columns])
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        ax.stackplot(wide.index.to_numpy(), *[wide[c].to_numpy() for c in wide.columns],
                     labels=list(wide.columns), colors=palette[: len(wide.columns)], alpha=0.9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(var)
        ax.set_ylabel("population fraction")
        ax.set_title(f"Outcome composition vs {var}")
        ax.legend(loc="upper center", ncol=len(wide.columns), fontsize=8, frameon=False,
                  bbox_to_anchor=(0.5, -0.15))
        paths.append(_save(fig, os.path.join(outdir, f"outcomes_{var}.png")))
    return paths


# --------------------------------------------------------------------------- #
# Interaction plots: E[H] and mean intervention vs each lever, per IC level
# --------------------------------------------------------------------------- #


def plot_interactions(interactions: pd.DataFrame, outdir: str) -> list[str]:
    """Line plots of E[H] and mean intervention vs lever value, by IC level."""
    paths: list[str] = []
    ic_levels = sorted(interactions["ic_level"].unique())
    palette = sns.color_palette("flare", n_colors=len(ic_levels))
    for metric in ["mean_intervention", "expected_harm"]:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
        for ax, lever in zip(axes, ["SC", "PC"]):
            sub = interactions[interactions["lever"] == lever]
            for c, ic in zip(palette, ic_levels):
                s = sub[sub["ic_level"] == ic].sort_values("value")
                ax.plot(s["value"], s[metric], marker="o", ms=3, color=c, label=f"IC={ic:g}")
            full = "Semantic Clarity" if lever == "SC" else "Phenomenological Caution"
            ax.set_title(full)
            ax.set_xlabel(f"{full} value")
            ax.set_xlim(0, 1)
        axes[0].set_ylabel(metric.replace("_", " "))
        axes[1].legend(title="interpretive capacity", fontsize=8)
        fig.suptitle(f"{metric.replace('_', ' ').title()} vs each lever, by interpretive capacity")
        paths.append(_save(fig, os.path.join(outdir, f"interactions_{metric}.png")))
    return paths


# --------------------------------------------------------------------------- #
# H1 margin: slope_SC - slope_PC vs interpretive capacity
# --------------------------------------------------------------------------- #


def plot_h1_margin(h1_table: pd.DataFrame, outdir: str,
                   threshold: float | None = None) -> str:
    """The decisive H1 figure: margin = slope_SC - slope_PC vs IC.

    margin > 0  <=>  PC reduces the target more steeply than SC at that capacity.
    A downward zero crossing is the threshold H1 predicts.
    """
    t = h1_table.sort_values("interpretive_capacity")
    ic = t["interpretive_capacity"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharex=True)
    for ax, target in zip(axes, ["intervention", "expected_harm"]):
        m = t[f"margin_{target}"].to_numpy()
        ax.axhline(0.0, color="0.4", lw=1, ls="--")
        ax.plot(ic, m, marker="o", color="#b5462f")
        ax.fill_between(ic, m, 0, where=m > 0, alpha=0.18, color="tab:blue",
                        label="PC stronger brake")
        ax.fill_between(ic, m, 0, where=m <= 0, alpha=0.18, color="tab:red",
                        label="SC stronger brake")
        ax.set_xlabel("interpretive capacity")
        ax.set_title(f"margin on {target.replace('_', ' ')}")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("slope_SC − slope_PC")
    if threshold is not None:
        for ax in axes:
            ax.axvline(threshold, color="k", lw=1, ls=":")
        axes[0].annotate(f"IC* ≈ {threshold:.3f}", xy=(threshold, 0),
                         xytext=(threshold, axes[0].get_ylim()[1] * 0.6),
                         fontsize=9, ha="center")
    fig.suptitle("H1 test — does PC out-brake SC, and only below a capacity threshold?")
    return _save(fig, os.path.join(outdir, "h1_margin.png"))


# --------------------------------------------------------------------------- #
# Sensitivity tornado
# --------------------------------------------------------------------------- #


def plot_sensitivity(ranking: pd.DataFrame, outdir: str, top: int = 17) -> str:
    """Tornado-style bar chart of total-order Sobol indices (with S1 overlaid)."""
    r = ranking.head(top).iloc[::-1]  # smallest at bottom
    names = [str(i) for i in r.index]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(names) + 1.2))
    ax.barh(y, r["ST"].to_numpy(), color="#4c6ef5", alpha=0.85, label="ST (total order)")
    ax.barh(y, r["S1"].to_numpy(), color="#f59f00", alpha=0.95, height=0.5, label="S1 (first order)")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Sobol index of Expected Harm")
    ax.set_title("Parameter importance for E[H] (uniform priors)")
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, os.path.join(outdir, "sensitivity_tornado.png"))


# --------------------------------------------------------------------------- #
# Convenience: render everything for an experiment result
# --------------------------------------------------------------------------- #


def render_all(result: "Experiment001Result", outdir: str,
               ranking: pd.DataFrame | None = None) -> list[str]:
    """Render every figure for a finished Experiment 001 run."""
    os.makedirs(outdir, exist_ok=True)
    paths: list[str] = []
    paths += plot_heatmaps(result.heatmaps, outdir, metric="disturbance_rate")
    paths += plot_heatmaps(result.heatmaps, outdir, metric="expected_harm")
    paths.append(plot_phase_diagram(result.heatmaps, outdir))
    paths += plot_outcome_distributions(result.outcome_distribution, outdir)
    paths += plot_interactions(result.interactions, outdir)
    paths.append(plot_h1_margin(result.h1.table, outdir, result.h1.threshold_intervention))
    if ranking is not None:
        paths.append(plot_sensitivity(ranking, outdir))
    return paths


__all__ = [
    "plot_heatmaps",
    "plot_phase_diagram",
    "plot_outcome_distributions",
    "plot_interactions",
    "plot_h1_margin",
    "plot_sensitivity",
    "render_all",
]
