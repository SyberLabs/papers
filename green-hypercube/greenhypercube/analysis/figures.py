"""Matplotlib/seaborn figures for the strategy comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(style="whitegrid", context="talk")


def plot_discovery_curves(curves: dict[str, np.ndarray], n_useful: int, out: Path) -> Path:
    """Mean discovery curves: useful species found vs. experiments spent."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for name, curve in curves.items():
        ax.plot(np.arange(1, len(curve) + 1), curve, label=name, linewidth=2.2)
    if n_useful > 0:
        ax.axhline(n_useful, color="0.4", linestyle=":", linewidth=1.5,
                   label=f"all useful (n={n_useful})")
    ax.set_xlabel("experiments (costly trials)")
    ax.set_ylabel("useful species discovered")
    ax.set_title("Searching the Green Hypercube: discovery efficiency")
    ax.legend(loc="upper left", fontsize=11, frameon=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_summary_bars(summary: pd.DataFrame, out: Path) -> Path:
    """Bar charts of AUDC and reward coverage with replicate spread."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    order = (
        summary.groupby("strategy")["audc"].mean().sort_values(ascending=False).index
    )
    sns.barplot(data=summary, x="audc", y="strategy", order=order, ax=axes[0],
                errorbar="sd", hue="strategy", legend=False)
    axes[0].set_title("Area under discovery curve")
    axes[0].set_xlabel("AUDC (1.0 = instant)")
    axes[0].set_ylabel("")
    sns.barplot(data=summary, x="reward_coverage", y="strategy", order=order, ax=axes[1],
                errorbar="sd", hue="strategy", legend=False)
    axes[1].set_title("Reward-mass coverage")
    axes[1].set_xlabel("fraction of total utility discovered")
    axes[1].set_ylabel("")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_null_controls(per_replicate: pd.DataFrame, out: Path) -> Path:
    """AUDC per strategy under the real landscape vs. each negative control.

    A valid result shows each structured strategy high under 'real' and dropping
    to the random baseline under the nulls (especially 'permute_reward').
    """
    fig, ax = plt.subplots(figsize=(12, 6.5))
    order = (
        per_replicate[per_replicate["condition"] == "real"]
        .groupby("strategy")["audc"].mean().sort_values(ascending=False).index.tolist()
    )
    cond_order = ["real"] + [c for c in per_replicate["condition"].unique() if c != "real"]
    sns.barplot(
        data=per_replicate, x="strategy", y="audc", hue="condition",
        order=order, hue_order=cond_order, errorbar="sd", ax=ax,
    )
    ax.set_title("Negative controls: does the advantage survive structure ablation?")
    ax.set_xlabel("")
    ax.set_ylabel("AUDC")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="condition", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_coupling(coupling: pd.DataFrame, out: Path) -> Path:
    """Measured cue->reward coupling per channel, against the permutation null.

    Each bar is the observed Spearman coupling; the shaded band is the
    label-permutation null's +/-2 sigma envelope. Bars outside the band carry
    real, exploitable signal; bars inside it are statistically indistinguishable
    from no coupling (the anti-tautology check on real data).
    """
    df = coupling.copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(df))
    band = 2.0 * df["null_std"].to_numpy()
    ax.bar(x, df["observed"], color=np.where(df["p_value"] < 0.05, "#2a7", "#bbb"),
           edgecolor="0.3", zorder=3)
    ax.errorbar(x, df["null_mean"], yerr=band, fmt="none", ecolor="0.4",
                elinewidth=8, alpha=0.35, zorder=2, label="null +/-2 sigma")
    for xi, (_, r) in zip(x, df.iterrows()):
        ax.annotate(f"p={r['p_value']:.3f}", (xi, r["observed"]),
                    ha="center", va="bottom" if r["observed"] >= 0 else "top",
                    fontsize=10)
    ax.axhline(0, color="0.3", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["channel"])
    ax.set_ylabel("Spearman coupling (reward vs. cue)")
    ax.set_title("Measured cue->reward coupling (real data)")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_multivariate_coupling(tbl: pd.DataFrame, out: Path) -> Path:
    """Partial R² for joint channel coupling vs permutation null."""
    row = tbl.iloc[0]
    fig, ax = plt.subplots(figsize=(6, 5))
    obs = row["observed"]
    band = 2.0 * row["null_std"]
    color = "#2a7" if row["p_value"] < 0.05 else "#bbb"
    ax.bar([0], [obs], color=color, edgecolor="0.3", width=0.5, zorder=3)
    ax.errorbar([0], [row["null_mean"]], yerr=band, fmt="none", ecolor="0.4",
                elinewidth=12, alpha=0.35, zorder=2, label="null +/-2 sigma")
    ax.annotate(f"p={row['p_value']:.3f}", (0, obs),
                ha="center", va="bottom", fontsize=11)
    ax.set_xticks([0])
    ax.set_xticklabels(["sensory+phylo+eco+bio"])
    ax.set_ylabel("Partial R² (rank-based)")
    ax.set_title("Multivariate cue→reward coupling")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_m2_advantage(advantage: pd.DataFrame, out: Path) -> Path:
    """Grouped bar chart of strategy advantage across M2 ladder conditions."""
    focus = advantage[
        advantage["strategy"].isin(["ecological", "cultural", "sensory", "random", "degree_random"])
    ].copy()
    fig, ax = plt.subplots(figsize=(11, 6))
    order = ["real", "permute_reward", "permute_reward_within_effort", "permute_features"]
    focus["condition"] = pd.Categorical(focus["condition"], categories=order, ordered=True)
    sns.barplot(
        data=focus, x="condition", y="advantage_over_random", hue="strategy", ax=ax,
    )
    ax.axhline(0, color="0.3", linewidth=1)
    ax.set_ylabel("AUDC advantage over uniform random")
    ax.set_xlabel("landscape condition")
    ax.set_title("M2 ladder: where structured-search advantage lives")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(fontsize=9, title="strategy")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_sensitivity(df: pd.DataFrame, x: str, out: Path, ylabel: str = "AUDC") -> Path:
    """Line plot of a metric vs. a swept parameter, one line per strategy."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    sns.lineplot(data=df, x=x, y="audc", hue="strategy", marker="o", ax=ax, errorbar="sd")
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Sensitivity of discovery efficiency to {x}")
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _advantage_over_random(df: pd.DataFrame) -> pd.DataFrame:
    """Mean AUDC advantage over random by (landscape, reward_top_frac, strategy)."""
    rows: list[dict] = []
    landscape = df["landscape"].iloc[0] if "landscape" in df.columns else "default"
    if "landscape" not in df.columns:
        df = df.copy()
        df["landscape"] = landscape
    for (land, tf), grp in df.groupby(["landscape", "reward_top_frac"]):
        rand = grp.loc[grp["strategy"] == "random", "audc"].mean()
        n_useful = int(grp["n_total_useful"].iloc[0])
        for strategy, sg in grp.groupby("strategy"):
            if strategy == "random":
                continue
            rows.append({
                "landscape": land,
                "reward_top_frac": tf,
                "n_useful": n_useful,
                "strategy": strategy,
                "advantage": float(sg["audc"].mean() - rand),
            })
    return pd.DataFrame(rows)


def plot_matched_density_advantage(
    sweep_paths: dict[str, Path | str],
    out: Path,
    strategies: tuple[str, ...] = ("sensory", "cultural", "ecological"),
) -> Path:
    """Compare advantage-over-random vs reward_top_frac across NAEB landscapes.

    ``sweep_paths`` maps landscape label -> ``sweep_reward_top_frac.csv`` path.
    """
    parts = []
    for label, path in sweep_paths.items():
        df = pd.read_csv(path)
        df = df.copy()
        df["landscape"] = label
        parts.append(df)
    combined = pd.concat(parts, ignore_index=True)
    adv = _advantage_over_random(combined)
    adv = adv[adv["strategy"].isin(strategies)]

    fig, axes = plt.subplots(1, len(strategies), figsize=(5 * len(strategies), 5), sharey=True)
    if len(strategies) == 1:
        axes = [axes]
    for ax, strategy in zip(axes, strategies):
        sub = adv[adv["strategy"] == strategy]
        sns.lineplot(
            data=sub, x="reward_top_frac", y="advantage", hue="landscape",
            marker="o", ax=ax, linewidth=2.2,
        )
        ax.set_title(strategy.capitalize())
        ax.set_xlabel("reward_top_frac")
        ax.set_ylabel("advantage over random (AUDC)")
        ax.axhline(0, color="0.5", linewidth=0.8, linestyle="--")
    fig.suptitle("Matched-density strategy advantage: enriched vs full flora", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
