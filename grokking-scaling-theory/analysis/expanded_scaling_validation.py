"""
Expanded scaling validation suite for the grokking effective theory.

This script does four things:
1. Builds an expanded dataset from published run-table points plus trace-backed local runs.
2. Jointly optimizes the log exponent q and weight-decay exponent beta.
3. Runs leave-one-out cross-validation on the grouped scaling law.
4. Compares q profiles across MLP and residual architectures on local runs.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXAMPLES_DIR = ROOT / "examples"
FIGURES_DIR = ROOT / "analysis" / "figures"


@dataclass(frozen=True)
class ScalingDatum:
    label: str
    group: str
    architecture: str
    modulus: int
    weight_decay: float
    grokking_epoch: int
    source: str
    trace_path: str = ""
    notes: str = ""
    measured: bool = True


LOCAL_TRACE_SPECS = [
    {
        "label": "local_mlp_p59_wd1",
        "group": "local_mlp",
        "architecture": "mlp",
        "modulus": 59,
        "weight_decay": 1.0,
        "trace_path": EXAMPLES_DIR / "representational_crossing_p59_mlp_wd1_extended.csv",
    },
    {
        "label": "local_mlp_p59_wd2",
        "group": "local_mlp",
        "architecture": "mlp",
        "modulus": 59,
        "weight_decay": 2.0,
        "trace_path": EXAMPLES_DIR / "representational_crossing_p59_mlp_wd2.csv",
    },
    {
        "label": "local_mlp_p97_wd05",
        "group": "local_mlp",
        "architecture": "mlp",
        "modulus": 97,
        "weight_decay": 0.5,
        "trace_path": EXAMPLES_DIR / "representational_crossing_wd05.csv",
    },
    {
        "label": "local_mlp_p97_wd1",
        "group": "local_mlp",
        "architecture": "mlp",
        "modulus": 97,
        "weight_decay": 1.0,
        "trace_path": EXAMPLES_DIR / "representational_crossing_wd1.csv",
    },
    {
        "label": "local_mlp_p97_wd15",
        "group": "local_mlp",
        "architecture": "mlp",
        "modulus": 97,
        "weight_decay": 1.5,
        "trace_path": EXAMPLES_DIR / "representational_crossing_wd15.csv",
    },
    {
        "label": "local_mlp_p97_wd2",
        "group": "local_mlp",
        "architecture": "mlp",
        "modulus": 97,
        "weight_decay": 2.0,
        "trace_path": EXAMPLES_DIR / "representational_crossing_wd2.csv",
    },
    {
        "label": "local_mlp_p113_wd1",
        "group": "local_mlp",
        "architecture": "mlp",
        "modulus": 113,
        "weight_decay": 1.0,
        "trace_path": EXAMPLES_DIR / "representational_crossing_p113_mlp_wd1.csv",
    },
    {
        "label": "local_mlp_p113_wd2",
        "group": "local_mlp",
        "architecture": "mlp",
        "modulus": 113,
        "weight_decay": 2.0,
        "trace_path": EXAMPLES_DIR / "representational_crossing_p113_mlp_wd2.csv",
    },
    {
        "label": "local_residual_p59_wd1",
        "group": "local_residual",
        "architecture": "residual",
        "modulus": 59,
        "weight_decay": 1.0,
        "trace_path": EXAMPLES_DIR / "representational_crossing_p59_residual_wd1.csv",
    },
    {
        "label": "local_residual_p59_wd2",
        "group": "local_residual",
        "architecture": "residual",
        "modulus": 59,
        "weight_decay": 2.0,
        "trace_path": EXAMPLES_DIR / "representational_crossing_p59_residual_wd2.csv",
    },
    {
        "label": "local_residual_p97_wd1",
        "group": "local_residual",
        "architecture": "residual",
        "modulus": 97,
        "weight_decay": 1.0,
        "trace_path": EXAMPLES_DIR / "representational_crossing_p97_residual_wd1.csv",
    },
    {
        "label": "local_residual_p97_wd2",
        "group": "local_residual",
        "architecture": "residual",
        "modulus": 97,
        "weight_decay": 2.0,
        "trace_path": EXAMPLES_DIR / "representational_crossing_p97_residual_wd2.csv",
    },
    {
        "label": "local_residual_p113_wd1",
        "group": "local_residual",
        "architecture": "residual",
        "modulus": 113,
        "weight_decay": 1.0,
        "trace_path": EXAMPLES_DIR / "representational_crossing_p113_residual_wd1.csv",
    },
    {
        "label": "local_residual_p113_wd2",
        "group": "local_residual",
        "architecture": "residual",
        "modulus": 113,
        "weight_decay": 2.0,
        "trace_path": EXAMPLES_DIR / "representational_crossing_p113_residual_wd2.csv",
    },
]


def extract_grokking_epoch(trace_path: Path, threshold: float = 95.0) -> tuple[int, float]:
    frame = pd.read_csv(trace_path)
    if "epoch" not in frame.columns or "val_acc" not in frame.columns:
        raise ValueError(f"Trace missing required columns: {trace_path}")
    mask = frame["val_acc"] >= threshold
    if not bool(mask.any()):
        raise ValueError(f"Trace never reaches {threshold}% validation accuracy: {trace_path}")
    epoch = int(frame.loc[mask, "epoch"].iloc[0])
    max_val = float(frame["val_acc"].max())
    return epoch, max_val


def load_published_points() -> list[ScalingDatum]:
    csv_path = DATA_DIR / "empirical_scaling_runs.csv"
    rows: list[ScalingDatum] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            include = row.get("include_in_scaling_fit", "True").lower()
            if include in {"false", "0", "no"}:
                continue
            censored = row.get("is_censored", "False").lower()
            if censored in {"true", "1", "yes"}:
                continue
            tau = row.get("grokking_epoch", "").strip()
            if not tau:
                continue
            source = row.get("source", "published")
            rows.append(
                ScalingDatum(
                    label=f"published_{row['modulus']}_{row['weight_decay']}_{source}",
                    group="published_mlp",
                    architecture=row.get("architecture", "mlp") or "mlp",
                    modulus=int(float(row["modulus"])),
                    weight_decay=float(row["weight_decay"]),
                    grokking_epoch=int(float(tau)),
                    source=source,
                    trace_path=row.get("trace_path", ""),
                    notes=row.get("notes", ""),
                    measured=source != "extrapolated",
                )
            )
    return rows


def load_local_trace_points() -> list[ScalingDatum]:
    rows: list[ScalingDatum] = []
    for spec in LOCAL_TRACE_SPECS:
        trace_path = Path(spec["trace_path"])
        epoch, max_val = extract_grokking_epoch(trace_path)
        rows.append(
            ScalingDatum(
                label=spec["label"],
                group=spec["group"],
                architecture=spec["architecture"],
                modulus=spec["modulus"],
                weight_decay=spec["weight_decay"],
                grokking_epoch=epoch,
                source="trace",
                trace_path=str(trace_path.relative_to(ROOT)),
                notes=f"max_val_acc={max_val:.2f}",
                measured=True,
            )
        )
    return rows


def build_expanded_dataset() -> pd.DataFrame:
    data = load_published_points() + load_local_trace_points()
    frame = pd.DataFrame([asdict(row) for row in data]).sort_values(
        ["group", "architecture", "modulus", "weight_decay"]
    )
    return frame.reset_index(drop=True)


def fit_grouped_constants(frame: pd.DataFrame, q: float, beta: float) -> tuple[dict[str, float], np.ndarray]:
    transformed = (
        np.log(frame["grokking_epoch"].to_numpy(dtype=float))
        - 2.0 * np.log(frame["modulus"].to_numpy(dtype=float))
        + q * np.log(np.log(frame["modulus"].to_numpy(dtype=float)))
        + beta * np.log(frame["weight_decay"].to_numpy(dtype=float))
    )
    constants: dict[str, float] = {}
    preds = np.zeros(len(frame), dtype=float)
    for group, idx in frame.groupby("group").groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        log_c = float(np.mean(transformed[idx_arr]))
        constants[group] = float(np.exp(log_c))
        preds[idx_arr] = (
            np.exp(log_c)
            * frame.iloc[idx_arr]["modulus"].to_numpy(dtype=float) ** 2
            / (
                np.log(frame.iloc[idx_arr]["modulus"].to_numpy(dtype=float)) ** q
                * frame.iloc[idx_arr]["weight_decay"].to_numpy(dtype=float) ** beta
            )
        )
    return constants, preds


def evaluate_fit(frame: pd.DataFrame, q: float, beta: float) -> dict[str, float | dict[str, float]]:
    constants, preds = fit_grouped_constants(frame, q, beta)
    obs = frame["grokking_epoch"].to_numpy(dtype=float)
    log_obs = np.log(obs)
    log_preds = np.log(preds)
    rel = preds / obs - 1.0
    return {
        "q": q,
        "beta": beta,
        "constants": constants,
        "rmse_log": float(np.sqrt(np.mean((log_preds - log_obs) ** 2))),
        "mae_pct": float(np.mean(np.abs(rel)) * 100.0),
        "max_err_pct": float(np.max(np.abs(rel)) * 100.0),
        "cv_norm": float(np.std(obs / preds) / np.mean(obs / preds)),
        "preds": preds,
    }


def grid_search(
    frame: pd.DataFrame,
    q_values: np.ndarray,
    beta_values: np.ndarray,
) -> tuple[dict[str, float | dict[str, float]], np.ndarray]:
    surface = np.zeros((len(beta_values), len(q_values)), dtype=float)
    best: dict[str, float | dict[str, float]] | None = None
    best_score = float("inf")
    for i, beta in enumerate(beta_values):
        for j, q in enumerate(q_values):
            fit = evaluate_fit(frame, float(q), float(beta))
            score = float(fit["rmse_log"])
            surface[i, j] = score
            if score < best_score:
                best_score = score
                best = fit
    assert best is not None
    return best, surface


def two_stage_grid_search(frame: pd.DataFrame) -> tuple[dict[str, float | dict[str, float]], np.ndarray, np.ndarray, np.ndarray]:
    coarse_q = np.linspace(0.0, 3.0, 31)
    coarse_beta = np.linspace(0.30, 1.00, 36)
    coarse_best, _ = grid_search(frame, coarse_q, coarse_beta)

    q_center = float(coarse_best["q"])
    beta_center = float(coarse_best["beta"])
    fine_q = np.linspace(max(0.0, q_center - 0.25), min(3.0, q_center + 0.25), 31)
    fine_beta = np.linspace(max(0.30, beta_center - 0.12), min(1.00, beta_center + 0.12), 25)
    fine_best, fine_surface = grid_search(frame, fine_q, fine_beta)
    return fine_best, fine_surface, fine_q, fine_beta


def bootstrap_uncertainty(
    frame: pd.DataFrame,
    n_boot: int = 120,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for group, group_frame in frame.groupby("group"):
        parts.append(group_frame.reset_index(drop=True))
    records = []
    for _ in range(n_boot):
        sampled = []
        for group_frame in parts:
            idx = rng.integers(0, len(group_frame), size=len(group_frame))
            sampled.append(group_frame.iloc[idx].reset_index(drop=True))
        boot_frame = pd.concat(sampled, ignore_index=True)
        best, _, _, _ = two_stage_grid_search(boot_frame)
        records.append({"q": float(best["q"]), "beta": float(best["beta"])})
    return pd.DataFrame(records)


def leave_one_out_cv(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for idx in range(len(frame)):
        train = frame.drop(index=idx).reset_index(drop=True)
        test = frame.iloc[idx]
        best, _, _, _ = two_stage_grid_search(train)
        constants, _ = fit_grouped_constants(train, float(best["q"]), float(best["beta"]))
        group = test["group"]
        if group not in constants:
            raise ValueError(f"Missing calibration group during LOO: {group}")
        pred = (
            constants[group]
            * test["modulus"] ** 2
            / ((np.log(test["modulus"]) ** float(best["q"])) * (test["weight_decay"] ** float(best["beta"])))
        )
        err = (pred - test["grokking_epoch"]) / test["grokking_epoch"]
        rows.append(
            {
                "label": test["label"],
                "group": group,
                "architecture": test["architecture"],
                "modulus": int(test["modulus"]),
                "weight_decay": float(test["weight_decay"]),
                "tau_obs": float(test["grokking_epoch"]),
                "tau_pred": float(pred),
                "relative_error_pct": float(err * 100.0),
                "q_fit": float(best["q"]),
                "beta_fit": float(best["beta"]),
            }
        )
    return pd.DataFrame(rows)


def architecture_sweep(
    frame: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    local = frame[frame["group"].isin(["local_mlp", "local_residual"])].copy()
    for architecture, subset in local.groupby("architecture"):
        best, surface, q_values, _ = two_stage_grid_search(subset.reset_index(drop=True))
        q_profile = surface.min(axis=0)
        loo = leave_one_out_cv(subset.reset_index(drop=True))
        results[architecture] = {
            "best": best,
            "surface": surface,
            "q_profile": q_profile,
            "q_values": q_values,
            "loo": loo,
            "count": len(subset),
        }
    return results


def subset_summaries(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    subsets = {
        "published_all": frame[frame["group"] == "published_mlp"].reset_index(drop=True),
        "published_measured_only": frame[
            (frame["group"] == "published_mlp") & (frame["measured"])
        ].reset_index(drop=True),
        "local_mlp": frame[frame["group"] == "local_mlp"].reset_index(drop=True),
        "local_residual": frame[frame["group"] == "local_residual"].reset_index(drop=True),
    }
    results: dict[str, dict[str, float]] = {}
    for name, subset in subsets.items():
        best, _, _, _ = two_stage_grid_search(subset)
        loo = leave_one_out_cv(subset)
        results[name] = {
            "count": float(len(subset)),
            "q": float(best["q"]),
            "beta": float(best["beta"]),
            "mae_pct": float(best["mae_pct"]),
            "loo_mae_pct": float(loo["relative_error_pct"].abs().mean()),
        }
    return results


def write_results_markdown(
    output_path: Path,
    frame: pd.DataFrame,
    best: dict[str, float | dict[str, float]],
    bootstrap: pd.DataFrame,
    loo: pd.DataFrame,
    arch_results: dict[str, dict[str, object]],
    subset_results: dict[str, dict[str, float]],
) -> None:
    q_ci = np.quantile(bootstrap["q"], [0.025, 0.5, 0.975])
    beta_ci = np.quantile(bootstrap["beta"], [0.025, 0.5, 0.975])
    group_counts = frame.groupby("group").size().to_dict()
    measured_count = int(frame["measured"].sum())

    lines = [
        "# Expanded Scaling Validation Results",
        "",
        "## Dataset Expansion",
        "",
        f"- Total points: {len(frame)}",
        f"- Measured points: {measured_count}",
        f"- Group counts: {group_counts}",
        "- Published group retains earlier anchors, including the two extrapolated low-modulus points.",
        "- Local groups are trace-backed measurements extracted from actual grokking runs.",
        "",
        "## Joint (q, beta) Optimization",
        "",
        f"- Best q: {best['q']:.2f}",
        f"- Best beta: {best['beta']:.2f}",
        f"- In-sample log-RMSE: {best['rmse_log']:.4f}",
        f"- Mean absolute percent error: {best['mae_pct']:.2f}%",
        f"- Max absolute percent error: {best['max_err_pct']:.2f}%",
        f"- Normalized collapse CV: {best['cv_norm']:.4f}",
        "",
        "### Group Calibration Constants",
        "",
    ]
    for group, value in best["constants"].items():  # type: ignore[index]
        lines.append(f"- {group}: C = {value:.2f}")

    lines.extend(
        [
            "",
            "## Subset Fits",
            "",
            "| Subset | n | q | beta | In-sample MAE | LOO MAE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, result in subset_results.items():
        lines.append(
            f"| {name} | {int(result['count'])} | {result['q']:.2f} | {result['beta']:.2f} | "
            f"{result['mae_pct']:.2f}% | {result['loo_mae_pct']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Bootstrap Uncertainty (120 stratified resamples)",
            "",
            f"- q median [{q_ci[0]:.2f}, {q_ci[2]:.2f}] with median {q_ci[1]:.2f}",
            f"- beta median [{beta_ci[0]:.2f}, {beta_ci[2]:.2f}] with median {beta_ci[1]:.2f}",
            "",
            "## Leave-One-Out Cross-Validation",
            "",
            f"- Mean absolute percent error: {loo['relative_error_pct'].abs().mean():.2f}%",
            f"- Median fitted q across folds: {loo['q_fit'].median():.2f}",
            f"- Median fitted beta across folds: {loo['beta_fit'].median():.2f}",
            f"- q range across folds: [{loo['q_fit'].min():.2f}, {loo['q_fit'].max():.2f}]",
            f"- beta range across folds: [{loo['beta_fit'].min():.2f}, {loo['beta_fit'].max():.2f}]",
            "",
            "## Architecture Sweep",
            "",
        ]
    )
    for architecture, result in arch_results.items():
        best_arch = result["best"]
        loo_arch = result["loo"]
        lines.extend(
            [
                f"### {architecture}",
                "",
                f"- Points: {result['count']}",
                f"- Best q: {best_arch['q']:.2f}",
                f"- Best beta: {best_arch['beta']:.2f}",
                f"- In-sample MAE: {best_arch['mae_pct']:.2f}%",
                f"- LOO MAE: {loo_arch['relative_error_pct'].abs().mean():.2f}%",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "- The denser mixed dataset lets us test the exponent pair jointly rather than fixing beta by hand.",
            "- The pooled fit is intentionally harsh: if a single shared exponent pair does not survive the mixed dataset, that is evidence against naive universality.",
            "- The published MLP anchor set still prefers q near 2 with low error, but the local MLP and residual ladders do not yet exhibit the same asymptotic regime.",
            "- The architecture split should therefore be read as exploratory and currently points to regime dependence rather than established universality.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(
    q_values: np.ndarray,
    beta_values: np.ndarray,
    surface: np.ndarray,
    best: dict[str, float | dict[str, float]],
    arch_results: dict[str, dict[str, object]],
) -> None:
    if not HAS_MATPLOTLIB:
        return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(
        surface,
        origin="lower",
        aspect="auto",
        extent=[q_values[0], q_values[-1], beta_values[0], beta_values[-1]],
        cmap="viridis",
    )
    ax.scatter([best["q"]], [best["beta"]], color="white", edgecolor="black", s=60, zorder=3)
    ax.set_xlabel("q")
    ax.set_ylabel(r"$\beta$")
    ax.set_title("Grouped Log-RMSE Surface")
    fig.colorbar(im, ax=ax, label="log-RMSE")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "q_beta_heatmap.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for architecture, result in arch_results.items():
        profile = np.asarray(result["q_profile"], dtype=float)
        profile = profile - profile.min()
        ax.plot(result["q_values"], profile, label=architecture, linewidth=2)
        ax.axvline(float(result["best"]["q"]), linestyle="--", alpha=0.6)
    ax.set_xlabel("q")
    ax.set_ylabel(r"$\Delta$ log-RMSE")
    ax.set_title("Architecture-Specific q Profiles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "architecture_q_profiles.png", dpi=200)
    plt.close(fig)


def main() -> None:
    frame = build_expanded_dataset()
    expanded_csv = ROOT / "analysis" / "expanded_scaling_runs.csv"
    frame.to_csv(expanded_csv, index=False)

    best, surface, q_values, beta_values = two_stage_grid_search(frame)
    bootstrap = bootstrap_uncertainty(frame, n_boot=120, seed=0)
    loo = leave_one_out_cv(frame)
    arch_results = architecture_sweep(frame)
    subset_results = subset_summaries(frame)
    bootstrap.to_csv(ROOT / "analysis" / "bootstrap_q_beta.csv", index=False)
    loo.to_csv(ROOT / "analysis" / "loo_predictions.csv", index=False)

    make_figures(q_values, beta_values, surface, best, arch_results)
    write_results_markdown(
        ROOT / "analysis" / "EXPANDED_SCALING_VALIDATION.md",
        frame,
        best,
        bootstrap,
        loo,
        arch_results,
        subset_results,
    )

    print("Expanded dataset:", len(frame))
    print(f"Best q={best['q']:.2f}, beta={best['beta']:.2f}, MAE={best['mae_pct']:.2f}%")
    print(
        "LOO MAE:",
        f"{loo['relative_error_pct'].abs().mean():.2f}%",
        "| q median:",
        f"{loo['q_fit'].median():.2f}",
        "| beta median:",
        f"{loo['beta_fit'].median():.2f}",
    )
    for architecture, result in arch_results.items():
        best_arch = result["best"]
        loo_arch = result["loo"]
        print(
            f"{architecture}: q={best_arch['q']:.2f}, beta={best_arch['beta']:.2f}, "
            f"LOO_MAE={loo_arch['relative_error_pct'].abs().mean():.2f}%"
        )
    for name, result in subset_results.items():
        print(
            f"{name}: q={result['q']:.2f}, beta={result['beta']:.2f}, "
            f"LOO_MAE={result['loo_mae_pct']:.2f}%"
        )


if __name__ == "__main__":
    main()
