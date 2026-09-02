"""
isomorphic.derived.trace_enrichment
-----------------------------------
Enrich raw grokking traces with lightweight proxy diagnostics.

These proxies are not mechanistic measurements. They are scaffold signals that
let the structured dataset carry:

- generalization gap
- validation velocity / acceleration
- cleanup progress inferred from L2 decay
- coarse algorithmic-strength and memorization-burden proxies
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def enrich_trace_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Add proxy diagnostic columns to a raw grokking trace."""
    required = {"epoch", "train_acc", "val_acc", "l2_norm"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Trace is missing required columns: {sorted(missing)}")

    enriched = frame.copy().sort_values("epoch").reset_index(drop=True)
    epoch_delta = enriched["epoch"].diff().replace(0, pd.NA)
    val_delta = enriched["val_acc"].diff()
    val_velocity = (val_delta / epoch_delta).fillna(0.0)
    val_acceleration = (val_velocity.diff() / epoch_delta).fillna(0.0)

    enriched["generalization_gap"] = (enriched["train_acc"] - enriched["val_acc"]).clip(lower=0.0)
    enriched["generalization_ratio"] = (
        enriched["val_acc"] / enriched["train_acc"].clip(lower=1e-6)
    ).clip(lower=0.0, upper=1.0)
    enriched["val_acc_velocity"] = val_velocity
    enriched["val_acc_acceleration"] = val_acceleration

    peak_l2 = enriched["l2_norm"].max()
    final_l2 = enriched["l2_norm"].iloc[-1]
    denom = max(peak_l2 - final_l2, 1e-9)
    cleanup_progress = (peak_l2 - enriched["l2_norm"]) / denom
    cleanup_progress = cleanup_progress.clip(lower=0.0, upper=1.0)

    enriched["cleanup_progress"] = cleanup_progress
    enriched["memorization_burden"] = (enriched["generalization_gap"] / 100.0).clip(0.0, 1.0)
    enriched["algorithmic_mode_mass"] = enriched["generalization_ratio"]
    enriched["dominant_mode_fraction"] = enriched["generalization_ratio"]
    enriched["phase_alignment"] = enriched["generalization_ratio"]

    return enriched


def main() -> None:
    """Enrich a raw trace CSV and write the result."""
    args = parse_args()
    frame = pd.read_csv(args.input)
    enriched = enrich_trace_dataframe(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output, index=False)
    print(f"Enriched trace written to {args.output}")


if __name__ == "__main__":
    main()
