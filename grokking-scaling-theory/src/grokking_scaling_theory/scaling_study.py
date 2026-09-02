"""
isomorphic.derived.scaling_study
--------------------------------
Structured dataset utilities for grokking scaling experiments.

The existing codebase stores grokking evidence as small in-module tables or as
single CSV traces. This module provides a stricter representation so the
research program can distinguish:

- experimental conditions
- per-run diagnostics
- observed vs censored grokking times
- spectral and memorization summaries

This module is deliberately lightweight and depends only on the standard
library plus NumPy so it can serve as the common substrate for later analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import csv

import numpy as np


@dataclass(frozen=True)
class ExperimentCondition:
    """Training condition for one grokking run."""

    modulus: int
    weight_decay: float
    learning_rate: float
    optimizer: str = "adamw"
    architecture: str = "mlp"
    width: Optional[int] = None
    depth: Optional[int] = None
    dataset_name: str = "modular_addition"
    task_name: str = "modular_addition"
    noise_level: float = 0.0


@dataclass(frozen=True)
class DiagnosticSnapshot:
    """Checkpointed observables during training."""

    epoch: int
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    train_acc: Optional[float] = None
    val_acc: Optional[float] = None
    l2_norm: Optional[float] = None
    algorithmic_mode_mass: Optional[float] = None
    effective_mode_count: Optional[float] = None
    dominant_mode_fraction: Optional[float] = None
    memorization_burden: Optional[float] = None
    cleanup_progress: Optional[float] = None
    phase_alignment: Optional[float] = None
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class GrokkingRun:
    """
    One training run with optional checkpointed diagnostics.

    A run is "censored" when grokking was not observed before the training
    budget ended. This is distinct from "never groks."
    """

    condition: ExperimentCondition
    seed: int
    max_epochs: int
    source: str = "unknown"
    diagnostics: List[DiagnosticSnapshot] = field(default_factory=list)
    grokking_threshold: float = 95.0
    observed_grokking_epoch: Optional[int] = None
    include_in_scaling_fit: bool = True
    notes: str = ""

    @property
    def is_censored(self) -> bool:
        """True when grokking was not observed within the training budget."""
        return self.grokking_epoch is None

    @property
    def grokking_epoch(self) -> Optional[int]:
        """Observed grokking epoch, if any."""
        if self.observed_grokking_epoch is not None:
            return self.observed_grokking_epoch

        for snapshot in sorted(self.diagnostics, key=lambda s: s.epoch):
            if snapshot.val_acc is not None and snapshot.val_acc >= self.grokking_threshold:
                return snapshot.epoch
        return None

    @property
    def last_epoch(self) -> int:
        """Last observed epoch for the run."""
        if not self.diagnostics:
            return self.max_epochs
        return max(snapshot.epoch for snapshot in self.diagnostics)

    def first_snapshot_passing(self, field_name: str, threshold: float) -> Optional[DiagnosticSnapshot]:
        """Return the first snapshot whose numeric field reaches threshold."""
        for snapshot in sorted(self.diagnostics, key=lambda s: s.epoch):
            value = getattr(snapshot, field_name, None)
            if value is not None and value >= threshold:
                return snapshot
        return None

    def to_record(self) -> Dict[str, object]:
        """Serialize the run into a flat dictionary suitable for CSV/JSON export."""
        return {
            "task_name": self.condition.task_name,
            "dataset_name": self.condition.dataset_name,
            "modulus": self.condition.modulus,
            "weight_decay": self.condition.weight_decay,
            "learning_rate": self.condition.learning_rate,
            "optimizer": self.condition.optimizer,
            "architecture": self.condition.architecture,
            "width": self.condition.width,
            "depth": self.condition.depth,
            "noise_level": self.condition.noise_level,
            "seed": self.seed,
            "max_epochs": self.max_epochs,
            "grokking_threshold": self.grokking_threshold,
            "grokking_epoch": self.grokking_epoch,
            "is_censored": self.is_censored,
            "include_in_scaling_fit": self.include_in_scaling_fit,
            "num_snapshots": len(self.diagnostics),
            "source": self.source,
            "notes": self.notes,
        }

    def diagnostic_summary(self) -> Dict[str, object]:
        """Summarize the most useful diagnostics available for the run."""
        summary: Dict[str, object] = {
            "seed": self.seed,
            "source": self.source,
            "grokking_epoch": self.grokking_epoch,
            "is_censored": self.is_censored,
            "num_snapshots": len(self.diagnostics),
        }
        if not self.diagnostics:
            return summary

        ordered = sorted(self.diagnostics, key=lambda snapshot: snapshot.epoch)
        summary["final_epoch"] = ordered[-1].epoch

        val_points = [(snapshot.epoch, snapshot.val_acc) for snapshot in ordered if snapshot.val_acc is not None]
        if val_points:
            best_epoch, best_val = max(val_points, key=lambda item: item[1])
            summary["best_val_acc"] = best_val
            summary["best_val_epoch"] = best_epoch
            summary["final_val_acc"] = val_points[-1][1]

        l2_points = [(snapshot.epoch, snapshot.l2_norm) for snapshot in ordered if snapshot.l2_norm is not None]
        if l2_points:
            peak_l2_epoch, peak_l2 = max(l2_points, key=lambda item: item[1])
            final_l2 = l2_points[-1][1]
            summary["peak_l2_norm"] = peak_l2
            summary["peak_l2_epoch"] = peak_l2_epoch
            summary["final_l2_norm"] = final_l2

            if peak_l2 > final_l2:
                cleanup_target = peak_l2 - 0.1 * (peak_l2 - final_l2)
                cleanup_epoch = next(
                    (
                        epoch for epoch, value in l2_points
                        if epoch >= peak_l2_epoch and value <= cleanup_target
                    ),
                    None,
                )
                summary["cleanup_onset_epoch"] = cleanup_epoch

        mode_mass_points = [
            (snapshot.epoch, snapshot.algorithmic_mode_mass)
            for snapshot in ordered
            if snapshot.algorithmic_mode_mass is not None
        ]
        if mode_mass_points:
            best_mode_epoch, best_mode = max(mode_mass_points, key=lambda item: item[1])
            summary["peak_algorithmic_mode_mass"] = best_mode
            summary["peak_algorithmic_mode_epoch"] = best_mode_epoch

        mode_count_points = [
            snapshot.effective_mode_count
            for snapshot in ordered
            if snapshot.effective_mode_count is not None
        ]
        if mode_count_points:
            summary["final_effective_mode_count"] = mode_count_points[-1]

        memorization_points = [
            snapshot.memorization_burden
            for snapshot in ordered
            if snapshot.memorization_burden is not None
        ]
        if memorization_points:
            summary["final_memorization_burden"] = memorization_points[-1]

        if ordered and "val_acc_velocity" in ordered[0].metadata:
            velocity_points = [
                (snapshot.epoch, snapshot.metadata["val_acc_velocity"])
                for snapshot in ordered
                if "val_acc_velocity" in snapshot.metadata
            ]
            if velocity_points:
                peak_velocity_epoch, peak_velocity = max(velocity_points, key=lambda item: item[1])
                summary["peak_val_acc_velocity"] = peak_velocity
                summary["peak_val_acc_velocity_epoch"] = peak_velocity_epoch

        if ordered and "val_acc_acceleration" in ordered[0].metadata:
            acceleration_points = [
                (snapshot.epoch, snapshot.metadata["val_acc_acceleration"])
                for snapshot in ordered
                if "val_acc_acceleration" in snapshot.metadata
            ]
            if acceleration_points:
                peak_acc_epoch, peak_acc = max(acceleration_points, key=lambda item: item[1])
                summary["peak_val_acc_acceleration"] = peak_acc
                summary["peak_val_acc_acceleration_epoch"] = peak_acc_epoch

        return summary

    @classmethod
    def from_trace_csv(
        cls,
        csv_path: str | Path,
        condition: ExperimentCondition,
        seed: int = 0,
        max_epochs: Optional[int] = None,
        source: str = "csv_trace",
        grokking_threshold: float = 95.0,
    ) -> "GrokkingRun":
        """
        Build a run from a CSV containing epoch-aligned diagnostics.

        The CSV may include any subset of the DiagnosticSnapshot fields. Unknown
        columns are stored in the snapshot metadata when numeric.
        """
        path = Path(csv_path)
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            diagnostics = [cls._snapshot_from_row(row) for row in reader]

        inferred_max_epochs = max((snapshot.epoch for snapshot in diagnostics), default=0)
        return cls(
            condition=condition,
            seed=seed,
            max_epochs=max_epochs if max_epochs is not None else inferred_max_epochs,
            source=source,
            diagnostics=diagnostics,
            grokking_threshold=grokking_threshold,
        )

    @staticmethod
    def _snapshot_from_row(row: Dict[str, str]) -> DiagnosticSnapshot:
        """Parse one CSV row into a DiagnosticSnapshot."""
        numeric_fields = {
            "epoch",
            "train_loss",
            "val_loss",
            "train_acc",
            "val_acc",
            "l2_norm",
            "algorithmic_mode_mass",
            "effective_mode_count",
            "dominant_mode_fraction",
            "memorization_burden",
            "cleanup_progress",
            "phase_alignment",
        }

        parsed: Dict[str, Optional[float]] = {}
        metadata: Dict[str, float] = {}

        for key, raw_value in row.items():
            if raw_value is None or raw_value == "":
                continue

            if key in numeric_fields:
                value = float(raw_value)
                parsed[key] = int(value) if key == "epoch" else value
            else:
                try:
                    metadata[key] = float(raw_value)
                except ValueError:
                    continue

        return DiagnosticSnapshot(
            epoch=int(parsed.get("epoch", 0)),
            train_loss=parsed.get("train_loss"),
            val_loss=parsed.get("val_loss"),
            train_acc=parsed.get("train_acc"),
            val_acc=parsed.get("val_acc"),
            l2_norm=parsed.get("l2_norm"),
            algorithmic_mode_mass=parsed.get("algorithmic_mode_mass"),
            effective_mode_count=parsed.get("effective_mode_count"),
            dominant_mode_fraction=parsed.get("dominant_mode_fraction"),
            memorization_burden=parsed.get("memorization_burden"),
            cleanup_progress=parsed.get("cleanup_progress"),
            phase_alignment=parsed.get("phase_alignment"),
            metadata=metadata,
        )


@dataclass
class ScalingDataset:
    """Collection of grokking runs with convenience methods for analysis."""

    runs: List[GrokkingRun]

    def observed_runs(self) -> List[GrokkingRun]:
        """Runs with observed grokking time."""
        return [run for run in self.runs if not run.is_censored]

    def fit_runs(self) -> List[GrokkingRun]:
        """Observed runs currently included in scaling-law fits."""
        return [
            run for run in self.runs
            if run.include_in_scaling_fit and not run.is_censored
        ]

    def censored_runs(self) -> List[GrokkingRun]:
        """Runs without observed grokking time inside the budget."""
        return [run for run in self.runs if run.is_censored]

    def filter(
        self,
        *,
        task_name: Optional[str] = None,
        optimizer: Optional[str] = None,
        architecture: Optional[str] = None,
    ) -> "ScalingDataset":
        """Return a filtered dataset by common condition fields."""
        filtered = self.runs
        if task_name is not None:
            filtered = [run for run in filtered if run.condition.task_name == task_name]
        if optimizer is not None:
            filtered = [run for run in filtered if run.condition.optimizer == optimizer]
        if architecture is not None:
            filtered = [run for run in filtered if run.condition.architecture == architecture]
        return ScalingDataset(filtered)

    def summary(self) -> Dict[str, object]:
        """High-level dataset summary for diagnostics and sanity checks."""
        observed = self.observed_runs()
        fit_runs = self.fit_runs()
        moduli = sorted({run.condition.modulus for run in self.runs})
        wds = sorted({run.condition.weight_decay for run in self.runs})
        optimizers = sorted({run.condition.optimizer for run in self.runs})
        architectures = sorted({run.condition.architecture for run in self.runs})

        grokking_epochs = [run.grokking_epoch for run in observed if run.grokking_epoch is not None]

        return {
            "num_runs": len(self.runs),
            "num_observed": len(observed),
            "num_censored": len(self.censored_runs()),
            "num_fit_runs": len(fit_runs),
            "moduli": moduli,
            "weight_decays": wds,
            "optimizers": optimizers,
            "architectures": architectures,
            "mean_grokking_epoch": float(np.mean(grokking_epochs)) if grokking_epochs else None,
            "median_grokking_epoch": float(np.median(grokking_epochs)) if grokking_epochs else None,
        }

    def diagnostic_runs(self) -> List[GrokkingRun]:
        """Runs that include at least one checkpointed diagnostic snapshot."""
        return [run for run in self.runs if run.diagnostics]

    def diagnostic_summaries(self) -> List[Dict[str, object]]:
        """Summaries for runs carrying checkpointed diagnostics."""
        return [
            {
                "task_name": run.condition.task_name,
                "modulus": run.condition.modulus,
                "weight_decay": run.condition.weight_decay,
                "optimizer": run.condition.optimizer,
                "architecture": run.condition.architecture,
                **run.diagnostic_summary(),
            }
            for run in self.diagnostic_runs()
        ]

    @classmethod
    def from_runs(cls, runs: Iterable[GrokkingRun]) -> "ScalingDataset":
        """Build a dataset from any iterable of runs."""
        return cls(list(runs))

    @classmethod
    def from_run_table_csv(cls, csv_path: str | Path) -> "ScalingDataset":
        """
        Load a structured run table.

        Each row describes one run. If `trace_path` is provided, checkpointed
        diagnostics are loaded from that CSV and attached to the run.
        """
        path = Path(csv_path)
        runs: List[GrokkingRun] = []

        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                runs.append(_run_from_table_row(row, path.parent))

        return cls(runs)


def _run_from_table_row(row: Dict[str, str], base_dir: Path) -> GrokkingRun:
    """Parse one run-table row into a GrokkingRun."""
    condition = ExperimentCondition(
        modulus=_parse_required_int(row, "modulus"),
        weight_decay=_parse_required_float(row, "weight_decay"),
        learning_rate=_parse_required_float(row, "learning_rate"),
        optimizer=row.get("optimizer", "adamw") or "adamw",
        architecture=row.get("architecture", "mlp") or "mlp",
        width=_parse_optional_int(row.get("width")),
        depth=_parse_optional_int(row.get("depth")),
        dataset_name=row.get("dataset_name", "modular_addition") or "modular_addition",
        task_name=row.get("task_name", "modular_addition") or "modular_addition",
        noise_level=_parse_optional_float(row.get("noise_level"), default=0.0),
    )

    seed = _parse_optional_int(row.get("seed"), default=0)
    max_epochs = _parse_required_int(row, "max_epochs")
    grokking_threshold = _parse_optional_float(row.get("grokking_threshold"), default=95.0)
    include_in_scaling_fit = _parse_optional_bool(row.get("include_in_scaling_fit"), default=True)
    source = row.get("source", "run_table") or "run_table"
    notes = row.get("notes", "") or ""
    observed_grokking_epoch = _parse_optional_int(row.get("grokking_epoch"))

    trace_path = row.get("trace_path", "") or ""
    if trace_path:
        trace_file = Path(trace_path)
        if not trace_file.is_absolute():
            trace_file = (base_dir / trace_file).resolve()
        if not trace_file.exists():
            # Trace files are machine-local artifacts and may be absent from
            # the repository checkout. Fall back to the summary row rather
            # than failing the entire dataset load.
            return GrokkingRun(
                condition=condition,
                seed=seed,
                max_epochs=max_epochs,
                source=source,
                grokking_threshold=grokking_threshold,
                observed_grokking_epoch=observed_grokking_epoch,
                include_in_scaling_fit=include_in_scaling_fit,
                notes=(notes + " [trace file unavailable; summary row only]").strip(),
            )
        run = GrokkingRun.from_trace_csv(
            csv_path=trace_file,
            condition=condition,
            seed=seed,
            max_epochs=max_epochs,
            source=source,
            grokking_threshold=grokking_threshold,
        )
        run.notes = notes
        run.include_in_scaling_fit = include_in_scaling_fit
        if observed_grokking_epoch is not None:
            run.observed_grokking_epoch = observed_grokking_epoch
        return run

    return GrokkingRun(
        condition=condition,
        seed=seed,
        max_epochs=max_epochs,
        source=source,
        grokking_threshold=grokking_threshold,
        observed_grokking_epoch=observed_grokking_epoch,
        include_in_scaling_fit=include_in_scaling_fit,
        notes=notes,
    )


def _parse_required_int(row: Dict[str, str], key: str) -> int:
    """Parse a required integer field from a row."""
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required integer field: {key}")
    return int(float(value))


def _parse_required_float(row: Dict[str, str], key: str) -> float:
    """Parse a required float field from a row."""
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required float field: {key}")
    return float(value)


def _parse_optional_int(value: Optional[str], default: Optional[int] = None) -> Optional[int]:
    """Parse an optional integer field."""
    if value is None or value == "":
        return default
    return int(float(value))


def _parse_optional_float(value: Optional[str], default: Optional[float] = None) -> Optional[float]:
    """Parse an optional float field."""
    if value is None or value == "":
        return default
    return float(value)


def _parse_optional_bool(value: Optional[str], default: bool = True) -> bool:
    """Parse an optional boolean field from common string forms."""
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Could not parse boolean value: {value}")
