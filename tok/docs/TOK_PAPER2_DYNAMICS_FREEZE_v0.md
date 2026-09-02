# TOK Paper 2 Dynamics Freeze v0

Date: 2026-06-18

Status: frozen internal dynamics reproduction manifest for Paper 2.
Production effect: `none_research_only`.

## Freeze Object

The freeze manifest is:

`research/paper2_dynamics_freeze_v0/generated/paper2_dynamics_freeze_manifest_v0.json`

It records hashes for the corpus manifest, atlas result bundles, behavioral
probe bundles, HPC receipts, source scripts, tests, docs, figures, and minimal
runtime requirements.

Current manifest identity:

```text
artifact_count=42
result_bundle_hash=b2772a4dc0694153111dc4bd17104dc5ce467b19899d5d509b88adb5a17fd918
source_bundle_hash=2084b4ca84f2d06db6cfbbbef51632ebcc68347ec9766bef95712018be56b461
bundle_hash=f1d3dbbf051d87e298a53acc418e736b184ba631a56dfb645bf13c42d8008b4a
```

## Included Evidence

- Dynamics corpus: 12 systems, 7 regimes, no uncovered regimes.
- Admissibility atlas v1: `116,640` fits, all acceptance gates passed.
- Collapse atlas v1: `144` transects, all acceptance gates passed.
- Behavioral probe bundle v0: controlled perturbation assay, all acceptance
  gates passed.
- Behavioral response-window calibration v0: fixed-window and natural-timescale
  normalized transects, all acceptance gates passed.
- Response assay contract HPC artifact `1465448`: provenance receipt retained,
  all acceptance gates passed.

## Boundary

This freeze supports claims about the tested implementations, corpus, ordinal
ladder, recoverability grid, collapse transects, and behavioral response-window
sensitivity.

It does not support broad invalidation of LTC/LNN model classes, empirical
causal validity, autonomous intervention policy, or production learning/evidence
updates.

## Commands

Build:

```powershell
.venv\Scripts\python.exe -B -m research.paper2_dynamics_freeze_v0.build_paper2_dynamics_freeze
```

Validate:

```powershell
.venv\Scripts\python.exe -B -m research.paper2_dynamics_freeze_v0.test_paper2_dynamics_freeze_v0
```

Root smoke:

```powershell
.venv\Scripts\python.exe -B tools\run_tok_smoke_suite.py
```
