# TOK Paper 1 Benchmark Freeze v0

Date: 2026-06-18

Status: frozen internal benchmark manifest for Paper 1 and Paper 3 preparation.
Production effect: `none_research_only`.

## Freeze Object

The freeze manifest is:

`research/paper1_benchmark_freeze_v0/generated/paper1_benchmark_freeze_manifest_v0.json`

It records hashes for public blinded packets, classifier outputs, scoring
scripts, private scoring maps, private scorecards, Error Atlas artifacts, and
paper-facing documentation.

Current manifest identity:

```text
artifact_count=60
public_bundle_hash=21ea78fa30e57da55a9ed90f981debb0b1732388ee79b40b215343f3e40ef14a
private_bundle_hash=71763440a5ed42df54493f35b2b3cdbefae5d6cb2c6d3e2ce5c46716d400dc43
bundle_hash=91343a4cff7d548516b393f81c9455759446c8fcaaaa0b1772dda74de72060c8
```

## Included Evidence

Primary benchmark:

- Naturalistic holdout Stage B blinded v1.
- 240 problems.
- Exact coordinate accuracy: `173/240 = 0.7208`.
- Axis decision accuracy: `885/960 = 0.9219`.
- Reviewed Error Atlas: 67 rows, 20 transition clusters, 6 review batches.
- Derivatives: 39 reauthoring rows, 23 compositional boundary rows, 5 classifier
  follow-up rows.

Repair / ablation benchmark:

- Boundary-note mini-holdout Stage B blinded v0.
- 83 problems.
- Exact coordinate accuracy: `69/83 = 0.8313`.
- Axis decision accuracy: `315/332 = 0.9488`.
- Replacement lane: `39/39 = 1.0000`.
- Clean-control lane: `28/39 = 0.7179`.
- Classifier-follow-up lane: `2/5 = 0.4000`.
- Residual mini Error Atlas: 14 rows, 9 transition clusters, 2 review batches.
- Reviewed residual adjudication accepted as `Mateo+Codex`:
  - `retain_fixture_classifier_miss`: 6 rows.
  - `reauthor_before_clean_reuse`: 6 rows.
  - `boundary_case_exclude_from_clean_score`: 2 rows.

## Boundary

This freeze supports claims about synthetic benchmark hygiene, semantic leakage
audits, direct semantic classification, Error Atlas adjudication, and
boundary-note repair effects.

It does not support claims of real-world causal truth, autonomous intervention
quality, automatic taxonomy revision, or production evidence updates.

Private artifacts are included by hash for internal reproducibility only. A
public release must exclude `private_ground_truth` files and use only the
manifest's public partitions.

## Commands

Build:

```powershell
.venv\Scripts\python.exe -B -m research.paper1_benchmark_freeze_v0.build_paper1_benchmark_freeze
```

Validate:

```powershell
.venv\Scripts\python.exe -B -m research.paper1_benchmark_freeze_v0.test_paper1_benchmark_freeze_v0
```

Root smoke:

```powershell
.venv\Scripts\python.exe -B tools\run_tok_smoke_suite.py
```
