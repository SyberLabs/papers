# TOK Demo Freeze v0

Date: 2026-06-18

Status: research-only cockpit replay freeze.
Production effect: `none_research_only`.

## Purpose

Demo Freeze v0 is the canonical local replay for showing TOK as an interactive
epistemic sandbox without turning candidate artifacts into truth.

It exercises this flow:

```text
source graph
-> candidate binding inspection
-> prior DYNDIV
-> reviewed CSV observations
-> seen-data registry
-> evidence-transition proposal
-> causal-status and boundary-breach artifacts
-> loose observation triage
-> triage-to-causal bridge
-> dynamic signal scout
-> data-driven workbench candidate
-> read-only cockpit trace
```

## Commands

Run from the repository root:

```powershell
.venv\Scripts\python.exe -B -m research.demo_freeze_v0.run_demo_freeze
.venv\Scripts\python.exe -B -m research.demo_freeze_v0.test_demo_freeze_v0
```

## Frozen Artifacts

Primary manifest:

```text
research/demo_freeze_v0/generated/demo_freeze_manifest_v0.json
```

Primary session:

```text
research/demo_freeze_v0/generated/demo_session/
```

Primary cockpit trace:

```text
research/demo_freeze_v0/generated/demo_session/reports/cockpit_trace_v0.json
```

## Claim Boundary

The demo supports a reproducible local replay of the research artifact pipeline.
It does not support production readiness, autonomous evidence transitions,
autonomous intervention recommendations, private benchmark browsing, or
`wisdom_db`/learning updates.

## Demo Standard

A demo is valid only if:

- every governed artifact validates;
- the registry preserves `not_evidence`;
- the evidence proposal has `update_effect.applied = false`;
- the cockpit trace reports zero invalid artifacts;
- all observed production effects remain `none_research_only`.

