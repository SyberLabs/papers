# TOK Smoke Suite

Date: 2026-06-18

Status: implemented local Python research baseline.
Production effect: `none_research_only`.

## Command

Run from the repository root:

```powershell
.venv\Scripts\python.exe -B tools\run_tok_smoke_suite.py
```

List included checks without running them:

```powershell
.venv\Scripts\python.exe -B tools\run_tok_smoke_suite.py --list
```

Stream child process output:

```powershell
.venv\Scripts\python.exe -B tools\run_tok_smoke_suite.py --verbose
```

## Scope

The suite is the green baseline for safe local research reproducibility. It
runs 26 checks covering:

- the research CLI invariant suite;
- dynamics corpus completeness;
- admissibility and collapse small harnesses;
- behavioral response-window calibration tests;
- causal status, observation, registry, triage, and bridge invariants;
- dynamic kernel, signal scout, data-driven dynamics, shadow ensemble, recovery,
  and sensitivity checks;
- Gemini coordinate and naturalistic holdout validators;
- naturalistic scorecard and Error Atlas derivative tests;
- boundary-note mini-holdout scoring and residual atlas tests;
- Paper 1 benchmark freeze manifest validation;
- Paper 2 dynamics atlas reproduction freeze validation;
- Demo Freeze v0 cockpit replay validation.

Last verified local result:

```text
SMOKE PASSED in 48.98s
```

## Exclusions

The default smoke suite intentionally excludes:

- HPC jobs and Slurm wrappers;
- networked LLM or API calls;
- large synthetic corpus generation;
- frontend lint/build;
- private benchmark export workflows;
- production-memory or `wisdom_db` mutation.

Those checks are important, but they are separate gates. The smoke suite should
stay fast, local, deterministic, and safe enough to run before paper edits or
research cockpit changes.
