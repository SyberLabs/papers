# TOK Development Freeze v0

Date: 2026-06-18

Status: Python research spine frozen for demo and paper preparation.
Production effect: `none_research_only`.

## Frozen Baseline

The development freeze baseline is:

```powershell
.venv\Scripts\python.exe -B tools\run_tok_smoke_suite.py
```

The smoke suite includes the Paper 1 benchmark freeze, Paper 2 dynamics freeze,
and Demo Freeze v0 cockpit replay. Heavy HPC jobs, networked LLM calls, private
release export workflows, frontend build/lint, and production-memory mutation
remain separate gates.

## What Is Frozen

- Paper 1/3 benchmark package manifest and score artifacts.
- Paper 2 dynamics atlas reproduction manifest.
- Demo Freeze v0 session replay and cockpit trace.
- Research-only CLI semantics around observations, evidence proposals, causal
  status, triage, and dynamic workbench candidates.

## What Is Not Frozen

- The frontend as a production app.
- The Clarity Engine production promotion pathway.
- Public benchmark release packaging.
- HPC inference configuration.
- Any autonomous evidence, learning, or intervention mechanism.

## Change Rule

After this freeze, changes that affect the research spine should either:

- update the relevant freeze manifest and rerun the smoke suite, or
- be explicitly marked as post-freeze experimental work.

No demo or paper draft should cite a changed artifact unless its hash appears in
the corresponding freeze manifest.

