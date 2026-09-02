# Paper 2 Dynamics Freeze v0

Status: manifest-first freeze for the Structural Recoverability Atlas paper.
Production effect: `none_research_only`.

This package freezes the result-of-record dynamics artifacts for Paper 2:

- 12-system dynamics corpus manifest;
- admissibility atlas v1;
- collapse atlas v1;
- behavioral probe atlas and response-window calibration;
- response assay contract HPC artifact and receipts;
- source scripts, tests, runbooks, docs, figures, and minimal requirements.

Build:

```powershell
.venv\Scripts\python.exe -B -m research.paper2_dynamics_freeze_v0.build_paper2_dynamics_freeze
```

Validate:

```powershell
.venv\Scripts\python.exe -B -m research.paper2_dynamics_freeze_v0.test_paper2_dynamics_freeze_v0
```
