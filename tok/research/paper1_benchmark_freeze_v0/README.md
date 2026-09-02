# Paper 1 Benchmark Freeze v0

Status: manifest-first freeze for Paper 1 / Paper 3 benchmark evidence.
Production effect: `none_research_only`.

This package freezes the benchmark artifacts used for:

- governed mechanism inference claims over blinded causal-coordinate packets;
- boundary hygiene claims over semantic leakage audits and Error Atlas review;
- boundary-note repair evidence from the mini-holdout.

The freeze manifest records file hashes and partitions. It does not create a
public release archive because private answer keys, maps, scorecards, and Error
Atlas rows must remain separate from blinded public inputs.

Build:

```powershell
.venv\Scripts\python.exe -B -m research.paper1_benchmark_freeze_v0.build_paper1_benchmark_freeze
```

Validate:

```powershell
.venv\Scripts\python.exe -B -m research.paper1_benchmark_freeze_v0.test_paper1_benchmark_freeze_v0
```
