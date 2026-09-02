# Selected Public Artifacts

This package is a curated public release of the TOK research spine. It is not a
mirror of the full research workspace.

## Orientation

- `docs/PUBLIC_SCOPE.md`
- `docs/TOK_RIGOROUS_TIGHTENING_STAGE.md`
- `docs/TOK_DEVELOPMENT_FREEZE_v0.md`
- `docs/TOK_SMOKE_SUITE.md`

## Paper 1 / Paper 3 Benchmark Hygiene

- `docs/TOK_PAPER1_BENCHMARK_FREEZE_v0.md`
- `research/paper1_benchmark_freeze_v0/generated/paper1_public_summary_v0.json`

The public summary records benchmark results and claim boundaries. Private
ground-truth files, private maps, and private scorecards are intentionally
excluded.

## Paper 2 Dynamics Atlas

- `docs/TOK_PAPER2_DYNAMICS_FREEZE_v0.md`
- `docs/TOK_ATLAS_RESULTS_v1.md`
- `docs/TOK_BEHAVIORAL_PROBE_ATLAS_v0.md`
- `research/paper2_dynamics_freeze_v0/generated/paper2_dynamics_public_summary_v0.json`

This evidence supports claims about the frozen corpus, tested procedures,
ordinal ladder, admissibility atlas, collapse atlas, and behavioral
response-window calibration.

## Demo Freeze / Cockpit Replay

- `docs/TOK_DEMO_FREEZE_v0.md`
- `research/demo_freeze_v0/generated/demo_public_summary_v0.json`
- `research/demo_freeze_v0/generated/demo_session/reports/cockpit_trace_v0.json`
- `research/demo_freeze_v0/generated/demo_session/reports/evidence_transition_proposal_v0.json`
- `research/demo_freeze_v0/generated/demo_session/reports/causal_status_assessment_v0.json`
- `research/demo_freeze_v0/generated/demo_session/registry/research_observation_registry_v0.json`

The replay is research-only. It demonstrates validated artifacts and human-gated
status transitions, not autonomous evidence learning.

## Toy Reference Pipeline

- `examples/toy_reference_pipeline/README.md`
- `examples/toy_reference_pipeline/run_toy_reference_pipeline.py`
- `examples/toy_reference_pipeline/generated/toy_reference_run_v0.json`

This public toy pipeline is not the private TOK engine. It is a compact,
synthetic reference path that demonstrates the sequence from narrated situation
to candidate mechanism graph, dynamic trace, reviewed observation registry,
evidence-transition proposal, and causal-status assessment while preserving
`candidate_not_evidence` and `may_update_evidence: false`.

## Public Boundary

Included artifacts may mention private partitions by summary or hash, but this
package must not include private benchmark answer keys or `private_ground_truth`
directories.

Detailed private-workspace runbooks are intentionally not included. The public
Paper 1 summary is the citable release surface for benchmark results.
