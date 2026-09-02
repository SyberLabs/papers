# TOK Rigorous Tightening Stage

Date: 2026-06-18

Status: planning checkpoint for interactive software and academic paper preparation.
Production effect: `none_research_only`.

## Executive Diagnosis

TOK has crossed from speculative architecture into a research program with
multiple gated artifacts:

- A regime-complete 12-system dynamics atlas.
- A naturalistic causal-coordinate holdout with private scoring and Error Atlas.
- A revised boundary-note mini-holdout with a completed blinded classifier run
  and private scorecard.
- A guarded research CLI that can replay graph, dynamics, observation, registry,
  evidence, causal-status, triage, and synthetic recovery flows.
- A growing theory layer around source-bounded mechanism edges, dynamic
  divergence, and epistemic state transitions.

The next risk is not lack of ideas. The risk is dilution: more gates, more
specifications, and more generated artifacts without a single hardened
interactive workflow or a paper-grade claim boundary.

The rigorous tightening stage should therefore have two outputs:

1. **Interactive Software:** a governed research cockpit that lets a human
   replay the TOK pipeline end to end without touching private benchmarks or
   production memory.
2. **Academic Papers:** a small paper stack with frozen claims, methods,
   datasets, negative results, and reproducible scripts.

## Current State

### Green Assets

1. **Research CLI passes its invariant suite**

   The CLI can run the canonical boiler workflow end to end and maintains
   research-only boundaries. Verified from the intended module directory with
   the project virtual environment.

2. **Dynamics corpus completeness gate passes**

   The full 12-system, 7-regime corpus has complete static touchpoints,
   generators, libraries, rollout paths, fit execution, and regime coverage.

3. **Atlas results are result-of-record quality**

   `TOK_ATLAS_RESULTS_v1.md` records a regime-complete run:

   - 116,640 admissibility fits.
   - 144 collapse transects.
   - 12 systems across 7 regimes.
   - all acceptance gates passed.

4. **Naturalistic holdout and Error Atlas discipline are mature**

   The project now distinguishes:

   - classifier misses,
   - authored ambiguities,
   - entangled coordinates,
   - compositional boundary cases,
   - follow-up rows.

   This is publishable methodology if framed carefully as benchmark hygiene,
   not as evidence that TOK is already correct in the world.

5. **Boundary-note mini-holdout Stage B is now scored**

   Stage A replacements are complete, revised, audited, and human-approved.
   The blinded Stage B packet was classified and privately scored:

   - overall exact coordinate accuracy: `69/83 = 0.8313`;
   - overall axis decision accuracy: `315/332 = 0.9488`;
   - replacement lane: `39/39 = 1.0000`;
   - clean-control lane: `28/39 = 0.7179`;
   - classifier-follow-up lane: `2/5 = 0.4000`.

   This is a strong repair signal for Error Atlas boundary-note rewrites, with
   residual errors concentrated in controls and follow-up cases.

### Yellow Assets

1. **Interactive frontend exists but is not hardened**

   The Vite/React frontend is present and connected to Flask endpoints, but the
   build/lint surface is not green:

   - `npm run lint` currently fails on 24 `no-explicit-any` violations.
   - `npm run build` transforms modules but fails while clearing `dist` with
     `EPERM`, and reports a Node version warning: Vite wants Node `20.19+` or
     `22.12+`, while the current environment reports `20.18.0`.

   These are product-hardening issues, not research invalidations.

2. **Python packaging remains mixed, but root smoke discovery is unified**

   Many research tests are still runnable as standalone modules. A root smoke
   suite now normalizes those invocation differences for the safe local research
   checks. Full packaging cleanup can wait until the research cockpit needs
   stable import boundaries.

3. **Generated artifacts are numerous**

   The repo contains many generated packets, zips, JSON bundles, and HPC outputs.
   This is normal for the research phase, but it complicates version control,
   review, and paper reproduction.

4. **Production Clarity Engine and research TOK are partially braided**

   `clarityengine_main.py` still carries production-ish endpoints, wisdom-db
   mutation paths, structural priors, confirmation routes, and graph endpoints.
   The research CLI has stronger boundaries than the main app surface.

### Red Risks

1. **Overclaim risk**

   The atlas supports a strong claim about the tested implementations and
   frozen ladder definitions. It does not prove that LTCs or LNNs cannot model
   dynamics in general, nor that TOK has discovered real-world causal laws.

2. **Benchmark leakage risk**

   The holdout work has good blinding discipline now, but future interactive
   demos must never expose private maps, source IDs, lane membership, or answer
   keys to a classifier or UI agent.

3. **Evidence mutation risk**

   The best architectural discipline in the repo is the repeated use of
   `candidate_not_evidence`, `none_research_only`, and explicit human gates.
   The production app must not bypass these controls by writing to `wisdom_db`
   from a research artifact.

4. **Narrative coherence risk**

   The system has several valid research threads. A paper or demo that tries to
   present all of them as one monolithic breakthrough will be weaker than a
   paper that names one claim and tests it rigorously.

## Tightening Principle

Stop expanding the ontology until the existing architecture can repeatedly
answer this sequence:

```text
source material
-> provisional mechanism graph
-> explicit exclusions and withheld inferences
-> optional dynamics-template inspection
-> reviewed observations
-> dynamic divergence or evidence-transition proposal
-> human review decision
```

Every stage must expose what it is allowed to claim and what it is forbidden to
claim.

## Track A: Interactive Software

### Target

Build a local research cockpit over the existing CLI, not a new production app.

The cockpit should let a human run and inspect the canonical TOK workflow:

1. Create or load a session.
2. Enter natural language or attach CSV observations.
3. Create a provisional HLMG graph.
4. Browse graph nodes, edges, exclusions, and source grounding.
5. Inspect candidate dynamics-template bindings.
6. Run implemented shadow dynamics only when parameters are human supplied.
7. Attach reviewed observations.
8. Inspect DYNDIV residuals and divergence triggers.
9. Register observations as seen data, not evidence.
10. Generate evidence-transition and causal-status proposals.

### Immediate Software Gates

1. **Define a green baseline command**

   Status: complete for the Python research spine.

   The single documented smoke command is:

   ```powershell
   .venv\Scripts\python.exe -B tools\run_tok_smoke_suite.py
   ```

   It runs:

   - research CLI invariant suite,
   - dynamics corpus completeness gate,
   - atlas small sweeps,
   - naturalistic holdout scorer tests,
   - naturalistic Error Atlas derivative tests,
   - boundary-note mini-holdout tests,
   - Paper 1 benchmark freeze manifest validation,
   - Paper 2 dynamics atlas reproduction freeze validation,
   - Demo Freeze v0 cockpit replay validation.

   Heavy HPC scripts, networked LLM calls, large-corpus generation, frontend
   build/lint, and production-memory mutation remain opt-in. See
   `docs/TOK_SMOKE_SUITE.md`.

2. **Harden frontend build**

   - Fix TypeScript `any` violations or explicitly type research payloads.
   - Decide whether `dist/` is source-controlled or generated-only.
   - Update Node to a Vite-compatible patch version or pin Vite to the local
     Node version.
   - Add a frontend smoke path: build, lint, and one mocked API render.

3. **Freeze a CLI-first cockpit replay**

   Status: complete for Demo Freeze v0.

   The first cockpit artifact is not the frontend. It is a deterministic local
   session replay:

   ```text
   research/demo_freeze_v0/generated/demo_freeze_manifest_v0.json
   research/demo_freeze_v0/generated/demo_session/reports/cockpit_trace_v0.json
   ```

   This gives the demo and future UI a stable flight recorder: graph, binding,
   DYNDIV, observations, registry, evidence proposal, causal status, boundary
   breach, triage, bridge, scout, and workbench artifacts are all present in one
   session tree with explicit `none_research_only` boundaries.

4. **Wrap CLI commands behind a session API**

   Do not call raw research functions from the frontend. Expose session-level
   endpoints:

   - `GET /api/research/session/:id/status`
   - `POST /api/research/session`
   - `POST /api/research/session/:id/graph`
   - `POST /api/research/session/:id/binding-suggestions`
   - `POST /api/research/session/:id/dyndiv`
   - `POST /api/research/session/:id/observations`
   - `POST /api/research/session/:id/evidence-proposal`

   Each endpoint returns an artifact path, validation result, and boundary
   statement.

5. **Separate production memory from research artifacts**

   The interactive cockpit should write only to session folders under
   `research/tok_research_cli_v0/sessions/` or another explicit sandbox root.
   It should not write to `wisdom_db` unless a separate production promotion
   protocol is built later.

6. **Make private benchmark workspaces non-interactive**

   The UI should not browse private ground-truth folders by default. Paper
   reproduction scripts can read them; the interactive classifier/demo cockpit
   should not.

### Software Definition Of Done

The interactive software stage is complete when a new user can:

```text
start server
open cockpit
create session
submit one problem
inspect provisional graph and exclusions
inspect candidate binding
attach sample CSV
view DYNDIV residuals
register observations
produce evidence proposal
see why no production evidence changed
```

with a green smoke suite and no private benchmark leakage.

## Track B: Academic Papers

### Paper 1: Governed Mechanism Inference

Core claim:

> LLM-assisted systems reasoning becomes more reliable when outputs are
> represented as source-bounded mechanism edges with explicit forbidden
> inferences, rather than as global labels or automatic causal claims.

Evidence base:

- response-event grammar gates,
- mechanism edge-set fidelity gate,
- source-raised hypothesis discipline,
- naturalistic holdout Error Atlas,
- boundary-note mini-holdout result once Stage B is scored.

What this paper must not claim:

- that the extracted graph is causal truth,
- that the system can recommend interventions autonomously,
- that benchmark labels are real-world evidence.

Tightening needed:

- Use the completed mini-holdout Stage B scorecard.
- Report replacement/control/follow-up performance separately.
- Add a small ablation: classifier with boundary notes versus without boundary
  notes, or original ambiguous rows versus revised rows.
- Freeze all prompts, packets, maps, scoring scripts, and exclusion criteria.

### Paper 2: Structural Recoverability Atlas

Core claim:

> Under a frozen ordinal ladder for sparse topological preservation, recoverable
> structure is sharply regime- and observability-dependent; dense continuous-time
> fits tested here do not clear the sparse topology threshold across the
> 12-system corpus.

Evidence base:

- admissibility atlas v1,
- collapse atlas v1,
- behavioral probe window calibration,
- full 12-system dynamics corpus.

What this paper must not claim:

- that LTC/LNN methods are broadly invalid,
- that one implementation exhausts the model class,
- that synthetic recoverability proves empirical causal validity.

Tightening status:

- Reproduction freeze manifest exists:
  `research/paper2_dynamics_freeze_v0/generated/paper2_dynamics_freeze_manifest_v0.json`.
- Result-of-record bundles, source scripts, tests, runbooks, docs, figures, and
  minimal requirements are hash-recorded.
- Add a method appendix defining every ladder level.
- Run one negative-control or robustness variant: e.g. `ltc_rk4` as a fourth
  procedure, not a replacement for the existing LTC result.

### Paper 3: Boundary Hygiene For Synthetic Causal Benchmarks

Core claim:

> Natural-language causal-coordinate benchmarks require post-authoring semantic
> leakage audits and human Error Atlas adjudication; raw accuracy alone is
> misleading because authored ambiguity, coordinate entanglement, and classifier
> misses are different error types.

Evidence base:

- coordinate workpacket,
- naturalistic holdout,
- semantic leakage audit,
- private Error Atlas,
- accepted derivative packets,
- boundary-note mini-holdout.

What this paper must not claim:

- that the taxonomy is final,
- that synthetic labels are empirical laws,
- that a high score equals causal understanding.

Tightening needed:

- Use the completed mini-holdout Stage B scorecard.
- Report before/after effects of boundary-note rewrites.
- Include examples of rejected rows and why they were excluded.
- Release a sanitized public artifact with private keys removed.

### Paper 4: Interactive Epistemic Sandbox

Core claim:

> A research interface can make AI systems reasoning safer by forcing every
> transition through artifact validation, explicit status, and human review.

Evidence base:

- research CLI,
- session workflow,
- observation registry,
- DYNDIV,
- evidence transition gate,
- causal status ladder,
- future cockpit demo.

What this paper must not claim:

- production readiness,
- medical/legal/financial reliability,
- autonomous evidence learning.

Tightening needed:

- Build the cockpit MVP.
- Record a reproducible demo trace.
- Include all artifacts generated in the trace.
- Show that every artifact has `production_effect: none_research_only`.

## Recommended Sequence

### Gate 1: Finish The Mini-Holdout Measurement

Status: complete.

Stage B classification and private scoring are complete. Results:

```text
replacement: 39/39 exact
clean_control: 28/39 exact
classifier_followup: 2/5 exact
overall: 69/83 exact, 315/332 axis decisions
```

The follow-up action is to build a residual mini Error Atlas for:

- 11 clean-control errors,
- 3 classifier-follow-up errors,
- 0 replacement errors.

Status update: this residual mini Error Atlas now exists as both the raw private
queue and an accepted reviewed copy. The accepted dispositions are:

- `retain_fixture_classifier_miss`: 6 rows;
- `reauthor_before_clean_reuse`: 6 rows;
- `boundary_case_exclude_from_clean_score`: 2 rows.

### Gate 2: Create The Green Baseline

Status: Python research smoke complete.

The root-level smoke suite is:

```powershell
.venv\Scripts\python.exe -B tools\run_tok_smoke_suite.py
```

Current verified result:

```text
26 commands passed in 48.98s.
```

Still separate from this green baseline:

- frontend lint/build,
- private benchmark validation,
- artifact diff check.

No paper draft should be considered stable unless this smoke command is green
and any paper-specific private benchmark validation has also been run.

### Gate 3: Build The Research Cockpit MVP

Status: CLI-first replay freeze complete; frontend cockpit still pending.

Do not start with the full Clarity Engine UI. Start with a local cockpit around
session artifacts:

- session list,
- artifact graph,
- JSON inspector,
- graph viewer,
- DYNDIV residual chart,
- boundary/evidence status panel.

The first UI should be boring and inspectable. That is a feature.

The Demo Freeze v0 replay is now the canonical data fixture for this UI.

### Gate 4: Freeze Paper 1 Dataset

Status: complete.

The benchmark freeze package for Paper 1/3 is:

```text
research/paper1_benchmark_freeze_v0/generated/paper1_benchmark_freeze_manifest_v0.json
```

It includes:

- public blinded packet,
- private map hash,
- scoring script,
- Error Atlas derivative summary,
- audit reports,
- final result table.

### Gate 5: Freeze Dynamics Paper Bundle

Status: complete for manifest-first reproduction freeze.

The dynamics freeze package for Paper 2 is:

```text
research/paper2_dynamics_freeze_v0/generated/paper2_dynamics_freeze_manifest_v0.json
```

It records hashes for result-of-record bundles, dynamics corpus manifest,
behavioral calibration artifacts, HPC receipts, source scripts, tests, runbooks,
docs, figures, and minimal requirements. The next decision is whether to add
`ltc_rk4` or another fourth procedure as a robustness variant.

## Review Summary

The strongest current contribution is not "TOK has found universal laws." It is:

> TOK is becoming a governed epistemic architecture for moving from narrated
> situations to source-bounded mechanism hypotheses, audited dynamics
> candidates, and human-reviewed evidence transitions without collapsing
> candidates into truth.

That is serious. It is also narrower than the mythology that naturally grows
around the project. The tightening stage should protect the serious version.

## Next Action

Use the green research smoke suite as the baseline, then either adjudicate the
residual mini Error Atlas or begin the research cockpit MVP. Frontend hardening
remains the next software risk before the cockpit becomes demo-stable.
