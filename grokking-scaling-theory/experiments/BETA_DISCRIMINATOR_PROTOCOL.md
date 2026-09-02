# Protocol: Beta Discriminator, Seed Variance, and the p^-2 Degeneracy

Pre-registered experimental protocol, 2026-07. Companion documents:
`paper/BETA_BOTTLENECK_SECTION.md`, `analysis/SURVIVAL_FIT_RESULTS.md`.

These three experiments share one training harness and resolve the three
open questions created by the July revision: (P3) where sublinear beta
lives, (S) whether the transition zone is a fluctuation-enhanced basin
boundary, and (N) whether the zone's p^-2 scaling is modulus scaling or
dataset-size scaling in disguise.

## Schema Additions (required before any run)

Add to the run-table schema and record for every new run:

- `train_fraction` (float): fraction of the p^2 pair table used for training
- `batch_size` (int) and `steps_per_epoch` (int): so tau is convertible
  between epochs and optimizer steps; a fixed batch size makes
  steps_per_epoch grow like p^2 and silently changes what "epoch scaling"
  means
- `budget_epochs` (int): the true training budget, distinct from
  `grokking_epoch`, so censoring is detectable structurally rather than
  from notes (published anchors currently encode max_epochs ==
  grokking_epoch, which forced notes-based censoring detection in
  `survival_fit.py`)

## Experiment 1 (P3): Where Does Sublinear Beta Live?

**Question.** Measured beta_eff in MLPs is 0.6-0.9. Two accounts within
(M, R, D):

- H1 (crossover): microscopic beta = 1; sublinearity is stopping-time
  geometry. Predicts beta_eff drifts across a wide wd ladder and tracks
  the trace-identified bottleneck; beta_eff -> 1 where cleanup binds.
- H2 (fractional): heterogeneous dissipation spectrum yields a genuine
  fractional exponent. Predicts beta_eff stable across decades of wd
  even as the bottleneck changes.

**Design.** One protocol throughout. Architecture: MLP (256, 2) and
residual, both. Fixed p = 97, lr = 1e-3, AdamW, fixed train_fraction.

- wd ladder: {0.01, 0.03, 0.1, 0.3, 1.0, 3.0}
- seeds: >= 3 per point (5 preferred at wd <= 0.03)
- budget: 150k epochs for wd <= 0.03, 30k otherwise; record censoring
- traces: R(t) (rule signal) and D(t) (deployment metric) logged every
  20 epochs for every run

Runs: 6 wd x 3 seeds x 2 archs = 36 minimum. At current local run costs
this is small-GPU/overnight scale; the wd = 0.01 runs dominate wall time.

**Analysis.** Local pairwise beta_eff between adjacent wd rungs, per seed,
via `beta_diagnostic.py`. Fit beta_eff vs log(wd) slope with seed-level
bootstrap CI. Independently classify the binding bottleneck per rung from
traces (early-R/late-D vs late-R).

**Decision rule (pre-registered).**

- If the beta_eff-vs-log(wd) slope CI excludes 0 AND rung-level beta_eff
  correlates with the trace bottleneck (higher where M-gated): H1 wins.
- If the slope CI includes 0 with half-width < 0.1 across >= 2 decades
  while the trace bottleneck changes: H2 wins.
- Anything else: report as unresolved; do not adjudicate post hoc.

Either outcome is publishable inside the framework; what changes is
whether sublinearity is stopping-time geometry or dissipation-spectrum
structure.

## Experiment 2 (S): Seed Variance Near the Basin Boundary

**Question.** The spinodal reading of the transition zone predicts
fluctuation enhancement near the boundary: the seed-variance of tau
should peak in the zone and shrink deep inside either basin.

**Design.** Moduli {31, 43, 53, 71, 97, 113}, wd = 1.0, local protocol,
8 seeds each, budget 40k epochs (so p=31 censoring status is itself
replicated). 48 runs.

**Analysis.** Per-modulus: mean, CV, and shape of the tau distribution.
Gumbel vs normal fit comparison (an extreme-value stopping time predicts
right-skewed, Gumbel-like tau; a deterministic flow time predicts tight
symmetric spread).

**Decision rule.** Spinodal reading supported if CV(tau) at p in {43, 53}
exceeds CV at p in {97, 113} by a bootstrap-significant margin. The p=31
replication also settles whether "lookup basin: tau -> infinity" is a
property of the modulus or of one unlucky seed.

## Experiment 3 (N): Break the p^-2 vs N^-1 Degeneracy

**Question.** The corrected zone exponent (a = -2.04 +/- 0.20) is
consistent with tau ~ 1/N where N ~ p^2 is training-set size at fixed
train fraction. On existing data, modulus scaling and data scaling are
perfectly confounded.

**Design.** Fix p = 53 (mid-zone). Vary train_fraction in
{0.3, 0.4, 0.5, 0.6, 0.7}, 3 seeds each, wd = 1.0, budget 40k. 15 runs.
Optionally mirror at p = 97 (Fourier regime) for contrast.

**Decision rule.**

- If tau ~ N^-1 at fixed p (log-log slope near -1): the zone inversion is
  dataset-size kinetics; the "coordination burden p^2" language should be
  re-examined in the Fourier regime too, since N ~ p^2 there as well.
- If tau is flat in N at fixed p: the inversion is genuinely a modulus
  effect and the basin-competition mechanism needs a p-dependent, not
  N-dependent, driver.
- A diverging tau at small train fraction reproduces the known critical-
  fraction behavior and bounds the usable range; record, do not discard.

## Reporting

All runs enter the run table with the extended schema. Censored runs are
first-class observations and enter fits via `survival_fit.py`, never by
exclusion. Negative and unresolved outcomes are reported with the same
prominence as confirmations, consistent with repository practice.
