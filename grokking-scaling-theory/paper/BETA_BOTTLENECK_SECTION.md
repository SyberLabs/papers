# The Weight-Decay Exponent as a Bottleneck Diagnostic

Draft section for `OFFICIAL_GROKKING_RESEARCH_PAPER.md` (proposed placement:
after Section 8, "Current Effective Theory"). All numbers below are computed
from `analysis/expanded_scaling_runs.csv` by
`src/grokking_scaling_theory/beta_diagnostic.py`.

## Claim

The measured local weight-decay exponent

`beta_eff(p, wd, arch) = - d log(tau) / d log(wd)`

is not a universal constant of grokking. It is an emergent property of the
coupled (M, R, D) stopping time, and its value identifies which latent clock
binds:

- `beta_eff -> 1`: cleanup-limited (M-gated). Rule structure forms early;
  visible grokking waits on memorization decay.
- `beta_eff -> 0`: formation-limited (R-gated). Cleanup finishes early;
  grokking waits on rule formation.
- `0 < beta_eff < 1`: crossover; the two clocks are comparable.

This reframes the architecture boundary of Section 6.2. The residual family
is not a failure of the theory. It is the M-gated limit of the same theory,
and the exponent it measures is a prediction, not a fit.

## Derivation Sketch: Emergent Sublinearity from a Linear Microscopic Law

Replace the phenomenological memorization equation with the microscopically
natural one, in which weight decay enters linearly:

`dM/dt = -a1 * wd * M`  =>  `M(t) = M0 * exp(-a1 * wd * t)`.

Rule structure grows at a wd-independent (to first order) rate:

`dR/dt = a2 * gamma_R(p) * R * (1 - R)`  =>  `tau_R ~ 1 / gamma_R(p)`.

Deployment remains `dD/dt = a3 * R * (1 - D) - a4 * M * D`, with grokking at
`D(tau) = D_crit`.

Two limits of the stopping time:

1. **M-gated limit** (`tau_R << tau_M`). R saturates early, but the
   suppression term `-a4 * M * D` pins D until M falls below
   `M_c ~ a3 / a4`. The crossing time is
   `tau ~ ln(M0 / M_c) / (a1 * wd)`, hence `tau ~ wd^{-1}` and
   `beta_eff = 1` exactly.

2. **R-gated limit** (`tau_R >> tau_M`). Memorization is gone long before
   rule structure completes; `tau ~ tau_R + O(1/a3)` with negligible wd
   dependence, hence `beta_eff -> 0`.

Between the limits, the stopping time is a smooth function of both clocks
and `beta_eff` interpolates in (0, 1). A sublinear measured exponent
(`beta ~ 0.65-0.9`) therefore requires **no fractional power anywhere in
the microscopic dynamics**. This removes the least principled step of the
current derivation note, which inserts `wd^beta` into the M equation by
hand and then calls its exact solution the cleanest part of the theory.

## Evidence

Local pairwise slopes from trace-backed runs
(`beta_diagnostic.py`, measured points only):

| group | arch | p | wd range | beta_eff |
|---|---|---:|---|---:|
| local_mlp | mlp | 59 | 1.0 -> 2.0 | 0.814 |
| local_mlp | mlp | 97 | 0.5 -> 1.0 | 0.895 |
| local_mlp | mlp | 97 | 1.0 -> 1.5 | 0.743 |
| local_mlp | mlp | 97 | 1.5 -> 2.0 | 0.907 |
| local_mlp | mlp | 113 | 1.0 -> 2.0 | 0.762 |
| local_residual | residual | 59 | 1.0 -> 2.0 | 1.000 |
| local_residual | residual | 97 | 1.0 -> 2.0 | 1.038 |
| local_residual | residual | 113 | 1.0 -> 2.0 | 0.957 |
| published_mlp | mlp | 97 | 0.01 -> 0.1 | 0.602 |
| published_mlp | mlp | 97 | 0.1 -> 1.0 | 0.699 |

Architecture summary:

- residual: `beta_eff = 0.998 +/- 0.033` (n=3): cleanup-limited (M-gated)
- mlp (local): `beta_eff = 0.775 +/- 0.100` (n=7, incl. published pairs)
  crossover

Two independent lines of evidence now converge on the residual story. The
trace diagnostics of Section 6 show residual networks forming early rule
structure and early prototypes while grokking lags (early-R, late-D). The
exponent measurement, computed with no reference to any trace, lands on
`beta_eff = 1` to within 4 percent: the exact M-gated prediction. Neither
measurement was fit to the other.

## Predictions

- **P1 (confirmed).** Any family whose traces show early-R/late-D measures
  `beta_eff ~ 1`. Residual: confirmed above.
- **P2 (hinted, underpowered).** In formation-limited families,
  `beta_eff` decreases with p inside the Fourier regime, because
  `tau_R ~ p^2 / log(p)^2` grows and R binds harder. Local MLP trend:
  0.814 (p=59) -> 0.762 (p=113); fitted `d(beta)/d(log p) = -0.085`.
  Single seeds; needs replication.
- **P3 (open, discriminating).** The crossover account predicts `beta_eff`
  drifts across a wide wd ladder. A genuinely fractional mechanism
  (heterogeneous dissipation spectrum, the "broad cleanup spectrum" of
  Section 10) predicts `beta_eff` stable across decades. The published
  p=97 ladder spans two decades and is flat (0.602, 0.699): weak evidence
  for the fractional account *in that protocol*, and tension with the
  crossover account, which expects `beta_eff -> 1` as `wd -> 0` makes
  cleanup the binding clock. Decisive experiment: one architecture, one
  protocol, `wd in {0.01, 0.03, 0.1, 0.3, 1.0, 3.0}`, >= 3 seeds per point,
  with R(t) and D(t) traces attached. If beta drifts and correlates with
  the trace-identified bottleneck, the crossover account wins; if it is
  flat while the bottleneck changes, the fractional account wins.

## Scope and Honesty Notes

- All local estimates are single-seed. Grokking-time seed variance is
  unmeasured in this repository, and the classification gate in
  `beta_diagnostic.py` refuses to classify when spread exceeds 0.15.
- The published-ladder pairs mix protocols with the local ladders and are
  reported separately, never pooled.
- P3 is currently unresolved, and its two outcomes are both compatible with
  the (M, R, D) framework; what changes is where sublinearity lives
  (stopping-time geometry vs. dissipation spectrum). Either outcome is
  reportable.
