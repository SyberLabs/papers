# Phase 2 Pre-Registration: Categorical Order Parameters for Grokking

Status: pre-registration draft, 2026-07. No confirmatory analysis has been run.
Companion to phase 1 (`paper/OFFICIAL_GROKKING_RESEARCH_PAPER.md`,
`paper/BETA_BOTTLENECK_SECTION.md`). This document fixes hypotheses, metrics,
and decision rules *before* implementation so that outcomes: including null
results: are interpretable rather than chosen post hoc.

## 0. One-Paragraph Summary

Phase 1 treats grokking as a thermodynamic / kinetic transition governed by
three latent variables (memorization M, rule structure R, deployment D). Its
open frontier is that D has no architecture-universal operationalization: the
variance-based D(t) times grokking in MLPs but not residuals. Phase 2 asks
whether the *categorical* structure of the network supplies a better D. We
build two order parameters from two independent categorical lineages
a geometric one (cellular-sheaf distance-to-global-section, Hansen-Ghrist /
Bodnar-Bronstein) and a logical one (decidability of the layer's logical
cells, Belfiore-Bennequin), and test them against each other and against
the existing variance D on the phase-1 traces. This is a measurement study,
not a training study: it runs on existing runs plus a small confirmatory
sweep, and it is designed to be able to fail.

## 1. Background and Positioning

### 1.1 What the Huawei lineage actually provides

Two publicly verifiable artifacts anchor the categorical view of DNNs:

- Belfiore & Bennequin, *Topos and Stacks of Deep Neural Networks*
  (arXiv 2106.14587): every DNN is an object in a Grothendieck topos;
  learning is a flow of morphisms; layer invariances are Giraud stacks;
  semantic functioning is the ability to express theories in an internal
  language. This is a *statics* of semantics: structure at a functioning
  point, with no account of the transition to that point.
- Belfiore, Bennequin & Giraud, *Logical Information Cells I*
  (arXiv 2108.04751, public): an *empirical* companion. Trains small MLPs
  on propositional and predicate tasks and measures the spontaneous
  emergence of "logical cells." Two results are directly relevant here:
  (i) a depth-driven bifurcation in which shallow nets do Fourier analysis
  and deeper nets (3 hidden layers) develop quantized logical cells while
  "the Fourier analysis completely vanishes"; (ii) a computable logical
  score for groups of cells that correlates with the L1 norm of their
  output weights ("weights precisely perform the proofs").

The companions cited by (ii): *Logico-probabilistic Information* (2020)
and *Logical Information Cells II* (2021): remain internal Huawei technical
reports and are not used here. We use only the public measurement.

### 1.2 The cautionary structure

The Huawei topos program is a five-year, deep, descriptive effort whose most
ambitious computational deliverables stayed internal, which is cited downstream
as an ancestor rather than deployed as a method, and whose visible 2025
output is Lean formalization of topos theory: moving toward formal math,
away from ML measurement. The failure mode phase 2 must avoid is doing topos
theory *about* grokking instead of extracting a *measurement* from categorical
structure. The line adjacent to Belfiore-Bennequin that did become method is
cellular sheaf theory (Hansen-Ghrist sheaf Laplacians; Bodnar et al. neural
sheaf diffusion; sheaf cohomology of predictive coding, NeurReps 2025):
computable objects, spectral quantities, benchmarks. Phase 2 sits in that
computational lineage and treats the topos papers as vocabulary and framing.

### 1.3 Why this is not a repackaging of phase 1

Two concrete, non-cosmetic gains:

- The *Logical Information Cells* bifurcation is a second control axis
  (depth) for the same Fourier -> structured-rule transition phase 1 studies
  along the time axis. It predicts a 2D phase diagram (depth x modulus) with
  a bifurcation surface, testable directly.
- The logical score gives a decidability reading of D(t): deployment as the
  fraction of the rule-equivalence structure that the layer can *prove*, not
  merely represent. This is a sharper object than within-class variance.

## 2. Definitions

Let a trace be a sequence of checkpoints at epochs t with hidden
representations phi(x, t) for each training example x, and a task rule class
c(x) (for modular addition mod p, c(x) = (a + b) mod p for x = (a, b)).

### 2.1 Variance order parameter (phase-1 baseline)

    D_var(t) = 1 - E_c[ Var( phi(x, t) | c(x) = c ) ] / Var( phi(x, t) )

Normalized so D_var in [0, 1]. This is the incumbent; phase 2 must beat or
explain it, not merely reproduce it.

### 2.2 Sheaf order parameter (geometric leg)

Construct a cellular sheaf F over a graph G:

- Vertices: training examples x (optionally a stratified subsample; see 4.4).
- Edges: connect x, x' when c(x) = c(x') (same rule class). For scale, use
  a k-nearest-within-class graph rather than the complete within-class graph.
- Stalks: F(x) = R^d, the hidden representation space (or a PCA/random
  projection to fixed d; see 4.4).
- Restriction maps: for edge e = (x, x'), learn or fix linear maps
  F_{x<e}, F_{x'<e}. Baseline: identity onto a shared subspace after
  per-class Procrustes alignment (the cheapest defensible choice). Ablation:
  Hansen-Ghrist learned restriction maps from smooth signals.

Two variants are pre-registered, and they make different claims.
Conflating them was an error in draft v1 of this document; they are now
separated explicitly (see amendment log, Section 7).

**Variant A (label-built, fixed sheaf: diagnostic only).** Graph from class
labels as above, identity restriction maps onto a shared subspace. Then the
sheaf Laplacian L_F is *constant in time* and only the section
s_t (s_t(x) = phi(x, t)) evolves. Define

    E_A(t) = s_t^T L_F s_t / (s_t^T s_t)
    D_sheaf_A(t) = 1 - E_A(t) / E_A(t_0)

Honest characterization: with identity restrictions on a label-built graph,
E_A is a graph-weighted generalization of within-class variance, i.e. a
spectral refinement of D_var, not an independent categorical claim. It is
kept because it is nearly free to compute and calibrates the pipeline.
Because L_F is fixed, its spectrum does not evolve; Variant A **cannot**
support any spectral-gap hypothesis.

**Variant B (representation-built, time-dependent sheaf: the categorical
claim).** At each checkpoint t, build the graph as k-nearest-neighbors in
representation space (no labels), and learn restriction maps from the local
representation geometry (Hansen-Ghrist smooth-signal estimation). This yields
a time-dependent Laplacian L_F(t). Define

    D_sheaf_B(t) = dim-weighted harmonic mass of s_t under L_F(t)
                 = 1 - E_B(t) / E_B^rand(t)

where E_B^rand(t) is the energy of s_t on a degree-matched random graph at
the same checkpoint (this contrast normalization also neutralizes the
trivial-collapse degeneracy: global representation collapse sends both
energies to zero together). Class labels touch nothing in the construction;
they are used only afterward, to evaluate whether the harmonic structure
that emerges aligns with rule classes (fraction of H^0(t) explained by the
class partition). This label-free construction is what dissolves the
circularity threat of Section 5.

Spectral note (the phase-1 / phase-2 bridge: Variant B only): L_F(t) is a
diffusion operator whose spectrum now legitimately evolves. The claim that
connects the two phases is that D_sheaf_B crossing threshold coincides with
the spectral gap lambda_1(t) of L_F(t) opening: grokking as a heat equation
on the (representation-derived) example graph acquiring a harmonic section.
This is where phase-1 thermodynamics and phase-2 category theory share one
object.

### 2.3 Decidability order parameter (logical leg)

Port the *Logical Information Cells* measurement to modular arithmetic:

1. For each hidden unit a and each candidate proposition P (a subset of rule
   classes, e.g. "c(x) in S"), collect the conditioned activity distribution
   { phi_a(x, t) : x satisfies P }.
2. Partition activity into {-1, 0, +1} by thresholds (-1/3, +1/3) after
   per-unit standardization. Unit a is *logical for P at t* if >= 80% of its
   P-conditioned activity falls in one outer bin (their criterion).
3. A group of logical units is *conclusive* at t if the Boolean combination of
   its asserted/excluded propositions reconstructs the rule class c(x) for a
   target fraction of examples.
4. Define

       D_logic(t) = fraction of examples whose class is decidable at t
                    by the layer's conclusive logical groups.

For modular addition the "objectives algebra" is large (p classes), so the
practical variant restricts P to a fixed, pre-registered proposition family:
residue-in-coset propositions for the subgroups of Z_p and low-order Fourier
sign propositions. This keeps the search finite and, importantly, lets the
Fourier-vs-logical competition from *Logical Information Cells* appear as a
measurable split between the two proposition families.

## 3. Hypotheses and Decision Rules

All thresholds and windows are fixed here. "Times grokking" means the order
parameter's transition epoch falls within a pre-set window W = +/- 10% of the
val-accuracy grok epoch tau, computed per run. The transition epoch is the
half-maximum epoch of a four-parameter logistic fit to the order-parameter
curve (fit in normalized time); "steepest rise" of raw curves is NOT used, as
it is noise-fragile and would reopen a tuning surface. Runs where the logistic
fit fails (R^2 < 0.8) are reported as unclassifiable, not dropped silently.

- **H0 (null).** No categorical order parameter improves on D_var. Concretely:
  neither D_sheaf nor D_logic has a smaller median |t_steep - tau| / tau than
  D_var across the MLP runs, and neither resolves the residual case.
  *This is a real, publishable outcome and the document is written to allow it.*

- **H1 (geometric universality).** D_sheaf_B times grokking in *both* MLP and
  residual families (median timing error within W in each), where D_var
  succeeds only in MLPs. Support => the sheaf gluing condition is the
  architecture-universal order parameter phase 1 was missing.

- **H2 (logical deployment).** D_logic times grokking and, in residual runs,
  rises *late* (tracking D, not R), matching the phase-1 finding that residuals
  form rule structure early but deploy late. Support => decidability is the
  correct reading of deployment.

- **H3 (cross-validation).** D_sheaf and D_logic agree (|t_steep difference|
  within W) despite independent construction. Support => the two categorical
  lineages measure one underlying transition, strengthening both.

- **H4 (bridge; Variant B only).** The spectral gap lambda_1(t) of the
  time-dependent Laplacian L_F(t) opens within W of tau, linking the
  diffusion (thermodynamic) and gluing (categorical) readings on one
  operator. Variant A cannot address H4 (fixed spectrum), by construction. Support => phases 1 and 2 share a mechanism, not
  just a vocabulary.

Discriminating outcomes (pre-registered):

- H1 true, H2 false: grokking is geometric consistency, not logical
  decidability; the logical-cell framing is a special case that fails off
  the *Logical Information Cells* task family.
- H2 true, H1 false: decidability leads; sheaf energy is a lagging correlate.
- H1 and H2 true, H3 false: two distinct transitions exist and the "single
  order parameter" goal is mistaken: also a substantive result.
- All false (H0): categorical structure adds no measurement power beyond
  variance statistics on these tasks. Report it plainly.

## 4. Implementation Plan

### 4.1 Module layout (mirrors phase-1 conventions)

- `src/grokking_scaling_theory/sheaf_order_parameter.py`
  - build_class_graph(examples, classes, k) -> graph
  - sheaf_laplacian(graph, stalks, restriction="procrustes"|"learned")
  - dirichlet_energy(L, section), spectral_gap(L)
  - d_sheaf(trace) -> array over checkpoints
- `src/grokking_scaling_theory/logical_cells.py`
  - conditioned_activity(trace_checkpoint, proposition_family)
  - logical_units(activity, bin_thresholds=(-1/3, 1/3), purity=0.80)
  - conclusive_groups(units, max_group=3) with Boolean reconstruction
  - d_logic(trace, proposition_family) -> array over checkpoints
- `src/grokking_scaling_theory/order_parameter_compare.py`
  - aligns D_var, D_sheaf, D_logic, lambda_1 on one epoch axis
  - steepest-rise timing, timing error vs tau, per-family summary
- Docs: this file, plus `analysis/ORDER_PARAMETER_RESULTS.md` (written only
  after runs) reporting each hypothesis with effect sizes and CIs.

### 4.2 Data

- Reuse phase-1 traces that carry per-example hidden states. The current run
  table stores scalar diagnostics (algorithmic_mode_mass, phase_alignment,
  etc.) and trace paths; per-example phi(x, t) must be present or regenerated.
  Where only scalar traces exist, D_var and D_logic-lite (using the scalar
  Fourier/mode fields) are still computable, but D_sheaf needs per-example
  states, so a small confirmatory sweep (4.3) is required regardless.
- All machine-local trace paths must be relativized first (the phase-1 loader
  fix already tolerates absence; phase 2 needs the actual arrays).

### 4.3 Confirmatory sweep (minimal)

To test H1-H4 with matched MLP/residual runs and stored per-example states:

- moduli: p in {59, 97, 113} (in the phase-1 Fourier regime, avoiding the
  transition zone so the order-parameter test is not confounded by regime).
- architectures: MLP (256, 2) and residual, matched width/params.
- wd = 1.0, AdamW, >= 3 seeds per cell. 3 x 2 x 3 = 18 runs.
- checkpoint per-example hidden states every 20 epochs to grok + 50%.
- storage: subsample to 2000 examples per run for the sheaf graph (4.4).

### 4.4 Scale controls (pre-committed)

- Graph size: k-nearest-within-class with k = 10, not complete within-class
  graphs (O(N k) vs O(N^2 / p)).
- Stalk dimension: project phi to d = 64 via PCA fit at grok epoch, applied to
  all checkpoints of that run (fixed basis, so cross-time comparison is valid).
- Restriction maps: Variant A uses identity maps only. Variant B uses
  Hansen-Ghrist learned maps; per-class Procrustes is explicitly EXCLUDED
  from Variant B because it injects label information into the construction
  under test.
- Randomization: report D_sheaf against a label-shuffled null (classes
  permuted) to confirm the order parameter tracks real class structure, not
  graph connectivity artifacts.

### 4.5 Statistics

- Primary estimand: median |t_steep - tau| / tau per order parameter per
  family, with seed-level bootstrap 95% CIs.
- H3/H4 agreement: paired differences of steepest-rise epochs, bootstrap CI.
- No p-hacking surface: thresholds (W, purity 0.80, bins +/-1/3, k, d) are
  fixed above. Any deviation forced by data is reported as a protocol
  amendment with date and reason, not folded silently.

## 5. Threats to Validity

- **Circularity.** Variant A is circular by design and is labeled a
  diagnostic, not evidence: its graph uses the same class labels D_var
  conditions on, and with identity restrictions its energy is a spectral
  refinement of within-class variance. All confirmatory weight rests on
  Variant B, whose construction (representation-kNN graph, learned
  restrictions) never sees labels; labels enter only at evaluation.
  Additional guards: the label-shuffled null (4.4, applies to evaluation
  alignment) and the requirement that D_sheaf_B beat D_var on *timing in
  residuals*, where D_var fails.
- **Proposition-family dependence.** D_logic depends on the chosen family;
  a rich enough family makes everything decidable and D_logic saturates early.
  Mitigation: the family is fixed and deliberately lean (cosets + low-order
  Fourier signs), and the Fourier/logical split is reported, not averaged away.
- **Small n.** 18 runs, single wd. This is a mechanism-isolation study, not a
  universality proof; claims are scoped accordingly, matching phase-1 practice.
- **Descriptive trap.** The whole risk of the Huawei lineage. Guard: every
  hypothesis has a numeric decision rule and a null; a categorical construction
  that only *describes* grokking without improving timing is reported as H0.

## 6. Deliverables and Order of Work

1. Relativize trace paths; confirm per-example state availability (blocker).
2. Implement `sheaf_order_parameter.py`; validate on one p=97 MLP trace;
   check label-shuffled null gives no transition.
3. Implement `logical_cells.py`; reproduce a Fourier-vs-logical split on one
   trace as a sanity check against *Logical Information Cells*.
4. Run the 18-run confirmatory sweep with per-example checkpointing.
5. `order_parameter_compare.py`; compute all timing errors and CIs.
6. Write `analysis/ORDER_PARAMETER_RESULTS.md` reporting H0-H4 as they fall.

The intended contribution is a single, honest sentence in the end state:
either "a categorical gluing/decidability order parameter is the
architecture-universal deployment variable phase 1 lacked," or "categorical
structure does not improve on variance statistics for timing grokking on these
tasks, and here is the evidence." Both are worth publishing; the protocol is
built so the data, not the author, decides which.

## 7. Amendment Log

- **2026-07, amendment 1 (pre-implementation).** Draft v1 conflated a fixed,
  label-built sheaf with a time-dependent, representation-built sheaf, and
  stated a spectral-gap hypothesis (H4) that is incoherent for the fixed
  construction (constant Laplacian, static spectrum). v2 separates Variant A
  (diagnostic, label-built, fixed L, honestly characterized as a spectral
  refinement of D_var) from Variant B (label-free, time-dependent L_F(t),
  the sole carrier of H1/H4 and of the categorical claim). The timing metric
  was also hardened from raw steepest-rise to a pre-registered logistic
  half-maximum fit, and per-class Procrustes restrictions were excluded from
  Variant B as label leakage. No data had been collected at the time of this
  amendment.

- **2026-07, amendment 2 (at prototype implementation, pre-data).** Two
  operationalizations fixed by `sheaf_order_parameter.py`: (i) the Variant B
  spectral event is the p-th eigengap of the scalar kNN-graph Laplacian
  (with identity restrictions the sheaf Laplacian factorizes as
  L_graph (x) I_d, so the scalar spectrum carries all content), reported
  alongside the near-kernel dimension approaching p; (ii) the contrast
  normalizer is an edge-count-matched random graph (degree-matched is
  deferred to the learned-restriction ablation). Synthetic validation gates
  passed: planted-transition timing recovered within 0.4% across seeds
  (R^2 = 1.00), shuffled-label alignment at chance, collapse flagged
  degenerate. Under 3x noise the timing estimate degrades gracefully
  (+8% bias) rather than failing silently: noted as a known bias source
  for low-SNR traces.

- **2026-07, amendment 3 (post methodological review; before analysis of the
  full sweep and before all arms declared below).** Disclosure of data seen
  at amendment time: the 18-run confirmatory sweep had completed training,
  but only the two p=59 seed-0 traces (one MLP, one residual) had been fed
  through the order-parameter analysis. No other trace had been analyzed.
  The amendment fixes the following, in response to an external
  methodological review:

  1. **Grok-epoch threshold pinned.** tau = first checkpoint epoch with
     val accuracy >= 90% (matching the executed sweep and the phase-1
     `t_90` observable). A 95% sensitivity re-derivation is pre-declared
     and becomes possible once traces store the accuracy series (item 2).
     The run table gains a `grokking_threshold` column.
  2. **Trace schema extension.** The .npz gains `val_acc` and `train_acc`
     arrays aligned with `epochs`, so tau is re-derivable at any threshold
     without re-running. The 18-run sweep is re-executed with identical
     seeds into a v2 directory (deterministic re-run; doubles as a
     determinism check and removes a float32/float16 storage inconsistency
     affecting exactly one v1 trace). v2 grok epochs must match v1
     exactly; any mismatch is reported as a determinism failure.
  3. **Amplitude disclosure.** Every logistic timing fit reports the fitted
     amplitude |hi - lo| alongside R^2. For the original 18 runs amplitude
     is reported but NOT gated (a gate would be post hoc for the two
     analyzed traces). For the control arms declared in item 5, a
     pre-registered amplitude gate applies: fitted amplitude < 0.05 =>
     the fit cannot count toward a within-W verdict (reported as
     low-amplitude instead).
  4. **Family decision rule reverted to registered form.** H1's family
     rule is median relative timing error <= W, exactly as section 3
     states. An implementation had strengthened this with an unregistered
     majority-within-W conjunct; that variant is demoted to a clearly
     labeled secondary robustness check.
  5. **Bypass/parameter control arm (pre-declared).** The executed
     "residual" model (ResidualRose) carries +65,792 parameters (+60-74%)
     over the executed MLP (SoftRose) and adds a linear bypass; both have
     exactly one ReLU in the hidden stack, so the contrast is
     bypass+parameters, NOT nonlinearity depth. Control: **NoSkipRose**,
     hidden = ReLU(W2 W1 e): parameter-identical to ResidualRose,
     function-class-identical to SoftRose (W2 W1 collapses to one matrix).
     9 runs (p in {59, 97, 113} x 3 seeds, wd = 1.0, same protocol).
     Decision rule: if NoSkipRose's D_sheaf_B is unclassifiable (SoftRose-
     like), the linear bypass is load-bearing for the sheaf signal; if it
     is classifiable with residual-like timing, the signal follows
     parametrization/optimization, not the bypass. A secondary
     width-matched arm (SoftRose with hidden_dim = 448, approximately
     parameter-matched to ResidualRose) is declared as optional, to be run
     only after the primary arm is analyzed.
  6. **Regime scope correction.** The fresh sweep shows grok time
     decreasing monotonically from p=59 to p=113 in both families under
     the local protocol: the sweep sits in the inverted-scaling
     (transition) regime, not the Fourier regime as section 4.3 assumed.
     Section 4.3's "avoiding the transition zone" rationale is void; all
     H1-H4 conclusions from this sweep are scoped to the inverted regime
     of this protocol. The widened zone (extending to at least p=113
     locally) is itself a reportable extension of the transition-zone
     finding in `analysis/SURVIVAL_FIT_RESULTS.md`.

- **2026-07, amendment 4 (before any D_logic computation).** Disclosure: at
  amendment time the sheaf/D_var results over all 27 traces were known
  (`analysis/ORDER_PARAMETER_RESULTS.md`), but NO decidability quantity had
  been computed on any trace. The parameters below are fixed before the
  first D_logic run; where a choice could plausibly be tuned toward an H2/H3
  outcome, the conventional value is taken and a sensitivity report is
  pre-committed instead of a choice.

  1. **Proposition family for prime moduli.** Section 2.3's
     "residue-in-coset propositions for the subgroups of Z_p" is degenerate
     for prime p (no proper additive subgroups). The family is fixed as:
     (a) low-order Fourier sign propositions
     `cos(2 pi k c / p) > 0` and `sin(2 pi k c / p) > 0` for k = 1..4
     (8 propositions), and (b) the quadratic-residue proposition
     `c in QR(p)`, c=0 excluded from QR (1 proposition): the multiplicative
     stand-in for the coset leg. Lean, fixed, and it preserves the
     Fourier-vs-logical split as the contrast between legs (a) and (b).
  2. **Ceiling normalization.** The 9-proposition signature partitions the
     p classes into at most ~40 sign-arcs x 2 QR cells; not all classes are
     distinguishable, and the ceiling falls with p (~0.68 of examples at
     p=59 vs ~0.35 at p=113 for the Fourier legs alone, before QR). D_logic
     is therefore reported as raw decidable fraction AND normalized by the
     family's theoretical ceiling (fraction of examples whose class has a
     unique proposition signature, computable from the family alone with no
     data). Timing metrics use the normalized series; the ceiling makes
     levels, not timing, comparable across p.
  3. **Logical-unit criterion.** Per-checkpoint per-unit standardization;
     ternary states with bins (-1/3, +1/3); purity 0.80 in one outer bin on
     the P-conditioned side (all per section 2.3, fixed pre-data).
     Added: a **two-sided informativeness condition**: the unit must also
     concentrate (>= 0.80) in the *opposite* outer bin on the
     not-P-conditioned side. A unit saturated in one bin regardless of P is
     logical for every proposition under the one-sided rule and carries no
     evidence; the two-sided condition is what "cells assert or exclude
     propositions" requires operationally.
  4. **Conclusive groups and decidability.** For each proposition, a
     conclusive group is <= 3 informative units whose majority vote (over
     non-zero states; abstain on tie/all-zero) attains balanced accuracy
     >= 0.90 for P vs not-P; greedy selection (best single unit, else best
     triple). A proposition is *covered* at t if a conclusive group exists.
     An example is *decidable* at t if the intersection of its covered-
     proposition verdicts is the singleton {c(x)} (singleton but wrong
     counts as not decidable). D_logic(t) = decidable fraction / ceiling.
     Sensitivity report pre-committed: balanced-accuracy target at
     {0.80, 0.90, 0.95}, primary 0.90.
  5. **H2/H3 decision rules (registered here).**
     - H2 leg 1 ("times grokking"): median |t_Dlogic - tau| / tau <= W per
       family, same family rule as H1.
     - H2 leg 2 ("rises late in residuals, tracking D not R"): median of
       paired differences (t_Dlogic - t_Dsheaf_B_raw) over residual runs
       where both are classifiable is positive with seed-level bootstrap
       95% CI excluding 0.
     - H3 ("the two categorical legs agree"): median
       |t_Dlogic - t_Dsheaf_B_raw| / tau <= W, per family.
  6. **Fit sanity gate (all timing fits from this amendment onward,
     including every D_logic fit).** A logistic fit is
     degenerate-unclassifiable if fitted amplitude > 1.5x the observed
     series range, or t_half lies outside the observed epoch span, or
     fitted amplitude < 0.05. Motivated by the pathological MLP PCA-64
     fits documented in the results (amplitudes 63-92, t_half at -32180);
     NOT applied retroactively to pre-amendment-4 fits.
  7. **Validation gates before real data (mirroring the sheaf module).**
     L1: planted decidable code recovers the planted transition epoch
     within 10% with normalized D_logic rising to ~1. L2: label-shuffled
     classes yield D_logic ~ 0 with no transition. L3: units encoding only
     Fourier signs produce covered Fourier propositions with QR uncovered
     (the split is measurable). All three must pass before any trace is
     analyzed.
     Construction note (fixed at gate implementation, before any real-trace
     D_logic): the L1 planted schedule is a fast onset (linear ramp over 6%
     of the span), and the truth compared against is the schedule's own
     half-maximum. A slow sigmoid schedule is inappropriate for L1 because
     D_logic is a threshold detector (units count only once purity crosses
     0.80) and fires at the detectability threshold of any slow schedule
     (s ~ 0.1 at gate amplitudes), which is a property of threshold
     detection, not a timing error. Gates passed: L1 err 5.6%, L2 peak
     0.000, L3 split 8/8 Fourier vs 0 QR (p=29 ceiling 0.72).

- **2026-07, amendment 5 (the depth arm; before any run).** Disclosure: at
  amendment time the full 27-run results including the D_logic null were
  known (`analysis/ORDER_PARAMETER_RESULTS.md`, findings 10-11: zero logical
  cells, max two-sided purity 0.42-0.43, uniform across the three
  one-nonlinearity families). This arm is the experiment that null points
  to: crossing the *Logical Information Cells* depth bifurcation
  (arXiv 2108.04751: quantized logical cells at ~3 hidden layers, Fourier
  vanishing) to test whether cells, and a functioning D_logic: appear at
  depth. This also instantiates the depth axis of the depth x modulus
  phase diagram promised in section 1.3.

  1. **Architecture (fixed).** DeepRose: embedding(p, 128) -> concat(256)
     -> Linear(256,256)+ReLU -> Linear(256,256)+ReLU -> Linear(256,256)+ReLU
     -> readout(256, p). Three hidden nonlinearities (vs one in all prior
     families). Parameter count ~220k at p=59 (vs 88k/154k shallow);
     params are NOT matched. Deconfound argument, stated in advance: in the
     shallow data, max two-sided purity was flat (0.42-0.43) across an
     88k -> 175k parameter range and across parametrizations, so parameter
     count is not a credible driver of purity; nonlinearity depth is the
     variable this arm moves. A param-matched wide-shallow control is
     declared optional, to be run only if D1 is positive and the param
     confound is contested.
  2. **Design.** p in {59, 97, 113} x 3 seeds = 9 runs; wd = 1.0,
     lr = 1e-3, AdamW, train_fraction 0.45, tau threshold 90: identical to
     prior arms. Budget 60,000 epochs (vs 40k), fixed in advance because
     optimization difficulty at depth is unknown and censoring at 40k would
     be ambiguous between "depth impedes grokking" and "budget too small".
     Censored runs are first-class: their traces are still analyzed for
     cell formation (cells may form without deployment: itself
     informative); timing metrics are skipped where tau is undefined.
  3. **Multi-layer logging (false-null guard).** Cells may form in any
     hidden layer, so all three are logged: final hidden layer (feeding the
     readout) at the standard 20-epoch cadence under the canonical `hidden`
     key (pipeline-compatible, comparable to prior arms); layers 1 and 2 at
     100-epoch cadence under `hidden_l1`, `hidden_l2` with `epochs_aux`.
     Purity/coverage/D_logic are computed per layer; if cells appear only
     in an earlier layer, its ~60+ point series still supports the logistic
     timing fit. Example subsample stays 2000 (4.4).
  4. **Decision rules.**
     - **D1 (cell formation, primary).** Outcome = number of runs with >= 1
       covered proposition in ANY layer at any checkpoint. D1 positive if
       >= 5 of 9 runs have coverage. Purity distributions per layer are
       reported regardless; a purity rise that stays below criterion is
       reported as movement toward the surface, not as support.
     - **D2 (D_logic functions at depth).** If coverage exists, D_logic
       (per the amendment-4 pipeline, sanity-gated) classifiable in the
       covering layer for >= half of the covering runs. If D2 holds, H2/H3
       are adjudicated per amendment-4 item 5, restricted to the depth
       family and so scoped.
     - **D3 (Fourier vanishing, LIC's second prediction).** Metric:
       `fourier_concentration`: per unit, the class-conditional mean
       activation profile m(c) is Fourier-analyzed; concentration = max
       non-DC harmonic power / total non-DC power; reported as the mean
       over units with non-trivial profile variance, and the 90th
       percentile, per layer. Known limitation, stated in advance: a
       sign-quantized sinusoid (a logical cell of a half-circle
       proposition) retains ~0.81 base-harmonic concentration, so this
       metric separates class-structured from unstructured units, not sine
       from square; the (concentration, purity) PAIR is the discriminator
       (Fourier: high/low; logical cell: high/high; unstructured: low/-).
       D3 supported if final-hidden-layer concentration in the depth arm is
       lower than the shallow families' hidden-layer concentration at
       matched p (medians, bootstrap CI excluding 0 difference).
     - **D4 (cascade at depth).** D_var -> D_sheaf_B -> tau ordering
       (finding 4) re-tested on the depth family's final layer; reported
       either way, no support threshold.
  5. **New-metric gate (pre-data).** F1: planted sinusoidal units give
     concentration > 0.9; unstructured random units give < 0.2. Must pass
     before any real-trace computation. All existing gates (L1-L3, sheaf
     gates) must still pass after code changes.
  6. All timing fits in this arm carry the amendment-4 item-6 sanity gate.

- **2026-07, amendment 6 (sharpening test + deeper arm; before either).**
  Disclosure: at amendment time the 36-run results were known, including
  finding 13 (aggregate Fourier concentration AND purity rise together at
  depth: the "sharpening" signature) and finding 12 (depth-graded purity,
  no coverage). What has NOT been examined anywhere is the *unit-level*
  structure of that rise; all Part A statistics operate at that unexamined
  level and their thresholds are fixed here.

  **Part A: the sharpening-hypothesis test (existing traces only; no new
  training).** Competing worlds for the depth-induced rise:
  - *Sharpening (continuous)*: individual units morph sinusoid -> square;
    the same units carry structure throughout.
  - *Replacement (discrete, the LIC picture)*: sinusoidal units fade out
    while distinct cell-units are born already-quantized; the structured
    population turns over.
  A naive "concentration and purity rise together" test does NOT
  discriminate (newborn cell-units also raise both at once); the
  discriminators are within-unit trajectories and unit identity:
  1. **Per-unit metrics** (per checkpoint, per layer): conc_a (dominant
     non-DC harmonic share of the class profile, as in amendment 5);
     purity_a (best two-sided purity over the amendment-4 proposition
     family); overtone_a = P(3k* alias) / P(k*), the third-to-first
     harmonic power ratio of the class profile (square-wave reference
     value 1/9 ~ 0.111; scale-invariant, so a fading or growing unit keeps
     its waveform's value).
  2. **Structured units**: conc_a >= 0.5. Early window = first third of
     checkpoints; late window = last third.
  3. **Statistics (per deep run, layers l2 and final; shallow hidden
     reported as context):**
     - S1 (identity continuity): Jaccard overlap of the structured-unit
       sets, early vs late window (using each window's last checkpoint).
     - S2a (morphing): median within-unit slope of overtone_a over
       normalized time, among units structured in the EARLY window;
       bootstrap CI over units.
     - S2b (onset waveform): among units structured only in the LATE
       window, median overtone_a at the first checkpoint where they cross
       conc >= 0.5.
     - S3 (bimodality): on purity_a of late-window structured units,
       2- vs 1-component 1D Gaussian mixture BIC (in-house EM);
       "bimodal" = delta-BIC > 10 with upper-component mean > 0.8.
  4. **Adjudication (fixed):** SHARPENING supported if median S1 > 0.5 AND
     S2a CI > 0. REPLACEMENT supported if median S1 < 0.3 AND S2a CI not
     above 0 AND (S2b >= 0.06 OR S3 bimodal in >= half of runs). Anything
     else: mixed/unresolved, reported as such. Family-level medians with
     bootstrap over runs.
  5. **Gate SG (pre-data, mandatory):** both worlds are synthesized
     (World A: units morph cos -> sign(cos) on a schedule; World B:
     disjoint populations crossfade, cells born with square waveform) and
     the statistics must classify each correctly before any real trace is
     touched. All prior gates must still pass.
     Gate-forced refinements (recorded before any real-trace computation;
     three SG iterations): (i) the synthetic transition occupies the middle
     third of the trace so the early/late windows are clean: a crossfade
     already underway at the early window is an ill-posed turnover test;
     (ii) S2a's retention mask is conc >= 0.5 (structured), because a
     fading unit's noise-dominated checkpoints inflate P3/P1 and fake a
     morph; (iii) the sharpening/replacement split on S2a is effect-size
     based: the median slope must be MATERIAL (>= 1/4 of the square-wave
     overtone, i.e. >= 0.0278) as well as statistically positive: SNR
     creep produces significant-but-immaterial slopes (~1% of the
     sine->square change) in pure-replacement worlds.
  6. Deliverables: `src/grokking_scaling_theory/sharpening.py`,
     `analysis/SHARPENING_RESULTS.md`.

  **Part B: the deeper arm (after Part A reports).** [Adjudicated
  2026-07-07: D1 negative 0/9, purity ladder stalled, D3 reversed at
  +0.25, D4 unmeasurable, churn replicated; 4/9 censored. See
  `analysis/DEEP5_RESULTS.md`. The contingent narrow arm below is thereby
  authorized.] DeepRose generalized
  to depth 5 (five hidden ReLUs, width 256; only depth moves relative to
  amendment 5). p in {59, 97, 113} x 3 seeds = 9 runs, wd 1.0, lr 1e-3,
  budget 80,000 epochs (depth-5 optimization difficulty unknown; same
  rationale as amendment 5 item 2). Aux layers l1..l4 logged at 100-epoch
  cadence, final at 20. Decision rules D1-D4 as amendment 5 (D1 threshold
  5/9 on any-layer coverage), plus Part A's S-statistics computed on the
  new traces as replication. A narrow variant (width 64, closer to the
  LIC nets) is declared as a contingent secondary arm, to be run only if
  the primary depth-5 arm is D1-negative. All fits carry the amendment-4
  sanity gate.

- **2026-07, amendment 7 (the narrow arm, pinning the amendment-6
  contingency; before any run).** Disclosure: the deep5 adjudication was
  known (D1 negative (the authorizing condition) plus the purity stall,
  the D3 dose-response, and the trainability collapse). The contingency
  left the narrow arm's depth unpinned; it is fixed here with rationale:
  1. **Architecture: DeepRose, n_hidden = 3, hidden width 64** (embedding
     128 unchanged; ~36k params at p=59). Depth 3, not 5, because (a) 3
     hidden layers is the LIC bifurcation regime itself, (b) depth 3 is
     this study's best purity approach (max 0.80) and trains reliably
     (9/9 grokked), whereas depth 5's trainability collapse would
     confound a cell null with an optimization failure. This isolates
     WIDTH at fixed depth: deep3@256 (done) vs deep3@64 (this arm). The
     hypothesis it tests: narrowness concentrates representational
     pressure per unit and forces quantization: the last LIC-faithful
     corner (their nets were small and narrow) not yet explored.
  2. **Design:** p in {59, 97, 113} x 3 seeds = 9 runs; wd 1.0, lr 1e-3,
     train_fraction 0.45, tau threshold 90; budget 60,000 epochs (deep3
     grokked within 8.6k at width 256; narrowness may slow grokking
     headroom without deep5's ambiguity). Aux l1-l2 at 100-epoch cadence,
     final at 20. Censored runs analyzed structurally as before.
  3. **Rules:** D1 (>= 5/9 any-layer coverage), D2, D3 (fourier
     concentration vs the shallow pool, grokked-only primary), D4
     (reported), and the Part A S-statistics as replication: all
     unchanged, all fits under the amendment-4 sanity gate.
  4. **Interpretive frame, fixed in advance:** D1 positive here =>
     narrowness, not depth, is the missing LIC ingredient: the cell
     regime exists on this task and the program continues into it. D1
     negative => every registered corner of the LIC architecture space
     (deep, deeper, narrow-deep) is exhausted on this task family; the
     program's end state is declared and the write-up proceeds with the
     wave-code conclusion as final for modular arithmetic under standard
     training.
