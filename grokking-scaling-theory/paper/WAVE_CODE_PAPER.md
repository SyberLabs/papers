# Grokking Without Logic: A Pre-Registered Dissection of Categorical Order Parameters Finds a Churning, Strengthening Wave Code

**Phase 2 companion to** `OFFICIAL_GROKKING_RESEARCH_PAPER.md` (the (M, R, D)
effective theory of grokking). **Canonical evidence base:**
`analysis/ORDER_PARAMETER_RESULTS.md` (18 findings),
`analysis/SHARPENING_RESULTS.md`, `analysis/DEEP5_RESULTS.md`,
`analysis/DEEPNARROW_RESULTS.md`. **Protocol:**
`experiments/PHASE2_CATEGORICAL_ORDER_PARAMETERS.md` (pre-registration +
7 amendments). Status: draft v1, 2026-07-07.

---

## Abstract

Grokking — delayed generalization long after training accuracy saturates —
invites two readings of what a network's internal representation is doing
during the delay: a *geometric* reading (representation space gradually
organizes until a consistency condition is met) and a *logical* reading
(discrete, quantized "logical cells" form and render the task decidable, as
reported by Belfiore, Bennequin & Giraud on propositional tasks). We test
both readings as candidate *order parameters* for grokking on modular
addition, under a pre-registered protocol with synthetic validation gates,
across 63 training runs spanning five architectures (three one-nonlinearity
families; three- and five-layer deep stacks; a narrow-deep variant), three
moduli, and unit-level trajectory statistics. The results reject both
registered hypotheses and replace them with a sharper picture. (1) A
label-free sheaf/kNN order parameter beats the variance incumbent on timing
grokking in every family, but is a *leading* indicator (typically 10-25%
early); its strength tracks weight-matrix parametrization and depth, never
the function class. (2) The logical reading fails with its own mechanism
measured: no logical cell ever assembles into a decision-grade group, and
unit-level statistics show a *churning wave code* — structured units turn
over continuously, newcomers are always born sinusoidal (onset overtone
<= 0.032 vs. the square-wave reference 0.111 in 63/63 tested cells), and no
unit ever morphs. (3) Depth *amplifies* the Fourier code monotonically
(concentration 0.47 -> 0.54 -> 0.72 across 1 -> 3 -> 5 ReLUs) — the clean
opposite of the predicted Fourier-to-logic replacement — while trainability
collapses (4/9 five-layer runs never grok in 80k epochs). (4) Width, not
depth, is the dominant quantization pressure: only narrow (width-64)
networks produce units crossing the 0.80 logical-cell purity criterion
(max 0.88), and only narrow networks yield internal transitions coincident
with behavioral grokking (sheaf timing within +/-10% of the grok epoch).
We conclude that grokked modular arithmetic is a drifting, strengthening
wave code at every trainable depth and width, and that the logical-cell
bifurcation, if it exists for this task family, lies beyond the reach of
standard training, which fails first.

---

## 1. Introduction

Phase 1 of this program modeled grokking as a coupled three-variable
kinetic system — memorization M, rule formation R, deployment D — and
measured scaling laws for the grok epoch tau. Its open frontier was
operational: D (deployment) had no architecture-universal measurement. The
within-class variance statistic that timed grokking in one family failed in
another.

Phase 2 asked whether *categorical* structure supplies the missing order
parameter, drawing on two lineages:

- **Geometric leg.** Cellular-sheaf methods (Hansen-Ghrist; Bodnar et al.):
  build a graph over training examples from the representation itself (no
  labels), and measure the Dirichlet energy contrast of the hidden states —
  "gluing" of a section over the example graph. We call this D_sheaf_B.
- **Logical leg.** *Logical Information Cells* (Belfiore, Bennequin &
  Giraud, arXiv 2108.04751; framework in arXiv 2106.14587): small MLPs on
  propositional tasks spontaneously develop quantized ternary units that
  assert/exclude propositions, with a depth bifurcation — shallow nets do
  Fourier analysis, deeper nets (~3 hidden layers) develop cells while
  "the Fourier analysis completely vanishes." We port their measurement to
  modular arithmetic as a decidability order parameter, D_logic.

The registered hypotheses (H0-H4) and all decision rules were fixed before
data collection; seven dated amendments record every operationalization,
threshold, and gate-forced refinement, each declared before the data it
governs. Every measurement instrument passed synthetic validation gates —
including two-world discriminability gates that must classify planted
"sharpening" and "replacement" dynamics correctly — before touching a real
trace.

**Contributions.**

1. A gate-validated, pre-registered measurement suite for grokking order
   parameters: variance baseline, label-free sheaf energy (raw and PCA),
   spectral eigengap, ceiling-normalized decidability, class-profile
   Fourier concentration, and unit-level trajectory statistics
   (identity continuity, overtone morphing, onset waveform, purity
   bimodality).
2. The finding that the sheaf order parameter is a robust *leading*
   indicator of grokking that beats the variance incumbent everywhere, with
   strength controlled by weight parametrization (via a
   function-class-preserving control architecture) and depth — and that it
   becomes *coincident* with grokking only at narrow width.
3. A measured mechanism for the failure of the logical reading: the
   churning wave code (population-composition drift over ever-sinusoidal,
   turning-over units), replicated across five architectures.
4. A depth dose-response showing Fourier structure is *amplified*, not
   extinguished, by depth (0.47 -> 0.54 -> 0.72), against the LIC
   bifurcation prediction, together with a trainability collapse (grok
   epoch 1.8-5k at one nonlinearity; 11.7-50.1k with 44% censoring at
   five) implying the logic regime is unreachable by standard training on
   this task.
5. A width result inverting the expected axis: narrowness, not depth, is
   the strongest quantization pressure (first above-criterion units,
   purity to 0.88), yet still yields no decision-grade logic.

## 2. Methods

### 2.1 Task and training protocol

Modular addition (a + b) mod p, p in {59, 97, 113}, train fraction 0.45,
full-batch AdamW, lr 1e-3, weight decay 1.0, CPU. The grok epoch tau is the
first checkpoint (20-epoch cadence) with validation accuracy >= 90%
(pinned in amendment 3; a 95% sensitivity re-derivation is computed from
stored accuracy series and reported alongside). Budgets: 40k epochs
(shallow, depth-3), 80k (depth-5), 60k (narrow), each fixed in advance
with rationale. Censored runs are first-class observations: excluded from
timing, included in structural diagnostics.

### 2.2 Architectures (all embeddings 128; hidden width 256 unless noted)

| family | hidden stack | params (p=59) | role |
|---|---|---:|---|
| mlp (SoftRose) | ReLU(W1 e) | 88.5k | phase-1 baseline |
| residual (ResidualRose) | W1 e + ReLU(W2 W1 e) | 154.3k | phase-1 "residual" family |
| noskip (NoSkipRose) | ReLU(W2 W1 e) | 154.3k | control: param-identical to residual, function-class-identical to mlp |
| deep (DeepRose, 3) | [Linear+ReLU]^3 | 220.1k | LIC bifurcation depth |
| deep5 (DeepRose, 5) | [Linear+ReLU]^5 | 351.7k | beyond the bifurcation depth |
| deepnarrow (3 @ width 64) | [Linear+ReLU]^3, width 64 | 36.2k | LIC-faithful narrowness |

3 seeds per (architecture x modulus) cell; 63 runs total (58 grokked,
5 censored). Every run logs per-example hidden states (final hidden layer
at 20-epoch cadence; earlier layers of deep stacks at 100-epoch cadence),
float16, with train/val accuracy series, to a documented .npz schema.
Training is bit-deterministic given the seed: an 18-run re-execution
reproduced all 18 grok epochs exactly.

### 2.3 Order parameters and unit-level statistics

- **D_var** (incumbent): 1 - E_c[Var(phi|c)] / Var(phi).
- **D_sheaf_B** (geometric leg): per checkpoint, a k-NN graph (k=10) on the
  hidden representations (labels never used in construction); Dirichlet
  edge energy contrasted against an edge-count-matched random graph;
  degenerate-collapse guard. Raw-width and PCA-64 variants (PCA basis fit
  at the grok epoch, fixed across time).
- **lambda_gap** (spectral bridge): the p-th eigengap of the graph
  Laplacian.
- **D_logic** (logical leg): the LIC measurement ported to prime moduli
  (amendment 4): proposition family = Fourier half-circle signs (k = 1..4,
  cos and sin) + quadratic residues; ternary unit states (bins +/- 1/3,
  per-checkpoint standardization); two-sided informativeness at purity
  0.80; conclusive groups (<= 3 units, majority vote, balanced accuracy
  >= 0.90); an example is decidable iff the covered-proposition verdicts
  intersect to exactly its class. Reported raw and normalized by the
  family's theoretical decidability ceiling (which falls with p).
- **fourier_concentration**: per unit, the dominant non-DC harmonic share
  of the class-conditional mean activation profile.
- **S-statistics** (unit-level, amendment 6): S1 identity continuity
  (Jaccard of structured-unit sets early vs. late), S2a within-unit
  overtone slope among early-structured units (overtone = third-to-first
  harmonic power ratio; square-wave reference 1/9), S2b overtone at
  structure onset for late-arriving units, S3 purity bimodality
  (2-vs-1-component GMM BIC).

**Timing metric** (fixed): half-maximum of a 4-parameter logistic fit;
R^2 >= 0.8 to classify; fitted amplitude always reported; a sanity gate
(amplitude in [0.05, 1.5x observed range], half-max inside the data span)
applies to all fits from amendment 4 onward. "Times grokking" = transition
within W = +/-10% of tau; family rule = median relative error <= 0.10.

### 2.4 Validation gates (all pre-data, all passing)

Sheaf: planted-transition timing within 10% (achieved 0.2%),
label-shuffle null, collapse flagged. D_logic: planted decidable code
timing within 10% (5.6%), shuffle null (0.000), Fourier-vs-QR split
measurable (8/8 vs 0). Fourier concentration: sinusoid > 0.9 (1.00),
random < 0.2 (0.14). S-statistics: a two-world gate — synthetic
"sharpening" (units morph sine -> square) and "replacement" (cells born
square while sinusoids fade) must both classify correctly; three
documented iterations of this gate forced refinements (clean transition
windows; structured-only retention masking; effect-size materiality) that
were recorded in the amendment before any real trace was analyzed.

## 3. Results

### 3.1 Scope: the inverted-scaling regime is architecture-universal

Grok time *decreases* monotonically from p=59 to p=113 in every family
(e.g. mlp means ~4430 -> ~2660 -> ~2130), placing the entire study in the
inverted (transition-zone) regime documented in
`analysis/SURVIVAL_FIT_RESULTS.md` (zone exponent a = -2.04 +/- 0.20,
censoring-aware). The pre-registration's assumption that these moduli
avoid the zone is void under the local protocol; all conclusions are
scoped accordingly. The zone's extension to p >= 113, in four independent
architecture families, is itself a replication-grade extension of the
phase-1 transition-zone discovery.

### 3.2 The incumbent fails everywhere; the sheaf leads everywhere

D_var classifies 27/27 shallow runs but lands within W in 0/27 (median
relative error 0.32-0.46 per family — always early). D_sheaf_B improves
on it in every family (mlp 0.080 vs 0.460; residual 0.224 vs 0.369;
noskip 0.192 vs 0.323), rejecting H0's comparative clause — categorical
structure *does* add timing power — while H1 (architecture-universal
within-W timing) fails. The stable ordering **D_var -> D_sheaf_B -> tau**
holds in essentially every classifiable run: variance collapse precedes
representation-graph gluing precedes deployment. Through phase 1's
(M, R, D): R-like structure forms first, gluing second, D last. The sheaf
transition is a *leading indicator* of grokking, typically 10-25% early.

### 3.3 Parametrization, not architecture

The v1 preview suggested an "MLP vs residual" split (sheaf silent in MLPs
at p=59, clean in residuals). A pre-declared control — NoSkipRose,
parameter-identical to the residual but function-class-identical to the
MLP (the two stacked matrices collapse to one) — behaved like the
residual: 9/9 classifiable sheaf transitions with robust amplitudes. The
split follows the *factorized parametrization* (its optimization/implicit
bias), not the skip connection and not the function class. Additionally,
MLP sheaf classifiability is modulus-dependent (0/3 at p=59, 3/3 at
p=113), so no single-modulus reading is reliable; and the one cell meeting
the registered timing standard (mlp raw, median 0.080) is
threshold-sensitive (0.134 under tau_95).

### 3.4 The logical leg: a null that carries its own mechanism

Across all 63 runs and every layer measured, **not one proposition is ever
covered by a decision-grade conclusive group** (D_logic = 0 everywhere;
all fits degenerate-gated). This is not a near-miss in shallow families:
max two-sided purity is 0.42-0.43, identical across mlp/noskip/residual
and flat across an 88k-352k parameter range. The unit-level S-statistics
identify the mechanism, uniform across families (45 shallow/deep cells +
18 deep5 + 18 narrow cells, every one "mixed/unresolved" under the
registered rule with both worlds failing their own signatures):

- structured units churn (S1 identity overlap median 0.17; deep 0.06);
- survivors never morph (S2a immaterial or negative in every cell);
- newcomers are always **born sinusoidal** (S2b onset overtone median
  0.012, maximum 0.032 — an order of magnitude below the square-wave
  reference 0.111);
- purity distributions are often bimodal but with sub-cell upper modes.

The representation is a **churning wave code**: the population's
statistics drift while its membership turns over, and no unit-level
quantization event ever occurs. Aggregate "sharpening" (Section 3.5) is
population-composition drift, not unit transformation.

### 3.5 Depth: amplification, stall, and collapse

Three hidden layers (the LIC bifurcation depth) moved the purity
diagnostic for the first time — a within-network depth gradient (l1 0.41,
l2 0.64, final 0.66, max 0.80) — but produced zero coverage. Five layers
*reversed* the approach: purity plateaus at 0.45-0.71 (max 0.71 < 0.80),
while trainability collapses — 4/9 runs censored at 80k epochs and the
grokkers take 11.7k-50.1k (vs 2.8-8.6k at depth 3, 1.8-5k shallow).
Meanwhile the class-profile Fourier concentration of grokked networks
rises monotonically with depth:

| depth (ReLUs) | 1 | 3 | 5 |
|---|---:|---:|---:|
| median concentration | 0.47 | 0.54 | 0.72 |

**Depth amplifies the wave code** — the clean opposite of the LIC
prediction that Fourier analysis vanishes as cells form. At depth 5 the
order-parameter timing machinery itself breaks (every logistic fit
pathological, amplitudes ~10^3): internal structure saturates within the
first few percent of training while deployment takes 10-40x longer. The
spectral eigengap, unclassifiable in all 27 shallow runs, becomes
classifiable in 4/9 depth-3 runs (firing early) — depth concentrates class
structure into forms the graph Laplacian can see, without ever quantizing
it.

### 3.6 Width: the strongest pressure, and the first coincident timings

The final registered arm — depth 3 at width 64 (~36k params) — closed the
program (D1 negative, 0/9 coverage) while producing its two most
remarkable measurements:

1. **First above-criterion units.** Three runs contain individual units
   crossing the 0.80 two-sided purity criterion (0.81, 0.82, 0.88) — the
   only such units in the study — yet none assemble into a conclusive
   group at balanced accuracy 0.90. The program-wide purity-max ladder:

   | shallow (256) | deep3 (256) | deep5 (256) | deep3 narrow (64) |
   |---:|---:|---:|---:|
   | 0.43 | 0.80 | 0.71 | **0.88** |

   **Width, not depth, is the dominant driver of unit quantization.**

2. **First coincident timings.** At p in {97, 113}, the sheaf transition
   lands *within* W in three runs (relative error 0.049, 0.059, 0.092),
   with D_var close behind — after 54 runs of everything firing early.
   Narrowness couples internal reorganization to behavioral deployment.

The churn signature replicates at width 64 (S2b <= 0.026), and Fourier
concentration remains elevated (+0.18 over shallow). One p=113 run never
trains at all (val 0.4%), showing narrowness has its own fragility edge.

## 4. Discussion

**The wave-code thesis.** Every instrument, at every trainable depth and
width, describes the same object: grokked modular arithmetic is carried by
a distributed, sinusoidal, class-structured code whose *population
statistics* strengthen (concentration and saturation rise with depth and
narrowness) while its *unit membership* churns, and which never — under
any registered condition — localizes into stable, quantized, decision-
grade logical units. Where the LIC program reports a depth bifurcation
from Fourier to logic on propositional tasks, modular arithmetic shows a
continuous, monotone *amplification* of the Fourier code with depth, a
quantization pressure that lives on the width axis instead, and a
training process that collapses before either axis reaches functioning
logic. If the bifurcation exists for this task family, it is not reachable
by standard AdamW training: grokking dies first.

**What the sheaf leg bought.** The label-free gluing parameter is the
first internal quantity in this program that (a) beats the variance
baseline in every family, (b) carries interpretable structure (its
strength tracks parametrization and depth), and (c) becomes coincident
with grokking under capacity pressure. The narrow-width coincidence is the
study's most suggestive positive finding: when representational capacity
per unit is scarce, internal consistency and behavioral deployment happen
together — as if the slack that lets wide networks organize long before
deploying is exactly what narrowness removes. This is a concrete,
falsifiable target for a successor study (width < 64; W-window tests at
larger p).

**Honest scope.** Everything here is within one protocol (full-batch
AdamW, wd = 1.0, train fraction 0.45), one task family, three moduli in
the inverted-scaling regime, three seeds per cell, CPU scale. The
registered verdicts are robust within that scope and unwarranted beyond
it. In particular the LIC comparison is a *transfer* of their measurement
to a task they did not study; our null constrains the generality of the
bifurcation, not its original report.

## 5. Threats to validity

- **Regime confound (disclosed, finding 1):** the study sits entirely in
  the inverted-scaling zone; order-parameter behavior in the true Fourier
  regime (p >> 113 under this protocol) is unmeasured.
- **Timing-metric fragility at extremes:** logistic timing fails on
  depth-5 traces (all fits degenerate); conclusions there are structural,
  not temporal. The amplitude-disclosure rule (amendment 3) is what makes
  these failures visible rather than silently wrong.
- **Small n:** 3 seeds per cell; the coincidence finding rests on 3
  within-W runs. Bootstrap CIs are reported but n is n.
- **Proposition-family dependence:** D_logic's family (Fourier signs + QR)
  is lean and fixed; a richer family could in principle reveal coverage
  the registered family misses — though the purity diagnostic, which is
  family-maximal, argues otherwise.
- **tau threshold:** one timing cell flips between tau_90 and tau_95;
  both are always reported.

## 6. Successor directions (out of scope, registered as such)

1. Width < 64 at depth 3: does the purity ladder extrapolate through the
   criterion to decision-grade groups, or does the trainability edge
   (the dead p=113 run) close in first?
2. Non-modular tasks — LIC's own propositional family — under this
   program's gates and unit-level statistics: does the born-square
   signature (S2b ~ 0.111) appear where cells were originally reported?
3. Optimizers/regularizers that survive depth (the trainability collapse
   gates everything at n_hidden >= 5).
4. The Fourier-regime replication at p >> 113, where the coordination
   burden scaling (phase 1) predicts qualitatively different kinetics.

## 7. Reproducibility

All training is bit-deterministic given the recorded seeds (verified
18/18 on a full re-execution). Run tables with the extended schema
(train_fraction, batch_size, steps_per_epoch, budget_epochs,
grokking_threshold) and per-run JSON sidecars are committed; per-example
traces (~29 GB) are regenerable exactly via
`scripts/train_grokking_traces.py`. The measurement suite lives in
`src/grokking_scaling_theory/` (sheaf_order_parameter, logical_cells,
sharpening, order_parameter_compare); every instrument's validation gates
run via `python -m`. The pre-registration and its seven dated amendments —
including every gate-forced refinement and every disclosure of what was
known when — are in `experiments/PHASE2_CATEGORICAL_ORDER_PARAMETERS.md`.

## References

- Belfiore, J. & Bennequin, D. *Topos and Stacks of Deep Neural Networks.*
  arXiv:2106.14587.
- Belfiore, J., Bennequin, D. & Giraud, X. *Logical Information Cells I.*
  arXiv:2108.04751.
- Hansen, J. & Ghrist, R. *Toward a Spectral Theory of Cellular Sheaves.*
- Bodnar, C. et al. *Neural Sheaf Diffusion.*
- Power, A. et al. *Grokking: Generalization Beyond Overfitting on Small
  Algorithmic Datasets.*
- Nanda, N. et al. *Progress Measures for Grokking via Mechanistic
  Interpretability.*
- Phase 1: `paper/OFFICIAL_GROKKING_RESEARCH_PAPER.md` (this repository);
  transition-zone kinetics in `analysis/SURVIVAL_FIT_RESULTS.md` and
  `analysis/HOUND_DOG_FINDINGS.md`.
