# A Latent-Variable Effective Theory of Grokking in Modular Arithmetic

## Abstract

We study grokking in modular arithmetic as a problem in delayed generalization and
effective dynamical theory. A previously identified empirical scaling law suggests that
grokking time `tau` depends on modulus `p` and weight decay `wd` approximately as

`tau = C * p^2 / ((log p)^k * wd^beta)`,

with a logarithmic correction exponent `k` near `1.5-2.0` and a sublinear regularization
exponent `beta` near `0.65`. Rather than treating this relation as a stand-alone fit, we
reconstruct it as a compressed summary of hidden representational dynamics. Using
trace-backed diagnostics across weight decay, modulus, neighboring task, and
architecture, we find that the strongest current mechanistic account is a delayed
transition from example-specific organization to rule-structured organization. In shallow
MLPs under AdamW, peak representational-transfer velocity tracks peak grokking velocity
almost exactly and survives stress tests across regularization, modulus, and modular
subtraction. This predictor is not architecture-invariant: residual networks can form
early rule structure and early output-readable prototypes while visible grokking still
lags. That failure isolates a second bottleneck, which is better described as delayed
deployment or consolidation of rule-structured representations across examples.
Thresholded deployment metrics improve the residual-family explanation but do not yet
yield a universal order parameter. The resulting theory is therefore not a completed
first-principles derivation, but a falsifiable effective framework based on three latent
variables: memorization burden, rule structure, and cross-example deployment.

## 1. Introduction

Grokking is the striking phenomenon in which a neural network achieves high training
accuracy long before it achieves high validation accuracy, followed by a comparatively
late and rapid transition to generalization. Modular arithmetic has become one of the
clearest settings for studying this effect because the task admits both algorithmic
structure and memorizing solutions.

The central scientific question is:

> What is the network waiting for before generalization becomes possible?

This question can be posed in several competing ways. Is the network waiting for a
correct rule to be discovered, for memorization to be erased, for a hidden harmonic
structure to phase-lock, or for a representation that already contains the rule to become
usable at the level of actual examples?

Earlier work in this repository identified a promising empirical scaling law for grokking
time as a function of problem size and regularization. That finding motivated a deeper
goal: transform a good fit into a defensible scientific framework with explicit latent
variables, rejected alternatives, known boundaries, and high-value falsification paths.

This paper reports the current synthesis of that effort.

## 2. Empirical Starting Point

The empirical candidate law is

`tau = C * p^2 / ((log p)^k * wd^beta)`.

At face value, this law says three things:

- there is a leading quadratic burden with modulus
- there is a logarithmic correction that accelerates learning relative to naive
  mean-field scaling
- weight decay shortens the grokking time through a sublinear channel

By itself, however, the law is not a theory. The same curve can be produced by many
different hidden mechanisms. The scientific problem is therefore to identify which parts
of the law are structural and which are contingent, and to determine what hidden process
the exponents are summarizing.

Our current stance is deliberately conservative:

- the law is a candidate effective description
- the exponents remain phenomenological
- the mechanistic content must come from independent diagnostics, not the fit alone

## 3. Minimal Effective Variables

The current experiments support a coarse-grained description in terms of three latent
variables:

- `M(t)`, memorization burden:
  example-specific structure that supports training accuracy without stable
  generalization

- `R(t)`, rule structure:
  degree to which internal representation is organized by the true task variable, such
  as modular sum class or modular difference class

- `D(t)`, cross-example deployment:
  degree to which rule-structured representation is sufficiently aligned across examples
  within the same rule class to drive correct behavior

This yields a minimal stopping-event view of grokking:

`tau = inf { t : G(R(t), M(t), D(t)) >= 0 }`

for some unknown effective criterion `G`.

This formulation is intentionally modest. It does not claim microscopic completeness. It
claims only that grokking time is better understood as the first time a latent
representational condition becomes true than as a direct function of loss alone.

## 4. Minimal Dynamical System

To move from latent-variable language to mechanism, we propose the following minimal
effective flow:

`dM/dt = -a1 * wd^beta * M`

`dR/dt = a2 * gamma_R(p) * R * (1 - R)`

`dD/dt = a3 * R * (1 - D) - a4 * M * D`

with positive coefficients `a1, a2, a3, a4`.

These equations are not presented as a microscopic derivation. They are the smallest
coupled system that captures the current empirical picture:

- `M(t)` decays under regularization and cleanup
- `R(t)` grows at an effective rate `gamma_R(p)` set by problem-size-dependent
  coordination burden and cooperative gain
- `D(t)` is helped by rule structure and hindered by residual memorization burden

Grokking is then defined by a stopping condition:

`tau = inf { t : D(t) >= D_crit }`

for a deployment threshold `D_crit`.

This is the central upgrade from a descriptive theory to a mechanistic one. It explains
why shallow MLPs can show near-coincident transfer and grokking, while residual
architectures can show early rule formation but late behavioral generalization: in the
residual case, `D(t)` decouples from `R(t)` for longer.

## 5. Competing Mechanism Hypotheses

Several plausible mechanism stories were considered.

### 5.1 Harmonic Locking

One early hypothesis was that grokking corresponds to a harmonic locking event in which
task-relevant internal modes become phase-aligned, analogous to a tuning fork entering a
clean resonant state. This was a strong intuition because modular arithmetic naturally
admits a Fourier description.

This hypothesis was useful because it forced a precise question about internal
coherence. However, the best harmonic coherence metrics tested did not behave like the
primary order parameter of the transition. They turned on too early or tracked the wrong
part of the process.

Conclusion: harmonic organization may be part of the story, but simple harmonic
coherence is not the leading explanatory variable.

### 5.2 Static Crossing

A second hypothesis was that grokking occurs when rule structure exceeds memorization
burden in a static sense.

This too was too simple. The static crossing often occurred too late, and in some
conditions it failed to give the most informative timing signal.

Conclusion: grokking is not well described by a single static threshold comparing two
quantities.

### 5.3 Dynamic Transfer

The strongest positive mechanism result emerged from a dynamic reinterpretation:
grokking is closely tied not to a static crossing, but to the **rate** at which
representational mass transfers from example-specific organization into rule-structured
organization.

In shallow MLPs under AdamW, peak representational-transfer velocity matched peak
grokking velocity extremely closely and survived several perturbations.

This became the leading local mechanism.

### 5.4 Consolidation and Deployment

When the same analysis was extended to residual networks, the transfer story no longer
fully explained the transition. Residual models could develop:

- early rule structure
- early output-readable prototypes
- early basin occupancy

while still showing delayed generalization.

This failure was highly informative. It suggested that rule discovery is not enough.
There is a second bottleneck involving whether examples have sufficiently consolidated
around the correct rule prototypes to make the representation behaviorally deployable.

## 6. Main Experimental Findings

### 6.1 Within-Family Robustness

Within the shallow MLP family under AdamW, the transfer mechanism survived:

- weight-decay sweeps
- a second modulus
- generalization from modular addition to modular subtraction

This is the strongest evidence in the project so far that a genuine latent mechanism has
been isolated, rather than merely an after-the-fact story attached to a fit.

### 6.2 Architecture Boundary

Transfer was not architecture-invariant. Residual models showed that early rule
structure does not guarantee early grokking.

This established an important boundary:

- some latent variables are family-local
- any serious theory must explain both the success in MLPs and the failure in residuals

### 6.3 Deployment Metrics

Prototype consolidation metrics improved the explanation of residual delay.
Thresholded deployment metrics, defined in terms of stricter example-level basin
confidence, improved it further within the residual family.

However, these metrics do not yet provide a fully unified invariant across both MLP and
residual cases. In MLPs they are often too early or too flat to act as the shared timing
signal on their own.

This is the current frontier of the mechanism work.

## 7. What Exactly Is D(t)?

The weakest variable in the current framework is also the most important one.

`D(t)` should not be read as vague "deployment sufficiency." The sharper interpretation
is:

> `D(t)` measures cross-example alignment of rule-structured representations.

Operationally, this means that examples sharing the same task rule should not merely
point toward the same prototype in aggregate. They should cluster tightly enough around
that rule representation to support reliable classification across the equivalence class.

A concrete family of observables is:

- low variance of hidden states within each rule class
- high margin between correct and nearest competing rule prototypes
- high fraction of examples lying deeply inside the correct prototype basin

In idealized form one can write:

`D(t) = 1 - E_class[ Var( phi(x, t) | class ) ]`

where `phi(x, t)` is the hidden representation at time `t`.

This sharpened definition explains the key residual result:

- rule structure can be present
- output readability can be present
- but cross-example deployment can still be incomplete

## 8. Current Effective Theory

The cleanest current theory statement is:

> Grokking is a kinetically delayed transition in which rule-structured representations
> form early but only produce generalization once they become sufficiently aligned and
> deployable across examples, under competing dynamics of memorization decay and
> structure formation.

This statement is preferred over earlier alternatives because it:

- explains the strong transfer result in MLPs
- explains why residual models can exhibit early rule structure without early grokking
- naturally incorporates regularization as cleanup or deployment assistance
- connects directly to measurable hidden-state observables

The theory is therefore simultaneously:

- mechanistic, because it posits latent variables and transition criteria
- systems-level, because it identifies a multistage invariant process
- skeptical, because it explicitly marks the point where current variables stop being
  universal

## 9. Phase Transition Interpretation

The phrase "phase transition" should be used carefully. The current evidence supports a
nonequilibrium, kinetically delayed transition rather than an equilibrium critical point.

The relevant ingredients are:

- control parameters:
  modulus `p` and weight decay `wd`

- order parameter:
  deployment `D(t)`

- hidden field:
  rule structure `R(t)`

Under this interpretation, grokking is not a sharp bifurcation in loss. It is a delayed
crossing of a deployment threshold after hidden structure has been forming for a long
time under competing cleanup and consolidation dynamics.

This is why the visible transition can be sudden while the hidden variables evolve over
much longer timescales.

## 10. Reinterpreting the Scaling Law

The scaling law

`tau = C * p^2 / ((log p)^k * wd^beta)`

should now be read as an **effective asymptotic summary** of the latent dynamics rather
than as a stand-alone law.

Current working interpretation:

- `p^2`:
  a baseline coordination burden that grows with the number of pairwise constraints or
  effective alignments required across the modular domain

- `(log p)^k`:
  an enhancement of `R(t)` growth associated with distributed rule formation across many
  weakly coordinated components

- `wd^beta`:
  a cleanup or deployment-assistance factor, plausibly arising from heterogeneous
  dissipation of memorizing modes

The key conceptual upgrade is:

> The scaling law is not fundamental. It is the emergent solution time of a coupled
> dynamical system over `(M, R, D)`.

Equivalently, the law emerges from the time required for rule structure to both form and
dominate deployment under competing decay and growth processes.

The log correction is therefore not just a better fit than a pure power law. It is
currently best understood as a signature of distributed rule formation. Likewise, the
sublinear regularization exponent is best understood as evidence that weight decay acts
through a broad cleanup spectrum rather than a single linear channel.

These interpretations remain provisional. What has improved is not that the exponents
have become final, but that the latent variables they summarize have become clearer.

### 10.1 Candidate Origin of the `log(p)^2` Term

The most plausible current mechanism for the logarithmic correction is a product of two
distinct logarithmic effects in Fourier space.

First, modular arithmetic admits a natural decomposition into Fourier characters on
`Z_p`. If rule formation is distributed across many weak task-aligned modes with an
effective envelope `a_m ~ 1/m`, then the coherent recruitment gain scales as

`H(p) = sum_{m <= p} 1/m ~ log(p)`.

Second, if coordination among those modes is governed by a marginal or nearly marginal
coarse-grained field, then the corresponding susceptibility can contribute another
logarithmic factor:

`chi(p) ~ log(p)`.

Combining a baseline coordination burden `p^2` with the cooperative gain `H(p) * chi(p)`
gives an effective rule-formation rate

`gamma_R(p) ~ log(p)^2 / p^2`.

This implies a rule-formation timescale

`tau_R ~ p^2 / log(p)^2`.

The key point is that the logarithmic factor should be interpreted as an enhancement of
the effective rule-formation rate relative to the quadratic baseline, not as an
additional drag term. This remains a heuristic mechanism rather than a closed RG
derivation, but it is now specific enough to generate direct empirical tests.

### 10.2 Controlled Asymptotic Derivation

The derivation target is:

`tau ~ p^2 / ((log p)^2 * wd^beta)`.

The cleanest route starts by separating cleanup and rule-formation dynamics.

First, the memorization equation is exactly solvable:

`dM/dt = -a1 * wd^beta * M`

so that

`M(t) = M0 * exp(-a1 * wd^beta * t)`.

This yields a cleanup timescale

`tau_M ~ wd^{-beta}`.

Second, the leading `p^2` burden cannot come from a logistic equation of the form
`dR/dt ~ R(1-R) / (log p)^k` alone. Such a flow only produces a logarithmic timescale.
Therefore the true size dependence must live in the effective rate itself:

`dR/dt = a2 * gamma_R(p) * R(1-R)`.

The controlled asymptotic claim is then

`gamma_R(p) ~ log(p)^2 / p^2`.

Under this rate law, the rule-formation timescale is

`tau_R ~ p^2 / log(p)^2`.

The physical origin of `gamma_R` is the two-log mechanism described above:

- one logarithm from harmonic accumulation of weak task-aligned modes
- one logarithm from marginal coordination susceptibility

The deployment variable is then governed by

`dD/dt = a3 * R * (1-D) - a4 * M * D`.

At late times, once `R` is large and `M` is decaying exponentially, `D` becomes the
gated release variable that determines when visible generalization occurs.

This means the exact grokking time is not generically a strict product of independent
timescales. In the full theory it is a stopping time of a coupled system. However, the
empirical data suggest an approximately separable regime in which:

- `p` primarily controls the rule-formation rate `gamma_R(p)`
- `wd` primarily controls cleanup through `M(t)`

In that regime the observed factorized ansatz

`tau ~ p^2 / ((log p)^2 * wd^beta)`

is a controlled effective approximation rather than an arbitrary fit.

## 11. What Has Been Ruled Down

A scientific framework gains strength not only by what it explains, but by what it
excludes.

The current work rules down the following as primary explanations:

- simple harmonic coherence as the main order parameter
- a purely static rule-versus-memorization crossing
- output readability alone as the trigger of generalization

These hypotheses were useful stepping stones. They are not the best current summary of
the data.

## 12. Limitations

Several limitations remain important.

First, the scaling-law dataset is still too small to justify strong universality claims.
The fit is promising, but the exponents should still be treated as under active test.

Second, the best current mechanism is architecture-sensitive. This is a feature for
scientific honesty, but it means the final order parameter has not yet been identified.

Third, optimizer robustness remains open. Initial SGD attempts did not cleanly enter the
same grokking regime, so optimizer dependence has not yet been resolved.

Fourth, the present dynamical system is intentionally minimal. It is constrained by the
current data, but it is still an effective model rather than a derived microscopic flow.

These limitations do not weaken the current framework. They define its correct scope.

## 13. Critical Experiments

Three experiments now have unusually high value.

### 13.1 Distinguish `log p` From `log^2 p`

Sweep modulus densely over a broader prime ladder and compare:

- `tau ~ p^2`
- `tau ~ p^2 / log p`
- `tau ~ p^2 / log^2 p`

Prediction:

- the stronger logarithmic correction should win at larger `p` if distributed rule
  formation is the right mechanism class

### 13.2 Break Deployment While Preserving Rule Structure

Construct a perturbation that preserves coarse rule prototypes but degrades
cross-example alignment within equivalence classes.

One route is to shuffle or scramble examples within rule-conditioned subsets in a way
that preserves class-level averages while disrupting within-class consolidation.

Prediction:

- `R(t)` can remain relatively intact while `D(t)` is damaged
- grokking should be delayed or disappear

### 13.3 Compare MLP and Residual Dynamics Directly

Measure `R(t)` and `D(t)` side by side in matched MLP and residual runs.

Prediction:

- MLP: transfer and deployment timing should remain tightly coupled
- residual: `R(t)` should rise early while `D(t)` lags substantially

These experiments directly test whether the proposed phase-transition variables are the
right ones.

## 14. Research Base

This work sits at the intersection of five bodies of science.

### 14.1 Renormalization Group

RG contributes the language of leading scaling, logarithmic corrections, and effective
flows. The present contribution is not a formal RG derivation, but an RG-like treatment
of learning-time corrections in a non-equilibrium training process.

### 14.2 Statistical Mechanics of Learning

This literature contributes phase transitions, order parameters, and delayed
generalization phenomena. The distinctive addition here is the separation of rule
formation from rule deployment.

### 14.3 Neural Scaling Laws

Modern scaling-law work mostly studies loss, model size, data size, and compute. The
present contribution is different: time-to-generalization scaling for a symbolic task,
with explicit mechanistic variables.

### 14.4 Dynamical Systems

The coupled `(M, R, D)` formulation places grokking in the language of competing flows,
metastability, and delayed threshold crossing.

### 14.5 Representation Learning Theory

This work also belongs to feature learning and implicit-bias theory, but adds an
explicit third variable. Representation can exist without immediate behavioral
deployment, and that separation is one of the central empirical contributions of the
project.

## 15. Research Program

The next stage of the work should proceed along three connected lines.

### 15.1 Validation

Expand the empirical dataset:

- more moduli
- denser weight-decay sweeps
- more seeds
- explicit treatment of censored no-grokking runs

### 15.2 Formalization

Analyze the minimal dynamical model over `(M, R, D)` and ask what asymptotic form of
`tau` it induces, including whether an approximate `p^2 / log^2 p` regime can be
recovered.

This is a better next formal step than inserting additional correction factors into a
one-variable ansatz.

### 15.3 Generalization

Test whether the same latent-variable framework survives:

- additional algorithmic tasks
- more architectures
- optimizer variation once comparable grokking regimes are found

## 16. Implications for the Isomorphic Library

This work also serves as a serious test case for the broader `isomorphic` project.
It shows that the library is most credible not when it declares deep structural unity in
advance, but when it supplies a disciplined workflow for finding recurring latent
processes, rejecting attractive wrong hypotheses, and isolating where a candidate
invariant breaks.

In that sense, the grokking case study is not merely an application. It is one of the
first places where the library begins to justify itself scientifically.

## 17. Conclusion

The strongest current conclusion is not that we have solved grokking, and not that we
have proven a universal renormalization theory of learning.

The strongest current conclusion is narrower and more useful:

we now have a principled effective theory in which grokking is a delayed
representational transition governed by memorization burden, rule formation, and
deployment sufficiency.

That theory has:

- one strong positive mechanism result
- several explicitly rejected alternatives
- a known architecture boundary
- a coherent path back to the scaling law

That is enough to stop local mechanism hunting for now and move into the next scientific
phase: validation, formalization, and communication.
