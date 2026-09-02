# DTBR-MC v0.2 — H3 Specification (Pre-Registered)

**Status:** pre-registration. Functional forms and falsification conditions below
are fixed *before* any output is inspected. No parameter, form, or threshold in
this document may be changed to make a hypothesis pass. Changes after the first
run must be logged as amendments with a dated rationale, and any result obtained
under an amended form is reported as exploratory, not confirmatory.

**One-line purpose:** rebuild the deep-time intrusion model so that every
structural element traces to a *measured* behavioral primitive, the headline
hypothesis is falsifiable stage-by-stage, and the deep-time extrapolation is
quarantined from the contemporary anchors.

---

## 0. Epistemic frame (binding on all outputs)

Every quantity this model emits carries exactly one of three labels, and the
label travels with the number into every figure, table, and sentence:

- **AUDIT** — a deductive consequence of the construction. Tells us which of our
  assumptions are load-bearing. Makes no claim about the world.
- **EXTRAPOLATION** — a conditional forecast resting on contemporary measured
  primitives, valid only under explicitly stated assumptions.
- **CARTOGRAPHY** — a possibility result ("regime X is reachable when Y"),
  asserting reachability, never actuality.

The v0.1 post-mortem is the reason this section exists: H1's "direction-only"
verdict was a clean AUDIT mislabeled as a forecast, and the severity-dominance
result was the objective function admiring itself (AUDIT) mislabeled as an
empirical recovery. The label is not optional metadata; it is the result.

**The one rule that prevents the v0.1 failure from recurring:** added structure
is licensed *only* by a measured drop-out or sign-flip in the literature. Every
stage and moderator in §3–§5 maps to a row in the provenance table (§10). A term
with provenance "ASSUMPTION" may exist but may never be point-estimated, may only
be swept, and contaminates any output it touches with the CARTOGRAPHY label.

---

## 1. What changed from v0.1, and why (falsification ledger)

1. **H1 was unfalsifiable in the model class it was tested in.** In any additive
   intervention model the lever margin (slope_SC − slope_PC) is a quotient of
   fixed coefficients, hence constant in interpretive capacity — a *theorem*, not
   a finding. A capacity threshold requires a non-separable lever×capacity
   interaction the form did not contain. v0.2 must therefore be explicitly
   non-additive and the interaction must be named, not latent.

2. **The moderator was on the wrong axis.** The warning-compliance literature
   (perceived hazard → avoidance; graphic/fear cues do *not* backfire for
   non-acquisitive actors) and the 5,000-year grave-robbing record (elaborate
   defense → *more* intrusion for acquisitive actors) jointly locate the
   sign-flip on **acquisitive motive × behavioral stage**, not on interpretive
   capacity.

3. **Backfire is two mechanisms, both upstream, both attenuating.**
   *Value-signaling* (defense read as evidence of worth; acquisitive actors) and
   *reactance / forbidden-fruit* (prohibition raises curiosity; near-universal).
   The reactance effect is robust at the curiosity/attention stage (FitzGibbon
   et al., card-selection, N≈2,141, ages 5–79) but attenuates toward null at the
   level of consummatory action (media-ratings field studies show no consistent
   behavioral effect). A single monotone curiosity→drive→intervention chain
   cannot represent this; the model must have stages with attrition.

4. **Severity is the weak deterrent lever; perceived certainty is the strong
   one; the referent lacks certainty.** Becker's expected-utility model plus the
   criminological certainty-over-severity aphorism imply deterrence runs through
   *perceived certainty of personal consequence*, and that the risk-seeking
   minority who actually intrude are the population least moved by severity. A
   buried radiological hazard supplies almost no perceptible certainty (harm is
   latent, invisible, delayed), which is precisely the near-undeterrable regime.
   The model must therefore (a) introduce a `perceived_certainty` quantity, (b)
   decouple *severity-of-harm* (objective sizing) from *severity-of-perceived-
   consequence* (deterrence), and (c) cap achievable certainty by a referent
   ceiling that is a property of the hazard, not of the message.

---

## 2. Hypothesis H3 (falsifiable)

> **H3.** The effect of a phenomenological/defensive cue on intrusion is not
> single-signed and is not primarily moderated by interpretive capacity. It is
> moderated by (i) the actor's **acquisitive appraisal** and (ii) the
> **behavioral stage** at which the outcome is measured. Cues that raise
> perceived hazard *deter* non-acquisitive actors; the same cues, read as
> evidence of value, *attract* acquisitive actors. A near-universal reactance
> term raises curiosity for everyone but attenuates across the
> notice→appraise→act funnel. Deterrence operates through **perceived certainty
> of personal consequence**, not severity; and because a radiological referent
> has an intrinsically low certainty ceiling, intrusion by the risk-seeking
> minority is weakly deterrable regardless of marker design.

H3 decomposes into four sub-claims, each tied to a literature and each with a
pre-registered falsification condition (§7):

- **H3a (motive moderates sign).** ∂P(act)/∂PC < 0 for low acquisitiveness and
  > 0 for high acquisitiveness, with a crossover in acquisitiveness — *at
  parameter values bounded by data*, not arbitrary ones.
- **H3b (reactance is upstream and attenuates).** The reactance contribution is
  large at the notice/appraise stages and ≈0 at the act stage, with a single
  attenuation parameter reconciling the robust curiosity effect and the near-null
  field-behavior effect.
- **H3c (certainty ≻ severity; ceiling binds).** ∂P(act)/∂(signal certainty)
  exceeds ∂P(act)/∂(perceived severity) in magnitude, and behavior is insensitive
  to signal certainty pushed above the referent ceiling; under a low (radiological)
  ceiling the risk-seeking minority is weakly deterrable.
- **H3d (coupling yields genuine emergence).** Any regime shift produced by
  social-modeling coupling survives comparison against a matched-marginal
  coupling-free null.

---

## 3. Architecture: the C-HIP funnel

Replace the single intervention scalar with a staged information-processing
funnel (after the communication–human-information-processing model that organizes
the warning literature). Each stage emits a pass-probability in [0,1]; the
population thins at every stage, matching the measured 88%→46%→27% noticing→
reading→complying attrition.

```
encounter → notice → comprehend → appraise → intend → act
   p_enc      p_not     c          (value vs deterrence)   p_int   p_act
```

- Stages compose so that the probability of reaching the act decision is the
  product of upstream pass-probabilities; this guarantees attrition and gives
  each measured drop-out a home.
- `comprehend` is the *only* stage gated by interpretive capacity. This is the
  demotion H1 earned: capacity acts on understanding, not on the lever balance.
- `appraise` is a Becker expected-utility competition between perceived value and
  perceived deterrence (§5). This is where both backfire channels live.
- `act` carries the social-modeling coupling term (§5), the only route to
  non-deductive output.

New variables introduced (and *why*, with provenance in §10):

| variable | meaning | why it exists |
| --- | --- | --- |
| `acquisitiveness` | agent's motive to extract value | the empirical sign-flip moderator (warning vs looting) |
| `signal_certainty` | how strongly markers assert *certain, immediate* consequence | the strong deterrent lever (criminology) — a design choice |
| `referent_certainty_ceiling` | max achievable perceived certainty given the hazard's nature | encodes that radiological harm is latent/invisible/delayed |
| `perceived_certainty` | appraised certainty after comprehension, capped by the ceiling | the operative deterrent quantity |

`repository_severity` is retained but **re-scoped**: it sizes *harm* in the
objective (§6) and must not be reused as the deterrence term. Perceived
consequence-to-self is a separate, comprehension-gated, certainty-capped quantity.

---

## 4. Variables (∈ [0,1] unless noted)

**Agent (carried from v0.1):** curiosity, risk_tolerance, technical_capability,
interpretive_capacity (IC), institutional_strength, prestige_sensitivity,
economic_pressure, ritualization_tendency. **Added:** acquisitiveness.
Explorer upper-tail minority retained — now interpreted as the *risk-seeking
offender* subpopulation (Becker: certainty-aphorism ⇒ intruders are risk-seeking),
and explorer boosts should load on risk_tolerance and acquisitiveness.

**Environment (carried):** visibility, accessibility, resource_attractiveness,
marker_clarity (SC), phenomenological_caution (PC), artificial_intentionality,
prestige_risk, historical_memory, repository_severity (now = harm magnitude).
**Added:** signal_certainty, referent_certainty_ceiling (a hazard property, set
LOW for the radiological baseline).

A composite, used below: `defense_level = aggregate(PC, artificial_intentionality,
prestige_risk, marker effort)` — how heavily the site reads as *defended*. This is
the quantity an acquisitive actor mines for evidence of value.

---

## 5. Functional forms (FIXED — pre-registered)

All forms are configurable in code so the literal/alternative readings remain
reinstatable (the v0.1 discipline), but the forms below are the registered
defaults under which H3 is tested. Weights shown are placeholders to be set from
§10 provenance and the calibration step; the *forms* are what is pre-registered,
not the exact weights.

**5.1 Encounter / notice.**
```
p_encounter = w1·visibility + (1−w1)·accessibility
p_notice    = clip( base_notice + k_con·conspicuity − k_load·cognitive_load )
```
conspicuity rises with marker prominence/artificial_intentionality; baseline
high (warning lit: ~88% notice).

**5.2 Comprehension (the only capacity-gated stage).**
```
comprehension c = w_ic·IC + (1−w_ic)·marker_clarity      # default w_ic = 0.7
```

**5.3 Appraisal — perceived value (where backfire lives).**
```
material_value   = resource_attractiveness · acquisitiveness
value_signaling  = γ · defense_level · acquisitiveness            # channel 2: defense ⇒ inferred worth
info_reward      = δ · mystery,   mystery = PC · (1 − c)          # channel 3: curiosity-as-reward
reactance_bump   = ρ · prohibition_salience                       # channel 3: forbidden-fruit
perceived_value  = material_value + value_signaling + info_reward + reactance_bump
```
- `value_signaling` is the formal statement of the looting result. **Anti-
  tautology guard:** its existence is not evidence for H3a; only the *bounded*
  magnitude of γ (from looting base rates, §10) and whether the empirically
  required sign pattern emerges *within that bound* count as a test (§7).
- `info_reward` + `reactance_bump` are the upstream curiosity channels; they are
  attenuated at the act stage by §5.6.

**5.4 Appraisal — perceived deterrence (certainty, not severity).**
```
perceived_certainty = min( referent_certainty_ceiling,
                           c · (cert_base + κ·signal_certainty) )
perceived_consequence = c · repository_severity          # comprehension-gated read of self-harm
perceived_deterrence  = w_det · perceived_certainty · perceived_consequence
```
The `min(…, ceiling)` is the apex inference made mechanical: no marker design
(`signal_certainty`) can push deterrence past what the *referent* can make
certain. Radiological baseline sets `referent_certainty_ceiling` LOW.

**5.5 Intent — Becker expected utility with risk attitude.**
```
EU       = perceived_value − perceived_deterrence − k_cost·(1 − accessibility)
p_intend = logistic( a·(EU − θ(risk_tolerance)) )
```
θ is the decision threshold, *lowered* by risk_tolerance (risk-seeking actors
intend at lower EU). This is where the explorer/offender minority does its work.

**5.6 Act — stage attenuation of curiosity + social coupling.**
```
# curiosity channels mostly fail to reach action (media-ratings near-null):
value_at_act   = perceived_value − (1 − α)·(info_reward + reactance_bump)
# α ∈ [0,1], α small ⇒ strong attenuation; α is bounded toward small (§7 H3b)
p_act_solo     = p_intend · opportunity(accessibility, technical_capability)
p_act          = clip( p_act_solo · (1 + λ · neighbor_act_fraction) )   # social modeling
```
`λ` (coupling strength) is direction-anchored (social modeling is robust) but
magnitude-unanchored → ASSUMPTION → swept, never point-estimated. `neighbor_act_
fraction` requires an agent graph (lattice or random graph; the graph family is
itself an identifiability variable, §7 H3d).

---

## 6. Objective and outcomes

```
intervention_score = p_act                         # in [0,1]
outcome ∈ {AVOID, OBSERVE, PRESERVE, INVESTIGATE, EXCAVATE}   # thresholds as v0.1
disturbance = outcome ∈ {INVESTIGATE, EXCAVATE}
E[H] = P(encounter) · P(reach act ∧ disturb) · repository_severity
```
`repository_severity` here is **harm magnitude only**. It must not appear in any
deterrence term. The v0.1 sensitivity result (severity dominates E[H]) is expected
to recur and is to be labeled AUDIT — it is a property of the multiplicative
objective, not a behavioral finding.

---

## 7. Falsification conditions (the heart of the pre-registration)

Each sub-claim states what output would prove it WRONG. If the falsifying pattern
appears, H3 (or that limb) is reported as falsified — no reframing to rescue it.

**H3a — motive moderates the sign.**
- *Test:* sweep PC at fixed comprehension across `acquisitiveness` ∈ [0,1], with γ
  fixed at its data-bounded value (§10).
- *Predicted:* ∂P(act)/∂PC < 0 at low acquisitiveness, > 0 at high, with an
  interior crossover.
- *FALSIFIED if:* the sign of ∂P(act)/∂PC is independent of acquisitiveness (no
  crossover), **or** a crossover appears only when γ is set outside its
  data-bounded range. (The second clause is the anti-tautology test: building
  `value_signaling` in does not count as confirming H3a.)

**H3b — reactance is upstream and attenuates.**
- *Test:* calibrate α so the act-stage reactance contribution matches the field
  near-null; independently check the appraise-stage contribution matches the
  robust curiosity effect.
- *FALSIFIED if:* no single α reconciles both (robust upstream, null at action) —
  i.e., the funnel cannot simultaneously honor FitzGibbon and the media-ratings
  result. That would mean the staged structure is wrong.

**H3c — certainty ≻ severity; ceiling binds.**
- *Test:* compare |∂P(act)/∂signal_certainty| vs |∂P(act)/∂perceived_consequence|;
  sweep signal_certainty above and below the ceiling; run the radiological
  (low-ceiling) parameterization against the risk-seeking minority.
- *FALSIFIED if:* the severity lever dominates the certainty lever (contradicts
  CAP), **or** behavior keeps responding to signal_certainty pushed above the
  ceiling (ceiling not binding ⇒ modeling error), **or** the low-ceiling
  radiological case is nonetheless easily deterred (would refute the apex
  inference that the referent, not the message, is the binding constraint).

**H3d — coupling yields genuine emergence.**
- *Test:* every regime shift (any qualitative change in the disturbance surface)
  is re-run against a coupling-free null with matched stage marginals (λ = 0,
  marginals frozen).
- *FALSIFIED (coupling adds nothing) if:* the null reproduces the shift. Then the
  result was deductive (AUDIT), not emergent, and must be relabeled.

**Identifiability protocol (applies to every claim).** For each contested stage,
implement ≥2 defensible alternative functional forms (e.g., logistic vs.
piecewise-linear intent; lattice vs. random-graph coupling). If the alternatives
yield observationally indistinguishable output, report the **non-identification**
and refuse to privilege one form. The v0.1 cross-model sign-disagreement
(baseline flat, backfire +, linear −) is the template: divergent structures with
equal plausibility is a result about the model's limits, not a menu to pick from.

---

## 8. Calibration protocol

- Parameters are bounded, not point-fit, wherever the anchor is a *qualitative*
  or *small/contested* effect. Bounds come from the provenance table (§10).
- A parameter whose only anchor is "direction known, magnitude unknown" (λ
  coupling, all deep-time decay rates) is **ASSUMPTION**: swept across its full
  plausible range, never set to a single value, and every output that depends on
  it is labeled CARTOGRAPHY.
- The deep-time horizon mapping (how IC, historical_memory, and the referent's
  perceived properties decay over 10⁴ years) is the master ASSUMPTION. No data
  touches it. It is represented as a family of decay scenarios, and *no single
  scenario is ever presented as the forecast*.

---

## 9. Pre-registered experiments

| id | question | levers swept | falsification | label |
| --- | --- | --- | --- | --- |
| 002 | Does motive flip the cue sign? | PC × acquisitiveness (γ fixed, bounded) | H3a | EXTRAPOLATION |
| 003 | Is reactance upstream-only? | reactance contribution by stage; fit α | H3b | EXTRAPOLATION |
| 004 | Certainty vs severity; ceiling | signal_certainty, perceived_consequence, ceiling | H3c | EXTRAPOLATION |
| 005 | Does coupling emerge? | λ vs matched-marginal null; graph family | H3d | CARTOGRAPHY |
| 006 | Deep-time horizon | decay-scenario family over 10⁴ yr | none (assumption sweep) | CARTOGRAPHY |

Experiment 004 carries the headline claim and should be reported first. Every run
reproduces the v0.1 hygiene: seeded determinism, common random numbers across
swept cells, Poisson-bootstrap CIs, and the identifiability check from §7.

---

## 10. Provenance table (every structural element → its license)

| element | functional role | provenance | status |
| --- | --- | --- | --- |
| staged funnel + attrition | core architecture | warning-compliance C-HIP; 88→46→27% attrition | ANCHORED (qualitative) |
| comprehension gate (IC, SC) | gates understanding only | warning lit: comprehension is a bottleneck | ANCHORED (qualitative) |
| hazard→avoidance brake | perceived danger deters | warning lit: compliance rises with perceived hazard | ANCHORED (qualitative) |
| value_signaling (γ) | defense ⇒ inferred worth | grave-robbing record (elaboration ⇒ more looting) | ANCHORED (bound only) |
| info_reward + reactance (δ, ρ) | curiosity backfire upstream | FitzGibbon forbidden-fruit (curiosity measures) | ANCHORED (upstream only) |
| attenuation α | curiosity fails to reach action | media-ratings field null vs lab curiosity | ANCHORED (bracketed) |
| Becker EU at appraise | value vs expected cost | Becker 1968; subjective-EU criminology | ANCHORED (form) |
| certainty ≻ severity | deterrent weight on certainty | certainty-aphorism (Nagin, Paternoster); small/contested | ANCHORED (direction; weak/contested magnitude) |
| risk-seeking intruders (θ↓) | explorer minority intends sooner | Becker: CAP ⇒ offenders risk-seeking | ANCHORED (qualitative) |
| referent_certainty_ceiling | caps deterrence below hazard's reality | inference from latency/invisibility of radiological harm | REASONED, not measured → treat as ASSUMPTION when set low |
| coupling λ | social modeling at act stage | social-modeling robust in direction only | ASSUMPTION (magnitude) |
| deep-time decay (IC, memory, perception) | 10⁴-yr horizon | none | ASSUMPTION (master) |

---

## 11. What this spec does not claim

- It does not predict the behavior of any actual future society. The deep-time
  layer is a scenario family, not a forecast.
- It does not estimate the behavioral primitives; it imports their *direction and
  rough magnitude* from contemporary, mostly WEIRD, short-horizon studies, several
  with small or contested effects.
- It does not resolve whether the certainty-aphorism holds as a clean elasticity;
  it adopts it as a bounded, falsifiable assumption (§7 H3c).
- A "regime exists" result (CARTOGRAPHY) is never to be read as "this will
  happen." The label is the result.

---

*End of pre-registration. First amendment, if any, goes below this line with date
and rationale.*

---

## Amendment 1 (2026-06-15) — phenomenological caution needs a brake pathway

**Trigger:** first run of the H3a motive-moderator test. Under the pre-registered
§5.3–5.4 form, `phenomenological_caution` (PC) fed *only* the value/backfire
channels (defense-signaling, mystery, reactance) and entered no deterrence term.
Result: `slope_pc > 0` at every acquisitiveness level — PC could not brake, so the
predicted brake→backfire crossover was structurally impossible. The registered
form was under-specified relative to the warning-label anchor (comprehended
hazard deters).

**Change:** add a comprehended-dread brake. `hazard_salience = PC · comprehension`
is added to `perceived_deterrence` (both deterrence forms), weighted by
`dread_weight` (default 0.4). The complement `mystery = PC · (1 − comprehension)`
stays on the backfire side. Same cue, opposite effect by comprehension: dread for
those who understand it, mystery for those who don't. The dread channel is also
the one route that partly bypasses the certainty ceiling (immediate affect rather
than cognitive risk appraisal) — which is what phenomenological warnings are for.

**Status of results obtained after this amendment:** EXPLORATORY, not
confirmatory. Post-amendment H3a shows a brake→backfire crossover at
acquisitiveness ≈ 0.82 with γ inside its [0,1] bound, and Exp 004 (H3c) is
unaffected. `dread_weight` is bounded-not-fitted and the crossover location
depends on it; it is not presented as a measured quantity.

