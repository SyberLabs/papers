# The Marker Is Not the Binding Constraint

### Physical substrate dominates symbolic design in deep-time hazard deterrence: a falsification-oriented Monte Carlo study

*Working paper — DTBR-MC program. Draft for internal review. All quantitative
claims carry an epistemic-status label (§2); bibliographic details in §References
require verification against primary sources before any submission.*

---

## Abstract

For four decades the problem of warning distant-future societies away from buried
nuclear hazards has been framed as a problem of *signs*: which markers, monuments,
myths, or institutions could carry meaning across ten millennia. We report a
falsification-oriented modelling program that began by testing a representative
hypothesis from that tradition — that below a threshold of interpretive capacity,
phenomenological caution (dread-inducing design) deters intrusion more effectively
than semantic clarity (legible warning) — and ended by relocating the problem
entirely. The capacity-threshold hypothesis is shown to be not merely false but
*unfalsifiable in the model class that motivated it*: in any additive behavioural
model the relative potency of two communication levers is a ratio of constants,
independent of capacity, so a threshold is algebraically impossible. Anchoring the
behavioural primitives to contemporary measured literatures (warning compliance,
the archaeology of grave-robbing, reactance and curiosity, and the criminology of
deterrence) relocates the operative moderator from *interpretive capacity* to the
*acquisitive motive of the actor* and the *behavioural stage* at which an effect is
measured. Three structurally independent experiments then converge on a single
result that was not designed into any of them: the binding constraint on expected
harm is physical, not symbolic. Severity of the hazard, the certainty ceiling
imposed by the hazard's own latency and invisibility, and the physical opportunity
(access and capability) required to disturb a site each dominate, in turn, the best
available communicative lever. We argue this convergence reflects a difference of
logical type rather than magnitude: physical factors enter the harm function as
*necessary multiplicative gates*, while symbolic factors enter as *modulators*
within the space those gates permit — and no modulator can move an outcome past a
gate. We are explicit that the simulator itself is, in its current form, a
consistency auditor rather than a predictive instrument: a coupling/contagion
experiment designed as the decisive test of whether the model can produce
genuinely emergent (non-deductive) behaviour returned a clean negative. The
contribution is therefore the reframing, audited for internal consistency, not a
forecast. The practical implication for marker design is direct: the symbolic
program optimises a second-order term, and the one symbolic move that survives the
analysis — manufacturing immediate affective certainty of consequence to partly
bypass the hazard's certainty ceiling — works only for non-acquisitive actors who
comprehend it, and backfires for the acquisitive few.

---

## 1. Introduction

The United States' Waste Isolation Pilot Plant and comparable deep geological
repositories pose a communication problem with no precedent in human practice: to
deter intrusion for on the order of ten thousand years, across a gulf in which
language, writing systems, institutions, and even the cognitive baseline of the
receiver cannot be assumed stable. The canonical responses are by now familiar —
the Human Interference Task Force convened to study marker feasibility; Sebeok's
proposal of a self-renewing "atomic priesthood" to carry interpretive authority;
the expert panels that produced menacing-earthwork concepts and the now-iconic
draft inscription insisting that the place is "not a place of honor"; and at the
fringe the "ray cat" proposal to breed animals that visibly react to radiation.
What unites these proposals is a shared premise: that the difficulty is one of
*meaning transmission*, and that the solution, if one exists, is a better sign.

It is worth stating at the outset what the present study is and is not. It is not
an empirical study of future human behaviour, which is unobservable. It is a
modelling program whose value lies in two places: first, in forcing a vague
hypothesis into a form precise enough to fail; and second, in anchoring its
behavioural assumptions to the contemporary literatures where the relevant
primitives *are* observable, and then asking what those primitives imply for the
deep-time case under explicitly labelled assumptions. The program's most important
output is negative and structural, and we have organised the paper to make the
honesty of that output auditable rather than rhetorical.

We take as our point of departure a hypothesis representative of the symbolic
tradition's more sophisticated wing, which we label H1:

> **H1.** Below a threshold of interpretive capacity, increasing phenomenological
> caution (PC) reduces intervention more than increasing semantic clarity (SC).

H1 is attractive because it captures a real intuition: that for receivers who
cannot read our warnings, a visceral, architecturally encoded sense of dread might
succeed where text fails. The remainder of this paper records what happened when we
tried, in earnest, to falsify it.

---

## 2. Methods I: an epistemic framework

A model with no free parameters fit to data cannot tell its author anything the
author did not put in; its outputs are deductive consequences of its construction.
This is not a defect to be hidden but a property to be tracked, and the central
methodological commitment of this program is that every result is assigned one of
three labels, and the label travels with the result into every claim:

- **AUDIT** — a deductive consequence of the model's construction. It reveals which
  of the author's assumptions are load-bearing; it makes no claim about the world.
- **EXTRAPOLATION** — a conditional forecast resting on contemporary measured
  primitives, valid only under stated assumptions about their persistence.
- **CARTOGRAPHY** — a reachability result ("regime *X* obtains when parameter
  *Y* exceeds *Y\**"), which asserts possibility, never actuality.

The discipline is not cosmetic. As §4 shows, the program's first apparent
"finding" — that one communication lever dominated another — was AUDIT mislabelled
as discovery, and recognising this was the hinge of the entire study. We adopt three
further safeguards. **Pre-registration:** functional forms and falsification
conditions are fixed before output is inspected; post hoc changes are logged as
dated amendments and any result obtained under an amended form is reported as
exploratory. **The anti-tautology rule:** added model structure is licensed only by
a measured drop-out or sign-flip in the literature, and a built-in mechanism never
counts as evidence for the hypothesis it encodes — only the *bounded magnitude* at
which it produces the predicted pattern can count. **Identifiability checking:** for
every contested functional form we implement at least two defensible alternatives;
if they yield observationally indistinguishable output, we report the
non-identification rather than privileging the prettier form.

---

## 3. Methods II: the model

### 3.1 Objective

The quantity of interest is expected harm per encountered agent,

E[H] = P(encounter) × P(intervention | encounter) × Severity,

evaluated over a Monte Carlo population of heterogeneous agents drawn against a
heterogeneous site. The implementation is vectorised, seeded for exact
reproducibility, and accompanied by a regression suite (37 tests in the released
code). All sampling, the behavioural models, the metrics, and the experiments are
modular and registered by name so that any equation can be replaced and the
literal alternative reinstated.

### 3.2 The behavioural funnel

The intervention probability is not a single equation but a staged
information-processing funnel adapted from the Communication–Human Information
Processing (C-HIP) framework that organises the warning-compliance literature:

> encounter → notice → comprehend → appraise → intend → act,

with attrition at every stage. This structure is itself an empirical commitment:
warning-compliance studies document large stage-wise drop-out (a representative
sequence falls from roughly nine in ten noticing a warning, to under half reading
it, to roughly a quarter complying), and a single monotone "drive minus caution"
equation cannot represent attrition. Comprehension is the only stage gated by
interpretive capacity — the demotion that H1, as we will see, earns. The appraisal
stage is an expected-utility competition, after Becker, between perceived value and
perceived deterrence; both the backfire channels and the deterrence channels live
there.

### 3.3 Behavioural primitives and their anchors

Each model term is licensed by a measured contemporary literature, summarised here
and developed in §4:

| Model term | Empirical anchor | Direction taken from data |
| --- | --- | --- |
| hazard → avoidance (brake) | warning-compliance research | perceived danger raises compliance |
| stage-wise attrition | C-HIP / warning funnels | large drop-out at each stage |
| value-signalling (defense → worth) | archaeology of grave-robbing | elaboration increased looting |
| curiosity / forbidden-fruit backfire | reactance & curiosity research | prohibition raises curiosity (upstream) |
| curiosity attenuation to action | media-ratings field studies | curiosity effect ≈ null at behaviour |
| expected-utility appraisal | Becker's economics of crime | act when reward exceeds expected cost |
| certainty ≻ severity (deterrence) | perceptual-deterrence criminology | certainty deters; severity weakly (small, contested) |
| risk-seeking intruders | implication of the certainty aphorism | offenders are disproportionately risk-seeking |
| social modelling (coupling) | warning research | people model others' (non)compliance |

### 3.4 The equation-interpretation note

A methodological hazard inherited from the original specification deserves explicit
statement, because all quantitative results are conditional on its resolution. The
source equations were written with multiplication between every term; read
literally this produces a degenerate product in which caution *increases*
intervention, inverting the research question. Because four of five equations have
coefficients summing to unity and the question requires caution to brake, the
coefficients are read as weighted linear combinations with caution as a
multiplicative brake. This is a documented choice, not a silent one; the forms are
configurable and the literal reading is reinstatable.

---

## 4. Results: a sequence of refutations

We present the results as the falsification trail actually unfolded, because the
sequence is the argument.

### 4.1 H1 is unfalsifiable in the model class that motivated it [AUDIT]

The natural first test sweeps each communication lever at fixed interpretive
capacity and compares the marginal slopes; the quantity of interest is the margin
*m* = slope(SC) − slope(PC), with *m* > 0 meaning PC is the stronger brake. H1
predicts *m* > 0 at low capacity and a downward crossing — a threshold — as capacity
rises.

In the additive model the margin is flat: identical at every interpretive-capacity
level, with no crossing. The reason is not numerical but algebraic. With additive
contributions, PC enters the brake directly while SC enters only through a
comprehension term at a throttled coefficient; the ratio of their potencies is a
quotient of fixed constants and therefore independent of capacity. A threshold
requires the *relative* effectiveness of the levers to vary with capacity, which
demands a non-separable lever×capacity interaction that an additive form does not
contain. H1 is thus not merely unsupported; its truth conditions live in interaction
terms absent from the model, making it untestable there. A robustness check across
three behavioural forms (additive brake, prestige-inversion backfire, and a linear
variant) sharpened the point into a non-identifiability result: the three forms
disagreed even on the *sign* of the capacity-dependence (flat, increasing, and
decreasing respectively), so "interpretive capacity moderates the lever balance" is
not a claim the model class can adjudicate — it is a free choice of nonlinearity.

The lesson generalises: **regime and threshold claims are claims about interaction
terms, and cannot be tested in a model that has none.** This result is AUDIT — a
property of additive algebra — but it is the most decisive thing the program
established, and it redirected everything that followed.

### 4.2 Relocating the moderator [EXTRAPOLATION from contemporary literature]

If capacity is the wrong axis, what is the right one? We turned to the literatures
in which the relevant behaviour is measured rather than imagined.

The nuclear-semiotics canon itself, examined first, proved to be almost entirely
expert *proposal* and *subjective-probability elicitation* — the Sandia marker
panels elicited judgments about deterrence efficacy that the panellists themselves
made conditional on the intruder's motive and the society's technological level —
and contained the backfire intuition ("why would so much effort defend nothing of
value?") only as folk reasoning, never as measurement. The field is proposal-rich
and data-poor precisely where a hypothesis needs data.

The warning-compliance literature supplies the brake: perceived hazard reliably
raises avoidance, and — directly contradicting the backfire intuition for ordinary
actors — vivid, fear- and disgust-inducing warnings *increase* protective behaviour
rather than glamorising the hazard. But this literature studies *non-acquisitive*
actors, people with nothing to gain by intruding. The archaeology of grave-robbing
supplies the mirror case: across five millennia, inscribed curses and supernatural
threats did little to deter determined robbers, and increasingly elaborate
defensive architecture coincided with increasingly thorough looting, the defenses
themselves reading as evidence of buried value. The reactance and curiosity
literatures add a third channel — prohibition raises curiosity, robustly and across
ages, even absent any inferred value — but with a decisive qualification: the effect
is strong on *curiosity and attention* and attenuates toward null at the level of
consummatory *behaviour* in field settings.

These findings jointly relocate the operative moderator. The sign of a
phenomenological cue's effect is governed not by the receiver's interpretive
capacity but by (i) the receiver's **acquisitive motive** and (ii) the **behavioural
stage** at which the outcome is measured. The same ominous cue deters the many who
have nothing to gain and tempts the dedicated few who suspect they do; and the
curiosity it provokes mostly fails to survive the funnel to action.

### 4.3 Certainty, not severity — and the ceiling that binds [mixed AUDIT/EXTRAPOLATION]

The criminology of deterrence supplies the appraisal stage's form. Becker's
expected-utility account makes intrusion a comparison of reward against the
*product* of the probability and the severity of consequence; the robust empirical
deviation from that account — the certainty aphorism — is that perceived *certainty*
of consequence deters substantially while *severity* deters only weakly, an
asymmetry whose magnitude is itself small and contested, and which formally implies
that those who do offend are disproportionately risk-seeking.

This carries a sharp consequence for the deep-time case that the experiment
(Experiment 004) makes mechanical. Deterrence in the model runs through a perceived
certainty of personal consequence that is capped by a *referent ceiling* — a
property not of the message but of the hazard. A buried radiological hazard is, from
an intruder's vantage, the worst possible referent on this axis: its harm is
invisible, latent, and delayed, supplying almost no perceptible certainty of
immediate personal consequence. Under such a ceiling the experiment finds that no
amount of marker "certainty signalling" moves behaviour once the ceiling is
reached, and the risk-seeking minority who actually intrude remain weakly deterrable
regardless of marker quality.

We are deliberate about the epistemic status here. That a low ceiling caps
deterrence is near-analytic — it follows from a minimum operator and is AUDIT. The
empirical content is the single claim that the radiological referent's ceiling *is*
low, which is reasoned from the physics of latency and invisibility but not
measured, and is therefore treated as a labelled assumption. The certainty-over-
severity ordering was tested for robustness across two defensible deterrence forms
(a symmetric Becker product and a certainty-gated form) and held in both, so it is
identified rather than an artifact of one form. The defensible reading is thus
modest and is the reframing this paper turns on: **the markers are not principally
fighting a comprehension problem or a phenomenology problem; they are fighting a
certainty-of-consequence problem against a hazard that intrinsically cannot supply
certainty.** The much-derided draft inscription insisting that the danger "is still
present, in your time, as it is in ours" was, on this account, groping toward the
correct lever — immediacy and certainty — without the theory to know why.

### 4.4 The brake/backfire crossover, and a logged amendment [EXPLORATORY]

A test of the relocated hypothesis (that acquisitive motive flips the sign of the
phenomenological cue) failed on its first run in an informative way, and we record
the failure rather than smoothing it. Under the pre-registered form, phenomenological
caution had been routed only into the value/backfire channels and given no braking
pathway, so its marginal effect on intervention was positive at every level of
acquisitiveness and the predicted crossover was structurally impossible. The
omission contradicted the warning-compliance anchor, under which comprehended dread
deters.

The fix is logged as a dated amendment and is theoretically principled rather than a
tuning convenience: the same cue is split by comprehension, reading as *hazard* to
those who understand it (a brake, scaling with comprehension) and as *mystery* to
those who do not (a backfire). Under the amended form the brake-to-backfire
crossover appears at high acquisitiveness, and — passing the anti-tautology guard — it
appears with the value-signalling coefficient inside its data-bounded range rather
than only at implausible values. Because the form was shaped after a result, this
finding is labelled exploratory, not confirmatory. Its interpretation is
nonetheless clean: phenomenological caution protects the great majority and tempts
only the most acquisitive, which is precisely the warning-versus-looting split made
continuous.

### 4.5 The decisive test: coupling produces no emergence [CARTOGRAPHY, negative]

A model whose every output is a deductive consequence of its inputs is a consistency
device, not a simulator. The single source of genuinely non-deductive behaviour
available to an agent model is *coupling* — feedback in which one agent's action
alters another's propensity — which can, in principle, produce emergent regime
shifts (cascades, tipping, hysteresis) not implied by any individual equation. We
therefore designed a coupling experiment (Experiment 005) as the decisive test of
whether the model could graduate from auditor to instrument.

Coupling was introduced as an act-stage social pull routed through the decision's
nonlinearity — the standard threshold-contagion form, which *can* produce
bistability. The emergent signature sought was hysteresis: two stable population
basins reachable from identical parameters depending on initial conditions, a
property no independent-agent model can reproduce regardless of its marginals, which
is what makes a matched-marginal null a real test rather than a trivial comparison.

No emergence appeared. The population's order parameter retained a unique fixed
point at coupling strengths swept to implausibly high values; cold-start and
hot-start trajectories always converged together; the result held under two distinct
definitions of the social signal (rare visible excavation versus general
engagement) and under both mean-field and local-graph topologies. Coupling
*amplified* the disturbance rate severalfold, but smoothly and reproducibly by a
shifted-marginal independent model — deductive amplification, not emergence. The
mechanism of the non-result is itself informative: population **heterogeneity**
suppresses tipping, because a diverse distribution of decision thresholds sums many
steep individual responses into a gently sloped aggregate with a single fixed point.
Within this model, deep-time intrusion is not a contagion phenomenon, and a
risk-seeking minority cannot ignite a cascade because there is no cascade to ignite.
(This conclusion is conditional on heterogeneous, independent priors; concentrated or
strongly correlated priors could in principle tip, and we have not anchored those.)

Because the coupling strength is unanchored, even a positive result here would have
been CARTOGRAPHY — a statement of reachability, never of actuality. The negative
result is correspondingly clean: the model declined the opportunity to surprise us.

---

## 5. The convergent finding: physical substrate is first-order

The experiments above were built to probe different questions — the algebra of
thresholds, the certainty of deterrence, the dynamics of contagion — yet a single
structural fact surfaced independently in three of them, designed into none.

In the foundational sensitivity analysis, the dominant driver of expected harm was
the hazard's **severity**, a fixed physical property, well above any communication
lever. In the deterrence experiment, the binding constraint was the **referent
ceiling**, the hazard's intrinsic inability to make consequence feel certain. In the
coupling experiment, the disturbance rate was hard-capped by **opportunity** — the
physical access and technical capability required to excavate — at a level the
strongest conceivable social pull could not exceed, because most agents simply
cannot reach a site they cannot get into.

It would be too easy, and we resist it, to read this as a triumphant empirical
discovery. Two of the three are at least partly AUDIT: severity dominates a
variance-based index in part because it is a wide-variance direct multiplier in the
objective, and the opportunity cap follows from opportunity entering as a
multiplicative gate. The honest and more interesting reading is that the convergence
reflects a difference of **logical type, not magnitude**. Physical factors —
severity, the certainty ceiling, opportunity — enter the harm function as *necessary
multiplicative conditions*: gates and caps through which the outcome must pass.
Symbolic and behavioural factors — clarity, dread, prestige, curiosity, social
modelling — enter as *modulators* within the space those gates permit. A modulator,
however well designed, cannot move an outcome past a gate it does not control. That
the physical factors are gates and the symbolic factors are modulators is itself the
empirically defensible claim: you cannot excavate a repository you cannot reach, you
cannot be deterred by a consequence you cannot perceive, and the harm is whatever the
waste makes it regardless of what the sign says.

This is the paper's thesis, and it inverts the field's working premise. **The marker
is not the binding constraint.** Forty years of semiotic ingenuity has been spent
optimising a modulator.

---

## 6. Discussion

### 6.1 Implications for marker design

If the physical factors are gates, the highest-leverage interventions are physical:
maximising the depth, inaccessibility, and technical difficulty of intrusion
(tightening the opportunity gate) does more than any sign, and it is the one lever
that acts on the risk-seeking, acquisitive minority who are by construction least
responsive to communication. This is not an argument against markers; it is an
argument about their order of magnitude.

There is, however, one symbolic move the analysis endorses, and it is a specific
one. The deterrence ceiling binds because the hazard cannot make *cognitive*
consequence feel certain. Affective dread — visceral, immediate, requiring no
inference — is the single channel that partly bypasses the ceiling, because it
operates before the cognitive risk appraisal that the latency of radiological harm
defeats. But dread is double-edged: it brakes the comprehending, non-acquisitive
majority and signals value to the acquisitive few. The design implication is precise:
phenomenological warning should aim to manufacture immediate, certain-feeling threat
(the *certainty* lever, not the *severity* lever) for the ordinary visitor, while
doing as little as possible to advertise the site as a defended prize to the
treasure-seeker — and it should expect to fail on the latter, who are better
addressed by the opportunity gate.

### 6.2 What the simulator was for

The simulator did not predict the future and was never able to. Its function was to
force a vague hypothesis into a falsifiable form, to make the consequences of each
modelling commitment inspectable, and — when asked directly, through the coupling
experiment, whether it could generate knowledge not already implicit in its
construction — to answer honestly that it could not. That negative answer is what
licenses the paper's modest framing: the contribution is a reframing, audited for
internal consistency against measured contemporary primitives, not a forecast.

### 6.3 Limitations

The limitations are not marginal and we state them plainly. Every behavioural
primitive is imported from contemporary, largely WEIRD, short-horizon studies, and
several rest on small or contested effects (the certainty-over-severity asymmetry in
particular). No data touches the deep-time horizon; the decay of interpretive
capacity, institutional memory, and referent perception over ten millennia is the
master assumption, represented as a scenario family and never as a forecast. The
behavioural equations are stipulated functional forms, not estimated, and the
quantitative results are conditional on the interpretation choice of §3.4. The
no-cascade result is conditional on population heterogeneity. The referent-ceiling
claim is reasoned from physics, not measured. And the convergent thesis of §5,
while we have argued it reflects a real type-asymmetry, is partly a consequence of
how physical and symbolic factors were encoded; a reader who rejects the gate/
modulator encoding can reject the conclusion, and we have tried to make that
encoding visible enough to be rejected.

---

## 7. Conclusion

We set out to falsify a hypothesis about which sign best deters a distant future
from a buried hazard, and discovered that the question was mis-posed at the level of
its independent variable. The threshold it posited was algebraically impossible in
its native model class; its real moderator was the intruder's motive, not the
intruder's capacity; its deterrent lever was certainty, not severity, against a
hazard that cannot supply certainty; and the social cascade one might fear, or hope,
would amplify a warning's reach does not arise in a heterogeneous population. Beneath
all of it lay a single structural fact, arrived at three independent ways: the
constraints that bind are physical gates, and the signs are modulators within them.
The deep-time warning problem is, in its first-order structure, a problem of physics
and access, not of semiotics. The most sophisticated marker imaginable is a
second-order correction to a containment problem — and the honest task of a model
in this domain is to say so, and to show its work.

---

## Epistemic-status ledger

| Result | Status | Note |
| --- | --- | --- |
| H1 threshold impossible in additive models (§4.1) | AUDIT | algebraic; the program's hinge |
| three-form sign disagreement (§4.1) | AUDIT | non-identification of capacity-dependence |
| moderator is motive × stage (§4.2) | EXTRAPOLATION | from warning/looting/reactance literatures |
| certainty ≻ severity ordering (§4.3) | EXTRAPOLATION | identified across two deterrence forms; effect small/contested |
| referent ceiling binds (§4.3) | AUDIT + assumption | min-operator (AUDIT); low radiological ceiling is reasoned, not measured |
| brake/backfire crossover (§4.4) | EXPLORATORY | post-amendment; passes anti-tautology bound |
| no coupling emergence (§4.5) | CARTOGRAPHY (negative) | robust to signal, topology, coupling magnitude; conditional on heterogeneity |
| physical substrate first-order (§5) | AUDIT-leaning | reflects gate/modulator type-asymmetry; partly encoded |

## Reproducibility

The model, experiments, and 37-test regression suite are released as DTBR-MC
(v0.1 additive models and v0.2 funnel). All runs are seeded and deterministic; the
H3 model, its experiments, and the pre-registration with its logged Amendment 1 are
included so that the falsification trail above can be reproduced and contested.

## References

*The works below are cited from synthesis; bibliographic details (years, volumes,
report numbers) should be verified against primary sources before submission, and
several secondary syntheses should be replaced by the primary studies they report.*

- Argo, J. J., & Main, K. J. Meta-analytic review of the effectiveness of warning labels. *Journal of Public Policy & Marketing.*
- Becker, G. S. (1968). Crime and punishment: an economic approach. *Journal of Political Economy.*
- Brehm, J. W. (1966). *A Theory of Psychological Reactance.*
- Bromberg-Martin, E. S., & Hikosaka, O. Midbrain dopamine neurons and the coding of information as reward. (curiosity / information-as-reward.)
- FitzGibbon, L., et al. The forbidden-fruit effect: prohibition increases curiosity. (OSF 2020; subsequently published.)
- Granovetter, M. (1978). Threshold models of collective behavior. *American Journal of Sociology.*
- Human Interference Task Force (1981/1984). Reducing the likelihood of future human activities that could affect geologic high-level waste repositories. (US DOE / Office of Nuclear Waste Isolation.)
- Murayama, K., and colleagues. The reward value of information / curiosity as motivated information-seeking.
- Nagin, D. S. Deterrence in the twenty-first century. (certainty vs severity.)
- Paternoster, R. The perceptual deterrence tradition; certainty, severity, and perceived risk.
- Sebeok, T. A. (1984). Communication measures to bridge ten millennia. (Office of Nuclear Waste Isolation; the "atomic priesthood.")
- Trauth, K. M., Hora, S. C., & Guzowski, R. V. (1993). Expert judgment on markers to deter inadvertent human intrusion into the Waste Isolation Pilot Plant. Sandia National Laboratories, SAND92-1382.
- Wogalter, M. S. (ed.). *Handbook of Warnings*; the Communication–Human Information Processing (C-HIP) model and warning-compliance moderators (familiarity, cost of compliance, social modelling).
