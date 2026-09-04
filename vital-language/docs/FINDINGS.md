# Vital Language: Interim Findings

*Status: exploratory, single-machine, small base models (Qwen2.5-0.5B / 1.5B,
CPU). The headline caveat below governs everything.*

## The one caveat that governs all "vitality" claims

**All human-judgment data is n=1, and that rater is the assistant model itself**
(the same system generating hypotheses). PCA on those ratings shows PC1 = 85% of
variance: a near-unidimensional "good↔bad" gradient consistent with a halo
effect. Therefore every claim below that involves *felt vitality* is a
**hypothesis about sign**, not an established fact, until independent human
raters are collected. Claims about **coherence / degeneracy** are
rater-independent (counts of collapse into non-prose) and are solid.

---

## What we set out to test

> Inject controlled **deterministic chaos** into generation → increase
> **multifractal** structure → increase **perceived vitality**, without
> collapsing coherence. (Chaos should beat ordinary randomness specifically.)

We built a falsifiable apparatus: one decode loop; logit modulation
`L' = L + ε·C·M` with scale-relative ε; chaotic signals (Lorenz) vs. a
**matched-autocorrelation OU-noise control** (same 2nd-order stats, stochastic
not deterministic) vs. white noise vs. plain sampling; from-scratch MFDFA with
finite-size guards; a base-model self-perplexity coherence gate; sentence-
embedding trajectory metrics; and a blind human-rating harness.

The MFDFA tool was **validated on real literature**: on sentence-length series,
Joyce *Ulysses* width ≈ 5.0, Woolf ≈ 6.3, Austen ≈ 0.25 (SoC ≈ 20× conventional
prose), and a **shuffle surrogate** collapses Joyce 5.0→0.35 but barely moves
Austen: confirming the multifractality is in temporal ordering, not the value
distribution. The instrument works; the question is what it measures on
small-model output.

---

## Findings, in the order the data forced them

### F1. Temporal structure of perturbation >> magnitude (solid)
White-noise logit modulation destroys coherence fast (self-perplexity 7.8→176
across ε=0→4); chaotic modulation at the same ε stays legible (~10). A 17×
coherence gap from identical perturbation magnitude, differing only in temporal
structure. Unstructured noise ≠ structure.

### F2. Chaos beats matched-noise on surprisal-width, but not plain sampling (solid stat, hollow meaning: see F4)
Paired ε-sweep (0.5B, 6 prompts × 3 seeds, legibility-gated, bootstrap CIs):
chaos > matched-OU on surprisal MFDFA width at ε=1.5 (+0.35*), 2.0 (+0.46*),
3.0 (+0.14*). **But chaos vs. plain sampling is null at every ε.** Replicated in
direction (not amplified) on 1.5B. So: chaos moves surprisal-structure beyond
its matched-randomness control (a real deterministic-vs-stochastic effect) but
not beyond ordinary sampling.

### F3. The literary (sentence-length) multifractal signal is NULL on our output
Chaos ≈ sampling (~0.29 vs 0.30) and chaos's small width does NOT survive the
shuffle surrogate (drop 0.05) → not genuine long-range structure. The signal
needs hundreds of sentences; small base models degenerate before sustaining
them. **The literary-multifractality question is not well-posed at this scale.**

### F4. Surprisal-width does NOT track felt vitality: because width is a degeneracy proxy (the pivotal result)
Raw: width vs. vitality = **−0.51**. This looked like "our target metric has the
wrong sign." It does not survive scrutiny:
- partial, controlling a template/quiz/loop **degeneracy flag**: −0.40
- **among clean (non-degenerate) passages only: −0.135** (collapses).

So **width was largely detecting collapse into training substrate** (math word
problems, multiple-choice quizzes, code scaffolds), which inflates MFDFA width
*and* reads as dead. The founding multifractal intuition is not refuted; it is
**untestable at this scale** because the substrate is dominated by collapse.
(Eyeball confirms: the highest-width passages are candle/ship arithmetic
problems.) Method note: **the clean-subset test is the project's backbone: it
impartially killed our metric, GPT's metric, and the advisor's metrics.**

### F5. What DOES track (n=1) vitality among coherent passages: semantic travel
Of everything tried, only simple sentence-embedding trajectory metrics survive
the clean subset:
- **global_range** (mean distance from the passage's own centroid): **+0.51 clean**
- **drift** (consecutive-sentence distance): **+0.45 clean**
- semantic-volume beats mean-surprisal as the "novelty carrier" (partials +0.41
  vs +0.28) → vitality tracks *covering new semantic ground*, not token entropy.
Disconfirmed on clean text: coherent-advance-rate (+0.04), concreteness density
(+0.06), recurrence (mostly a stuckness confound: −0.60→−0.43 clean). Eyeball:
low-range coherent passages are *topically stuck* (circling silence/stone);
high-range ones *travel* (the bird "coming to life", the recursive microwave
litany). Working hypothesis: **vitality ≈ sustained semantic travel under
coherence**, NOT multifractality / entropy / recurrence / concreteness.

### F6. The reliable lever is prompt-level agency, and it buys non-collapse (solid)
Agency test (persistent-speaker scaffold vs. chaos vs. matched vs. plain):
- **degeneracy: agency 0/18 vs plain 3/18 vs chaos/matched 2/18**: the cleanest,
  most reliable effect in the project. A "one remembering speaker" prompt keeps
  the model from falling out of first-person into substrate.
- but **agency does NOT raise the vitality proxy** (agency−plain global_range =
  +0.001, dead null). It keeps text coherent; it doesn't make it travel.
- **chaos *lowers* semantic breadth vs plain** (plain−chaos = +0.051*). Token
  chaos is, on the surviving proxy, counterproductive.
- eyeball caveat: among already-clean passages, the agency scaffold can trade
  *strangeness for blandness* (Hallmark-inspirational drift): a possible new
  dead-end, the "machine basin" we set out to escape.

---

## Where the center of gravity has moved

The founding bet inverted. **Token-level chaos is the wrong intervention**: it
doesn't beat sampling on anything meaningful and *reduces* the surviving vitality
proxy. The intervention that reliably *works* is **higher-layer (prompt/agency)**,
and what it buys is **non-collapse**, not multifractal vitality: exactly where
the theoretical advisor pointed ("vitality lives above the token layer; the
reachable object may be non-collapse").

**The honestly defensible project right now** is narrower and truer than the
original: *what keeps a small LM sustaining a coherent first-person voice instead
of collapsing into training substrate, and can we additionally push semantic
travel without flattening into blandness?*

---

## What would change these conclusions

- **Real human raters** (the gate). If humans' vitality ratings do NOT correlate
  with global_range among coherent passages, F5 evaporates and the vitality axis
  must be rebuilt. If they do, F5 becomes a real target. Collect forced-choice +
  separated dimensions to break the halo.
- The advisor's **agency conjecture** (vitality = reader's inference of a
  persistent mind): partially supported (agency→non-collapse) but its strong form
  (agency beats dynamics on vitality) is NOT supported by the proxy.

## Candidate next experiment (only after human raters validate F5)
Single signed axis: **directed semantic departure pressure** at sentence level
(reward next-sentence embeddings that increase semantic volume / leave the recent
centroid), swept negative (return) → 0 (sampling) → positive (departure), under a
hard perplexity gate AND watched for blandness. **Critical control:** a
random-direction pressure matched in magnitude (isolates *directed* travel from
generic perturbation). Pre-registered abandon criterion: if no level beats
baseline on (human) vitality within the coherence band, and directed ≈ random,
the "semantic travel is injectable vitality" direction is falsified.

## F7. THE UNIFYING RESULT: at this scale, the only large axis of quality variation is collapse-vs-coherence

After the reverse (literature-first) approach produced metrics that order the SOC
tradition (F3 concrete⇄abstract oscillation: Lispector 0.154 > Joyce 0.136 >
Woolf 0.116 > Austen 0.084; imagistic CLIP channel dissociable from semantic),
we tested them against MODEL-output ratings. They collapse on the clean subset
exactly like all prior metrics: F3 +0.50 ALL → **+0.011 CLEAN**; imagery weak.

The cause, found directly: **the clean (coherent) subset is a uniformly-middling
band.** Vitality SD = 15.6 (all) → **8.1 (clean)**; range [12,75] → [35,62]. The
degenerate passages carry nearly all the variance.

UNIFYING STATEMENT: *At 0.5–1.5B scale, the only large, reliable axis of variation
in felt quality is collapse-vs-coherence. Among coherent passages the model emits
a narrow band of competent-but-flat prose with little vitality variance, so every
"vitality metric" that appeared to work (width, recurrence, advance, F3, imagery)
was detecting DEGENERACY, not discriminating aliveness.* This explains the entire
project: why metrics kept being degeneracy proxies, why the agency scaffold's real
effect was non-collapse, why literary metrics separate Joyce/Austen (real range)
but not model passages (no range). The vitality the project sought does not VARY
enough to study in coherent sub-2B output. Two regimes split it (LIVING_LANGUAGE_
SPEC): intrinsic (Lispector, measurable but barely present in our outputs) vs
relational (Joyce allusion/phonetics, out of scale entirely).

Caveat unchanged: vitality ratings are n=1/self. But this result makes the missing
human study LESS likely to rescue the vitality axis: the variance to explain is
small in the clean band regardless of rater. The robust, rater-independent finding
stands: agency-framing prevents collapse (F6).

## F8. FRONTIER IMITATION: vitality is a structural discipline, not a nameable register (the literary-arc capstone)

If vitality barely varies in *small*-model output (F7), the next question is
whether *frontier* models (which hold the cultural manifold) can produce it. We
tested this by imitation, treating "can a frontier model become author X" as a
probe of WHAT in literary vitality is reachable. Design: Sonnet 4.6 generated, per
author (Woolf / Joyce / Lispector), an **imitation** AND a **matched control**
(same scene, no author named, "good normal prose"). We measured the distance
traveled (`imitation − control`) against where the real author sits relative to
conventional prose (`real − Austen`), on all channels (semantic & imagistic
trajectories, F3 oscillation, sentence rhythm, phonetic flow). Prompts never named
the features measured (naming a metric turns a score into mere compliance).

**The three authors split by reachability, and the split is the result.**

- **Woolf: CAPTURED.** The imitation moves toward real Woolf on every measurable
  axis (semantic drift/range, sentence-rhythm), often *overshooting* the
  tradition gap. Her perception⇄memory glide is an intrinsic, register-level
  signature a frontier model reaches.

- **Joyce: SURFACE captured, DEPTH not.** Trajectory metrics moved toward Joyce,
  and the model reproduced his late-style **run-on so completely the imitation is
  one 1088-word sentence** (it broke our sentence splitter: extreme surface
  fidelity). But the *referential* depth is a different story (see allusion probe
  below).

- **Lispector: FAILED, and instructively.** Her structural signature is
  counterintuitive: **high semantic drift / LOW imagistic drift** (meaning churns
  while one image-field is held still). Measured sem/img ratio: real Lispector
  **5.67**; Sonnet's generic control **5.93**; Sonnet's *deliberate imitation*
  **4.45**. By full-signature distance the imitation is CLOSER to its own generic
  control (0.124) than to real Lispector (0.182). **Asked to imitate her, the
  model moved AWAY**: it added imagistic variety (its notion of "poetic"), the
  opposite of her held-image discipline. The CLIP imagistic channel was the
  discriminator that exposed this; semantic geometry alone could not see it.

### Allusion probe (the relational/Joyce depth): blind, cross-model, with a confound we caught

A capable model first close-read the Joyce imitation and found it allusively rich
(Proust, Synge, Ecclesiastes, Eliot, Bergson, Baudelaire, Plato, Irish-ballad,
Molly Bloom). But that was circular (same model family) and unblinded. So we ran a
**blind, length-matched, cross-model** probe: 400-word excerpts of {real Joyce,
imitation, control}, labelled only PASSAGE 1/2/3, scored for *nameable* resonance
density by **Grok, Gemini, and ChatGPT** independently.

Densities (P1=control, P2=real Joyce, P3=imitation):

| Judge | control | real Joyce | imitation |
|---|---|---|---|
| Grok | 0.023 | ~0.33 | ~0.29 |
| Gemini | 0.131 | 0.791 | 0.074 |
| ChatGPT | 0.07 | 0.32 | 0.02 |

**CONTAMINATION (caught): all three judges RECOGNIZED the real Joyce as "Oxen of
the Sun"** (named the episode + characters). You cannot blind-probe canonical text
for depth: the model has it memorized, so its high score is partly *recall*, not
reading. So "real > imitation" is NOT cleanly established from this.

**The contamination-FREE signal: the imitation is novel, unrecognizable:** the
judges SPLIT hard on it (Grok 0.29 high; Gemini 0.074 and ChatGPT 0.02: *below
the plain control*). The split is principled: Grok counted Joycean *mode/technique*
as resonance; Gemini & ChatGPT required *specific nameable external references* and
found almost none. ChatGPT: "almost all significance generated internally, not
externally nameable cultural references." So even allowing the confound: the
imitation has the **texture** of allusive depth but not the **referential pointing**;
2 of 3 blind judges scored it *below a plain insomnia passage* on nameable
resonance.

### The capstone thesis

Lispector (image-discipline) and Joyce (allusive pointing) converge, by
independent methods, on one statement:

> **Frontier models reproduce the nameable STYLE REGISTER of literary vitality
> the glide, the run-on, the lyrical-modernist surface, but not its STRUCTURAL
> DISCIPLINE: Lispector's image held still under churning meaning, Joyce's
> load-bearing reference into a shared manifold. Asked to "be" an author, the
> model reaches for the recognizable register and defaults to its own generic
> "literary" attractor for the rest.** Woolf is "captured" precisely because her
> signature largely IS a register; the two whose vitality is a discipline rather
> than a register are not.

This reframes the relational/intrinsic boundary (LIVING_LANGUAGE_SPEC) one final
time: the boundary is **register (imitable) vs. discipline (not imitable on
command)**: orthogonal to model scale. A bigger model gets a richer register, not
the discipline.

### Method lessons (durable)
- Canonical text is unusable in blind LLM-judge comparisons (memorization). Use
  novel text, or human experts, for depth probes.
- The CLIP imagistic channel earned its place: it caught the Lispector failure
  that semantic geometry was blind to. The semantic⇄imagistic *relationship* (not
  either channel's level) is the discriminating quantity.
- Caveats: n=1 model, 1 passage/condition (directional, not powered); real-author
  windows are single samples; Woolf corpus is *Voyage Out* (her conventional-
  leaning novel), softening "captured Woolf."

## Artifacts
- Frontier imitation: `experiments/frontier_imitation/` (PROMPTS, analyze2.py,
  RESULTS.md, ALLUSION_PROBE.md, the 3 judge analyses, _probe_KEY.json)
- Living-language theory: `LIVING_LANGUAGE_SPEC.md`
- Apparatus: `vitality/`, `scripts/`, `configs/`
- Literary benchmark: `scripts/benchmark_literary.py`, `corpus/`
- Reanalyses: `scripts/stage0_reanalysis.py`, `test_advance_vs_vitality.py`,
  `agency_test.py`, `probe_metrics_vs_vitality.py`
- Human study (ready, needs raters): `study/rate.html`, `vitality/study/`
- Theory: `ADVISOR_BRIEF.md` (+ its response in conversation)
