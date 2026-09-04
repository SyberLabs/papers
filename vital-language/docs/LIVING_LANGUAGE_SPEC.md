# Living Language: Operationalized Feature Spec (Stream of Consciousness)

*Reverse approach: derive the target from the tradition, THEN build metrics to
match: instead of measuring first and theorizing after. Each feature below names
(a) the literary mechanism, (b) which author exemplifies it, (c) a COMPUTABLE
metric, (d) its prediction, and (e) how Lispector tests it.*

## Why this document exists

Our empirical pipeline kept discovering its metrics were proxies for degeneracy
because we had no independent theory of the target. And a first probe is already
warning us: on a sampled window, our current best proxies (drift, global_range)
**barely separate Joyce/Woolf from Austen** (drift 0.69–0.73, range 0.45–0.54),
while **sentence-length CV does** (Woolf 0.96 > Austen 0.71 > Joyce 0.57). That
suggests SOC's distinctiveness is **syntactic/rhythmic, not semantic-travel**
i.e. F5 ("vitality = semantic travel") may be measuring the wrong layer.

## The three authors are three different machines (do not average them)

| Author | Core mechanism | What "alive" means there |
|---|---|---|
| **Woolf** | free indirect discourse; the "luminous halo"; perception⇄memory glide | a narrated mind moving fluidly across time; expanding/contracting sentences around "moments of being" |
| **Joyce** (Penelope) | syntactic dissolution; associative chaining; punctuation removed | the sentence AS an unbroken associative current; sound-driven drift |
| **Lispector** | thought failing to grasp itself; abstraction made visceral; **deliberate non-advancement** | pressure on a single inapprehensible thing; circling, not roaming |

NB: our corpus has *The Voyage Out* (1915): Woolf's CONVENTIONAL early novel,
NOT her SOC. Real Woolf-SOC features below are from *Mrs Dalloway / To the
Lighthouse / The Waves* and must be measured on those, not Voyage Out.
Lispector is in copyright (d. 1977): describe features, don't corpus-mine.

---

## Operationalized features

### Feature 1: Rhythmic heterogeneity (sentence/clause length variance)
- **Mechanism**: SOC abandons uniform sentence shape; bursts of short fragments
  against long subordinated flows. (Woolf's expansion/contraction; Joyce's run-ons.)
- **Metric**: sentence-length CV; clause-length CV (split on commas/semicolons);
  and the **MFDFA width of the sentence-length series** (our existing IFJ signal).
- **Prediction**: high in all three; the most robust SOC discriminator (already
  separates Woolf 0.96 / Austen 0.71 in the probe).
- **Lispector test**: she SHOULD score high too (her fragments + long meditative
  sentences). If a metric ranks her low, it's too tuned to Joyce's run-ons.

### Feature 2: Syntactic parataxis / connective dissolution
- **Mechanism**: coordination ("and… and…") and juxtaposition replace
  subordination and logical connectives; punctuation thins (Joyce extreme).
- **Metric**: ratio of coordinating ("and/but/then") to subordinating
  ("because/although/which") connectives; punctuation density (commas/periods per
  word); mean inter-punctuation run length.
- **Prediction**: Joyce extreme; Woolf moderate; Austen low (she subordinates).
- **Lispector test**: MIXED: she uses short declaratives AND paradox; this
  metric may not capture her. Flags that parataxis is Joyce-specific, not
  SOC-general.

### Feature 3: Perception⇄abstraction oscillation (the Woolf/Lispector signature)
- **Mechanism**: rapid alternation between concrete sensory detail and abstract
  reflection: the mind touching the world then pulling back to think.
- **Metric**: per-sentence concreteness score (concreteness-norm lexicon, better
  than our crude sensory list), then the **alternation rate / variance** of
  concreteness across sentences (not the mean: the *oscillation*).
- **Prediction**: high alternation in Woolf & Lispector; LOW in Austen (steady
  register) and in machine-quiz collapse (flat).
- **Lispector test**: THIS is her signature: "abstraction made visceral." If our
  metric suite is going to capture Lispector at all, it's here. Strong test.

### Feature 4: Recursive return / motif pressure (NOT mere recurrence)
- **Mechanism**: a motif returns TRANSFORMED, under accumulating pressure, not
  verbatim repetition (dead) and not never-returning (mere travel).
- **Metric**: for embedding-near sentence pairs at distance ≥k, measure the
  *residual* novelty (1 − sim) AT the lexical level despite semantic closeness
  i.e. "same idea, new words." Return-with-transformation, not return-as-loop.
- **Prediction**: present in all three; the thing our crude recurrence_rate
  (−0.60) CONFLATED with stuck-looping. This separates artful return from
  degeneration.
- **Lispector test**: SHE IS THE CASE. Her circling is high recurrence + high
  transformation. If this metric ranks her high where plain recurrence_rate ranks
  her low, it vindicates the feature AND explains F5's failure.

### Feature 5: Delayed / withheld closure
- **Mechanism**: syntactic and semantic resolution is deferred; sentences and
  thoughts stay open; the "completion pressure" LLMs satisfy too fast is resisted.
- **Metric**: rate of sentence-final hedges/openings vs. declaratives; proportion
  of clauses left grammatically suspended; (harder) a "closure" classifier.
- **Prediction**: low closure-rate in SOC, high in Austen and in machine prose
  (which closes every idea immediately: our original "Standard Prediction Stream"
  complaint).
- **Lispector test**: extreme: she refuses closure hardest. Good discriminator.

### Feature 6: Single sustained consciousness (the agency thread)
- **Mechanism**: one continuous remembering perceiver; deixis (I/here/now) stays
  anchored; no collapse into impersonal exposition.
- **Metric**: first-person pronoun continuity; referential consistency of the
  speaker; absence of register-breaks (our degeneracy flag is the crude version).
- **Prediction**: this is what our AGENCY scaffold already enforces (F6: 0/18
  collapse). It's necessary-not-sufficient: keeps a mind present, doesn't make it
  vital.
- **Lispector test**: present (intense first person) but not her distinctive
  feature: confirms F6 is a floor, not the target.

---

## How this reorganizes the project

- **F5 ("vitality = semantic travel") is probably Joyce-biased and incomplete.**
  The probe + Feature 3/4 suggest the real signal is **rhythmic heterogeneity
  (F1) + perception⇄abstraction oscillation (F3) + transformed-return (F4)**, NOT
  raw semantic distance traveled.
- **Lispector is the unit test for the whole suite**: any metric set claiming to
  measure vitality must rank her HIGH despite her low semantic travel and high
  recurrence. F1/F3/F4/F5 should catch her; F2 (parataxis) should not, and that
  pattern itself validates which features are SOC-general vs Joyce-specific.
- **The agency floor (F6) stands**: necessary substrate, not the target.

## Lispector measured (Água Viva, Losada ES translation, multilingual embeddings)

We now have real text (agua_vida.md). Direct read + measurement CONFIRM she
falsifies "vitality = semantic travel":
- **global_range = 0.473 (modest), drift = 0.651 (high), recurrence ~0.01.**
  She has MOTION WITHOUT TRAVEL: high local sentence-to-sentence movement held
  within a BOUNDED semantic region. She circles one nucleus (el "it" / el "es" /
  el instante-ya) intensely, dozens of returns, never roaming far. Our F5 metric
  (global_range) would rank the most vital text in the tradition only MIDDLING.
- The close read foregrounds exactly Features 3/4/5: perception⇄abstraction
  oscillation ("la cuarta dimensión del instante" slammed into "escurría limón
  sobre una ostra viva y veía cómo se retorcía"); transformed return (the "it"
  returns as oyster/placenta/God/stone, NOT verbatim); withheld closure (she
  STATES it: "quiero lo no concluido", "quiero la experiencia de una falta de
  construcción").

REFINED TARGET (from text): vitality ≈ **high local drift held within a bounded
semantic region** (restless movement around a fixed obsession) + perception⇄
abstraction oscillation + transformed return, NOT raw global_range/travel.
Candidate metric: drift / global_range is WRONG sign (we found −0.41); the right
shape is high drift AND moderate range: i.e. a "local agitation, global
containment" ratio, or trajectory that fills a bounded region densely
(space-filling) rather than escaping it.

TRANSLATION CAVEAT: agua_vida.md is Losada's SPANISH of Lispector's Portuguese.
Syntactic features (F2: parataxis/punctuation) are translation-ALTERED: measure
them only on original-language text. Semantic/structural features (F3,4,5) are
translation-robust (properties of thought-movement a good translator preserves).
Conveniently, F3/4/5 are the SOC-general features and F2 was already Joyce-specific.
Use multilingual embeddings (paraphrase-multilingual-MiniLM) for any ES/PT text.

## RESULT: perception⇄abstraction oscillation (F3) separates the tradition

Built two refined metrics (vitality/metrics/soc_features.py, multilingual embeds):
- **bounded_agitation = local_drift × (1−global_range): FAILS.** Ranks Austen
  (0.349) ABOVE Joyce/Woolf: the containment term carries it and conventional
  prose is also contained. A vitality metric that ranks Austen 2nd is wrong. Drop it.
- **perception⇄abstraction OSCILLATION (F3): WORKS, robustly.** = std of per-
  sentence (concrete-anchor sim − abstract-anchor sim), embedding-anchored so
  language-fair. Across 4 windows/author it MONOTONICALLY orders the tradition
  with tiny within-author variance:
    Lispector 0.154 > Joyce 0.136 > Woolf(VoyageOut) 0.116 > Austen 0.084
  Austen ~45% below Lispector, no overlap. This is the strongest result in the
  project: grounded in LITERARY ground truth (not n=1 ratings), survives the
  Lispector falsification case (she scores HIGHEST: the metric built to catch
  her, catches her), and is interpretable (a mind oscillating concrete⇄abstract).
  Woolf's transitional Voyage Out landing between Joyce and Austen even fits.

STILL UNPROVEN: (a) does F3-oscillation track MODEL-output vitality / human
ratings? (separating literary SOC ≠ predicting felt vitality in 0.5–1.5B prose);
(b) is it manipulable during generation? (c) n on authors is small (4 windows).
But this is the first metric earned from the tradition rather than from our own
halo-prone ratings, and the first that the falsification case CONFIRMS.

## IMAGERY + PHONETICS hypothesis (user/GPT): NLP compresses language to SEMANTICS, discarding IMAGE + SOUND; SOC lives in the discarded channels

CONCEPTUAL FRAME (strong, retro-explains our failures): Language = Semantic ⊗
Imagistic ⊗ Phonetic. Semantic-travel metrics barely separated Joyce/Woolf from
Austen because Austen ALSO travels semantically: what she lacks is image-
metamorphosis + phonetic drift. "Intelligent but not alive" = meaning survives
compression, sound+image don't (next-token optimizes the surviving channel).

IMAGISTIC EMBEDDING EXISTS: CLIP text-encoder embeds a sentence by evoked VISUAL
SCENE, not proposition: get imagistic similarity from TEXT alone (no images).
Built scripts/test_imagistic.py (open_clip ViT-B-32). RESULTS (4-window robust):
- Channels are REAL & DISSOCIABLE. Lispector: SEM drift 0.67 (high) vs IMG drift
  0.089 (very low, range 0.047 = 1/4 of others, stable 0.082-0.097). She is
  RESTLESS IN MEANING, STILL IN IMAGE: holds one dark-wet-enclosed image-world
  (oyster/placenta/cave/stone/blood) fixed while concepts churn around it.
  Confirms "tight image manifold under intense mutation" as a MEASURED fact.
- BUT naive "SOC = more imagistic drift" FAILS: Joyce 0.18 ≈ Austen 0.15. Image
  channel is real; "more is better" is not a law.
- The discriminator is the INTER-CHANNEL RELATIONSHIP, not within-channel motion:
  Lispector = high-semantic/low-imagistic. img-sem drift xcorr orders
  Lispector .44 > Joyce .37 > Austen .28 > Woolf .16 (image&meaning FUSED in
  SOC vs meaning-moves-image-inert in conventional). The braid is about ALIGNMENT.
- Joyce's expected 'imagistic whirlpool' did NOT appear in CLIP: consistent with
  Joyce being the PHONETIC case, Lispector the IMAGE case. Phonetics UNTESTED
  (and only valid on original-language text, not the ES Lispector translation).

REVISED TARGET: vitality signature is per-author RELATIONSHIP among Semantic /
Imagistic / (Phonetic) trajectories, not a level on any one. Feature 3
(concrete⇄abstract oscillation) was a partial probe of the Image↔Concept axis,
which is why it worked. Two encoders (MiniLM + CLIP-text) per sentence = two
trajectories; their divergence/alignment = the real measurable.

## VERTICAL ALLUSIVE DEPTH (user, the Joyce insight): one image resonates across SCALES at once

User: Joyce's "riverrun, past Eve and Adam's... commodius vicus of recirculation"
= ONE surface image (river) functioning simultaneously as Genesis + Vico's
cyclical history + samsara + the Liffey + the book's own loop structure. This is
NOT horizontal image-MOVEMENT (Joyce's CLIP imagistic-drift was unremarkable ≈
Austen). It is VERTICAL SUPERPOSITION: density of resonance AT A POINT, polysemy
across scales, held together by a single concrete image.

Three axes now distinguished (was conflating 2nd and 3rd):
1. semantic trajectory (MiniLM): meaning movement
2. horizontal imagistic trajectory (CLIP-text): visual-scene movement (Lispector's
   held image-field; MEASURABLE, demonstrated)
3. VERTICAL allusive depth: scales co-activated at one point (Joyce)

POLYSEMY PROXY TESTED (embedding neighborhood-dispersion): FAILS. 'riverrun'
spread 0.86 ≈ 'table' 0.79 ≈ 'spoon' 0.79; 'genesis' LOWEST (0.66). No signal.
WHY IT FAILS = the key insight: allusion is NOT in the word's local geometry
it's an edge in a CULTURAL/INTERTEXTUAL graph that lives in the READER's
knowledge ('riverrun→Vico' requires having read Vico). Small text encoders never
held it; co-occurrence stats ≠ erudition.

SCOPE CONSEQUENCE (real boundary): Joyce-style allusive depth requires a model
that HOLDS the cultural manifold. CLIP/MiniLM/0.5-1.5B base models neither hold
it nor can generate into it. So vertical allusive vitality is OUT OF REACH at our
scale, not a measurement gap but a generative-capacity gap. Only route to
measure it = LLM-as-probe with a model large enough to KNOW the allusions (asked
'what scales does this image activate?'): changes project character to LLM-judge
work, needs resources we mostly lack locally.

CLEAN SPLIT: horizontal imagistic (Lispector) = small-model-tractable, measurable
now. Vertical allusive (Joyce) = needs world-knowledge, large-model only. These
are DIFFERENT phenomena we'd been calling 'imagistic'.

## PHONETIC channel TESTED: null with this instrument (vitality/metrics/phonetic.py, CMUdict via pronouncing, English-only)

Three attempts, all flat/against-prediction:
- static phonetic_flow (alliteration+phoneme-recurrence): Joyce 0.153 ≈ Austen
  0.140 ≈ MACHINE 0.149. No separation; machine even > Austen (opposite of 'dead
  channel'). 
- stress_variance: only separator, but INVERSE: Austen 0.69 > Woolf 0.44 >
  machine 0.36 > Joyce 0.33 (Joyce's run-ons are rhythmically EVEN). = sentence
  architecture, not vitality.
- cross-channel ('sound bridges meaning-gaps': phon overlap high where semantic
  sim low): phon-sem xcorr Joyce +0.20, Woolf +0.03, Austen +0.11: all POSITIVE
  (sound&meaning weakly co-move), Joyce not distinctively negative. Null.

WHY (two reasons, both partly true): (1) instrument too crude: word-level
phoneme-SET Jaccard discards SEQUENCE/POSITION (alliteration=initial,
assonance=ordered vowel runs, Joyce's effects sub-word & rhythmic-across-line).
Real metric needs ordered-phoneme-stream alignment over clauses. (2) DEEPER &
unifying: Joyce's phonetics, like his allusion, is sound-as-POINTER
'commodius vicus' works because you half-hear 'commodious'+'Vico'; the sound
resonates against the cultural ear. Phoneme stats can't capture sound-as-pointer
any more than embedding geometry captured allusion-as-pointer.

## UNIFYING SCOPE BOUNDARY (the session's core finding)
Vitality splits by WHERE it lives:
- INTRINSIC (in the text's own structure) → MEASURABLE at our scale, small-model-
  relevant. Lispector's held-image-field under conceptual churn is the exemplar.
- RELATIONAL (surface that POINTS into the reader's vast associative/cultural
  manifold: Joyce's allusion AND his phonetics) → NOT recoverable from local text
  geometry, NOT generable by small models. Needs a model that HOLDS the manifold.
The project's tractable core is the INTRINSIC regime (Lispector-type: semantic ×
imagistic channel relationship). The relational regime (Joyce) is a real
phenomenon but out of scale, only approachable via large-LLM-as-probe.

## Next concrete steps
1. Get a copyright-clean SOC corpus for measurement: real Woolf SOC (*Mrs
   Dalloway* is US-PD; *To the Lighthouse* PD in many jurisdictions), more Joyce
   (Penelope episode), + a conventional-prose control. Lispector via short
   fair-use excerpts for qualitative feature-checking only.
2. Implement Features 1–5 as metrics; run on SOC vs conventional corpus; keep only
   those that separate them (the literary ground truth).
3. Re-run the SURVIVING features against our model-output ratings (and future
   human ratings). A feature that separates Joyce-from-Austen AND tracks human
   vitality is a real target.
4. THEN design injection toward those features (likely Feature 3/4/5 at the
   semantic/decoding layer, plus the F6 agency floor), not token chaos.
