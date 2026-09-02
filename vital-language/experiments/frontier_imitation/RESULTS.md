# Frontier Imitation Experiment — Results

Sonnet 4.6 (web) generating imitation + matched control per author; compared to
real-author ground truth on all channels. Distance-traveled design:
`imitation − control` should match `real_author − Austen` if the model captured
the signature. (n=1 model, 1 passage/condition — directional, not powered.)

## Headline: the three authors split by reachability, as the intrinsic/relational theory predicted

| Author | Captured? | Reading |
|---|---|---|
| **Woolf** | **YES** | imitation moves toward real on sem_drift/range, sent_cv, img — often overshooting the tradition gap. Intrinsic, frontier-reproducible. |
| **Joyce** | **Surface yes, deep unclear** | trajectory metrics captured; sent_cv "wrong-dir" is an ARTIFACT — the imitation is ONE 1088-word sentence (out-Joyced the run-on). Phonetic flow moved toward Joyce. But the relational/allusive depth (the real Joyce) is untested by geometry. |
| **Lispector** | **NO** | imitation FAILS her signature. img_drift/range go WRONG direction (she is still-in-image; imitation roams). pa_osc (her defining concrete⇄abstract oscillation) goes wrong way. |

## The Lispector result (the most important, non-obvious finding)

Lispector's structural signature = high semantic drift / LOW imagistic drift
(meaning churns while one image-field is held). Measured sem/img ratio:
- real Lispector: **5.67**
- Sonnet control (generic "reflect on time/language"): **5.93**
- Sonnet Lispector imitation: **4.45**

And by full-signature distance, the imitation is CLOSER to its own generic control
(0.124) than to real Lispector (0.182).

**So: asked to imitate Lispector, the frontier model moved AWAY from her actual
structure** — it added imagistic variety ("being poetic") which is the opposite of
her held-image discipline. Its *generic* control accidentally sits nearer her
signature than its deliberate imitation does. The model reproduces "philosophical-
poetic voice" (a style register it knows) but not the specific image-stillness/
meaning-churn relationship that constitutes her vitality.

## Interpretation

- Frontier models reproduce signatures that are **surface-nameable and
  imitable-by-register** (Woolf's perception-glide, Joyce's run-on) but miss
  signatures that are **structurally counterintuitive** (Lispector: holding an
  image still WHILE meaning moves — the model's notion of "poetic" pushes toward
  image-variety, the wrong way).
- This is evidence that some literary vitality is a **specific dynamical
  discipline**, not a style you can invoke by name — even a frontier model
  defaults to its generic "literary" attractor rather than the author's actual
  constraint.
- Joyce's deep (allusive/phonetic-pointer) vitality remains untestable by
  text-geometry; the run-on fidelity is real but is the surface, not the depth.
  (See resonances_close_reading.md for the LLM-as-probe route.)

## Allusion probe (resonances_close_reading.md) — REVISES the relational-regime claim

A capable model close-read the Joyce IMITATION for cross-scale resonance and found
it DENSE with recoverable allusion: Proustian involuntary memory, Synge's omen-
gulls, Ecclesiastes' tidal catalogue, Eliot's Thames, Hypnos/Morpheus cave of
sleep, Bergson's durée vs clock-time, Baudelaire's beauty-in-rot, Plato's Phaedrus
horses, Irish emigration-ballad tradition, Molly Bloom's looping close.

This RUNS AGAINST my prediction ("frontier can't produce allusive depth"). Update:
- FALSIFIED: 'frontier models cannot generate allusive depth.' They can; a manifold-
  holding probe can recover it.
- SURVIVES (sharper): the boundary is MODEL-SCALE + INSTRUMENT, not generability.
  0.5B: can't produce OR detect allusion. Frontier: can produce AND detect.
  Text-geometry (CLIP/MiniLM): CANNOT detect it at any scale (polysemy proxy failed
  on real Joyce too). So relational vitality = "measurable only by a knowledge-
  holding model, invisible to pure geometry" — NOT "unreachable."

CRITICAL CAVEAT (circularity): same model FAMILY wrote the imitation AND probed it.
"AI writes allusive-seeming text, AI confirms allusion" is weak — primed to find
depth in its own kind of output. PROPER TEST (not yet run): run the IDENTICAL probe
on (a) real Joyce, (b) the imitation, (c) the control; compare allusion density. If
imitation≈real>control → real. If depth found everywhere → probe over-generous.

ALSO: the probe itself flagged the imitation as Molly-Bloom + Dylan Thomas +
Irish-ballad COMPOSITE — a 'lyrical-Celtic-modernist REGISTER', not specifically
Joyce. Consistent with the Lispector finding: model reaches a known stylistic
register (here richly) rather than the specific author's structural discipline.
Allusions may be register-appropriate FURNITURE, not load-bearing the way
'riverrun' is in Joyce.

## BLIND CROSS-MODEL allusion probe (Grok + Gemini + ChatGPT), KEY: P1=control, P2=real Joyce, P3=imitation

| Judge | Control | Real Joyce | Imitation |
|---|---|---|---|
| Grok | 0.023 | ~0.33 | ~0.29 |
| Gemini | 0.131 | 0.791 | 0.074 |
| ChatGPT | 0.07 | 0.32 | 0.02 |

CONTAMINATION CONFIRMED (user's caveat, verified): all three judges RECOGNIZED P2
as Joyce's "Oxen of the Sun" — named the episode, characters (Costello, Lynch),
Dublin. So P2's high score is confounded by RECOGNITION/reputation, not blind
reading. Blinding fails when the text is memorized. => "real Joyce > imitation" is
NOT cleanly established; the canonical-text advantage is partly recall.

THE CONTAMINATION-FREE FINDING (imitation is NOVEL, unrecognizable): judges SPLIT
hard on it — Grok 0.29 (high) vs Gemini 0.074 / ChatGPT 0.02 (below the control!).
The split is principled: Grok counted "sustained Joycean MODE/modernist technique"
as resonance; Gemini & ChatGPT required SPECIFIC NAMEABLE external references and
found almost none. ChatGPT: "almost all significance generated internally, not
externally nameable cultural references." Gemini: "decorative, not structurally
necessary."

RIGOROUS CONCLUSION: the imitation reproduces the TEXTURE of allusive depth
(modernist interiority, run-on, sensory layering) but not actual REFERENTIAL depth
(specific pointers into the cultural manifold); 2 of 3 blind judges scored it BELOW
the plain control on nameable resonance. = the Lispector finding generalized:
frontier models reach the STYLE REGISTER of depth, not the STRUCTURAL DISCIPLINE /
pointing of it. Held across 3 independent judges on the uncontaminated text.

## Caveats
- n=1 model, 1 passage per condition; sent_cv on the Joyce run-on is degenerate.
- Real-author windows are single ~9k-char samples; Woolf is Voyage Out
  (conventional-leaning), which makes "captured Woolf" a softer claim.
- Embedding/CLIP metrics are proxies; the Lispector finding rests on the sem/img
  ratio being a true signature (validated earlier across 4 windows).
