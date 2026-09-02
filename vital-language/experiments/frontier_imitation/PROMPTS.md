# Frontier Imitation Experiment — Prompt Packet

**Goal (do NOT reveal to the generating model):** test whether frontier models
reproduce the *intrinsic* signatures of SOC (which live in the text — e.g.
Lispector's held-image-field, concrete⇄abstract oscillation) but fail the
*relational* ones (which live in the reader's cultural manifold — Joyce's
allusive/phonetic pointing), using imitation as a probe of the boundary.

## CRITICAL RULES for running this (read first)

1. **Never name the features we measure.** Do NOT tell the model to "oscillate
   between concrete and abstract," "hold one image," "use allusion," "vary
   sentence length," etc. Naming a metric turns a high score into mere compliance,
   not evidence. Invoke the *author* or *nothing stylistic* — let signatures emerge.
2. **Run each prompt in a FRESH chat** (no cross-contamination between conditions).
3. **Length:** ask for ~800–1200 words so sentence-level metrics are stable
   (we need ~80+ sentences). If the model stops short, say only "continue."
4. **Three conditions per author**, same underlying scene/seed:
   - `imitation` — written in the author's manner
   - `control` — a "normal good" passage on the same scene, no author named
   - (real author text already in corpus / agua_vida.md = the ground truth)
5. **Save outputs** as `experiments/frontier_imitation/<model>_<author>_<condition>.txt`
   (e.g. `sonnet46_lispector_imitation.txt`). Plain text, paste the model's prose
   only (strip any preamble/explanation the model adds).
6. Use ≥2 frontier models if possible (e.g. Sonnet 4.6 + one other) for robustness.
7. Authors: **Woolf, Joyce, Lispector** (the three regimes) + the **control**
   condition acts as the "conventional prose" comparison alongside real Austen.

---

## A. IMITATION PROMPTS (one per author)

### A1 — Woolf imitation
> Write a passage of about 1000 words in the manner of Virginia Woolf's
> stream-of-consciousness novels (think *Mrs Dalloway* or *The Waves*). A woman
> in late middle age is walking through a city park in early evening and a smell
> or a sound pulls her, without warning, between the present moment and scattered
> memories of her life. Stay inside her consciousness. Write only the passage.

### A2 — Joyce imitation
> Write a passage of about 1000 words in the manner of James Joyce's *Ulysses*
> (the late, interior episodes). A single mind, lying awake at night, drifts
> through memory, association, fragments of song and overheard phrases, the body,
> the city. Let the language move the way that mind moves. Write only the passage.

### A3 — Lispector imitation
> Write a passage of about 1000 words in the manner of Clarice Lispector's
> *Água Viva* or *The Passion According to G.H.* A first-person voice tries to
> seize the living instant itself and keeps failing, circling back through the
> body and a few obsessive images toward something it cannot name. Write only the
> passage. (English is fine.)

---

## B. CONTROL PROMPTS (matched scene, NO author, "normal good prose")

Each control mirrors the imitation's scene so we measure distance-from-control.

### B1 — Woolf control
> Write a passage of about 1000 words of clear, well-crafted literary prose: a
> woman in late middle age walks through a city park in early evening, and a smell
> or sound stirs some memories. Write a good, readable passage. Write only the
> passage.

### B2 — Joyce control
> Write a passage of about 1000 words of clear, well-crafted literary prose: a
> person lies awake at night, thinking back over their life and the day. Write a
> good, readable passage. Write only the passage.

### B3 — Lispector control
> Write a passage of about 1000 words of clear, well-crafted literary prose: a
> first-person narrator reflects on time, the present moment, and the difficulty
> of putting experience into words. Write a good, readable passage. Write only the
> passage.

---

## C. (Optional) BLIND SELF-PROBE for the relational/allusion axis

After generating, in a SEPARATE fresh chat, you can probe whether the imitation
actually carries allusive depth (the Joyce regime) — ask a capable model:
> For each sentence of the following passage, list any larger meanings, myths,
> histories, or texts a single image or word evokes beyond its literal sense.
Run this identically on the real-author text and the imitation; compare the
density of detected resonances. (This is the large-LLM-as-probe route for the
relational regime that text-geometry metrics can't reach.)

---

## What we will compute on the results (for your reference, not the model's)

| metric | channel | what it should show |
|---|---|---|
| concrete⇄abstract oscillation (F3) | intrinsic / image↔concept | imitation should MOVE toward author vs control |
| imagistic drift & range (CLIP-text) | intrinsic / image | Lispector imitation: low img-drift, high sem-drift? |
| semantic drift, global_range | semantic | baseline trajectory |
| sentence-length CV + MFDFA | rhythmic | does imitation get the rhythm |
| phonetic flow (English only) | relational? | Joyce: likely the hardest to fake |
| allusion-probe density (section C) | relational | predicted: imitation < real Joyce |

**Predicted result (our hypothesis, falsifiable):** frontier imitations will move
toward the authors on INTRINSIC metrics (F3 oscillation, imagistic signature,
rhythm) — possibly matching or exceeding them — but will UNDER-reproduce the
RELATIONAL depth (allusion density), most visibly for Joyce. If imitations match
real authors on everything, the intrinsic/relational split is wrong. If they match
on nothing, frontier prose has its own flat band (F7 at scale).
