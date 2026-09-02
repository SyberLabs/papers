"""Refined SOC vitality metrics, derived from reading the tradition (esp. the
Lispector falsification case). Language-fair: uses multilingual sentence
embeddings and embedding-anchored concreteness (no English-only lexicon).

Two metrics:

A) bounded_agitation -- the Lispector signature: high LOCAL drift held within a
   BOUNDED semantic region (restless circling of an obsession, not roaming).
   Operationalized as drift / global_range is the WRONG framing (we found that
   ratio negative). Instead:
     local_drift   = mean consecutive-sentence distance        (agitation)
     global_range  = mean distance from centroid               (containment, LOW=contained)
     fill_ratio    = total path length / (range * n)           (space-filling: how
                     much the trajectory MOVES per unit of territory it covers)
   bounded_agitation = local_drift * (1 - global_range)  -> HIGH when it moves a
   lot locally while staying contained. This is the quantity that should rank
   Lispector (move-a-lot, stay-put) ABOVE a text that wanders calmly.

B) perception_abstraction_oscillation (Feature 3) -- the Woolf/Lispector engine.
   Embedding-anchored: each sentence gets a concreteness score = sim(concrete
   anchors) - sim(abstract anchors) in the shared multilingual space. The METRIC
   is the OSCILLATION (std + lag-1 sign-flip rate) of that score across
   sentences -- the mind touching the world then pulling back to think. Mean
   concreteness is NOT the signal; the back-and-forth is.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# Anchors are given in BOTH English and Spanish so the multilingual embedder
# centroids are robust regardless of text language.
CONCRETE_ANCHORS = [
    "the cold wet stone in my hand", "blood on the white sheet",
    "light flickering on the water", "the oyster writhing under lemon juice",
    "her hair against the rough bark", "the smell of rain on hot dust",
    "la piedra fría y mojada en mi mano", "sangre sobre la sábana blanca",
    "la luz temblando sobre el agua", "la ostra retorciéndose bajo el limón",
    "el olor de la lluvia sobre el polvo caliente", "el casco seco del caballo",
]
ABSTRACT_ANCHORS = [
    "the concept of being itself", "time as pure abstraction",
    "the meaning that escapes all meaning", "the idea of the eternal present",
    "consciousness reflecting on consciousness", "the limits of language",
    "el concepto del ser mismo", "el tiempo como pura abstracción",
    "el sentido que escapa a todo sentido", "la idea del presente eterno",
    "la conciencia que se refleja a sí misma", "los límites del lenguaje",
]


@dataclass
class SOCMetrics:
    n_sent: int
    local_drift: float
    global_range: float
    bounded_agitation: float
    fill_ratio: float
    concreteness_mean: float
    pa_oscillation_std: float
    pa_flip_rate: float


def split_sents(text, min_words=3):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
            if len(s.split()) >= min_words]


def compute_soc(text, model, max_sents=400):
    sents = split_sents(text)[:max_sents]
    n = len(sents)
    if n < 8:
        return SOCMetrics(n, *([float("nan")] * 7))

    E = np.asarray(model.encode(sents, normalize_embeddings=True,
                                show_progress_bar=False), dtype=np.float64)
    cA = np.asarray(model.encode(CONCRETE_ANCHORS, normalize_embeddings=True,
                                 show_progress_bar=False)).mean(0)
    cB = np.asarray(model.encode(ABSTRACT_ANCHORS, normalize_embeddings=True,
                                 show_progress_bar=False)).mean(0)
    cA /= np.linalg.norm(cA); cB /= np.linalg.norm(cB)

    consec = np.array([1.0 - float(E[i] @ E[i + 1]) for i in range(n - 1)])
    local_drift = float(consec.mean())
    cen = E.mean(0); cen /= (np.linalg.norm(cen) + 1e-9)
    global_range = float((1.0 - E @ cen).mean())
    path_len = float(consec.sum())
    fill_ratio = path_len / (global_range * n + 1e-9)
    bounded_agitation = local_drift * (1.0 - global_range)

    # concreteness score per sentence, then oscillation
    conc = (E @ cA) - (E @ cB)
    conc = (conc - conc.mean())  # center; we care about variation
    pa_osc_std = float(conc.std())
    # sign-flip rate of the de-meaned concreteness series (back-and-forth)
    signs = np.sign(conc)
    flips = np.sum(signs[:-1] * signs[1:] < 0)
    pa_flip_rate = float(flips / (n - 1))

    return SOCMetrics(
        n_sent=n, local_drift=round(local_drift, 4),
        global_range=round(global_range, 4),
        bounded_agitation=round(bounded_agitation, 4),
        fill_ratio=round(fill_ratio, 4),
        concreteness_mean=round(float(((E @ cA) - (E @ cB)).mean()), 4),
        pa_oscillation_std=round(pa_osc_std, 4),
        pa_flip_rate=round(pa_flip_rate, 4),
    )
