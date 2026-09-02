"""Phonetic-flow metrics — the third channel (Semantic x Imagistic x PHONETIC).

Hypothesis (user/GPT): SOC, Joyce especially, is driven partly by SOUND, not only
meaning — "thought rides sound." Machine prose optimizes next-token meaning/syntax
and lets phonetic momentum die, producing the "airless" quality: every sentence
MEANS, nothing RESONATES. We measure hidden musicality, not poetry.

ENGLISH ONLY, original-language only. Uses CMUdict (via `pronouncing`). OOV words
(coined: "riverrun") are skipped, not guessed.

Sub-measures (per passage, over the token sequence):
  alliteration_density : rate of shared word-initial consonant in nearby windows
  phoneme_recurrence   : how much a word's phoneme set overlaps the PRECEDING
                         word's — sonic carry-forward (the "drift rides sound")
  vowel_continuity     : similarity of stressed-vowel sequences between adjacent
                         words (assonance / vowel-pattern continuity)
  stress_variance      : variance of per-word stress-pattern length/shape — rhythm
                         irregularity (uniform = metronomic machine cadence)
  phonetic_flow        : composite = recurrence + alliteration carried ACROSS
                         positions (the momentum signal), the headline number

Prediction: Joyce highest (phonetic whirlpool); machine prose low (dead channel).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pronouncing as pr

_VOWELS = set("AEIOU")  # ARPABET vowels start with these (AA, EH, IY, ...)


@dataclass
class PhoneticMetrics:
    n_words: int
    coverage: float            # fraction of words found in CMUdict
    alliteration_density: float
    phoneme_recurrence: float
    vowel_continuity: float
    stress_variance: float
    phonetic_flow: float


def _phones(word):
    p = pr.phones_for_word(word.lower())
    return p[0].split() if p else None


def _initial_consonant(phones):
    for ph in phones:
        base = re.sub(r"\d", "", ph)
        if base[0] not in _VOWELS:
            return base
        return None  # starts with vowel
    return None


def _vowel_seq(phones):
    return [re.sub(r"\d", "", ph) for ph in phones if re.sub(r"\d", "", ph)[0] in _VOWELS]


def _stress_pattern(phones):
    return "".join(ch for ph in phones for ch in ph if ch.isdigit())


def compute_phonetic(text, window=4):
    words = re.findall(r"[A-Za-z']+", text)
    seq = []  # list of (word, phones) for in-vocab words, in order
    for w in words:
        ph = _phones(w)
        if ph:
            seq.append((w, ph))
    n_in = len(seq)
    if n_in < 10:
        return PhoneticMetrics(len(words), n_in / max(len(words), 1),
                               *( [float("nan")] * 5 ))

    inits = [_initial_consonant(ph) for _, ph in seq]
    # alliteration: within a sliding window, fraction of consonant-initial pairs
    # that share initial consonant
    allit_hits, allit_tot = 0, 0
    for i in range(n_in):
        for j in range(i + 1, min(i + window, n_in)):
            if inits[i] and inits[j]:
                allit_tot += 1
                if inits[i] == inits[j]:
                    allit_hits += 1
    alliteration = allit_hits / allit_tot if allit_tot else 0.0

    # phoneme recurrence: Jaccard overlap of phoneme sets between adjacent words
    recs = []
    for i in range(1, n_in):
        a = set(re.sub(r"\d", "", p) for p in seq[i-1][1])
        b = set(re.sub(r"\d", "", p) for p in seq[i][1])
        if a or b:
            recs.append(len(a & b) / len(a | b))
    phoneme_recurrence = float(np.mean(recs)) if recs else 0.0

    # vowel continuity: overlap of stressed-vowel sequences between adjacent words
    vconts = []
    for i in range(1, n_in):
        va, vb = set(_vowel_seq(seq[i-1][1])), set(_vowel_seq(seq[i][1]))
        if va or vb:
            vconts.append(len(va & vb) / len(va | vb))
    vowel_continuity = float(np.mean(vconts)) if vconts else 0.0

    # stress variance: variance of stress-pattern lengths (rhythmic irregularity)
    slens = [len(_stress_pattern(ph)) for _, ph in seq]
    stress_variance = float(np.var(slens))

    # composite flow: sonic carry-forward (recurrence + local alliteration)
    phonetic_flow = phoneme_recurrence + alliteration

    return PhoneticMetrics(
        n_words=len(words), coverage=round(n_in / len(words), 3),
        alliteration_density=round(alliteration, 4),
        phoneme_recurrence=round(phoneme_recurrence, 4),
        vowel_continuity=round(vowel_continuity, 4),
        stress_variance=round(stress_variance, 4),
        phonetic_flow=round(phonetic_flow, 4),
    )
