"""Advisor's candidate vitality metrics (Stage 0 follow-up).

Three computable metrics, each with a stated prediction. The real test is the
CLEAN-subset correlation: Stage 0 showed raw correlations are inflated by
degeneracy (template/quiz/loop substrate), so a metric that merely detects
broken text is useless. We want one that discriminates vitality AMONG coherent
passages.

1. coherent_advance_rate  -- fraction of sentences that move to new semantic
   ground (high drift) WHILE staying on-manifold (coherent with recent context).
   Penalizes both salad (drift, no coherence) and templated mush (coherence, no
   drift). Prediction: beats raw drift/global_range (+0.56) by excluding both
   failure modes.

2. concreteness_density    -- rate of concrete/sensory/proper-noun/particular
   tokens per 100 words. Readers reliably call concrete prose 'alive', abstract
   prose 'dead', and none of our dynamical metrics capture this. Prediction:
   positive, possibly rivaling trajectory metrics; partly confounded with advance.

3. semantic_predictability_curve -- how much sentence t constrains t+k as k
   grows (here: mean cosine sim at lag k). Dead/looping text: high at all lags.
   Salad: ~0 even at k=1. Vital text: moderate, slowly decaying. We summarize the
   curve by (sim@lag1, decay slope). Prediction: vital text has moderate lag1 +
   gentle decay; this reconstructs the long-range-correlation intuition at the
   semantic layer. (Pooled across passages for stability; per-passage here as a
   first look.)
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .surface import split_sentences, tokenize_words

# crude but defensible concreteness lexicon cues. Real version would use a
# concreteness norm table (Brysbaert); this is a computable proxy: sensory verbs,
# body/nature/object nouns, and we add proper-noun + numeral density separately.
_SENSORY = set("""
saw see seen look looked watching watched glance stare gaze
heard hear sound noise silence ring hum whisper rustle echo
touch felt feel cold warm hot wet dry rough smooth soft hard
smell scent fragrance reek
taste bitter sweet salt sour
light dark shadow glow gleam shine bright dim
red blue green yellow black white gray golden silver
stone water river sea ocean tree wood grass field sky moon sun rain wind snow
hand face eye eyes hair skin mouth lips breath
door window wall floor roof table chair bed lamp candle key glass
""".split())


@dataclass
class AdvanceMetrics:
    n_sent: int
    coherent_advance_rate: float
    concreteness_density: float
    proper_numeral_density: float
    sem_lag1: float
    sem_decay: float


def _embed(sents, model):
    return np.asarray(model.encode(sents, normalize_embeddings=True,
                                   show_progress_bar=False), dtype=np.float64)


def compute_advance(text, model, drift_hi=0.35, coh_floor=0.25, ctx=3):
    sents = [s for s in split_sentences(text) if len(s.split()) >= 2]
    n = len(sents)
    words = tokenize_words(text)
    nw = max(len(words), 1)

    # concreteness: sensory-lexicon hits per 100 words
    sens = sum(1 for w in words if w in _SENSORY)
    concreteness = 100.0 * sens / nw
    # proper nouns (capitalized mid-sentence) + numerals per 100 words
    propers = len(re.findall(r"(?<=[a-z\s])\b[A-Z][a-z]{2,}", text))
    numerals = len(re.findall(r"\b\d+\b", text))
    proper_num = 100.0 * (propers + numerals) / nw

    if n < 6:
        return AdvanceMetrics(n, float("nan"), round(concreteness, 3),
                              round(proper_num, 3), float("nan"), float("nan"))

    E = _embed(sents, model)
    # coherent advance: drift high AND coherence with running context above floor
    adv = 0
    for i in range(1, n):
        drift_i = 1.0 - float(E[i] @ E[i - 1])
        c0 = max(0, i - ctx)
        ctx_vec = E[c0:i].mean(0); ctx_vec /= (np.linalg.norm(ctx_vec) + 1e-9)
        coh_i = float(E[i] @ ctx_vec)
        if drift_i >= drift_hi and coh_i >= coh_floor:
            adv += 1
    coherent_advance_rate = adv / (n - 1)

    # semantic predictability curve: mean cos sim at lag k
    sims = E @ E.T
    def lag_sim(k):
        d = np.array([sims[i, i + k] for i in range(n - k)])
        return float(d.mean()) if len(d) else np.nan
    lag1 = lag_sim(1)
    # decay slope over lags 1..min(6,n-1)
    K = min(6, n - 1)
    lags = np.arange(1, K + 1)
    vals = np.array([lag_sim(k) for k in lags])
    ok = np.isfinite(vals)
    decay = float(np.polyfit(lags[ok], vals[ok], 1)[0]) if ok.sum() >= 2 else np.nan

    return AdvanceMetrics(
        n_sent=n,
        coherent_advance_rate=round(coherent_advance_rate, 4),
        concreteness_density=round(concreteness, 3),
        proper_numeral_density=round(proper_num, 3),
        sem_lag1=round(lag1, 4),
        sem_decay=round(decay, 5),
    )
