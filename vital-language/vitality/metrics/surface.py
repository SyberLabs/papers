"""Surface + structural metrics over a generated passage.

Kept dependency-light (regex + numpy). Embedding-based metrics live in
structural.py so the basic suite runs without loading a sentence model.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

_WORD = re.compile(r"[A-Za-z']+")
_SENT = re.compile(r"[^.!?]+[.!?]+|\S[^.!?]*$")


def tokenize_words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.findall(text) if s.strip()]


def mattr(words: list[str], window: int = 50) -> float:
    """Moving-Average Type-Token Ratio: TTR is length-biased; MATTR isn't.

    This matters because conditions may produce different lengths; raw TTR would
    confound lexical diversity with passage length."""
    if len(words) < window:
        return len(set(words)) / len(words) if words else 0.0
    ratios = [
        len(set(words[i : i + window])) / window
        for i in range(len(words) - window + 1)
    ]
    return float(np.mean(ratios))


def burstiness(values: np.ndarray) -> float:
    """Goh-Barabasi burstiness B = (sigma - mu)/(sigma + mu) in [-1, 1].
    +1 bursty, 0 Poisson-ish, -1 regular. Computed here on inter-event style
    series (e.g. sentence lengths or surprisal)."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2:
        return 0.0
    mu, sigma = v.mean(), v.std()
    return float((sigma - mu) / (sigma + mu)) if (sigma + mu) > 0 else 0.0


@dataclass
class SurfaceMetrics:
    n_words: int
    n_sentences: int
    mattr_50: float
    ttr: float
    sent_len_mean: float
    sent_len_std: float
    sent_len_cv: float  # coefficient of variation -- rhythm irregularity
    sent_len_burstiness: float
    hapax_ratio: float  # share of words occurring exactly once
    punct_rate: float  # punctuation marks per word
    surprisal_mean: float
    surprisal_std: float
    surprisal_burstiness: float


def compute_surface(text: str, surprisal: np.ndarray | None = None) -> SurfaceMetrics:
    words = tokenize_words(text)
    sents = split_sentences(text)
    sent_lens = np.array([len(tokenize_words(s)) for s in sents], dtype=np.float64)
    counts = Counter(words)
    hapax = sum(1 for c in counts.values() if c == 1)
    n_punct = len(re.findall(r"[.,;:!?\-—()]", text))

    s = np.asarray(surprisal, dtype=np.float64) if surprisal is not None else np.array([])

    cv = float(sent_lens.std() / sent_lens.mean()) if len(sent_lens) and sent_lens.mean() > 0 else 0.0
    return SurfaceMetrics(
        n_words=len(words),
        n_sentences=len(sents),
        mattr_50=round(mattr(words), 4),
        ttr=round(len(set(words)) / len(words), 4) if words else 0.0,
        sent_len_mean=round(float(sent_lens.mean()) if len(sent_lens) else 0.0, 3),
        sent_len_std=round(float(sent_lens.std()) if len(sent_lens) else 0.0, 3),
        sent_len_cv=round(cv, 4),
        sent_len_burstiness=round(burstiness(sent_lens), 4),
        hapax_ratio=round(hapax / len(words), 4) if words else 0.0,
        punct_rate=round(n_punct / len(words), 4) if words else 0.0,
        surprisal_mean=round(float(s.mean()), 4) if len(s) else 0.0,
        surprisal_std=round(float(s.std()), 4) if len(s) else 0.0,
        surprisal_burstiness=round(burstiness(s), 4) if len(s) else 0.0,
    )
