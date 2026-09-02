"""Coherence guardrail metrics for the epsilon sweep.

The whole point of Week 2 is to push epsilon up until structure appears, but
NOT past the point where prose stops being legible. We need a coherence number
that actually degrades with eps. Two notes on why the obvious choice fails:

  - argmax-preservation is useless here: the topk_complement mask protects the
    argmax by construction, so it is ~1.0 regardless of eps.
  - effective-surprisal inflation measures how hard we *pushed*, not whether the
    result is coherent -- a strong push that the language still absorbs is fine.

So the primary guardrail is SELF-PERPLEXITY under the UNPERTURBED base model:
re-score the generated text with the clean model. Text the model finds
implausible (word salad, broken syntax) scores high. This is the operational
meaning of "the model no longer recognizes this as well-formed language."

We pair it with a degeneration check (n-gram repetition), because the other
failure mode of over-perturbation is collapse into loops, which perplexity
alone can rate as "confident."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class CoherenceMetrics:
    self_perplexity: float  # base-model PPL of the generated text (lower=coherent)
    distinct_2: float  # distinct bigram ratio (lower => more repetition)
    distinct_3: float
    rep_4gram: float  # share of 4-grams that are repeats (higher=degenerate)


@torch.no_grad()
def self_perplexity(model, tokenizer, text: str, device: str = "cpu") -> float:
    """Perplexity of `text` under the clean model (teacher-forced)."""
    if not text.strip():
        return float("nan")
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    if ids.shape[1] < 2:
        return float("nan")
    out = model(ids, labels=ids)
    return float(torch.exp(out.loss).item())


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def degeneration(tokens: list[str]) -> tuple[float, float, float]:
    """distinct-2, distinct-3, and repeated-4gram fraction."""
    def distinct(n):
        g = _ngrams(tokens, n)
        return len(set(g)) / len(g) if g else 0.0

    g4 = _ngrams(tokens, 4)
    rep4 = 1.0 - (len(set(g4)) / len(g4)) if g4 else 0.0
    return distinct(2), distinct(3), rep4


def compute_coherence(
    model, tokenizer, text: str, words: list[str], device: str = "cpu"
) -> CoherenceMetrics:
    d2, d3, rep4 = degeneration(words)
    return CoherenceMetrics(
        self_perplexity=round(self_perplexity(model, tokenizer, text, device), 3),
        distinct_2=round(d2, 4),
        distinct_3=round(d3, 4),
        rep_4gram=round(rep4, 4),
    )
