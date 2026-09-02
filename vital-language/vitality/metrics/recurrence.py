"""Recurrence / drift-with-return metrics on a passage's sentence embeddings.

Motivation (from the GPT consult + our own probe): surprisal WIDTH anti-tracks
vitality; what may track it is *uneven preservation of unresolved structure* --
i.e. the text wanders semantically but RETURNS to earlier material (transformed),
rather than marching to closure or looping verbatim. These metrics try to
capture that, to be correlated against human vitality ratings BEFORE we build
any injection mechanism.

Metrics (per passage, on sentence-embedding trajectory e_1..e_n):
  drift            : mean cosine distance between consecutive sentences (= H,
                     local heterogeneity; expected to behave like surprisal std)
  global_range     : mean distance from the centroid (how far it wanders overall)
  recurrence_rate  : fraction of non-adjacent sentence pairs that are CLOSE
                     (cosine sim > thresh) -- motif return, recurrence-plot density
  return_distance  : for each "return" (a later sentence close to a much earlier
                     one), the index gap; mean gap = how delayed the returns are
                     (delayed recurrence = the SoC signature, not immediate repeat)
  drift_return_ratio: drift / global_range. HIGH = moves a lot locally but stays
                     bounded globally (wanders-and-returns). This is the candidate
                     "vitality" signature: local life without global runaway.
  verbatim_loop    : fraction of adjacent pairs that are NEAR-IDENTICAL (sim>0.97)
                     -- penalizes degenerate repetition (the dead kind of return)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .surface import split_sentences


@dataclass
class RecurrenceMetrics:
    n_sent: int
    drift: float
    global_range: float
    recurrence_rate: float
    return_distance: float
    drift_return_ratio: float
    verbatim_loop: float


def _embed(sentences, model):
    emb = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float64)


def compute_recurrence(text, model, near=0.6, far_gap=5, loop_thresh=0.97):
    sents = split_sentences(text)
    sents = [s for s in sents if len(s.split()) >= 2]
    n = len(sents)
    if n < 6:
        return RecurrenceMetrics(n, *( [float("nan")] * 6 ))

    E = _embed(sents, model)              # [n, d], unit-normalized
    sim = E @ E.T                          # cosine similarity matrix
    dist = 1.0 - sim

    # local drift: consecutive distances
    consec = np.array([dist[i, i + 1] for i in range(n - 1)])
    drift = float(consec.mean())

    # global range: distance from centroid
    centroid = E.mean(0); centroid /= (np.linalg.norm(centroid) + 1e-9)
    global_range = float((1.0 - E @ centroid).mean())

    # recurrence: non-adjacent close pairs
    iu = np.triu_indices(n, k=2)           # exclude diagonal + adjacent
    pair_sim = sim[iu]
    recurrence_rate = float((pair_sim > near).mean())

    # return distance: among close non-adjacent pairs, the index gap
    gaps = (iu[1] - iu[0])[pair_sim > near]
    return_distance = float(gaps.mean()) if len(gaps) else 0.0

    drift_return_ratio = float(drift / (global_range + 1e-9))

    verbatim_loop = float((consec < (1.0 - loop_thresh)).mean())

    return RecurrenceMetrics(
        n_sent=n, drift=round(drift, 4), global_range=round(global_range, 4),
        recurrence_rate=round(recurrence_rate, 4),
        return_distance=round(return_distance, 2),
        drift_return_ratio=round(drift_return_ratio, 4),
        verbatim_loop=round(verbatim_loop, 4),
    )
