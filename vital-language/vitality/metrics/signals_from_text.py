"""Turn a text into the time-series we run MFDFA on.

Two signals, treated symmetrically:

  surprisal       : per-token -log p (dense, ~hundreds of points per passage).
                    Our primary signal; what the intervention directly perturbs.

  sentence_length : words per sentence, as a series across the text. This is the
                    EXACT signal used by the IFJ PAN group (Drozdz, Oswiecimka et
                    al.) whose Finnegans Wake multifractality result motivated the
                    project, so it is our bridge to the external literature. The
                    catch: it needs MANY sentences (they used whole novels, 1e4+).
                    On a 220-token passage we get ~15 -- far too few for credible
                    MFDFA. Use the long-passage generation mode for this signal.
"""

from __future__ import annotations

import numpy as np

from .surface import split_sentences, tokenize_words


def surprisal_series(surprisal: np.ndarray) -> np.ndarray:
    s = np.asarray(surprisal, dtype=np.float64)
    return s[np.isfinite(s)]


def sentence_length_series(text: str) -> np.ndarray:
    """Words per sentence, in document order (the IFJ signal)."""
    sents = split_sentences(text)
    return np.array([len(tokenize_words(s)) for s in sents], dtype=np.float64)


def series_from_text(text: str, surprisal: np.ndarray | None, name: str) -> np.ndarray:
    if name == "surprisal":
        if surprisal is None:
            raise ValueError("surprisal series requested but no surprisal provided")
        return surprisal_series(surprisal)
    if name == "sentence_length":
        return sentence_length_series(text)
    raise ValueError(f"unknown series '{name}'")
