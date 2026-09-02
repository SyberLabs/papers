"""Go/no-go gate: does OUR MFDFA estimator reproduce the known finding that
stream-of-consciousness prose is more strongly multifractal than conventional
prose, on the SAME signal (sentence length) the IFJ PAN group used?

If a strongly-SOC text (Ulysses) does not show a wider singularity spectrum than
conventional prose (Austen) on sentence-length series, our measurement tool is
not trustworthy and nothing downstream means anything.

We also report a phase-randomized surrogate for each text: shuffling sentence
order destroys long-range correlation. A genuinely multifractal series should
LOSE most of its spectrum width under shuffling; if the shuffled width is as
wide as the real one, the "multifractality" was an artifact of the value
distribution, not temporal structure.
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vitality.metrics.signals_from_text import sentence_length_series
from vitality.metrics.mfdfa import mfdfa

CORPUS = {
    "ulysses (Joyce, SOC)": "corpus/ulysses.txt",
    "voyage_out (Woolf, modernist)": "corpus/woolf_voyage.txt",
    "pride (Austen, conventional)": "corpus/austen_pride.txt",
}


def strip_gutenberg(t: str) -> str:
    m = re.search(r"\*\*\* ?START OF.*?\*\*\*", t, re.S)
    if m:
        t = t[m.end():]
    m = re.search(r"\*\*\* ?END OF", t, re.S)
    if m:
        t = t[: m.start()]
    return t


def main():
    print(f"{'text':32s} {'n_sent':>7} {'width':>7} {'hurst':>7} {'fit_r2':>7} "
          f"{'shuf_w':>7}")
    print("-" * 74)
    results = {}
    for label, path in CORPUS.items():
        raw = open(path, encoding="utf-8", errors="ignore").read()
        text = strip_gutenberg(raw)
        series = sentence_length_series(text)
        # clip absurd outliers (chapter headings etc.) at the 99.5th pct
        cap = np.percentile(series, 99.5)
        series = np.clip(series, 1, cap)

        r = mfdfa(series, n_scales=16)

        # surrogate: shuffle sentence order -> kills temporal correlation
        rng = np.random.default_rng(0)
        shuf = series.copy()
        rng.shuffle(shuf)
        rs = mfdfa(shuf, n_scales=16)

        results[label] = r
        print(f"{label:32s} {r.n:7d} {r.width:7.3f} {r.hurst:7.3f} "
              f"{r.fit_r2:7.3f} {rs.width:7.3f}")

    # the test
    print("\n--- gate ---")
    uly = results["ulysses (Joyce, SOC)"].width
    aus = results["pride (Austen, conventional)"].width
    print(f"Ulysses width ({uly:.3f}) vs Austen width ({aus:.3f}): "
          f"{'PASS - SOC wider' if uly > aus else 'FAIL - estimator suspect'}")
    print("(Also check: shuf_w should be NOTABLY < width for the SOC texts,\n"
          " confirming the multifractality is in the ordering, not the values.)")


if __name__ == "__main__":
    main()
