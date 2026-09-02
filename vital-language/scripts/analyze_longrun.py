"""Analyze the focused long run: pooled sentence-length multifractality per
condition, with block-bootstrap CIs and a shuffle surrogate.

Why pooled: a single ~1500-token passage gives ~60-120 sentences -- borderline
for MFDFA. Concatenating all passages of a condition gives several hundred
sentences for a stable estimate. We BLOCK-bootstrap (resample whole passages,
not individual sentences) so within-passage long-range correlation -- the actual
signal -- is preserved in each resample.

Surrogate: shuffle the pooled series. Genuine multifractality from long-range
ordering should collapse under shuffling (as it did for Joyce/Woolf in the
literary benchmark). If a condition's width does NOT drop when shuffled, its
"multifractality" is a distributional artifact, not vitality.

The headline test: is logit_chaos's pooled width > logit_matched's, with
non-overlapping bootstrap CIs, AND does chaos's width survive shuffling worse
(i.e. depend more on ordering)?

Usage: python scripts/analyze_longrun.py outputs/longrun_XXXX/longrun.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vitality.metrics.mfdfa import mfdfa

MIN_LEN = 120  # pooled, so we can afford the proper floor


def pooled_width(series_list, rng=None):
    """Concatenate passage series and return MFDFA width (nan if too short)."""
    if not series_list:
        return float("nan")
    pooled = np.concatenate(series_list)
    if len(pooled) < MIN_LEN:
        return float("nan")
    return mfdfa(pooled, min_len=MIN_LEN, n_scales=16).width


def block_bootstrap_ci(series_list, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    k = len(series_list)
    widths = []
    for _ in range(n_boot):
        idx = rng.integers(0, k, k)  # resample whole passages
        w = pooled_width([series_list[i] for i in idx])
        if np.isfinite(w):
            widths.append(w)
    if not widths:
        return float("nan"), float("nan"), float("nan")
    return (float(np.mean(widths)),
            float(np.percentile(widths, 2.5)),
            float(np.percentile(widths, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()

    by_cond = defaultdict(list)   # condition -> list of sentence-length arrays
    surp_by_cond = defaultdict(list)  # condition -> list of per-passage surprisal width
    coh_by_cond = defaultdict(list)
    excluded = defaultdict(int)
    # QUALITY GATE: passages whose tail degenerated into 1-word/0-word "sentences"
    # produce square-wave series that MFDFA misreads as huge multifractality (we
    # saw width=17 with fit_r2=0.63). We must not pool garbage. A passage enters
    # the pool only if it is legible AND not degenerate by sentence structure.
    PPL_MAX = 25.0          # legibility ceiling (~3x typical base PPL)
    MIN_SENT = 30           # need enough sentences to matter
    MAX_DEGEN_FRAC = 0.30   # reject if >30% of sentences are <=1 word (degenerate)
    for line in open(args.path, encoding="utf-8"):
        r = json.loads(line)
        sl_raw = np.array(r["mfdfa"]["sentence_length_series"], dtype=np.float64)
        sl = sl_raw[sl_raw >= 1]  # drop 0-word (punctuation-only) fragments
        ppl = r["coherence"]["self_perplexity"]
        degen_frac = float(np.mean(sl_raw <= 1)) if len(sl_raw) else 1.0
        coh_by_cond[r["condition"]].append(ppl)
        # gate
        if (len(sl) >= MIN_SENT and np.isfinite(ppl) and ppl <= PPL_MAX
                and degen_frac <= MAX_DEGEN_FRAC):
            by_cond[r["condition"]].append(sl)
        else:
            excluded[r["condition"]] += 1
        if r["mfdfa"]["surprisal"]["ok"] and np.isfinite(ppl) and ppl <= PPL_MAX:
            surp_by_cond[r["condition"]].append(r["mfdfa"]["surprisal"]["width"])

    print("Quality gate (legible, >=%d sentences, <=%.0f%% degenerate):" %
          (MIN_SENT, MAX_DEGEN_FRAC * 100))
    for c in sorted(set(list(by_cond) + list(excluded))):
        print(f"  {c:16s} kept {len(by_cond[c]):2d}, excluded {excluded[c]:2d}")
    print()

    print("POOLED sentence-length multifractality (the IFJ signal)\n")
    print(f"{'condition':16s} {'npass':>5} {'nsent':>6} {'width':>7} "
          f"{'CI95':>16} {'shuf_w':>7} {'meanPPL':>8}")
    print("-" * 72)
    summary = {}
    for cond, series_list in sorted(by_cond.items()):
        nsent = sum(len(s) for s in series_list)
        w = pooled_width(series_list)
        mean_w, lo, hi = block_bootstrap_ci(series_list)
        # shuffle surrogate on the pool
        pooled = np.concatenate(series_list)
        shuf = pooled.copy(); np.random.default_rng(0).shuffle(shuf)
        shuf_w = mfdfa(shuf, min_len=MIN_LEN, n_scales=16).width if len(shuf) >= MIN_LEN else float("nan")
        ppl = float(np.mean(coh_by_cond[cond]))
        summary[cond] = (w, lo, hi, shuf_w, ppl)
        print(f"{cond:16s} {len(series_list):5d} {nsent:6d} {w:7.3f} "
              f"[{lo:5.2f},{hi:5.2f}] {shuf_w:7.3f} {ppl:8.2f}")

    print("\nPer-passage surprisal multifractality (mean width):")
    for cond in sorted(surp_by_cond):
        vals = surp_by_cond[cond]
        print(f"  {cond:16s} {np.mean(vals):.3f} (n={len(vals)})")

    # headline contrast
    print("\n--- HEADLINE: chaos vs matched (sentence-length, pooled) ---")
    if "logit_chaos" in summary and "logit_matched" in summary:
        cw, clo, chi, csh, _ = summary["logit_chaos"]
        mw, mlo, mhi, msh, _ = summary["logit_matched"]
        print(f"  chaos   width={cw:.3f} CI[{clo:.2f},{chi:.2f}] shuffled={csh:.3f}")
        print(f"  matched width={mw:.3f} CI[{mlo:.2f},{mhi:.2f}] shuffled={msh:.3f}")
        sep = (clo > mhi) or (mlo > chi)
        print(f"  CIs {'DO NOT overlap -> SEPARATION' if sep else 'overlap -> no clear separation'}")
        print(f"  ordering-dependence (width drop when shuffled): "
              f"chaos {cw - csh:+.3f}, matched {mw - msh:+.3f}")
        print("  (vitality from real long-range structure should show a LARGE drop)")


if __name__ == "__main__":
    main()
