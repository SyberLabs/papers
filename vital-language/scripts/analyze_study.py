"""Analyze blind ratings: de-anonymize, aggregate by condition, and test the
central validation question (H3) -- does MEASURED surprisal multifractality
correlate with HUMAN-rated vitality?

If it does NOT, our whole metric stack is measuring something readers don't
feel, and the multifractality numbers are decoration. If it does, the metric
has perceptual grounding and the chaos>matched result becomes meaningful.

Inputs:
  study/key.json            (passage_uid -> condition + metrics)
  ratings_*.json            (one or more rater files exported from rate.html)

Reports:
  - mean rating per (condition, dimension), pooled over raters
  - "most alive" win-rate per condition
  - correlation(surprisal_width, human vitality) across passages  <-- H3
  - inter-rater agreement on vitality (if >=2 raters)

Usage: python scripts/analyze_study.py study/key.json ratings_*.json
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict

import numpy as np

DIMS = ["vitality", "originality", "coherence", "reread", "interiority",
        "resonance", "stimulation", "artificiality"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key")
    ap.add_argument("ratings", nargs="+", help="ratings_*.json files / globs")
    args = ap.parse_args()

    key = json.load(open(args.key, encoding="utf-8"))
    files = []
    for g in args.ratings:
        files.extend(glob.glob(g))
    if not files:
        raise SystemExit("no ratings files matched")

    # collect: per passage, per dim, list of scores across raters
    by_uid = defaultdict(lambda: defaultdict(list))
    alive_votes = defaultdict(lambda: defaultdict(int))  # uid -> condition unused; track per uid
    raters = []
    for f in files:
        data = json.load(open(f, encoding="utf-8"))
        raters.append(data.get("rater", f))
        for uid, r in data["ratings"].items():
            if uid not in key:
                continue
            for k, v in r.get("scores", {}).items():
                by_uid[uid][k].append(v)
            if r.get("alive"):
                alive_votes[uid]["alive"] += 1

    print(f"raters: {raters}\n")

    # aggregate by condition
    cond_dim = defaultdict(lambda: defaultdict(list))
    cond_alive = defaultdict(int)
    cond_total = defaultdict(int)
    vit_pairs = []  # (surprisal_width, mean_vitality) per passage for H3
    for uid, dims in by_uid.items():
        cond = key[uid]["condition"]
        for k, vals in dims.items():
            cond_dim[cond][k].append(np.mean(vals))
        cond_total[cond] += 1
        cond_alive[cond] += alive_votes[uid].get("alive", 0)
        if "vitality" in dims:
            sw = key[uid]["surprisal_width"]
            if np.isfinite(sw):
                vit_pairs.append((sw, np.mean(dims["vitality"])))

    print("Mean rating by condition x dimension:")
    conds = sorted(cond_dim)
    print(f"  {'dim':14s} " + " ".join(f"{c:>14s}" for c in conds))
    for k in DIMS:
        row = " ".join(f"{np.mean(cond_dim[c][k]) if cond_dim[c][k] else float('nan'):14.1f}"
                       for c in conds)
        print(f"  {k:14s} {row}")

    print("\n'Most alive' win-rate by condition:")
    for c in conds:
        wr = cond_alive[c] / cond_total[c] if cond_total[c] else 0
        print(f"  {c:16s} {cond_alive[c]:3d}/{cond_total[c]:3d}  = {wr:.2f}")

    # H3: correlation of measured multifractality with felt vitality
    print("\n--- H3: does measured surprisal-width track FELT vitality? ---")
    if len(vit_pairs) >= 5:
        sw = np.array([p[0] for p in vit_pairs])
        vt = np.array([p[1] for p in vit_pairs])
        # Pearson + Spearman
        r_p = np.corrcoef(sw, vt)[0, 1]
        rank = lambda x: np.argsort(np.argsort(x))
        r_s = np.corrcoef(rank(sw), rank(vt))[0, 1]
        # bootstrap CI on Pearson
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(5000):
            i = rng.integers(0, len(sw), len(sw))
            if sw[i].std() > 0 and vt[i].std() > 0:
                boot.append(np.corrcoef(sw[i], vt[i])[0, 1])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  n={len(vit_pairs)} passages")
        print(f"  Pearson  r = {r_p:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
        print(f"  Spearman r = {r_s:+.3f}")
        verdict = ("POSITIVE: measured multifractality tracks felt vitality"
                   if lo > 0 else
                   "NULL/UNCLEAR: metric may not capture felt vitality" if hi > 0
                   else "NEGATIVE: metric anti-correlates with felt vitality")
        print(f"  -> {verdict}")
    else:
        print(f"  not enough rated passages yet (have {len(vit_pairs)}, need >=5)")

    if len(raters) >= 2:
        print("\nInter-rater vitality agreement: collect >=2 full rater files, "
              "then compare per-passage vitality means (extend here as needed).")


if __name__ == "__main__":
    main()
