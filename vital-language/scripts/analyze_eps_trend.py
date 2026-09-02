"""Does the chaos-vs-matched surprisal-multifractality effect strengthen with eps?

This is the focused question for the eps sweep. For each eps we compute the
PAIRED (by prompt+seed) difference in surprisal MFDFA width between conditions,
gated by legibility (drop passages whose self-perplexity blew past the ceiling,
since an incoherent passage's multifractality is meaningless).

Reports, per eps:
  - chaos - matched  (paired mean + bootstrap 95% CI)
  - chaos - sampling
  - mean self-perplexity per condition (the coherence cost)
and finds the eps with the largest SIGNIFICANT positive chaos-matched gap that
still keeps chaos legible -- the operating point worth taking to a bigger model.

Usage: python scripts/analyze_eps_trend.py outputs/sweep_XXXX/sweep.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import numpy as np

PPL_MAX = 25.0  # legibility gate; incoherent passages excluded


def load(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        rows.append({
            "cond": r["condition"], "eps": r["eps"],
            "prompt": r["prompt_id"], "seed": r["seed"],
            "surpW": r["mfdfa"]["width"] if r["mfdfa"]["ok"] else np.nan,
            "ppl": r["coherence"]["self_perplexity"],
        })
    return rows


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def paired(rows, eps, a, b):
    """Paired (prompt,seed) surpW[a]-surpW[b] at this eps, legible only."""
    idx = defaultdict(dict)
    for r in rows:
        if r["eps"] != eps:
            continue
        if not np.isfinite(r["surpW"]) or not np.isfinite(r["ppl"]) or r["ppl"] > PPL_MAX:
            continue
        idx[(r["prompt"], r["seed"])][r["cond"]] = r["surpW"]
    diffs = [v[a] - v[b] for v in idx.values() if a in v and b in v]
    return np.array(diffs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()
    rows = load(args.path)
    eps_vals = sorted({r["eps"] for r in rows})

    print("Surprisal multifractality: paired chaos vs controls, by eps "
          f"(legible only, ppl<={PPL_MAX:.0f})\n")
    print(f"{'eps':>4} | {'chaos-matched (CI95)':>30} | {'chaos-sampling (CI95)':>30} "
          f"| {'pplC':>6} {'pplM':>6} {'pplS':>6}")
    print("-" * 100)

    best = None
    for eps in eps_vals:
        cm = paired(rows, eps, "logit_chaos", "logit_matched")
        # sampling is eps-invariant (stored at eps=0); compare chaos@eps to sampling@0
        idx = defaultdict(dict)
        for r in rows:
            if not np.isfinite(r["surpW"]) or not np.isfinite(r["ppl"]) or r["ppl"] > PPL_MAX:
                continue
            if r["cond"] == "logit_chaos" and r["eps"] == eps:
                idx[(r["prompt"], r["seed"])]["c"] = r["surpW"]
            if r["cond"] == "sampling":
                idx[(r["prompt"], r["seed"])]["s"] = r["surpW"]
        cs = np.array([v["c"] - v["s"] for v in idx.values() if "c" in v and "s" in v])

        def ppl(cond, e):
            v = [r["ppl"] for r in rows if r["cond"] == cond and r["eps"] == e
                 and np.isfinite(r["ppl"])]
            return np.mean(v) if v else np.nan

        m_cm, lo_cm, hi_cm = boot_ci(cm)
        m_cs, lo_cs, hi_cs = boot_ci(cs)
        sig_cm = "*" if (lo_cm > 0) else " "
        pC, pM = ppl("logit_chaos", eps), ppl("logit_matched", eps)
        pS = ppl("sampling", 0.0)
        print(f"{eps:4.1f} | {m_cm:+.3f} [{lo_cm:+.3f},{hi_cm:+.3f}]{sig_cm:>2} (n={len(cm):2d})"
              f" | {m_cs:+.3f} [{lo_cs:+.3f},{hi_cs:+.3f}]   (n={len(cs):2d})"
              f" | {pC:6.1f} {pM:6.1f} {pS:6.1f}")

        if eps > 0 and lo_cm > 0 and np.isfinite(pC) and pC <= PPL_MAX:
            if best is None or m_cm > best[1]:
                best = (eps, m_cm)

    print()
    if best:
        print(f">>> Largest SIGNIFICANT chaos>matched gap at legible eps: "
              f"eps={best[0]} (delta={best[1]:+.3f}). Take this eps to the bigger model.")
    else:
        print(">>> No eps shows a SIGNIFICANT chaos>matched surprisal gap. "
              "The 0.5B trend does not reach significance at any eps -> the "
              "bigger-model test is the decisive next step.")


if __name__ == "__main__":
    main()
