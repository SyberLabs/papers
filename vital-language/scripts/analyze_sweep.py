"""Analyze an eps-sweep: structural gain vs coherence cost, and whether chaotic
modulation separates from matched-autocorrelation noise.

Outputs (next to the input jsonl):
  - sweep_summary.csv   : mean +/- sem per (condition, eps)
  - structure_vs_eps.png: MFDFA width, sentence-CV, surprisal burstiness vs eps,
                          with the self-perplexity coherence curve overlaid
  - chaos_vs_matched.png: per-eps delta(chaos - matched) with paired-bootstrap CIs

Prints:
  - the coherence cliff (eps where PPL exceeds tolerance x sampling baseline)
  - the recommended operating eps
  - whether chaos beats matched noise on structure within the legible band

Usage: python scripts/analyze_sweep.py outputs/sweep_XXXX/sweep.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STRUCT_KEYS = {
    "mfdfa_width": ("mfdfa", "width"),
    "sent_len_cv": ("surface", "sent_len_cv"),
    "surprisal_burstiness": ("surface", "surprisal_burstiness"),
    "mattr_50": ("surface", "mattr_50"),
}
COH_PPL = ("coherence", "self_perplexity")


def load(path: str) -> pd.DataFrame:
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        flat = {
            "condition": r["condition"], "eps": r["eps"], "seed": r["seed"],
            "prompt_id": r["prompt_id"], "n_tokens": r["n_tokens"],
            "self_perplexity": r["coherence"]["self_perplexity"],
            "rep_4gram": r["coherence"]["rep_4gram"],
            "mfdfa_ok": r["mfdfa"]["ok"],
        }
        for name, (grp, key) in {**STRUCT_KEYS}.items():
            flat[name] = r[grp][key]
        rows.append(flat)
    return pd.DataFrame(rows)


def sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--ppl-tol", type=float, default=1.5,
                    help="legible if PPL <= tol * sampling-baseline PPL")
    args = ap.parse_args()
    out_dir = os.path.dirname(os.path.abspath(args.path))
    df = load(args.path)

    metrics = list(STRUCT_KEYS) + ["self_perplexity", "rep_4gram"]
    agg = (
        df.groupby(["condition", "eps"])[metrics]
        .agg(["mean", sem])
        .reset_index()
    )
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
    agg.to_csv(os.path.join(out_dir, "sweep_summary.csv"), index=False)

    # baseline coherence reference = unmodulated sampling
    samp_ppl = df[df.condition == "sampling"]["self_perplexity"].mean()
    ppl_ceiling = args.ppl_tol * samp_ppl
    print(f"sampling baseline PPL = {samp_ppl:.2f}; legibility ceiling "
          f"(x{args.ppl_tol}) = {ppl_ceiling:.2f}\n")

    modulated = ["logit_chaos", "logit_matched", "logit_white"]
    eps_vals = sorted(df.eps.unique())

    # ---- coherence cliff per condition ----
    print("Coherence: largest eps with mean PPL <= ceiling (the legible band)")
    legible_eps = {}
    for c in modulated:
        sub = agg[agg.condition == c].sort_values("eps")
        ok = sub[sub.self_perplexity_mean <= ppl_ceiling]
        top = float(ok.eps.max()) if len(ok) else 0.0
        legible_eps[c] = top
        print(f"  {c:14s}: eps <= {top:.1f}")
    # The operating point is set by the INTERVENTION we intend to deploy (chaos)
    # and its matched control -- NOT by white noise, which is a control we expect
    # to fail. Dragging the operating eps down to white's cliff would hide the
    # whole usable band. The chaos-vs-matched test runs wherever BOTH are legible.
    operating = min(legible_eps.get("logit_chaos", 0.0),
                    legible_eps.get("logit_matched", 0.0))
    print(f"\n>>> white-noise cliff at eps<= {legible_eps.get('logit_white', 0.0):.1f} "
          f"(expected: unstructured noise fails fast)")
    print(f">>> operating eps for chaos-vs-matched test (both legible): {operating:.1f}\n")

    # ---- chaos vs matched within legible band (paired by prompt+seed+eps) ----
    print("Chaos - Matched structural delta (paired bootstrap 95% CI), legible eps:")
    for metric in ["mfdfa_width", "sent_len_cv", "surprisal_burstiness"]:
        for eps in [e for e in eps_vals if 0 < e <= operating]:
            a = df[(df.condition == "logit_chaos") & (df.eps == eps)]
            b = df[(df.condition == "logit_matched") & (df.eps == eps)]
            merged = a.merge(b, on=["prompt_id", "seed"], suffixes=("_c", "_m"))
            if len(merged) == 0:
                continue
            d = (merged[f"{metric}_c"] - merged[f"{metric}_m"]).to_numpy(float)
            d = d[np.isfinite(d)]
            if len(d) < 2:
                continue
            rng = np.random.default_rng(0)
            boot = [rng.choice(d, len(d), replace=True).mean() for _ in range(2000)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            star = "*" if (lo > 0 or hi < 0) else " "
            print(f"  {metric:22s} eps={eps:.1f}: "
                  f"{d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] {star}")
    print("  (* = CI excludes 0: chaos differs from matched noise)\n")

    _plot_structure(agg, modulated, ppl_ceiling, out_dir)
    _plot_chaos_vs_matched(df, eps_vals, operating, out_dir)
    print(f"plots + summary -> {out_dir}")


def _plot_structure(agg, modulated, ppl_ceiling, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = ["mfdfa_width", "sent_len_cv", "surprisal_burstiness", "self_perplexity"]
    colors = {"logit_chaos": "C3", "logit_matched": "C0", "logit_white": "C7"}
    for ax, metric in zip(axes.flat, panels):
        for c in modulated:
            sub = agg[agg.condition == c].sort_values("eps")
            ax.errorbar(sub.eps, sub[f"{metric}_mean"], yerr=sub[f"{metric}_sem"],
                        marker="o", capsize=3, label=c, color=colors[c])
        ax.set_xlabel("epsilon"); ax.set_ylabel(metric); ax.set_title(metric)
        if metric == "self_perplexity":
            ax.axhline(ppl_ceiling, ls="--", color="k", label="legibility ceiling")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "structure_vs_eps.png"), dpi=120)
    plt.close(fig)


def _plot_chaos_vs_matched(df, eps_vals, operating, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    xs, ys, los, his = [], [], [], []
    for eps in [e for e in eps_vals if e > 0]:
        a = df[(df.condition == "logit_chaos") & (df.eps == eps)]
        b = df[(df.condition == "logit_matched") & (df.eps == eps)]
        merged = a.merge(b, on=["prompt_id", "seed"])
        d = (merged["mfdfa_width_x"] - merged["mfdfa_width_y"]).to_numpy(float)
        d = d[np.isfinite(d)]
        if len(d) < 2:
            continue
        rng = np.random.default_rng(0)
        boot = [rng.choice(d, len(d), replace=True).mean() for _ in range(2000)]
        xs.append(eps); ys.append(d.mean())
        los.append(np.percentile(boot, 2.5)); his.append(np.percentile(boot, 97.5))
    if xs:
        ys = np.array(ys)
        ax.errorbar(xs, ys, yerr=[ys - np.array(los), np.array(his) - ys],
                    marker="o", capsize=4, color="C3")
        ax.axhline(0, ls="-", color="k", lw=0.8)
        ax.axvline(operating, ls="--", color="green", label=f"operating eps={operating}")
        ax.set_xlabel("epsilon"); ax.set_ylabel("MFDFA width: chaos - matched")
        ax.set_title("Does chaos add multifractality beyond matched noise?")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "chaos_vs_matched.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
