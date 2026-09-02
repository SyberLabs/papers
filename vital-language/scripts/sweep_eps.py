"""Week 2: epsilon sweep.

Goal: find the largest modulation strength that still preserves legibility, and
test whether chaotic modulation separates from matched-autocorrelation noise
*before* coherence collapses.

For each eps in the grid we generate the three modulated conditions
(logit_chaos, logit_matched, logit_white). The eps-invariant references
(baseline, sampling, prompt_only) are generated once at eps=0 and reused. Each
cell records structural metrics + the coherence guardrail (base-model
self-perplexity, n-gram repetition).

Usage:
  python scripts/sweep_eps.py --config configs/default.yaml
  python scripts/sweep_eps.py --config configs/default.yaml --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vitality.generation.model import load_model
from vitality.generation.harness import SamplingParams, generate
from vitality.generation.conditions import build_conditions
from vitality.metrics.surface import compute_surface, tokenize_words
from vitality.metrics.mfdfa import mfdfa
from vitality.metrics.coherence import compute_coherence

MODULATED = {"logit_chaos", "logit_matched", "logit_white"}
REFERENCES = {"baseline", "sampling", "prompt_only"}


def evaluate(model, tok, res, device):
    words = tokenize_words(res.text)
    surf = compute_surface(res.text, res.surprisal_base)
    mf = mfdfa(res.surprisal_base)
    coh = compute_coherence(model, tok, res.text, words, device)
    return {
        "text": res.text,
        "n_tokens": res.meta["n_tokens"],
        "surface": asdict(surf),
        "mfdfa": {"width": mf.width, "hurst": mf.hurst, "fit_r2": mf.fit_r2, "ok": mf.ok},
        "coherence": asdict(coh),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--eps-grid", default="0.0,0.5,1.0,1.5,2.0,3.0,4.0")
    ap.add_argument("--base", action="store_true",
                    help="base (non-instruct) model: raw-prompt, no prompt_only")
    ap.add_argument("--model", default=None, help="override model name (HPC/GPU)")
    ap.add_argument("--device", default=None, help="override device (e.g. cuda)")
    ap.add_argument("--quick", action="store_true", help="2 prompts, 1 seed, fewer eps")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(cfg["prompts_file"]) as f:
        prompts = yaml.safe_load(f)

    use_chat = not args.base
    # base model can't follow "write multifractally", so only 'sampling' is a
    # meaningful eps-invariant reference (greedy baseline kept; prompt_only dropped)
    refs = REFERENCES if not args.base else {"baseline", "sampling"}

    eps_grid = [float(x) for x in args.eps_grid.split(",")]
    seeds = cfg["seeds"]
    if args.quick:
        prompts = prompts[:2]
        seeds = seeds[:1]
        eps_grid = [0.0, 1.0, 2.0, 4.0]

    model_name = args.model or cfg["model"]
    device = args.device or cfg["device"]
    print(f"[load] {model_name} on {device} ...", flush=True)
    model, tok = load_model(model_name, device)
    m = cfg["modulation"]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(cfg["out_dir"], f"sweep_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "sweep.jsonl")
    records = []

    def emit(rec):
        records.append(rec)
        with open(path, "a", encoding="utf-8") as f:  # incremental: survive crashes
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _log(rec)

    for seed in seeds:
        base_sampling = SamplingParams(
            temperature=cfg["sampling"]["temperature"],
            top_p=cfg["sampling"]["top_p"],
            top_k=cfg["sampling"]["top_k"],
            max_new_tokens=cfg["sampling"]["max_new_tokens"],
            greedy=cfg["sampling"]["greedy"],
            seed=seed,
        )

        for p in prompts:
            # --- eps-invariant references: generate once ---
            ref_conditions = build_conditions(
                base_sampling, eps=0.0, chaotic_map=m["chaotic_map"],
                mask_mode=m["mask_mode"], mask_topk=m["mask_topk"],
                every_k=m["every_k"], warmup=m["warmup"], seed=seed,
            )
            for cond in ref_conditions:
                if cond.name not in refs:
                    continue
                res = generate(model, tok, cond.prompt_prefix + p["text"],
                               cond.sampling, cond.name,
                               modulator=cond.make_modulator(),
                               use_chat_template=use_chat, device=device)
                emit({"run_id": run_id, "prompt_id": p["id"], "category": p["category"],
                      "condition": cond.name, "seed": seed, "eps": 0.0,
                      **evaluate(model, tok, res, device)})

            # --- modulated conditions across the eps grid ---
            for eps in eps_grid:
                conditions = build_conditions(
                    base_sampling, eps=eps, chaotic_map=m["chaotic_map"],
                    mask_mode=m["mask_mode"], mask_topk=m["mask_topk"],
                    every_k=m["every_k"], warmup=m["warmup"], seed=seed,
                )
                for cond in conditions:
                    if cond.name not in MODULATED:
                        continue
                    res = generate(model, tok, cond.prompt_prefix + p["text"],
                                   cond.sampling, cond.name,
                                   modulator=cond.make_modulator(),
                                   use_chat_template=use_chat, device=device)
                    emit({"run_id": run_id, "prompt_id": p["id"],
                          "category": p["category"], "condition": cond.name,
                          "seed": seed, "eps": eps,
                          **evaluate(model, tok, res, device)})

    print(f"\n[done] {len(records)} records -> {path}")
    print("Next: python scripts/analyze_sweep.py " + path)


def _log(rec):
    mf = rec["mfdfa"]
    print(
        f"  [{rec['condition']:14s}] eps={rec['eps']:.1f} {rec['prompt_id']:14s} "
        f"ppl={rec['coherence']['self_perplexity']:7.2f} "
        f"mfw={(mf['width'] if mf['ok'] else float('nan')):.3f} "
        f"sl_cv={rec['surface']['sent_len_cv']:.3f} "
        f"rep4={rec['coherence']['rep_4gram']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
