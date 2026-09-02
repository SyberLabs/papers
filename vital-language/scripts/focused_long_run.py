"""Focused long-passage run: does chaotic modulation reach the 'Joyce corner'
(wide sentence-length multifractal spectrum) more than matched-autocorrelation
noise -- at an eps that keeps prose legible?

Fixed eps (the operating point), long passages (~1500 tok -> enough sentences
for valid sentence-length MFDFA), four conditions:
    sampling, prompt_only, logit_chaos, logit_matched.

For each generation we compute MFDFA on BOTH signals (surprisal, sentence_length)
and, for sentence_length, the shuffle-surrogate width -- the same credibility
check that validated the literary benchmark. A real move toward the Joyce corner
should show width that SURVIVES shuffling poorly (i.e. width >> shuffled width).

Usage:
  python scripts/focused_long_run.py --config configs/default.yaml --eps 2.0 \
      --max-new 1500 --seeds 0,1,2
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
from vitality.metrics.signals_from_text import (
    surprisal_series, sentence_length_series,
)

KEEP_INSTRUCT = {"sampling", "prompt_only", "logit_chaos", "logit_matched"}
# Base models can't follow a "write multifractally" instruction, so prompt_only
# is dropped; the continuation seeds in prompts_base.yaml carry the prompt.
KEEP_BASE = {"sampling", "logit_chaos", "logit_matched"}
BASE_MODEL = "Qwen/Qwen2.5-0.5B"


def mfdfa_block(text, surprisal):
    """Per-passage MFDFA. Sentence-length at single-passage scale is borderline
    (~60-120 sentences), so the AUTHORITATIVE sentence-length estimate is the
    POOLED one computed in analysis from the saved series. We keep per-passage
    numbers for reference and persist the raw sentence-length series for pooling.
    """
    surp = surprisal_series(surprisal)
    sl = sentence_length_series(text)
    r_surp = mfdfa(surp)
    r_sl = mfdfa(sl, min_len=80, n_scales=14)
    return {
        "surprisal": {"width": r_surp.width, "hurst": r_surp.hurst,
                      "fit_r2": r_surp.fit_r2, "ok": r_surp.ok, "n": r_surp.n},
        "sentence_length": {"width": r_sl.width, "hurst": r_sl.hurst,
                            "fit_r2": r_sl.fit_r2, "ok": r_sl.ok, "n": r_sl.n},
        # raw series -> pooled MFDFA in analyze step (the trustworthy estimate)
        "sentence_length_series": [int(v) for v in sl],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--eps", type=float, default=2.0)
    ap.add_argument("--max-new", type=int, default=1500)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--base", action="store_true",
                    help="use base (non-instruct) model + raw-prompt continuation")
    ap.add_argument("--model", default=None,
                    help="override model name (e.g. Qwen/Qwen2.5-3B for HPC/GPU)")
    ap.add_argument("--device", default=None, help="override device (e.g. cuda)")
    ap.add_argument("--quick", action="store_true", help="1 prompt, 1 seed, 600 tok")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    keep = KEEP_BASE if args.base else KEEP_INSTRUCT
    use_chat = not args.base
    model_name = args.model or (BASE_MODEL if args.base else cfg["model"])
    prompts_path = "prompts/prompts_base.yaml" if args.base else cfg["prompts_file"]
    with open(prompts_path) as f:
        prompts = yaml.safe_load(f)

    seeds = [int(s) for s in args.seeds.split(",")]
    max_new = args.max_new
    if args.quick:
        prompts = prompts[:1]; seeds = [0]; max_new = 600

    print(f"[load] {model_name} ({'base' if args.base else 'instruct'}) | "
          f"eps={args.eps} max_new={max_new} seeds={seeds} "
          f"prompts={len(prompts)} conditions={sorted(keep)}", flush=True)
    device = args.device or cfg["device"]
    model, tok = load_model(model_name, device)
    m = cfg["modulation"]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(cfg["out_dir"], f"longrun_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "longrun.jsonl")
    records = []

    for seed in seeds:
        base_sampling = SamplingParams(
            temperature=cfg["sampling"]["temperature"], top_p=cfg["sampling"]["top_p"],
            top_k=cfg["sampling"]["top_k"], max_new_tokens=max_new, seed=seed,
        )
        conditions = build_conditions(
            base_sampling, eps=args.eps, chaotic_map=m["chaotic_map"],
            mask_mode=m["mask_mode"], mask_topk=m["mask_topk"],
            every_k=m["every_k"], warmup=m["warmup"], seed=seed,
        )
        for p in prompts:
            for cond in conditions:
                if cond.name not in keep:
                    continue
                res = generate(model, tok, cond.prompt_prefix + p["text"],
                               cond.sampling, cond.name,
                               modulator=cond.make_modulator(),
                               use_chat_template=use_chat, device=device)
                words = tokenize_words(res.text)
                surf = compute_surface(res.text, res.surprisal_base)
                coh = compute_coherence(model, tok, res.text, words, device)
                mf = mfdfa_block(res.text, res.surprisal_base)
                rec = {"run_id": run_id, "prompt_id": p["id"],
                       "category": p["category"], "condition": cond.name,
                       "seed": seed, "eps": args.eps, "text": res.text,
                       "n_tokens": res.meta["n_tokens"],
                       "surface": asdict(surf), "coherence": asdict(coh),
                       "mfdfa": mf}
                records.append(rec)
                # write incrementally so a late crash never loses the whole run
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                slb = mf["sentence_length"]
                slw = slb["width"] if slb["ok"] else float("nan")
                print(f"  [{cond.name:14s}] {p['id']:14s} s{seed} "
                      f"ntok={res.meta['n_tokens']:4d} nsent={slb['n']:3d} "
                      f"ppl={coh.self_perplexity:6.2f} "
                      f"SLw={slw:.2f} surpW={mf['surprisal']['width']:.2f}",
                      flush=True)

    print(f"\n[done] {len(records)} records -> {path}")


if __name__ == "__main__":
    main()
