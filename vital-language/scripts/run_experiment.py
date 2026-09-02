"""Run all (prompt x condition x seed) generations and save outputs + metrics.

Usage:
  python scripts/run_experiment.py --config configs/default.yaml
  python scripts/run_experiment.py --config configs/default.yaml --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vitality.generation.model import load_model
from vitality.generation.harness import SamplingParams, generate
from vitality.generation.conditions import build_conditions
from vitality.metrics.surface import compute_surface
from vitality.metrics.mfdfa import mfdfa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true", help="1 prompt, 1 seed, short")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(cfg["prompts_file"]) as f:
        prompts = yaml.safe_load(f)

    seeds = cfg["seeds"]
    smax = cfg["sampling"]["max_new_tokens"]
    if args.smoke:
        prompts = prompts[:1]
        seeds = seeds[:1]
        smax = 64

    print(f"[load] {cfg['model']} on {cfg['device']} ...", flush=True)
    model, tok = load_model(cfg["model"], cfg["device"])

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(cfg["out_dir"], "smoke" if args.smoke else run_id)
    os.makedirs(out_dir, exist_ok=True)
    records = []

    for seed in seeds:
        base_sampling = SamplingParams(
            temperature=cfg["sampling"]["temperature"],
            top_p=cfg["sampling"]["top_p"],
            top_k=cfg["sampling"]["top_k"],
            max_new_tokens=smax,
            greedy=cfg["sampling"]["greedy"],
            seed=seed,
        )
        m = cfg["modulation"]
        conditions = build_conditions(
            base_sampling,
            eps=m["eps"],
            chaotic_map=m["chaotic_map"],
            mask_mode=m["mask_mode"],
            mask_topk=m["mask_topk"],
            every_k=m["every_k"],
            warmup=m["warmup"],
            seed=seed,
        )

        for p in prompts:
            for cond in conditions:
                full_prompt = cond.prompt_prefix + p["text"]
                res = generate(
                    model, tok, full_prompt, cond.sampling, cond.name,
                    modulator=cond.make_modulator(), device=cfg["device"],
                )
                surf = compute_surface(res.text, res.surprisal_base)
                mf = mfdfa(res.surprisal_base)
                rec = {
                    "run_id": run_id,
                    "prompt_id": p["id"],
                    "category": p["category"],
                    "condition": cond.name,
                    "seed": seed,
                    "text": res.text,
                    "n_tokens": res.meta["n_tokens"],
                    "tok_per_s": res.meta["tok_per_s"],
                    "surface": asdict(surf),
                    "mfdfa": {
                        "width": mf.width, "hurst": mf.hurst,
                        "fit_r2": mf.fit_r2, "ok": mf.ok, "n": mf.n, "note": mf.note,
                    },
                }
                records.append(rec)
                print(
                    f"  [{cond.name:14s}] {p['id']:14s} seed={seed} "
                    f"n={res.meta['n_tokens']:3d} "
                    f"mfw={mf.width if mf.ok else float('nan'):.3f} "
                    f"sl_cv={surf.sent_len_cv:.3f} mattr={surf.mattr_50:.3f}",
                    flush=True,
                )

    with open(os.path.join(out_dir, "records.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[done] {len(records)} records -> {out_dir}/records.jsonl")


if __name__ == "__main__":
    main()
