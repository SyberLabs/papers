"""Agency test (advisor's sharpest hypothesis): does enforcing a persistent,
remembering first-person SPEAKER move vitality-proxies more than token-level
dynamical intervention (Lorenz chaos vs matched OU noise)?

If vitality is the reader's inference of a continuing intending mind, a
speaker-consistency SCAFFOLD (no dynamical modulation) should beat the dynamical
conditions on the surviving vitality proxy (global_range / drift, clean subset),
and produce LESS degeneracy. If the scaffold does nothing beyond plain sampling
while chaos/matched also do nothing, the object isn't reachable here either.

Conditions (all base model, raw-prompt continuation, eps=2.0 where modulated):
  plain        : seed prompt, sampling, no scaffold, no modulation
  agency       : scaffold-prefixed seed, sampling, no modulation
  logit_chaos  : seed prompt, Lorenz logit modulation
  logit_matched: seed prompt, matched-OU logit modulation

Proxy outcome (NOT human truth): global_range & drift (clean subset), and the
degeneracy rate per condition. Honest framing: a proxy test that sets up the
human study, not a substitute for it.
"""
import json, os, re, sys
from datetime import datetime, timezone

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vitality.generation.model import load_model
from vitality.generation.harness import SamplingParams, generate
from vitality.generation.conditions import build_conditions
from vitality.metrics.recurrence import compute_recurrence
from vitality.metrics.surface import tokenize_words
from sentence_transformers import SentenceTransformer

SCAFFOLD = ("I am one person, and I remember as I go. Everything I say stays "
            "tied to a single life — mine — carried forward by one voice that "
            "keeps its own thread. ")

TEMPLATE_PAT = re.compile(
    r"(\b[A-D]\)\s|\bAnswer:|\bA\.\s.*\bB\.\s|\\\(|\\\[|\\boxed|def |Input\b|"
    r"lexicographically|Creative Commons|licensed under)", re.I)


def degen(text):
    w = tokenize_words(text)
    g4 = [tuple(w[i:i+4]) for i in range(len(w)-3)]
    rep4 = 1 - len(set(g4))/len(g4) if g4 else 0
    return 1 if (len(TEMPLATE_PAT.findall(text)) >= 2 or rep4 > 0.15) else 0


def main():
    cfg = yaml.safe_load(open("configs/base.yaml"))
    prompts = yaml.safe_load(open("prompts/prompts_base.yaml"))
    seeds = cfg["seeds"]
    model, tok = load_model("Qwen/Qwen2.5-0.5B", "cpu")
    emb = SentenceTransformer("all-MiniLM-L6-v2")
    m = cfg["modulation"]

    out_dir = os.path.join("outputs",
                           "agency_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "agency.jsonl")
    recs = []

    for seed in seeds:
        sp = SamplingParams(temperature=cfg["sampling"]["temperature"],
                            top_p=cfg["sampling"]["top_p"], max_new_tokens=400, seed=seed)
        conds = {c.name: c for c in build_conditions(
            sp, eps=m["eps"], chaotic_map=m["chaotic_map"], mask_mode=m["mask_mode"],
            mask_topk=m["mask_topk"], warmup=m["warmup"], seed=seed)}
        for p in prompts:
            jobs = [
                ("plain", p["text"], None),
                ("agency", SCAFFOLD + p["text"], None),
                ("logit_chaos", p["text"], conds["logit_chaos"].make_modulator()),
                ("logit_matched", p["text"], conds["logit_matched"].make_modulator()),
            ]
            for name, prompt_text, mod in jobs:
                r = generate(model, tok, prompt_text, sp, name, modulator=mod,
                             use_chat_template=False)
                rec = compute_recurrence(r.text, emb)
                row = {"condition": name, "prompt_id": p["id"], "seed": seed,
                       "n_tokens": r.meta["n_tokens"], "text": r.text,
                       "drift": rec.drift, "global_range": rec.global_range,
                       "recurrence_rate": rec.recurrence_rate,
                       "degen": degen(r.text)}
                recs.append(row)
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"  [{name:14s}] {p['id']:14s} s{seed} "
                      f"ntok={r.meta['n_tokens']:4d} gr={rec.global_range:.3f} "
                      f"drift={rec.drift:.3f} degen={row['degen']}", flush=True)

    # summary
    print("\n=== proxy summary (clean only for trajectory; degen rate over all) ===")
    by = {}
    for r in recs:
        by.setdefault(r["condition"], []).append(r)
    print(f"  {'condition':14s} {'degen_rate':>10} {'gr_clean':>9} {'drift_clean':>11}")
    for c, rows in by.items():
        dr = np.mean([r["degen"] for r in rows])
        cln = [r for r in rows if r["degen"] == 0]
        gr = np.mean([r["global_range"] for r in cln]) if cln else float("nan")
        df = np.mean([r["drift"] for r in cln]) if cln else float("nan")
        print(f"  {c:14s} {dr:10.2f} {gr:9.3f} {df:11.3f}  (n_clean={len(cln)})")
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
