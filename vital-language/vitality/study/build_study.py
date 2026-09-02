"""Build a blind rating study from one or more run jsonl files.

Each study ITEM is a (prompt_id, seed) group shown as a set of anonymized
PASSAGES (one per condition), in randomized order. The rater never sees the
condition; we keep a separate KEY mapping passage_uid -> condition so ratings
can be de-anonymized at analysis time.

Why grouped/paired rather than one-passage-at-a-time: humans rate "which of
these is more alive" far more reliably than an absolute vitality score in a
vacuum. Showing matched-prompt outputs together turns vitality into a relative
judgment, which is what we can actually measure.

Output:
  study/manifest.json  -> items shown to raters (NO condition labels)
  study/key.json       -> passage_uid -> {condition, run, prompt_id, seed, metrics}

Usage:
  python -m vitality.study.build_study outputs/longrun_*/longrun.jsonl \
      --out study --min-tokens 200 --seed 0
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
from collections import defaultdict


def passage_uid(rec, run_tag):
    h = hashlib.sha1(
        f"{run_tag}|{rec['prompt_id']}|{rec['seed']}|{rec['condition']}".encode()
    ).hexdigest()[:10]
    return f"p_{h}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="+", help="run jsonl files / globs")
    ap.add_argument("--out", default="study")
    ap.add_argument("--min-tokens", type=int, default=200,
                    help="skip too-short passages (incomplete generations)")
    ap.add_argument("--max-chars", type=int, default=1600,
                    help="truncate displayed text to keep rating tractable")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-items", type=int, default=0,
                    help="cap study length for rater fatigue (0 = all); "
                         "selection is balanced across categories")
    args = ap.parse_args()

    files = []
    for g in args.globs:
        files.extend(glob.glob(g))
    if not files:
        raise SystemExit("no files matched")

    rng = random.Random(args.seed)
    # group passages by (run, prompt_id, seed)
    groups = defaultdict(list)
    key = {}
    for f in sorted(files):
        run_tag = os.path.basename(os.path.dirname(f))
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["n_tokens"] < args.min_tokens:
                continue
            uid = passage_uid(r, run_tag)
            text = r["text"].strip()
            if len(text) > args.max_chars:
                # truncate at a sentence boundary near the cap
                cut = text.rfind(".", 0, args.max_chars)
                text = text[: cut + 1] if cut > args.max_chars // 2 else text[: args.max_chars]
            groups[(run_tag, r["prompt_id"], r["seed"])].append((uid, text, r))
            key[uid] = {
                "condition": r["condition"], "run": run_tag,
                "prompt_id": r["prompt_id"], "seed": r["seed"],
                "category": r["category"], "n_tokens": r["n_tokens"],
                "surprisal_width": r["mfdfa"]["surprisal"]["width"],
                "self_perplexity": r["coherence"]["self_perplexity"],
            }

    # build items: only groups with >= 2 conditions present (a real comparison)
    items = []
    for (run_tag, pid, seed), passages in groups.items():
        if len(passages) < 2:
            continue
        shown = [{"uid": uid, "text": text} for uid, text, _ in passages]
        rng.shuffle(shown)  # randomize order within the item
        items.append({
            "item_id": f"{run_tag}_{pid}_s{seed}",
            "category": passages[0][2]["category"],
            "passages": shown,
        })
    # optional cap: balanced across categories (round-robin so a short study
    # still spans all prompt types) before final shuffle
    if args.max_items and len(items) > args.max_items:
        by_cat = defaultdict(list)
        for it in items:
            by_cat[it["category"]].append(it)
        for v in by_cat.values():
            rng.shuffle(v)
        picked, cats = [], list(by_cat)
        while len(picked) < args.max_items and any(by_cat.values()):
            for c in cats:
                if by_cat[c]:
                    picked.append(by_cat[c].pop())
                    if len(picked) >= args.max_items:
                        break
        items = picked

    rng.shuffle(items)  # randomize item order across the study

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"items": items}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "key.json"), "w", encoding="utf-8") as f:
        json.dump(key, f, ensure_ascii=False, indent=2)

    n_pass = sum(len(it["passages"]) for it in items)
    print(f"[study] {len(items)} items, {n_pass} passages from {len(files)} run(s)")
    print(f"  -> {args.out}/manifest.json  (blind, for raters)")
    print(f"  -> {args.out}/key.json       (condition key, for analysis only)")


if __name__ == "__main__":
    main()
