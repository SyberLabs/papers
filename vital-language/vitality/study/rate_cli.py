"""CLI rater: same study, same output format as study/rate.html.

Two modes:
  interactive : prompts for each dimension score + 'most alive' pick per item.
  --submit FILE: ingest a JSON of pre-decided ratings (for non-TTY raters, e.g.
                 an LLM rater) and write it out in the canonical ratings format.

Output format (identical to rate.html, consumed by scripts/analyze_study.py):
  {"rater": "<id>", "ratings": {uid: {"scores": {dim: 0-100, ...},
                                       "alive": bool}}}

The --submit FILE format is a convenience superset:
  {"<item_id>": {"<uid>": {"vitality":N, ..., "alive":true|false}}, ...}
or directly the canonical {uid: {scores, alive}} form. Either is accepted.
"""

from __future__ import annotations

import argparse
import json
import sys

DIMS = ["vitality", "originality", "coherence", "reread", "interiority",
        "resonance", "stimulation", "artificiality"]
DIM_HELP = {
    "vitality": "does the language feel alive vs inert",
    "originality": "non-generic",
    "coherence": "legibility / holds together",
    "reread": "desire to reread",
    "interiority": "felt inner life",
    "resonance": "emotional resonance",
    "stimulation": "cognitive stimulation",
    "artificiality": "feels machine-made (HIGH = more artificial)",
}


def load(path):
    return json.load(open(path, encoding="utf-8"))


def interactive(manifest, rater):
    ratings = {}
    items = manifest["items"]
    for i, it in enumerate(items):
        print(f"\n{'='*70}\nITEM {i+1}/{len(items)}  [{it['category']}]")
        for j, p in enumerate(it["passages"]):
            print(f"\n--- passage {j+1} ({p['uid']}) ---\n{p['text']}\n")
            scores = {}
            for d in DIMS:
                while True:
                    raw = input(f"  {d} (0-100, {DIM_HELP[d]}): ").strip()
                    try:
                        v = float(raw);
                        if 0 <= v <= 100:
                            scores[d] = v; break
                    except ValueError:
                        pass
                    print("   enter a number 0-100")
            ratings[p["uid"]] = {"scores": scores, "alive": False}
        # most-alive pick
        labels = {str(j+1): p["uid"] for j, p in enumerate(it["passages"])}
        while True:
            pick = input(f"  MOST ALIVE passage # (1-{len(it['passages'])}): ").strip()
            if pick in labels:
                ratings[labels[pick]]["alive"] = True; break
    return ratings


def from_submission(manifest, sub):
    """Accept three shapes:
      (a) canonical   {uid: {"scores": {...}, "alive": bool}}
      (b) flat-by-uid {uid: {"vitality": N, ..., "alive": bool}}
      (c) nested      {item_id: {uid: {"vitality": N, ...}}}
    """
    valid_uids = {p["uid"] for it in manifest["items"] for p in it["passages"]}

    # (a) canonical
    if all(isinstance(v, dict) and "scores" in v for v in sub.values()):
        return sub

    # (b) flat-by-uid: top-level keys are uids, values carry dimension scores
    if all(k in valid_uids for k in sub):
        ratings = {}
        for uid, d in sub.items():
            ratings[uid] = {"scores": {k: float(d[k]) for k in DIMS if k in d},
                            "alive": bool(d.get("alive", False))}
        return ratings

    # (c) nested-by-item
    ratings = {}
    for _item, passages in sub.items():
        for uid, d in passages.items():
            if uid not in valid_uids:
                sys.exit(f"submission has unknown uid {uid}")
            ratings[uid] = {"scores": {k: float(d[k]) for k in DIMS if k in d},
                            "alive": bool(d.get("alive", False))}
    return ratings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--rater", required=True)
    ap.add_argument("--submit", help="JSON file of pre-decided ratings (non-TTY)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    manifest = load(args.manifest)
    if args.submit:
        ratings = from_submission(manifest, load(args.submit))
    else:
        ratings = interactive(manifest, args.rater)

    out = args.out or f"study/ratings_{args.rater}.json"
    json.dump({"rater": args.rater, "ratings": ratings},
              open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    rated = len(ratings)
    alive = sum(1 for r in ratings.values() if r.get("alive"))
    print(f"\n[rated] {rated} passages, {alive} 'most alive' picks -> {out}")


if __name__ == "__main__":
    main()
