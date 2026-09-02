"""Probe: which generation statistics track (my) felt vitality?

Tests GPT's "multifractality != entropy" claim by putting surprisal WIDTH
(noisy per-passage MFDFA), surprisal MEAN (entropy proxy), surprisal STD
(burstiness), and perplexity side by side against vitality ratings.
"""
import glob, hashlib, json, os
import numpy as np

key = json.load(open("study/key.json", encoding="utf-8"))
rt = json.load(open("study/ratings_claude.json", encoding="utf-8"))["ratings"]


def uid_of(run, pid, seed, cond):
    h = hashlib.sha1(f"{run}|{pid}|{seed}|{cond}".encode()).hexdigest()[:10]
    return "p_" + h


sm = {}  # uid -> (mean_surprisal, std_surprisal)
for f in glob.glob("outputs/longrun_*/longrun.jsonl"):
    run = os.path.basename(os.path.dirname(f))
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        u = uid_of(run, r["prompt_id"], r["seed"], r["condition"])
        sm[u] = (r["surface"]["surprisal_mean"], r["surface"]["surprisal_std"])

rows = []
for u, rr in rt.items():
    if u not in key or u not in sm:
        continue
    w = key[u]["surprisal_width"]
    if not np.isfinite(w):
        continue
    mean_s, std_s = sm[u]
    rows.append((w, mean_s, std_s, key[u]["self_perplexity"], rr["scores"]["vitality"]))

W, MS, SS, P, V = map(np.array, zip(*rows))


def c(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return round(float(np.corrcoef(a[m], b[m])[0, 1]), 3)


print(f"n={len(W)}")
print("vitality vs surprisal WIDTH (multifractal-ish):", c(W, V))
print("vitality vs surprisal MEAN  (entropy proxy)   :", c(MS, V))
print("vitality vs surprisal STD   (burstiness)      :", c(SS, V))
print("vitality vs perplexity                        :", c(P, V))
print("--- are width and entropy the same axis? ---")
print("width vs mean-surprisal:", c(W, MS))
print("width vs std-surprisal :", c(W, SS))
