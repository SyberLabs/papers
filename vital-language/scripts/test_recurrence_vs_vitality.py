"""Go/no-go for the recurrence hypothesis: do drift-with-return metrics track
felt vitality better than surprisal-width did (-0.51)?

Computes recurrence metrics on every rated passage (need its text -> reload runs)
and correlates each against vitality ratings.
"""
import glob, hashlib, json, os, sys
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vitality.metrics.recurrence import compute_recurrence

key = json.load(open("study/key.json", encoding="utf-8"))
rt = json.load(open("study/ratings_claude.json", encoding="utf-8"))["ratings"]


def uid_of(run, pid, seed, cond):
    h = hashlib.sha1(f"{run}|{pid}|{seed}|{cond}".encode()).hexdigest()[:10]
    return "p_" + h


# recover passage text by uid
text_by_uid = {}
for f in glob.glob("outputs/longrun_*/longrun.jsonl"):
    run = os.path.basename(os.path.dirname(f))
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        text_by_uid[uid_of(run, r["prompt_id"], r["seed"], r["condition"])] = r["text"]

print("loading sentence embedder ...", flush=True)
model = SentenceTransformer("all-MiniLM-L6-v2")

fields = ["drift", "global_range", "recurrence_rate", "return_distance",
          "drift_return_ratio", "verbatim_loop"]
data = {k: [] for k in fields}
vit = []
for uid, rr in rt.items():
    if uid not in text_by_uid:
        continue
    m = compute_recurrence(text_by_uid[uid], model)
    if not np.isfinite(m.drift):
        continue
    for k in fields:
        data[k].append(getattr(m, k))
    vit.append(rr["scores"]["vitality"])

vit = np.array(vit)


def c(a):
    a = np.array(a)
    m = np.isfinite(a) & np.isfinite(vit)
    if m.sum() < 5 or a[m].std() == 0:
        return float("nan")
    return round(float(np.corrcoef(a[m], vit[m])[0, 1]), 3)


print(f"\nn={len(vit)} passages. Correlation of each recurrence metric vs vitality:")
print("  (reference: surprisal-width vs vitality = -0.51)\n")
for k in fields:
    print(f"  {k:20s} {c(data[k]):+.3f}")
