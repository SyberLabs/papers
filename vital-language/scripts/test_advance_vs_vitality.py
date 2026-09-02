"""Test advisor's candidate metrics vs vitality, ALL and CLEAN subset.

Clean = not flagged degenerate (Stage 0 template/quiz/loop detector). The
clean-subset correlation is the real test: a metric that only works by detecting
broken text is useless. Bench against the current best (semantic_volume +0.41
clean-ish, trajectory drift/range +0.56 raw).
"""
import glob, hashlib, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vitality.metrics.advance import compute_advance
from vitality.metrics.surface import tokenize_words
from sentence_transformers import SentenceTransformer

key = json.load(open("study/key.json", encoding="utf-8"))
rt = json.load(open("study/ratings_claude.json", encoding="utf-8"))["ratings"]

TEMPLATE_PAT = re.compile(
    r"(\b[A-D]\)\s|\bAnswer:|\bA\.\s.*\bB\.\s|multiple choice|"
    r"\\\(|\\\[|\\boxed|def |Input\b|lexicographically|"
    r"Creative Commons|licensed under|\bn ≥|integers?\b.*\bpositive)", re.I)


def degen(text):
    w = tokenize_words(text)
    g4 = [tuple(w[i:i+4]) for i in range(len(w)-3)]
    rep4 = 1 - len(set(g4))/len(g4) if g4 else 0
    return 1 if (len(TEMPLATE_PAT.findall(text)) >= 2 or rep4 > 0.15) else 0


def uid_of(run, pid, seed, cond):
    h = hashlib.sha1(f"{run}|{pid}|{seed}|{cond}".encode()).hexdigest()[:10]
    return "p_" + h


text_by_uid = {}
for f in glob.glob("outputs/longrun_*/longrun.jsonl"):
    run = os.path.basename(os.path.dirname(f))
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        text_by_uid[uid_of(run, r["prompt_id"], r["seed"], r["condition"])] = r["text"]

model = SentenceTransformer("all-MiniLM-L6-v2")

fields = ["coherent_advance_rate", "concreteness_density",
          "proper_numeral_density", "sem_lag1", "sem_decay"]
data = {k: [] for k in fields}
vit, dg = [], []
for uid, rr in rt.items():
    if uid not in text_by_uid:
        continue
    m = compute_advance(text_by_uid[uid], model)
    if not np.isfinite(m.coherent_advance_rate):
        continue
    for k in fields:
        data[k].append(getattr(m, k))
    vit.append(rr["scores"]["vitality"])
    dg.append(degen(text_by_uid[uid]))

vit = np.array(vit); dg = np.array(dg); clean = dg == 0


def c(a, mask=None):
    a = np.array(a, float); v = vit
    if mask is not None:
        a, v = a[mask], vit[mask]
    m = np.isfinite(a) & np.isfinite(v)
    if m.sum() < 6 or a[m].std() == 0:
        return float("nan")
    return round(float(np.corrcoef(a[m], v[m])[0, 1]), 3)


print(f"n={len(vit)}, clean={int(clean.sum())}")
print("references: drift +0.53 / global_range +0.56 (raw); width -0.135 (clean)\n")
print(f"  {'metric':24s} {'ALL':>7} {'CLEAN':>7}")
for k in fields:
    print(f"  {k:24s} {c(data[k]):+7.3f} {c(data[k], clean):+7.3f}")
