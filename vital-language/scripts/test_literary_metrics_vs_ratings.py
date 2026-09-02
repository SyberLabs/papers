"""Bridge test: do the LITERATURE-derived metrics predict felt vitality in MODEL
output, where semantic-travel/width/recurrence all failed the clean-subset test?

Metrics tested vs the 33 existing ratings (ALL and CLEAN subset):
  - F3 perception<->abstraction oscillation (soc_features.pa_oscillation_std)
  - imagistic drift / range (CLIP-text trajectory)
  - semantic x imagistic relationship: img-sem drift xcorr, and sem/img drift ratio
Reference clean-subset correlations to beat: global_range +0.51, width -0.135.
"""
import glob, hashlib, json, os, re, sys
import numpy as np
import torch, open_clip
from sentence_transformers import SentenceTransformer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vitality.metrics.soc_features import compute_soc
from vitality.metrics.surface import tokenize_words

TP = re.compile(r"(\b[A-D]\)\s|\bAnswer:|\\\(|\\\[|\\boxed|def |Input\b|"
                r"lexicographically|Creative Commons|licensed under)", re.I)


def degen(t):
    w = tokenize_words(t); g4 = [tuple(w[i:i+4]) for i in range(len(w)-3)]
    rep4 = 1 - len(set(g4))/len(g4) if g4 else 0
    return 1 if (len(TP.findall(t)) >= 2 or rep4 > 0.15) else 0


def uid_of(run, pid, seed, cond):
    return "p_" + hashlib.sha1(f"{run}|{pid}|{seed}|{cond}".encode()).hexdigest()[:10]


key = json.load(open("study/key.json", encoding="utf-8"))
rt = json.load(open("study/ratings_claude.json", encoding="utf-8"))["ratings"]
tx = {}
for f in glob.glob("outputs/longrun_*/longrun.jsonl"):
    run = os.path.basename(os.path.dirname(f))
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        tx[uid_of(run, r["prompt_id"], r["seed"], r["condition"])] = r["text"]

sem = SentenceTransformer("all-MiniLM-L6-v2")
soc_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_tok = open_clip.get_tokenizer("ViT-B-32"); clip_model.eval()


def sents(t, mn=3):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if len(s.split()) >= mn]


def clip_traj(ss):
    with torch.no_grad():
        f = clip_model.encode_text(clip_tok(ss)).float()
        f = f / f.norm(dim=-1, keepdim=True)
    E = f.numpy()
    consec = np.array([1 - float(E[i] @ E[i+1]) for i in range(len(E)-1)])
    cen = E.mean(0); cen /= np.linalg.norm(cen)+1e-9
    return consec, float((1-E@cen).mean())


def feats(text):
    ss = sents(text)[:120]
    if len(ss) < 8:
        return None
    soc = compute_soc(text, soc_model)
    Es = np.asarray(sem.encode(ss, normalize_embeddings=True, show_progress_bar=False))
    sc = np.array([1-float(Es[i]@Es[i+1]) for i in range(len(Es)-1)])
    ic, irange = clip_traj(ss)
    m = min(len(sc), len(ic))
    xcorr = float(np.corrcoef(sc[:m], ic[:m])[0, 1]) if m > 3 else np.nan
    return {
        "pa_osc": soc.pa_oscillation_std,
        "img_drift": float(ic.mean()), "img_range": irange,
        "img_sem_xcorr": xcorr,
        "sem_img_ratio": float(sc.mean()) / (float(ic.mean())+1e-9),
    }


rows = []
for uid, rr in rt.items():
    if uid not in tx:
        continue
    fv = feats(tx[uid])
    if fv is None:
        continue
    fv["vit"] = rr["scores"]["vitality"]; fv["degen"] = degen(tx[uid])
    rows.append(fv)

V = np.array([r["vit"] for r in rows]); D = np.array([r["degen"] for r in rows])
clean = D == 0
fields = ["pa_osc", "img_drift", "img_range", "img_sem_xcorr", "sem_img_ratio"]


def c(key_, mask=None):
    a = np.array([r[key_] for r in rows], float); v = V
    if mask is not None:
        a, v = a[mask], V[mask]
    ok = np.isfinite(a) & np.isfinite(v)
    if ok.sum() < 6 or a[ok].std() == 0:
        return float("nan")
    return round(float(np.corrcoef(a[ok], v[ok])[0, 1]), 3)


print(f"n={len(rows)}, clean={int(clean.sum())}")
print("ref to beat (clean): global_range +0.51, width -0.135\n")
print(f"  {'literary metric':18s} {'ALL':>7} {'CLEAN':>7}")
for k in fields:
    print(f"  {k:18s} {c(k):+7.3f} {c(k, clean):+7.3f}")
