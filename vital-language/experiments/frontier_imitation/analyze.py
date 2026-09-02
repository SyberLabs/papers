"""Analyze frontier imitation vs control vs real-author on all channels.

Reads experiments/frontier_imitation/<model>_<author>_<condition>.txt and the
real-author corpus, runs the SAME metric pipeline, and prints the key contrast:
does imitation move toward the author (vs its control) on INTRINSIC metrics, and
does it under-reproduce the RELATIONAL ones?

Run incrementally as you add files:
  python experiments/frontier_imitation/analyze.py
"""
import glob, os, re, sys
import numpy as np
import torch, open_clip
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from vitality.metrics.soc_features import compute_soc
from vitality.metrics.phonetic import compute_phonetic
from vitality.metrics.surface import split_sentences, tokenize_words

HERE = os.path.dirname(os.path.abspath(__file__))

sem = SentenceTransformer("all-MiniLM-L6-v2")
soc_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_tok = open_clip.get_tokenizer("ViT-B-32"); clip_model.eval()


def strip_g(t):
    m = re.search(r"\*\*\* ?START OF.*?\*\*\*", t, re.S); t = t[m.end():] if m else t
    m = re.search(r"\*\*\* ?END OF", t, re.S); return t[:m.start()] if m else t


def sents(t, mn=3):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if len(s.split()) >= mn]


def all_metrics(text, english=True):
    ss = sents(text)[:150]
    if len(ss) < 10:
        return None
    Es = np.asarray(sem.encode(ss, normalize_embeddings=True, show_progress_bar=False))
    sc = np.array([1 - float(Es[i] @ Es[i+1]) for i in range(len(Es)-1)])
    cen = Es.mean(0); cen /= np.linalg.norm(cen)+1e-9
    with torch.no_grad():
        f = clip_model.encode_text(clip_tok(ss)).float(); f = f/f.norm(dim=-1, keepdim=True)
    Ei = f.numpy()
    ic = np.array([1 - float(Ei[i] @ Ei[i+1]) for i in range(len(Ei)-1)])
    icen = Ei.mean(0); icen /= np.linalg.norm(icen)+1e-9
    soc = compute_soc(text, soc_model)
    slen = [len(tokenize_words(s)) for s in split_sentences(text) if s.strip()]
    slcv = float(np.std(slen)/np.mean(slen)) if slen and np.mean(slen) else 0.0
    ph = compute_phonetic(text).phonetic_flow if english else float("nan")
    return {
        "sem_drift": round(float(sc.mean()), 3),
        "sem_range": round(float((1-Es@cen).mean()), 3),
        "img_drift": round(float(ic.mean()), 3),
        "img_range": round(float((1-Ei@icen).mean()), 3),
        "pa_osc": soc.pa_oscillation_std,
        "sent_cv": round(slcv, 3),
        "phon_flow": round(ph, 3) if ph == ph else float("nan"),
    }


# real-author ground truth (windowed)
def win(p, frac=0.45, n=9000, g=True):
    t = open(p, encoding="utf-8", errors="ignore").read()
    if g: t = strip_g(t)
    s = int(len(t)*frac); return t[s:s+n]


liz = "\n".join(open(os.path.join(ROOT, "agua_vida.md"), encoding="utf-8").read().split("\n")[28:])[:9000]
real = {
    "REAL Woolf": (win(os.path.join(ROOT, "corpus/woolf_voyage.txt")), True),
    "REAL Joyce": (win(os.path.join(ROOT, "corpus/ulysses.txt")), True),
    "REAL Lispector": (liz, False),
    "REAL Austen": (win(os.path.join(ROOT, "corpus/austen_pride.txt")), True),
}

cols = ["sem_drift", "sem_range", "img_drift", "img_range", "pa_osc", "sent_cv", "phon_flow"]
print(f"{'source':32s} " + " ".join(f"{c:>9}" for c in cols))
print("-" * 105)
table = {}
for label, (txt, eng) in real.items():
    m = all_metrics(txt, eng)
    if m:
        table[label] = m
        print(f"{label:32s} " + " ".join(f"{m[c]:9.3f}" for c in cols))

# frontier outputs
files = sorted(glob.glob(os.path.join(HERE, "*_*_*.txt")))
if files:
    print()
    for f in files:
        name = os.path.basename(f)[:-4]
        eng = "lispector" not in name.lower() or True  # imitations are English here
        m = all_metrics(open(f, encoding="utf-8").read(), english=eng)
        if m:
            table[name] = m
            print(f"{name:32s} " + " ".join(f"{m[c]:9.3f}" for c in cols))
else:
    print("\n(no frontier output files yet — drop <model>_<author>_<condition>.txt here)")

# key contrast: imitation moved toward author (vs control)?
print("\n--- KEY CONTRAST (per author, per model): does imitation move toward real? ---")
for f in files:
    n = os.path.basename(f)[:-4]
    if "imitation" not in n:
        continue
    base = n.replace("imitation", "")
    ctrl = base + "control"
    parts = n.split("_")
    author = parts[1] if len(parts) > 1 else "?"
    realkey = next((k for k in table if k.startswith("REAL") and author.lower() in k.lower()), None)
    if ctrl in table and n in table and realkey:
        for metric in ["pa_osc", "img_drift", "img_range", "sent_cv", "phon_flow"]:
            imi, con, rl = table[n][metric], table[ctrl][metric], table[realkey][metric]
            if not (np.isfinite(imi) and np.isfinite(con) and np.isfinite(rl)):
                continue
            moved = imi - con
            toward = "TOWARD" if (rl > con and moved > 0) or (rl < con and moved < 0) else "away"
            print(f"  {n:30s} {metric:9s}: ctrl={con:.3f} imi={imi:.3f} real={rl:.3f}  [{toward}]")
