"""Analyze frontier imitations using the ACTUAL filenames in this folder:
  <author>.md            = imitation
  <author>_control.md    = matched control
vs real-author ground truth. Prints the distance-traveled contrast:
  imitation - control   (did the model move toward the author?)
  vs  real_author - Austen   (where the tradition actually sits)
"""
import os, re, sys
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
    s = [x.strip() for x in re.split(r"(?<=[.!?])\s+", t) if len(x.split()) >= mn]
    # Fallback for long-flow prose (Joyce run-ons): if too few sentence units,
    # segment on clause boundaries so trajectory metrics have enough points.
    if len(s) < 12:
        s = [x.strip() for x in re.split(r"[.!?,;:—]+\s+|\s+and\s+", t)
             if len(x.split()) >= mn]
    return s


def metrics(text, english=True):
    ss = sents(text)[:150]
    if len(ss) < 10:
        return None
    Es = np.asarray(sem.encode(ss, normalize_embeddings=True, show_progress_bar=False))
    sc = np.array([1-float(Es[i]@Es[i+1]) for i in range(len(Es)-1)])
    cen = Es.mean(0); cen /= np.linalg.norm(cen)+1e-9
    with torch.no_grad():
        f = clip_model.encode_text(clip_tok(ss)).float(); f = f/f.norm(dim=-1, keepdim=True)
    Ei = f.numpy()
    ic = np.array([1-float(Ei[i]@Ei[i+1]) for i in range(len(Ei)-1)])
    icen = Ei.mean(0); icen /= np.linalg.norm(icen)+1e-9
    soc = compute_soc(text, soc_model)
    slen = [len(tokenize_words(s)) for s in split_sentences(text) if s.strip()]
    slcv = float(np.std(slen)/np.mean(slen)) if slen and np.mean(slen) else 0.0
    ph = compute_phonetic(text).phonetic_flow if english else float("nan")
    return {"sem_drift": round(float(sc.mean()),3), "sem_range": round(float((1-Es@cen).mean()),3),
            "img_drift": round(float(ic.mean()),3), "img_range": round(float((1-Ei@icen).mean()),3),
            "pa_osc": soc.pa_oscillation_std, "sent_cv": round(slcv,3),
            "phon_flow": round(ph,3) if ph==ph else float("nan")}


def win(p, frac=0.45, n=9000):
    t = strip_g(open(p, encoding="utf-8", errors="ignore").read()); s=int(len(t)*frac); return t[s:s+n]


liz = "\n".join(open(os.path.join(ROOT,"agua_vida.md"),encoding="utf-8").read().split("\n")[28:])[:9000]
real = {
    "woolf": (win(os.path.join(ROOT,"corpus/woolf_voyage.txt")), True),
    "joyce": (win(os.path.join(ROOT,"corpus/ulysses.txt")), True),
    "lispector": (liz, False),
}
austen = metrics(win(os.path.join(ROOT,"corpus/austen_pride.txt")), True)

cols = ["sem_drift","sem_range","img_drift","img_range","pa_osc","sent_cv","phon_flow"]
print(f"{'source':26s} " + " ".join(f"{c:>9}" for c in cols))
print("-"*100)
print(f"{'CONVENTIONAL: Austen':26s} " + " ".join(f"{austen[c]:9.3f}" for c in cols))
T = {"austen": austen}
for a,(txt,eng) in real.items():
    m = metrics(txt, eng); T[f"real_{a}"]=m
    print(f"{'REAL '+a:26s} " + " ".join(f"{m[c]:9.3f}" for c in cols))
print()
for a in ["woolf","joyce","lispector"]:
    for cond in ["control",""]:
        fn = os.path.join(HERE, f"{a}{'_'+cond if cond else ''}.md")
        if os.path.exists(fn):
            eng = (a != "lispector")
            m = metrics(open(fn,encoding="utf-8").read(), eng)
            lab = f"{a} {'CONTROL' if cond else 'IMITATION'}"
            if m is None:
                print(f"{lab:26s}  (too few splittable sentences: likely one long flow)")
                continue
            T[f"{a}_{cond or 'imit'}"]=m
            print(f"{lab:26s} " + " ".join(f"{m[c]:9.3f}" for c in cols))

print("\n=== DISTANCE TRAVELED: imitation-control  vs  real_author-Austen ===")
print("(same sign + imitation reaches toward real = model captured that signature)\n")
for a in ["woolf","joyce","lispector"]:
    imi, ctrl, rl = T.get(f"{a}_imit"), T.get(f"{a}_control"), T.get(f"real_{a}")
    if not (imi and ctrl and rl):
        continue
    print(f"-- {a.upper()} --")
    for c in cols:
        if not all(np.isfinite([imi[c],ctrl[c],rl[c],austen[c]])):
            print(f"   {c:10s}: n/a"); continue
        model_move = imi[c]-ctrl[c]
        trad_gap  = rl[c]-austen[c]
        agree = "captured" if (np.sign(model_move)==np.sign(trad_gap) and abs(model_move)>0.01) else ("flat" if abs(model_move)<=0.01 else "WRONG-DIR")
        print(f"   {c:10s}: model_move={model_move:+.3f}  trad_gap={trad_gap:+.3f}  [{agree}]")
