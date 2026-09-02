"""Test the imagery hypothesis: does a CLIP-text ('imagistic') trajectory separate
the SOC tradition differently/better than a semantic (MiniLM) trajectory, and is
the IMAGE<->CONCEPT divergence the real signal (decomposed Feature 3)?

CLIP text-encoder embeds a sentence by its evoked-visual-scene content, not its
propositional meaning -- our 'imagistic embedding'. We compute both trajectories
and compare.
"""
import os, re, sys
import numpy as np
import torch
import open_clip
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def strip_g(t):
    m = re.search(r"\*\*\* ?START OF.*?\*\*\*", t, re.S); t = t[m.end():] if m else t
    m = re.search(r"\*\*\* ?END OF", t, re.S); return t[:m.start()] if m else t


def split_sents(t, mn=3):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if len(s.split()) >= mn]


# encoders
sem = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_tok = open_clip.get_tokenizer("ViT-B-32")
clip_model.eval()


def clip_embed(sents):
    with torch.no_grad():
        toks = clip_tok(sents)
        f = clip_model.encode_text(toks).float()
        f = f / f.norm(dim=-1, keepdim=True)
    return f.numpy()


def traj_stats(E):
    consec = np.array([1.0 - float(E[i] @ E[i+1]) for i in range(len(E)-1)])
    cen = E.mean(0); cen /= (np.linalg.norm(cen) + 1e-9)
    grange = float((1.0 - E @ cen).mean())
    return float(consec.mean()), grange, consec


def analyze(label, text, nmax=120):
    sents = split_sents(text)[:nmax]
    Es = np.asarray(sem.encode(sents, normalize_embeddings=True, show_progress_bar=False))
    Ei = clip_embed(sents)
    Ei = Ei / (np.linalg.norm(Ei, axis=1, keepdims=True) + 1e-9)
    sdrift, srange, sc = traj_stats(Es)
    idrift, irange, ic = traj_stats(Ei)
    # cross-channel divergence: correlation of the two per-step drift series.
    # LOW correlation = image and meaning move independently (the 'braid' signature)
    m = min(len(sc), len(ic))
    cross = float(np.corrcoef(sc[:m], ic[:m])[0, 1]) if m > 3 else float("nan")
    return label, len(sents), sdrift, srange, idrift, irange, cross


def strip_liz():
    L = open("agua_vida.md", encoding="utf-8").read().split("\n")
    return "\n".join(L[28:])


def win(p, frac=0.45, n=9000):
    t = strip_g(open(p, encoding="utf-8", errors="ignore").read())
    s = int(len(t) * frac); return t[s:s+n]


samples = {
    "Lispector (ES,SOC)": strip_liz()[:9000],
    "Joyce (SOC)": win("corpus/ulysses.txt"),
    "Woolf VoyageOut": win("corpus/woolf_voyage.txt"),
    "Austen (conventional)": win("corpus/austen_pride.txt"),
}

print(f"{'text':22s} {'n':>3} | {'SEM drift':>9} {'SEM rng':>7} | "
      f"{'IMG drift':>9} {'IMG rng':>7} | {'img-sem xcorr':>13}")
print("-" * 86)
rows = []
for label, txt in samples.items():
    r = analyze(label, txt)
    rows.append(r)
    print(f"{r[0]:22s} {r[1]:3d} | {r[2]:9.3f} {r[3]:7.3f} | "
          f"{r[4]:9.3f} {r[5]:7.3f} | {r[6]:13.3f}")

print("\nKey questions:")
print("- Does IMG drift/range separate SOC from Austen better than SEM did?")
print("- Is img-sem xcorr LOWER for SOC (image & meaning move independently = braid)?")
