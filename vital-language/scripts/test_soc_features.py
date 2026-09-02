"""Test refined SOC metrics on the literary corpus with ONE multilingual model
(comparable across languages). Decisive question: does bounded_agitation rank
Lispector HIGH (where global_range ranked her middling), and does
perception-abstraction oscillation separate SOC from Austen?
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vitality.metrics.soc_features import compute_soc
from sentence_transformers import SentenceTransformer


def strip_g(t):
    m = re.search(r"\*\*\* ?START OF.*?\*\*\*", t, re.S)
    if m: t = t[m.end():]
    m = re.search(r"\*\*\* ?END OF", t, re.S)
    if m: t = t[:m.start()]
    return t


def window(path, start_frac=0.45, nchars=9000, gutenberg=True):
    t = open(path, encoding="utf-8", errors="ignore").read()
    if gutenberg:
        t = strip_g(t)
    s = int(len(t) * start_frac)
    return t[s:s + nchars]


model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Lispector: skip front matter (~line 28), take a body window
liz_lines = open("agua_vida.md", encoding="utf-8").read().split("\n")
liz = "\n".join(liz_lines[28:])[:9000]

samples = {
    "Lispector/AguaViva (ES, SOC)": liz,
    "Joyce/Ulysses (SOC)": window("corpus/ulysses.txt"),
    "Woolf/VoyageOut (conventional-ish)": window("corpus/woolf_voyage.txt"),
    "Austen/Pride (conventional)": window("corpus/austen_pride.txt"),
}

print(f"{'text':34s} {'n':>3} {'drift':>6} {'range':>6} {'bnd_agit':>8} "
      f"{'fill':>6} {'pa_osc':>6} {'pa_flip':>7}")
print("-" * 86)
rows = {}
for label, txt in samples.items():
    m = compute_soc(txt, model)
    rows[label] = m
    print(f"{label:34s} {m.n_sent:3d} {m.local_drift:6.3f} {m.global_range:6.3f} "
          f"{m.bounded_agitation:8.3f} {m.fill_ratio:6.3f} "
          f"{m.pa_oscillation_std:6.3f} {m.pa_flip_rate:7.3f}")

print("\nDECISIVE CHECKS:")
liz_m = rows["Lispector/AguaViva (ES, SOC)"]
aus_m = rows["Austen/Pride (conventional)"]
print(f"  Lispector bounded_agitation ({liz_m.bounded_agitation:.3f}) vs "
      f"Austen ({aus_m.bounded_agitation:.3f}): "
      f"{'LISPECTOR HIGHER (good)' if liz_m.bounded_agitation > aus_m.bounded_agitation else 'NOT higher (metric fails her)'}")
print(f"  Lispector pa_oscillation ({liz_m.pa_oscillation_std:.3f}) vs "
      f"Austen ({aus_m.pa_oscillation_std:.3f}): "
      f"{'LISPECTOR HIGHER (good)' if liz_m.pa_oscillation_std > aus_m.pa_oscillation_std else 'NOT higher'}")
print(f"  Recall: global_range ranked Lispector only MIDDLING (~0.47). "
      f"bounded_agitation should re-rank her up if she is local-agitation/contained.")
