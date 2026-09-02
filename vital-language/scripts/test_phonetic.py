"""Does the PHONETIC channel separate Joyce (predicted 'phonetic whirlpool') from
conventional prose, where the imagistic channel did NOT? English-only corpus.
Multi-window for robustness. Includes a machine-output sample (dead channel?)."""
import glob, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vitality.metrics.phonetic import compute_phonetic
import json


def strip_g(t):
    m = re.search(r"\*\*\* ?START OF.*?\*\*\*", t, re.S); t = t[m.end():] if m else t
    m = re.search(r"\*\*\* ?END OF", t, re.S); return t[:m.start()] if m else t


def wins(p, g=True, n=4, sz=6000):
    t = open(p, encoding="utf-8", errors="ignore").read()
    if g: t = strip_g(t)
    return [t[int(len(t)*(0.2+0.15*i)):int(len(t)*(0.2+0.15*i))+sz] for i in range(n)]


corpus = {
    "Joyce/Ulysses": wins("corpus/ulysses.txt"),
    "Woolf/VoyageOut": wins("corpus/woolf_voyage.txt"),
    "Austen/Pride": wins("corpus/austen_pride.txt"),
}

# add machine-output samples (concatenate a few legible 1.5B passages)
mach = []
for f in glob.glob("outputs/longrun_20260617_081901/longrun.jsonl"):
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        if r["n_tokens"] > 300 and r["condition"] == "sampling":
            mach.append(r["text"])
if mach:
    corpus["MACHINE/1.5B sampling"] = mach[:4]

print(f"{'text':24s} | {'allit':>6} {'ph_rec':>6} {'vow_cont':>8} "
      f"{'stress_var':>10} {'FLOW':>6}")
print("-" * 72)
results = {}
for name, ws in corpus.items():
    ms = [compute_phonetic(w) for w in ws]
    ms = [m for m in ms if m.phonetic_flow == m.phonetic_flow]
    if not ms:
        continue
    a = np.mean([m.alliteration_density for m in ms])
    pr_ = np.mean([m.phoneme_recurrence for m in ms])
    vc = np.mean([m.vowel_continuity for m in ms])
    sv = np.mean([m.stress_variance for m in ms])
    fl = np.mean([m.phonetic_flow for m in ms])
    results[name] = fl
    print(f"{name:24s} | {a:6.3f} {pr_:6.3f} {vc:8.3f} {sv:10.3f} {fl:6.3f}")

print("\nKEY: does Joyce phonetic_flow exceed Austen / machine?")
if "Joyce/Ulysses" in results and "Austen/Pride" in results:
    j, au = results["Joyce/Ulysses"], results["Austen/Pride"]
    print(f"  Joyce {j:.3f} vs Austen {au:.3f}: "
          f"{'JOYCE HIGHER' if j > au else 'not higher'}")
    if "MACHINE/1.5B sampling" in results:
        print(f"  machine {results['MACHINE/1.5B sampling']:.3f} "
              f"(dead channel if lowest?)")
