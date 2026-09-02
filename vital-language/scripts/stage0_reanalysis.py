"""Stage 0 (advisor): decisive cheap reanalysis, zero generation.

Tests the advisor's central conjecture: width's -0.51 vs vitality is largely a
TEMPLATE/DEGENERACY confound. If width's negative correlation ATTENUATES below
~|0.25| once we partial out a degeneracy flag, the advisor is right (width tracks
structured-substrate collapse, not an anti-vital text axis per se). If it stays
strongly negative among clean passages, the advisor is wrong about the mechanism.

Also: recurrence_rate vs vitality within the clean subset (stuckness-confound
test), and mean-surprisal vs semantic-volume as the 'novelty carrier'.
"""
import glob, hashlib, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vitality.metrics.recurrence import compute_recurrence
from vitality.metrics.surface import split_sentences, tokenize_words

key = json.load(open("study/key.json", encoding="utf-8"))
rt = json.load(open("study/ratings_claude.json", encoding="utf-8"))["ratings"]


def uid_of(run, pid, seed, cond):
    h = hashlib.sha1(f"{run}|{pid}|{seed}|{cond}".encode()).hexdigest()[:10]
    return "p_" + h


text_by_uid, surf_by_uid = {}, {}
for f in glob.glob("outputs/longrun_*/longrun.jsonl"):
    run = os.path.basename(os.path.dirname(f))
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        u = uid_of(run, r["prompt_id"], r["seed"], r["condition"])
        text_by_uid[u] = r["text"]
        surf_by_uid[u] = r["surface"]


# --- degeneracy flag: template/quiz/code/loop substrate ---
TEMPLATE_PAT = re.compile(
    r"(\b[A-D]\)\s|\bAnswer:|\bA\.\s.*\bB\.\s|multiple choice|"
    r"\\\(|\\\[|\\boxed|def |Input\b|lexicographically|"
    r"Creative Commons|licensed under|\bn ≥|integers?\b.*\bpositive)",
    re.I,
)


def degeneracy_score(text):
    words = tokenize_words(text)
    # loop: repeated-4gram rate
    g4 = [tuple(words[i:i+4]) for i in range(len(words)-3)]
    rep4 = 1 - len(set(g4)) / len(g4) if g4 else 0
    # template hits per passage (normalized)
    hits = len(TEMPLATE_PAT.findall(text))
    return rep4, hits, (1 if (hits >= 2 or rep4 > 0.15) else 0)


def partial_corr(x, y, z):
    """corr(x,y | z) via residuals."""
    x, y, z = map(np.asarray, (x, y, z))
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    if len(x) < 6 or z.std() == 0:
        return np.corrcoef(x, y)[0, 1], len(x)
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return float(np.corrcoef(rx, ry)[0, 1]), len(x)


# embeddings for semantic volume
from sentence_transformers import SentenceTransformer
emb_model = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_volume_rate(text):
    sents = [s for s in split_sentences(text) if len(s.split()) >= 2]
    if len(sents) < 6:
        return np.nan
    E = np.asarray(emb_model.encode(sents, normalize_embeddings=True,
                                    show_progress_bar=False), dtype=np.float64)
    # log-det of covariance (regularized) = spread of retained semantic ground
    C = np.cov(E.T) + 1e-4 * np.eye(E.shape[1])
    sign, logdet = np.linalg.slogdet(C)
    return float(logdet) / len(sents)


rows = []
for uid, rr in rt.items():
    if uid not in text_by_uid:
        continue
    t = text_by_uid[uid]
    w = key[uid]["surprisal_width"]
    if not np.isfinite(w):
        continue
    rep4, hits, degen = degeneracy_score(t)
    rec = compute_recurrence(t, emb_model)
    rows.append({
        "uid": uid, "vit": rr["scores"]["vitality"], "width": w,
        "ppl": key[uid]["self_perplexity"],
        "mean_surp": surf_by_uid[uid]["surprisal_mean"],
        "degen": degen, "rep4": rep4, "hits": hits,
        "recurrence": rec.recurrence_rate, "drift": rec.drift,
        "sem_vol": semantic_volume_rate(t),
    })

V = np.array([r["vit"] for r in rows])
W = np.array([r["width"] for r in rows])
D = np.array([r["degen"] for r in rows], float)
clean = D == 0

print(f"n={len(rows)}, degenerate flagged={int(D.sum())}, clean={int(clean.sum())}\n")

print("=== TEST 1: width vs vitality, controlling for degeneracy ===")
raw = np.corrcoef(W, V)[0, 1]
pc, n = partial_corr(W, V, D)
print(f"  raw width~vitality        = {raw:+.3f}")
print(f"  partial width~vitality|deg = {pc:+.3f}  (advisor: should attenuate <|0.25|)")
wc, vc = W[clean], V[clean]
if clean.sum() >= 6:
    print(f"  width~vitality CLEAN only  = {np.corrcoef(wc, vc)[0,1]:+.3f} (n={int(clean.sum())})")

print("\n=== TEST 2: recurrence vs vitality within clean subset ===")
R = np.array([r["recurrence"] for r in rows])
print(f"  recurrence~vitality ALL    = {np.corrcoef(R, V)[0,1]:+.3f}")
if clean.sum() >= 6:
    rc, vc = R[clean], V[clean]
    mm = np.isfinite(rc)
    print(f"  recurrence~vitality CLEAN  = {np.corrcoef(rc[mm], vc[mm])[0,1]:+.3f} "
          f"(advisor: should weaken if stuckness-confound)")

print("\n=== TEST 3: novelty carrier — mean-surprisal vs semantic-volume ===")
MS = np.array([r["mean_surp"] for r in rows])
SV = np.array([r["sem_vol"] for r in rows])
mm = np.isfinite(SV)
print(f"  mean_surprisal~vitality    = {np.corrcoef(MS, V)[0,1]:+.3f}")
print(f"  semantic_volume~vitality   = {np.corrcoef(SV[mm], V[mm])[0,1]:+.3f}")
pc_ms, _ = partial_corr(MS, V, SV)
pc_sv, _ = partial_corr(SV[mm], V[mm], MS[mm])
print(f"  mean_surp~vit | sem_vol    = {pc_ms:+.3f}")
print(f"  sem_vol~vit | mean_surp    = {pc_sv:+.3f}  (which survives = the carrier)")
