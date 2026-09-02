"""
grokking_scaling_theory.sharpening
----------------------------------
The sharpening-hypothesis test (PHASE2 amendment 6, Part A).

Question: the depth-induced joint rise of Fourier concentration and
quantization purity (finding 13) — is it individual units morphing
sinusoid -> square (SHARPENING, continuous), or sinusoidal units fading
while distinct cell-units are born already-quantized (REPLACEMENT, the
discrete LIC picture)?

The aggregate rise cannot discriminate; within-unit trajectories and unit
identity can (amendment 6, Part A):

  S1  identity continuity: Jaccard overlap of structured-unit sets,
      early vs late window.
  S2a morphing: within-unit slope of the overtone ratio
      P(3k*)/P(k*) over normalized time, among EARLY-structured units.
      Square-wave reference: (1/3)^2 = 1/9 ~ 0.111.
  S2b onset waveform: overtone ratio at structure onset for units that
      become structured only LATE (replacement: born ~square, >= 0.06;
      sharpening: born ~sinusoidal, ~0).
  S3  bimodality: 2-vs-1 component GMM BIC on late-window purity of
      structured units (replacement predicts two modes mid-transition).

Adjudication (fixed pre-data): SHARPENING if median S1 > 0.5 and S2a
CI > 0. REPLACEMENT if median S1 < 0.3, S2a CI not above 0, and
(S2b >= 0.06 or S3 bimodal in >= half of runs). Otherwise mixed.

Gate SG (mandatory before real traces): both worlds are synthesized and
the statistics must classify each correctly.

Usage:
    python -m grokking_scaling_theory.sharpening            # gate SG
    python -m grokking_scaling_theory.sharpening --run DIR  # real traces
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from grokking_scaling_theory.sheaf_order_parameter import TraceArrays
from grokking_scaling_theory.logical_cells import (
    proposition_family,
    unit_states,
    PURITY,
)

CONC_STRUCTURED = 0.5  # amendment 6 Part A item 2
SQUARE_OVERTONE = 1.0 / 9.0
ONSET_SQUARE_MIN = 0.06
JACCARD_SHARPENING = 0.5
JACCARD_REPLACEMENT = 0.3
DELTA_BIC = 10.0
# Gate-SG-forced refinement (pre-real-data, recorded in amendment 6): a
# morph slope must be MATERIAL on the waveform scale, not merely
# statistically nonzero — SNR creep in fading units produces slopes ~1% of
# the sine->square change. Materiality = 1/4 of the square-wave overtone.
S2A_MATERIAL = SQUARE_OVERTONE / 4.0


# ---------------------------------------------------------------------------
# Per-unit metrics (amendment 6, Part A item 1)
# ---------------------------------------------------------------------------


def class_profiles(X: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Class-conditional mean activation profiles, [p, d]."""
    p = int(classes.max()) + 1
    prof = np.zeros((p, X.shape[1]))
    for c in range(p):
        m = classes == c
        if m.any():
            prof[c] = np.asarray(X[m], dtype=np.float64).mean(axis=0)
    return prof


def _alias(k: int, p: int) -> int:
    """Fold harmonic k into the rfft range [1, (p-1)//2] for prime p."""
    k = k % p
    if k > p // 2:
        k = p - k
    return k


def unit_metrics(X: np.ndarray, classes: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-unit conc, overtone, purity at one checkpoint.

    conc_a     = max non-DC harmonic share of the class profile
    overtone_a = P(alias(3 k*)) / P(k*), scale-invariant waveform measure
    purity_a   = best two-sided purity over the amendment-4 proposition
                 family (max over propositions of the two-sided minimum)
    """
    p = int(classes.max()) + 1
    prof = class_profiles(X, classes)
    F = np.fft.rfft(prof, axis=0)
    power = np.abs(F[1:]) ** 2  # k = 1 .. p//2
    total = power.sum(axis=0)
    ok = total > 1e-12
    kstar = np.argmax(power, axis=0) + 1  # dominant harmonic (1-indexed)
    conc = np.where(ok, power.max(axis=0) / np.maximum(total, 1e-12), 0.0)

    overtone = np.zeros(X.shape[1])
    for a in range(X.shape[1]):
        if not ok[a]:
            continue
        k1 = int(kstar[a])
        k3 = _alias(3 * k1, p)
        p1 = power[k1 - 1, a]
        p3 = power[k3 - 1, a] if k3 >= 1 else 0.0
        overtone[a] = float(p3 / max(p1, 1e-12))

    # per-unit best two-sided purity over the proposition family
    states = unit_states(np.asarray(X, dtype=np.float64))
    props = proposition_family(p)
    best = np.zeros(X.shape[1])
    for mask in props.values():
        member = mask[classes]
        P_, nP = states[member], states[~member]
        if len(P_) == 0 or len(nP) == 0:
            continue
        fp_pos = (P_ == 1).mean(axis=0)
        fp_neg = (P_ == -1).mean(axis=0)
        fn_pos = (nP == 1).mean(axis=0)
        fn_neg = (nP == -1).mean(axis=0)
        two_sided = np.maximum(
            np.minimum(fp_pos, fn_neg), np.minimum(fp_neg, fn_pos)
        )
        best = np.maximum(best, two_sided)

    return {"conc": conc, "overtone": overtone, "purity": best}


# ---------------------------------------------------------------------------
# Trajectories and statistics (items 2-3)
# ---------------------------------------------------------------------------


@dataclass
class LayerSharpening:
    """S-statistics for one (run, layer)."""

    label: str
    n_early_structured: int
    n_late_structured: int
    n_late_only: int
    s1_jaccard: float
    s2a_median_slope: float
    s2a_ci: Tuple[float, float]
    s2b_onset_overtone: float
    s3_delta_bic: float
    s3_upper_mean: float
    s3_bimodal: bool
    final_overtone_median: float


def _gmm_1d_bic(x: np.ndarray, n_comp: int, n_iter: int = 200, seed: int = 0) -> float:
    """BIC of an n_comp-component 1D Gaussian mixture (tiny EM)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    rng = np.random.default_rng(seed)
    if n_comp == 1:
        mu, var = float(np.mean(x)), float(np.var(x) + 1e-9)
        ll = float(np.sum(_norm_logpdf(x, mu, var)))
        k = 2
        return -2 * ll + k * np.log(n)
    # init: split at median
    mus = np.array([np.percentile(x, 25), np.percentile(x, 75)], dtype=float)
    vars_ = np.array([np.var(x) + 1e-6] * 2)
    pis = np.array([0.5, 0.5])
    for _ in range(n_iter):
        logp = np.stack(
            [np.log(pis[j] + 1e-12) + _norm_logpdf(x, mus[j], vars_[j]) for j in range(2)]
        )
        logp -= logp.max(axis=0, keepdims=True)
        r = np.exp(logp)
        r /= r.sum(axis=0, keepdims=True)
        for j in range(2):
            w = r[j].sum() + 1e-12
            mus[j] = float((r[j] * x).sum() / w)
            vars_[j] = float((r[j] * (x - mus[j]) ** 2).sum() / w + 1e-9)
            pis[j] = float(w / n)
    ll = float(
        np.sum(
            np.log(
                sum(
                    pis[j] * np.exp(_norm_logpdf(x, mus[j], vars_[j]))
                    for j in range(2)
                )
                + 1e-300
            )
        )
    )
    k = 5
    _gmm_1d_bic.last_upper_mean = float(max(mus))  # type: ignore[attr-defined]
    return -2 * ll + k * np.log(n)


def _norm_logpdf(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    return -0.5 * (np.log(2 * np.pi * var) + (x - mu) ** 2 / var)


def layer_sharpening(
    hidden: np.ndarray,  # [T, N, d]
    classes: np.ndarray,
    label: str,
    n_boot: int = 2000,
    seed: int = 0,
) -> LayerSharpening:
    """All S-statistics for one layer's trajectory."""
    T = hidden.shape[0]
    metrics = [unit_metrics(hidden[t], classes) for t in range(T)]
    conc = np.stack([m["conc"] for m in metrics])       # [T, d]
    over = np.stack([m["overtone"] for m in metrics])   # [T, d]
    pur = np.stack([m["purity"] for m in metrics])      # [T, d]

    third = max(T // 3, 1)
    early_idx, late_idx = third - 1, T - 1
    early_set = set(np.flatnonzero(conc[early_idx] >= CONC_STRUCTURED))
    late_set = set(np.flatnonzero(conc[late_idx] >= CONC_STRUCTURED))
    late_only = sorted(late_set - early_set)

    # S1: identity continuity
    union = early_set | late_set
    s1 = float(len(early_set & late_set) / len(union)) if union else float("nan")

    # S2a: within-unit overtone slope among EARLY-structured units
    tnorm = np.linspace(0.0, 1.0, T)
    slopes: List[float] = []
    for a in sorted(early_set):
        # Overtone is only meaningful while the unit's profile IS structured
        # (registered S2a definition): a fading unit's noise-dominated
        # checkpoints would otherwise inflate P3/P1 and fake a morph.
        mask = conc[:, a] >= CONC_STRUCTURED
        if mask.sum() >= 4:
            slopes.append(float(np.polyfit(tnorm[mask], over[mask, a], 1)[0]))
    if slopes:
        rng = np.random.default_rng(seed)
        boots = [
            float(np.median(rng.choice(slopes, len(slopes), replace=True)))
            for _ in range(n_boot)
        ]
        s2a = float(np.median(slopes))
        s2a_ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    else:
        s2a, s2a_ci = float("nan"), (float("nan"), float("nan"))

    # S2b: onset overtone of late-only structured units
    onsets: List[float] = []
    for a in late_only:
        crossings = np.flatnonzero(conc[:, a] >= CONC_STRUCTURED)
        if len(crossings):
            onsets.append(float(over[crossings[0], a]))
    s2b = float(np.median(onsets)) if onsets else float("nan")

    # S3: bimodality of late-window purity among structured units
    late_pur = pur[late_idx, sorted(late_set)] if late_set else np.array([])
    if len(late_pur) >= 8:
        bic1 = _gmm_1d_bic(late_pur, 1)
        bic2 = _gmm_1d_bic(late_pur, 2)
        upper = getattr(_gmm_1d_bic, "last_upper_mean", float("nan"))
        dbic = float(bic1 - bic2)
        bimodal = (dbic > DELTA_BIC) and (upper > 0.8)
    else:
        dbic, upper, bimodal = float("nan"), float("nan"), False

    final_over = (
        float(np.median(over[late_idx, sorted(late_set)])) if late_set else float("nan")
    )

    return LayerSharpening(
        label=label,
        n_early_structured=len(early_set),
        n_late_structured=len(late_set),
        n_late_only=len(late_only),
        s1_jaccard=s1,
        s2a_median_slope=s2a,
        s2a_ci=s2a_ci,
        s2b_onset_overtone=s2b,
        s3_delta_bic=dbic,
        s3_upper_mean=upper,
        s3_bimodal=bimodal,
        final_overtone_median=final_over,
    )


def classify(res: LayerSharpening) -> str:
    """Amendment 6 Part A item 4 adjudication for one (run, layer), with the
    gate-SG-forced materiality refinement: the morph slope must be both
    statistically positive AND material (>= S2A_MATERIAL) for sharpening;
    an immaterial slope (< S2A_MATERIAL) counts as no-morph for replacement
    regardless of statistical sign."""
    s1, s2a_lo = res.s1_jaccard, res.s2a_ci[0]
    material = np.isfinite(res.s2a_median_slope) and res.s2a_median_slope >= S2A_MATERIAL
    if (
        np.isfinite(s1) and s1 > JACCARD_SHARPENING
        and np.isfinite(s2a_lo) and s2a_lo > 0 and material
    ):
        return "SHARPENING"
    repl_id = np.isfinite(s1) and s1 < JACCARD_REPLACEMENT
    repl_slope = not material
    repl_wave = (
        np.isfinite(res.s2b_onset_overtone)
        and res.s2b_onset_overtone >= ONSET_SQUARE_MIN
    ) or res.s3_bimodal
    if repl_id and repl_slope and repl_wave:
        return "REPLACEMENT"
    return "mixed/unresolved"


# ---------------------------------------------------------------------------
# Gate SG: two synthetic worlds (item 5)
# ---------------------------------------------------------------------------


def _world_trace(
    world: str, p: int = 29, per_class: int = 12, d: int = 48, T: int = 30,
    noise: float = 0.05, seed: int = 0,
) -> TraceArrays:
    """World A (sharpening): units morph cos -> sign(cos), same units
    throughout. World B (replacement): sinusoid population fades while a
    disjoint population is born already-square."""
    # The transition occupies the MIDDLE third of the trace, leaving clean
    # pre- and post-transition windows. (A crossfade already underway at the
    # early window is an ill-posed test of identity turnover — caught by the
    # first Gate SG run; construction fixed, statistics untouched.)
    rng = np.random.default_rng(seed)
    classes = np.repeat(np.arange(p), per_class)
    N = len(classes)
    epochs = np.arange(T) * 20
    hidden = np.zeros((T, N, d))
    ks = rng.integers(1, 5, size=d)
    phase = classes.astype(float)
    for i in range(T):
        s = float(np.clip((i / (T - 1) - 1.0 / 3.0) * 3.0, 0.0, 1.0))
        for a in range(d):
            wave_sin = np.cos(2 * np.pi * ks[a] * phase / p)
            wave_sq = np.sign(wave_sin)
            if world == "A":
                # every unit morphs continuously, amplitude constant
                wave = (1 - s) * wave_sin + s * wave_sq
                hidden[i][:, a] = 2.0 * wave
            else:
                # first half of units: sinusoids fading out;
                # second half: square cells fading in (disjoint identities)
                if a < d // 2:
                    hidden[i][:, a] = 2.0 * (1 - s) * wave_sin
                else:
                    hidden[i][:, a] = 2.0 * s * wave_sq
        hidden[i] += noise * rng.normal(size=(N, d))
    return TraceArrays(epochs=epochs, hidden=hidden, classes=classes)


def run_gate_sg(seed: int = 0, verbose: bool = True) -> Dict[str, object]:
    """Both worlds must be classified correctly. AssertionError on failure."""
    ta = _world_trace("A", seed=seed)
    ra = layer_sharpening(ta.hidden, ta.classes, "worldA")
    va = classify(ra)
    assert va == "SHARPENING", f"Gate SG FAIL: world A classified {va} ({ra})"

    tb = _world_trace("B", seed=seed)
    rb = layer_sharpening(tb.hidden, tb.classes, "worldB")
    vb = classify(rb)
    assert vb == "REPLACEMENT", f"Gate SG FAIL: world B classified {vb} ({rb})"

    if verbose:
        print("=" * 74)
        print(" SHARPENING TEST: GATE SG (two synthetic worlds, amendment 6)")
        print("=" * 74)
        for tag, r, v in (("World A", ra, va), ("World B", rb, vb)):
            print(
                f" {tag}: {v:12s} S1={r.s1_jaccard:.2f} "
                f"S2a={r.s2a_median_slope:+.3f} CI[{r.s2a_ci[0]:+.3f},{r.s2a_ci[1]:+.3f}] "
                f"S2b={r.s2b_onset_overtone:.3f} S3bimodal={r.s3_bimodal}"
            )
        print(" Gate SG PASS")
        print("=" * 74)
    return {"worldA": ra, "worldB": rb}


# ---------------------------------------------------------------------------
# Real-trace runner
# ---------------------------------------------------------------------------


def analyze_trace(npz_path: Path) -> List[Tuple[str, LayerSharpening, str]]:
    """S-statistics for the deep layers of one trace (l2, final; l1 skipped
    as unstructured per finding 12). Shallow traces: final layer only."""
    raw = np.load(npz_path)
    classes = np.asarray(raw["classes"], dtype=int)
    out: List[Tuple[str, LayerSharpening, str]] = []

    def _capped(key: str, max_t: int = 800) -> np.ndarray:
        # Memory discipline: keep the stored float16 dtype (per-checkpoint
        # upcasting happens inside unit_metrics) and cap the analysis
        # cadence — giant censored traces log 4000 checkpoints, and the
        # S-statistics are insensitive to cadence (the deep-3 verdicts were
        # grounded on 100-epoch aux cadence).
        arr = raw[key]
        stride = max(1, int(np.ceil(len(arr) / max_t)))
        return arr[::stride]

    layers = [("final", _capped("hidden"))]
    aux_keys = sorted(
        (k for k in raw.files if k.startswith("hidden_l")),
        key=lambda k: int(k.rsplit("hidden_l", 1)[1]),
    )
    if aux_keys:
        deepest = aux_keys[-1]  # l2 for deep-3 traces, l4 for deep-5
        layers.insert(0, (deepest.replace("hidden_", ""), _capped(deepest)))
    for lname, arr in layers:
        res = layer_sharpening(arr, classes, lname)
        out.append((lname, res, classify(res)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=str, default=None,
                        help="trace dir: analyze real traces (deep family + shallow context)")
    parser.add_argument("--out", type=str, default="analysis/SHARPENING_RESULTS.md")
    args = parser.parse_args()

    run_gate_sg()  # gate always runs first; hard-fails if broken
    if not args.run:
        return

    trace_dir = Path(args.run)
    lines: List[str] = []
    lines.append("# Sharpening-Hypothesis Test Results (amendment 6, Part A)")
    lines.append("")
    lines.append(
        "Generated by `grokking_scaling_theory.sharpening`. Statistics, "
        "thresholds, and the adjudication rule are fixed in PHASE2 amendment "
        "6; Gate SG (two synthetic worlds) passed before this analysis."
    )
    lines.append("")
    lines.append("| trace | layer | early/late/new units | S1 Jacc | S2a slope [CI] | S2b onset | S3 dBIC | verdict |")
    lines.append("|---|---|---|---:|---|---:|---:|---|")

    verdicts: Dict[str, List[str]] = {}
    for npz in sorted(trace_dir.glob("trace_*.npz")):
        meta = json.loads(npz.with_suffix(".json").read_text())
        arch = meta["config"]["arch"]
        is_deep = arch == "deep" or "hidden_l2" in np.load(npz)
        # deep family = confirmatory; one shallow trace per family/p as context
        for lname, res, verdict in analyze_trace(npz):
            if not is_deep and lname != "final":
                continue
            tag = npz.stem.replace("trace_", "")
            s2a = (
                f"{res.s2a_median_slope:+.3f} [{res.s2a_ci[0]:+.3f},{res.s2a_ci[1]:+.3f}]"
                if np.isfinite(res.s2a_median_slope) else "n/a"
            )
            s2b = f"{res.s2b_onset_overtone:.3f}" if np.isfinite(res.s2b_onset_overtone) else "n/a"
            dbic = f"{res.s3_delta_bic:.1f}" if np.isfinite(res.s3_delta_bic) else "n/a"
            lines.append(
                f"| {tag} | {lname} | {res.n_early_structured}/{res.n_late_structured}/{res.n_late_only} "
                f"| {res.s1_jaccard:.2f} | {s2a} | {s2b} | {dbic} | {verdict} |"
            )
            if is_deep:
                verdicts.setdefault(lname, []).append(verdict)

    lines.append("")
    lines.append("## Family adjudication (deep runs)")
    lines.append("")
    for lname, vs in sorted(verdicts.items()):
        from collections import Counter
        counts = Counter(vs)
        lines.append(f"- **{lname}**: " + ", ".join(f"{k} x{v}" for k, v in counts.most_common()))
    lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
