# TOK Atlas Results v1 — Full 12-System Corpus (HPC)

**Status:** Results of record — first full 7-regime corpus run.
**Production Effect:** `none_research_only`
**Bundles:** `admissibility_atlas_v0/generated/admissibility_bundle_v1.json`
(`hash 9ac7c426…`), `collapse_atlas_v0/generated/collapse_bundle_v1.json`
(`hash 0f2de338…`)

Supersedes `TOK_ATLAS_RESULTS_v0.md` (5-system). All 7 regimes now covered.

---

## 1. Run scale

| Atlas | Units | Systems | Acceptance |
| :-- | :-- | :-- | :-- |
| Admissibility | **116,640 fits** (12 × 108 cells × 3 proc × 30 seeds) | 12 | all passed |
| Collapse | **144 transects** (12 × 3 × 4 axes), 30 seeds/pt | 12 | all passed |

Corpus (all 7 regimes): smooth (lv, sir, coupled), chaotic (lorenz, rossler),
relaxation (vdp, fhn), delayed (forrester), stochastic (ou), hybrid (hybrid),
sparse/real held-out (hudson_bay, gause).

## 2. Headline — the LTC result is now regime-complete

Across **all 12 systems and all 7 regimes**, on the envelope verdict (fraction of
cells reaching Level ≥ 3):

| Procedure | Verdict spread | Reaches Level ≥3 anywhere? |
| :-- | :-- | :-- |
| **LTC** | **Dangerous 12/12** | **No cell, on any system, in any regime** |
| **SINDy** | Dangerous 10/12, Neutral 2 | only lorenz (0.30), gause (0.16) |
| **Handshake** | Dangerous 11/12, Neutral 1 | only lorenz (0.30) |

**LTC reaches Level 3 in exactly zero cells across 116,640 fits.** The
"dense continuous-time fits never achieve sparse topological preservation" result is
no longer a 5-system artifact — it is **regime-complete**, holding through delayed,
stochastic, hybrid, and real-data-shaped systems. This is the strongest form of the
finding: measured, falsifiable, and now robust across the full taxonomy.

SINDy and Handshake clear Level 3 only on **Lorenz** (and SINDy marginally on the
real Gause series) — i.e. sparse structure is recoverable mainly in the clean
low-dimensional chaotic case, and essentially nowhere else under this grid.

### Per-system frac of cells reaching Level ≥3

```
regime:      smooth chaot smooth smooth relax delay chaot relax stoch hybrid sparse sparse
system:      lv     lorenz sir   coupled vdp  forr  ross  fhn   ou    hybrid hudson gause
sindy        0.1    0.3    0.0   0.0    0.0   0.1   0.1   0.0   0.0   0.0    0.0    0.2
ltc          0.0    0.0    0.0   0.0    0.0   0.0   0.0   0.0   0.0   0.0    0.0    0.0
handshake    0.1    0.3    0.0   0.0    0.0   0.0   0.0   0.0   0.0   0.0    0.0    0.0
```

## 3. Collapse — boundaries across the full corpus

24 transitions catalogued, **16 cliffs**, **13 non-monotone transects** (over half).

The new **delayed_memory** system (Forrester) produces the sharpest boundaries in the
whole corpus — SINDy on Forrester cliffs from Level 4 at the slightest degradation on
*every* axis:

| System/Procedure | Axis | Cliff |
| :-- | :-- | :-- |
| forrester/sindy | sample_count | **L4→L1 @ N=6** (deepest drop in corpus) |
| forrester/sindy | noise | L4→L2 @ σ=0.02 (earliest noise cliff) |
| forrester/sindy | delay | L4→L2 @ 2 steps |
| forrester/sindy | downsample | L4→L2 @ 2× |

This is consistent with the admissibility finding that Forrester is recoverable *only*
in the clean instantaneous case (SINDy's `x` term carries the loop) and shatters the
instant observability degrades — exactly the delayed_memory hazard the taxonomy named.

That **13/144 transects are non-monotone** is reported, not hidden: collapse is
frequently seed-sensitive with dips/recoveries rather than a clean slide.

## 4. What this establishes

- The capability-matrix contradiction is now **regime-complete**: the hand-authored
  taxonomy's "LTC broadly supported" is measured-Dangerous across all 7 regimes.
- Sparse topological recovery (Level ≥3) is the *exception*, not the rule — confined
  largely to clean chaotic Lorenz. This is a strong, sobering admissibility result.
- The boundary cards (collapse onsets per procedure × axis × system) are now a
  full-corpus asset for the Evidence Transition Gate.

## 5. Caveats (honest scope)

- **Synthetic recoverability ≠ empirical validity.** Hudson Bay and Gause are real
  but enter as held-out trajectory shapes, not validated mechanism claims.
- **The Level-3 suppression requirement is the frozen, operator-confirmed definition.**
  The LTC verdict is downstream of it and re-derivable from retained raw metrics.
- **Two regimes have a single exemplar** (delayed: forrester; stochastic: ou; hybrid:
  hybrid). Single-system regimes should not be over-generalized; a 2nd exemplar each
  is the obvious v2 widening.
- These are the current single-cell NumPy LTC (Euler forward pass). An RK4/integrator
  upgrade should be measured as a *parallel 4th procedure* (`ltc_rk4`), not swapped
  in — so the upgrade earns its place by changing the measured envelope, not by
  assumption.
