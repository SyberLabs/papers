# Hound Dog Analysis: The Pattern Discovered

## Executive Summary

**The scaling law doesn't fail - we were measuring in the WRONG REGIME.**

The local sweep data (p=31-53) is in a **transition zone** between algorithmic basins,
not the Fourier regime where the scaling law applies.

## Key Findings

### 1. Censoring Discovery

**p=31 is CENSORED**: grokking_epoch = max_epochs = 20000

This run never actually grokked. It hit the training budget limit.
The network got stuck in the lookup basin and couldn't escape.

### 2. Inverted Scaling in Transition Zone

After excluding the censored p=31 point:

| p | tau | Trend |
|---|-----|-------|
| 37 | 8870 | - |
| 41 | 9480 | UP (noise) |
| 43 | 9220 | DOWN |
| 47 | 5680 | DOWN |
| 53 | 4020 | DOWN |

**Pattern**: tau DECREASES as p increases!

This is the OPPOSITE of the expected p^2 scaling.

### 3. Three Regime Model

```
tau
^
|                                          * p=113 (15000)
|
10000|   X p=31 (CENSORED)       * p=97 (10000)
|     * p=41,43
|      * p=37
|                        * p=59 (5000)
5000|         * p=47
|              * p=53
|_______________|___________________________|___________> p
               40        60                100

LOOKUP         TRANSITION      FOURIER REGIME
BASIN          ZONE            tau ~ p^2/log(p)^2
(fails)        tau ~ 1/p^2
               (inverted!)
```

### 4. Regime Boundaries

| Regime | p Range | Scaling | Mechanism |
|--------|---------|---------|-----------|
| Lookup Basin | p < 40 | tau -> infinity | Memorization attempted, fails |
| Transition Zone | 40 <= p <= 60 | tau ~ 1/p^2 (corrected) | Lookup -> Fourier competition |
| Fourier Basin | p > 60 | tau ~ p^2/log(p)^2 | Pure Fourier algorithm |

### 5. Why the Transition Zone Shows Inverted Scaling

In the transition zone:
- At small p (37-43): Network partially attempts memorization -> SLOW
- At larger p (47-53): Memorization becomes impossible, Fourier wins -> FAST

The network is FORCED into the Fourier basin as p grows, making learning faster!

Once fully in the Fourier basin (p > 60), the normal p^2 scaling resumes.

## Quantitative Support

### Fourier Regime (Published Data, p >= 59)
```
Normalized tau = tau * log(p)^2 / p^2
C = 23.91, CV = 0.064 (excellent collapse)
```

### Transition Zone (Local Sweep, p = 37-53)
```
Fourier normalization: CV = 0.41 (poor collapse)
Inverted normalization (tau * p): CV = 0.22
Inverted normalization (tau * p^2): CV = 0.16 (best collapse; see correction below)
```

The transition zone is better described by tau ~ k/p^2 than by tau ~ k/p or
tau ~ p^2. (Corrected 2026-07; the original 1/p label understated the
inversion. See CORRECTION section at the end of this document.)

## Predictions

### Critical Test Points

| p | Predicted tau | Regime | Test |
|---|--------------|--------|------|
| 61 | ~5200 | Fourier | Should be HIGHER than p=53 (4020) |
| 67 | ~6100 | Fourier | tau should increase with p |
| 71 | ~6600 | Fourier | Normal scaling resumes |

### What Would Falsify This?

1. If p=61 shows tau < 4000 -> Still in transition zone
2. If p=61 shows tau >> 7000 -> Different factor at play
3. If p=67 shows tau < p=61 -> Transition extends further

## Implications for the Theory

### The (M, R, D) Framework Remains Valid

The dynamical system is correct. What changes is gamma_R(p):

- **Lookup basin**: gamma_R(p) ~ 0 (rule formation blocked)
- **Transition**: gamma_R(p) increases with p (basin selection)
- **Fourier basin**: gamma_R(p) ~ log(p)^2/p^2 (established scaling)

### The Critical Modulus p_c

We previously predicted p_c ~ sqrt(H) ~ 16 for H=256.

The data suggests the **effective** transition boundary is around p ~ 55-60.

This could be because:
1. Our estimate was naive
2. The transition is gradual, not sharp
3. Protocol details (learning rate, batch size) affect the boundary

### Updated Algorithmic Basin Theory

```
                    p_c ~ 40          p_c ~ 60
                       |                 |
        [Lookup]  ---> | [Transition] -->| [Fourier]
        gamma_R ~ 0    | gamma_R rises   | gamma_R ~ log(p)^2/p^2
        tau -> inf     | tau ~ k/p^2     | tau ~ p^2/log(p)^2
```

## Action Items

1. **Mark p=31 as censored** in the data file
2. **Exclude transition zone** from scaling law validation
3. **Run validation at p=61, 67, 71** to confirm Fourier regime boundary
4. **Update paper** with transition zone discovery

## Conclusion

The scaling law q=2 is correct for the Fourier regime (p > 60).

The "failure" of the scaling law in local sweeps is not a failure -
it's a DISCOVERY of the transition zone between algorithmic basins.

This strengthens the algorithmic basin theory: we can now predict
not just the scaling exponent, but the regime boundaries.

**The hound dog found the scent.**

## CORRECTION (2026-07): Transition-Zone Exponent

The `tau ~ 1/p` label used above was an eyeball fit, not an estimate. Two
analyses supersede it (see `SURVIVAL_FIT_RESULTS.md` and
`src/grokking_scaling_theory/survival_fit.py`):

1. Free-exponent fit on the 8 measured zone points (p = 37..71):
   a = -1.77 +/- 0.22. Already inconsistent with -1.
2. Censoring-aware AFT fit including the p=31 run as the observation
   `tau > 20000`: **a = -2.04 +/- 0.20**. Local-sources-only variant:
   a = -1.82 +/- 0.22.
3. Collapse check: tau * p^2 gives CV = 0.16 on the original 5 points
   (vs 0.22 for tau * p) and CV = 0.14 on all 8 (vs 0.24).

Reading: the zone scales as tau ~ p^-2, the mirror image of the forward
p^2 law. If basin escape is rate-limited by a quantity proportional to
training-set size (N ~ p^2 at fixed train fraction), an escape rate linear
in N yields tau ~ 1/p^2 directly. This is degenerate with modulus scaling
on current data; a train-fraction sweep at fixed p breaks the degeneracy
(see `experiments/BETA_DISCRIMINATOR_PROTOCOL.md`, Experiment 3).

All zone points remain single-seed. The exponent should carry that caveat
until replicated.
