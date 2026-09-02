# DTBR-MC Experiment 001 — Report

*Generated 2026-06-15 21:40:07 UTC · model = `baseline` · N = 100,000 agents · seed = 0*

## Headline verdict

**H1 DIRECTION-ONLY (threshold claim UNSUPPORTED)**

PC reduces intervention more steeply than SC at EVERY interpretive-capacity level, with no crossover. H1's qualitative direction (PC is the stronger brake) holds, but its central claim — that this advantage is specific to LOW capacity, below a threshold — is unsupported: there is no threshold and the advantage does not vanish (or even grows) as capacity rises.

> H1 (as stated): *Below a threshold of interpretive capacity, increasing Phenomenological Caution (PC) reduces intervention more than increasing Semantic Clarity (SC).* The simulator was built to look for the failure of this claim, not to confirm it.

## Baseline population

A reference population under fully random (uniform) environment and agent priors, before any lever is pinned:

| Metric | Estimate (95% CI) |
| --- | --- |
| expected harm | 0.0635 (95% CI 0.0631–0.0638) |
| encounter rate | 0.5004 (95% CI 0.4992–0.5017) |
| intervention rate | 0.0010 (95% CI 0.0008–0.0012) |
| excavation rate | 0.0000 (95% CI 0.0000–0.0000) |
| avoidance rate | 0.5315 (95% CI 0.5283–0.5346) |
| preservation rate | 0.0548 (95% CI 0.0534–0.0562) |
| mean hesitation proxy | 0.5007 (95% CI 0.4996–0.5017) |
| mystery to curiosity index | 0.0042 (95% CI -0.0021–0.0103) |
| prestige inversion index | -0.6597 (95% CI -0.6628–-0.6564) |
| behavioral degradation gradient | 0.0273 (95% CI 0.0260–0.0285) |

## Does H1 hold? The lever crossover test

For each interpretive-capacity (IC) level we hold one lever at 0.5 and sweep the other over [0,1] under common random numbers, then fit the marginal slope of mean intervention against each lever. We define

    margin = slope_SC − slope_PC

so that **margin > 0 means PC is the stronger brake** at that capacity. H1 predicts margin > 0 at low IC and a downward crossing (margin ≤ 0) as IC rises — i.e. a *threshold* above which clarity wins.

- At lowest IC (0.00): slope_PC = -0.2542, slope_SC = -0.0458, margin = 0.2085.
- At highest IC (1.00): slope_PC = -0.2542, slope_SC = -0.0458, margin = 0.2085.
- Crossover (intervention): **none within [0,1]** — the levers never swap ranking, so there is no threshold of the kind H1 posits.

See `h1_margin.png` and `phase_diagram.png`.

## What actually drives Expected Harm? (sensitivity)

Total-order Sobol indices over all 17 inputs (uniform priors). Top drivers:

| Rank | Variable | ST | S1 | OAT range |
| --- | --- | --- | --- | --- |
| 1 | repository_severity | 0.5460 | 0.3906 | 0.1250 |
| 2 | phenomenological_caution | 0.1640 | 0.0996 | 0.0625 |
| 3 | accessibility | 0.1577 | 0.0987 | 0.0625 |
| 4 | visibility | 0.1522 | 0.0963 | 0.0625 |
| 5 | economic_pressure | 0.0613 | 0.0332 | 0.0375 |

See `sensitivity_tornado.png`.

## Counterintuitive findings & where the theory fails

1. **The threshold is the weak point of H1.** In the baseline model the PC advantage does not switch off at high capacity — it is roughly flat or even widens. H1's *direction* (PC brakes harder) can hold while its *structure* (only below a capacity threshold) does not. A result that merely shows "PC > SC on average" should not be read as confirming H1.
2. **Prestige inversion index = -0.6597.** A negative value means caution is, on net, suppressing intervention rather than glamorising the site. The backfire channel (mystery → curiosity) only bites where comprehension is low; wherever people partly understand the marker, caution stops being alluring. Whether backfire ever dominates is a property of `backfire_strength`, not a law of the system — try the `backfire` model with a higher strength to see the sign flip.
3. **Severity dominates the levers.** Sensitivity analysis puts repository_severity (a fixed property of the waste, not a messaging choice) well above either lever for E[H]. Communication design moves a second-order term. Any policy claim from this model should foreground that ordering.

## Cross-model robustness of the verdict

The same H1 crossover test was run under several behaviour structures (full N). `margin = slope_SC − slope_PC` on mean intervention; `slope_pc` is PC's own marginal effect at low capacity (negative = PC brakes, positive = PC backfires).

| Configuration | Verdict | low-IC slope_PC | low-IC margin | high-IC margin | IC\* |
| --- | --- | --- | --- | --- | --- |
| `baseline` | DIRECTION-ONLY (threshold claim UNSUPPORTED) | -0.254 | +0.208 | +0.208 | — |
| `backfire` | DIRECTION-ONLY (threshold claim UNSUPPORTED) | -0.210 | +0.145 | +0.195 | — |
| `linear` | DIRECTION-ONLY (threshold claim UNSUPPORTED) | -0.359 | +0.290 | +0.132 | — |
| `backfire_strong` | DIRECTION-ONLY (threshold claim UNSUPPORTED) | -0.206 | +0.132 | +0.143 | — |
| `pc_brake_off` | FALSIFIED | +0.141 | -0.282 | -0.143 | — |

Reading this table:

- Under every model that keeps PC as a direct brake — including `backfire` with strength raised well above default — the verdict is **direction-only**: PC out-brakes SC at *all* capacities and no threshold appears. Cranking the backfire channel does not flip the sign, because PC's suppressive effect enters the caution term *before* the `(1 − caution)` multiplier while its curiosity backfire only enters the drive that the same multiplier then attenuates.
- In `linear` the PC advantage at least *shrinks* as capacity rises (margin falls with IC), which is the qualitative direction H1 expects — yet it still never crosses zero, so the threshold is absent there too.
- H1 only **falsifies** when PC's direct brake is removed (`pc_brake_off`): then PC acts solely through curiosity inflation, its low-capacity slope turns positive, and increasing caution *increases* intervention (the Alternative hypothesis).

**Implication.** Whether H1 holds is governed by a structural modelling choice — does phenomenological caution primarily *deter* or primarily *intrigue*? — far more than by interpretive capacity. The hypothesis's framing around a capacity threshold mislocates the real dependency.

## Limitations

- The behavioural equations are a stipulated functional form, not estimated from data. The spec's equations were ambiguous (written with `*` between every term); we render them as weighted linear combinations with caution as a multiplicative brake, because the literal product is degenerate and inverts the research question. The literal reading is reinstatable via config — results are conditional on this choice.
- No historical calibration. Priors are illustrative; the calibration hooks load files but ship with example scenarios only.
- Agents are independent draws with a static environment; there is no time dynamics, no social diffusion, no institutional decay process, and no feedback from one agent's intervention to another's.
- ‘Interpretive capacity’, ‘semantic clarity’ and ‘phenomenological caution’ are scalar abstractions on [0,1]; mapping them to real markers, languages, or monuments is outside the model.
- Sobol indices use uniform priors over the unit hypercube; under realistic correlated priors the importance ranking can change.
- Expected Harm uses a multiplicative severity term, so the model cannot distinguish ‘rare but catastrophic’ from ‘common but mild’ beyond their product.

## Artifacts

Figures and tables emitted by this run:

- `h1_margin.png`
- `heatmap_disturbance_rate_ic0.15.png`
- `heatmap_disturbance_rate_ic0.5.png`
- `heatmap_disturbance_rate_ic0.85.png`
- `heatmap_expected_harm_ic0.15.png`
- `heatmap_expected_harm_ic0.5.png`
- `heatmap_expected_harm_ic0.85.png`
- `interactions_expected_harm.png`
- `interactions_mean_intervention.png`
- `outcomes_interpretive_capacity.png`
- `outcomes_marker_clarity.png`
- `outcomes_phenomenological_caution.png`
- `phase_diagram.png`
- `sensitivity_tornado.png`
- `baseline_metrics.csv`, `heatmaps.csv`, `sweeps_1d.csv`, `outcome_distribution.csv`, `interactions.csv`, `h1_table.csv`, `sensitivity_ranking.csv`, `summary.json`
