# Vital Language: Experimental Systems for Coherence, Agency, and Literary Vitality

Vital Language is a research software project investigating whether language-model prose can be made to feel more alive without collapsing coherence. The project began with a falsifiable hypothesis about chaotic logit modulation and evolved into a more disciplined result: at small-model scale, most apparent "vitality" metrics are really detecting collapse-versus-coherence, while stronger structure seems to live above the token layer.

The system is not a productized writing tool and does not claim to solve literary generation. It is an empirical harness for testing interventions, measuring failure modes, and separating attractive stories from measured effects.

## Research Question

Can controlled generation dynamics increase the structural vitality of prose while preserving coherence?

The first bet was:

1. Inject deterministic chaotic modulation into token logits.
2. Increase multifractal structure in generated text.
3. Improve perceived vitality without degeneracy.

The evidence forced a narrower and more useful conclusion: token-level chaos can create measurable signal changes, but it does not reliably outperform ordinary sampling on meaningful vitality proxies. Prompt-level agency scaffolding is more reliable for preventing collapse, and literary vitality appears tied to higher-level structural discipline rather than raw entropy, randomness, or stylistic imitation.

## What Is Built

- A shared generation harness for baseline sampling, prompt-only controls, chaotic logit modulation, matched-noise controls, and white-noise controls.
- Chaotic and stochastic modulation signals, including Lorenz, Henon, logistic, multifrequency, white noise, and matched Ornstein-Uhlenbeck controls.
- A scale-relative logit injector with masking modes for protecting high-confidence logits.
- Metrics for surface diversity, MFDFA spectrum width, coherence, recurrence, semantic advance, phonetic structure, and stream-of-consciousness features.
- A blind rating study scaffold for collecting human judgments separately from hidden conditions.
- Analysis scripts for epsilon sweeps, long runs, literary benchmarks, agency tests, and frontier-model imitation probes.

## Core Findings

The results are deliberately framed as interim and conditional.

- **Temporal structure matters for coherence.** White-noise perturbation destroys coherence much faster than structured chaotic modulation at the same magnitude.
- **Chaos beats a matched-noise control on some surprisal-width measurements, but not plain sampling.** This is a real control result, not a practical win for vitality.
- **Many "vitality" metrics were degeneracy detectors.** Width, recurrence, advance, and some literary metrics looked meaningful until collapsed passages were separated from clean passages.
- **Agency scaffolding is the most reliable lever found so far.** A persistent-speaker scaffold reduced collapse, but did not by itself create stronger vitality.
- **The literary branch suggests vitality is structural discipline, not just register.** Frontier models can imitate recognizable style, but the harder signals appear to involve constraints such as held image-fields, concrete/abstract oscillation, transformed return, and allusive pointing.

## Repository Structure

```text
vitality/
  generation/      model loading, decode harness, conditions
  modulation/      chaotic and stochastic signals, logit injector
  metrics/         MFDFA, recurrence, coherence, SOC features, phonetics
  study/           blind rating study builders and CLI helpers

scripts/           experiment runners and analysis scripts
configs/           generation and modulation settings
prompts/           prompt sets for experiments
tests/             fast non-model checks for core components
docs/              findings and theory notes
study/             browser-based rating interface
experiments/       frontier imitation probe summaries and analysis scripts
outputs/selected/  selected public pilot plots and summaries
```

## Quick Start

Install the lightweight dependencies needed for the non-model tests:

```bash
python -m pip install -r requirements.txt
python tests/test_core.py
```

Run a small model smoke experiment after installing the optional model dependencies:

```bash
python scripts/run_experiment.py --config configs/default.yaml --smoke
```

The default config uses `Qwen/Qwen2.5-0.5B-Instruct` on CPU. Full model runs require downloading model weights through Hugging Face and can take significant time.

## Selected Artifacts

- [Interim findings](docs/FINDINGS.md)
- [Living-language feature specification](docs/LIVING_LANGUAGE_SPEC.md)
- [Blind rating study protocol](study/README.md)
- [Frontier imitation results](experiments/frontier_imitation/RESULTS.md)
- [Selected epsilon sweep plot](outputs/selected/structure_vs_eps.png)
- [Chaos vs matched-control plot](outputs/selected/chaos_vs_matched.png)

## Public Release Boundaries

This public package intentionally excludes:

- raw generated-output dumps,
- private condition keys and rating files,
- copyrighted literary source text,
- hidden probe answer keys,
- local cache files and bytecode.

The included results should be read as an empirical research trail, not a finished scientific claim. The strongest current claim is methodological: the project built falsifiable controls and used them to reject several tempting explanations.

## Current Status

Vital Language is best presented as an exploratory research system with a high standard for self-correction. Its most portfolio-relevant value is the combination of experimental design, metric construction, falsification controls, and honest narrowing of claims after the evidence did not support the original story.
