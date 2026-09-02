# DTBR-MC: Deep-Time Behavioral Risk Monte Carlo

DTBR-MC is a falsification-oriented Monte Carlo simulator for studying
deep-time hazard communication: how a future society might encounter, interpret,
and disturb a buried hazard when language, institutions, and interpretive
capacity cannot be assumed stable.

The simulator is not a forecast of future civilizations. It is a consistency
auditor for modelling assumptions around long-term warning design.

## Research Question

The project began with a representative nuclear-semiotics hypothesis:

> Below a threshold of interpretive capacity, phenomenological caution reduces
> intrusion more effectively than semantic clarity.

The model was built to search for the failure of that hypothesis, not to confirm
it. The resulting paper argues that the original symbolic framing is second-order:
physical substrate, access, hazard severity, and certainty-of-consequence gates
dominate the communicative marker.

## Core Result

The strongest defensible claim is narrow:

> In this model family, symbolic factors act as modulators, while physical
> factors act as gates. A marker can modulate behavior inside the feasible space,
> but it cannot move outcomes past physical access, capability, severity, or
> certainty ceilings.

This does not prove how distant-future people will behave. It shows which
assumptions are load-bearing when the warning-design problem is made explicit.

## Architecture

```text
dtbr_mc/
  config/schemas.py     pydantic configs for priors and behavior weights
  distributions.py      unit-interval sampling and correlated traits
  agents.py             heterogeneous receiver population sampler
  environment.py        site/environment sampler
  behavior.py           registered behavior models
  simulation.py         seeded vectorized simulation
  metrics.py            expected harm, rates, bootstrap CIs
  experiments.py        H1, sensitivity, and report-generation experiments
  experiments_h3.py     funnel, certainty ceiling, coupling, and H3 variants
  visualization.py      figures and diagnostic plots
  main.py               Typer CLI
examples/               illustrative priors and scenarios
tests/                  regression and falsifiability tests
```

## Epistemic Labels

Every major result is treated with a status label:

- `AUDIT`: a consequence of model construction.
- `EXTRAPOLATION`: conditional on contemporary empirical primitives.
- `CARTOGRAPHY`: a reachability or regime result, not an actuality claim.
- `EXPLORATORY`: post-amendment or not confirmatory.

This is the main discipline of the project: the simulator can expose structure,
but it cannot launder stipulated assumptions into empirical truth.

## Included Artifacts

- [PAPER.md](PAPER.md): working paper draft.
- [SPEC_H3.md](SPEC_H3.md): H3/funnel specification and amendments.
- `dtbr_mc/`: simulator source.
- `tests/`: deterministic regression and falsifiability tests.
- `examples/`: illustrative, non-calibrated scenario inputs.
- `experiment_001_outputs/`: representative outputs and figures from the first
  experiment bundle.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
pip install pytest
```

Run tests:

```bash
python -m pytest tests -q
```

Run a small demo:

```bash
python -m dtbr_mc.main demo --outdir outputs_demo
```

Run the first experiment bundle:

```bash
python -m dtbr_mc.main experiment001 --n-agents 100000 --outdir outputs
```

## Boundaries

DTBR-MC does not provide operational nuclear-waste policy, empirical prediction,
or calibrated future-behavior estimates. Its value is methodological: it turns a
symbolic hypothesis into replaceable equations, shows where the hypothesis fails,
and records which conclusions are audit, extrapolation, or cartography.
