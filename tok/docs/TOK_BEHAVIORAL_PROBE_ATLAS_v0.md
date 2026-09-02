# TOK Behavioral Probe Atlas v0

**Status:** research-only local CPU lane
**Production effect:** `none_research_only`

## Question

When a dynamical system is repeatedly perturbed, what response motifs emerge,
and which structural hypotheses could those motifs discriminate between?

## Design

Each intervention-capable corpus system receives repeated normalized negative
state displacements. Every pulse is compared with a cloned, unperturbed
counterfactual twin from the same pre-pulse state. For stochastic systems, the
clone preserves random-generator state so both paths receive matched future
noise.

The retained observable bundle is:

- peak displacement
- integrated deviation
- endpoint residual
- target endpoint shift
- recovery duration
- exploratory phase-lag proxy
- treatment stability
- counterfactual stability

The deterministic classifier records:

- stable response
- habituation
- sensitization
- discrete-transition candidate
- non-monotonic adaptation
- insufficient recovery
- perturbation destabilization
- baseline instability

These are assay outputs, not automatic claims of learning, causation, or template
confirmation.

The v0 pilot uses a fixed simulation window (`dt * interval_steps`) after each
pulse. That makes the local assay simple and reproducible, but it is not yet a
cross-system natural-timescale normalization. Response-window sensitivity is the
next calibration axis. See `TOK_BEHAVIORAL_PROBE_WINDOW_CALIBRATION_v0.md`.

## Corpus Boundary

The canonical twelve-system registry is retained. Ten synthetic systems can
receive controlled interventions. Hudson Bay and Gause remain explicit
`observational_only` records because historical data cannot be experimentally
pulsed.

## Local Run

```bash
python -m research.behavioral_probe_atlas_v0.test_behavioral_probe_atlas_v0
python -m research.behavioral_probe_atlas_v0.run_local
```

## Optional CPU Cluster Run

```bash
sbatch research/behavioral_probe_atlas_v0/run_cpu_sbatch.sh
```

The cluster launcher widens amplitudes, seeds, pulse count, and response windows.
It remains numpy-only and API-free.
