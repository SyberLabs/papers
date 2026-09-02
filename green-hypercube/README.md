# Green Hypercube

Coverage-aware search benchmarks in computational ethnobotany.

Green Hypercube is a research software package for testing how structured search
strategies behave in high-dimensional plant-data landscapes. It combines
phylogeny, chemistry/sensory salience, spatial co-occurrence, and animal
association cues, then asks whether those cues help discover hidden reward under
explicit sparsity, coverage, and negative-control checks.

The central caution of the project is that biodiversity, phytochemistry, and
ethnobotanical databases often measure research attention as much as biological
structure. Green Hypercube is built to separate raw cue-reward coupling from
coverage artifacts before crediting a search strategy with real signal.

## What This Is

- A reproducible Python package for sequential search over plant manifolds.
- A benchmark harness for random, phylogenetic, sensory, ecological, social, and
  integrative search strategies.
- A validation suite with reward permutation, graph rewiring, phylogeny shuffles,
  coupling tests, residualization, and matched-density sweeps.
- A research artifact for studying database bias, reward sparsity, and
  reproducibility in computational ethnobotany.

## What This Is Not

- It is not a claim about Indigenous discovery mechanisms or representative
  community knowledge.
- It is not a bioprospecting recommendation engine.
- It does not treat documented ethnobotanical use as a direct proxy for latent
  biological value.
- It does not publish raw external database caches or generated live-data result
  bundles.

Where documented-use datasets are involved, they are treated as limited,
aggregated, historically mediated records. The analysis is framed around coverage
artifacts and claim boundaries, not around extracting or validating community
knowledge.

## Model

Each plant species is represented as a point in a manifold. Cue channels are
available to strategies before testing; reward is hidden until a strategy spends
budget on a species.

| Face | Source family | Role |
| --- | --- | --- |
| Phylogeny | Open Tree of Life | cue |
| Chemistry / sensory salience | Dr. Duke's-style phytochemistry tables | cue |
| Spatial co-occurrence | GBIF-style occurrence records | cue |
| Animal associations | GloBI-style interaction records | cue |
| Hidden reward | synthetic assay, ChEMBL potency, or documented-use labels depending on config | target |

The synthetic default path requires no network access and is intended for tests,
smoke runs, and method inspection. Live-data configurations are included, but they
rebuild caches locally and depend on external services.

## Strategies

1. `random` - unstructured baseline.
2. `phylogenetic` - follows clade proximity around discovered hits.
3. `sensory` - follows chemistry/sensory salience.
4. `ecological` - follows co-occurrence structure.
5. `social` - shares discoveries over a social graph.
6. `cultural` - integrates multiple cue channels with online reweighting.

## Quickstart

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS/Linux

pip install -r requirements.txt
pip install -e .
pytest
```

Run the offline synthetic comparison:

```bash
python -m greenhypercube run --config configs/default.yaml
```

Inspect the assembled manifold:

```bash
python -m greenhypercube info --config configs/default.yaml
```

Run negative controls:

```bash
python -m greenhypercube controls --config configs/default.yaml --output results/controls
```

Run a coupling check:

```bash
python -m greenhypercube coupling --config configs/default.yaml --output results/coupling
```

## Reproducibility Surface

The repository includes:

```text
greenhypercube/
  config.py            pydantic config schema and YAML loader
  data/                data adapters and normalized schemas
  hypercube/           phylogeny, manifold, and builder code
  strategies/          search strategies and registry
  simulation/          environment, episode engine, metrics, study runner
  validation/          controls, coupling, residualization, M2 ladder, phylo community
  analysis/            figure helpers
  cli.py               Typer CLI

configs/               offline and live-data configuration files
experiments/           headline comparison config
tests/                 pytest suite for pipeline, strategies, controls, coupling, and metrics
```

Generated caches, raw external source data, virtual environments, and live result
bundles are intentionally excluded from version control. Rebuild them from the
configs when needed.

## Claim Boundary

The strongest public result is methodological: apparent structure in integrated
plant-data benchmarks can be inflated by coverage and reward-density artifacts.
Green Hypercube therefore reports raw coupling beside controlled and residualized
coupling, and treats matched-density comparisons as necessary before interpreting
cross-pool strategy advantage.

The code is research software, not a production system. It is designed to make
assumptions inspectable and failure modes visible.
