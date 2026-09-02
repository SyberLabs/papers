# Toy Reference Pipeline

This directory contains a minimal executable reference path for the public TOK
package. It is intentionally small and synthetic. Its purpose is to show the
artifact discipline without exposing private benchmark data or implying that
the full internal TOK engine is included here.

Run:

```powershell
python examples\toy_reference_pipeline\run_toy_reference_pipeline.py --write
```

The script writes:

```text
examples/toy_reference_pipeline/generated/toy_reference_run_v0.json
```

The generated run demonstrates:

- narrated situation to candidate mechanism graph;
- candidate dynamics binding;
- dynamic divergence trace;
- reviewed observation registry;
- evidence-transition proposal;
- causal-status assessment;
- explicit refusal to update evidence, production state, or `wisdom_db`.

The public validator checks this example alongside the frozen public summaries:

```powershell
python tools\validate_public_tok_package.py
```
