# Validation Protocol: Transition Zone Boundary

## Objective

Confirm that the Fourier regime boundary lies near p ~ 60 by running controlled
experiments at p = 61, 67, 71 and verifying that normal p^2/log(p)^2 scaling resumes.

## Hypothesis

If the three-regime model is correct:
- p = 61, 67, 71 should be in the **Fourier basin**
- Grokking times should **increase** with p (unlike transition zone)
- Fourier normalization should yield CV < 0.15

## Predictions

| p | Predicted tau (Fourier) | 95% CI |
|---|-------------------------|--------|
| 61 | 5262 | [4700, 5800] |
| 67 | 6068 | [5400, 6700] |
| 71 | 6631 | [5900, 7300] |

Using: tau = 23.9 * p^2 / (log(p)^2 * wd^0.65) with wd = 1.0

## Protocol

### Hardware/Software Requirements
- Same setup as local sweep runs
- Python 3.10+, PyTorch
- GPU recommended but not required for small p

### Training Configuration

Match the local sweep protocol exactly:
```python
config = {
    "task": "modular_addition",
    "optimizer": "adamw",
    "learning_rate": 0.001,
    "weight_decay": 1.0,
    "architecture": "mlp",
    "width": 256,
    "depth": 2,
    "max_epochs": 30000,  # Extended to avoid censoring
    "grokking_threshold": 95.0,
    "seed": 42,
}
```

### Runs to Execute

| Run ID | p | wd | max_epochs | Expected tau |
|--------|---|----|-----------:|-------------:|
| val_p61_wd1 | 61 | 1.0 | 30000 | ~5200 |
| val_p67_wd1 | 67 | 1.0 | 30000 | ~6100 |
| val_p71_wd1 | 71 | 1.0 | 30000 | ~6600 |

### Success Criteria

1. **No censoring**: All runs must reach 95% val_acc before max_epochs
2. **Increasing tau**: tau(61) < tau(67) < tau(71)
3. **Fourier scaling**: Combined with published data, CV < 0.15 at q=2
4. **Boundary confirmed**: tau(61) > tau(53) (transition ends)

### Failure Modes

| Outcome | Interpretation |
|---------|----------------|
| tau(61) < 4000 | Transition zone extends past p=60 |
| tau(61) > 7000 | Different factor at play |
| tau(67) < tau(61) | Still in transition zone |
| Any run censored | Increase max_epochs or investigate |

## Analysis Steps

### Step 1: Run experiments
```bash
python run_validation_sweep.py --moduli 61,67,71 --wd 1.0
```

### Step 2: Extract grokking times
```python
from analysis.log_exponent_collapse import load_scaling_data, compute_collapse_metric

# Load all data including new runs
points = load_scaling_data(Path("data/empirical_scaling_runs.csv"))

# Filter to Fourier regime (p >= 59)
fourier_points = [p for p in points if p.modulus >= 59]

# Compute CV at q=2
cv, C, std = compute_collapse_metric(fourier_points, q=2.0, beta=0.65)
print(f"Fourier regime CV: {cv:.4f}")
```

### Step 3: Update data file
Add validated runs to `data/empirical_scaling_runs.csv` with:
- `source`: `validation_sweep`
- `include_in_scaling_fit`: `True`
- `notes`: `Fourier regime validation point`

### Step 4: Regenerate collapse analysis
```bash
python analysis/log_exponent_collapse.py --output-dir analysis/figures/
```

## Expected Outcome

If validation succeeds:
- Fourier regime (p >= 59): 8 points, CV < 0.10, q = 2.0
- Transition zone (37 <= p <= 53): 5 points, excluded from Fourier fit
- Lookup basin (p <= 31): Censored/excluded

The scaling law claim becomes:
> tau ~ 23.9 * p^2 / (log(p)^2 * wd^0.65) for p > 60 (Fourier regime)

## Stretch Goals

If time permits, additional validation:

1. **Boundary sharpening**: Run p = 57, 59 to pinpoint transition edge
2. **Multi-seed**: Run 3 seeds per p to estimate variance
3. **Weight decay sweep**: At p = 61, sweep wd to confirm beta ~ 0.65
4. **Fourier concentration**: Extract hidden activations and compute DFT concentration

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Setup | 1 hour | Config files, scripts |
| Training | 6-12 hours | Raw traces |
| Analysis | 1 hour | Updated figures, CV metrics |
| Documentation | 30 min | Updated COLLAPSE_ANALYSIS_RESULTS.md |

## Contact

For questions about the protocol, see:
- `analysis/HOUND_DOG_FINDINGS.md` - Pattern discovery
- `analysis/ALGORITHMIC_BASIN_THEORY.md` - Theoretical framework
- `paper/grokking_effective_theory.tex` - Current paper draft
