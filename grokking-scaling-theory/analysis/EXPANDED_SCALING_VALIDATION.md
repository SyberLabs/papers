# Expanded Scaling Validation Results

## Dataset Expansion

- Total points: 21
- Measured points: 19
- Group counts: {'local_mlp': 8, 'local_residual': 6, 'published_mlp': 7}
- Published group retains earlier anchors, including the two extrapolated low-modulus points.
- Local groups are trace-backed measurements extracted from actual grokking runs.

## Joint (q, beta) Optimization

- Best q: 3.00
- Best beta: 0.68
- In-sample log-RMSE: 0.4507
- Mean absolute percent error: 33.24%
- Max absolute percent error: 94.15%
- Normalized collapse CV: 0.5286

### Group Calibration Constants

- local_mlp: C = 23.68
- local_residual: C = 24.55
- published_mlp: C = 98.27

## Subset Fits

| Subset | n | q | beta | In-sample MAE | LOO MAE |
|---|---:|---:|---:|---:|---:|
| published_all | 7 | 1.95 | 0.64 | 4.43% | 8.79% |
| published_measured_only | 5 | 1.58 | 0.63 | 4.67% | 14.16% |
| local_mlp | 8 | 3.00 | 0.73 | 45.16% | 58.95% |
| local_residual | 6 | 3.00 | 1.00 | 44.25% | 60.87% |

## Bootstrap Uncertainty (120 stratified resamples)

- q median [3.00, 3.00] with median 3.00
- beta median [0.60, 0.97] with median 0.67

## Leave-One-Out Cross-Validation

- Mean absolute percent error: 41.32%
- Median fitted q across folds: 3.00
- Median fitted beta across folds: 0.68
- q range across folds: [3.00, 3.00]
- beta range across folds: [0.67, 0.75]

## Architecture Sweep

### mlp

- Points: 8
- Best q: 3.00
- Best beta: 0.73
- In-sample MAE: 45.16%
- LOO MAE: 58.95%

### residual

- Points: 6
- Best q: 3.00
- Best beta: 1.00
- In-sample MAE: 44.25%
- LOO MAE: 60.87%

## Interpretation

- The denser mixed dataset lets us test the exponent pair jointly rather than fixing beta by hand.
- The pooled fit is intentionally harsh: if a single shared exponent pair does not survive the mixed dataset, that is evidence against naive universality.
- The published MLP anchor set still prefers q near 2 with low error, but the local MLP and residual ladders do not yet exhibit the same asymptotic regime.
- The architecture split should therefore be read as exploratory and currently points to regime dependence rather than established universality.

