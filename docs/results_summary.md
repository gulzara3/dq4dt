# Results summary

All numbers below come from the CSVs in `results/`. See the paper for interpretation.

## Clean baselines (seed 42, after sigma calibration)

| Subset | Model | RMSE | PHM08 | ECE pre -> post | cov90 pre -> post |
|---|---|---|---|---|---|
| FD001 | LSTM | 16.00 | 330 | 0.101 -> 0.080 | 0.68 -> 0.88 |
| FD001 | Transformer | 16.67 | 623 | 0.085 -> 0.065 | 0.79 -> 0.85 |
| FD001 | Physics TCN | 13.33 | 243 | 0.072 -> 0.081 | 0.66 -> 0.85 |
| FD004 | LSTM | 15.32 | 1334 | 0.060 -> 0.141 | 0.73 -> 0.99 |
| FD004 | Transformer | 15.75 | 1175 | 0.039 -> 0.070 | 0.93 -> 0.96 |
| FD004 | Physics TCN | 15.85 | 1143 | 0.105 -> 0.081 | 0.78 -> 0.94 |

## Normalized elasticity (higher is more fragile)

| Dimension | FD001 LSTM | FD001 TCN | FD001 Transf. | FD004 LSTM | FD004 TCN | FD004 Transf. |
|---|---|---|---|---|---|---|
| accuracy | 0.32 | 0.63 | 0.39 | 0.43 | 0.58 | 0.72 |
| accuracy_bias | 2.22 | 1.69 | 2.52 | 1.26 | 0.81 | 1.60 |
| completeness_MCAR | 0.35 | 0.41 | 0.18 | 0.33 | 0.42 | 0.29 |
| completeness_MAR | 0.40 | 0.56 | 0.20 | 0.42 | 0.40 | 0.19 |
| completeness_MNAR | 1.44 | 1.83 | 1.52 | 1.40 | 1.41 | 1.54 |
| timeliness | 0.54 | 0.70 | 0.34 | 0.99 | 0.92 | 0.67 |
| consistency | 2.20 | 0.98 | 1.70 | 3.19 | 1.33 | 3.31 |
| drift | 2.04 | 2.24 | 0.94 | 3.48 | 3.59 | 2.42 |

## Collapse points

| Dimension | FD001 | FD004 | Spread |
|---|---|---|---|
| completeness_MNAR | 0.390 | 0.388 | 0.002 |
| accuracy_bias | 0.188 | 0.234 | 0.046 |

## Coverage decay (mean 90% coverage by severity)

| Subset | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| FD001 | 0.86 | 0.86 | 0.78 | 0.73 | 0.65 | 0.58 |
| FD004 | 0.95 | 0.95 | 0.90 | 0.85 | 0.79 | 0.70 |

## Interactions (FD001, n = 8 seeds)

| Model | Pair | Gap (cycles) | p raw | p Holm | r |
|---|---|---|---|---|---|
| LSTM | MNAR x timeliness | -2.77 | 0.0039 | 0.0156 | -1.00 |
| Transformer | MNAR x timeliness | -0.12 | 0.527 | 0.527 | 0.00 |
| LSTM | bias x consistency | +7.59 | 0.039 | 0.117 | +0.72 |
| Transformer | bias x consistency | +2.41 | 0.191 | 0.383 | +0.39 |

## Injection point (LSTM, FD001, delta RMSE vs clean)

| Dimension | Severity | Inference | Train | Both |
|---|---|---|---|---|
| accuracy_bias | 3 | +6.1 | +10.1 | +6.2 |
| accuracy_bias | 5 | +16.8 | +9.1 | +5.7 |
| completeness_MNAR | 3 | +1.1 | -1.7 | -0.9 |
| completeness_MNAR | 5 | +12.5 | +16.4 | +17.2 |

## Gating ablation (all Holm p = 1.0)

| Dimension | slope off | slope on | delta | p raw |
|---|---|---|---|---|
| accuracy_bias | 0.41 | 0.27 | -0.14 | 0.25 |
| completeness_MNAR | 0.95 | 1.12 | +0.17 | 0.875 |
| consistency | 0.70 | 0.56 | -0.14 | 0.375 |
| drift | 0.02 | 0.06 | +0.04 | 0.875 |
