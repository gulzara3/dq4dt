# Methodology

This page states the definitions the code implements. Section numbers refer to the paper.

## Problem setup

A surrogate twin maps a sensor window X of length L over d channels to a predictive RUL distribution, f_theta: X -> p(RUL | X). A data-quality dimension j is degraded to quality q_j in [0, 1], where 1 is pristine. An injection operator O_j(.; s) realizes the degradation at ordinal severity s. The twin observes O_j(X; s) in place of X.

## Fidelity vector

phi = [RMSE, S_PHM08, ECE, cov_90]

- RMSE in cycles.
- S_PHM08 is the asymmetric score with denominators 13 (early) and 10 (late), so late predictions cost more.
- ECE is the quantile-based expected calibration error over ten levels from 0.05 to 0.95.
- cov_90 is the empirical coverage of the central 90% prediction interval.

Implementation: `dq4dt/metrics.py`.

## Degradation operators

Severity s in {1, ..., 5} maps linearly to a physical parameter:

| Dimension | Parameter | Range |
|---|---|---|
| accuracy | Gaussian noise std (normalized units) | 0.05 to 0.60 |
| accuracy_bias | same noise plus constant offset equal to the std | 0.05 to 0.60 |
| completeness_MCAR / MAR / MNAR | missing fraction, LOCF imputed | 0.05 to 0.50 |
| timeliness | delay in cycles, plus sample-and-hold factor | 1 to 8, 1 to 5 |
| consistency | offset and scale error on about half the channels | 0.1 to 1.0 |
| drift | linear ramp magnitude across the window | 0.1 to 1.0 |

MNAR masks values with probability increasing in the value itself, which makes the missingness non-ignorable. Each operator returns a per-channel quality estimate u in [0, 1]; the sweep records the batch mean as q.

Implementation: `dq4dt/injectors.py`.

## Elasticity and collapse

eta_j = d phi / d(1 - q_j), estimated as the least-squares slope of the dose-response curve. The normalized form divides by the clean RMSE. Confidence intervals come from a 500-resample bootstrap over seeds.

Collapse points come from a hinge regression y = a + b x + c max(x - brk, 0). A breakpoint is reported when the segmented fit reduces SSE by at least 20% relative to a linear fit.

Analytical bound: if the prediction depends on channel averages over L_eff samples, zero-mean noise perturbs the prediction by sigma / sqrt(L_eff) while a constant bias b passes through. The predicted ratio eta_bias / eta_noise is about sqrt(L_eff), which is 5.5 for L = 30.

Implementation: `dq4dt/analysis.py`, `dq4dt/stats.py`.

## Surrogates and training

Three architectures share a heteroscedastic Gaussian head (mu, log sigma^2): a two-layer LSTM (96 units), a two-layer transformer encoder (d_model 128, mean plus last-token pooling), and a dilated TCN with GroupNorm and physics penalties (RUL monotone non-increasing and non-negative, weight 0.01).

Training: AdamW, cosine schedule with 5% warmup, five MSE-only epochs then Gaussian NLL, early stopping with patience 15 and improvement margin 0.05 RMSE, gradient clipping at 1.0. After training, a single variance scale s^2 = mean(((y - mu) / sigma)^2) is fitted on validation and applied to all predicted sigmas.

Implementation: `dq4dt/models.py`, `dq4dt/training.py`.

## Statistical protocol

- Five seeds per (subset, architecture) in the main sweep.
- Interaction tests use eight seeds, one-sided Wilcoxon in the observed direction, Holm correction within the family, and matched-pairs rank-biserial effect sizes.
- Architecture signatures are discovered on FD001, pre-registered, and confirmed on FD004. With five seeds the per-test Wilcoxon floor is 0.031, so the primary confirmatory statistic is the aggregate sign test.
- The main sweep has 2 x 3 x 5 x 8 x 6 = 1,440 cells.

## Monitor

The live monitor estimates completeness (1 minus NaN fraction, before imputation), accuracy (variance ratio to training statistics), and timeliness (fraction of exact repeats) from a raw window, and returns

c = 1 - sum_j w_j (1 - q_hat_j),  w_j = |eta_rel_j| / sum_k |eta_rel_k|

Implementation: `dq4dt/monitor.py`.
