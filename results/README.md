# Released results

These files are the unmodified outputs of the paper run (July 2026, NVIDIA L4).

| File | Rows | Description |
|---|---|---|
| sweep_results_v3.csv | 1,440 | one row per (subset, model, seed, dim, sev) evaluation cell |
| elasticity_v3.csv | 48 | slope and bootstrap CI per (subset, model, dim), raw and normalized |
| breakpoints_v3.csv | 48 | segmented-regression collapse points and SSE gains |
| c3_cross_dataset.csv | 2 | cross-dataset breakpoint consistency |
| c4_signatures.csv | 12 | architecture signature tests |
| interactions_v3.csv | 4 | pairwise interaction tests (8 seeds) |
| injection_points_v3.csv | 12 | train vs inference vs both study (per seed) |
| qcg_sweep_v3.csv | 96 | gating ablation raw evaluations |
| qcg_result_v3.csv | 4 | gating ablation per-dimension test |
| tables/ | | the paper tables as CSV, LaTeX, and HTML |

Column notes for `sweep_results_v3.csv`:

- `sev` is the ordinal severity, 0 is clean.
- `one_minus_q` is the degradation level used as the x-axis for elasticity.
- `rmse`, `score`, `ece`, `cov90` are the fidelity vector after sigma calibration.
- `rmse_base` is the clean RMSE of the same model and seed, used for normalization.

Regenerate everything downstream with `python scripts/analyze_results.py`.
