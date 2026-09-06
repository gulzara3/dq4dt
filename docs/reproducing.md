# Reproducing the paper

There are three levels of reproduction, from seconds to hours.

## Level 1: recompute the analysis from released results (seconds, CPU)

The sweep CSV in `results/` is the raw output of the paper run. Everything downstream can be regenerated from it.

```bash
python scripts/analyze_results.py --sweep results/sweep_results_v3.csv
python scripts/make_figures.py --results results --out figures/regenerated
```

The first command prints the normalized elasticity tables, the cross-dataset breakpoint check (0.390 vs 0.388), the signature tests, and the coverage decay. It matches `results/elasticity_v3.csv` to machine precision.

## Level 2: smoke test the full pipeline (about an hour, one GPU)

```bash
python scripts/run_benchmark.py --root ./runs/quick --quick
```

This runs FD001 only, three seeds, three severities, and shortened training. It exercises every stage including download, training, the sweep, and all follow-up studies. Numbers will differ from the paper because of the reduced budget; the qualitative findings should still appear.

## Level 3: full benchmark (4 to 8 GPU hours)

```bash
python scripts/run_benchmark.py --root ./runs/full
```

Or from the original notebook: open `notebooks/DQ4DT_benchmark_v3_final.ipynb` in Google Colab on an L4 runtime and run all cells. The notebook persists to Google Drive and resumes after disconnects.

Outputs land in `./runs/full`:

| File | Content |
|---|---|
| sweep_results.csv | one row per evaluation cell |
| elasticity.csv | slopes with bootstrap CIs, raw and normalized |
| breakpoints.csv | segmented-regression collapse points |
| c3_cross_dataset.csv | cross-dataset consistency check |
| c4_signatures.csv | architecture signature tests |
| interactions.csv | pairwise interaction tests |
| injection_points.csv | train vs inference vs both study |
| qcg_sweep.csv, qcg_result.csv | gating ablation |
| checkpoints/ | every trained model (not committed) |

## Resuming

Every stage checkpoints. Models save after each epoch and finalize on completion. The sweep appends to its CSV after each (subset, model, seed) block. The injection-point study appends per row. Re-running the same command continues from the last completed unit. To force a redo, delete the relevant checkpoint or CSV rows.

## Hardware and numerical notes

The paper run used one NVIDIA L4, PyTorch 2.x, bfloat16 autocast, and cuDNN in non-deterministic mode for speed (results are averaged over seeds). On other hardware, expect RMSE differences at the second decimal. The elasticity orderings, the invariant collapse point, and the twelve signature directions are the claims that should replicate.

## Extending

- Add FD002 and FD003: set `Config(subsets=["FD001", "FD002", "FD003", "FD004"])`. Both are multi-regime and use the same preprocessing path as FD004.
- Add an operator, dataset, or model: see `CONTRIBUTING.md`.
