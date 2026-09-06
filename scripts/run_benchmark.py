#!/usr/bin/env python
"""Run the full DQ4DT benchmark end to end.

Usage:
    python scripts/run_benchmark.py --root ./runs/full
    python scripts/run_benchmark.py --root ./runs/quick --quick

Everything is resumable. Re-run the same command after an interruption and it
picks up where it stopped.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dq4dt.analysis import (breakpoint_table, compute_elasticity,  # noqa: E402
                            cross_dataset_consistency, gating_ablation,
                            injection_point_study, run_interactions,
                            signature_confirmation)
from dq4dt.config import DEFAULT_CONFIG, QUICK_CONFIG  # noqa: E402
from dq4dt.data import build_subset, download_cmapss  # noqa: E402
from dq4dt.seeding import get_device, set_determinism  # noqa: E402
from dq4dt.sweep import run_sweep  # noqa: E402
from dq4dt.training import ModelCache  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./runs/full", help="output directory")
    ap.add_argument("--data", default="./data", help="where C-MAPSS is cached")
    ap.add_argument("--quick", action="store_true", help="smoke-test configuration")
    ap.add_argument("--skip-studies", action="store_true",
                    help="run only the main sweep and elasticity analysis")
    args = ap.parse_args()

    cfg = QUICK_CONFIG if args.quick else DEFAULT_CONFIG
    os.makedirs(args.root, exist_ok=True)
    ckpt_dir = os.path.join(args.root, "checkpoints")
    device = get_device()
    print(f"device: {device} | cells: {cfg.n_cells} | subsets: {cfg.subsets}")
    set_determinism(cfg.global_seed)

    cmapss = download_cmapss(args.data)
    datasets = {s: build_subset(s, cmapss, cfg.window, cfg.stride, cfg.rul_cap,
                                cfg.n_regimes, cfg.val_fraction, cfg.global_seed)
                for s in cfg.subsets}
    cache = ModelCache(cfg, ckpt_dir, device)

    sweep = run_sweep(cfg, datasets, cache, os.path.join(args.root, "sweep_results.csv"))
    elast = compute_elasticity(sweep, n_boot=cfg.bootstrap_resamples, seed=cfg.global_seed)
    elast.to_csv(os.path.join(args.root, "elasticity.csv"), index=False)
    breaks = breakpoint_table(sweep, cfg.breakpoint_min_gain)
    breaks.to_csv(os.path.join(args.root, "breakpoints.csv"), index=False)
    c3 = cross_dataset_consistency(breaks, tol=cfg.cross_dataset_tolerance)
    c3.to_csv(os.path.join(args.root, "c3_cross_dataset.csv"), index=False)
    print("\nCross-dataset consistency:\n", c3.to_string(index=False) if len(c3) else "n/a")

    c4 = pd.concat([signature_confirmation(sweep, s, cfg.seeds) for s in cfg.subsets])
    c4.to_csv(os.path.join(args.root, "c4_signatures.csv"), index=False)
    print("\nSignature tests:\n", c4.round(4).to_string(index=False))

    if args.skip_studies:
        return
    primary = datasets[cfg.subsets[0]]
    inter = run_interactions(primary, cache, cfg)
    pd.DataFrame([{k: v for k, v in r.items() if k != "per_seed"} for r in inter]) \
        .to_csv(os.path.join(args.root, "interactions.csv"), index=False)
    injection_point_study(primary, cache, cfg, os.path.join(args.root, "injection_points.csv"))
    qsweep, qres = gating_ablation(primary, cache, cfg)
    qsweep.to_csv(os.path.join(args.root, "qcg_sweep.csv"), index=False)
    qres.to_csv(os.path.join(args.root, "qcg_result.csv"), index=False)
    print("\nDone. Results in", args.root)


if __name__ == "__main__":
    main()
