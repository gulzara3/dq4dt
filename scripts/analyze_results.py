#!/usr/bin/env python
"""Recompute elasticity, breakpoints, and cross-dataset consistency from a
sweep CSV. Runs on CPU in seconds. Defaults to the released results.

    python scripts/analyze_results.py --sweep results/sweep_results_v3.csv
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dq4dt.analysis import (breakpoint_table, compute_elasticity,  # noqa: E402
                            cross_dataset_consistency, signature_confirmation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="results/sweep_results_v3.csv")
    ap.add_argument("--out", default="results/recomputed")
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sweep = pd.read_csv(args.sweep)
    elast = compute_elasticity(sweep, n_boot=args.boot)
    breaks = breakpoint_table(sweep)
    c3 = cross_dataset_consistency(breaks)
    seeds = sorted(sweep["seed"].unique())
    c4 = pd.concat([signature_confirmation(sweep, s, seeds) for s in sweep["subset"].unique()])

    elast.to_csv(os.path.join(args.out, "elasticity.csv"), index=False)
    breaks.to_csv(os.path.join(args.out, "breakpoints.csv"), index=False)
    c3.to_csv(os.path.join(args.out, "c3_cross_dataset.csv"), index=False)
    c4.to_csv(os.path.join(args.out, "c4_signatures.csv"), index=False)

    for sub in sweep["subset"].unique():
        print(f"\n== normalized elasticity, {sub} ==")
        print(elast[elast.subset == sub].pivot(index="model", columns="dim",
                                               values="elasticity_rel").round(2).to_string())
    print("\n== cross-dataset breakpoint consistency ==")
    print(c3.to_string(index=False))
    print("\n== coverage decay (mean cov90 by severity) ==")
    print(sweep.groupby(["subset", "sev"])["cov90"].mean().round(2).unstack().to_string())


if __name__ == "__main__":
    main()
