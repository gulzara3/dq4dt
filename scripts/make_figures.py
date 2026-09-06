#!/usr/bin/env python
"""Regenerate the figures that depend only on CSV results.

    python scripts/make_figures.py --results results --out figures/regenerated
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dq4dt import plotting  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures/regenerated")
    args = ap.parse_args()
    plotting.set_style()

    sweep = pd.read_csv(os.path.join(args.results, "sweep_results_v3.csv"))
    elast = pd.read_csv(os.path.join(args.results, "elasticity_v3.csv"))
    for sub in sweep["subset"].unique():
        plotting.fig_dose_response(sweep, sub, args.out)
        plotting.fig_elasticity(elast, sub, args.out)
    if sweep["subset"].nunique() >= 2:
        plotting.fig_cross_dataset(sweep, args.out)
    plotting.fig_coverage_decay(sweep, args.out)

    ip_path = os.path.join(args.results, "injection_points_v3.csv")
    if os.path.exists(ip_path):
        plotting.fig_injection_points(pd.read_csv(ip_path), args.out)
    q_path = os.path.join(args.results, "qcg_result_v3.csv")
    if os.path.exists(q_path):
        plotting.fig_gating(pd.read_csv(q_path), args.out)
    print("figures written to", args.out)


if __name__ == "__main__":
    main()
