"""Publication figures. Every function takes dataframes and writes PDF + PNG."""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

TOL = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB", "#000000"]


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 11, "font.family": "serif", "mathtext.fontset": "cm",
        "text.usetex": False,
        "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _save(figdir, name):
    os.makedirs(figdir, exist_ok=True)
    plt.savefig(os.path.join(figdir, f"{name}.pdf"))
    plt.savefig(os.path.join(figdir, f"{name}.png"))
    plt.close()


def fig_dose_response(sweep: pd.DataFrame, subset: str, figdir: str):
    sub = sweep[sweep.subset == subset]
    dims = sorted(sub["dim"].unique())
    models = list(dict.fromkeys(sub["model"]))
    n_seeds = sub["seed"].nunique()
    fig, axes = plt.subplots(1, len(models), figsize=(4.0 * len(models), 3.9), sharey=True)
    axes = np.atleast_1d(axes)
    handles = []
    for ax, m in zip(axes, models):
        s2 = sub[sub.model == m]
        for i, d in enumerate(dims):
            gg = s2[s2.dim == d].groupby("one_minus_q")["rmse"].agg(["mean", "std"]).reset_index()
            ci = 1.96 * gg["std"] / np.sqrt(n_seeds)
            (line,) = ax.plot(gg["one_minus_q"], gg["mean"], "o-", ms=3.5, lw=1.4,
                              color=TOL[i % len(TOL)], label=d)
            ax.fill_between(gg["one_minus_q"], gg["mean"] - ci, gg["mean"] + ci,
                            color=TOL[i % len(TOL)], alpha=0.15, lw=0)
            if ax is axes[0]:
                handles.append(line)
        ax.set_title(f"{m} ({subset})")
        ax.set_xlabel(r"Degradation $(1-q)$")
    axes[0].set_ylabel("Test RMSE (cycles)")
    fig.legend(handles, dims, loc="lower center", ncol=4, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    _save(figdir, f"fig1_dose_response_{subset}")


def fig_elasticity(elast: pd.DataFrame, subset: str, figdir: str):
    e2 = elast[elast.subset == subset]
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.0))
    for ax, col, ttl, fmt in [(axes[0], "elasticity", rf"Raw $\eta$ ({subset})", ".1f"),
                              (axes[1], "elasticity_rel", rf"Normalized $\eta^{{rel}}$ ({subset})", ".2f")]:
        pivot = e2.pivot(index="model", columns="dim", values=col)
        sns.heatmap(pivot, annot=True, fmt=fmt, cmap="cividis", ax=ax, cbar_kws={"shrink": 0.9})
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
    fig.tight_layout()
    _save(figdir, f"fig2_elasticity_{subset}")


def fig_cross_dataset(sweep: pd.DataFrame, figdir: str, dim: str = "completeness_MNAR"):
    subsets = sweep["subset"].unique()
    models = list(dict.fromkeys(sweep["model"]))
    fig, axes = plt.subplots(1, len(models), figsize=(4.0 * len(models), 3.6))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        for k, sub in enumerate(subsets):
            g = sweep[(sweep.subset == sub) & (sweep.model == m) & (sweep.dim == dim)]
            gg = g.groupby("one_minus_q")["rmse"].agg(["mean", "std"]).reset_index()
            ax.plot(gg["one_minus_q"], gg["mean"] / g["rmse_base"].mean(), "o-", ms=3.5, lw=1.5,
                    color=TOL[k], label=sub)
        ax.set_title(m)
        ax.set_xlabel(r"Degradation $(1-q)$")
        ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    axes[0].set_ylabel(r"RMSE / RMSE$_{clean}$")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(figdir, "fig7_cross_dataset")


def fig_interactions(results, figdir: str):
    fig, axes = plt.subplots(1, len(results), figsize=(3.6 * len(results), 3.5))
    for ax, r in zip(np.atleast_1d(axes), results):
        df = r["per_seed"]
        means, stds = df[["dA", "dB", "dAB"]].mean(), df[["dA", "dB", "dAB"]].std()
        vals = [means["dA"], means["dB"], means["dA"] + means["dB"], means["dAB"]]
        errs = [stds["dA"], stds["dB"], np.sqrt(stds["dA"] ** 2 + stds["dB"] ** 2), stds["dAB"]]
        ax.bar([r"$\Delta_A$", r"$\Delta_B$", r"$\Delta_A{+}\Delta_B$", r"$\Delta_{AB}$"], vals,
               yerr=errs, capsize=3, color=[TOL[0], TOL[0], TOL[2], TOL[1]])
        a = r["dim_a"].replace("completeness_", "").replace("accuracy_", "")
        b = r["dim_b"].replace("completeness_", "").replace("accuracy_", "")
        ax.set_title(f"{a} $\\times$ {b}\n{r['model']} | $p_{{Holm}}$={r['p_holm']:.3f}", fontsize=9)
        ax.set_ylabel(r"$\Delta$RMSE (cycles)")
    fig.tight_layout()
    _save(figdir, "fig3_interactions")


def fig_injection_points(ip: pd.DataFrame, figdir: str):
    agg = ip.groupby(["dim", "sev"])[["d_inference", "d_train", "d_both"]].agg(["mean", "std"])
    cells = list(agg.index)
    xpos, width = np.arange(len(cells)), 0.25
    plt.figure(figsize=(1.7 * len(cells) + 2, 3.6))
    for k, (col, lab) in enumerate([("d_inference", "inference"), ("d_train", "train"),
                                    ("d_both", "both")]):
        plt.bar(xpos + (k - 1) * width, agg[(col, "mean")], width, yerr=agg[(col, "std")],
                capsize=3, color=TOL[k], label=lab)
    plt.xticks(xpos, [f"{d.replace('completeness_', '').replace('accuracy_', '')}\nsev {s}"
                      for d, s in cells], fontsize=8)
    plt.ylabel(r"$\Delta$RMSE vs clean (cycles)")
    plt.legend(frameon=False, fontsize=8, title="injection point")
    plt.tight_layout()
    _save(figdir, "fig8_injection_points")


def fig_gating(result: pd.DataFrame, figdir: str):
    x, width = np.arange(len(result)), 0.35
    plt.figure(figsize=(1.5 * len(result) + 2, 3.4))
    plt.bar(x - width / 2, result["eta_qcg_off"], width, color=TOL[6], label="gate off")
    plt.bar(x + width / 2, result["eta_qcg_on"], width, color=TOL[2], label="gate on")
    plt.xticks(x, [d.replace("completeness_", "").replace("accuracy_", "") for d in result["dim"]],
               fontsize=8)
    plt.ylabel("RMSE slope per severity step")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    _save(figdir, "fig9_qcg_ablation")


def fig_coverage_decay(sweep: pd.DataFrame, figdir: str):
    """Mean 90% coverage against severity, one line per subset. New in the repo."""
    plt.figure(figsize=(4.6, 3.4))
    for k, sub in enumerate(sweep["subset"].unique()):
        g = sweep[sweep.subset == sub].groupby("sev")["cov90"].mean()
        plt.plot(g.index, g.values, "o-", ms=4, lw=1.6, color=TOL[k], label=sub)
    plt.axhline(0.90, color="gray", ls="--", lw=1, label="nominal 0.90")
    plt.xlabel("Severity (ordinal 0-5)")
    plt.ylabel("Mean 90% interval coverage")
    plt.ylim(0.4, 1.0)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    _save(figdir, "fig10_coverage_decay")
