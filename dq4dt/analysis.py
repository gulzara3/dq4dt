"""Post-sweep analysis.

Everything here works on the sweep CSV (plus a ModelCache for the studies
that need fresh evaluations). Functions are split so that elasticity and
breakpoint analysis can be run on the released results without a GPU.
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from .config import Config
from .injectors import MIX_DIMS, degrade, mixed_degrade
from .stats import fit_breakpoint, holm, rank_biserial
from .training import ModelCache, evaluate, make_loader


# ---------------------------------------------------------------- elasticity
def compute_elasticity(df: pd.DataFrame, fidelity: str = "rmse", n_boot: int = 500,
                       seed: int = 42) -> pd.DataFrame:
    """Slope of the dose-response per (subset, model, dim) with bootstrap CIs."""
    rows = []
    rng = np.random.default_rng(seed)
    for (sub, m, d), g in df.groupby(["subset", "model", "dim"]):
        agg = g.groupby("one_minus_q")[fidelity].mean().reset_index()
        if len(agg) < 2:
            continue
        slope = np.polyfit(agg["one_minus_q"], agg[fidelity], 1)[0]
        base = g["rmse_base"].mean()
        seeds = g["seed"].unique()
        boots = []
        for _ in range(n_boot):
            samp = rng.choice(seeds, len(seeds), replace=True)
            s2 = (pd.concat([g[g.seed == s] for s in samp])
                  .groupby("one_minus_q")[fidelity].mean().reset_index())
            if len(s2) >= 2:
                boots.append(np.polyfit(s2["one_minus_q"], s2[fidelity], 1)[0])
        lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (slope, slope)
        rows.append(dict(subset=sub, model=m, dim=d, elasticity=slope, ci_lo=lo, ci_hi=hi,
                         elasticity_rel=slope / base, rel_ci_lo=lo / base, rel_ci_hi=hi / base))
    return pd.DataFrame(rows).sort_values(["subset", "dim", "model"])


def breakpoint_table(df: pd.DataFrame, min_gain: float = 0.2) -> pd.DataFrame:
    rows = []
    for (sub, m, d), g in df.groupby(["subset", "model", "dim"]):
        agg = g.groupby("one_minus_q")["rmse"].mean().reset_index()
        if len(agg) < 4:
            continue
        brk, sse_seg, sse_lin = fit_breakpoint(agg["one_minus_q"], agg["rmse"])
        gain = 1.0 - sse_seg / (sse_lin + 1e-12)
        rows.append(dict(subset=sub, model=m, dim=d,
                         breakpoint=(round(brk, 3) if (brk is not None and gain >= min_gain)
                                     else np.nan),
                         sse_gain=round(gain, 2)))
    return pd.DataFrame(rows).sort_values(["dim", "subset", "model"])


def cross_dataset_consistency(breaks: pd.DataFrame, dims=("completeness_MNAR", "accuracy_bias"),
                              tol: float = 0.10) -> pd.DataFrame:
    rows = []
    for dim in dims:
        sub_means = (breaks[breaks.dim == dim].dropna(subset=["breakpoint"])
                     .groupby("subset")["breakpoint"].mean())
        if len(sub_means) >= 2:
            spread = float(sub_means.max() - sub_means.min())
            rows.append(dict(dim=dim, **{f"brk_{s}": round(v, 3) for s, v in sub_means.items()},
                             spread=round(spread, 3), consistent=bool(spread <= tol)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- interactions
def _chained(dims: List[str], sev: int):
    def fn(xb, g):
        u = torch.ones(xb.shape[0], xb.shape[2], device=xb.device)
        for d in dims:
            xb, ui = degrade(xb, d, sev, g)
            u = torch.minimum(u, ui)
        return xb, u
    return fn


def interaction_test(model_name: str, dim_a: str, dim_b: str, data: Dict, cache: ModelCache,
                     cfg: Config, seeds: List[int], sev: int = 5) -> Dict:
    device = cache.device
    te = make_loader(data["Xte"], data["yte"], cfg.batch_size, False, cfg.global_seed,
                     2 if device.type == "cuda" else 0)
    per_seed = []
    for seed in seeds:
        mdl, meta = cache.get(model_name, data, seed=seed, verbose=False)
        ss = meta["sigma_scale"]
        base = evaluate(mdl, te, sigma_scale=ss, device=device)["rmse"]
        gen = torch.Generator(device=device)
        gen.manual_seed(7 + seed)
        d_a = evaluate(mdl, te, _chained([dim_a], sev), gen, ss, device)["rmse"] - base
        d_b = evaluate(mdl, te, _chained([dim_b], sev), gen, ss, device)["rmse"] - base
        d_ab = evaluate(mdl, te, _chained([dim_a, dim_b], sev), gen, ss, device)["rmse"] - base
        per_seed.append(dict(seed=seed, dA=d_a, dB=d_b, dAB=d_ab, gap=d_ab - (d_a + d_b)))
    df = pd.DataFrame(per_seed)
    gaps = df["gap"].values
    direction = "greater" if np.mean(gaps) > 0 else "less"
    p = (wilcoxon(gaps, alternative=direction).pvalue
         if len(gaps) >= 3 and np.any(gaps != 0) else np.nan)
    return dict(subset=data["subset"], model=model_name, dim_a=dim_a, dim_b=dim_b,
                mean_gap=float(np.mean(gaps)), p_raw=float(p), r_rb=rank_biserial(gaps),
                verdict="SUPER-additive" if np.mean(gaps) > 0 else "SUB-additive",
                per_seed=df)


def run_interactions(data: Dict, cache: ModelCache, cfg: Config) -> List[Dict]:
    pairs = [("lstm", "completeness_MNAR", "timeliness"),
             ("transformer", "completeness_MNAR", "timeliness"),
             ("lstm", "accuracy_bias", "consistency"),
             ("transformer", "accuracy_bias", "consistency")]
    results = [interaction_test(m, a, b, data, cache, cfg, cfg.interaction_seeds)
               for m, a, b in pairs]
    for res, pa in zip(results, holm([r["p_raw"] for r in results])):
        res["p_holm"] = float(pa)
    return results


# ---------------------------------------------------------------- signatures
SIGNATURE_TESTS = [
    ("consistency", "pinn", "lstm"), ("consistency", "pinn", "transformer"),
    ("drift", "transformer", "lstm"), ("drift", "transformer", "pinn"),
    ("accuracy_bias", "lstm", "transformer"), ("completeness_MNAR", "lstm", "transformer"),
]


def _per_seed_rel_slopes(sweep: pd.DataFrame, subset, model, dim, seeds):
    out = []
    for s in seeds:
        g = sweep[(sweep.subset == subset) & (sweep.model == model)
                  & (sweep.dim == dim) & (sweep.seed == s)]
        agg = g.groupby("one_minus_q")["rmse"].mean().reset_index()
        if len(agg) >= 2:
            out.append(np.polyfit(agg["one_minus_q"], agg["rmse"], 1)[0] / g["rmse_base"].mean())
    return np.array(out)


def signature_confirmation(sweep: pd.DataFrame, subset: str, seeds: List[int]) -> pd.DataFrame:
    rows = []
    for dim, robust, fragile in SIGNATURE_TESTS:
        a = _per_seed_rel_slopes(sweep, subset, robust, dim, seeds)
        b = _per_seed_rel_slopes(sweep, subset, fragile, dim, seeds)
        n = min(len(a), len(b))
        diff = a[:n] - b[:n]
        p = (wilcoxon(diff, alternative="less").pvalue
             if n >= 3 and np.any(diff != 0) else np.nan)
        rows.append(dict(subset=subset, dim=dim, robust=robust, fragile=fragile,
                         eta_robust=round(a.mean(), 2), eta_fragile=round(b.mean(), 2),
                         confirmed=bool(a.mean() < b.mean()), p_raw=float(p)))
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p_raw"].values)
    return df


# ---------------------------------------------------------------- injection point
def injection_point_study(data: Dict, cache: ModelCache, cfg: Config, out_csv: str,
                          model: str = "lstm", dims=("accuracy_bias", "completeness_MNAR"),
                          sevs=(3, 5), seeds=(0, 1, 2)) -> pd.DataFrame:
    device = cache.device
    te = make_loader(data["Xte"], data["yte"], cfg.batch_size, False, cfg.global_seed,
                     2 if device.type == "cuda" else 0)
    done = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()
    done_keys = ({(r.dim, r.sev, r.seed) for r in done.itertuples()} if len(done) else set())
    for dim in dims:
        for sev in sevs:
            for seed in seeds:
                if (dim, sev, seed) in done_keys:
                    continue
                base_mdl, base_meta = cache.get(model, data, seed=seed, verbose=False)
                bss = base_meta["sigma_scale"]
                clean = evaluate(base_mdl, te, sigma_scale=bss, device=device)["rmse"]
                gen = torch.Generator(device=device)
                gen.manual_seed(77 + seed)
                deg_fn = (lambda xb, g, d=dim, s=sev: degrade(xb, d, s, g))
                inf_r = evaluate(base_mdl, te, deg_fn, gen, bss, device)["rmse"]
                tp_fn = (lambda xb, g, d=dim, s=sev: degrade(xb, d, s, g))
                mdl, meta = cache.get(model, data, seed=seed, tag=f"_tp_{dim}{sev}",
                                      train_degrade_fn=tp_fn, verbose=False)
                ss = meta["sigma_scale"]
                train_r = evaluate(mdl, te, sigma_scale=ss, device=device)["rmse"]
                gen2 = torch.Generator(device=device)
                gen2.manual_seed(77 + seed)
                both_r = evaluate(mdl, te, deg_fn, gen2, ss, device)["rmse"]
                row = dict(dim=dim, sev=sev, seed=seed, clean=clean, d_inference=inf_r - clean,
                           d_train=train_r - clean, d_both=both_r - clean)
                done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
                done.to_csv(out_csv, index=False)
    return done


# ---------------------------------------------------------------- gating ablation
def gating_ablation(data: Dict, cache: ModelCache, cfg: Config, seeds=(0, 1, 2),
                    sevs=(1, 3, 5)):
    device = cache.device
    te = make_loader(data["Xte"], data["yte"], cfg.batch_size, False, cfg.global_seed,
                     2 if device.type == "cuda" else 0)
    rows = []
    for use_qcg in (False, True):
        for seed in seeds:
            mdl, meta = cache.get("lstm", data, seed=seed, use_qcg=use_qcg,
                                  train_degrade_fn=mixed_degrade, tag="_mix", verbose=False)
            ss = meta["sigma_scale"]
            base = evaluate(mdl, te, sigma_scale=ss, device=device)["rmse"]
            for dim in MIX_DIMS:
                for sev in [0] + list(sevs):
                    gen = torch.Generator(device=device)
                    gen.manual_seed(500 + sev)
                    deg_fn = (lambda xb, g, d=dim, s=sev: degrade(xb, d, s, g))
                    r = evaluate(mdl, te, deg_fn, gen, ss, device)
                    rows.append(dict(qcg=use_qcg, seed=seed, dim=dim, sev=sev,
                                     rmse=r["rmse"], rmse_base=base))
    sweep = pd.DataFrame(rows)

    out = []
    for dim in sweep["dim"].unique():
        diffs, etas = [], {True: [], False: []}
        for seed in sorted(sweep["seed"].unique()):
            per = {}
            for flag in (False, True):
                g = sweep[(sweep.qcg == flag) & (sweep.seed == seed) & (sweep.dim == dim)]
                agg = g.groupby("sev")["rmse"].mean().reset_index()
                if len(agg) >= 2:
                    per[flag] = np.polyfit(agg["sev"], agg["rmse"], 1)[0]
                    etas[flag].append(per[flag])
            if len(per) == 2:
                diffs.append(per[True] - per[False])
        diffs = np.array(diffs)
        p = (wilcoxon(diffs, alternative="less").pvalue
             if len(diffs) >= 3 and np.any(diffs != 0) else np.nan)
        out.append(dict(dim=dim, eta_qcg_off=round(float(np.mean(etas[False])), 2),
                        eta_qcg_on=round(float(np.mean(etas[True])), 2),
                        mean_diff=round(float(diffs.mean()), 2), p_raw=float(p)))
    result = pd.DataFrame(out)
    result["p_holm"] = holm(result["p_raw"].values)
    return sweep, result
