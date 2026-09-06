"""The main degradation sweep.

Runs every (subset, model, seed) block and, inside each, every (dimension,
severity) pair including the clean baseline. Results are appended to a CSV
after each block, so the sweep resumes at block granularity.
"""

import os
from typing import Dict

import pandas as pd
import torch
from tqdm.auto import tqdm

from .config import Config
from .injectors import degrade, mean_quality
from .training import ModelCache, evaluate, make_loader


def run_sweep(cfg: Config, datasets: Dict[str, Dict], cache: ModelCache,
              out_csv: str) -> pd.DataFrame:
    device = cache.device
    if os.path.exists(out_csv):
        done_df = pd.read_csv(out_csv)
        expected = len(cfg.dims) * (len(cfg.severities) + 1)
        done_keys = {k for k, g in done_df.groupby(["subset", "model", "seed"])
                     if len(g) >= expected}
        print(f"Sweep resume: {len(done_keys)} completed blocks.")
    else:
        done_df, done_keys = pd.DataFrame(), set()

    loaders = {s: make_loader(d["Xte"], d["yte"], cfg.batch_size, False, cfg.global_seed,
                              2 if device.type == "cuda" else 0)
               for s, d in datasets.items()}
    combos = [(sub, m, s) for sub in cfg.subsets for m in cfg.models for s in cfg.seeds]

    for sub, model_name, seed in tqdm(combos, desc="subset x arch x seed"):
        if (sub, model_name, seed) in done_keys:
            continue
        data, te = datasets[sub], loaders[sub]
        probe = torch.from_numpy(data["Xte"][:256]).to(device)
        mdl, meta = cache.get(model_name, data, seed=seed, verbose=False)
        ss = meta["sigma_scale"]
        base = evaluate(mdl, te, sigma_scale=ss, device=device)
        block = []
        for dim in cfg.dims:
            for sev in [0] + list(cfg.severities):
                gen = torch.Generator(device=device)
                gen.manual_seed(1000 + 31 * sev + sum(map(ord, dim)) % 997)
                deg_fn = (lambda xb, g, d=dim, s=sev: degrade(xb, d, s, g))
                res = evaluate(mdl, te, degrade_fn=deg_fn, gen=gen, sigma_scale=ss, device=device)
                q = 1.0 if sev == 0 else mean_quality(dim, sev, probe)
                block.append(dict(subset=sub, model=model_name, seed=seed, dim=dim, sev=sev,
                                  one_minus_q=1.0 - q, rmse=res["rmse"], score=res["score"],
                                  ece=res["ece"], cov90=res["cov90"], rmse_base=base["rmse"]))
        done_df = pd.concat([done_df, pd.DataFrame(block)], ignore_index=True)
        done_df.to_csv(out_csv, index=False)
    return done_df
