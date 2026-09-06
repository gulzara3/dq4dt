"""Live data-quality monitor.

Estimates per-dimension quality from a raw (pre-imputation) window and turns
it into a single fidelity-confidence score using elasticity weights. The
monitor sees NaNs before any imputation, which is what makes the completeness
estimate honest.
"""

from typing import Dict

import numpy as np
import pandas as pd
import torch

from .injectors import _ffill


def estimate_quality(raw_window: torch.Tensor, train_std: torch.Tensor) -> Dict[str, float]:
    """Per-dimension quality estimates for one window of shape (L, d).

    completeness: 1 - fraction of NaNs
    accuracy: 1 - |variance ratio to training - 1|, clipped to [0, 1]
    timeliness: 1 - fraction of exact repeats between consecutive observed rows
    """
    nan_frac = torch.isnan(raw_window).float().mean().item()
    q_completeness = 1.0 - nan_frac
    filled = _ffill(raw_window.unsqueeze(0)).squeeze(0)
    var_ratio = (filled.std(dim=0) / (train_std + 1e-6)).mean().item()
    q_accuracy = float(np.clip(1.0 - abs(var_ratio - 1.0), 0.0, 1.0))
    obs = filled[~torch.isnan(raw_window).any(dim=1)]
    repeats = ((obs[1:] == obs[:-1]).float().mean().item() if len(obs) > 1 else 0.0)
    q_timeliness = float(np.clip(1.0 - repeats, 0.0, 1.0))
    return {"completeness": q_completeness, "accuracy": q_accuracy, "timeliness": q_timeliness}


def fidelity_confidence(q_hat: Dict[str, float], elast: pd.DataFrame, subset: str = "FD001",
                        model: str = "transformer") -> float:
    """c = 1 - sum_j w_j (1 - q_j), with w_j proportional to |eta_rel_j|."""
    sub = elast[(elast.model == model) & (elast.subset == subset)]
    eta = {"completeness": float(sub[sub.dim == "completeness_MNAR"]["elasticity_rel"].mean()),
           "accuracy": float(sub[sub.dim == "accuracy"]["elasticity_rel"].mean()),
           "timeliness": float(sub[sub.dim == "timeliness"]["elasticity_rel"].mean())}
    total = sum(abs(v) for v in eta.values()) + 1e-9
    w = {k: abs(v) / total for k, v in eta.items()}
    return float(np.clip(1.0 - sum(w[k] * (1.0 - q_hat.get(k, 1.0)) for k in w), 0.0, 1.0))
