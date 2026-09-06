"""Data-quality injection operators.

Each operator takes a batch of windows `x` with shape (B, L, d), an ordinal
severity in 1..5, and a torch Generator, and returns the degraded batch plus a
per-channel quality estimate `u` with shape (B, d) in [0, 1]. Severity maps to
a physical parameter linearly between the bounds listed in the paper.

Eight dimensions are exposed through INJECTORS:
    accuracy, accuracy_bias, completeness_MCAR, completeness_MAR,
    completeness_MNAR, timeliness, consistency, drift
"""

from typing import Callable, Dict, Optional, Tuple

import torch

Tensor = torch.Tensor


def _sev_scale(s: int, lo: float, hi: float) -> float:
    """Linear map from severity 1..5 to [lo, hi]."""
    return lo + (hi - lo) * (max(1, s) - 1) / 4.0


def _ffill(x: Tensor) -> Tensor:
    """Last-observation-carried-forward along the time axis; NaN at t=0 -> 0."""
    out = x.clone()
    for t in range(1, out.shape[1]):
        cur, prev = out[:, t, :], out[:, t - 1, :]
        nanmask = torch.isnan(cur)
        cur[nanmask] = prev[nanmask]
        out[:, t, :] = cur
    out[torch.isnan(out)] = 0.0
    return out


def inject_accuracy(x: Tensor, s: int, bias: bool = False,
                    gen: Optional[torch.Generator] = None) -> Tuple[Tensor, Tensor]:
    std = _sev_scale(s, 0.05, 0.60)
    noise = torch.randn(x.shape, generator=gen, device=x.device) * std
    xd = x + noise + (std if bias else 0.0)
    u = torch.full((x.shape[0], x.shape[2]), 1.0 - min(std, 1.0), device=x.device)
    return xd, u


def _missing_mask(x: Tensor, s: int, mech: str, gen) -> Tensor:
    p = _sev_scale(s, 0.05, 0.50)
    if mech == "MCAR":
        return torch.rand(x.shape, generator=gen, device=x.device) < p
    if mech == "MAR":
        driver = x[..., :1]
        prob = torch.sigmoid(3.0 * (driver - driver.mean())) * (2.0 * p)
        return torch.rand(x.shape, generator=gen, device=x.device) < prob
    # MNAR: probability of missing depends on the value itself
    prob = torch.sigmoid(3.0 * (x - x.mean())) * (2.0 * p)
    return torch.rand(x.shape, generator=gen, device=x.device) < prob


def make_missing(x: Tensor, s: int, mech: str = "MCAR", gen=None) -> Tuple[Tensor, Tensor]:
    """Return the raw window with NaNs (before imputation) and the mask."""
    mask = _missing_mask(x, s, mech, gen)
    xd = x.clone()
    xd[mask] = float("nan")
    return xd, mask


def inject_completeness(x: Tensor, s: int, mech: str = "MCAR", gen=None):
    xd_raw, mask = make_missing(x, s, mech, gen)
    return _ffill(xd_raw), 1.0 - mask.float().mean(dim=1)


def inject_timeliness(x: Tensor, s: int, gen=None):
    delay = int(round(_sev_scale(s, 1, 8)))
    xd = torch.roll(x, shifts=delay, dims=1)
    xd[:, :delay, :] = x[:, :1, :]
    factor = int(round(_sev_scale(s, 1, 5)))
    if factor > 1:
        idx = (torch.arange(x.shape[1], device=x.device) // factor) * factor
        xd = xd[:, idx.clamp(max=x.shape[1] - 1), :]
    u = torch.full((x.shape[0], x.shape[2]),
                   1.0 - min((delay + factor) / 13.0, 1.0), device=x.device)
    return xd, u


def inject_consistency(x: Tensor, s: int, gen=None):
    off = _sev_scale(s, 0.1, 1.0)
    n_feat = x.shape[2]
    affected = torch.rand(n_feat, generator=gen, device=x.device) < 0.5
    scale = torch.where(affected, torch.tensor(1.0 + off, device=x.device),
                        torch.tensor(1.0, device=x.device))
    shift = torch.where(affected, torch.tensor(off, device=x.device),
                        torch.tensor(0.0, device=x.device))
    u = (1.0 - off * affected.float()).unsqueeze(0).expand(x.shape[0], -1).clone()
    return x * scale + shift, u


def inject_drift(x: Tensor, s: int, gen=None):
    mag = _sev_scale(s, 0.1, 1.0)
    ramp = torch.linspace(0.0, mag, x.shape[1], device=x.device).view(1, -1, 1)
    u = torch.full((x.shape[0], x.shape[2]), 1.0 - min(mag, 1.0), device=x.device)
    return x + ramp, u


INJECTORS: Dict[str, Callable] = {
    "accuracy":          lambda x, s, g: inject_accuracy(x, s, False, g),
    "accuracy_bias":     lambda x, s, g: inject_accuracy(x, s, True, g),
    "completeness_MCAR": lambda x, s, g: inject_completeness(x, s, "MCAR", g),
    "completeness_MAR":  lambda x, s, g: inject_completeness(x, s, "MAR", g),
    "completeness_MNAR": lambda x, s, g: inject_completeness(x, s, "MNAR", g),
    "timeliness":        lambda x, s, g: inject_timeliness(x, s, g),
    "consistency":       lambda x, s, g: inject_consistency(x, s, g),
    "drift":             lambda x, s, g: inject_drift(x, s, g),
}


def degrade(x: Tensor, dim: Optional[str], severity: int, gen=None) -> Tuple[Tensor, Tensor]:
    """Apply one operator. severity=0 or dim=None returns the clean input."""
    if severity == 0 or dim is None:
        return x, torch.ones(x.shape[0], x.shape[2], device=x.device)
    return INJECTORS[dim](x, severity, gen)


# Dimensions used for mixed-degradation training in the gating ablation.
MIX_DIMS = ["accuracy_bias", "completeness_MNAR", "consistency", "drift"]


def mixed_degrade(xb: Tensor, gen) -> Tuple[Tensor, Tensor]:
    """Sample one (dimension, severity) per batch; about 1/6 of batches are clean."""
    pick = int(torch.randint(0, len(MIX_DIMS) + 1, (1,), generator=gen, device=xb.device).item())
    if pick == len(MIX_DIMS):
        return degrade(xb, None, 0, gen)
    sev = int(torch.randint(1, 6, (1,), generator=gen, device=xb.device).item())
    return degrade(xb, MIX_DIMS[pick], sev, gen)


def mean_quality(dim: str, severity: int, probe_x: Tensor) -> float:
    """Mean quality scalar q for a (dim, severity) pair, estimated on a probe batch."""
    gen = torch.Generator(device=probe_x.device)
    gen.manual_seed(0)
    _, u = degrade(probe_x, dim, severity, gen)
    return float(u.mean().cpu())
