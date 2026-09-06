"""Training loop with per-epoch checkpointing and post-hoc sigma calibration.

Every trained model is keyed by (subset, model, seed, qcg, tag). A finished
model is saved as final_<key>.pt and loads instantly on the next call. A
partially trained model is saved as last_<key>.pt after every epoch and
resumes from there, so an interrupted run never repeats work.
"""

import copy
import math
import os
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .config import Config
from .metrics import fidelity_vector, fit_sigma_scale, gaussian_nll, physics_penalty
from .models import build_model
from .seeding import get_device, seed_worker, set_determinism


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int = 256,
                shuffle: bool = True, seed: int = 42, workers: int = 2) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    gen = torch.Generator()
    gen.manual_seed(seed)
    kw = dict(num_workers=workers, pin_memory=torch.cuda.is_available(),
              worker_init_fn=seed_worker, generator=gen)
    if workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle, **kw)


def cosine_warmup(step: int, total: int, warmup: int, base_lr: float, min_lr: float = 1e-5) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * prog))


@torch.no_grad()
def evaluate(model, loader, degrade_fn: Optional[Callable] = None, gen=None,
             sigma_scale: float = 1.0, device=None) -> Dict:
    """Evaluate on a loader, optionally degrading each batch first.

    `degrade_fn(xb, gen)` must return (x_degraded, u). The quality estimate u is
    passed to the model so that QCG variants can use it.
    """
    device = device or get_device()
    model.eval()
    mus, sigs, ys = [], [], []
    use_amp = device.type == "cuda"
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        u = None
        if degrade_fn is not None:
            xb, u = degrade_fn(xb, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            mu, lv = model(xb, u)
        mus.append(mu.float().cpu().numpy())
        sigs.append(torch.exp(0.5 * lv).float().cpu().numpy())
        ys.append(yb.numpy())
    mu = np.concatenate(mus)
    sig = np.concatenate(sigs) * sigma_scale
    y = np.concatenate(ys)
    out = fidelity_vector(mu, sig, y)
    out.update(mu=mu, sigma=sig, y=y)
    return out


def ckpt_paths(ckpt_dir: str, subset: str, name: str, seed: int, use_qcg: bool, tag: str = ""):
    stem = f"{subset}_{name}_s{seed}_qcg{int(use_qcg)}{tag}"
    return (os.path.join(ckpt_dir, f"final_{stem}.pt"),
            os.path.join(ckpt_dir, f"last_{stem}.pt"))


def train_model(name: str, data: Dict, cfg: Config, ckpt_dir: str, use_qcg: bool = False,
                seed: Optional[int] = None, train_degrade_fn: Optional[Callable] = None,
                tag: str = "", verbose: bool = True, device=None) -> Tuple[torch.nn.Module, Dict]:
    """Train (or resume, or load) one surrogate. Returns (model, blob).

    blob contains the best state dict, validation history, best validation
    RMSE, and the fitted sigma_scale.
    """
    device = device or get_device()
    seed = cfg.global_seed if seed is None else seed
    os.makedirs(ckpt_dir, exist_ok=True)
    final_p, last_p = ckpt_paths(ckpt_dir, data["subset"], name, seed, use_qcg, tag)
    base_lr = cfg.lr[name]
    use_physics = (name == "pinn")
    label = f"[{data['subset']} {name} s{seed}{tag}]"

    if os.path.exists(final_p):
        blob = torch.load(final_p, map_location=device, weights_only=False)
        model = build_model(name, data["n_feat"], use_qcg, device)
        model.load_state_dict(blob["state"])
        if verbose:
            print(f"{label} loaded final checkpoint (best val {blob['best_rmse']:.2f}).")
        return model, blob

    set_determinism(seed, strict=False)
    model = build_model(name, data["n_feat"], use_qcg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    workers = 2 if device.type == "cuda" else 0
    tr = make_loader(data["Xtr"], data["ytr"], cfg.batch_size, True, cfg.global_seed, workers)
    va = make_loader(data["Xva"], data["yva"], cfg.batch_size, False, cfg.global_seed, workers)
    total_steps = cfg.epochs * len(tr)
    lr_warm = int(0.05 * total_steps)
    dgen = torch.Generator(device=device)
    dgen.manual_seed(seed)
    start_ep, step, bad = 0, 0, 0
    best_rmse, best_state, history = float("inf"), None, []
    use_amp = device.type == "cuda"

    if os.path.exists(last_p):
        blob = torch.load(last_p, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        start_ep, step, bad = blob["epoch"] + 1, blob["step"], blob["bad"]
        best_rmse, best_state, history = blob["best_rmse"], blob["best_state"], blob["history"]
        if verbose:
            print(f"{label} resuming at epoch {start_ep} (best {best_rmse:.2f}).")

    t0 = time.time()
    for ep in range(start_ep, cfg.epochs):
        model.train()
        warm = ep < cfg.warmup_epochs
        for xb, yb in tr:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            u = None
            if train_degrade_fn is not None:
                xb, u = train_degrade_fn(xb, dgen)
            for pg in opt.param_groups:
                pg["lr"] = cosine_warmup(step, total_steps, lr_warm, base_lr)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                mu, lv = model(xb, u)
                if warm:
                    loss = F.mse_loss(mu, yb)
                else:
                    loss = gaussian_nll(mu, lv, yb)
                    if use_physics:
                        order = torch.argsort(yb, descending=True)
                        loss = loss + cfg.physics_weight * physics_penalty(mu[order])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
        val_rmse = evaluate(model, va, device=device)["rmse"]
        history.append(val_rmse)
        if val_rmse < best_rmse - cfg.improve_margin:
            best_rmse, best_state, bad = val_rmse, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if verbose and (ep % 10 == 0 or bad >= cfg.patience):
            print(f"{label} ep{ep:02d} val={val_rmse:6.2f} best={best_rmse:6.2f}")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=ep, step=step,
                        bad=bad, best_rmse=best_rmse, best_state=best_state,
                        history=history), last_p)
        if bad >= cfg.patience and not warm:
            break

    model.load_state_dict(best_state)
    val_res = evaluate(model, va, device=device)
    s_scale = fit_sigma_scale(val_res["mu"], val_res["sigma"], val_res["y"])
    blob = dict(state=best_state, history=history, best_rmse=best_rmse, sigma_scale=s_scale)
    torch.save(blob, final_p)
    if os.path.exists(last_p):
        os.remove(last_p)
    if verbose:
        print(f"{label} done ({time.time() - t0:.0f}s) best {best_rmse:.2f} "
              f"sigma-scale {s_scale:.2f}")
    return model, blob


class ModelCache:
    """In-process cache so a model is trained or loaded at most once per run."""

    def __init__(self, cfg: Config, ckpt_dir: str, device=None):
        self.cfg, self.ckpt_dir = cfg, ckpt_dir
        self.device = device or get_device()
        self._cache = {}

    def get(self, name, data, seed=None, use_qcg=False, **kw):
        seed = self.cfg.global_seed if seed is None else seed
        key = (data["subset"], name, seed, use_qcg, kw.get("tag", ""))
        if key not in self._cache:
            self._cache[key] = train_model(name, data, self.cfg, self.ckpt_dir, use_qcg=use_qcg,
                                           seed=seed, device=self.device, **kw)
        return self._cache[key]
