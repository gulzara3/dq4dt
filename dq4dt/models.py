"""Surrogate twin architectures.

All three share a heteroscedastic Gaussian head that returns (mu, log_var).
An optional quality-conditioned gate (QCG) multiplies each input channel by a
learned function of the per-channel quality estimate. The gate is included so
the ablation in the paper can be reproduced; it did not improve robustness.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt


class QCG(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_feat, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_feat), nn.Sigmoid())

    def forward(self, x, u):
        return x * self.net(u).unsqueeze(1)


class HeteroHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.mu = nn.Linear(d, 1)
        self.lv = nn.Linear(d, 1)

    def forward(self, h):
        return self.mu(h).squeeze(-1), self.lv(h).squeeze(-1).clamp(-6.0, 6.0)


class LSTMTwin(nn.Module):
    def __init__(self, n_feat, hidden=96, layers=2, use_qcg=False):
        super().__init__()
        self.qcg = QCG(n_feat) if use_qcg else None
        self.rnn = nn.LSTM(n_feat, hidden, layers, batch_first=True, dropout=0.2)
        self.head = HeteroHead(hidden)

    def forward(self, x, u=None):
        if self.qcg is not None and u is not None:
            x = self.qcg(x, u)
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


class TransformerTwin(nn.Module):
    """Lightweight encoder: d_model 128, 2 layers, mean + last-token pooling."""

    def __init__(self, n_feat, d_model=128, nhead=4, layers=2, use_qcg=False, max_len=64):
        super().__init__()
        self.qcg = QCG(n_feat) if use_qcg else None
        self.proj = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model, dropout=0.1,
            batch_first=True, activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, layers, enable_nested_tensor=False)
        self.head = HeteroHead(2 * d_model)
        self.grad_ckpt = True

    def forward(self, x, u=None):
        if self.qcg is not None and u is not None:
            x = self.qcg(x, u)
        h = self.proj(x) + self.pos[:, :x.shape[1]]
        if self.grad_ckpt and self.training:
            h = ckpt.checkpoint(self.enc, h, use_reentrant=False)
        else:
            h = self.enc(h)
        pooled = torch.cat([h.mean(dim=1), h[:, -1]], dim=-1)
        return self.head(pooled)


class PINNTwin(nn.Module):
    """Dilated TCN with GroupNorm. Physics enters through the loss (see training.py)."""

    def __init__(self, n_feat, ch=64, use_qcg=False):
        super().__init__()
        self.qcg = QCG(n_feat) if use_qcg else None
        self.tcn = nn.Sequential(
            nn.Conv1d(n_feat, ch, 5, padding=2), nn.GELU(), nn.GroupNorm(8, ch),
            nn.Conv1d(ch, ch, 5, padding=4, dilation=2), nn.GELU(), nn.GroupNorm(8, ch),
            nn.Conv1d(ch, ch, 5, padding=8, dilation=4), nn.GELU(), nn.GroupNorm(8, ch))
        self.head = HeteroHead(ch)

    def forward(self, x, u=None):
        if self.qcg is not None and u is not None:
            x = self.qcg(x, u)
        h = self.tcn(x.transpose(1, 2))
        return self.head(h.mean(dim=-1))


MODEL_REGISTRY = {"lstm": LSTMTwin, "transformer": TransformerTwin, "pinn": PINNTwin}


def build_model(name: str, n_feat: int, use_qcg: bool = False, device=None) -> nn.Module:
    model = MODEL_REGISTRY[name](n_feat, use_qcg=use_qcg)
    return model.to(device) if device is not None else model
