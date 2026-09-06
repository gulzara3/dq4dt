"""Losses and the fidelity-vector metrics.

The fidelity vector is [RMSE, PHM08 score, ECE, coverage@90]. All metric
functions take numpy arrays and are safe to call on CPU.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm


# ---------------------------------------------------------------- losses
def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Heteroscedastic Gaussian negative log-likelihood, up to a constant."""
    return 0.5 * (logvar + (y - mu) ** 2 / torch.exp(logvar)).mean()


def physics_penalty(mu_sorted: torch.Tensor) -> torch.Tensor:
    """Penalise RUL that increases along a trajectory or goes negative.

    `mu_sorted` must be ordered by descending true RUL so that consecutive
    differences should be non-positive.
    """
    diff = mu_sorted[1:] - mu_sorted[:-1]
    return F.relu(diff).mean() + F.relu(-mu_sorted).mean()


# ---------------------------------------------------------------- metrics
def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def nasa_score(pred: np.ndarray, true: np.ndarray) -> float:
    """Asymmetric PHM08 score. Late predictions (d >= 0) are penalised more."""
    d = np.clip(pred - true, -500.0, 500.0)
    s = np.where(d < 0, np.exp(np.clip(-d / 13.0, None, 50)) - 1.0,
                 np.exp(np.clip(d / 10.0, None, 50)) - 1.0)
    return float(np.sum(s))


def ece_regression(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error in the quantile formulation."""
    ps = np.linspace(0.05, 0.95, n_bins)
    return float(np.mean([abs(np.mean(y <= mu + sigma * norm.ppf(p)) - p) for p in ps]))


def coverage(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray, level: float = 0.9) -> float:
    """Empirical coverage of the central `level` prediction interval."""
    z = norm.ppf(0.5 + level / 2.0)
    return float(np.mean((y >= mu - z * sigma) & (y <= mu + z * sigma)))


def fit_sigma_scale(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> float:
    """Closed-form MLE variance rescaling factor s, fitted on validation data.

    s^2 = mean(((y - mu) / sigma)^2). Multiply predicted sigma by s.
    """
    z2 = ((y - mu) / np.maximum(sigma, 1e-6)) ** 2
    return float(np.sqrt(np.mean(z2)))


def fidelity_vector(mu, sigma, y) -> dict:
    return dict(rmse=rmse(mu, y), score=nasa_score(mu, y),
                ece=ece_regression(mu, sigma, y), cov90=coverage(mu, sigma, y, 0.9))
