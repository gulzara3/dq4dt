import numpy as np
import torch

from dq4dt.metrics import (coverage, ece_regression, fit_sigma_scale, gaussian_nll,
                           nasa_score, physics_penalty, rmse)


def test_rmse_basic():
    assert rmse(np.array([1., 2.]), np.array([1., 4.])) == np.sqrt(2.0)


def test_nasa_score_asymmetric():
    # late prediction (pred > true) should cost more than early by same margin
    late = nasa_score(np.array([110.]), np.array([100.]))
    early = nasa_score(np.array([90.]), np.array([100.]))
    assert late > early > 0


def test_coverage_of_perfectly_calibrated_gaussian():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 20000)
    cov = coverage(np.zeros_like(y), np.ones_like(y), y, 0.9)
    assert abs(cov - 0.9) < 0.01


def test_ece_near_zero_when_calibrated():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 2, 20000)
    assert ece_regression(np.zeros_like(y), 2 * np.ones_like(y), y) < 0.01


def test_sigma_scale_recovers_true_factor():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 3, 20000)
    s = fit_sigma_scale(np.zeros_like(y), np.ones_like(y), y)
    assert abs(s - 3.0) < 0.05


def test_gaussian_nll_finite():
    mu, lv, y = torch.zeros(5), torch.zeros(5), torch.ones(5)
    assert torch.isfinite(gaussian_nll(mu, lv, y))


def test_physics_penalty_zero_for_monotone_nonneg():
    mu = torch.tensor([5., 4., 3., 2., 1.])
    assert physics_penalty(mu).item() == 0.0
    assert physics_penalty(torch.tensor([1., 2., 3.])).item() > 0
