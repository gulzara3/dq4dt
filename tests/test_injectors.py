import torch
import pytest

from dq4dt.injectors import INJECTORS, degrade, mixed_degrade, make_missing

torch.manual_seed(0)
X = torch.randn(8, 30, 17)


@pytest.mark.parametrize("dim", list(INJECTORS))
@pytest.mark.parametrize("sev", [1, 3, 5])
def test_operator_shapes_and_finite(dim, sev):
    g = torch.Generator().manual_seed(0)
    xd, u = degrade(X, dim, sev, g)
    assert xd.shape == X.shape
    assert u.shape == (X.shape[0], X.shape[2])
    assert torch.isfinite(xd).all()
    assert ((u >= 0) & (u <= 1)).all()


def test_severity_zero_is_identity():
    xd, u = degrade(X, "accuracy", 0)
    assert torch.equal(xd, X)
    assert torch.all(u == 1.0)


def test_quality_decreases_with_severity():
    g = torch.Generator().manual_seed(0)
    qs = [degrade(X, "accuracy", s, g)[1].mean().item() for s in (1, 3, 5)]
    assert qs[0] > qs[1] > qs[2]


def test_mnar_masks_more_than_mcar_on_high_values():
    # MNAR should target large values, so masked entries have higher mean
    g = torch.Generator().manual_seed(1)
    _, mask_mnar = make_missing(X, 5, "MNAR", g)
    g = torch.Generator().manual_seed(1)
    _, mask_mcar = make_missing(X, 5, "MCAR", g)
    assert X[mask_mnar].mean() > X[mask_mcar].mean()


def test_bias_shifts_mean_noise_does_not():
    g = torch.Generator().manual_seed(2)
    xn, _ = degrade(X, "accuracy", 5, g)
    g = torch.Generator().manual_seed(2)
    xb, _ = degrade(X, "accuracy_bias", 5, g)
    assert abs(xn.mean() - X.mean()) < 0.05
    assert xb.mean() - X.mean() > 0.4


def test_mixed_degrade_runs():
    g = torch.Generator().manual_seed(3)
    for _ in range(10):
        xd, u = mixed_degrade(X, g)
        assert xd.shape == X.shape and u.shape == (8, 17)
