import torch

from dq4dt.models import MODEL_REGISTRY, build_model


def test_all_models_forward_cpu():
    x = torch.randn(4, 30, 17)
    u = torch.rand(4, 17)
    for name in MODEL_REGISTRY:
        for qcg in (False, True):
            m = build_model(name, 17, use_qcg=qcg)
            m.eval()
            mu, lv = m(x, u if qcg else None)
            assert mu.shape == (4,) and lv.shape == (4,)
            assert torch.isfinite(mu).all() and torch.isfinite(lv).all()
