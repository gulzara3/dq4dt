import numpy as np

from dq4dt.stats import cliffs_delta, fit_breakpoint, holm, rank_biserial


def test_holm_monotone_and_bounded():
    adj = holm([0.01, 0.04, 0.03])
    assert np.all(adj <= 1.0) and np.all(adj >= np.array([0.01, 0.04, 0.03]))
    assert adj[0] == 0.03  # smallest p times m=3


def test_rank_biserial_extremes():
    assert rank_biserial([1, 2, 3]) == 1.0
    assert rank_biserial([-1, -2, -3]) == -1.0


def test_cliffs_delta_sign():
    assert cliffs_delta([5, 6, 7], [1, 2, 3]) == 1.0


def test_breakpoint_recovered_on_hinge_data():
    x = np.linspace(0, 1, 11)
    y = 1.0 + 0.5 * x + 4.0 * np.maximum(x - 0.4, 0.0)
    brk, sse_seg, sse_lin = fit_breakpoint(x, y)
    assert abs(brk - 0.4) < 1e-9
    assert sse_seg < 1e-9 and sse_lin > sse_seg
