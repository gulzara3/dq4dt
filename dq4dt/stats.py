"""Statistical helpers used throughout the analysis."""

from typing import Sequence, Tuple

import numpy as np


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values. Controls the family-wise error rate."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def rank_biserial(diffs: Sequence[float]) -> float:
    """Matched-pairs rank-biserial correlation, the effect size for Wilcoxon."""
    d = np.asarray(diffs, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()
    return float((w_pos - w_neg) / (w_pos + w_neg))


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return float((gt - lt) / (len(a) * len(b)))


def fit_breakpoint(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float, float]:
    """Segmented (hinge) regression y = a + b x + c max(x - brk, 0).

    Returns (best_breakpoint, sse_segmented, sse_linear). Candidate breakpoints
    are the interior x values.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    lin = np.polyfit(x, y, 1, full=True)
    sse_lin = float(lin[1][0]) if len(lin[1]) else 0.0
    best = (None, np.inf)
    for brk in x[1:-1]:
        design = np.column_stack([np.ones_like(x), x, np.maximum(x - brk, 0.0)])
        _, res, _, _ = np.linalg.lstsq(design, y, rcond=None)
        sse = float(res[0]) if len(res) else 0.0
        if sse < best[1]:
            best = (float(brk), sse)
    return best[0], best[1], sse_lin
