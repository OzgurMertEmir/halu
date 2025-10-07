from __future__ import annotations
import numpy as np

EPS = 1e-8

def trapz_area(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y, dtype=float); x = np.asarray(x, dtype=float)
    if y.size == 0 or x.size == 0:
        return float("nan")
    if y.size != x.size:
        raise ValueError("trapz_area: x and y must have same length")
    dx = np.diff(x)
    if not np.all(np.isfinite(dx)):
        return float("nan")
    mid = 0.5 * (y[1:] + y[:-1])
    return float((dx * mid).sum())

def pearson_from_centered(x: np.ndarray, y: np.ndarray, eps: float = EPS) -> float:
    """
    Pearson r for already-centered vectors (mean ~ 0).
    IMPORTANT: do NOT add eps to denominator unconditionally; only guard the
    zero-variance case. This makes perfect correlation hit exactly 1.0/-1.0.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    num = float((x * y).sum())
    den = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if den <= eps:
        return float("nan")
    return num / den

def top2_gap(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    v = np.sort(v)[::-1]
    return float(v[0] - v[1]) if v.size >= 2 else float(v[0])

def entropy_from_prob(p: np.ndarray) -> float:
    """
    Shannon entropy (nats). Zeros should contribute exactly 0 (limit p log p → 0),
    so we only sum over strictly positive probabilities.
    """
    p = np.asarray(p, dtype=float)
    s = p.sum()
    if s <= 0:
        return float("nan")
    q = p / s
    mask = q > 0.0
    return float(-(q[mask] * np.log(q[mask])).sum())