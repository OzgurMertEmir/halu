# Halu/analysis/eval_metrics.py
from __future__ import annotations
from typing import Tuple, Callable
import numpy as np
import pandas as pd
from halu.utils.math import trapz_area

# ---------------------------
# Calibration-style summaries
# ---------------------------
def ece_binary(y_true: np.ndarray, p_err: np.ndarray, bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE) for P(error).
    Conventions:
      - y_true ∈ {0,1}, where 1 = error/incorrect.
      - p_err is the model's predicted probability of error (risk).
      - We compare predicted correctness (1 - p_err) vs. observed accuracy per bin.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_err, dtype=float)
    if y.size == 0:
        return float("nan")
    if y.shape[0] != p.shape[0]:
        raise ValueError(f"ece_binary: y and p must have same length (got {y.shape[0]} vs {p.shape[0]})")

    edges = np.linspace(0, 1, bins + 1)
    inds = np.digitize(p, edges) - 1
    ece = 0.0
    for b in range(bins):
        m = (inds == b)
        if not m.any():
            continue
        # In-bin predicted correctness vs observed accuracy
        pred_corr = 1.0 - float(p[m].mean())
        obs_corr  = 1.0 - float(y[m].mean())
        ece += float(m.mean()) * abs(obs_corr - pred_corr)
    return float(ece)


def reliability_table(y_true: np.ndarray, p_err: np.ndarray, bins: int = 12) -> pd.DataFrame:
    """
    Reliability bins for P(error). Reports per-bin mean P(error) and empirical error rate.
    Columns: [p_lo, p_hi, n, p_mean(Perr), err_rate, cal_error_abs]
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_err, dtype=float)
    if y.shape[0] != p.shape[0]:
        raise ValueError(f"reliability_table: y and p must have same length (got {y.shape[0]} vs {p.shape[0]})")

    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(p, edges) - 1
    rows = []
    for b in range(bins):
        m = (idx == b)
        if not m.any():
            continue
        p_mean = float(p[m].mean())
        err_rate = float(y[m].mean())
        rows.append([float(edges[b]), float(edges[b + 1]), int(m.sum()),
                     p_mean, err_rate, abs(p_mean - err_rate)])
    return pd.DataFrame(rows, columns=["p_lo", "p_hi", "n", "p_mean(Perr)", "err_rate", "cal_error_abs"])


# --------------------------------
# Selective prediction curve family
# --------------------------------
def risk_coverage_curves(
    y_true: np.ndarray,
    p_err: np.ndarray,
    n: int = 0  # kept for API compat; ignored in prefix mode
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Prefix (sorted) risk-coverage curves:
    - cov[k]  = (k+1)/N
    - risk[k] = mean(y[:k+1]) after sorting by p_err ascending
    - acc[k]  = 1 - risk[k]
    Integration uses endpoints [(0,·), (cov,·)] so degenerate cases are exact.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_err, dtype=float)
    if y.shape[0] != p.shape[0]:
        raise ValueError("risk_coverage_curves: y and p must have same length")
    N = y.shape[0]
    if N == 0:
        return np.array([]), np.array([]), np.array([]), float("nan"), float("nan")

    order = np.argsort(p, kind="mergesort")
    y_sorted = y[order]

    cov = np.arange(1, N + 1, dtype=float) / float(N)
    csum = np.cumsum(y_sorted, dtype=float)
    risk = csum / np.arange(1, N + 1, dtype=float)
    acc  = 1.0 - risk

    # Integrate with (0,0) for risk and (0,1) for accuracy to make edge cases exact
    cov_i   = np.concatenate((np.array([0.0], dtype=float), cov))
    risk_i  = np.concatenate((np.array([0.0], dtype=float), risk))
    acc_i   = np.concatenate((np.array([1.0], dtype=float), acc))

    AURC  = trapz_area(risk_i, cov_i)
    AUACC = trapz_area(acc_i,  cov_i)
    return cov, risk, acc, float(AURC), float(AUACC)

def aurc_and_auacc(y_true: np.ndarray, p_err: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    cov, risk, acc, AURC, AUACC = risk_coverage_curves(y_true, p_err)
    return AURC, AUACC, cov, risk, acc

def coverage_at_accuracy(y_true: np.ndarray, p_err: np.ndarray, acc_target: float) -> float:
    """
    Max coverage with accuracy >= acc_target (using the same prefix curve).
    """
    cov, _, acc, _, _ = risk_coverage_curves(y_true, p_err)
    if cov.size == 0:
        return 0.0
    hit = (acc >= float(acc_target))
    return float(cov[hit].max()) if hit.any() else 0.0

# ------------------------------
# Generic stratified bootstrap CI
# ------------------------------
def bootstrap_ci(
    func: Callable[[np.ndarray, np.ndarray], float],
    y: np.ndarray,
    p: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 1337
) -> Tuple[float, float, float]:
    """
    Stratified bootstrap for a scalar metric func(y, p). Returns (lo, hi, point).
    - Resamples positives and negatives with replacement to preserve class balance.
    - Deterministic given `seed`.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if y.shape[0] != p.shape[0]:
        raise ValueError(f"bootstrap_ci: y and p must have same length (got {y.shape[0]} vs {p.shape[0]})")

    point = float(func(y, p))

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        # Degenerate case: only one class present
        return (float("nan"), float("nan"), point)

    vals = []
    for _ in range(n_boot):
        s0 = rng.choice(idx0, size=len(idx0), replace=True)
        s1 = rng.choice(idx1, size=len(idx1), replace=True)
        s = np.concatenate([s0, s1])
        ys = y[s]; ps = p[s]
        try:
            vals.append(float(func(ys, ps)))
        except Exception:
            # In case func fails on a pathological resample
            continue

    if not vals:
        return float("nan"), float("nan"), point

    lo = float(np.quantile(vals, alpha / 2.0))
    hi = float(np.quantile(vals, 1 - alpha / 2.0))
    return lo, hi, point