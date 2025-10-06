# halu/analysis/selective_eval.py
from __future__ import annotations
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple

# ---------------- Core helpers (existing) ----------------
def tau_for_abstain_frac(p_cal_hat: np.ndarray, abstain_frac: float) -> float:
    """Pick τ on calibration so that test coverage ≈ 1 - abstain_frac."""
    return float(np.quantile(p_cal_hat, 1 - abstain_frac, method="linear"))

def risk_coverage_curves(y: np.ndarray, p_hat: np.ndarray):
    """Sort by risk p_hat ascending; return Coverage, Risk(=kept error), and Accuracy."""
    order = np.argsort(p_hat)
    y_sorted = y[order].astype(int)
    n = len(y_sorted)
    cov = np.arange(1, n+1) / n
    kept_err = np.cumsum(y_sorted) / np.arange(1, n+1)
    acc = 1.0 - kept_err
    return cov, kept_err, acc

def aurc_and_auacc(y: np.ndarray, p_hat: np.ndarray):
    cov, risk, acc = risk_coverage_curves(y, p_hat)
    AURC  = float(np.trapz(risk, cov))
    AUACC = float(np.trapz(acc,  cov))
    return AURC, AUACC, cov, risk, acc

def coverage_at_accuracy(y: np.ndarray, p_hat: np.ndarray, target_acc: float) -> float:
    cov, _, acc = risk_coverage_curves(y, p_hat)
    ok = acc >= target_acc
    return float(cov[ok].max()) if ok.any() else 0.0

def _ece_binary_subset(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """
    ECE for y∈{0,1} where y=1 means error (risk), so 'confidence' for correct is (1 - p).
    """
    if len(p) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(p, edges) - 1
    ece = 0.0
    for b in range(bins):
        m = (idx == b)
        if not m.any(): 
            continue
        conf = 1.0 - float(p[m].mean())           # "confidence of being correct"
        acc  = 1.0 - float(y[m].mean())           # observed accuracy in bin
        ece += (m.mean()) * abs(acc - conf)
    return float(ece)

# ---------------- Extended abstention metrics ----------------
def abstention_metrics(
    y_true: np.ndarray,
    p_hat: np.ndarray,
    tau: float,
    *,
    bins_ece: int = 15
) -> Dict[str, float]:
    """
    y_true: 1 = error/hallucination, 0 = correct
    p_hat : predicted error probability
    tau   : abstain threshold; abstain if p_hat >= tau
    """
    y = y_true.astype(int)
    p = p_hat.astype(float)
    abstain = (p >= tau)
    keep    = ~abstain

    n = float(len(y))
    n_pos = float((y == 1).sum())  # errors (unsafe)
    n_neg = float((y == 0).sum())  # correct (safe)

    # Confusion for "abstain-on-error" detector
    TP = float(((abstain) & (y == 1)).sum())  # abstained & was error   (good abstention)
    FP = float(((abstain) & (y == 0)).sum())  # abstained & was correct (over-conservative)
    FN = float(((keep)    & (y == 1)).sum())  # kept & was error        (bad: hallucination slipped through)
    TN = float(((keep)    & (y == 0)).sum())  # kept & was correct      (good answer)

    # Rates
    coverage         = float(keep.mean())          # = acceptance_rate
    acceptance_rate  = coverage
    abstain_rate     = 1.0 - coverage

    # Accuracy among kept
    kept_error = float(y[keep].mean()) if keep.any() else float("nan")
    selective_accuracy = 1.0 - kept_error if keep.any() else float("nan")

    # Abstention classification metrics (treat "positive" as y=1 i.e., error)
    abstention_precision = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
    abstention_recall    = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
    if np.isfinite(abstention_precision) and np.isfinite(abstention_recall) and (abstention_precision + abstention_recall) > 0:
        abstention_f1 = 2 * abstention_precision * abstention_recall / (abstention_precision + abstention_recall)
    else:
        abstention_f1 = float("nan")

    # "Abstention accuracy" as a 2-class task: abstain on errors (TP) + keep on correct (TN)
    abstention_accuracy = (TP + TN) / n if n > 0 else float("nan")

    # Safety-side rates
    benign_answering_rate   = TN / n_neg if n_neg > 0 else float("nan")   # answered safely
    over_conservativeness   = FP / n_neg if n_neg > 0 else float("nan")   # abstained on safe
    abstained_on_safe_rate  = over_conservativeness                       # alias used in your table

    # Reliability-style summaries
    reliable_accuracy     = coverage * selective_accuracy if np.isfinite(coverage) and np.isfinite(selective_accuracy) else float("nan")
    effective_reliability = selective_accuracy  # by your prior table definition

    # Calibration in kept/abstained partitions
    brier_kept = float(np.mean((y[keep] - p[keep])**2)) if keep.any() else float("nan")
    abstain_ece_kept = _ece_binary_subset(y[keep], p[keep], bins=bins_ece) if keep.any() else float("nan")
    abstain_ece_abst = _ece_binary_subset(y[abstain], p[abstain], bins=bins_ece) if abstain.any() else float("nan")

    return dict(
        tau=float(tau),
        coverage=coverage,
        acceptance_rate=acceptance_rate,
        abstain_rate=abstain_rate,
        selective_accuracy=selective_accuracy,
        kept_error=kept_error,
        reliable_accuracy=reliable_accuracy,
        effective_reliability=effective_reliability,
        abstention_accuracy=abstention_accuracy,
        abstention_precision=abstention_precision,
        abstention_recall=abstention_recall,
        abstention_f1=abstention_f1,
        benign_answering_rate=benign_answering_rate,
        over_conservativeness=over_conservativeness,
        abstained_on_safe_rate=abstained_on_safe_rate,
        brier_kept=brier_kept,
        abstain_ece_kept=abstain_ece_kept,
        abstain_ece_abstained=abstain_ece_abst
    )

# ---------------- Bootstrap CIs for selective curves ----------------
def bootstrap_selective(y, p, B=2000, seed=2025, acc_targets=(0.80,0.85,0.90,0.95)):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    idx_pos = np.where(y==1)[0]; idx_neg = np.where(y==0)[0]
    n_pos, n_neg = len(idx_pos), len(idx_neg)
    AURCs, AUACCs = [], []
    CovAt = {a: [] for a in acc_targets}
    for _ in range(B):
        i_pos = rng.integers(0, n_pos, size=n_pos)
        i_neg = rng.integers(0, n_neg, size=n_neg)
        sel = np.r_[ idx_pos[i_pos], idx_neg[i_neg] ]
        yb, pb = y[sel], p[sel]
        aurc, auacc, *_ = aurc_and_auacc(yb, pb)
        AURCs.append(aurc); AUACCs.append(auacc)
        for a in acc_targets:
            CovAt[a].append(coverage_at_accuracy(yb, pb, a))
    def ci(x):
        lo, hi = np.quantile(x, [0.025, 0.975]); return float(lo), float(hi), float(np.mean(x))
    return dict(
        AURC  = ci(AURCs),
        AUACC = ci(AUACCs),
        CoverageAt = {a: ci(CovAt[a]) for a in acc_targets}
    )

# ---------------- Convenience: build abstention table & exports ----------------
def build_abstention_table_from_targets(
    y_test: np.ndarray,
    p_test_calibrated: np.ndarray,
    p_calib_blend: np.ndarray,
    abstain_targets = (0.05, 0.10, 0.20, 0.30, 0.40),
    bins_ece: int = 15
) -> pd.DataFrame:
    """
    - Picks τ on Calib to match each target abstain fraction (≈ quantile on p_calib_blend)
    - Evaluates metrics on Test (using calibrated p_test_calibrated)
    """
    rows = []
    for a in abstain_targets:
        tau = tau_for_abstain_frac(p_calib_blend, a)
        m = abstention_metrics(y_test, p_test_calibrated, tau, bins_ece=bins_ece)
        rows.append({"abstain_frac_target": float(a), **m})
    return pd.DataFrame(rows)

def export_selective_plots_and_tables(
    y_test: np.ndarray,
    p_test_calibrated: np.ndarray,
    abst_table: pd.DataFrame,
    out_dir: str = "paper_metrics",
    acc_targets = (0.80, 0.85, 0.90, 0.95)
) -> Dict[str, float]:
    out = Path(out_dir); out.mkdir(exist_ok=True, parents=True)

    # Curves + areas
    AURC, AUACC, cov, risk, acc = aurc_and_auacc(y_test, p_test_calibrated)

    # Plots
    plt.figure()
    plt.plot(cov, risk, lw=2)
    plt.xlabel("Coverage"); plt.ylabel("Risk (error rate among kept)")
    plt.title("Risk–Coverage (Test, calibrated)")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out / "risk_coverage.png", dpi=160)

    plt.figure()
    plt.plot(cov, acc, lw=2)
    plt.xlabel("Coverage"); plt.ylabel("Accuracy among kept")
    plt.title("Accuracy–Coverage (Test, calibrated)")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out / "accuracy_coverage.png", dpi=160)

    # Coverage@Acc sweep
    cov_at = {f"Coverage@Acc≥{a:.2f}": coverage_at_accuracy(y_test, p_test_calibrated, a) for a in acc_targets}

    # Save table
    abst_table.to_csv(out / "abstention_table.csv", index=False)

    return dict(AURC=AURC, AUACC=AUACC, **cov_at)
