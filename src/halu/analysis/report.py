# analysis/report.py
# report.py
from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
if os.environ.get("MPLBACKEND", "").lower() not in ("agg", "module://matplotlib_inline.backend_inline"):
    try:
        matplotlib.use("Agg", force=True)
    except Exception:
        pass
import matplotlib.pyplot as plt

# Small, consistent defaults
plt.rcParams.update({
    "figure.figsize": (6, 5),
    "figure.dpi": 160,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

from halu.analysis.eval_metrics import (
    ece_binary,
    reliability_table,
    risk_coverage_curves,
    coverage_at_accuracy,
    bootstrap_ci,
)

_HAS_EVALUATE = False
try:
    import evaluate
    _HAS_EVALUATE = True
except Exception:
    _HAS_EVALUATE = False

# ===== sklearn fallbacks =====
from sklearn.metrics import (
    roc_auc_score as _sk_roc_auc,
    average_precision_score as _sk_ap,
    brier_score_loss as _sk_brier,
    confusion_matrix as _sk_confusion_matrix,
    accuracy_score as _sk_accuracy,
)


def tau_for_abstain_frac(scores_calib: np.ndarray, abstain_frac: float) -> float:
    s = np.asarray(scores_calib, dtype=float)
    q = 1.0 - float(abstain_frac)
    return float(np.quantile(s, q, method="nearest"))

def _ece_binary_subset(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    if len(p) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(p, edges) - 1
    ece = 0.0
    for b in range(len(edges) - 1):
        m = (idx == b)
        if not m.any():
            continue
        conf = 1.0 - float(p[m].mean())  # predicted correctness
        acc  = 1.0 - float(y[m].mean())  # observed correctness
        ece += (m.mean()) * abs(acc - conf)
    return float(ece)

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
    tau   : abstain if p_hat >= tau
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_hat, dtype=float)
    abstain = (p >= tau)
    keep    = ~abstain

    n = float(len(y))
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())

    TP = float(((abstain) & (y == 1)).sum())
    FP = float(((abstain) & (y == 0)).sum())
    FN = float(((keep)    & (y == 1)).sum())
    TN = float(((keep)    & (y == 0)).sum())

    coverage         = float(keep.mean())
    abstain_rate     = 1.0 - coverage

    kept_error = float(y[keep].mean()) if keep.any() else float("nan")
    selective_accuracy = 1.0 - kept_error if keep.any() else float("nan")

    abstention_precision = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
    abstention_recall    = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
    if np.isfinite(abstention_precision) and np.isfinite(abstention_recall) and (abstention_precision + abstention_recall) > 0:
        abstention_f1 = 2 * abstention_precision * abstention_recall / (abstention_precision + abstention_recall)
    else:
        abstention_f1 = float("nan")

    abstention_accuracy = (TP + TN) / n if n > 0 else float("nan")
    benign_answering_rate   = TN / n_neg if n_neg > 0 else float("nan")
    over_conservativeness   = FP / n_neg if n_neg > 0 else float("nan")

    brier_kept = float(np.mean((y[keep] - p[keep])**2)) if keep.any() else float("nan")
    abstain_ece_kept = _ece_binary_subset(y[keep], p[keep], bins=bins_ece) if keep.any() else float("nan")
    abstain_ece_abst = _ece_binary_subset(y[abstain], p[abstain], bins=bins_ece) if abstain.any() else float("nan")

    return dict(
        tau=float(tau),
        coverage=coverage,
        abstain_rate=abstain_rate,
        selective_accuracy=selective_accuracy,
        kept_error=kept_error,
        abstention_accuracy=abstention_accuracy,
        abstention_precision=abstention_precision,
        abstention_recall=abstention_recall,
        abstention_f1=abstention_f1,
        benign_answering_rate=benign_answering_rate,
        over_conservativeness=over_conservativeness,
        brier_kept=brier_kept,
        abstain_ece_kept=abstain_ece_kept,
        abstain_ece_abstained=abstain_ece_abst
    )

def build_abstention_table_from_targets(
    y_test: np.ndarray,
    p_test_calibrated: np.ndarray,
    p_calib_blend: np.ndarray,
    abstain_targets = (0.05, 0.10, 0.20, 0.30, 0.40),
    bins_ece: int = 15
) -> pd.DataFrame:
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
    os.makedirs(out_dir, exist_ok=True)
    AURC, AUACC, cov, risk, acc = None, None, None, None, None
    cov, risk, acc, AURC, AUACC = risk_coverage_curves(y_test, p_test_calibrated)

    plt.figure()
    plt.plot(cov, risk, lw=2)
    plt.xlabel("Coverage"); plt.ylabel("Risk (error rate among kept)")
    plt.title("Risk–Coverage (Test, calibrated)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "risk_coverage.png")); plt.close()

    plt.figure()
    plt.plot(cov, acc, lw=2)
    plt.xlabel("Coverage"); plt.ylabel("Accuracy among kept)")
    plt.title("Accuracy–Coverage (Test, calibrated)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_coverage.png")); plt.close()

    cov_at = {f"Coverage@Acc≥{a:.2f}": coverage_at_accuracy(y_test, p_test_calibrated, a) for a in acc_targets}
    abst_table.to_csv(os.path.join(out_dir, "abstention_table.csv"), index=False)
    return dict(AURC=AURC, AUACC=AUACC, **cov_at)


# ------------------------------------------------------------------------------------
# HF "evaluate-metric" wrappers with safe fallbacks
# ------------------------------------------------------------------------------------
def _roc_auc(y_true: np.ndarray, p_score: np.ndarray) -> float:
    """
    Prefer HF evaluate-metric/roc_auc; fallback to sklearn.metrics.roc_auc_score.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_score, dtype=float)
    if np.unique(y).size < 2:
        return float("nan")
    if _HAS_EVALUATE:
        try:
            m = evaluate.load("evaluate-metric/roc_auc")
            out = m.compute(references=y.tolist(), predictions=p.tolist())
            return float(out.get("roc_auc", float("nan")))
        except Exception:
            pass
    return float(_sk_roc_auc(y, p))

def _brier(y_true: np.ndarray, p_prob: np.ndarray) -> float:
    """
    Prefer HF evaluate-metric/brier_score; fallback to sklearn brier_score_loss.
    (Note: expects probabilities for the "positive" label.)
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_prob, dtype=float)
    if _HAS_EVALUATE:
        try:
            m = evaluate.load("evaluate-metric/brier_score")
            out = m.compute(references=y.tolist(), predictions=p.tolist())
            # some versions return {'brier_score': ...}
            if "brier_score" in out:
                return float(out["brier_score"])
        except Exception:
            pass
    return float(_sk_brier(y, p))

def _accuracy(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> float:
    """
    Prefer HF evaluate-metric/accuracy; fallback to sklearn accuracy_score.
    """
    y = np.asarray(y_true_bin).astype(int)
    z = np.asarray(y_pred_bin).astype(int)
    if _HAS_EVALUATE:
        try:
            m = evaluate.load("evaluate-metric/accuracy")
            out = m.compute(references=y.tolist(), predictions=z.tolist())
            if "accuracy" in out:
                return float(out["accuracy"])
        except Exception:
            pass
    return float(_sk_accuracy(y, z))

def _confusion_matrix(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> np.ndarray:
    """
    Prefer HF evaluate-metric/confusion_matrix; fallback to sklearn confusion_matrix.
    Returns a 2x2 ndarray [[TN, FP], [FN, TP]].
    """
    y = np.asarray(y_true_bin).astype(int)
    z = np.asarray(y_pred_bin).astype(int)
    if _HAS_EVALUATE:
        try:
            m = evaluate.load("evaluate-metric/confusion_matrix")
            out = m.compute(references=y.tolist(), predictions=z.tolist())
            # HF returns a dict with 'confusion_matrix' list of lists
            cm = np.array(out.get("confusion_matrix", []), dtype=float)
            if cm.size == 4:
                return cm
        except Exception:
            pass
    # sklearn returns by label order; ensure labels=[0,1]
    return _sk_confusion_matrix(y, z, labels=[0, 1]).astype(float)

def tau_for_abstain_frac(scores_calib: np.ndarray, abstain_frac: float) -> float:
    """
    Given error scores s (higher = more likely error), choose τ s.t. fraction above τ ≈ abstain_frac.
    """
    s = np.asarray(scores_calib, dtype=float)
    q = 1.0 - float(abstain_frac)
    return float(np.quantile(s, q, method="nearest"))

def selective_metrics(y_true: np.ndarray, p_err: np.ndarray, tau: float, bins_ece: int = 15) -> Dict[str, float]:
    """
    Compute coverage and probability quality on the *kept* set (p_err < tau).
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_err, dtype=float)
    keep = p < tau
    coverage = float(keep.mean())
    if keep.any():
        kept_err = float(y[keep].mean())
        sel_acc  = 1.0 - kept_err
        brier_kept = _brier(y[keep], p[keep])  # Brier on P(error)
        ece_kept   = ece_binary(y[keep], p[keep], bins=bins_ece)
    else:
        kept_err = sel_acc = brier_kept = ece_kept = float("nan")
    return dict(
        coverage=coverage,
        selective_accuracy=sel_acc,
        kept_error=kept_err,
        brier_kept=brier_kept,
        ece_kept=ece_kept,
    )

# ------------------------------------------------------------------------------------
# Vanilla baseline (confidence) from df
# ------------------------------------------------------------------------------------
def vanilla_confidence_series(df: pd.DataFrame) -> np.ndarray:
    """
    Returns a confidence in [0,1] for the *vanilla model* being correct.
    Priority: 'p_opt_chosen' (probability mass of chosen option).
    Fallback: normalized 'p_opt_gap' if present; otherwise median-filled zeros.
    """
    c = pd.to_numeric(df.get("p_opt_chosen"), errors="coerce")
    if c.isna().all():
        margin = pd.to_numeric(df.get("p_opt_gap"), errors="coerce")
        if margin.notna().any():
            z = (margin - margin.min()) / (margin.max() - margin.min() + 1e-12)
            c = z
        else:
            c = pd.Series(0.0, index=df.index)
    c = c.fillna(c.median()).clip(0, 1)
    return c.values.astype(float)

# ------------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------------
def _save_reliability_plot(y: np.ndarray, p_err: np.ndarray, out_path: str, bins: int = 12, title: str = "Reliability (P(error))") -> None:
    tab = reliability_table(y, p_err, bins=bins)
    # Expected correct prob per bin is (1 - p_mean(Perr)), observed is (1 - err_rate)
    x = 0.5 * (tab["p_lo"].values + tab["p_hi"].values)
    yhat = 1.0 - tab["p_mean(Perr)"].values
    yobs = 1.0 - tab["err_rate"].values

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.scatter(yhat, yobs)
    for i, n in enumerate(tab["n"].values):
        plt.annotate(str(int(n)), (yhat[i], yobs[i]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    plt.xlabel("Predicted correctness (1 - P(error))")
    plt.ylabel("Observed correctness")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def _save_risk_coverage_plot(y: np.ndarray, p_err: np.ndarray, out_path: str, title: str = "Risk–Coverage Curve") -> Dict[str, float]:
    cov, risk, acc, AURC, AUACC = risk_coverage_curves(y, p_err)
    plt.figure(figsize=(6, 5))
    plt.plot(cov, risk)
    plt.xlabel("Coverage")
    plt.ylabel("Risk = error rate among kept")
    plt.title(f"{title}\nAURC={AURC:.4f}, AUACC={AUACC:.4f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return {"AURC": AURC, "AUACC": AUACC}

# ------------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------------
@dataclass
class ReportInputs:
    df_test: pd.DataFrame
    p_err_test: np.ndarray            # detector calibrated P(error) on test set
    df_calib: Optional[pd.DataFrame] = None
    p_err_calib: Optional[np.ndarray] = None
    abstain_targets: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.40)
    ece_bins: int = 15
    rel_bins: int = 12
    compute_bootstrap: bool = False
    n_boot: int = 1000
    alpha: float = 0.05
    seed: int = 1337
    # threshold for "binary" summaries (confusion matrix, accuracy)
    # If None, we pick 0.5 by default. If you pass a float, it will be used.
    classification_threshold: Optional[float] = 0.5

def generate_report(
    inp: ReportInputs,
    out_dir: str = "report_out",
    prefix: str = "truthfulqa"
) -> Dict[str, Any]:
    """
    Creates a set of metrics, tables, and plots. Saves:
    - JSON summary,
    - CSV tables (reliability bins, abstention table, improvements table),
    - PNG plots (reliability, risk-coverage).
    Returns a dict with all computed stats.
    """
    os.makedirs(out_dir, exist_ok=True)
    df_te = inp.df_test.copy()
    y_te = (df_te["chosen"].astype(str).str.upper() != df_te["gold"].astype(str).str.upper()).astype(int).values
    p_te = np.asarray(inp.p_err_test, dtype=float)

    # ======= Detector (no abstention) =======
    det_core = {
        "AUROC": float(_roc_auc(y_te, p_te)),
        "AUPRC": float(_sk_ap(y_te, p_te)),  # AP not in evaluate-metric; keep sklearn
        "Brier": float(_brier(y_te, p_te)),
        "ECE":   float(ece_binary(y_te, p_te, bins=inp.ece_bins)),
    }

    # Reliability (detector)
    rel_det = reliability_table(y_te, p_te, bins=inp.rel_bins)

    # ======= Vanilla baseline from confidence =======
    conf_te = vanilla_confidence_series(df_te)
    base_err_te = 1.0 - conf_te
    base_core = {
        "AUROC": float(_roc_auc(y_te, base_err_te)),
        "AUPRC": float(_sk_ap(y_te, base_err_te)),
        "Brier": float(_brier(y_te, base_err_te)),
        "ECE":   float(ece_binary(y_te, base_err_te, bins=inp.ece_bins)),
    }

    # ======= Binary summaries (confusion matrix / accuracy) for a fixed threshold =======
    thr = 0.5 if inp.classification_threshold is None else float(inp.classification_threshold)
    y_pred_det_bin  = (p_te >= thr).astype(int)           # predict "error" if P(error) >= thr
    y_pred_base_bin = (base_err_te >= thr).astype(int)

    det_binary = {
        "threshold": thr,
        "accuracy": float(_accuracy(y_te, y_pred_det_bin)),
        "confusion_matrix": _confusion_matrix(y_te, y_pred_det_bin).tolist(),  # [[TN,FP],[FN,TP]]
    }
    base_binary = {
        "threshold": thr,
        "accuracy": float(_accuracy(y_te, y_pred_base_bin)),
        "confusion_matrix": _confusion_matrix(y_te, y_pred_base_bin).tolist(),
    }

    # ======= Abstention table (taus chosen on calib if provided; else chosen on test as fallback) =======
    if inp.p_err_calib is not None:
        p_ca = np.asarray(inp.p_err_calib, dtype=float)
        df_ca = inp.df_calib if inp.df_calib is not None else df_te.sample(frac=0.25, random_state=inp.seed)
        conf_ca = vanilla_confidence_series(df_ca) if inp.df_calib is not None else vanilla_confidence_series(df_te)
        base_err_ca = 1.0 - conf_ca
    else:
        # Fallback (not ideal for a paper; but keeps code usable during exploration)
        p_ca = p_te.copy()
        df_ca = df_te.copy()
        conf_ca = conf_te.copy()
        base_err_ca = 1.0 - conf_ca

    rows_abst = []
    for a in inp.abstain_targets:
        tau_det  = tau_for_abstain_frac(p_ca, a)
        tau_base = tau_for_abstain_frac(base_err_ca, a)

        m_det  = selective_metrics(y_te, p_te,          tau_det,  bins_ece=inp.ece_bins)
        m_base = selective_metrics(y_te, base_err_te,   tau_base, bins_ece=inp.ece_bins)

        rows_abst.append({
            "abstain_target": float(a),
            "tau_det":  float(tau_det),
            "tau_base": float(tau_base),
            "coverage_det":  m_det["coverage"],
            "coverage_base": m_base["coverage"],
            "selective_acc_det":  m_det["selective_accuracy"],
            "selective_acc_base": m_base["selective_accuracy"],
            "kept_err_det":  m_det["kept_error"],
            "kept_err_base": m_base["kept_error"],
            "brier_kept_det":  m_det["brier_kept"],
            "brier_kept_base": m_base["brier_kept"],
            "ece_kept_det(Perr)":  m_det["ece_kept"],
            "ece_kept_base(Perr)": m_base["ece_kept"],
            # deltas (kept pool quality)
            "abs_err_reduction":  m_base["kept_error"] - m_det["kept_error"],
            "rel_err_reduction_%": 100.0 * ((m_base["kept_error"] - m_det["kept_error"]) / (m_base["kept_error"] + 1e-12)),
        })
    abst_table = pd.DataFrame(rows_abst)

    # ======= Global selective curves (AURC/AUACC) and coverage@accuracy =======
    _, _, _, AURC_det,  AUACC_det  = risk_coverage_curves(y_te, p_te)
    _, _, _, AURC_base, AUACC_base = risk_coverage_curves(y_te, base_err_te)
    curves = {
        "AURC_det":  float(AURC_det),
        "AURC_base": float(AURC_base),
        "ΔAURC":     float(AURC_base - AURC_det),
        "AUACC_det":  float(AUACC_det),
        "AUACC_base": float(AUACC_base),
        "ΔAUACC":     float(AUACC_det - AUACC_base),
        "Coverage@Acc≥0.80_det":  float(coverage_at_accuracy(y_te, p_te, 0.80)),
        "Coverage@Acc≥0.80_base": float(coverage_at_accuracy(y_te, base_err_te, 0.80)),
        "ΔCov@0.80":  float(coverage_at_accuracy(y_te, p_te, 0.80) - coverage_at_accuracy(y_te, base_err_te, 0.80)),
        "Coverage@Acc≥0.85_det":  float(coverage_at_accuracy(y_te, p_te, 0.85)),
        "Coverage@Acc≥0.85_base": float(coverage_at_accuracy(y_te, base_err_te, 0.85)),
        "ΔCov@0.85":  float(coverage_at_accuracy(y_te, p_te, 0.85) - coverage_at_accuracy(y_te, base_err_te, 0.85)),
        "Coverage@Acc≥0.90_det":  float(coverage_at_accuracy(y_te, p_te, 0.90)),
        "Coverage@Acc≥0.90_base": float(coverage_at_accuracy(y_te, base_err_te, 0.90)),
        "ΔCov@0.90":  float(coverage_at_accuracy(y_te, p_te, 0.90) - coverage_at_accuracy(y_te, base_err_te, 0.90)),
        "Coverage@Acc≥0.95_det":  float(coverage_at_accuracy(y_te, p_te, 0.95)),
        "Coverage@Acc≥0.95_base": float(coverage_at_accuracy(y_te, base_err_te, 0.95)),
        "ΔCov@0.95":  float(coverage_at_accuracy(y_te, p_te, 0.95) - coverage_at_accuracy(y_te, base_err_te, 0.95)),
    }

    # ======= Optional bootstrap CIs for AURC/AUACC (on detector only) =======
    cis = {}
    if inp.compute_bootstrap:
        lo, hi, pt = bootstrap_ci(lambda Y, P: risk_coverage_curves(Y, P)[3], y_te, p_te, n_boot=inp.n_boot, alpha=inp.alpha, seed=inp.seed)
        cis["AURC_det_CI"] = {"lo": lo, "hi": hi, "point": pt}
        lo, hi, pt = bootstrap_ci(lambda Y, P: risk_coverage_curves(Y, P)[4], y_te, p_te, n_boot=inp.n_boot, alpha=inp.alpha, seed=inp.seed)
        cis["AUACC_det_CI"] = {"lo": lo, "hi": hi, "point": pt}

    # ======= Plots =======
    rel_plot_path = os.path.join(out_dir, f"{prefix}_reliability_detector.png")
    _save_reliability_plot(y_te, p_te, rel_plot_path, bins=inp.rel_bins, title="Reliability (Detector P(error))")

    riskcov_plot_path = os.path.join(out_dir, f"{prefix}_risk_coverage_detector.png")
    rc_vals = _save_risk_coverage_plot(y_te, p_te, riskcov_plot_path, title="Risk–Coverage (Detector)")

    # Optional: vanilla reliability/risk-coverage for appendix
    rel_plot_base = os.path.join(out_dir, f"{prefix}_reliability_vanilla.png")
    _save_reliability_plot(y_te, base_err_te, rel_plot_base, bins=inp.rel_bins, title="Reliability (Vanilla P(error)=1-conf)")

    riskcov_plot_base = os.path.join(out_dir, f"{prefix}_risk_coverage_vanilla.png")
    _save_risk_coverage_plot(y_te, base_err_te, riskcov_plot_base, title="Risk–Coverage (Vanilla)")

    # ======= Save tables =======
    rel_csv = os.path.join(out_dir, f"{prefix}_reliability_bins_detector.csv")
    rel_det.to_csv(rel_csv, index=False)

    abst_csv = os.path.join(out_dir, f"{prefix}_abstention_table.csv")
    abst_table.to_csv(abst_csv, index=False)

    # Improvement snapshot (one-line summary)
    imp_summary = {
        "detector_core": det_core,
        "vanilla_core": base_core,
        "binary_detector@thr": det_binary,
        "binary_vanilla@thr":  base_binary,
        "abstention_table_path": abst_csv,
        "reliability_bins_path": rel_csv,
        "plots": {
            "detector_reliability_png": rel_plot_path,
            "detector_risk_coverage_png": riskcov_plot_path,
            "vanilla_reliability_png": rel_plot_base,
            "vanilla_risk_coverage_png": riskcov_plot_base,
        },
        "curves": curves,
        "bootstrap_CIs": cis,
        "notes": {
            "metrics_backend": "evaluate-metric (if available) + sklearn fallback",
            "ece_bins": inp.ece_bins,
            "reliability_bins": inp.rel_bins,
            "classification_threshold": thr,
        }
    }

    # JSON dump
    json_path = os.path.join(out_dir, f"{prefix}_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(imp_summary, f, indent=2)

    return imp_summary


# ------------------------------------------------------------------------------------
# (Optional) Predictor helper to produce p_err from your trained blend+calibrator
# ------------------------------------------------------------------------------------
def build_X_like_train(df: pd.DataFrame, meta: Dict[str, Any]) -> np.ndarray:
    """
    Build feature matrix in the *exact* training space:
      meta["num_cols"], meta["imputer"], meta["scaler"]
    """
    num_cols = meta["num_cols"]
    X_df = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
                         for c in num_cols})
    X_imp = meta["imputer"].transform(X_df)
    X_std = meta["scaler"].transform(X_imp)
    return X_std.astype(np.float32)

def predict_detector_p_err(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    nn_model,                     # trained MLP (PyTorch)
    tree_model,                   # LightGBM or HGB
    w_opt: float,
    calibrator,                   # isotonic / platt / beta (your best_cal)
    device: Optional[str] = None,
    batch: int = 4096,
) -> np.ndarray:
    """
    Generate *calibrated* detector P(error) for a new dataframe df using your final
    blend (w_opt) and calibrator. This mirrors your training notebook’s inference.
    """
    X = build_X_like_train(df, meta)
    # NN
    import torch
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    nn_model.to(dev).eval()
    ps_nn = []
    with torch.inference_mode():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i+batch]).to(dev)
            ps_nn.append(torch.sigmoid(nn_model(xb)).float().cpu().numpy().ravel())
    p_nn = np.concatenate(ps_nn)

    # Tree
    if hasattr(tree_model, "predict_proba"):
        if "pandas" in tree_model.__class__.__module__ or hasattr(tree_model, "booster_"):
            X_df = pd.DataFrame(X, columns=meta["num_cols"])
            p_tree = tree_model.predict_proba(X_df)[:, 1]
        else:
            p_tree = tree_model.predict_proba(X)[:, 1]
    else:
        # safety
        p_tree = np.zeros(len(X), dtype=float)

    p_blend = float(w_opt) * p_nn + (1.0 - float(w_opt)) * p_tree

    # Calibrate
    if hasattr(calibrator, "transform"):
        p_hat = calibrator.transform(p_blend)
    else:
        # isotonic in sklearn uses transform; this branch is for defensive coding
        p_hat = calibrator.predict_proba(p_blend.reshape(-1, 1))[:, 1]
    return p_hat.astype(float)