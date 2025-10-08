# src/halu/features/build.py
from __future__ import annotations
from typing import Dict, Iterable, Tuple, Any, Optional
import numpy as np
import pandas as pd
from halu.features.feature_metrics import (
    softmax_stable,
    entropy_from_logits,
    margin_from_logits,
    maxprob_from_logits,
)

def feature_row_from_logits(
    ex: Dict[str, Any],
    response_logits: np.ndarray,   # [T, V]
    letter_id_map: Dict[str, int], # e.g., {"A":0,"B":1,"C":2,"D":3}
) -> Dict[str, Any]:
    """
    Build a single feature row for an MCQ example from response-region logits.
    Policy:
      - If last-step logits contain any non-finite value, return NaNs for p_*, last_*,
        aggregates, empty pred_letter, and NaN is_error (strict propagation).
      - Otherwise compute aggregates with finite-only masking (robust to occasional NaNs earlier).
    """
    row: Dict[str, Any] = {
        "qid": ex.get("qid", ""),
        "dataset": ex.get("dataset", ""),
    }

    x = np.asarray(response_logits, dtype=float)
    if x.ndim == 2 and x.size > 0:
        T, V = int(x.shape[0]), int(x.shape[1])
    elif x.ndim == 1 and x.size > 0:
        T, V = 1, int(x.shape[0])
        x = x[None, :]  # normalize to [T,V]
    else:
        T, V = 0, 0
    row["T"], row["V"] = T, V

    # Empty/degenerate → fill NaNs and return
    if T == 0 or V == 0:
        row.update({
            "ent_mean": np.nan, "ent_std": np.nan,
            "margin_mean": np.nan, "maxprob_mean": np.nan,
            "last_entropy": np.nan, "last_maxprob": np.nan,
        })
        for L in letter_id_map.keys():
            row[f"p_{L}"] = np.nan
        row["pred_letter"] = ""
        row["is_error"] = np.nan
        return row

    # ---- Robust aggregates over time (finite-only) ----
    H = entropy_from_logits(x)       # [T]
    M = margin_from_logits(x)        # [T]
    Q = maxprob_from_logits(x)       # [T]

    def _mean_finite(v):
        v = np.asarray(v, dtype=float)
        m = np.isfinite(v)
        return float(v[m].mean()) if m.any() else np.nan

    def _std_finite(v):
        v = np.asarray(v, dtype=float)
        m = np.isfinite(v)
        return float(v[m].std(ddof=0)) if m.any() else np.nan

    row["ent_mean"]     = _mean_finite(H)
    row["ent_std"]      = _std_finite(H)
    row["margin_mean"]  = _mean_finite(M)
    row["maxprob_mean"] = _mean_finite(Q)
    row["last_entropy"] = float(H[-1]) if np.isfinite(H[-1]) else np.nan
    row["last_maxprob"] = float(Q[-1]) if np.isfinite(Q[-1]) else np.nan

    # ---- Last-step probabilities & prediction ----
    last_ok = np.all(np.isfinite(x[-1]))
    if not last_ok:
        # strict propagation for classification-side fields
        for L in letter_id_map.keys():
            row[f"p_{L}"] = np.nan
        row["pred_letter"] = ""
        row["is_error"] = np.nan
        return row

    P_last = softmax_stable(x[-1][None, :])[0]
    probs: Dict[str, float] = {}
    for L, idx in letter_id_map.items():
        idx = int(idx)
        if 0 <= idx < V:
            val = float(P_last[idx])
            probs[L] = val
            row[f"p_{L}"] = val
        # (If a letter is in the map but the index is out-of-range, we skip it—
        # consistent with prior behavior and tests.)

    pred_letter = max(probs, key=probs.get) if probs else ""
    row["pred_letter"] = pred_letter

    gold = (ex.get("gold_letter") or "").strip().upper()
    row["is_error"] = (0 if pred_letter == gold else 1) if (pred_letter and gold) else np.nan
    return row

def build_features_df_from_pairs(
    pairs: Iterable[Tuple[Dict[str, Any], np.ndarray]],
    letter_id_map: Dict[str, int],
) -> pd.DataFrame:
    """
    pairs: iterable of (example_dict, response_logits) produced by any runner.
    Returns a DataFrame with one row per example.
    """
    rows = [feature_row_from_logits(ex, logits, letter_id_map) for (ex, logits) in pairs]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# -----------------------------------------------------------------------------
# 2) Integration path (HFRunner + pipeline). Heavy imports moved inside function
# -----------------------------------------------------------------------------

def build_features_df(
    model,
    tokenizer,
    dataset_name: str,
    size: Optional[int] = None,
    seed: Optional[int] = 1337,
) -> pd.DataFrame:
    """
    End-to-end builder that:
      - runs HFRunner over a dataset
      - computes option-side features via MetricsPipeline
      - collapses to per-example features
    Heavy deps are imported here to keep module import light for unit tests.
    """
    # Local imports so `import halu.features.build` doesn't require torch/HF
    from halu.core.runner import HFRunner
    from halu.pipeline import MetricsPipeline
    from halu.data.registry import pick_dataset
    from halu.features.tables import build_option_table, collapse_option_side_features
    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(x, *a, **k):  # fallback if tqdm missing
            return x

    runner = HFRunner(
        model, tokenizer,
        store_device="cpu",
        store_dtype=getattr(model, "dtype", None)
    )
    pipe = MetricsPipeline(runner)
    dataset = pick_dataset(dataset_name)

    recs = []
    for ex in tqdm(dataset.iter_examples(sample_size=size, seed=seed), "Building examples"):
        recs.append(pipe.example_to_row(ex))

    opt = build_option_table(recs)
    out = collapse_option_side_features(opt)

    print(out.head())
    print(out.info())
    print(out.describe())

    return out