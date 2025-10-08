# tests/test_no_leakage.py
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoModelForCausalLM

from halu.model.ensemble import build_xy
from halu.engine import DetectorEnsemble, DetectorConfig
from halu.features.build import build_features_df


def _y_from_df(df: pd.DataFrame) -> np.ndarray:
    return (df["chosen"].astype(str).str.upper() != df["gold"].astype(str).str.upper()).astype(int).values


def _apply_target_by_editing_gold(df: pd.DataFrame, y_target: np.ndarray) -> pd.DataFrame:
    """
    Make the dataframe's labels equal to y_target **without** touching feature columns.
    y=0 -> set gold == chosen ; y=1 -> set gold != chosen (pick a different letter).
    """
    df2 = df.copy()
    letters = ["A", "B", "C", "D", "E"]
    g_idx = df2.columns.get_loc("gold")

    chosen_series = df2["chosen"].astype(str).str.upper().tolist()
    for i, (ch, y) in enumerate(zip(chosen_series, y_target)):
        if y == 0:
            df2.iat[i, g_idx] = ch
        else:
            alt = next(L for L in letters if L != ch)
            df2.iat[i, g_idx] = alt
    return df2


def _tiny_detector_cfg(seed=1337) -> DetectorConfig:
    # Light & fast; enough to learn real signal, not enough to learn noise.
    return DetectorConfig(
        seed=seed,
        hidden=64,
        epochs=4,
        lr=3e-3,
        weight_decay=2e-4,
        gamma=2.0,
        calib_size=0.25,
        val_size=0.20,
        blend_grid=11,
        device="cpu",
        batch=256,
    )


def test_build_xy_excludes_reserved_columns():
    # Synthetic frame with some reserved + numeric cols
    df = pd.DataFrame({
        "qid":   ["q0","q1","q2","q3"],
        "gold":  ["A","B","C","D"],
        "chosen":["A","A","D","B"],
        # numeric feature cols
        "p_opt_chosen": [0.9,0.2,0.6,0.5],
        "p_opt_gap":    [0.8,0.1,0.2,0.4],
        "fcm_entropy":  [0.1,0.6,0.3,0.2],
        # string non-numeric feature (should be excluded)
        "opt_top_letter": ["A","C","B","D"],
    })

    X, y, meta = build_xy(df)
    num_cols = meta["num_cols"]

    # Must not include direct labels / string columns
    forbidden = {"qid", "gold", "chosen", "opt_top_letter", "fcm_top_letter"}
    assert not (forbidden & set(num_cols)), f"Forbidden columns leaked into features: {forbidden & set(num_cols)}"

    # Types & shapes look sane
    assert X.dtype == np.float32
    assert y.dtype in (np.int64, np.int32)
    assert X.shape[0] == y.shape[0] == len(df)


@pytest.mark.hf
@pytest.mark.slow
def test_label_permutation_breaks_signal_qwen25_truthfulqa():
    """
    End-to-end leakage check on real features:
      - Train detector on TRAIN -> AUROC on TEST should be clearly > 0.5
      - Permute TRAIN labels and retrain -> AUROC on TEST should drop near 0.5
    If AUROC stays high after permutation, features or pipeline leak target info.
    """
    model_id = "Qwen/Qwen2.5-0.5B"

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"
    tok.truncation_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cpu", torch_dtype="auto", trust_remote_code=True, attn_implementation="eager"
    ).eval()

    df_all = build_features_df(model, tok, "truthfulqa", size=20, seed=1337)

    # Standard split; stratify if labels both present
    y_all = _y_from_df(df_all)
    strat = y_all if np.unique(y_all).size >= 2 else None
    df_tr, df_te = train_test_split(df_all, test_size=0.30, random_state=1337, stratify=strat)

    # Extra guard: no qid overlap
    assert set(df_tr["qid"]).isdisjoint(set(df_te["qid"])), "Train/Test qids overlap — split bug."

    # --- Baseline detector (true labels) ---
    det_cfg = _tiny_detector_cfg(seed=1337)
    det = DetectorEnsemble(det_cfg)
    det.fit(df_tr)
    p_te = det.predict_proba(df_te)
    y_te = _y_from_df(df_te)

    auroc_true = roc_auc_score(y_te, p_te)
    # Should be meaningfully above chance on real features; allow slack across HF/model versions
    assert auroc_true > 0.6, f"Unexpectedly weak signal with true labels (AUROC={auroc_true:.3f})."

    # --- Permutation test: destroy label-feature relation on TRAIN only ---
    rng = np.random.default_rng(42)
    y_tr = _y_from_df(df_tr)
    y_perm = rng.permutation(y_tr)
    df_tr_perm = _apply_target_by_editing_gold(df_tr, y_perm)

    det2 = DetectorEnsemble(_tiny_detector_cfg(seed=4242))
    det2.fit(df_tr_perm)
    p_te_perm = det2.predict_proba(df_te)

    auroc_perm = roc_auc_score(y_te, p_te_perm)

    # After permutation, AUROC on the *same* TEST labels should collapse toward chance.
    # We set a generous upper bound to allow small fluctuations.
    assert auroc_perm < 0.6, f"AUROC with permuted training labels is too high (AUROC={auroc_perm:.3f}) → potential leakage."

    # And the drop should be noticeable
    assert (auroc_true - auroc_perm) > 0.2, f"AUROC drop too small: true={auroc_true:.3f}, perm={auroc_perm:.3f}"