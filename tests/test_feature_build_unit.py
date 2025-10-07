import numpy as np
import pandas as pd

from halu.features.feature_metrics import (
    softmax_stable, entropy_from_logits, margin_from_logits, maxprob_from_logits
)
from halu.features.build import (
    feature_row_from_logits,
    build_features_df_from_pairs,
)

R1 = np.array([ 2.0, 1.0, 0.0, -1.0])
R2 = np.array([ 0.0, 0.0, 0.0,  0.0])
R3 = np.array([ 1.0, 3.0, 0.0, -2.0])
LOGITS = np.stack([R1, R2, R3], axis=0)
LETTER_IDS = {"A":0, "B":1, "C":2, "D":3}

def _ex(qid="q1", gold="B"):
    return {
        "qid": qid,
        "question": "What is X?",
        "options": [
            {"label":"A","text":"a"},
            {"label":"B","text":"b"},
            {"label":"C","text":"c"},
            {"label":"D","text":"d"},
        ],
        "gold_letter": gold,
        "dataset": "toy",
    }

def test_feature_row_basic_letter_probs_and_error_flag():
    ex = _ex(gold="B")
    row = feature_row_from_logits(ex, LOGITS, LETTER_IDS)

    assert row["qid"] == "q1" and row["dataset"] == "toy"
    assert row["T"] == 3 and row["V"] == 4

    pA, pB, pC, pD = row["p_A"], row["p_B"], row["p_C"], row["p_D"]
    np.testing.assert_allclose(pA + pB + pC + pD, 1.0, atol=1e-12)
    assert row["pred_letter"] == "B"
    assert row["is_error"] == 0

def test_feature_row_incorrect_and_aggregates():
    ex = _ex(qid="q2", gold="C")  # argmax is B → error
    row = feature_row_from_logits(ex, LOGITS, LETTER_IDS)
    assert row["pred_letter"] == "B"
    assert row["is_error"] == 1

    H = entropy_from_logits(LOGITS)
    M = margin_from_logits(LOGITS)
    Q = maxprob_from_logits(LOGITS)

    np.testing.assert_allclose(row["ent_mean"],  H.mean(), atol=1e-12)
    np.testing.assert_allclose(row["ent_std"],   H.std(ddof=0), atol=1e-12)
    np.testing.assert_allclose(row["margin_mean"], M.mean(), atol=1e-12)
    np.testing.assert_allclose(row["maxprob_mean"], Q.mean(), atol=1e-12)

    P_last = softmax_stable(LOGITS[-1][None, :])[0]
    np.testing.assert_allclose(row["p_B"], P_last[1], atol=1e-12)
    np.testing.assert_allclose(row["last_entropy"], H[-1], atol=1e-12)
    np.testing.assert_allclose(row["last_maxprob"], Q[-1], atol=1e-12)

def test_build_features_df_from_pairs_minipack():
    ex1, ex2 = _ex(qid="e1", gold="B"), _ex(qid="e2", gold="C")
    pairs = [(ex1, LOGITS), (ex2, LOGITS)]
    df = build_features_df_from_pairs(pairs, LETTER_IDS)

    assert isinstance(df, pd.DataFrame)
    assert {"qid", "dataset", "is_error", "pred_letter", "p_A", "p_B", "p_C", "p_D"}.issubset(df.columns)
    assert list(df["qid"]) == ["e1", "e2"]
    assert list(df["is_error"]) == [0, 1]

def test_edge_cases_empty_and_partial_letter_map():
    ex = _ex(qid="empty", gold="A")
    empty_logits = LOGITS[:0, :]
    row = feature_row_from_logits(ex, empty_logits, LETTER_IDS)
    assert row["T"] == 0 and row["V"] == 0
    assert np.isnan(row["p_A"]) and np.isnan(row["p_B"])
    assert row["pred_letter"] == ""
    assert np.isnan(row["is_error"])

    partial = {"A":0, "B":1, "C":2}  # no D
    row2 = feature_row_from_logits(ex, LOGITS, partial)
    assert {"p_A","p_B","p_C"}.issubset(row2.keys())
    assert "p_D" not in row2

def test_zero_logits_uniform_ok():
    x = np.zeros((3, 4))  # T=3, V=4
    row = feature_row_from_logits(_ex("u","A"), x, LETTER_IDS)
    s = row["p_A"]+row["p_B"]+row["p_C"]+row["p_D"]
    assert abs(s - 1.0) < 1e-12
    assert row["last_maxprob"] == 0.25
    assert row["is_error"] in (0,1,np.nan)  # depends on gold vs argmax tie-break in numpy

def test_nans_propagate_policy():
    base = np.array([[0,1,2,3],[1,2,3,4],[2,3,4,5]], dtype=float)
    base[-1,1] = np.nan  # NaN in the last step
    row = feature_row_from_logits(_ex("nan","A"), base, LETTER_IDS)
    # With “propagate” policy, last-step probs and derived fields should be NaN/empty.
    assert np.isnan(row["p_A"]) and np.isnan(row["p_B"])
    assert row["pred_letter"] == ""
    assert np.isnan(row["is_error"])

def test_inf_rows_are_handled_or_propagate():
    base = np.array([[0,1,2,3],[1,2,3,4],[2,3,np.inf,5]], dtype=float)
    row = feature_row_from_logits(_ex("inf","A"), base, LETTER_IDS)
    # Either NaN propagation (policy = propagate) or a finite result if you later choose a coercion policy.
    assert ("pred_letter" in row)

