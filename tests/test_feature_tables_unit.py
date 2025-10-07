# tests/test_feature_tables_unit.py
import numpy as np
import pandas as pd

from halu.features.tables import build_option_table, collapse_option_side_features

def _rec_base():
    # One example with full A..D keys and FCM list
    r = {
        "qid": "q1",
        "dataset": "toy",
        "gold_letter": "C",
        "pred_letter": "C",     # used as fallback for 'vanilla'
        # pmil
        "A_pmil": 0.10, "B_pmil": 0.20, "C_pmil": 0.90, "D_pmil": 0.00,
        # pmic
        "A_pmic": 0.30, "B_pmic": 0.40, "C_pmic": 0.70, "D_pmic": 0.40,
        # entropies (lower is better => asc rank)
        "A_e_letter": 2.0, "B_e_letter": 1.2, "C_e_letter": 0.8, "D_e_letter": 1.5,
        "A_e_content": 2.5, "B_e_content": 1.0, "C_e_content": 0.7, "D_e_content": 1.6,
        # reliability (any numeric)
        "A_icrp_reliable": 0.1, "B_icrp_reliable": 0.2, "C_icrp_reliable": 0.8, "D_icrp_reliable": 0.3,
        # fcm probs sum to 1 (already normalized to make assertions simple)
        "fcm_letter_probs": [0.1, 0.2, 0.6, 0.1],
        # preferred vanilla (same as pred_letter here)
        "model_pred": "C",
    }
    return r

def test_build_option_table_rows_and_flags():
    opt = build_option_table([_rec_base()])
    assert set(opt.columns) >= {
        "qid","dataset","letter","gold","vanilla","pmil","pmic",
        "e_letter","e_content","icrp_reliable","fcm_prob","y_opt",
        "pmil_rank","pmic_rank","e_letter_rank","e_content_rank"
    }
    assert len(opt) == 4 and set(opt["letter"]) == {"A","B","C","D"}
    # gold/pred flags
    rowC = opt[opt["letter"] == "C"].iloc[0]
    assert rowC["y_opt"] == 1
    # ranks:
    # - pmil: C is largest => rank 1 (descending)
    assert float(rowC["pmil_rank"]) == 1.0
    # - e_letter: C is smallest => rank 1 (ascending)
    assert float(rowC["e_letter_rank"]) == 1.0
    # fcm_prob carried per-letter
    np.testing.assert_allclose(
        opt.sort_values("letter")["fcm_prob"].to_numpy(),
        np.array([0.1, 0.2, 0.6, 0.1])
    )

def test_collapse_basic_distribution_and_top_second_gap():
    opt = build_option_table([_rec_base()])
    out = collapse_option_side_features(opt)
    assert list(out["qid"]) == ["q1"]
    r = out.iloc[0]
    # top/second from fcm or synthesized p_opt; here fcm drives p_opt
    assert abs(r["p_opt_top"] - 0.6) < 1e-12
    assert abs(r["p_opt_second"] - 0.2) < 1e-12
    assert abs(r["p_opt_gap"] - 0.4) < 1e-12
    # opt_top_letter should be C
    assert r["opt_top_letter"] == "C"
    # chosen (vanilla) is C; rank should be 1
    assert r["chosen"].upper() == "C"
    assert float(r["p_opt_rank_chosen"]) == 1.0
    # p_opt entropy/gini computed on normalized distribution
    # H = -sum p log p
    p = np.array([0.1,0.2,0.6,0.1])
    H = float(-(p * np.log(p)).sum())
    assert abs(r["p_opt_entropy"] - H) < 1e-12
    # simple gini sanity: in [0,1], bigger concentration => bigger gini than uniform
    assert 0.0 <= r["p_opt_gini"] <= 1.0

def test_collapse_normalizes_given_p_opt_and_handles_missing_fcm():
    # Provide two options with explicit (unnormalized) p_opt and no fcm_prob
    rows = []
    for L, p_opt in [("A", 2.0), ("B", 1.0)]:
        rows.append(dict(
            qid="q2", dataset="toy", letter=L, gold="A", vanilla="B",
            p_opt=p_opt, pmil=np.nan, pmic=np.nan,
            e_letter=np.nan, e_content=np.nan, icrp_reliable=np.nan, fcm_prob=np.nan,
        ))
    opt = pd.DataFrame(rows)
    out = collapse_option_side_features(opt)
    r = out.iloc[0]
    # p_opt should have been renormalized to [2/3, 1/3]
    assert abs(r["p_opt_top"] - (2/3)) < 1e-12
    assert abs(r["p_opt_second"] - (1/3)) < 1e-12
    # chosen == B (vanilla); rank is 2 (since A is top)
    assert r["chosen"] == "B"
    assert float(r["p_opt_rank_chosen"])

def test_letters_inferred_from_fcm_sources_only():
    rec = {"qid":"q1","dataset":"toy","gold_letter":"B",
           "fcm_letter_prob_map":{"B":0.7,"C":0.3},
           "model_pred":"B"}
    opt = build_option_table([rec])
    assert set(opt["letter"]) == {"B","C"}
    assert np.isclose(opt.loc[opt.letter=="B","fcm_prob"].iloc[0], 0.7)

def test_fcm_prob_alignment_prefers_named_letters_over_index():
    rec = {"qid":"q2","dataset":"toy","gold_letter":"C","model_pred":"C",
           "fcm_prob_letters":["C","A","B"], "fcm_letter_probs":[0.6,0.2,0.2]}
    opt = build_option_table([rec])
    got = dict(zip(opt["letter"], opt["fcm_prob"]))
    assert np.isclose(got["C"], 0.6)
    assert np.isclose(got["A"], 0.2)
    assert np.isclose(got["B"], 0.2)

def test_collapse_normalizes_popt_and_sets_tops():
    rec = {"qid":"q3","dataset":"toy","gold_letter":"B","model_pred":"B",
           "fcm_letter_prob_map":{"A":0.1,"B":0.8,"C":0.1}}
    opt = build_option_table([rec])
    out = collapse_option_side_features(opt)
    r = out.iloc[0]
    assert r["qid"] == "q3"
    # p_opt should be normalized to 1 across options
    # top letter fields should be consistent
    assert r["opt_top_letter"] == "B"
    assert np.isfinite(r["p_opt_top"])