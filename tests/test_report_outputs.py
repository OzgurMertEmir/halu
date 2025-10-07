# tests/test_report_outputs.py
import json, os
import numpy as np
import pandas as pd
from halu.analysis.report import ReportInputs, generate_report

def _make_df(n=140, seed=0):
    rng = np.random.default_rng(seed)
    qid = [f"q{i:04d}" for i in range(n)]
    p_top = rng.uniform(0.45, 0.95, n)
    p_second = np.clip(p_top - rng.uniform(0.05, 0.35, n), 0, 1)
    gap = np.clip(p_top - p_second, 0, 1)
    y = (rng.random(n) < (0.5*(1-p_top) + 0.3*(1-gap))).astype(int)
    df = pd.DataFrame({
        "qid": qid, "dataset": "toy", "gold": "A",
        "chosen": np.where(y == 1, "B", "A"),
        "p_opt_top": p_top, "p_opt_second": p_second, "p_opt_gap": gap
    })
    # turn the same signal into a probability
    z = -3.0 + 6.0 * (1 - p_top + 0.5*(1 - gap))
    p = 1 / (1 + np.exp(-z))
    return df, p

def test_generate_report_outputs(tmp_path):
    df, p = _make_df(160, seed=42)
    out_dir = tmp_path / "rep"
    out_dir.mkdir()
    summary = generate_report(
        ReportInputs(df_test=df, p_err_test=p, rel_bins=10, ece_bins=10, compute_bootstrap=False),
        out_dir=str(out_dir),
        prefix="unit"
    )
    # files
    expect = [
        "unit_report.json",
        "unit_reliability_bins_detector.csv",
        "unit_abstention_table.csv",
        "unit_reliability_detector.png",
        "unit_risk_coverage_detector.png",
        "unit_reliability_vanilla.png",
        "unit_risk_coverage_vanilla.png",
    ]
    for fn in expect:
        path = out_dir / fn
        assert path.exists(), f"missing {fn}"
        if fn.endswith(".png"):
            assert os.path.getsize(path) > 0

    # json schema
    with open(out_dir / "unit_report.json", "r", encoding="utf-8") as f:
        js = json.load(f)
    for k in ["detector_core", "vanilla_core", "binary_detector@thr", "binary_vanilla@thr", "curves", "plots"]:
        assert k in js

    # reliability CSV sanity
    bins = pd.read_csv(out_dir / "unit_reliability_bins_detector.csv")
    assert "n" in bins.columns and "err_rate" in bins.columns
    assert int(bins["n"].sum()) == len(df)