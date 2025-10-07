# tests/test_engine_detector.py
import json
import numpy as np
import pandas as pd
import pytest
from dataclasses import asdict

from halu.engine.detector import DetectorEnsemble, DetectorConfig

pytestmark = pytest.mark.integration

# --- small synthetic dataset builder ----------------------------------------
def _make_df(n=240, seed=0):
    rng = np.random.default_rng(seed)
    qids = [f"q{i:04d}" for i in range(n)]

    # Features in [0,1] with some signal
    p_opt_top     = rng.uniform(0.4, 0.95, n)
    p_opt_second  = np.clip(p_opt_top - rng.uniform(0.05, 0.4, n), 0.0, 1.0)
    p_opt_gap     = np.clip(p_opt_top - p_opt_second, 0.0, 1.0)
    pmic_gap      = rng.uniform(0.0, 1.0, n)
    pmil_gap      = rng.uniform(0.0, 1.0, n)
    last_maxprob  = np.clip(p_opt_top + 0.05*rng.normal(size=n), 0.0, 1.0)

    # Latent "erroriness" score — higher means more likely error
    score = (1.0 - p_opt_top) + 0.5*(1.0 - p_opt_gap) + 0.2*(1.0 - last_maxprob) + 0.1*rng.normal(size=n)
    is_error = (score > np.quantile(score, 0.6)).astype(int)

    # chosen/gold: gold="A"; chosen flips to "B" on error
    gold   = np.array(["A"]*n, dtype=object)
    chosen = np.where(is_error == 1, "B", "A")

    df = pd.DataFrame({
        "qid": qids,
        "dataset": ["toy"]*n,
        "gold": gold,
        "chosen": chosen,
        # numeric features (names don’t matter; engine picks numeric cols except reserved)
        "p_opt_top": p_opt_top,
        "p_opt_second": p_opt_second,
        "p_opt_gap": p_opt_gap,
        "pmic_gap_top1_top2": pmic_gap,
        "pmil_gap_top1_top2": pmil_gap,
        "last_maxprob": last_maxprob,
    })
    return df

# --- tests -------------------------------------------------------------------

def test_engine_fit_predict_save_load_roundtrip(tmp_path):
    df = _make_df(220, seed=1)
    cfg = DetectorConfig(
        epochs=6, hidden=16, lr=3e-3, weight_decay=2e-4,
        calib_size=0.25, val_size=0.20, blend_grid=21,
        device="cpu", batch=128, seed=123
    )
    det = DetectorEnsemble(cfg)

    # fit
    summary = det.fit(df)
    assert 0.0 <= summary["blend_weight"] <= 1.0
    assert summary["calibrator"] in {"isotonic", "platt", "beta"}
    for k in ["calib_brier", "train_brier_uncal", "train_brier_cal"]:
        assert 0.0 <= float(summary[k]) <= 1.0

    # predict
    p0 = det.predict_proba(df)
    assert p0.shape == (len(df),)
    assert np.all(np.isfinite(p0)) and np.all((p0 >= 0) & (p0 <= 1))

    # save artifacts
    outdir = tmp_path / "detector"
    det.save(str(outdir))
    for fname in ["config.json", "meta.joblib", "tree.joblib", "calibrator.joblib", "nn.pt"]:
        assert (outdir / fname).exists()

    # load and re-predict — must be bitwise-close
    det2 = DetectorEnsemble.load(str(outdir))
    p1 = det2.predict_proba(df)
    np.testing.assert_allclose(p0, p1, rtol=0, atol=1e-8)

    # config round-trip (the dicts should match exactly)
    with open(outdir / "config.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    assert "config" in raw and "w_opt" in raw
    assert asdict(det2.cfg) == asdict(cfg)
    assert np.isclose(det2.w_opt, det.w_opt)

def test_engine_predict_before_fit_raises():
    det = DetectorEnsemble(DetectorConfig(device="cpu", epochs=1))
    with pytest.raises(AssertionError):
        det.predict_proba(_make_df(10))

def test_engine_missing_columns_are_imputed_and_order_irrelevant():
    df = _make_df(180, seed=2)
    det = DetectorEnsemble(DetectorConfig(epochs=5, hidden=8, device="cpu", blend_grid=11))
    det.fit(df)

    # drop one numeric feature + shuffle columns
    df_small = df.drop(columns=["pmil_gap_top1_top2"]).copy()
    df_small = df_small[df_small.columns[::-1]]  # reverse order
    p = det.predict_proba(df_small)

    assert p.shape == (len(df_small),)
    assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))

def test_engine_save_manifest_fields(tmp_path):
    df = _make_df(140, seed=3)
    det = DetectorEnsemble(DetectorConfig(epochs=4, hidden=8, device="cpu"))
    det.fit(df)
    outdir = tmp_path / "savechk"
    det.save(str(outdir))

    with open(outdir / "config.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    assert set(info.keys()) == {"config", "w_opt"}
    assert isinstance(info["w_opt"], (float, int))