# tests/test_detector_ensemble_unit.py
import os
import numpy as np
import pandas as pd
from tempfile import TemporaryDirectory

from halu.engine.detector import DetectorConfig, DetectorEnsemble

def _make_df(n=300, seed=123):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, size=n)
    q = 1 / (1 + np.exp(-z))            # true P(error)
    y = rng.binomial(1, q).astype(int)

    df = pd.DataFrame({
        # a few signal-ish features
        "p_opt_top": 1 - q,
        "p_opt_gap": 0.4 * (1 - q) + 0.05*rng.standard_normal(n),
        "pmic_gap_top1_top2": 0.3 * (1 - q) + 0.05*rng.standard_normal(n),
        "pmil_gap_top1_top2": 0.2 * (1 - q) + 0.05*rng.standard_normal(n),
        "fcm_entropy": -(q*np.log(q) + (1-q)*np.log(1-q)),
        "junk_noise": rng.standard_normal(n),

        # labels for error supervision
        "gold": ["A"]*n,
        "chosen": np.where(y == 1, "B", "A"),
    })
    # sprinkle NaN/Inf to test robustness
    df.loc[:2, "junk_noise"] = np.nan
    df.loc[3, "pmic_gap_top1_top2"] = np.inf
    df.loc[4, "pmil_gap_top1_top2"] = -np.inf
    return df

def test_fit_predict_shapes_and_bounds():
    df = _make_df(240, seed=0)
    cfg = DetectorConfig(epochs=8, hidden=32, lr=3e-3, weight_decay=2e-4,
                         blend_grid=21, calib_size=0.25, val_size=0.2, device="cpu", batch=128)
    det = DetectorEnsemble(cfg)
    summary = det.fit(df)
    assert 0 <= summary["blend_weight"] <= 1
    p = det.predict_proba(df)
    assert p.shape == (len(df),)
    assert np.all(np.isfinite(p))
    assert np.all((p >= 0) & (p <= 1))

def test_determinism_same_seed_same_preds():
    df = _make_df(200, seed=1)
    cfg = DetectorConfig(epochs=6, hidden=16, seed=42, device="cpu", blend_grid=11)
    det1 = DetectorEnsemble(cfg); det1.fit(df)
    p1 = det1.predict_proba(df)
    det2 = DetectorEnsemble(cfg); det2.fit(df)
    p2 = det2.predict_proba(df)
    np.testing.assert_allclose(p1, p2, atol=1e-8)

def test_save_load_roundtrip_identical_preds():
    df = _make_df(180, seed=2)
    cfg = DetectorConfig(epochs=5, hidden=8, seed=7, device="cpu", blend_grid=9)
    det = DetectorEnsemble(cfg); det.fit(df)
    p0 = det.predict_proba(df)
    with TemporaryDirectory() as tmp:
        det.save(tmp)
        det2 = DetectorEnsemble.load(tmp)
        p1 = det2.predict_proba(df)
    np.testing.assert_allclose(p0, p1, atol=0)

def test_missing_columns_at_inference_are_imputed():
    df = _make_df(150, seed=3)
    cfg = DetectorConfig(epochs=5, hidden=8, seed=3, device="cpu", blend_grid=9)
    det = DetectorEnsemble(cfg); det.fit(df)
    # drop a feature column at inference time → imputer should supply value
    df_small = df.drop(columns=["pmil_gap_top1_top2"])
    p = det.predict_proba(df_small)
    assert p.shape == (len(df_small),)
    assert np.all(np.isfinite(p))

def test_handles_extreme_values_and_batching():
    df = _make_df(513, seed=4)  # 513 to exercise batching w/ non-divisible size
    cfg = DetectorConfig(epochs=4, hidden=8, seed=4, device="cpu", batch=64, blend_grid=7)
    det = DetectorEnsemble(cfg); det.fit(df)
    p = det.predict_proba(df)
    assert p.shape == (len(df),)
    assert np.isfinite(p).all()