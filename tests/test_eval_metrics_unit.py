# tests/test_eval_metrics_unit.py
import math
import numpy as np
import pandas as pd

from halu.analysis.eval_metrics import (
    ece_binary,
    reliability_table,
    risk_coverage_curves,
    aurc_and_auacc,
    coverage_at_accuracy,
    bootstrap_ci,
)

Y = np.array([0,1,0,1,0,0,1,1,0,1], dtype=int)
P = np.array([0.1,0.8,0.2,0.7,0.3,0.4,0.6,0.9,0.05,0.55], dtype=float)

def test_ece_binary_basic_and_shapes():
    # baseline value
    val = ece_binary(Y, P)  # default bins=15
    assert math.isfinite(val)
    assert abs(val - 0.25) < 1e-12

def test_ece_binary_edge_cases():
    # empty -> nan
    assert math.isnan(ece_binary(np.array([], dtype=int), np.array([], dtype=float)))
    # length mismatch -> ValueError
    try:
        _ = ece_binary(np.array([0,1]), np.array([0.1]))
        assert False, "Expected ValueError on length mismatch"
    except ValueError:
        pass

def test_reliability_table_schema_and_values():
    tab = reliability_table(Y, P, bins=4)
    # Schema
    assert list(tab.columns) == ["p_lo","p_hi","n","p_mean(Perr)","err_rate","cal_error_abs"]
    # Totals & bin counts
    assert int(tab["n"].sum()) == len(Y)
    assert len(tab) <= 4

    # Spot-check against the known values from your session
    # (tolerances keep the test robust to float noise)
    exp_n = np.array([3, 2, 3, 2])
    exp_pmean = np.array([0.116667, 0.350000, 0.616667, 0.850000])
    exp_err = np.array([0.0, 0.0, 1.0, 1.0])

    np.testing.assert_array_equal(tab["n"].to_numpy(), exp_n)
    np.testing.assert_allclose(tab["p_mean(Perr)"].to_numpy(), exp_pmean, rtol=0, atol=1e-6)
    np.testing.assert_allclose(tab["err_rate"].to_numpy(), exp_err, rtol=0, atol=1e-12)

def test_risk_coverage_curves_monotone_and_areas():
    cov, risk, acc, AURC, AUACC = risk_coverage_curves(Y, P, 10)
    # lengths
    assert len(cov) == len(Y) == len(risk) == len(acc)
    # coverage is strictly increasing 1/N .. 1
    np.testing.assert_allclose(cov, np.arange(1, len(Y)+1, dtype=float) / len(Y), rtol=0, atol=0)
    # acc + risk = 1
    np.testing.assert_allclose(acc + risk, np.ones_like(acc), atol=1e-12)
    # exact areas under the prefix curve (with endpoints)
    assert abs(AURC - 0.1521825396825397) < 1e-12
    assert abs(AUACC - 0.8478174603174603) < 1e-12
    # sanity: areas complement to 1
    assert abs((AURC + AUACC) - 1.0) < 1e-12

def test_risk_coverage_curves_degenerate_labels():
    N = 8
    cov0, risk0, acc0, A0, AA0 = risk_coverage_curves(np.zeros(N, dtype=int), np.linspace(0,1,N), 10)
    cov1, risk1, acc1, A1, AA1 = risk_coverage_curves(np.ones(N, dtype=int),  np.linspace(0,1,N), 10)
    # All-correct → risk=0, acc=1, AURC=0, AUACC=1
    assert np.allclose(risk0, 0.0)
    assert np.allclose(acc0, 1.0)
    assert abs(A0 - 0.0) < 1e-12 and abs(AA0 - 1.0) < 1e-12
    # All-errors (prefix + endpoints):
    # risk ≡ 1, so AURC = 1 - 1/(2N); AUACC = 1/(2N)
    assert np.allclose(risk1, 1.0)
    assert np.allclose(acc1, 0.0)
    N = 8
    assert abs(A1 - (1.0 - 1.0 / (2 * N))) < 1e-12
    assert abs(AA1 - (1.0 / (2 * N))) < 1e-12

def test_aurc_and_auacc_wrapper():
    A, AA, cov, risk, acc = aurc_and_auacc(Y, P)
    assert abs(A - 0.1521825396825397) < 1e-12
    assert abs(AA - 0.8478174603174603) < 1e-12
    # cov/risk/acc lengths agree
    assert len(cov) == len(risk) == len(acc) == len(Y)

def test_coverage_at_accuracy():
    # Known from your session
    assert abs(coverage_at_accuracy(Y, P, 0.80) - 0.60) < 1e-12
    # With prefix sorting, first 5 kept are correct → accuracy=1 at 50% coverage
    assert coverage_at_accuracy(Y, P, 0.9999) == 0.5

def test_bootstrap_ci_basic_and_reproducible():
    # Use AURC as the scalar metric
    f = lambda Y_, P_: aurc_and_auacc(Y_, P_)[0]
    lo1, hi1, pt1 = bootstrap_ci(f, Y, P, n_boot=200, seed=1337, alpha=0.05)
    lo2, hi2, pt2 = bootstrap_ci(f, Y, P, n_boot=200, seed=1337, alpha=0.05)
    # Reproducible with same seed
    assert (lo1, hi1, pt1) == (lo2, hi2, pt2)
    # Point estimate equals direct AURC
    A, *_ = aurc_and_auacc(Y, P)
    assert abs(pt1 - A) < 1e-12
    # CI ordering
    assert lo1 <= pt1 <= hi1

def test_bootstrap_ci_single_class_edge_case():
    Y0 = np.zeros_like(Y)
    f = lambda Y_, P_: aurc_and_auacc(Y_, P_)[0]
    lo, hi, pt = bootstrap_ci(f, Y0, P, n_boot=100, seed=1337)
    assert math.isnan(lo) and math.isnan(hi)
    # point estimate still computed on original sample
    assert math.isfinite(pt)