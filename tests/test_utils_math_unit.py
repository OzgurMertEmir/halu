# tests/test_utils_math_unit.py
import math
import numpy as np
from halu.utils.math import (
    trapz_area,
    pearson_from_centered,
    top2_gap,
    entropy_from_prob,
)

def test_trapz_area_linear_triangle():
    x = np.linspace(0.0, 1.0, 11)
    y = x.copy()  # area under y=x on [0,1] is 0.5
    a = trapz_area(y, x)
    assert abs(a - 0.5) < 1e-12

def test_trapz_area_errors_and_nans():
    # length mismatch
    try:
        _ = trapz_area(np.array([0,1]), np.array([0]))
        assert False, "expected ValueError"
    except ValueError:
        pass
    # empty -> nan
    assert math.isnan(trapz_area(np.array([]), np.array([])))

def test_pearson_from_centered_basic():
    x = np.array([-1.0, 0.0, 1.0])
    y = np.array([-1.0, 0.0, 1.0])
    z = np.array([ 1.0, 0.0,-1.0])
    assert abs(pearson_from_centered(x, y) - 1.0) < 1e-12
    assert abs(pearson_from_centered(x, z) + 1.0) < 1e-12

def test_pearson_from_centered_degenerate():
    x = np.array([0.0, 0.0, 0.0])  # zero variance
    y = np.array([-1.0, 0.0, 1.0])
    assert math.isnan(pearson_from_centered(x, y))

def test_top2_gap_cases():
    assert abs(top2_gap(np.array([5, 1, 1])) - 4.0) < 1e-12
    assert abs(top2_gap(np.array([np.nan, 2.0])) - 2.0) < 1e-12  # only one finite → returns top1
    assert math.isnan(top2_gap(np.array([np.nan, np.inf, -np.inf])))

def test_entropy_from_prob():
    u = np.ones(4) / 4.0
    assert abs(entropy_from_prob(u) - math.log(4.0)) < 1e-12
    one_hot = np.array([1.0, 0.0, 0.0, 0.0])
    assert abs(entropy_from_prob(one_hot)) < 1e-12
    assert math.isnan(entropy_from_prob(np.array([0.0, 0.0, 0.0])))