import math
import numpy as np

from halu.features.feature_metrics import (
    softmax_stable,
    logsoftmax_stable,
    token_logprobs_for_targets,
    entropy_from_logits,
    margin_from_logits,
    maxprob_from_logits,
    aggregate,
    letter_probs_last,
)

# Three toy steps (T=3), V=4
R1 = np.array([ 2.0, 1.0, 0.0, -1.0])
R2 = np.array([ 0.0, 0.0, 0.0,  0.0])
R3 = np.array([ 1.0, 3.0, 0.0, -2.0])
LOGITS = np.stack([R1, R2, R3], axis=0)
TARGETS = np.array([0, 1, 1], dtype=int)  # realized next-token ids

def test_softmax_stable_rows_sum_to_one_and_known_values():
    P = softmax_stable(LOGITS)
    # rows sum to 1
    np.testing.assert_allclose(P.sum(axis=-1), np.ones(3), atol=1e-12)
    # spot check against known values
    exp_r1 = np.array([0.6439142598879724, 0.23688281808991013, 0.08714431874203257, 0.03205860328008499])
    exp_r2 = np.full(4, 0.25)
    exp_r3 = np.array([0.11354961935954127, 0.8390245074625321, 0.04177257051508217, 0.005653302662844444])
    np.testing.assert_allclose(P[0], exp_r1, atol=1e-12)
    np.testing.assert_allclose(P[1], exp_r2, atol=1e-12)
    np.testing.assert_allclose(P[2], exp_r3, atol=1e-12)

def test_logsoftmax_matches_log_of_softmax():
    P = softmax_stable(LOGITS)
    L = logsoftmax_stable(LOGITS)
    np.testing.assert_allclose(L, np.log(P), atol=1e-12)

def test_token_logprobs_for_targets():
    lp = token_logprobs_for_targets(LOGITS, TARGETS)
    exp = np.array([
        -0.4401896985611953,   # log P(target=0 | R1)
        -1.3862943611198906,   # log P(target=1 | R2 uniform)
        -0.17551536261671444,  # log P(target=1 | R3)
    ])
    np.testing.assert_allclose(lp, exp, atol=1e-12)

def test_entropy_margin_maxprob_per_step():
    H = entropy_from_logits(LOGITS)
    M = margin_from_logits(LOGITS)
    Q = maxprob_from_logits(LOGITS)

    # known entropies
    np.testing.assert_allclose(H[0], 0.9475369639754255, atol=1e-12)
    np.testing.assert_allclose(H[1], math.log(4.0),     atol=1e-12)
    np.testing.assert_allclose(H[2], 0.5561988261936499, atol=1e-12)

    # logit margins (top1 - top2)
    np.testing.assert_allclose(M, np.array([1.0, 0.0, 2.0]), atol=1e-12)

    # max prob
    np.testing.assert_allclose(Q, np.array([0.6439142598879724, 0.25, 0.8390245074625321]), atol=1e-12)

def test_sequence_aggregates():
    M = margin_from_logits(LOGITS)  # [1.0, 0.0, 2.0]
    out = aggregate(M, stats=("mean","min","max","std"))
    assert set(out.keys()) == {"mean","min","max","std"}
    np.testing.assert_allclose(out["mean"], 1.0, atol=1e-12)
    np.testing.assert_allclose(out["min"],  0.0, atol=1e-12)
    np.testing.assert_allclose(out["max"],  2.0, atol=1e-12)
    # population std of [1,0,2] around mean=1 is sqrt((0^2+1^2+1^2)/3) = sqrt(2/3)
    np.testing.assert_allclose(out["std"], math.sqrt(2.0/3.0), atol=1e-12)

def test_letter_probs_last_step():
    # Map letters to ids in our toy vocab
    letter_ids = {"A":0, "B":1, "C":2, "D":3}
    last = LOGITS[-1]  # R3
    probs = letter_probs_last(last, letter_ids)
    # probabilities should align with softmax(R3)
    P = softmax_stable(R3[None, :])[0]
    np.testing.assert_allclose(
        [probs["A"], probs["B"], probs["C"], probs["D"]],
        [P[0], P[1], P[2], P[3]],
        atol=1e-12
    )
    # argmax is B
    assert max(probs, key=probs.get) == "B"

def test_edge_cases_empty_and_uniform_and_large():
    # Empty T=0 -> empty arrays, aggregates->nan
    empty_logits = np.zeros((0, 4))
    assert entropy_from_logits(empty_logits).size == 0
    assert margin_from_logits(empty_logits).size == 0
    assert maxprob_from_logits(empty_logits).size == 0
    out = aggregate(np.array([]))
    assert math.isnan(out["mean"]) and math.isnan(out["min"]) and math.isnan(out["max"]) and math.isnan(out["std"])

    # Stability with very large logits
    huge = np.array([[1000.0, 0.0]])
    p = softmax_stable(huge)
    np.testing.assert_allclose(p, np.array([[1.0, 0.0]]), atol=1e-15)