from __future__ import annotations
from typing import Dict, Iterable
import numpy as np

# --------- numerics ---------
def softmax_stable(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(logits, dtype=float)
    if x.size == 0:
        return np.empty_like(x)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    z = np.sum(e, axis=axis, keepdims=True)
    return e / z

def _logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)

def logsoftmax_stable(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(logits, dtype=float)
    if x.size == 0:
        return np.empty_like(x)
    lse = _logsumexp(x, axis=axis)
    # expand lse to subtract along axis
    shape = list(x.shape)
    shape[axis] = 1
    lse = lse.reshape(shape)
    return x - lse

# --------- per-token metrics ---------

def token_logprobs_for_targets(logits: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    """
    logits: [T, V], target_ids: [T]
    returns: [T] log P(target_t | logits_t)
    """
    L = logsoftmax_stable(logits)  # [T,V]
    idx_rows = np.arange(L.shape[0], dtype=int)
    tgt = np.asarray(target_ids, dtype=int)
    if tgt.shape[0] != L.shape[0]:
        raise ValueError("target_ids length must match logits T")
    return L[idx_rows, tgt]

def entropy_from_logits(logits: np.ndarray) -> np.ndarray:
    """
    Shannon entropy per step: -sum p * log p, using stable log-softmax.
    """
    L = logsoftmax_stable(logits)      # [T,V]
    P = np.exp(L)
    if P.size == 0:
        return np.array([], dtype=float)
    return -np.sum(P * L, axis=-1)

def margin_from_logits(logits: np.ndarray) -> np.ndarray:
    """
    logit margin per step: top1 - top2. Ties -> 0.
    """
    x = np.asarray(logits, dtype=float)
    if x.size == 0:
        return np.array([], dtype=float)
    s = np.sort(x, axis=-1)
    top1 = s[..., -1]
    top2 = s[..., -2] if x.shape[-1] >= 2 else np.full_like(top1, np.nan)
    return top1 - top2

def maxprob_from_logits(logits: np.ndarray) -> np.ndarray:
    """
    max softmax prob per step.
    """
    P = softmax_stable(logits)
    if P.size == 0:
        return np.array([], dtype=float)
    return np.max(P, axis=-1)

# --------- aggregates ---------

def aggregate(values: np.ndarray, stats: Iterable[str] = ("mean","min","max","std")) -> Dict[str, float]:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {k: float("nan") for k in stats}
    out: Dict[str, float] = {}
    for k in stats:
        if k == "mean":
            out[k] = float(v.mean())
        elif k == "min":
            out[k] = float(v.min())
        elif k == "max":
            out[k] = float(v.max())
        elif k == "std":
            # population std (like numpy default ddof=0)
            out[k] = float(v.std(ddof=0))
        else:
            # ignore unknowns to keep API forgiving
            pass
    return out

# --------- MCQ helpers ---------

def letter_probs_last(logits_last_step: np.ndarray,
                      letter_id_map: Dict[str, int]) -> Dict[str, float]:
    """
    Given 1D logits for the *last* step [V], return softmax probs for letters.
    """
    x = np.asarray(logits_last_step, dtype=float)
    if x.ndim == 2:
        # Accept [1,V] or [T,V]; take last row
        x = x[-1]
    p = softmax_stable(x[None, :])[0]
    out = {}
    for L, idx in letter_id_map.items():
        out[L] = float(p[int(idx)])
    return out