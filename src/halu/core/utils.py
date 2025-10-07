from __future__ import annotations
from typing import List, Dict, Iterable, Tuple, Any
import numpy as np
import torch
from . import prompts as _P
from halu.utils.math import pearson_from_centered

EPS = 1e-10
SEED = 1234
NEED_LOGITS="logits"; NEED_HIDDEN="hidden_layers"; NEED_ATTN="attn_layers"

def build_openllm_prompt(question: str, options_text: str) -> str:
    return _P.openllm_prompt(question, options_text)
'''
def build_prompt(ex, tokenizer=None) -> str:
    opts = "\n".join([f"{o.label}) {o.text}" for o in ex.options])
    txt, _ = _P.q_with_options(tokenizer, ex.question, opts)
    return txt

def build_prompt_uncond(ex, option_label: str, tokenizer=None):
    return _P.uncond_letter(tokenizer, option_label)  # (prompt, full, len)

def build_uncond_with_assistant_text(ex, assistant_text: str, tokenizer=None):
    return _P.uncond_assistant_text(tokenizer, assistant_text)

def build_prompt_noopts_letter(option_label: str, tokenizer=None):
    return _P.uncond_letter(tokenizer, option_label)  # (prompt, full, len)

def build_uncond_noopts_with_assistant_text(assistant_text: str, tokenizer=None):
    return _P.uncond_assistant_text(tokenizer, assistant_text)
'''

def build_prompt(ex, tokenizer=None) -> str:
    # q_with_options returns (prompt, prompt_len) – we need just the text
    opts = options_text_from_ex(ex)
    prompt, _ = _P.q_with_options(tokenizer, ex.question, opts)
    return prompt

def build_prompt_uncond(ex, option_label: str, tokenizer=None) -> Tuple[str, str]:
    # Include options in the prompt as required by tests
    opts = options_text_from_ex(ex)
    p, f, _ = _P.uncond_letter_with_options(tokenizer, opts, letter=option_label)
    return p, f

def build_uncond_with_assistant_text(ex, assistant_text: str, tokenizer=None) -> Tuple[str, str]:
    # uncond_assistant_text returns (prompt, full, prompt_len) – drop length
    p, f, _ = _P.uncond_assistant_text(tokenizer, assistant_text)
    return p, f

def build_prompt_noopts_letter(option_label: str, tokenizer=None) -> Tuple[str, str]:
    # alias for the same uncond_letter path
    p, f, _ = _P.uncond_letter(tokenizer, option_label)
    return p, f

def build_uncond_noopts_with_assistant_text(assistant_text: str, tokenizer=None) -> Tuple[str, str]:
    # alias for the same uncond_assistant_text path
    p, f, _ = _P.uncond_assistant_text(tokenizer, assistant_text)
    return p, f
def options_text_from_ex(ex) -> str:
    return "\n".join([f"{o.label}) {o.text}" for o in ex.options])

def labels_from_ex(ex) -> List[str]:
    return [o.label.upper() for o in ex.options]

def _mean_nll(logits_resp: torch.Tensor, targets_resp: torch.Tensor) -> torch.Tensor:
    """Compute mean negative log-likelihood with robust error handling."""
    try:
        if logits_resp is None or targets_resp is None:
            return torch.tensor(float('nan'))

        # align lengths
        if logits_resp.dim() != 2 or targets_resp.dim() != 1:
            return torch.tensor(float('nan'))
        if logits_resp.shape[0] != targets_resp.shape[0]:
            L = min(logits_resp.shape[0], targets_resp.shape[0])
            if L <= 0:
                return torch.tensor(float('nan'))
            logits_resp = logits_resp[:L]
            targets_resp = targets_resp[:L]

        # ---- NEW: align device + dtype for safety ----
        if targets_resp.device != logits_resp.device:
            targets_resp = targets_resp.to(logits_resp.device)
        logits_resp = logits_resp.to(torch.float32)

        logp = torch.log_softmax(logits_resp, dim=-1)               # [T,V]
        nll  = -logp.gather(-1, targets_resp.view(-1, 1)).squeeze() # [T]
        return nll.mean()
    except Exception:
        return torch.tensor(float('nan'))

def _safe_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    num = (a * b).sum(dim=-1)
    den = a.norm(dim=-1).clamp_min(EPS) * b.norm(dim=-1).clamp_min(EPS)
    return num / den


def pool_icr_features(
    *,
    delta_per_tok: torch.Tensor,               # [T_resp]
    dL_norm: torch.Tensor,                     # [L', T_resp]
    dT_norm: torch.Tensor | None,              # [T_resp-1] or None
    interlayer_cos_tok: torch.Tensor | None,
    temporal_cos: torch.Tensor | None,
    prompt_mass_tok: torch.Tensor | None
) -> Dict[str, Any]:
    pieces = []
    scalars = {}
    def _moments(x: torch.Tensor, prefix: str):
        x = x.float()
        scalars[prefix+"_mean"] = float(x.mean().item())
        scalars[prefix+"_std"]  = float(x.std(unbiased=False).item())
        scalars[prefix+"_p95"]  = float(torch.quantile(x, 0.95).item())
        pieces.extend([x.mean().item(), x.std(unbiased=False).item(), torch.quantile(x, 0.95).item()])
    _moments(delta_per_tok, "delta_tok")
    layer_spread = (dL_norm.std(dim=0))
    _moments(layer_spread, "deltaL_spread")
    #_moments(dL_norm.mean(dim=0), "deltaL_tok")
    if dT_norm is not None and dT_norm.numel() > 0:
        _moments(dT_norm, "deltaT")
    if interlayer_cos_tok is not None and interlayer_cos_tok.numel() > 0:
        _moments(interlayer_cos_tok, "interlayer_cos")
    if temporal_cos is not None and temporal_cos.numel() > 0:
        _moments(temporal_cos, "temporal_cos")
    if prompt_mass_tok is not None and prompt_mass_tok.numel() > 0:
        _moments(prompt_mass_tok, "prompt_mass")
    vec = np.array(pieces, dtype=np.float32)
    scalars["pooled_vector"] = vec
    if prompt_mass_tok is not None and prompt_mass_tok.numel() == delta_per_tok.numel():
        x = (delta_per_tok - delta_per_tok.mean()).cpu().numpy()
        y = (prompt_mass_tok - prompt_mass_tok.mean()).cpu().numpy()
        scalars["attn_delta_consistency"] = float(pearson_from_centered(x, y))
    else:
        scalars["attn_delta_consistency"] = float("nan")
    return scalars

def _letter_buckets(tokenizer, letters: List[str]) -> Tuple[List[List[int]], List[str]]:
    buckets: List[List[int]] = []
    reps: List[str] = []
    for L in letters:
        variants = [L, f" {L}", f"{L})", f" {L})", f"{L}) "]
        ids: List[int] = []
        for v in variants:
            toks = tokenizer(v, add_special_tokens=False).input_ids
            if isinstance(toks[0], list):
                toks = toks[0]
            if len(toks) >= 1:
                ids.append(toks[0])
        ids = sorted(set(ids))
        buckets.append(ids)
        reps.append(L)
    return buckets, reps

def _letter_dist_fullsoftmax(last_logits: torch.Tensor, letters: List[str], tokenizer, temperature: float = 1.0):
    with torch.no_grad():
        logits = last_logits.float()
        probs = torch.softmax(logits / max(temperature, 1e-8), dim=-1).cpu().numpy()
    buckets, reps = _letter_buckets(tokenizer, letters)
    p_letters = np.zeros(len(letters), dtype=np.float32)
    for i, b in enumerate(buckets):
        if b:
            p_letters[i] = float(probs[b].sum())
    mass_on_letters = float(p_letters.sum())
    if mass_on_letters > 0:
        p_letters = p_letters / mass_on_letters
    else:
        p_letters = np.zeros_like(p_letters)
    return p_letters, mass_on_letters, buckets

def get_single_row_features_from_option_metrics(per_option_dicts, option_labels):
    row = {}
    for lab, feats in zip(option_labels, per_option_dicts):
        for k, v in feats.items():
            row[f"{lab}_{k}"] = v
    return row
