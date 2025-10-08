import math
import numpy as np
import pandas as pd
import pytest

from transformers import AutoTokenizer, AutoModelForCausalLM

from halu.core import prompts as P
from halu.core import utils as U
from halu.features.build import build_features_df


# -------------------------
# Config
# -------------------------
MODEL_IDS = [
    # Qwen family (chat)
    "Qwen/Qwen2.5-0.5B",
]

DATASET = "truthfulqa"   # adapter present in repo
N_EXAMPLES = 12          # keep very small to stay fast on CPU
SEED = 1337


# -------------------------
# Helpers
# -------------------------
def _load_model_and_tok(model_id: str):
    """Load model/tokenizer in a way that works across families; skip on OOM/availability."""
    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)

        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        tok.padding_side = "right"
        tok.truncation_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu",
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        ).eval()
        return tok, model
    except Exception as e:
        pytest.skip(f"Could not load {model_id} on this machine: {e}")


def _is_finite(x) -> bool:
    try:
        return np.isfinite(x).all()
    except Exception:
        try:
            return math.isfinite(float(x))
        except Exception:
            return False


def _coverage(df: pd.DataFrame, col: str) -> float:
    return float(df[col].notna().mean()) if col in df.columns else 0.0


def _assert_between(name: str, x: float, lo: float, hi: float):
    assert lo <= x <= hi, f"{name}={x:.4f} not in [{lo}, {hi}]"


# -------------------------
# 1) Prompt / tokenization alignment
# -------------------------
@pytest.mark.hf
@pytest.mark.slow
@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_prompt_alignment_and_letter_buckets(model_id):
    tok, _ = _load_model_and_tok(model_id)

    # Simple MCQ
    question = "What is the capital of France?"
    labels = ["A", "B", "C", "D", "E"]
    texts  = ["Paris", "London", "Berlin", "Madrid", "Rome"]
    options_text = "\n".join([f"{L}) {t}" for L, t in zip(labels, texts)])

    # Canonical "question + options" prompt via our builder
    prompt, prompt_len = P.q_with_options(tok, question, options_text)
    assert isinstance(prompt, str) and len(prompt) > 10
    assert isinstance(prompt_len, int) and prompt_len > 0

    # Two sample responses; make sure token slices line up with prompt_len
    responses = ["A) Paris", "B) London"]
    texts = [prompt + r for r in responses]
    enc = tok(texts, return_tensors="pt", padding=True)
    ids0 = enc.input_ids[0]
    ids1 = enc.input_ids[1]

    # Check the first prompt_len tokens match the prompt tokens
    enc_prompt = tok(prompt, return_tensors="pt", add_special_tokens=False)
    pids = enc_prompt.input_ids[0]
    assert pids.shape[0] == prompt_len, "prompt_len must equal tokenized length"
    assert ids0[:prompt_len].tolist() == pids.tolist(), "Prompt token prefix mismatch (sample 0)"
    assert ids1[:prompt_len].tolist() == pids.tolist(), "Prompt token prefix mismatch (sample 1)"

    # Verify response region is non-empty
    total_len_0 = int(enc.attention_mask[0].sum().item())
    resp_len_0 = total_len_0 - prompt_len
    assert resp_len_0 > 0, "Empty response region (0)"

    # Letter buckets should be non-empty for most letters
    buckets, reps = U._letter_buckets(tok, labels)
    assert len(buckets) == len(labels) == len(reps)
    non_empty = sum(1 for b in buckets if len(b) > 0)
    assert non_empty >= 4, f"Too few letter buckets non-empty: {non_empty}/5"


# -------------------------
# 2) Feature population audit across families
# -------------------------
@pytest.mark.hf
@pytest.mark.slow
@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_feature_population_across_model_families(model_id):
    tok, model = _load_model_and_tok(model_id)

    # Build a tiny features DF end-to-end
    df = build_features_df(model, tok, DATASET, size=N_EXAMPLES, seed=SEED)
    assert isinstance(df, pd.DataFrame) and len(df) == N_EXAMPLES

    # Required schema (per-question aggregated table)
    must_have_cols = [
        # option prob features
        "p_opt_chosen", "p_opt_top", "p_opt_second", "p_opt_gap", "p_opt_rank_chosen",
        # PMI-derived deltas
        "pmil_gap_top1_top2", "pmil_rank_chosen",
        "delta_pmil_chosen_vs_othermax",
        # LLMCheck deltas (letter + content)
        "delta_e_letter_chosen_vs_othermax",
        "delta_e_content_chosen_vs_othermax",
        # ICRP probe delta
        "delta_icrp_rel_chosen_vs_othermax",
        # FCM summary columns (legacy friendly)
        "fcm_entropy", "fcm_gini", "fcm_letters_coverage", "fcm_mass_gap", "fcm_top_letter",
    ]
    missing = [c for c in must_have_cols if c not in df.columns]
    assert not missing, f"Missing expected feature columns: {missing}"

    # --- Coverage thresholds (these are deliberately strict; tune if needed) ---
    cov = {}
    cov["pmil_gap"]   = _coverage(df, "pmil_gap_top1_top2")
    cov["llmc_letter"]  = _coverage(df, "delta_e_letter_chosen_vs_othermax")
    cov["llmc_content"] = _coverage(df, "delta_e_content_chosen_vs_othermax")
    cov["icrp"]         = _coverage(df, "delta_icrp_rel_chosen_vs_othermax")
    cov["fcm_entropy"]  = _coverage(df, "fcm_entropy")
    cov["fcm_gini"]     = _coverage(df, "fcm_gini")

    # Reasonable minimums that catch template drift and metric wiring errors.
    assert cov["pmil_gap"]   >= 0.8,  f"PMI(letter) coverage too low: {cov['pmil_gap']:.2f}"
    assert cov["llmc_letter"]  >= 0.8,  f"LLMCheck(letter) coverage too low: {cov['llmc_letter']:.2f}"
    assert cov["llmc_content"] >= 0.5,  f"LLMCheck(content) coverage too low: {cov['llmc_content']:.2f}"
    assert cov["icrp"]         >= 0.6,  f"ICR probe coverage too low: {cov['icrp']:.2f}"
    assert cov["fcm_entropy"]  >= 0.5,  f"FCM coverage too low: {cov['fcm_entropy']:.2f}"
    assert cov["fcm_gini"]     >= 0.5,  f"FCM coverage too low: {cov['fcm_gini']:.2f}"

    # --- Option-probability sanity (avoid degenerate uniform/zero gaps) ---
    # Not all rows need to be non-uniform, but at least half of them should be.
    uniform_frac = float((df["p_opt_gap"].fillna(0.0) == 0.0).mean())
    assert uniform_frac <= 0.6, f"Too many uniform option distributions (p_opt_gap==0): {uniform_frac:.2f}"

    # p_opt_chosen should be "reasonable": not all ~0 or ~1
    mean_pchosen = float(df["p_opt_chosen"].mean())
    _assert_between("mean(p_opt_chosen)", mean_pchosen, 0.15, 0.85)

    # --- FCM internal coherence ---
    # top letter should look like a valid label and mass_gap finite when entropy present
    valid_letters = {"A","B","C","D","E"}
    ok_letters = df["fcm_top_letter"].astype(str).str.upper().isin(valid_letters).mean()
    assert ok_letters >= 0.8, f"FCM top letters look invalid too often ({ok_letters:.2f} valid)"

    # If entropy present, mass_gap must be finite (no NaNs) and letters_coverage > 0
    if "fcm_entropy" in df.columns:
        subset = df[df["fcm_entropy"].notna()]
        assert (subset["fcm_mass_gap"].apply(_is_finite)).all(), "FCM mass_gap has NaNs/Infs where entropy is present"
        assert (subset["fcm_letters_coverage"].fillna(0) > 0).mean() >= 0.8, "FCM letter coverage often zero"

    # --- LLMCheck delta sanity (values should not be all ~0) ---
    # We just ensure some variance exists (detects constant-zero due to prefix bug)
    assert df["delta_e_letter_chosen_vs_othermax"].std(skipna=True) > 1e-6, "LLMCheck(letter) delta variance ~0"
    # Content is harder; allow looser variance
    assert df["delta_e_content_chosen_vs_othermax"].std(skipna=True) > 1e-8, "LLMCheck(content) delta variance ~0"

    # --- PMI letter sanity ---
    assert df["delta_pmil_chosen_vs_othermax"].std(skipna=True) > 1e-6, "PMI(letter) delta variance ~0"

    # Debug (optional)
    # print({k: round(v, 3) for k, v in cov.items()})


# -------------------------
# 3) Minimal end-to-end smoke: build + small metrics slice
# -------------------------
@pytest.mark.hf
@pytest.mark.slow
@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_build_features_schema_and_basic_stats(model_id):
    tok, model = _load_model_and_tok(model_id)
    df = build_features_df(model, tok, DATASET, size=N_EXAMPLES, seed=SEED)

    # No NaN explosions in basic, always-on columns
    base_cols = ["p_opt_chosen", "p_opt_top", "p_opt_second", "p_opt_gap", "p_opt_rank_chosen"]
    for c in base_cols:
        assert c in df.columns, f"Missing base feature: {c}"
        assert df[c].notna().all(), f"Base feature {c} has NaNs"

    # Reliability of PMI(letter) and ICRP
    assert df["pmil_gap_top1_top2"].notna().mean() >= 0.8
    assert df["delta_icrp_rel_chosen_vs_othermax"].notna().mean() >= 0.6

    # Basic distribution sanity
    assert df["p_opt_gap"].mean() >= 0.01, "Average option gap too small — looks uniform"
    assert df["p_opt_rank_chosen"].between(1, 5).mean() > 0.9, "Rank outside [1,5] too often"