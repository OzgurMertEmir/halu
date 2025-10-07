import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from halu.core.types import MCQExample, MCQOption
from halu.core.runner import HFRunner
from halu.pipeline import MetricsPipeline
from halu.features.tables import build_option_table, collapse_option_side_features

pytestmark = [pytest.mark.hf, pytest.mark.slow]

MODEL_ID = "Qwen/Qwen2.5-0.5B"  # small & friendly for CI; can swap to -Instruct too

def _mk_example():
    # Simple 4-option MCQ
    return MCQExample(
        qid="q_demo",
        question="Which animal is a mammal?",
        options=[
            MCQOption("A", "Shark"),
            MCQOption("B", "Crocodile"),
            MCQOption("C", "Dolphin"),
            MCQOption("D", "Seahorse"),
        ],
        gold_letter="C",
        dataset="unit"
    )

def test_qwen25_pipeline_e2e_cpu():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,      # CPU friendly
        device_map="cpu"
    ).eval()

    # Runner config: store on CPU, standard dtypes
    runner = HFRunner(
        tokenizer=tok,
        model=model,
        store_device="cpu",
        store_dtype=torch.float32,
        include_embeddings=False,
        keep_logits_full=False
    )

    # Use ICR + LLMCheck + FCM. If runtime is tight, set use_icr=False to skip attention pooling.
    pipe = MetricsPipeline(runner, use_icr=True, use_llmcheck=True, use_fcm=True)
    ex = _mk_example()

    row = pipe.example_to_row(ex)
    assert row["qid"] == "q_demo"
    assert row["gold_letter"] == "C"

    # Sanity: some expected metric keys present (don’t assert exact values)
    # - LLMCheck targets
    assert any(k.endswith("_llmc_letter_ce_logpmi") for k in row)
    assert any(k.endswith("_llmc_content_ce_logpmi") for k in row)
    # - Centered features & gaps
    assert "llmc_content_ce_logpmi_gap_top1_top2" in row
    assert "llmc_letter_ce_logpmi_gap_top1_top2" in row
    # - FCM bundle
    assert "fcm_letter_probs" in row or "fcm_letter_prob_map" in row

    # Turn per-option metrics into per-option table + per-question collapse
    opt = build_option_table([row])
    assert set(opt["letter"]) == {"A", "B", "C", "D"}
    out = collapse_option_side_features(opt)
    assert list(out["qid"]) == ["q_demo"]
    r = out.iloc[0]

    # Basic shape/typing sanity
    assert isinstance(r["p_opt_top"], float)
    assert isinstance(r["p_opt_entropy"], float)

    # P(error) proxy sanity: top letter by p_opt should be among the options
    assert isinstance(r["opt_top_letter"], str) and r["opt_top_letter"] in {"A", "B", "C", "D"}