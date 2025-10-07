import pytest
from halu.core.runner import HFRunner

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TR = True
except Exception:
    HAS_TR = False

pytestmark = pytest.mark.skipif(not HAS_TR, reason="transformers not installed")


@pytest.fixture(scope="session")
def tiny_hf():
    # Tiny CPU-friendly causal LM
    model_id = "sshleifer/tiny-gpt2"
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()
    return model, tok


def test_runner_logits_slicing_basic(tiny_hf):
    model, tok = tiny_hf
    runner = HFRunner(model, tok, store_device="cpu", include_embeddings=False, keep_logits_full=False)

    prompt = "ABC"      # 3 tokens
    response = " WXYZ"  # 5 tokens (leading space helps GPT2)
    pack = runner.prompt_response_forward(prompt, response, {"logits"})

    # basic shape checks
    assert pack.logits_full is None
    assert pack.logits_resp is not None
    assert pack.resp_len > 0
    assert pack.logits_resp.shape[0] == pack.resp_len
    assert pack.logits_resp.shape[1] == model.config.vocab_size

    # verify the slicing rule matches runner’s implementation
    enc_full = tok(prompt + response, return_tensors="pt")
    total_len = enc_full.input_ids.shape[1]
    prompt_len = tok(prompt, return_tensors="pt").input_ids.shape[1]
    resp_len = total_len - prompt_len
    assert pack.resp_len == resp_len


def test_runner_needs_normalization_and_states(tiny_hf):
    model, tok = tiny_hf
    runner = HFRunner(model, tok, store_device="cpu", include_embeddings=True, keep_logits_full=True)

    prompt = "hello"
    response = " world!"
    pack = runner.prompt_response_forward(prompt, response, {"attn", "hidden", "logits_full"})

    # logits_full present when requested
    assert pack.logits_full is not None
    assert pack.logits_full.shape[0] == pack.total_len
    assert pack.logits_full.shape[1] == model.config.vocab_size

    # attentions: list of per-layer [H, T, T]
    assert pack.attn_layers is not None and len(pack.attn_layers) >= 1
    H, T, T2 = pack.attn_layers[-1].shape
    assert T == pack.total_len and T2 == pack.total_len
    assert H >= 1

    # hidden states: list of per-layer [T, D] (+ optional embeddings)
    assert pack.hidden_layers is not None and len(pack.hidden_layers) >= 1
    T_h, D = pack.hidden_layers[-1].shape
    assert T_h == pack.total_len and D == model.config.n_embd

    # embeddings (optional) present when include_embeddings=True
    assert pack.hidden_embed is not None
    assert pack.hidden_embed.shape[0] == pack.total_len
    assert pack.hidden_embed.shape[1] == model.config.n_embd