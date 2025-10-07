import re
from halu.core.utils import (
    build_prompt,
    build_prompt_uncond,
    build_prompt_noopts_letter,
    build_uncond_noopts_with_assistant_text,
    options_text_from_ex,
)
from halu.core.types import MCQExample, MCQOption

# --- Fixtures ---------------------------------------------------------------

def _ex():
    return MCQExample(
        qid="q1",
        question="What is the capital of France?",
        options=[
            MCQOption("A", "Berlin"),
            MCQOption("B", "Madrid"),
            MCQOption("C", "Paris"),
            MCQOption("D", "Rome"),
        ],
        gold_letter="C",
        dataset="toy",
    )

class _EncArr:
    """Minimal tensor-ish stub with the attributes HF code expects."""
    def __init__(self, n):
        self.shape = (1, int(n))  # [batch=1, seq_len]
        self.ndim = 2

    def __len__(self):
        return self.shape[0]

    def to(self, *args, **kwargs):
        return self  # mimic torch.Tensor.to

    def __getitem__(self, item):
        # We don't actually use indexing in these tests, but make it safe.
        return self

class _Enc:
    def __init__(self, n):
        self.input_ids = _EncArr(n)

    def to(self, *args, **kwargs):
        return self  # mimic BatchEncoding.to

def _approx_token_len(text: str) -> int:
    return len(text)

class _NoChatTok:
    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        return _Enc(_approx_token_len(text))

class _ChatTok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        parts = []
        for m in messages:
            r = m.get("role", "")
            c = m.get("content", "")
            parts.append(f"<{r.upper()}>{c}</{r.upper()}>")
        s = "".join(parts)
        if add_generation_prompt:
            s += "<ASSISTANT>"
        return s

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        return _Enc(_approx_token_len(text))
# --- Tests: fallback (no chat template) -------------------------------------

def test_build_prompt_fallback_plain_text():
    ex = _ex()
    tok = _NoChatTok()
    prompt = build_prompt(ex, tokenizer=tok)

    # contains system, question, options, and trailing "Answer:"
    assert "You are a helpful, factual assistant." in prompt
    assert "Question: What is the capital of France?" in prompt
    assert "Options:" in prompt
    assert "A) Berlin" in prompt and "C) Paris" in prompt
    assert prompt.strip().endswith("Answer:")

def test_build_prompt_uncond_includes_options_and_letter_in_full():
    ex = _ex()
    tok = _NoChatTok()
    prompt_u, full_u = build_prompt_uncond(ex, option_label="B", tokenizer=tok)

    # prompt has options and ends with "Answer:"
    assert "Options:" in prompt_u and "Answer:" in prompt_u
    # full contains "B)" immediately after prompt (allow a single separating space)
    assert re.search(r"Answer:\s*B\)", full_u) is not None

def test_build_prompt_noopts_letter_and_uncond_text():
    # letter-only (no options list)
    prompt_l, full_l = build_prompt_noopts_letter(option_label="D", tokenizer=_NoChatTok())
    assert "Options:" not in prompt_l and "Answer:" in prompt_l
    assert re.search(r"Answer:\s*D\)", full_l) is not None

    # free assistant text (no options list)
    p2, f2 = build_uncond_noopts_with_assistant_text("C) Paris", tokenizer=_NoChatTok())
    assert "Options:" not in p2 and "Answer:" in p2
    assert f2.strip().endswith("C) Paris")

def test_options_text_helper_is_consistent():
    ex = _ex()
    txt = options_text_from_ex(ex)
    # four lines, A..D
    lines = txt.splitlines()
    assert len(lines) == 4
    assert lines[0].startswith("A)") and "Berlin" in lines[0]
    assert lines[2].startswith("C)") and "Paris" in lines[2]

# --- Tests: chat-template path (no external deps) ---------------------------

def test_build_prompt_with_chat_template_stub():
    ex = _ex()
    tok = _ChatTok()
    out = build_prompt(ex, tokenizer=tok)
    # should be wrapped by our synthetic tags and end with <ASSISTANT>
    assert out.startswith("<SYSTEM>") and "<USER>" in out and out.endswith("<ASSISTANT>")
    # ensure the question made it through
    assert "What is the capital of France?" in out
    # options are in the user message
    assert "A) Berlin" in out and "C) Paris" in out

def test_uncond_variants_with_chat_template_stub():
    ex = _ex()
    tok = _ChatTok()

    # letter target
    pL, fL = build_prompt_noopts_letter("B", tokenizer=tok)
    assert pL.endswith("<ASSISTANT>")
    assert fL.endswith("</ASSISTANT>") is False  # we don't add assistant block here
    assert "Answer:" in pL and "B)" in fL

    # assistant full text
    p2, f2 = build_uncond_noopts_with_assistant_text("B) Madrid", tokenizer=tok)
    assert p2.endswith("<ASSISTANT>")
    assert "B) Madrid" in f2