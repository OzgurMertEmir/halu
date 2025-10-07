# tests/test_core_prompts_edge_cases_unit.py
import re
import pytest

from halu.core.types import MCQExample, MCQOption
from halu.core.utils import (
    build_prompt,
    build_prompt_uncond,
    build_prompt_noopts_letter,
    build_uncond_noopts_with_assistant_text,
    options_text_from_ex,
)
from halu.core import prompts as P

# --- Minimal stubs (copied to keep this file self-contained) ----------------

class _EncArr:
    def __init__(self, n): self.shape = (1, int(n)); self.ndim = 2
    def __len__(self): return self.shape[0]
    def to(self, *args, **kwargs): return self
    def __getitem__(self, item): return self

class _Enc:
    def __init__(self, n): self.input_ids = _EncArr(n)
    def to(self, *args, **kwargs): return self

def _approx_len(text: str) -> int: return len(text)

class _NoChatTok:
    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        return _Enc(_approx_len(text))

class _ChatTok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        parts = []
        for m in messages:
            r = m.get("role","").upper()
            c = m.get("content","")
            parts.append(f"<{r}>{c}</{r}>")
        s = "".join(parts)
        if add_generation_prompt: s += "<ASSISTANT>"
        return s
    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        return _Enc(_approx_len(text))

# --- Helpers ----------------------------------------------------------------

def _ex_unicode():
    return MCQExample(
        qid="q_uni",
        question="東京の首都はどこ？ 🇯🇵",  # (intentionally wrong wording to test unicode; just needs to roundtrip)
        options=[
            MCQOption("A", "東京"),
            MCQOption("B", "京都"),
            MCQOption("C", "大阪"),
            MCQOption("D", "名古屋 🏯"),
        ],
        gold_letter="A",
        dataset="toy",
    )

def _ex_weird_labels():
    return MCQExample(
        qid="q_weird",
        question="Pick the odd one out:",
        options=[
            MCQOption("A", "triangle"),
            MCQOption("Z", "circle"),
            MCQOption("β", "square"),
            MCQOption("4", "hexagon"),
        ],
        gold_letter="Z",
        dataset="toy",
    )

def _ex_two_options():
    return MCQExample(
        qid="q2",
        question="Is the sky blue?",
        options=[MCQOption("A","Yes"), MCQOption("B","No")],
        gold_letter="A",
        dataset="toy",
    )

def _ex_five_options():
    return MCQExample(
        qid="q5",
        question="Which planet is known as the Red Planet?",
        options=[
            MCQOption("A","Mercury"),
            MCQOption("B","Venus"),
            MCQOption("C","Earth"),
            MCQOption("D","Mars"),
            MCQOption("E","Jupiter"),
        ],
        gold_letter="D",
        dataset="toy",
    )

# --- Tests ------------------------------------------------------------------

def test_unicode_roundtrips_in_plain_and_chat():
    ex = _ex_unicode()

    # Plain (no chat template)
    p1 = build_prompt(ex, tokenizer=_NoChatTok())
    assert "東京の首都はどこ？" in p1
    assert "名古屋 🏯" in p1
    assert p1.strip().endswith("Answer:")

    # Chat-template path
    p2 = build_prompt(ex, tokenizer=_ChatTok())
    assert "東京の首都はどこ？" in p2 and "名古屋 🏯" in p2
    assert p2.endswith("<ASSISTANT>")

def test_options_text_varied_counts_and_labels():
    # 2 options
    txt2 = options_text_from_ex(_ex_two_options())
    lines2 = txt2.splitlines()
    assert len(lines2) == 2
    assert lines2[0].startswith("A)") and "Yes" in lines2[0]
    assert lines2[1].startswith("B)") and "No" in lines2[1]

    # 5 options
    txt5 = options_text_from_ex(_ex_five_options())
    lines5 = txt5.splitlines()
    assert len(lines5) == 5
    assert lines5[3].startswith("D)") and "Mars" in lines5[3]

    # Weird labels preserved (uppercased when applicable)
    weird = options_text_from_ex(_ex_weird_labels()).splitlines()
    # The function uppercases the label text; verify formatting is preserved
    assert weird[0].startswith("A)") and "triangle" in weird[0]
    assert weird[1].startswith("Z)") and "circle" in weird[1]
    # Non-Latin 'β' will remain as is (no upper for Greek beta by .upper() in many locales)
    assert weird[2].startswith("Β)") or weird[2].startswith("β)")  # allow either depending on .upper() behavior
    assert "square" in weird[2]
    assert weird[3].startswith("4)") and "hexagon" in weird[3]

def test_lowercase_letter_normalization_in_uncond_noopts():
    p, f = build_prompt_noopts_letter("b", tokenizer=_NoChatTok())
    assert "Options:" not in p and "Answer:" in p
    assert re.search(r"Answer:\s*B\)", f) is not None  # normalized to uppercase B)

def test_long_prompt_does_not_break():
    long_q = "What?" + (" verylong" * 2000)
    ex = MCQExample(
        qid="q_long",
        question=long_q,
        options=[MCQOption("A","x"), MCQOption("B","y"), MCQOption("C","z")],
        gold_letter="A",
        dataset="toy",
    )
    s = build_prompt(ex, tokenizer=_NoChatTok())
    assert len(s) > 5000
    assert s.strip().endswith("Answer:")

def test_custom_prompt_style_fallback_and_chat():
    style = P.PromptStyle(
        system="Eres un asistente útil y veraz.",
        question_label="Pregunta:",
        options_label="Opciones:",
        answer_label="Respuesta:",
    )
    ex = _ex_two_options()
    # Directly call prompts to pass style (utils does not expose style yet)
    tok_plain = _NoChatTok()
    txt_plain, _ = P.q_with_options(tok_plain, ex.question, options_text_from_ex(ex), style=style)
    assert "Eres un asistente útil y veraz." in txt_plain
    assert "Pregunta:" in txt_plain and "Opciones:" in txt_plain and txt_plain.strip().endswith("Respuesta:")

    tok_chat = _ChatTok()
    txt_chat, _ = P.q_with_options(tok_chat, ex.question, options_text_from_ex(ex), style=style)
    assert txt_chat.startswith("<SYSTEM>Eres un asistente útil y veraz.</SYSTEM>")
    assert txt_chat.endswith("<ASSISTANT>")
    assert "Pregunta:" in txt_chat and "Opciones:" in txt_chat

def test_uncond_assistant_text_unicode_and_chat_concat():
    # No options; assistant free text with unicode should be appended to open assistant prompt (no closing tag)
    p_plain, f_plain = build_uncond_noopts_with_assistant_text("Β) Αθήνα 🏛", tokenizer=_NoChatTok())
    assert "Options:" not in p_plain and "Answer:" in p_plain
    assert f_plain.strip().endswith("Β) Αθήνα 🏛")

    p_chat, f_chat = build_uncond_noopts_with_assistant_text("Β) Αθήνα 🏛", tokenizer=_ChatTok())
    assert p_chat.endswith("<ASSISTANT>")
    # We should *not* wrap a completed assistant turn here; we concat to the open tag
    assert f_chat.endswith("</ASSISTANT>") is False
    assert "Αθήνα" in f_chat

def test_light_integration_with_real_hf_tokenizer():
    # Optional: exercise _tok_len path with an actual HF tokenizer (no chat template)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    except Exception:
        pytest.skip("transformers/tokenizer not available; skipping integration test")

    ex = _ex_five_options()
    # Use prompts directly to retrieve (text, token_len)
    prompt_text, tok_len = P.q_with_options(tok, ex.question, options_text_from_ex(ex))
    # Sanity: non-empty text and a positive integer token length
    assert isinstance(prompt_text, str) and len(prompt_text) > 0
    assert isinstance(tok_len, int) and tok_len > 0