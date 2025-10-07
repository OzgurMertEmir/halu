# src/halu/core/prompts.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional

def _apply_chat_template(tokenizer, messages, *, add_generation_prompt: bool) -> Optional[str]:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        except Exception:
            return None
    return None

def _tok_len(tokenizer, text: str) -> int:
    if tokenizer is None:
        return len(text)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids
    return int(ids.shape[1] if getattr(ids, "ndim", 1) == 2 else ids[0].shape[0])

def option_line(label: str, text: str) -> str:
    return f"{(label or 'A').strip().upper()}) {text}"

def options_block(labels: List[str], texts: List[str]) -> str:
    return "\n".join([option_line(L, t) for L, t in zip(labels, texts)])

@dataclass
class PromptStyle:
    system: str = "You are a helpful, factual assistant."
    question_label: str = "Question:"
    options_label: str = "Options:"
    answer_label: str = "Answer:"

# ---------- Canonical builders ----------

def q_only(tokenizer, question: str, style: PromptStyle = PromptStyle()) -> Tuple[str, int]:
    system = style.system
    user   = f"{style.question_label} {question}\n{style.answer_label}"
    prompt = _apply_chat_template(
        tokenizer,
        [{"role":"system","content":system},{"role":"user","content":user}],
        add_generation_prompt=True,
    ) or (system + "\n\n" + user)
    return prompt, _tok_len(tokenizer, prompt)

def q_with_options(tokenizer, question: str, options_text: str, style: PromptStyle = PromptStyle()) -> Tuple[str, int]:
    system = style.system
    user   = f"{style.question_label} {question}\n{style.options_label}\n{options_text}\n{style.answer_label}"
    prompt = _apply_chat_template(
        tokenizer,
        [{"role":"system","content":system},{"role":"user","content":user}],
        add_generation_prompt=True,
    ) or (system + "\n\n" + user)
    return prompt, _tok_len(tokenizer, prompt)

def uncond_letter(tokenizer, letter: str, style: PromptStyle = PromptStyle()) -> Tuple[str, str, int]:
    """No options; assistant begins with 'B)'. Return (prompt, full, prompt_len)."""
    system = style.system
    user   = f"{style.answer_label}"
    assistant = f"{(letter or 'A').strip().upper()})"
    prompt = _apply_chat_template(
        tokenizer,
        [{"role":"system","content":system},{"role":"user","content":user}],
        add_generation_prompt=True,
    ) or (system + "\n\n" + user)
    # IMPORTANT: keep assistant tag open (tests expect no closing tag in 'full')
    full = prompt + " " + assistant
    return prompt, full, _tok_len(tokenizer, prompt)

def uncond_assistant_text(tokenizer, assistant_text: str, style: PromptStyle = PromptStyle()) -> Tuple[str, str, int]:
    """No options; assistant free text. Return (prompt, full, prompt_len)."""
    system = style.system
    user   = f"{style.answer_label}"
    prompt = _apply_chat_template(
        tokenizer,
        [{"role":"system","content":system},{"role":"user","content":user}],
        add_generation_prompt=True,
    ) or (system + "\n\n" + user)
    # Same rationale: append text to an open assistant prompt
    full = prompt + " " + (assistant_text or "")
    return prompt, full, _tok_len(tokenizer, prompt)

def uncond_letter_with_options(
    tokenizer,
    options_text: str,
    *,
    letter: str = "A",
    style: PromptStyle = PromptStyle(),
) -> Tuple[str, str, int]:
    """
    Options block included; assistant begins with 'B)'. Return (prompt, full, prompt_len).
    Used by build_prompt_uncond(...) tests that require 'Options:' in the prompt.
    """
    system = style.system
    user   = f"{style.options_label}\n{options_text}\n{style.answer_label}"
    assistant = f"{(letter or 'A').strip().upper()})"
    prompt = _apply_chat_template(
        tokenizer,
        [{"role":"system","content":system},{"role":"user","content":user}],
        add_generation_prompt=True,
    ) or (system + "\n\n" + user)
    full = prompt + " " + assistant
    return prompt, full, _tok_len(tokenizer, prompt)

def openllm_prompt(question: str, options_text: str, style: PromptStyle = PromptStyle()) -> str:
    return f"{style.question_label} {question}\n{style.options_label}\n{options_text}\n{style.answer_label}"