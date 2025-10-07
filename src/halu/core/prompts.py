# halu/core/prompts.py
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
        return len(text)  # fallback (not used for slicing logits, only for defaults)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids
    return int(ids.shape[1] if ids.ndim == 2 else ids[0].shape[0])

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
    """
    System + (Question ... Answer:) — NO options. Returns (prompt_text, prompt_len).
    """
    system = style.system
    user   = f"{style.question_label} {question}\n{style.answer_label}"
    # Prefer chat template if available
    prompt = _apply_chat_template(
        tokenizer,
        [{"role":"system","content":system},{"role":"user","content":user}],
        add_generation_prompt=True,
    )
    if prompt is None:
        prompt = system + "\n\n" + user
    return prompt, _tok_len(tokenizer, prompt)

def q_with_options(tokenizer, question: str, options_text: str, style: PromptStyle = PromptStyle()) -> Tuple[str, int]:
    """
    System + (Question + Options + Answer:) — for *letter* scoring / full MCQ.
    Returns (prompt_text, prompt_len).
    """
    system = style.system
    user   = f"{style.question_label} {question}\n{style.options_label}\n{options_text}\n{style.answer_label}"
    prompt = _apply_chat_template(
        tokenizer,
        [{"role":"system","content":system},{"role":"user","content":user}],
        add_generation_prompt=True,
    )
    if prompt is None:
        prompt = system + "\n\n" + user
    return prompt, _tok_len(tokenizer, prompt)

def uncond_letter(tokenizer, letter: str, style: PromptStyle = PromptStyle()) -> Tuple[str, str, int]:
    """
    Unconditional scaffold WITHOUT options; assistant begins with 'B)' etc.
    Returns (prompt_text, full_text, prompt_len).
    """
    system = style.system
    user   = f"{style.answer_label}"
    assistant = f"{(letter or 'A').strip().upper()})"
    prompt = _apply_chat_template(
        tokenizer,
        [{"role":"system","content":system},{"role":"user","content":user}],
        add_generation_prompt=True,
    )
    full = _apply_chat_template(
        tokenizer,
        [{"role":"system","content":system},{"role":"user","content":user},{"role":"assistant","content":assistant}],
        add_generation_prompt=False,
    )
    if prompt is None or full is None:
        prompt = (system + "\n\n" + user)
        full   = prompt + " " + assistant
    return prompt, full, _tok_len(tokenizer, prompt)

def uncond_assistant_text(tokenizer, assistant_text: str, style: PromptStyle = PromptStyle()) -> Tuple[str, str, int]:
    """
    Unconditional scaffold WITHOUT options; assistant is full free text (e.g. 'B) Mars ...').
    Returns (prompt_text, full_text, prompt_len).
    """
    system = style.system
    user   = f"{style.answer_label}"
    prompt = _apply_chat_template(
        tokenizer,
        [{"role":"system","content":system},{"role":"user","content":user}],
        add_generation_prompt=True,
    )
    full = _apply_chat_template(
        tokenizer,
        [{"role":"system","content":system},{"role":"user","content":user},{"role":"assistant","content":assistant_text}],
        add_generation_prompt=False,
    )
    if prompt is None or full is None:
        prompt = (system + "\n\n" + user)
        full   = prompt + " " + (assistant_text or "")
    return prompt, full, _tok_len(tokenizer, prompt)

def openllm_prompt(question: str, options_text: str, style: PromptStyle = PromptStyle()) -> str:
    """
    Plain-text prompt for models w/o chat templates, used by FCM.
    """
    return f"{style.question_label} {question}\n{style.options_label}\n{options_text}\n{style.answer_label}"