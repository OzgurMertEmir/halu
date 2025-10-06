from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import torch

@dataclass
class MCQOption:
    label: str
    text: str

@dataclass
class MCQExample:
    qid: Optional[str]
    question: str
    options: List[MCQOption]
    gold_letter: Optional[str] = None
    dataset: Optional[str] = None

@dataclass
class ForwardPack:
    # prompt/response boundary
    prompt: str
    response: str
    prompt_len: int
    total_len: int
    resp_len: int

    # logits
    logits_resp: Optional[torch.Tensor]                 # [T_resp, V]
    logits_full: Optional[torch.Tensor] = None          # [T, V] (optional)

    # attentions: per-layer only
    attn_layers: Optional[List[torch.Tensor]] = None    # length L, each [H, T, T]

    # hidden states: per-layer only
    hidden_layers: Optional[List[torch.Tensor]] = None  # length L, each [T, D]
    hidden_embed: Optional[torch.Tensor] = None         # [T, D] (embeddings at index 0, optional)
