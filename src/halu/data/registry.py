# Halu/data/registry.py
from __future__ import annotations
from halu.data.truthfulqa import TruthfulQADataset
from halu.data.open_llm import OpenLLMDataset

def pick_dataset(name: str):
    name = (name or "").strip().lower()
    if name in {"truthfulqa", "truthful_qa", "tqa"}:
        return TruthfulQADataset()
    if name in {"open_llm", "open-llm", "openllm"}:
        return OpenLLMDataset()
    raise ValueError(f"Unknown dataset: {name}")