# data/open_llm.py
from __future__ import annotations
from typing import List
from datasets import load_dataset
from halu.data.base import Dataset, DatasetRow
from halu.core.types import MCQExample, MCQOption

class OpenLLMDataset(Dataset):
    def __init__(self, split: str = "train"):
        super().__init__()
        self.original_dataset = load_dataset("Open-Style/Open-LLM-Benchmark", "questions")[split]

    def _row_adapter(self, row) -> "OpenLLMDatasetRow":
        return OpenLLMDatasetRow(row)

class OpenLLMDatasetRow(DatasetRow):
    def row_to_mcq(self, qid: str) -> MCQExample:
        # Options already carry labels (e.g., "A","B","C","D")
        opts: List[dict] = list(self.row["options"])
        labels = [str(o["label"]).strip().upper() for o in opts]
        texts  = [str(o["text"]) for o in opts]

        # Build prompt question (passage + question) deterministically
        passage = str(self.row.get("passage", "-"))
        question = (passage + "\n\n" if passage and passage != "-" else "") + str(self.row["question"])

        gold_letter = str(self.row.get("answerKey", "")).strip().upper() or None
        dataset_name = str(self.row.get("dataset", "open_llm"))

        return MCQExample(
            qid=qid,
            question=question,
            options=[MCQOption(L, t) for L, t in zip(labels, texts)],
            gold_letter=gold_letter,
            dataset=dataset_name,
        )