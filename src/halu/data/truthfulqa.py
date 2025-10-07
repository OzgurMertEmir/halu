# data/truthfulqa.py
from __future__ import annotations
import numpy as np
from datasets import load_dataset
from halu.data.base import Dataset, DatasetRow, LETTERS
from halu.core.types import MCQExample, MCQOption

class TruthfulQADataset(Dataset):
    def __init__(self):
        super().__init__()
        self.original_dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice")["validation"]

    def _row_adapter(self, row) -> TruthfulQADatasetRow:
        return TruthfulQADatasetRow(row)

class TruthfulQADatasetRow(DatasetRow):
    def row_to_mcq(self, qid: str) -> MCQExample:
        q = self.row['question']
        mc1 = self.row['mc1_targets']
        option_texts = mc1['choices']
        labels = [LETTERS[i] for i in range(len(option_texts))]
        correct_idx = mc1['labels'].index(1)
        correct_option = option_texts[correct_idx]
        np.random.seed(42)
        options_copy = np.empty_like(option_texts)
        np.copyto(options_copy, labels)
        np.random.shuffle(options_copy)
        gold_idx = options_copy.index(correct_option)
        gold_letter = LETTERS[gold_idx]
        return MCQExample(
            qid=f"{qid}",
            question=q,
            options=[MCQOption(label, text) for label, text in zip(labels, options_copy)],
            gold_letter=gold_letter,
            dataset="truthfulqa"
        )
