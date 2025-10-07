# data/truthfulqa.py
from __future__ import annotations
import numpy as np
from datasets import load_dataset
from halu.data.base import Dataset, DatasetRow, LETTERS
from halu.core.types import MCQExample, MCQOption

class TruthfulQADataset(Dataset):
    def __init__(self, shuffle_options: bool = False, shuffle_seed: int | None = None):
        super().__init__()
        self.shuffle_options = bool(shuffle_options)
        self.shuffle_seed = int(shuffle_seed) if shuffle_seed is not None else None
        self.original_dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice")["validation"]

    def _row_adapter(self, row) -> TruthfulQADatasetRow:
        return TruthfulQADatasetRow(row, shuffle_options=self.shuffle_options, shuffle_seed=self.shuffle_seed)

class TruthfulQADatasetRow(DatasetRow):
    def __init__(self, row, shuffle_options: bool = False, shuffle_seed: int | None = None):
        super().__init__(row)
        self.shuffle_options = bool(shuffle_options)
        self.shuffle_seed = int(shuffle_seed) if shuffle_seed is not None else None

    def row_to_mcq(self, qid: str) -> MCQExample:
        q = self.row['question']
        mc1 = self.row['mc1_targets']
        option_texts = list(mc1['choices'])
        n = len(option_texts)
        labels = [LETTERS[i] for i in range(n)]
        correct_idx = mc1['labels'].index(1)

        # Optional deterministic shuffling of option texts only
        if self.shuffle_options and n > 1:
            base = self.shuffle_seed if self.shuffle_seed is not None else 0
            qid_int = int(qid)
            rng = np.random.default_rng(base + qid_int)
            perm = rng.permutation(n)
            texts_shuffled = [option_texts[i] for i in perm]
            # Find new gold position and letter
            gold_pos = list(perm).index(correct_idx)
            gold_letter = labels[gold_pos]
        else:
            texts_shuffled = option_texts
            gold_letter = labels[correct_idx]

        return MCQExample(
            qid=f"{qid}",
            question=q,
            options=[MCQOption(label, text) for label, text in zip(labels, texts_shuffled)],
            gold_letter=gold_letter,
            dataset="truthfulqa"
        )
