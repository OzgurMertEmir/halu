from .base import Dataset, DatasetRow, LETTERS
from datasets import load_dataset
from ..core.types import MCQExample, MCQOption
import numpy as np
from tqdm import tqdm

class TruthfulQADataset(Dataset):
    def __init__(self):
        super().__init__()
        self.original_dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice")['validation']
        self.dataset = []
        self.feature_dataset = []

    def build_feature_dataset(self, pipe, sample_size=None):
        if sample_size is not None:
            random_indices = np.random.choice(len(self.original_dataset), size=sample_size, replace=False)
            subset = self.original_dataset.select(random_indices)
        else:
            subset = self.original_dataset

        for i, row in tqdm(enumerate(subset), total=len(subset)):
            dataset_row = TruthfulQADatasetRow(row)
            ex = dataset_row.row_to_mcq(i)
            row = pipe.example_to_row(ex)
            self.feature_dataset.append(row)

        return self.feature_dataset

class TruthfulQADatasetRow(DatasetRow):
    def row_to_mcq(self, id):
        q = self.row['question']
        mc1 = self.row['mc1_targets']
        option_texts = mc1['choices']
        labels = [LETTERS[i] for i in range(len(option_texts))]
        correct_idx = mc1['labels'].index(1)
        correct_option = option_texts[correct_idx]
        np.random.shuffle(option_texts)
        gold_idx = option_texts.index(correct_option)
        gold_letter = LETTERS[gold_idx]
        ex = MCQExample(
            qid=f"{id}",
            question=q,
            options=[MCQOption(label, text) for label, text in zip(labels, option_texts)],
            gold_letter=gold_letter, dataset="truthfulqa"
        )
        return ex
