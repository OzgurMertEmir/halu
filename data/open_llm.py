from .base import Dataset, DatasetRow
from datasets import load_dataset
from ..core.types import MCQExample, MCQOption
import numpy as np
from tqdm import tqdm

class OpenLLMDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.original_dataset = load_dataset("Open-Style/Open-LLM-Benchmark", "questions")["train"]
        self.dataset = []
        self.feature_dataset = []

    def build_feature_dataset(self, pipe, sample_size=None):
        if sample_size is not None:
            random_indices = np.random.choice(len(self.original_dataset), size=sample_size, replace=False)
            subset = self.original_dataset.select(random_indices)
        else:
            subset = self.original_dataset

        for i, row in tqdm(enumerate(subset), total=len(subset)):
            dataset_row = OpenLLMDatasetRow(row)
            ex = dataset_row.row_to_mcq(i)
            row = pipe.example_to_row(ex)
            self.feature_dataset.append(row)

        return self.feature_dataset

class OpenLLMDatasetRow(DatasetRow):
    def row_to_mcq(self, id):
        option_texts = [opt["text"] for opt in self.row["options"]]
        labels = [opt["label"] for opt in self.row["options"]]
        if self.row["passage"] != "-":
            q = self.row["passage"] + "\n\n" + self.row["question"]
        else:
            q = self.row["question"]

        ex = MCQExample(
            qid=f"{id}",
            question=q,
            options=[MCQOption(label, text) for label, text in zip(labels, option_texts)],
            gold_letter=self.row["answerKey"], dataset=self.row["dataset"]
        )

        return ex
