# data/base.py
from abc import ABC, abstractmethod
from typing import Any, Optional, Iterator, Tuple
import numpy as np

class DatasetRow(ABC):
    def __init__(self, row):
        self.row = row
    @abstractmethod
    def row_to_mcq(self, qid):
        ...

class Dataset(ABC):
    """Slim base dataset: load once, iterate MCQExamples deterministically."""
    def __init__(self):
        self.original_dataset = None  # subclasses set this

    @abstractmethod
    def _row_adapter(self, row: Any) -> DatasetRow:
        """Return the DatasetRow adapter for a raw row."""
        ...

    def _iter_subset(self, sample_size: Optional[int], seed: Optional[int]) -> Iterator[Tuple[int, Any]]:
        ds = self.original_dataset
        n = len(ds)
        if sample_size is None or sample_size >= n:
            # full pass, deterministic order
            for i in range(n):
                yield i, ds[i]
        else:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n, size=sample_size, replace=False)
            for i in idx:
                yield int(i), ds[int(i)]

    def iter_examples(self, sample_size: Optional[int] = None, seed: Optional[int] = None):
        """Yield MCQExample objects. No shuffling of options here."""
        assert self.original_dataset is not None, "Subclass must set self.original_dataset"
        for i, row in self._iter_subset(sample_size, seed):
            ex = self._row_adapter(row).row_to_mcq(qid=str(i))
            yield ex

LETTERS = [chr(ord('A') + i) for i in range(26)]
