# metrics/base.py
from abc import ABC, abstractmethod

class DatasetRow(ABC):
    def __init__(self, row):
        self.row = row
    @abstractmethod
    def row_to_mcq(self, id):
        ...

class Dataset():
    @abstractmethod
    def __init__(self):
        ...

    @abstractmethod
    def build_feature_dataset(self, pipe, sample_size=None):
        ...

LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
