from __future__ import annotations
from typing import List, Dict
from halu.core.types import MCQExample
from halu.pipeline import MetricsPipeline

def extract_features(ex: MCQExample, pipeline: MetricsPipeline) -> List[Dict]:
    return [pipeline.example_to_row(ex)]
