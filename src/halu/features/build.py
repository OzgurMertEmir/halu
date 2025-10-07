# Halu/features/build.py
from __future__ import annotations
from typing import Optional
import pandas as pd
from halu.core.runner import HFRunner
from halu.pipeline import MetricsPipeline
from halu.data.registry import pick_dataset
from halu.features.tables import build_option_table, collapse_option_side_features
from tqdm import tqdm

def build_features_df(
    model,
    tokenizer,
    dataset_name: str,
    size: Optional[int] = None,
    seed: Optional[int] = 1337,
) -> pd.DataFrame:
    runner = HFRunner(model, tokenizer, store_device="cpu", store_dtype=getattr(model, "dtype", None))
    pipe = MetricsPipeline(runner)
    dataset = pick_dataset(dataset_name)

    recs = []
    for ex in tqdm(dataset.iter_examples(sample_size=size, seed=seed), "Building examples"):
        recs.append(pipe.example_to_row(ex))

    OPT = build_option_table(recs)
    OUT = collapse_option_side_features(OPT)
    return OUT