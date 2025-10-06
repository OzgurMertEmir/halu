from ..core.runner import HFRunner
from ..pipeline import MetricsPipeline
from data.table_builder import build_option_table, collapse_option_side_features, pick_dataset
import pandas as pd

def build_features_df(model, tokenizer, dataset_name, size=None) -> pd.DataFrame:
    runner = HFRunner(model, tokenizer, store_device="cpu", store_dtype=model.dtype)
    pipe = MetricsPipeline(runner)
    dataset = pick_dataset(dataset_name)
    recs = dataset.build_feature_dataset(pipe, size)
    OPT = build_option_table(recs)
    OUT = collapse_option_side_features(OPT)
    return OUT