# halu/__init__.py
from halu.core.types import MCQOption, MCQExample, ForwardPack
from halu.core.runner import HFRunner
from halu.pipeline import MetricsPipeline
from halu.engine import DetectorEnsemble, DetectorConfig

# Light re-exports for public API
from halu.features.build import build_features_df
from halu.analysis.eval_metrics import (
    ece_binary, reliability_table, aurc_and_auacc, coverage_at_accuracy, bootstrap_ci
)