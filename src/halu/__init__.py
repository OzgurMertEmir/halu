# halu/__init__.py
from importlib import import_module

__version__ = "0.1.0"
__all__ = ["__version__"]  # convenience names are provided lazily via __getattr__

def __getattr__(name: str):
    # Pure-math metrics (safe to import anytime)
    if name in ("ece_binary", "reliability_table", "aurc_and_auacc",
                "coverage_at_accuracy", "bootstrap_ci"):
        mod = import_module("halu.analysis.eval_metrics")
        return getattr(mod, name)

    # Convenience re-exports (lazy; only import torch/transformers if you actually touch these)
    if name == "HFRunner":
        return getattr(import_module("halu.core.runner"), "HFRunner")
    if name == "MetricsPipeline":
        return getattr(import_module("halu.pipeline"), "MetricsPipeline")
    if name == "build_features_df":
        return getattr(import_module("halu.features.build"), "build_features_df")

    raise AttributeError(f"module 'halu' has no attribute {name!r}")