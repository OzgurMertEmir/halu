from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Any, Dict
import json, os

try:
    import yaml  # optional; falls back to JSON if missing
except Exception:
    yaml = None

# ---- Nested sections --------------------------------------------------------

@dataclass
class DatasetCfg:
    name: str = "truthfulqa"
    size: Optional[int] = None
    seed: int = 1337

@dataclass
class RunnerCfg:
    store_device: str = "cpu"              # "cpu" | "gpu"
    store_dtype: Optional[str] = None      # "float16"|"bfloat16"|None -> auto
    include_embeddings: bool = False
    keep_logits_full: bool = False

@dataclass
class PipelineCfg:
    use_icr: bool = True
    use_llmcheck: bool = True
    use_fcm: bool = True

@dataclass
class DetectorCfg:
    hidden: int = 256
    epochs: int = 100
    k_folds: int = 5
    n_seeds_nn: int = 3
    n_seeds_tree: int = 3
    seed: int = 1337

@dataclass
class ReportCfg:
    ece_bins: int = 15
    rel_bins: int = 12
    compute_bootstrap: bool = False
    n_boot: int = 1000
    alpha: float = 0.05
    abstain_targets: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.40)

@dataclass
class PathsCfg:
    out_dir: str = "runs"
    report_dir: str = "report_out"
    cache_dir: str = ".cache"

# ---- Top-level --------------------------------------------------------------

@dataclass
class HaluConfig:
    seed: int = 1337
    deterministic: bool = True
    dataset: DatasetCfg = field(default_factory=DatasetCfg)
    runner: RunnerCfg = field(default_factory=RunnerCfg)
    pipeline: PipelineCfg = field(default_factory=PipelineCfg)
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    report: ReportCfg = field(default_factory=ReportCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)

    @staticmethod
    def from_file(path: str) -> "HaluConfig":
        with open(path, "r", encoding="utf-8") as f:
            s = f.read()
        data: Dict[str, Any]
        if yaml is not None and (path.endswith(".yml") or path.endswith(".yaml")):
            data = yaml.safe_load(s) or {}
        else:
            data = json.loads(s)

        def _merge(dc, cls):
            if dc is None: return cls()
            merged = cls(**{**asdict(cls()), **dc})
            return merged

        return HaluConfig(
            seed=data.get("seed", 1337),
            deterministic=data.get("deterministic", True),
            dataset=_merge(data.get("dataset"), DatasetCfg),
            runner=_merge(data.get("runner"), RunnerCfg),
            pipeline=_merge(data.get("pipeline"), PipelineCfg),
            detector=_merge(data.get("detector"), DetectorCfg),
            report=_merge(data.get("report"), ReportCfg),
            paths=_merge(data.get("paths"), PathsCfg),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)