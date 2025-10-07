from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from pathlib import Path
import os, json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from halu.core.determinism import set_seed_and_register
from halu.features.build import build_features_df
from halu.engine import DetectorEnsemble, DetectorConfig
from halu.analysis.report import ReportInputs, generate_report

# ---------------- Config ----------------
@dataclass
class RunConfig:
    # HF model + dataset
    model_id: str = "meta-llama/Llama-3.2-1B"
    dataset: str = "truthfulqa"         # see halu.data.registry
    n_examples: Optional[int] = None    # None = all
    seed: int = 1337

    # compute / loading
    device: Optional[str] = None        # "cuda" | "mps" | "cpu" | None(auto)
    dtype: str = "auto"                 # "auto" | "fp16" | "bf16" | "fp32"

    # output
    out_dir: str = "runs/halu_run"

    # detector training
    epochs: int = 80
    hidden: int = 256
    calib_size: float = 0.20
    val_size: float = 0.20
    blend_grid: int = 101

    # reporting
    ece_bins: int = 15
    rel_bins: int = 12
    compute_bootstrap: bool = False
    n_boot: int = 1000
    alpha: float = 0.05


# ---------------- Utilities ----------------
def _pin_blas_threads():
    # Optional: helps bit-for-bit repeatability on some boxes
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

def _pick_device(name: Optional[str]) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _pick_dtype(pref: str) -> torch.dtype:
    pref = (pref or "auto").lower()
    if pref == "fp32":
        return torch.float32
    if pref == "fp16":
        return torch.float16
    if pref == "bf16":
        return torch.bfloat16
    # auto:
    if torch.cuda.is_available():
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32

def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "-", s.strip())
    return re.sub(r"-{2,}", "-", s).strip("-")


# ---------------- Orchestration ----------------
def run_all(cfg: RunConfig) -> Dict[str, Any]:
    # 0) Determinism FIRST (before model/dataset/etc.)
    _pin_blas_threads()
    set_seed_and_register(cfg.seed, deterministic=True)

    # Prepare output folder
    tag = f"{_slug(cfg.dataset)}__{_slug(cfg.model_id.split('/')[-1])}"
    out = Path(cfg.out_dir) / tag
    (out / "artifacts").mkdir(parents=True, exist_ok=True)

    # 1) Load HF model + tokenizer
    device = _pick_device(cfg.device)
    dtype  = _pick_dtype(cfg.dtype)

    tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    # Keep encoding stable across runs
    tok.padding_side = "right"
    tok.truncation_side = "right"

    model = AutoModelForCausalLM.from_pretrained(cfg.model_id,
                                                 torch_dtype=dtype,
                                                 device_map="auto",
                                                 trust_remote_code=True,
                                                 attn_implementation="eager")
    model.to(device).eval()

    # 2) Build features in one pass (deterministic)
    df = build_features_df(model, tok, cfg.dataset, size=cfg.n_examples, seed=cfg.seed)
    df.to_csv(out / "features.csv", index=False)

    # 3) Train detector (blend + calibrate)
    det_cfg = DetectorConfig(
        seed=cfg.seed,
        hidden=cfg.hidden,
        epochs=cfg.epochs,
        lr=3e-3,
        weight_decay=2e-4,
        gamma=2.0,
        calib_size=cfg.calib_size,
        val_size=cfg.val_size,
        blend_grid=cfg.blend_grid,
        device=str(device),
        batch=4096,
    )
    det = DetectorEnsemble(det_cfg)
    train_summary = det.fit(df)
    det.save(str(out / "artifacts" / "detector"))

    # 4) Predict calibrated P(error) on the whole set
    p_err = det.predict_proba(df)

    # 5) Report (uses detector P(error); calib optional)
    report_inp = ReportInputs(
        df_test=df,
        p_err_test=p_err,
        df_calib=None,
        p_err_calib=None,
        ece_bins=cfg.ece_bins,
        rel_bins=cfg.rel_bins,
        compute_bootstrap=cfg.compute_bootstrap,
        n_boot=cfg.n_boot,
        alpha=cfg.alpha,
        seed=cfg.seed,
        classification_threshold=0.5,
    )
    rep = generate_report(report_inp, out_dir=str(out / "report"), prefix=cfg.dataset)

    # 6) Manifest
    manifest = {
        "config": asdict(cfg),
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "train_summary": train_summary,
        "report": rep,
        "paths": {
            "features_csv": str(out / "features.csv"),
            "artifacts_dir": str(out / "artifacts"),
            "report_dir": str(out / "report"),
        },
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest