from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from pathlib import Path
import os, json, re, torch
import numpy as np
from sklearn.model_selection import train_test_split
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

    # split for reporting
    test_size: float = 0.30             # <— NEW: held-out test proportion

    # reporting
    ece_bins: int = 15
    rel_bins: int = 12
    compute_bootstrap: bool = False
    n_boot: int = 1000
    alpha: float = 0.05


# ---------------- Utilities ----------------
def _pin_blas_threads():
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
    if pref == "fp32": return torch.float32
    if pref == "fp16": return torch.float16
    if pref == "bf16": return torch.bfloat16
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
    # 0) Determinism FIRST
    _pin_blas_threads()
    set_seed_and_register(cfg.seed, deterministic=True)

    # Prepare output folder
    tag = f"{_slug(cfg.dataset)}__{_slug(cfg.model_id.split('/')[-1])}"
    out = Path(cfg.out_dir) / tag
    art_dir = out / "artifacts"
    rep_dir = out / "report"
    art_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load HF model + tokenizer
    device = _pick_device(cfg.device)
    dtype = _pick_dtype(cfg.dtype)
    tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right";
    tok.truncation_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, torch_dtype=dtype, device_map="auto", trust_remote_code=True, attn_implementation="eager"
    ).eval()

    # 2) Build features once (deterministic)
    df_all = build_features_df(model, tok, cfg.dataset, size=cfg.n_examples, seed=cfg.seed)
    df_all.to_csv(out / "features.csv", index=False)

    # 2.1) Split into TRAIN / TEST for unbiased reporting
    y_all = (df_all["chosen"].astype(str).str.upper() != df_all["gold"].astype(str).str.upper()).astype(int).values
    strat = y_all if np.unique(y_all).size >= 2 else None
    df_train, df_test = train_test_split(
        df_all, test_size=cfg.test_size, random_state=cfg.seed, stratify=strat
    )
    df_train.to_csv(out / "features_train.csv", index=False)
    df_test.to_csv(out / "features_test.csv", index=False)

    # 3) Train detector (blend + calibrate) on TRAIN only
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
    train_summary = det.fit(df_train)
    det.save(str(out / "artifacts" / "detector"))

    # 4) Predict calibrated P(error)
    p_err_test = det.predict_proba(df_test)
    p_err_train = det.predict_proba(df_train)

    # Save test predictions for inspection
    (df_test.assign(p_err=p_err_test)).to_csv(rep_dir / f"{cfg.dataset}_predictions.csv", index=False)

    # 5) Report on TEST (taus derived from TRAIN to avoid leakage)
    report_inp = ReportInputs(
        df_test=df_test, p_err_test=p_err_test,
        df_calib=df_train, p_err_calib=p_err_train,
        ece_bins=cfg.ece_bins, rel_bins=cfg.rel_bins,
        compute_bootstrap=cfg.compute_bootstrap, n_boot=cfg.n_boot,
        alpha=cfg.alpha, seed=cfg.seed, classification_threshold=0.5,
    )
    rep = generate_report(report_inp, out_dir=str(rep_dir), prefix=cfg.dataset)

    # 6) Manifest
    manifest = {
        "config": asdict(cfg),
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "train_summary": train_summary,
        "report": rep,
        "paths": {
            "features_csv": str(out / "features.csv"),
            "artifacts_dir": str(art_dir),
            "report_dir": str(rep_dir),
        },
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest