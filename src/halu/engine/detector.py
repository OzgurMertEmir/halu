from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import os, json
import numpy as np
import pandas as pd
import torch
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

# Reuse tried-and-true bits from your model package
from halu.model.ensemble import (
    MLP, _train_one_nn, _train_one_tree, _predict_tree,
    pick_best_calibrator, build_xy,
)

# -------- Config --------
@dataclass
class DetectorConfig:
    seed: int = 1337
    hidden: int = 256
    epochs: int = 100
    lr: float = 3e-3
    weight_decay: float = 2e-4
    gamma: float = 2.0                 # focal loss gamma for NN
    calib_size: float = 0.20           # fraction (if df_calib not provided)
    val_size: float = 0.20             # NN/Tree early-stopping split from train
    blend_grid: int = 101              # weight search granularity
    device: Optional[str] = None       # "cuda", "mps", "cpu" or None = auto
    batch: int = 4096                  # predict batch size

# -------- Detector --------
class DetectorEnsemble:
    """
    Unified detector API:
      - fit(df[, df_calib]) : train NN + Tree, choose blend weight on calib, fit calibrator
      - predict_proba(df)   : calibrated P(error) for new rows
      - save(dir) / load(dir)
    Expects Halu’s canonical table columns (from features.tables.collapse_option_side_features):
      'chosen', 'gold' + numeric feature columns.
    """

    def __init__(self, config: DetectorConfig | None = None):
        self.cfg = config or DetectorConfig()
        self.meta: Dict[str, Any] = {}
        self.nn_model: Optional[MLP] = None
        self.tree_model = None
        self.w_opt: Optional[float] = None
        self.calibrator = None
        self._input_dim: Optional[int] = None
        self._fitted: bool = False

    # ----- internals -----
    def _device(self):
        if self.cfg.device:
            return torch.device(self.cfg.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    '''def _build_X_like_train(self, df: pd.DataFrame) -> np.ndarray:
        """Use saved meta to project new data into the training feature space."""
        num_cols = self.meta["num_cols"]
        X_df = pd.DataFrame({
            c: pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
            for c in num_cols
        })
        X_imp = self.meta["imputer"].transform(X_df)
        X_std = self.meta["scaler"].transform(X_imp)
        return X_std.astype(np.float32)'''

    def _build_X_like_train(self, df: pd.DataFrame) -> np.ndarray:
        """Use saved meta to project new data into the training feature space."""
        num_cols = self.meta["num_cols"]

        # 1) take the training schema columns, coerce to numeric, fill missing columns with NaN
        X_df = pd.DataFrame({
            c: pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
            for c in num_cols
        })

        # 2) replace ±inf with NaN so the imputer can handle them
        X_df = X_df.replace([np.inf, -np.inf], np.nan)

        # 3) impute + scale with the training-time transformers
        X_imp = self.meta["imputer"].transform(X_df)
        X_std = self.meta["scaler"].transform(X_imp)
        return X_std.astype(np.float32)

    def _predict_components(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Uncalibrated component probabilities: (p_nn, p_tree)."""
        dev = self._device()
        self.nn_model.to(dev).eval()
        ps = []
        with torch.inference_mode():
            for i in range(0, len(X), self.cfg.batch):
                xb = torch.from_numpy(X[i:i+self.cfg.batch]).to(dev)
                ps.append(torch.sigmoid(self.nn_model(xb)).float().cpu().numpy().ravel())
        p_nn = np.concatenate(ps) if ps else np.zeros(len(X), dtype=float)

        if hasattr(self.tree_model, "predict_proba"):
            try:
                # If the tree expects a DataFrame (e.g., LightGBM w/ column names)
                import pandas as _pd  # noqa
                X_df = pd.DataFrame(X, columns=self.meta["num_cols"])
                p_tree = self.tree_model.predict_proba(X_df)[:, 1]
            except Exception:
                p_tree = self.tree_model.predict_proba(X)[:, 1]
        else:
            p_tree = np.zeros(len(X), dtype=float)

        return p_nn.astype(float), p_tree.astype(float)

    def _apply_calibrator(self, p_raw: np.ndarray) -> np.ndarray:
        cal = self.calibrator
        if hasattr(cal, "transform"):
            return cal.transform(p_raw).astype(float)
        if hasattr(cal, "predict_proba"):
            return cal.predict_proba(p_raw.reshape(-1, 1))[:, 1].astype(float)
        if hasattr(cal, "predict"):
            return cal.predict(p_raw.reshape(-1, 1)).astype(float)
        # Last resort: identity (shouldn't happen if pick_best_calibrator does its job)
        return p_raw.astype(float)

    def _calibrated_preds(self, cal, p: np.ndarray) -> np.ndarray:
        if hasattr(cal, "transform"):
            return cal.transform(p).astype(float)
        if hasattr(cal, "predict_proba"):
            return cal.predict_proba(p.reshape(-1, 1))[:, 1].astype(float)
        # e.g. IsotonicRegression
        return cal.predict(p).astype(float)

    # ----- public API -----
    def fit(self, df: pd.DataFrame, df_calib: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Train NN + Tree on df (with internal train/val), pick blend weight on calib set,
        and fit the best calibrator on calib predictions.
        Returns a summary dict of key metrics for quick logging.
        """
        # Build X/y and normalization meta
        X_all, y_all, meta = build_xy(df.copy())
        self.meta = meta
        self._input_dim = X_all.shape[1]

        # Split train/val for early stopping
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_all, y_all, test_size=self.cfg.val_size,
            random_state=self.cfg.seed, stratify=y_all
        )

        # Train components
        self.nn_model = _train_one_nn(
            X_tr, y_tr, X_va, y_va,
            hidden=self.cfg.hidden, epochs=self.cfg.epochs,
            lr=self.cfg.lr, wd=self.cfg.weight_decay,
            gamma=self.cfg.gamma, seed=self.cfg.seed
        )

        self.tree_model = _train_one_tree(
            X_tr, y_tr, X_va, y_va, seed=self.cfg.seed, cols=None
        )

        # Build/choose calibration set
        if df_calib is None:
            # carve from df (disjoint from val used for early stopping)
            X_tr2, X_ca, y_tr2, y_ca = train_test_split(
                X_all, y_all, test_size=self.cfg.calib_size,
                random_state=self.cfg.seed + 1, stratify=y_all
            )
            X_for_calib, y_for_calib = X_ca, y_ca
        else:
            X_for_calib = self._build_X_like_train(df_calib)
            y_for_calib = (df_calib["chosen"].astype(str).str.upper()
                           != df_calib["gold"].astype(str).str.upper()).astype(int).values

        # Blend weight on calibration set
        p_nn_ca, p_tree_ca = self._predict_components(X_for_calib)

        ws = np.linspace(0, 1, self.cfg.blend_grid)
        briers = [brier_score_loss(y_for_calib, w*p_nn_ca + (1-w)*p_tree_ca) for w in ws]
        j = int(np.argmin(briers))
        self.w_opt = float(ws[j])

        p_blend_ca = self.w_opt * p_nn_ca + (1.0 - self.w_opt) * p_tree_ca

        # Calibrator selection
        best_name, best_cal, fits = pick_best_calibrator(y_for_calib, p_blend_ca)
        self.calibrator = best_cal

        # Quick snapshot metrics on df (uncalibrated & calibrated)
        p_nn_all, p_tree_all = self._predict_components(X_all)
        p_blend_all = self.w_opt * p_nn_all + (1.0 - self.w_opt) * p_tree_all
        p_cal_all = self._calibrated_preds(self.calibrator, p_blend_all)

        self._fitted = True
        return {
            "blend_weight": self.w_opt,
            "calibrator": best_name,
            "calib_brier": float(brier_score_loss(y_for_calib, p_blend_ca)),
            "train_brier_uncal": float(brier_score_loss(y_all, p_blend_all)),
            "train_brier_cal": float(brier_score_loss(y_all, p_cal_all)),
            "calibrator_fits": {k: f"{b:.4f}" for k, (_, b) in fits.items()},
        }

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return calibrated P(error) for each row of df."""
        assert self._fitted, "DetectorEnsemble is not fitted. Call fit(...) first."
        X = self._build_X_like_train(df)
        p_nn, p_tree = self._predict_components(X)
        p_blend = self.w_opt * p_nn + (1.0 - self.w_opt) * p_tree
        return self._calibrated_preds(self.calibrator, p_blend)

    def save(self, out_dir: str) -> None:
        """Save model + preproc to a directory."""
        assert self._fitted, "Nothing to save: fit the detector first."
        os.makedirs(out_dir, exist_ok=True)

        # Config & blend info
        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"config": asdict(self.cfg), "w_opt": self.w_opt}, f, indent=2)

        # Preprocessing/meta & tree & calibrator
        joblib.dump(self.meta,        os.path.join(out_dir, "meta.joblib"))
        joblib.dump(self.tree_model,  os.path.join(out_dir, "tree.joblib"))
        joblib.dump(self.calibrator,  os.path.join(out_dir, "calibrator.joblib"))

        # NN
        torch.save({
            "state_dict": self.nn_model.state_dict(),
            "d_in": self._input_dim,
            "hidden": self.cfg.hidden,
        }, os.path.join(out_dir, "nn.pt"))

    @classmethod
    def load(cls, in_dir: str) -> "DetectorEnsemble":
        """Load a saved detector from a directory."""
        with open(os.path.join(in_dir, "config.json"), "r", encoding="utf-8") as f:
            info = json.load(f)
        cfg = DetectorConfig(**info.get("config", {}))
        obj = cls(cfg)

        obj.meta        = joblib.load(os.path.join(in_dir, "meta.joblib"))
        obj.tree_model  = joblib.load(os.path.join(in_dir, "tree.joblib"))
        obj.calibrator  = joblib.load(os.path.join(in_dir, "calibrator.joblib"))

        nn_blob = torch.load(os.path.join(in_dir, "nn.pt"), map_location="cpu")
        d_in   = int(nn_blob.get("d_in"))
        hidden = int(nn_blob.get("hidden", cfg.hidden))
        model  = MLP(d_in, hidden)
        model.load_state_dict(nn_blob["state_dict"])
        obj.nn_model = model
        obj._input_dim = d_in

        obj.w_opt = float(info["w_opt"])
        obj._fitted = True
        return obj