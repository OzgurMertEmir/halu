# Halu/model/ensemble.py
from __future__ import annotations
import math, numpy as np, pandas as pd, torch
from typing import Dict
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from halu.analysis.eval_metrics import ece_binary, reliability_table

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier

# ---- build XY ----
def build_xy(df: pd.DataFrame):
    y = (df["chosen"].str.upper() != df["gold"].str.upper()).astype(int).values
    drop_cols = {"qid","dataset","gold","chosen","opt_top_letter","fcm_top_letter"}
    num_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    X = df[num_cols].copy()
    imputer = SimpleImputer(strategy="median"); scaler = StandardScaler()
    X_imp = imputer.fit_transform(X); X_std = scaler.fit_transform(X_imp)
    meta = dict(num_cols=num_cols, imputer=imputer, scaler=scaler)
    return X_std.astype(np.float32), y.astype(np.int64), meta

# ---- simple MLP ----
class MLP(torch.nn.Module):
    def __init__(self, d_in: int, h):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, h*2), torch.nn.ReLU(), torch.nn.BatchNorm1d(h*2), torch.nn.Dropout(0.1),
            torch.nn.Linear(h*2, h),   torch.nn.ReLU(), torch.nn.BatchNorm1d(h),   torch.nn.Dropout(0.1),
            torch.nn.Linear(h, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

def _evaluate_raw(model, X, y, device):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        xb = torch.from_numpy(X).to(device)
        logits = model(xb)
        p_err = torch.sigmoid(logits).float().cpu().numpy()
        ps.append(p_err); ys.append(y)
    y = np.concatenate(ys); p = np.concatenate(ps)
    return y, p

def _train_one_nn(X_tr, y_tr, X_val, y_val, hidden=256, epochs=100, lr=3e-3, wd=2e-4, gamma=2.0, seed=1337):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLP(X_tr.shape[1], hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
    pos_rate = float(y_tr.mean()); alpha = float(max(0.05, min(0.95, 1.0 - pos_rate)))

    def focal_loss_with_logits(logits, targets, alpha=alpha, gamma=gamma):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        p = torch.sigmoid(logits)
        p_t = p*targets + (1-p)*(1-targets)
        alpha_t = alpha*targets + (1-alpha)*(1-targets)
        return (alpha_t * (1 - p_t).pow(gamma) * bce).mean()

    best_key, best_state, n_bad = math.inf, None, 0
    for _ in range(epochs):
        model.train()
        xb = torch.from_numpy(X_tr).to(device); yb = torch.from_numpy(y_tr).to(device)
        loss = focal_loss_with_logits(model(xb), yb)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        if X_val is not None:
            yv, pv = _evaluate_raw(model, X_val, y_val, device)
            key = brier_score_loss(yv, pv)
            sched.step(key)
            if key < best_key:
                best_key, best_state, n_bad = key, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
            else:
                n_bad += 1
            if n_bad >= 10: break
    if best_state: model.load_state_dict(best_state)
    return model

def _train_one_tree(X_tr, y_tr, X_val, y_val, seed=1337, cols: list[str] | None = None):
    if _HAS_LGB:
        import pandas as pd
        Xtr_df = pd.DataFrame(X_tr, columns=cols) if cols else X_tr
        Xva_df = pd.DataFrame(X_val, columns=cols) if (cols and X_val is not None) else X_val
        clf = lgb.LGBMClassifier(
            objective="binary", n_estimators=2000, learning_rate=0.03, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0, random_state=seed,
            n_jobs=-1, verbose=-1, force_col_wise=True, deterministic=True
        )
        if X_val is not None:
            clf.fit(Xtr_df, y_tr, eval_set=[(Xva_df, y_val)],
                    eval_metric="binary_logloss",
                    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
        else:
            clf.fit(Xtr_df, y_tr, callbacks=[lgb.log_evaluation(-1)])
        return clf
    else:
        clf = HistGradientBoostingClassifier(
            max_depth=None, learning_rate=0.05, max_iter=600, l2_regularization=1.0,
            validation_fraction=0.1, early_stopping=True, random_state=seed
        )
        clf.fit(X_tr, y_tr)
        return clf

def _predict_tree(clf, X, cols: list[str] | None = None):
    if _HAS_LGB and cols:
        import pandas as pd
        X_df = pd.DataFrame(X, columns=cols)
        return clf.predict_proba(X_df)[:,1]
    else:
        return clf.predict_proba(X)[:,1]

# ---- calibrators ----
class PlattCalibrator:
    def __init__(self): self.lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    def fit(self, p, y):
        eps = 1e-6; z = np.log(np.clip(p, eps, 1-eps) / np.clip(1-p, eps, 1-eps)).reshape(-1,1)
        self.lr.fit(z, y); return self
    def transform(self, p):
        eps = 1e-6; z = np.log(np.clip(p, eps, 1-eps) / np.clip(1-p, eps, 1-eps)).reshape(-1,1)
        return self.lr.predict_proba(z)[:,1]

class BetaCalibrator:
    def __init__(self): self.lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    def fit(self, p, y):
        eps = 1e-6
        X = np.c_[np.log(np.clip(p, eps, 1-eps)), np.log(np.clip(1-p, eps, 1-eps))]
        self.lr.fit(X, y); return self
    def transform(self, p):
        eps = 1e-6
        X = np.c_[np.log(np.clip(p, eps, 1-eps)), np.log(np.clip(1-p, eps, 1-eps))]
        return self.lr.predict_proba(X)[:,1]

def pick_best_calibrator(y_cal, p_cal):
    cands = {
        "isotonic": IsotonicRegression(out_of_bounds="clip"),
        "platt": PlattCalibrator(),
        "beta":  BetaCalibrator()
    }
    best_name, best_obj, best_brier = None, None, float("inf")
    fits = {}
    for name, cal in cands.items():
        cal.fit(p_cal, y_cal)
        p_hat = cal.transform(p_cal) if hasattr(cal, "transform") else cal.predict(p_cal)
        if not hasattr(cal, "transform"): p_hat = cal.transform(p_cal)
        br = brier_score_loss(y_cal, p_hat)
        fits[name] = (cal, br)
        if br < best_brier:
            best_name, best_obj, best_brier = name, cal, br
    return best_name, best_obj, fits

def basic_metrics(y, p) -> Dict[str,float]:
    return dict(
        AUROC = roc_auc_score(y, p),
        AUPRC = average_precision_score(y, p),
        Brier = brier_score_loss(y, p),
        ECE   = ece_binary(y, p, bins=15)
    )

def cv_blend_and_calibrate(X_all, y_all, seed=1337, k_folds=5, n_seeds_nn=3, n_seeds_tree=3, hidden=256, epochs=100):
    X_trcal, X_te, y_trcal, y_te = train_test_split(X_all, y_all, test_size=0.20, random_state=seed, stratify=y_all)
    X_tr, X_ca, y_tr, y_ca = train_test_split(X_trcal, y_trcal, test_size=0.25, random_state=seed, stratify=y_trcal)  # 60/20/20
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    p_tr_oof_nn = np.zeros_like(y_tr, dtype=float)
    p_ca_nn_blend = np.zeros_like(y_ca, dtype=float)
    p_te_nn_blend = np.zeros_like(y_te, dtype=float)

    p_tr_oof_tree = np.zeros_like(y_tr, dtype=float)
    p_ca_tree_blend = np.zeros_like(y_ca, dtype=float)
    p_te_tree_blend = np.zeros_like(y_te, dtype=float)

    num_cols = list(range(X_all.shape[1]))  # for LGB fallback path

    for fold, (idx_tr_f, idx_val_f) in enumerate(skf.split(X_tr, y_tr), 1):
        Xf_tr, yf_tr = X_tr[idx_tr_f], y_tr[idx_tr_f]
        Xf_va, yf_va = X_tr[idx_val_f], y_tr[idx_val_f]

        # NN bag
        p_va_accum = np.zeros_like(yf_va, dtype=float)
        p_ca_accum = np.zeros_like(y_ca, dtype=float)
        p_te_accum = np.zeros_like(y_te, dtype=float)
        for s in range(n_seeds_nn):
            seed_s = seed + 1000*fold + s
            model = _train_one_nn(Xf_tr, yf_tr, Xf_va, yf_va, hidden=hidden, epochs=epochs, seed=seed_s)
            _, p_va = _evaluate_raw(model, Xf_va, yf_va, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            _, p_ca = _evaluate_raw(model, X_ca, y_ca, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            _, p_te = _evaluate_raw(model, X_te, y_te, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            p_va_accum += p_va; p_ca_accum += p_ca; p_te_accum += p_te
        p_tr_oof_nn[idx_val_f] = p_va_accum / n_seeds_nn
        p_ca_nn_blend += p_ca_accum / n_seeds_nn
        p_te_nn_blend += p_te_accum / n_seeds_nn

        # Tree bag
        p_va_t = np.zeros_like(yf_va, dtype=float)
        p_ca_t = np.zeros_like(y_ca, dtype=float)
        p_te_t = np.zeros_like(y_te, dtype=float)
        for s in range(n_seeds_tree):
            seed_s = seed + 2000*fold + s
            tree = _train_one_tree(Xf_tr, yf_tr, Xf_va, yf_va, seed=seed_s, cols=None)
            p_va_t += _predict_tree(tree, Xf_va, cols=None)
            p_ca_t += _predict_tree(tree, X_ca, cols=None)
            p_te_t += _predict_tree(tree, X_te, cols=None)
        p_tr_oof_tree[idx_val_f] = p_va_t / n_seeds_tree
        p_ca_tree_blend += p_ca_t / n_seeds_tree
        p_te_tree_blend += p_te_t / n_seeds_tree

    p_ca_nn_blend   /= k_folds
    p_te_nn_blend   /= k_folds
    p_ca_tree_blend /= k_folds
    p_te_tree_blend /= k_folds

    def best_blend_weight(y, p1, p2, grid=101):
        ws = np.linspace(0,1,grid)
        scores = [brier_score_loss(y, w*p1 + (1-w)*p2) for w in ws]
        j = int(np.argmin(scores)); return float(ws[j])

    w_opt = best_blend_weight(y_ca, p_ca_nn_blend, p_ca_tree_blend, grid=101)
    p_ca_blend = w_opt * p_ca_nn_blend + (1 - w_opt) * p_ca_tree_blend
    p_te_blend = w_opt * p_te_nn_blend + (1 - w_opt) * p_te_tree_blend

    raw = dict(
        train_nn= {k: f"{v:.4f}" for k,v in basic_metrics(y_tr, p_tr_oof_nn).items()},
        train_tree={k: f"{v:.4f}" for k,v in basic_metrics(y_tr, p_tr_oof_tree).items()},
        calib={k: f"{v:.4f}" for k,v in basic_metrics(y_ca, p_ca_blend).items()},
        test = {k: f"{v:.4f}" for k,v in basic_metrics(y_te, p_te_blend).items()},
        w_opt = w_opt
    )

    best_name, best_cal, fits = pick_best_calibrator(y_ca, p_ca_blend)
    p_te_cal = best_cal.transform(p_te_blend) if hasattr(best_cal, "transform") else best_cal.transform(p_te_blend)
    cal = dict(best=best_name, fits={k: f"{b:.4f}" for k,(_,b) in fits.items()},
               test={k: f"{v:.4f}" for k,v in basic_metrics(y_te, p_te_cal).items()},
               reliability=reliability_table(y_te, p_te_cal, bins=12))

    return (y_tr, p_tr_oof_nn, p_tr_oof_tree), (y_ca, p_ca_blend), (y_te, p_te_blend, p_te_cal), raw, cal
