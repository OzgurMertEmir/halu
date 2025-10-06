import pandas as pd
import numpy as np
import os, json, torch
from joblib import dump
from engine.detector import DetectorEnsemble
from analysis.report import generate_report
from features.build import build_features_df
from analysis.report import ReportInputs

def run_experiment(
    model,
    tokenizer,
    dataset,
    is_train: bool,
    out_dir: str = "runs/exp",
    config: dict | None = None,
    dataset_name: str | None = None
):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Adapt dataset → MCQExamples
    examples = adapt_dataset(dataset, name_hint=dataset_name)

    # 2) Extract features (collapsed, one row per example)
    df_all = extract_features_df(model, tokenizer, examples)

    # 3) If training: split + fit, else: load artifacts
    #    For now, reuse your notebook’s split logic externally and save artifacts
    #    If artifacts exist, we assume evaluation-only.
    has_artifacts = all(os.path.exists(os.path.join(out_dir, p)) for p in
                        ["nn.pt","tree.pkl","w_opt.json","meta.pkl","calibrator.pkl"])

    if is_train or not has_artifacts:
        # ==== TRAIN: build X,y and fit like in your notebook ====
        from sklearn.model_selection import train_test_split
        from joblib import dump

        # y: 1 if vanilla wrong
        y_all = (df_all["chosen"].astype(str).str.upper() != df_all["gold"].astype(str).str.upper()).astype(int).values

        # NOTE: You can keep your exact CV+bagging code here or refactor into a DetectorEnsemble class.
        # For brevity: split into train/calib/test; fit models (NN/Tree); find blend; fit calibrator; save artifacts.
        # Save meta (imputer, scaler, num_cols) too.

        # --- save placeholders to show the flow ---
        with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
            import pickle; pickle.dump({"num_cols":[c for c in df_all.columns if pd.api.types.is_numeric_dtype(df_all[c])],
                                        "imputer":None, "scaler":None}, f)
        with open(os.path.join(out_dir, "w_opt.json"), "w") as f:
            json.dump({"w_opt": 0.5}, f)
        torch.save({}, os.path.join(out_dir, "nn.pt"))
        from joblib import dump
        dump(object(), os.path.join(out_dir, "tree.pkl"))
        dump(object(), os.path.join(out_dir, "calibrator.pkl"))

        # p_err_test should be computed on test slice after calibration
        p_err_test = np.zeros(len(df_all), dtype=float)  # <-- replace with real predictions
        ri = ReportInputs(df_test=df_all, p_err_test=p_err_test)
        summary = generate_report(ri, out_dir=out_dir, prefix=(dataset_name or "dataset"))
        return summary

    else:
        # ==== EVAL: load artifacts and predict calibrated P(error) ====
        import pickle, joblib
        with open(os.path.join(out_dir, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        w_opt = json.load(open(os.path.join(out_dir, "w_opt.json")))["w_opt"]
        # load models
        # (here you would load your real nn_final, tree_final, calibrator)
        nn_final = torch.nn.Sequential()  # placeholder
        tree_final = joblib.load(os.path.join(out_dir, "tree.pkl"))
        calibrator = joblib.load(os.path.join(out_dir, "calibrator.pkl"))

        p_err_test = predict_detector_p_err(
            df=df_all, meta=meta,
            nn_model=nn_final, tree_model=tree_final,
            w_opt=w_opt, calibrator=calibrator
        )
        ri = ReportInputs(df_test=df_all, p_err_test=p_err_test)
        summary = generate_report(ri, out_dir=out_dir, prefix=(dataset_name or "dataset"))
        return summary