import json
from pathlib import Path
import pytest

from halu.orchestrate import run_all, RunConfig

pytestmark = [pytest.mark.hf, pytest.mark.slow]

def test_run_all_qwen25_truthfulqa_cpu(tmp_path):
    out_dir = tmp_path / "run"
    out_dir.mkdir()

    # Tiny run to keep it light but realistic.
    cfg = RunConfig(
        model_id="Qwen/Qwen2.5-0.5B",   # small Qwen 2.5 variant
        dataset="truthfulqa",
        n_examples=40,                   # small slice to keep CI fast
        seed=1337,
        device="cpu",
        dtype="fp32",
        out_dir=str(out_dir),
        # detector training kept short
        epochs=4,
        hidden=64,
        calib_size=0.25,
        val_size=0.20,
        blend_grid=11,
        # reporting
        ece_bins=10,
        rel_bins=10,
        compute_bootstrap=False,
    )

    manifest = run_all(cfg)

    # ---------- Manifest sanity ----------
    assert "config" in manifest
    assert manifest["config"]["dataset"] == "truthfulqa"
    assert "train_summary" in manifest
    assert "report" in manifest
    paths = manifest["paths"]

    # ---------- Files/materialized outputs ----------
    # Features CSV
    f_csv = Path(paths["features_csv"])
    assert f_csv.exists() and f_csv.stat().st_size > 0

    # Artifacts (detector)
    art_dir = Path(paths["artifacts_dir"]) / "detector"
    assert art_dir.exists()
    # core artifacts expected from DetectorEnsemble.save(...)
    for fname in ["config.json", "meta.joblib", "tree.joblib", "calibrator.joblib", "nn.pt"]:
        assert (art_dir / fname).exists()

    # Report directory and plots/CSVs
    rep_dir = Path(paths["report_dir"])
    assert rep_dir.exists()

    rep = manifest["report"]
    assert "detector_core" in rep and "vanilla_core" in rep
    assert "plots" in rep
    for k in [
        "detector_reliability_png",
        "detector_risk_coverage_png",
        "vanilla_reliability_png",
        "vanilla_risk_coverage_png",
    ]:
        p = Path(rep["plots"][k])
        assert p.exists() and p.stat().st_size > 0

    # CSVs saved
    assert Path(rep["abstention_table_path"]).exists()
    assert Path(rep["reliability_bins_path"]).exists()

    # Final manifest file present & parseable
    # tag is dataset__modeltail, where model tail is slug of last path component
    tag = "truthfulqa__Qwen2.5-0.5B"
    man_path = Path(out_dir) / tag / "manifest.json"
    assert man_path.exists()
    with open(man_path, "r", encoding="utf-8") as f:
        json.load(f)