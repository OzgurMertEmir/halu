from __future__ import annotations
import argparse, json
from halu.orchestrate import RunConfig, run_all


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="halu", description="Halu end-to-end orchestrator")
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--dataset", default="truthfulqa")
    p.add_argument("--n-examples", type=int, default=None)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default=None, help='"cuda" | "mps" | "cpu" | None(auto)')
    p.add_argument("--dtype", default="auto", help='"auto" | "fp16" | "bf16" | "fp32"')
    p.add_argument("--out", default="runs/halu_run")

    # detector knobs
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--calib-size", type=float, default=0.20)
    p.add_argument("--val-size", type=float, default=0.20)
    p.add_argument("--blend-grid", type=int, default=101)

    # reporting knobs
    p.add_argument("--ece-bins", type=int, default=15)
    p.add_argument("--rel-bins", type=int, default=12)
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--alpha", type=float, default=0.05)
    return p

def main(argv=None):
    args = _build_parser().parse_args(argv)
    cfg = RunConfig(
        model_id=args.model,
        dataset=args.dataset,
        n_examples=args.n_examples,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        out_dir=args.out,
        epochs=args.epochs,
        hidden=args.hidden,
        calib_size=args.calib_size,
        val_size=args.val_size,
        blend_grid=args.blend_grid,
        ece_bins=args.ece_bins,
        rel_bins=args.rel_bins,
        compute_bootstrap=bool(args.bootstrap),
        n_boot=args.n_boot,
        alpha=args.alpha,
    )
    manifest = run_all(cfg)
    print(json.dumps({
        "ok": True,
        "report_dir": manifest["paths"]["report_dir"],
        "artifacts_dir": manifest["paths"]["artifacts_dir"],
        "features_csv": manifest["paths"]["features_csv"],
    }, indent=2))

if __name__ == "__main__":
    main()