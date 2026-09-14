"""Did fine-tuning the trunk on fish cost us wheat? — the forgetting check.

Two arms, one wheat val loader, one metric function:

* **baseline** — the shipped wheat decoder on the frozen DINOv3 trunk, exactly
  as it is released.
* **tuned** — the same decoder with the CFD-fine-tuned trunk dropped underneath
  it (``load_checkpoint(..., backbone_from=RUN/best.pt)``).

The decoder is byte-identical across the two arms, so any difference is the
trunk's and nothing else. Runs locally on MPS in fp32 (``amp_dtype`` returns
``None`` off CUDA, so autocast is already a no-op there).

``--strict`` is the honesty gate: if the baseline arm does not reproduce the
published wheat numbers to ``--tolerance``, the delta measures the harness, not
forgetting, and the script exits 1 instead of printing a number.

Usage::

    python wheat_forgetting.py --tuned-checkpoint runs/cfd17_unfrozen_s0/best.pt \\
        --decoder weights/decoder_best.pt --data-root data --device mps --strict
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = ("baseline", "tuned")
METRIC_KEYS = ("count_mae", "count_rmse", "count_bias", "precision", "recall", "f1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wheat_forgetting.py",
        description="Wheat val, frozen trunk vs CFD-fine-tuned trunk, same decoder.",
    )
    parser.add_argument("--tuned-checkpoint", type=Path, required=True,
                        help="a CFD run's best.pt/last.pt — its 'backbone' key is the tuned trunk")
    parser.add_argument("--decoder", type=Path, default=Path("weights/decoder_best.pt"),
                        help="the shipped wheat point decoder (default weights/decoder_best.pt)")
    parser.add_argument("--data-root", type=Path, default=Path("data"),
                        help="wheat split root holding train/ and val/ (default data)")
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"),
                        help="directory holding the DINOv3 backbone checkpoint")
    parser.add_argument("--tau", type=float, default=0.35, help="decode threshold (default 0.35)")
    parser.add_argument("--k", type=int, default=3, help="local-max kernel (default 3)")
    parser.add_argument("--nms-radius", type=float, default=1.5, help="cells (default 1.5)")
    parser.add_argument("--match-radius-px", type=float, default=24.0,
                        help="TP match radius in pixels (default 24)")
    parser.add_argument("--output-stride", type=int, default=None,
                        help="default: whatever the decoder checkpoint's config says")
    parser.add_argument("--device", default="mps", help="torch device (default mps)")
    parser.add_argument("--out", type=Path, default=Path("results/wheat_forgetting.json"))
    parser.add_argument("--expect-mae", type=float, default=8.483607,
                        help="published baseline count MAE at tau 0.35")
    parser.add_argument("--expect-f1", type=float, default=0.728848,
                        help="published baseline F1 at tau 0.35")
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="absolute tolerance on the baseline reproduction (default 1e-3)")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 when the baseline arm misses the published numbers")
    return parser


def git_rev(repo: Path) -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is nice-to-have, never fatal
        return "unknown"


def checkpoint_provenance(path: Path) -> dict:
    """The small bookkeeping keys off a checkpoint payload — never the tensors."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    info = {
        "path": str(path),
        "keys": sorted(payload.keys()),
        "epoch": payload.get("epoch"),
        "best_epoch": payload.get("best_epoch"),
        "best_metric": payload.get("best_metric"),
        "run_name": (payload.get("config") or {}).get("run_name"),
        "has_backbone": "backbone" in payload,
    }
    del payload
    return info


def markdown(result: dict) -> str:
    header = "| arm | MAE | RMSE | bias | precision | recall | F1 |"
    lines = [header, "|" + "---|" * 7]
    for arm in ARMS:
        row = result["arms"][arm]
        lines.append(
            f"| {arm} | {row['count_mae']:.4f} | {row['count_rmse']:.4f} | "
            f"{row['count_bias']:+.4f} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} |"
        )
    delta = result["delta"]
    lines.append(
        f"| **delta (tuned - baseline)** | {delta['count_mae']:+.4f} | "
        f"{delta['count_rmse']:+.4f} | {delta['count_bias']:+.4f} | "
        f"{delta['precision']:+.4f} | {delta['recall']:+.4f} | {delta['f1']:+.4f} |"
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    from dataclasses import replace

    import torch

    from cropcounter.metrics import evaluate
    from cropcounter.train import build_loaders, load_checkpoint

    device = torch.device(args.device)
    if device.type == "cuda":
        print("warning: the published wheat reference is fp32; on CUDA the package "
              "autocasts, so a small drift here is precision, not forgetting.")

    print(f"decoder  {args.decoder}")
    print(f"tuned    {args.tuned_checkpoint}")
    baseline_model, base_cfg = load_checkpoint(
        args.decoder, device, weights_dir=args.weights_dir
    )
    tuned_model, _ = load_checkpoint(
        args.decoder, device, weights_dir=args.weights_dir,
        backbone_from=args.tuned_checkpoint,
    )
    baseline_model.eval()
    tuned_model.eval()

    # The wheat val loader exactly as train.py's point validation builds it: the
    # decoder checkpoint's own config, pointed at this data root.
    cfg = replace(base_cfg, data_root=args.data_root, task="point",
                  device=str(device), init_decoder_from=None)
    _, val_loader, _, val_recs = build_loaders(cfg, device)
    output_stride = args.output_stride or cfg.output_stride
    print(f"{len(val_recs)} wheat val images | {cfg.annotation_format} | device {device} "
          f"| tau {args.tau} k {args.k} nms_radius {args.nms_radius} "
          f"stride {output_stride} match_radius {args.match_radius_px}px")

    arms = {}
    for arm, model in (("baseline", baseline_model), ("tuned", tuned_model)):
        summary, _ = evaluate(
            model, val_loader, device, tau=args.tau, k=args.k,
            nms_radius=args.nms_radius, output_stride=output_stride,
            match_radius_px=args.match_radius_px,
            focal_alpha=cfg.focal_alpha, focal_beta=cfg.focal_beta,
            progress=True, desc=f"wheat val ({arm})",
        )
        arms[arm] = {key: float(summary[key]) for key in METRIC_KEYS}
        arms[arm]["n_images"] = int(summary["n_images"])
        arms[arm]["val_loss"] = float(summary["val_loss"])

    mae_off = abs(arms["baseline"]["count_mae"] - args.expect_mae)
    f1_off = abs(arms["baseline"]["f1"] - args.expect_f1)
    reproduced = mae_off <= args.tolerance and f1_off <= args.tolerance

    result = {
        "arms": arms,
        "delta": {key: arms["tuned"][key] - arms["baseline"][key] for key in METRIC_KEYS},
        "baseline_check": {
            "expect_mae": args.expect_mae, "expect_f1": args.expect_f1,
            "mae_abs_error": mae_off, "f1_abs_error": f1_off,
            "tolerance": args.tolerance, "reproduced": reproduced,
        },
        "decode": {
            "tau": args.tau, "k": args.k, "nms_radius": args.nms_radius,
            "output_stride": output_stride, "match_radius_px": args.match_radius_px,
            "autocast": "off (fp32)" if device.type != "cuda" else "on (cuda)",
        },
        "provenance": {
            "decoder": checkpoint_provenance(args.decoder),
            "tuned": checkpoint_provenance(args.tuned_checkpoint),
            "data_root": str(args.data_root),
            "annotation_format": cfg.annotation_format,
            "device": str(device),
            "torch": torch.__version__,
            "git_rev": git_rev(Path(__file__).resolve().parents[3]),
        },
    }

    table = markdown(result)
    result["markdown"] = table
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print()
    print(table)
    print()
    if reproduced:
        print(f"baseline reproduces the published wheat numbers "
              f"(MAE off {mae_off:.2e}, F1 off {f1_off:.2e}) — the delta is real.")
    else:
        print(f"BASELINE DID NOT REPRODUCE: MAE off {mae_off:.2e}, F1 off {f1_off:.2e} "
              f"> tolerance {args.tolerance:.1e}. The delta above measures the harness, "
              f"not forgetting.")
    print(f"wrote {args.out}")
    if args.strict and not reproduced:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
