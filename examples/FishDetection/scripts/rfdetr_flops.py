"""Measure RF-DETR inference GFLOPs and write them into baseline_metrics.json.

Standalone because ``torch.utils.flop_counter.FlopCounterMode`` raises
``AssertionError('Expected gradient function to be set')`` inside RF-DETR's forward
when run under ``torch.no_grad()`` (its deformable-attention path registers a custom
autograd Function). Counting with autograd enabled -- but without calling backward --
sidesteps that. Usage::

    python rfdetr_flops.py --weights <dir with <name>.pth> --metrics <baseline_metrics.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode


def count_gflops(checkpoint: Path, side: int, device: torch.device) -> float:
    from rfdetr import from_checkpoint

    model = from_checkpoint(str(checkpoint))
    core = model.model.model.to(device).eval()  # rfdetr wraps a LWDETR nn.Module
    x = torch.randn(1, 3, side, side, device=device, requires_grad=False)
    with torch.enable_grad(), FlopCounterMode(display=False) as fc:
        core(x)
    return fc.get_total_flops() / 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True, help="directory holding <name>.pth")
    ap.add_argument("--metrics", type=Path, required=True, help="baseline_metrics.json to update")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = json.load(open(args.metrics))
    for name in list(metrics):
        ckpt = args.weights / f"{name}.pth"
        if not ckpt.exists():
            print(f"{name}: no checkpoint at {ckpt}, skipped")
            continue
        side = int(name.split("_")[-1])
        try:
            g = count_gflops(ckpt, side, device)
            metrics[name]["gflops"] = round(g, 1)
            metrics[name]["gflops_input"] = f"{side}x{side}"
            print(f"{name} ({side}x{side}): {g:.1f} GFLOPs")
        except Exception as exc:  # noqa: BLE001 - report, never crash the pipeline
            metrics[name]["gflops_error"] = repr(exc)[:200]
            print(f"{name}: FLOP count failed: {exc!r}")
    json.dump(metrics, open(args.metrics, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
