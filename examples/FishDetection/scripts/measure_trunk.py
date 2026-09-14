"""Size, cost and speed of a whole CropCounter, per backbone size.

The point of the tiny/small conversions is a cheaper trunk, and "cheaper" has
to be a measured number rather than a parameter count: a ConvNeXt-Small has
*more* FLOPs per parameter than a Base at the same width, and on a fixed
image size throughput is what decides whether an experiment fits in a budget.
So this reports, for the full frozen-trunk + box-head model in eval mode:

* total and trainable parameters (the decoder is the trainable part),
* GFLOPs for one forward at the requested side, via
  ``torch.utils.flop_counter.FlopCounterMode``,
* images/s and ms/image after warm-up, device-synchronised,
* peak allocated GPU memory (CUDA only — MPS and CPU report null).

``--side`` is ``WIDTHxHEIGHT`` (e.g. the CFD frame shape ``1024x576``); both
must be multiples of 32, which is the trunk's total stride.

Usage::

    python measure_trunk.py --size tiny --device mps --side 1024x576 --iters 50
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from cropcounter.dinov3_pyramid import CropCounter, amp_dtype, autocast_context  # noqa: E402

#: The trunk's total stride; both sides must be a multiple of it.
STRIDE = 32


def parse_side(text: str) -> Tuple[int, int]:
    """``"1024x576"`` -> ``(width, height)``, both checked against the stride."""
    try:
        width, height = (int(part) for part in text.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--side must look like 1024x576, got {text!r}") from exc
    for name, value in (("width", width), ("height", height)):
        if value % STRIDE:
            raise argparse.ArgumentTypeError(f"{name} {value} is not a multiple of {STRIDE}")
    return width, height


def synchronize(device: torch.device) -> None:
    """Block until the device has actually finished — MPS and CUDA both queue."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def count_gflops(model: torch.nn.Module, shape: Tuple[int, ...], device: torch.device) -> Dict[str, object]:
    """One forward's GFLOPs, falling back to CPU if the device upsets the counter.

    ``FlopCounterMode`` works through ``__torch_dispatch__``; on MPS some ops
    dispatch in ways it does not see, so the count is retried on CPU and the
    device it actually ran on is reported rather than quietly assumed.
    """
    for target in (device, torch.device("cpu")):
        try:
            from torch.utils.flop_counter import FlopCounterMode

            probe = model.to(target)
            x = torch.randn(*shape, device=target)
            with torch.no_grad(), FlopCounterMode(display=False) as counter:
                probe(x)
            total = counter.get_total_flops()
            if total > 0:
                return {"gflops": total / 1e9, "flops_device": target.type, "flops_error": None}
            error = "FlopCounterMode counted 0 flops"
        except Exception as exc:  # noqa: BLE001 - fall through to the CPU retry
            error = repr(exc)[:200]
        if target.type == "cpu":
            return {"gflops": None, "flops_device": None, "flops_error": error}
    return {"gflops": None, "flops_device": None, "flops_error": "unreachable"}


def measure(
    size: str,
    device: torch.device,
    side: Tuple[int, int],
    batch: int,
    iters: int,
    warmup: int,
    weights_dir: Optional[Path],
) -> Dict[str, object]:
    """Build the model, count it, and time it."""
    width, height = side
    model = CropCounter(backbone_size=size, weights_dir=weights_dir, task="box").eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.trainable_parameters())
    backbone_params = sum(p.numel() for p in model.backbone.model.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())

    shape = (batch, 3, height, width)
    flops = count_gflops(model, shape, device)

    model = model.to(device)
    x = torch.randn(*shape, device=device)
    dtype = amp_dtype(device)

    with torch.no_grad():
        for _ in range(warmup):
            with autocast_context(device):
                model(x)
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(iters):
            with autocast_context(device):
                model(x)
        synchronize(device)
        elapsed = time.perf_counter() - start

    images = iters * batch
    return {
        "size": size,
        "device": device.type,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "input": {"batch": batch, "width": width, "height": height},
        "autocast_dtype": str(dtype) if dtype is not None else None,
        "params_total": total_params,
        "params_trainable": trainable_params,
        "params_backbone": backbone_params,
        "params_decoder": decoder_params,
        **flops,
        "iters": iters,
        "warmup": warmup,
        "seconds": elapsed,
        "images_per_s": images / elapsed,
        "ms_per_image": 1000.0 * elapsed / images,
        "peak_gpu_mem_mb": (
            torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None
        ),
    }


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="measure_trunk.py",
        description="Params, GFLOPs and throughput of a full CropCounter, per backbone size.",
    )
    parser.add_argument("--size", required=True, choices=("tiny", "small", "base", "large"))
    parser.add_argument("--device", default=None, choices=("cuda", "mps", "cpu"))
    parser.add_argument("--side", type=parse_side, default="1024x576")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--weights-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="also write the JSON here")
    args = parser.parse_args(argv)

    if args.device is None:
        args.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    side = args.side if isinstance(args.side, tuple) else parse_side(args.side)

    record = measure(
        args.size, torch.device(args.device), side, args.batch, args.iters, args.warmup,
        args.weights_dir,
    )
    text = json.dumps(record, indent=1)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
