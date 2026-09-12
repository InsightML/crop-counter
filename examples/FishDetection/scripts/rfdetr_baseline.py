"""Score a released RF-DETR checkpoint on a CFD val split, through OUR scorer.

The script form of ``3_inference.ipynb`` cells 6-7: download the checkpoint if it
is not already local, run it over every image in the val split, write standard
COCO results, then score them with ``cropcounter.det_metrics.coco_eval`` against
the same ``annotations.json`` our own runs are scored against. One scorer, one
GT file, identical images — that is the whole point of the baseline.

The checkpoint lands at ``<weights-dir>/<name>.pth`` so ``rfdetr_flops.py`` (which
looks for exactly that, and reads the input side off the name's last ``_`` field)
can measure GFLOPs into the same ``baseline_metrics.json`` afterwards.

Usage::

    python rfdetr_baseline.py --checkpoint-url <url> --weights-dir weights/baselines \\
        --images data/cfd17/val/images --annotations data/cfd17/val/annotations.json \\
        --out results/rfdetr_nano_640/predictions.json \\
        --metrics results/baseline_metrics.json --name rfdetr_nano_640 --side 640

``rfdetr`` is an optional, heavy dependency (``pip install rfdetr supervision``);
it is imported inside ``main`` so ``--help`` works in any environment.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rfdetr_baseline.py",
        description="Run a released RF-DETR checkpoint over a CFD val split and score it.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint-url", help="release URL; downloaded into --weights-dir")
    source.add_argument("--checkpoint", type=Path, help="a local .pth to use as-is")
    parser.add_argument("--weights-dir", type=Path, default=Path("weights/baselines"),
                        help="where a downloaded checkpoint is cached, as <name>.pth")
    parser.add_argument("--images", type=Path, required=True, help="val images directory")
    parser.add_argument("--annotations", type=Path, required=True,
                        help="the split's COCO annotations.json (the GT and the image ids)")
    parser.add_argument("--out", type=Path, required=True, help="COCO results json to write")
    parser.add_argument("--metrics", type=Path, default=Path("baseline_metrics.json"),
                        help="metrics json to merge this run's row into, keyed by --name")
    parser.add_argument("--name", required=True,
                        help="row key, e.g. rfdetr_nano_640 (the trailing field is the input side)")
    parser.add_argument("--threshold", type=float, default=0.001,
                        help="score floor; low so the PR curve keeps its tail (default 0.001)")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--side", type=int, default=None,
                        help="input side, recorded in the metrics row for the record only")
    parser.add_argument("--resolution", type=int, default=None,
                        help="force the model's input resolution; omit to let the checkpoint "
                             "declare its own (what the notebook does)")
    parser.add_argument("--limit", type=int, default=None, help="stop after N images (debug)")
    return parser


def resolve_device(prefer: str) -> str:
    import torch

    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_checkpoint(args: argparse.Namespace) -> Path:
    """Return a local checkpoint path, downloading the release asset if needed."""
    if args.checkpoint is not None:
        if not args.checkpoint.exists():
            raise SystemExit(f"no checkpoint at {args.checkpoint}")
        return args.checkpoint
    import urllib.request

    args.weights_dir.mkdir(parents=True, exist_ok=True)
    # <name>.pth is the layout rfdetr_flops.py expects.
    path = args.weights_dir / f"{args.name}.pth"
    if path.exists() and path.stat().st_size > 0:
        print(f"checkpoint present: {path} ({path.stat().st_size / 1e6:.1f} MB)")
        return path
    print(f"downloading {args.checkpoint_url} -> {path}")
    tmp = path.with_suffix(".pth.part")
    urllib.request.urlretrieve(args.checkpoint_url, tmp)
    tmp.replace(path)
    print(f"downloaded {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def load_model(checkpoint: Path, device: str, resolution):
    """``rfdetr.from_checkpoint``, degrading over optional kwargs it may not take.

    The notebook passes the path alone and lets the checkpoint declare its own
    input resolution, which is why nano-640 and medium-1024 both "just work"
    there. ``--resolution`` is an override for the case where a checkpoint does
    not carry it; if this build of ``rfdetr`` rejects the kwarg we fall back to
    the notebook's bare call rather than failing the run.
    """
    from rfdetr import from_checkpoint

    for kwargs in ({"device": device, "resolution": resolution},
                   {"resolution": resolution},
                   {"device": device},
                   {}):
        clean = {k: v for k, v in kwargs.items() if v is not None}
        try:
            model = from_checkpoint(str(checkpoint), **clean)
            print(f"from_checkpoint({checkpoint.name}, {clean}) ok")
            return model
        except TypeError as exc:
            print(f"from_checkpoint rejected {clean}: {exc}")
    raise SystemExit("from_checkpoint refused every argument combination")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        import rfdetr  # noqa: F401 - probed here so the failure message is useful
    except ImportError:
        print("rfdetr is not installed in this environment.\n"
              "    pip install rfdetr supervision\n"
              "(it is an optional baseline-only dependency; the rest of the "
              "benchmark does not need it)", file=sys.stderr)
        return 1

    from PIL import Image
    from tqdm.auto import tqdm

    from cropcounter.det_metrics import DEFAULT_CATEGORY_ID, coco_eval, write_coco_results

    device = resolve_device(args.device)
    checkpoint = ensure_checkpoint(args)

    with args.annotations.open(encoding="utf-8") as fh:
        gt = json.load(fh)
    # CFD image ids are STRINGS. They are taken off the annotations file by
    # basename and copied through with their own type — a cast would break the
    # join back to the GT inside pycocotools.
    id_by_name = {Path(str(im["file_name"])).name: im["id"] for im in gt["images"]}
    categories = gt.get("categories") or []
    category_id = categories[0]["id"] if categories else DEFAULT_CATEGORY_ID

    names = list(id_by_name)
    if args.limit is not None:
        names = names[: args.limit]
    print(f"{len(names)} val images | category_id {category_id} | device {device} "
          f"| threshold {args.threshold}")

    model = load_model(checkpoint, device, args.resolution)

    detections = []
    missing = 0
    started = time.time()
    for name in tqdm(names, desc=args.name):
        path = args.images / name
        if not path.exists():
            missing += 1
            continue
        image_id = id_by_name[name]
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        predicted = model.predict(image, threshold=args.threshold)
        for (x1, y1, x2, y2), score in zip(predicted.xyxy, predicted.confidence):
            detections.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            })
    seconds = time.time() - started
    scored = len(names) - missing
    if missing:
        print(f"warning: {missing} of {len(names)} images were not on disk", file=sys.stderr)
    if not scored:
        raise SystemExit("no images scored — check --images")

    write_coco_results(detections, args.out)
    row = coco_eval(args.annotations, detections)
    row = {key: float(row[key]) for key in ("ap", "ap50", "ap75", "ar100")}
    row.update({
        "seconds_per_image": seconds / scored,
        "n_images": scored,
        "n_detections": len(detections),
        "checkpoint": str(checkpoint),
        "threshold": args.threshold,
        "device": device,
        "side": args.side,
        "predictions": str(args.out),
    })

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics = {}
    if args.metrics.exists():
        with args.metrics.open(encoding="utf-8") as fh:
            metrics = json.load(fh)
    metrics[args.name] = {**metrics.get(args.name, {}), **row}
    with args.metrics.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=1)

    print(f"{args.name}: AP {row['ap']:.4f} AP50 {row['ap50']:.4f} AP75 {row['ap75']:.4f} "
          f"AR100 {row['ar100']:.4f} | {len(detections)} dets over {scored} images "
          f"| {row['seconds_per_image']:.3f} s/image -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
