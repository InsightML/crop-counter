#!/usr/bin/env python
"""Rescore box-detector predictions as POINTS against box ground truth.

The free first step of the fish point experiment: the box head and the released
RF-DETR-Nano already have COCO predictions on the Brackish val frames, so before
training a point head we can ask what point-in-box F1 those boxes reach when
reduced to their centres. That is the anchor the point run has to beat.

    python examples/FishDetection/scripts/rescore_boxes_as_points.py \
        --gt data/brackish/val/annotations.json \
        --pred /path/predictions.json:boxhead_best \
        --pred /path/rfdetr_nano_640_predictions.json:rfdetr_nano \
        --sweep --out results/free_first_step.json

Scoring is :mod:`point_in_box` — i.e. Liam's verbatim matcher. Nothing here is
random and nothing time-varying is written, so two runs with the same arguments
produce byte-identical JSON (that is the re-runnability check).

The ground truth's own image list defines the evaluation set: an image with no
prediction is scored as zero predictions, which is how the 1,740 empty Brackish
frames get to punish false positives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from point_in_box import (  # noqa: E402
    coco_boxes_by_image,
    coco_results_to_points,
    score_dataset,
    sweep_thresholds,
)

#: A prediction centre this far outside the GT frame means the two geometries
#: are in different coordinate systems (a `fetch --max-side` rescale that did
#: not apply to the GT, say). More than OUTSIDE_FRACTION_STOP of them and the
#: run aborts rather than quietly reporting a meaningless score.
FRAME_SLACK_PX = 1.0
OUTSIDE_FRACTION_STOP = 0.01

TABLE_COLUMNS = [
    ("system", 34, "{:<34}"),
    ("tau", 5, "{:>5.2f}"),
    ("P", 6, "{:>6.4f}"),
    ("R", 6, "{:>6.4f}"),
    ("F1", 6, "{:>6.4f}"),
    ("MAE", 6, "{:>6.3f}"),
    ("RMSE", 6, "{:>6.3f}"),
    ("bias", 7, "{:>7.3f}"),
    ("mean_acc", 8, "{:>8.4f}"),
    ("n_pred", 8, "{:>8d}"),
]


def parse_thresholds(spec: str) -> List[float]:
    """``"0.05:0.95:0.05"`` -> [0.05, 0.10, ... 0.95]; also accepts a comma list.

    The stop is inclusive and every value is rounded to 6dp, so the grid is
    exactly reproducible and free of binary-float dust (``np.arange`` would give
    0.15000000000000002 and a differing JSON between numpy versions).
    """
    if ":" in spec:
        start, stop, step = (float(v) for v in spec.split(":"))
        if step <= 0:
            raise ValueError("--thresholds step must be > 0")
        n = int(round((stop - start) / step)) + 1
        return [round(start + i * step, 6) for i in range(n)]
    return [round(float(v), 6) for v in spec.split(",") if v.strip()]


def parse_pred_arg(spec: str) -> Tuple[Path, str]:
    """``"/path/to/predictions.json:label"`` -> (path, label)."""
    if ":" not in spec:
        raise ValueError(f"--pred needs <file>:<label>, got {spec!r}")
    path, label = spec.rsplit(":", 1)
    if not path or not label:
        raise ValueError(f"--pred needs <file>:<label>, got {spec!r}")
    return Path(path), label


def sha256_of(path: Path) -> str:
    """Stream a hash of a prediction file, to identify it without its local path."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gt(gt_path: Path) -> Tuple[Dict, Dict]:
    """Return (boxes by image id, a summary of the GT document)."""
    boxes = coco_boxes_by_image(gt_path)
    with open(gt_path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    sizes = Counter((img["width"], img["height"]) for img in doc.get("images", []))
    n_boxes = sum(len(b) for b in boxes.values())
    return boxes, {
        "path": str(gt_path),
        "n_images": len(boxes),
        "n_boxes": int(n_boxes),
        "n_empty_images": int(sum(1 for b in boxes.values() if len(b) == 0)),
        "image_sizes": [{"width": w, "height": h, "n": n} for (w, h), n in sorted(sizes.items())],
        "cfd_scale_values": sorted(
            {img.get("cfd_scale") for img in doc.get("images", [])}, key=str
        ),
    }


def join_report(points_by_image: Dict, gt_boxes: Dict, gt_summary: Dict) -> Dict:
    """How well a system's image ids join to the GT, and whether the scales agree.

    Reported, not silently tolerated: a detector evaluated on frames the GT does
    not contain (or vice versa) is the classic way to get a flattering number.
    """
    gt_ids = set(gt_boxes)
    pred_ids = set(points_by_image)
    matched = gt_ids & pred_ids
    n_points = sum(len(v) for v in points_by_image.values())

    widths = [s["width"] for s in gt_summary["image_sizes"]]
    heights = [s["height"] for s in gt_summary["image_sizes"]]
    max_w, max_h = (max(widths) if widths else 0), (max(heights) if heights else 0)
    outside = 0
    max_x = max_y = 0.0
    for image_id in matched:
        arr = points_by_image[image_id]
        if not len(arr):
            continue
        max_x = max(max_x, float(arr[:, 0].max()))
        max_y = max(max_y, float(arr[:, 1].max()))
        outside += int(
            ((arr[:, 0] > max_w + FRAME_SLACK_PX) | (arr[:, 1] > max_h + FRAME_SLACK_PX)).sum()
        )
    n_matched_points = sum(len(points_by_image[i]) for i in matched)
    return {
        "n_pred_images": len(pred_ids),
        "n_gt_images": len(gt_ids),
        "n_gt_images_with_pred": len(matched),
        "gt_coverage": len(matched) / len(gt_ids) if gt_ids else 0.0,
        "n_pred_images_not_in_gt": len(pred_ids - gt_ids),
        "n_gt_images_without_pred": len(gt_ids - pred_ids),
        "n_raw_predictions": int(n_points),
        "max_centre_x": max_x,
        "max_centre_y": max_y,
        "gt_frame": [max_w, max_h],
        "n_centres_outside_gt_frame": outside,
        "fraction_outside_gt_frame": (outside / n_matched_points) if n_matched_points else 0.0,
    }


def best_by(sweep: List[Dict], key: str, largest: bool) -> Dict:
    """Pick one summary from a sweep; ties go to the LOWER threshold (stable)."""
    ordered = sorted(sweep, key=lambda s: (-s[key] if largest else s[key], s["conf_thr"]))
    return ordered[0]


def format_row(label: str, summary: Dict) -> str:
    values = [
        label[:34],
        summary["conf_thr"],
        summary["precision"],
        summary["recall"],
        summary["f1"],
        summary["count_mae"],
        summary["count_rmse"],
        summary["count_bias"],
        summary["mean_accuracy"],
        summary["n_pred"],
    ]
    return "  ".join(fmt.format(v) for (_, _, fmt), v in zip(TABLE_COLUMNS, values))


def header_row() -> str:
    head = "  ".join(f"{name:<{width}}" if i == 0 else f"{name:>{width}}"
                     for i, (name, width, _) in enumerate(TABLE_COLUMNS))
    return head + "\n" + "-" * len(head)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt", required=True, type=Path,
                    help="COCO annotations.json holding the GT boxes (defines the image set)")
    ap.add_argument("--pred", required=True, action="append", metavar="FILE:LABEL",
                    help="COCO results json and the name to report it under; repeatable")
    ap.add_argument("--sweep", action="store_true",
                    help="score the whole --thresholds grid and pick the best-F1 operating "
                         "point (default: score only --conf-thr)")
    ap.add_argument("--thresholds", default="0.05:0.95:0.05",
                    help="start:stop:step (stop inclusive) or a comma list; with --sweep")
    ap.add_argument("--conf-thr", type=float, default=0.0,
                    help="the single threshold scored when --sweep is not given")
    ap.add_argument("--out", required=True, type=Path, help="where to write the results JSON")
    ap.add_argument("--no-rows", action="store_true",
                    help="omit the per-image rows (keeps the committed copy small; the full "
                         "file lives on Drive)")
    args = ap.parse_args(argv)

    thresholds = parse_thresholds(args.thresholds) if args.sweep else [round(args.conf_thr, 6)]
    systems = [parse_pred_arg(spec) for spec in args.pred]

    gt_boxes, gt_summary = load_gt(args.gt)
    print(f"GT {args.gt}: {gt_summary['n_images']} images, {gt_summary['n_boxes']} boxes, "
          f"{gt_summary['n_empty_images']} empty images, "
          f"sizes {[(s['width'], s['height'], s['n']) for s in gt_summary['image_sizes']]}")
    print(f"thresholds: {thresholds[0]} .. {thresholds[-1]} ({len(thresholds)} values)\n")

    payload = {
        "gt": gt_summary,
        "thresholds": thresholds,
        "scorer": "examples/FishDetection/scripts/point_in_box.py — match_image copied "
                  "verbatim from examples/WheatHead/notebooks/4_evaluate.ipynb",
        "rows_threshold": "best_f1",
        "systems": [],
    }

    for pred_path, label in systems:
        with open(pred_path, "r", encoding="utf-8") as fh:
            results = json.load(fh)
        points = coco_results_to_points(results)
        join = join_report(points, gt_boxes, gt_summary)
        print(f"{label}: {join['n_raw_predictions']} predictions over "
              f"{join['n_pred_images']} images; joins {join['n_gt_images_with_pred']}/"
              f"{join['n_gt_images']} GT images ({join['gt_coverage']:.1%}); "
              f"max centre ({join['max_centre_x']:.1f}, {join['max_centre_y']:.1f}) "
              f"vs GT frame {join['gt_frame']}")
        if join["fraction_outside_gt_frame"] > OUTSIDE_FRACTION_STOP:
            print(
                f"STOP: {join['fraction_outside_gt_frame']:.2%} of {label}'s centres lie outside "
                f"the GT frame {join['gt_frame']} — the prediction and GT geometries are in "
                "different scales, so a point-in-box score would be meaningless.",
                file=sys.stderr,
            )
            return 2

        sweep = sweep_thresholds(points, gt_boxes, thresholds)
        best_f1 = best_by(sweep, "f1", largest=True)
        best_mae = best_by(sweep, "count_mae", largest=False)
        _, rows = score_dataset(points, gt_boxes, best_f1["conf_thr"])
        payload["systems"].append({
            "label": label,
            "pred_file": pred_path.name,
            "pred_sha256": sha256_of(pred_path),
            "join": join,
            "best_f1": best_f1,
            "best_count_mae": best_mae,
            "sweep": sweep,
            "rows": [] if args.no_rows else rows,
        })

    print("\nbest F1 operating point")
    print(header_row())
    for sysrec in payload["systems"]:
        print(format_row(sysrec["label"], sysrec["best_f1"]))

    print("\nbest count-MAE operating point (counting, not detection)")
    print(header_row())
    for sysrec in payload["systems"]:
        print(format_row(sysrec["label"], sysrec["best_count_mae"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        # Compact and key-ordered: the file is a re-runnability check, so its
        # bytes must depend only on the numbers.
        json.dump(payload, fh, separators=(",", ":"), sort_keys=True)
        fh.write("\n")
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
