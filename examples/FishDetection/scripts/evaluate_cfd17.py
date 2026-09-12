"""Score every saved prediction set on the CFD-17 val split — one scorer, no GPU.

Pure post-processing. It never builds a model and never touches CUDA: everything
comes out of the ``predictions.json`` files the runs and the baselines already
wrote, plus the split's own ``annotations.json``.

Per prediction set:

* ``coco_eval`` over the full val split (AP / AP50 / AP75 / AR100).
* a tau sweep via ``sweep_tau_from_detections`` — the detections were decoded
  once at ``ap_tau``, so every operating threshold above it is a mask over a set
  we already have. **Best tau = argmin count MAE**, with the F1-argmax tau
  printed beside it as the sanity check; where the two disagree the report has
  to say so rather than quote whichever flatters (``4_evaluate.ipynb`` §4).
* the same again per CFD source (the ``dataset`` field ``cfd.py`` copies onto
  every image record), so a 17-source average cannot hide a dead source.

AP / AP50 / AP75 / AR100 are deliberately absent from the sweep: they integrate
over the score axis, so no threshold can move them.

Usage::

    python evaluate_cfd17.py --data-root data/cfd17 --runs-dir runs \\
        --baselines-dir results --out results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CSV_COLUMNS = (
    "model", "scope", "n_images", "ap", "ap50", "ap75", "ar100",
    "best_tau_mae", "mae_at_best", "f1_at_best_tau", "best_tau_f1", "f1_max",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_cfd17.py",
        description="Score saved CFD-17 predictions: COCO AP, tau calibration, per source.",
    )
    parser.add_argument("--data-root", type=Path, required=True,
                        help="subset directory holding val/annotations.json")
    parser.add_argument("--predictions", nargs="*", action="extend", default=[],
                        metavar="NAME=PATH", help="explicit prediction sets to score")
    parser.add_argument("--runs-dir", type=Path, default=None,
                        help="auto-discover RUN/predictions.json (-> NAME_best) and "
                             "RUN/predictions_last.json (-> NAME_last)")
    parser.add_argument("--baselines-dir", type=Path, default=None,
                        help="auto-discover */predictions.json as a baseline row")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--taus", default="0.05:0.80:0.05",
                        help="start:stop:step, stop exclusive (default 0.05:0.80:0.05)")
    parser.add_argument("--match-iou", type=float, default=0.5,
                        help="IoU a detection needs to count as a TP (default 0.5)")
    parser.add_argument("--per-source", action=argparse.BooleanOptionalAction, default=True,
                        help="also score each CFD source separately (default on)")
    parser.add_argument("--history", action=argparse.BooleanOptionalAction, default=True,
                        help="pull per-epoch val_ap/val_ap50 out of each run's history.json "
                             "(default on)")
    return parser


def parse_taus(spec: str):
    """``"0.05:0.80:0.05"`` -> [0.05, 0.10, ... 0.75] — stop exclusive, as the notebook."""
    start, stop, step = (float(part) for part in spec.split(":"))
    taus, value = [], start
    while value < stop - 1e-12:
        taus.append(round(value, 4))
        value += step
    return taus


def discover(args) -> "list[tuple[str, Path, Path | None]]":
    """``(name, predictions path, run dir or None)`` for everything to score."""
    found: "list[tuple[str, Path, Path | None]]" = []
    seen = set()

    def add(name, path, run_dir=None):
        if name in seen or not Path(path).exists():
            return
        seen.add(name)
        found.append((name, Path(path), run_dir))

    for entry in args.predictions:
        if "=" not in entry:
            raise SystemExit(f"--predictions wants NAME=PATH, got {entry!r}")
        name, _, path = entry.partition("=")
        if not Path(path).exists():
            raise SystemExit(f"no predictions file at {path}")
        add(name, path)

    if args.runs_dir is not None:
        for run_dir in sorted(p for p in args.runs_dir.iterdir() if p.is_dir()):
            add(f"{run_dir.name}_best", run_dir / "predictions.json", run_dir)
            add(f"{run_dir.name}_last", run_dir / "predictions_last.json", run_dir)

    if args.baselines_dir is not None:
        for sub in sorted(p for p in args.baselines_dir.iterdir() if p.is_dir()):
            add(sub.name, sub / "predictions.json")

    return found


def calibrate(sweep):
    """The notebook's rule: best tau = argmin count MAE; F1-argmax beside it."""
    mae_row = min(sweep, key=lambda r: r["count_mae"])
    f1_row = max(sweep, key=lambda r: r["f1"])
    return {
        "best_tau_mae": float(mae_row["tau"]),
        "mae_at_best": float(mae_row["count_mae"]),
        "f1_at_best_tau": float(mae_row["f1"]),
        "best_tau_f1": float(f1_row["tau"]),
        "f1_max": float(f1_row["f1"]),
        "agree": abs(float(mae_row["tau"]) - float(f1_row["tau"])) < 1e-9,
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    import csv
    from collections import defaultdict

    import numpy as np

    from cropcounter.boxmap import boxes_xywh_to_xyxy
    from cropcounter.det_metrics import coco_eval, read_coco_results, sweep_tau_from_detections

    annotations_path = args.data_root / "val" / "annotations.json"
    if not annotations_path.exists():
        raise SystemExit(f"no val annotations at {annotations_path}")
    with annotations_path.open(encoding="utf-8") as fh:
        gt = json.load(fh)

    images = gt["images"]
    image_ids = [im["id"] for im in images]
    gt_boxes: "dict" = defaultdict(list)
    for ann in gt["annotations"]:
        gt_boxes[ann["image_id"]].append(ann["bbox"])
    # ``dataset`` is the per-image source key cfd._coco_image copies through.
    ids_by_source: "dict" = defaultdict(list)
    for im in images:
        ids_by_source[str(im.get("dataset") or "unknown")].append(im["id"])

    taus = parse_taus(args.taus)
    targets = discover(args)
    if not targets:
        raise SystemExit("nothing to score — pass --predictions, --runs-dir or --baselines-dir")
    print(f"{len(images)} val images | {sum(len(v) for v in gt_boxes.values())} GT boxes | "
          f"{len(ids_by_source)} sources | {len(taus)} taus {taus[0]}..{taus[-1]} | "
          f"{len(targets)} prediction sets")

    scopes = {"full": image_ids}
    if args.per_source:
        scopes.update({name: ids for name, ids in sorted(ids_by_source.items())})

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = {}

    for name, path, run_dir in targets:
        detections = read_coco_results(path)
        by_id: "dict" = defaultdict(list)
        for det in detections:
            by_id[det["image_id"]].append((det["bbox"], det["score"]))

        def triples(ids, by_id=by_id):
            for image_id in ids:
                entries = by_id.get(image_id, ())
                yield (
                    boxes_xywh_to_xyxy(np.array([b for b, _ in entries], dtype=np.float32)),
                    np.asarray([s for _, s in entries], dtype=np.float32),
                    boxes_xywh_to_xyxy(np.array(gt_boxes.get(image_id, []), dtype=np.float32)),
                )

        entry = {"predictions": str(path), "n_detections": len(detections), "scopes": {}}
        for scope, ids in scopes.items():
            stats = coco_eval(
                annotations_path, detections,
                image_ids=None if scope == "full" else ids,
            )
            sweep = sweep_tau_from_detections(triples(ids), taus, match_iou=args.match_iou)
            calibration = calibrate(sweep)
            record = {
                "n_images": len(ids),
                **{key: float(stats[key]) for key in ("ap", "ap50", "ap75", "ar100")},
                **calibration,
                "sweep": sweep,
            }
            entry["scopes"][scope] = record
            rows.append({
                "model": name, "scope": scope, "n_images": len(ids),
                **{key: round(record[key], 5)
                   for key in ("ap", "ap50", "ap75", "ar100", "best_tau_mae", "mae_at_best",
                               "f1_at_best_tau", "best_tau_f1", "f1_max")},
            })

        full = entry["scopes"]["full"]
        if full["agree"]:
            note = f"MAE-argmin and F1-argmax agree at tau {full['best_tau_mae']:.2f}"
        else:
            note = (f"MAE-argmin tau {full['best_tau_mae']:.2f} (F1 {full['f1_at_best_tau']:.4f}) "
                    f"DISAGREES with F1-argmax tau {full['best_tau_f1']:.2f} "
                    f"(F1 {full['f1_max']:.4f}) — compensating errors; report both")
        print(f"  {name}: AP {full['ap']:.4f} AP50 {full['ap50']:.4f} | {note}")

        if args.history and run_dir is not None:
            history_path = run_dir / "history.json"
            if history_path.exists():
                with history_path.open(encoding="utf-8") as fh:
                    history = json.load(fh)
                entry["history"] = {
                    key: history[key] for key in ("val_ap", "val_ap50") if key in history
                }
        summary[name] = entry

    csv_path = args.out / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.out / "results_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump({
            "annotations": str(annotations_path),
            "taus": taus,
            "match_iou": args.match_iou,
            "models": summary,
        }, fh, indent=1)

    header = [c for c in CSV_COLUMNS if c != "scope"]
    print()
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        if row["scope"] != "full":
            continue
        print("| " + " | ".join(str(row[c]) for c in header) + " |")
    print()
    print(f"wrote {csv_path} and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
