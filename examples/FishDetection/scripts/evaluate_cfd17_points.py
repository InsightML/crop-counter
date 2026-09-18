"""Score a POINT run on the CFD-17 val split — one scorer, no GPU.

The point-task twin of ``evaluate_cfd17.py``. Pure post-processing: it never
builds a model and never touches CUDA, working entirely from the
``predictions.json`` a point run wrote (see
:func:`cropcounter.metrics.write_point_results`) plus the split's own
``annotations.json``.

The judgement is **point-in-box**: a prediction is a true positive when it
falls inside a ground-truth box, one point per box, confidence-descending with
a nearest-box-centre tiebreak (``point_in_box.match_image``). That is the only
honest way to compare a point head against box ground truth, and it is the
metric the Brackish points memo reports — so this run's number and that one are
like for like.

Two joins matter and both are done here rather than in the trainer:

* **Predictions are keyed by file name.** A ``cfd points`` root renumbers image
  ids to consecutive ints; the bbox root keeps CFD's original strings. The file
  names are shared, and unique within a split, so the name is the only key that
  joins the two without carrying ``cfd_id_map.json`` around.
* **Ground truth comes from the BBOX root**, never the points root — scoring
  against the centres we ourselves derived would be marking our own homework.

Usage::

    python evaluate_cfd17_points.py --data-root data/cfd17 --runs-dir runs \\
        --out results
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import point_in_box as pib  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score point runs on the CFD-17 val split (CPU only).",
    )
    parser.add_argument("--data-root", type=Path, required=True,
                        help="the BBOX subset root — its val/annotations.json is the GT")
    parser.add_argument("--runs-dir", type=Path, default=None,
                        help="score every <runs-dir>/<name>/predictions.json found")
    parser.add_argument("--predictions", nargs="*", action="extend", default=[],
                        help="extra prediction files as name=path")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--taus", default="0.05:0.80:0.05",
                        help="threshold sweep as start:stop:step (inclusive of stop)")
    parser.add_argument("--per-source", action=argparse.BooleanOptionalAction, default=True,
                        help="also break every run down by CFD source dataset")
    return parser


def parse_taus(spec: str) -> List[float]:
    """``"0.05:0.80:0.05"`` -> [0.05, 0.10, ... 0.80], stop included."""
    start, stop, step = (float(part) for part in spec.split(":"))
    if step <= 0:
        raise ValueError(f"--taus step must be positive, got {step}")
    out, value = [], start
    while value <= stop + 1e-9:
        out.append(round(value, 6))
        value += step
    return out


def load_ground_truth(data_root: Path) -> tuple:
    """(gt_by_name, source_of_name) from the bbox root's val split.

    Every val image gets an entry, including the empty frames — they are where
    a detector's false positives show up, so dropping them would flatter it.
    """
    path = Path(data_root) / "val" / "annotations.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    name_of_id = {img["id"]: str(img["file_name"]) for img in document["images"]}
    source_of_name = {
        str(img["file_name"]): str(img.get("dataset", "unknown"))
        for img in document["images"]
    }
    by_id = pib.coco_boxes_by_image(path)
    gt_by_name = {name_of_id[i]: boxes for i, boxes in by_id.items() if i in name_of_id}
    return gt_by_name, source_of_name


def load_predictions(path: Path) -> Dict[str, np.ndarray]:
    """A point run's ``predictions.json`` -> ``{file name: (N, 3) x/y/score}``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "points" not in payload:
        raise ValueError(
            f"{path} carries no 'points' key — is it a BOX run's predictions.json? "
            "Those are scored by evaluate_cfd17.py, not this script."
        )
    return {
        name: np.asarray(points, dtype=float).reshape(-1, 3)
        for name, points in payload["points"].items()
    }


def discover_runs(runs_dir: Optional[Path]) -> List[tuple]:
    """Every ``<runs-dir>/<name>/predictions.json`` that is a POINT run's."""
    if runs_dir is None:
        return []
    found = []
    for path in sorted(Path(runs_dir).glob("*/predictions.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "points" in payload:
            found.append((path.parent.name, path))
    return found


def best_by(sweep: Sequence[Dict], key: str, largest: bool) -> Dict:
    """The sweep entry optimising ``key``."""
    return (max if largest else min)(sweep, key=lambda row: row[key])


def score_one(
    name: str,
    predictions: Dict[str, np.ndarray],
    gt_by_name: Dict[str, np.ndarray],
    source_of_name: Dict[str, str],
    taus: Sequence[float],
    per_source: bool,
) -> Dict[str, Any]:
    """Sweep one run, pick both operating points, and break it down by source."""
    missing = set(gt_by_name) - set(predictions)
    extra = set(predictions) - set(gt_by_name)
    print(f"\n=== {name} ===")
    print(f"  {len(gt_by_name)} GT images | {len(predictions)} predicted | "
          f"{len(missing)} unpredicted | {len(extra)} not in GT")
    if len(missing) == len(gt_by_name):
        print("  SKIP: the name join is empty — wrong data root, or a box run's file")
        return {}

    sweep = pib.sweep_thresholds(predictions, gt_by_name, taus)
    by_f1 = best_by(sweep, "f1", largest=True)
    by_mae = best_by(sweep, "count_mae", largest=False)
    print(f"  best F1  tau={by_f1['conf_thr']:.2f}  F1={by_f1['f1']:.3f} "
          f"P={by_f1['precision']:.3f} R={by_f1['recall']:.3f} "
          f"MAE={by_f1['count_mae']:.2f}")
    print(f"  best MAE tau={by_mae['conf_thr']:.2f}  MAE={by_mae['count_mae']:.2f} "
          f"F1={by_mae['f1']:.3f}")
    if abs(by_f1["conf_thr"] - by_mae["conf_thr"]) > 1e-9:
        print("  note: F1 and count MAE peak at DIFFERENT thresholds — a report "
              "must say which it quotes rather than pick the flattering one")

    result: Dict[str, Any] = {
        "name": name,
        "sweep": sweep,
        "best_f1": by_f1,
        "best_count_mae": by_mae,
        "n_gt_images": len(gt_by_name),
        "n_unpredicted_images": len(missing),
    }

    if per_source:
        groups: Dict[str, List[str]] = defaultdict(list)
        for image_name in gt_by_name:
            groups[source_of_name.get(image_name, "unknown")].append(image_name)
        per: Dict[str, Dict] = {}
        tau = by_f1["conf_thr"]
        for source, names in sorted(groups.items()):
            subset_gt = {n: gt_by_name[n] for n in names}
            subset_pred = {n: predictions[n] for n in names if n in predictions}
            summary, _rows = pib.score_dataset(subset_pred, subset_gt, tau)
            per[source] = summary
            print(f"    {source:<24} n={summary['n_images']:>6} "
                  f"F1={summary['f1']:.3f} MAE={summary['count_mae']:.2f}")
        result["per_source"] = {"tau": tau, "sources": per}
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    taus = parse_taus(args.taus)
    gt_by_name, source_of_name = load_ground_truth(args.data_root)
    print(f"{len(gt_by_name)} val images | "
          f"{sum(len(v) for v in gt_by_name.values())} GT boxes | "
          f"{len(set(source_of_name.values()))} sources")

    targets = discover_runs(args.runs_dir)
    for spec in args.predictions:
        name, _, path = spec.partition("=")
        targets.append((name, Path(path or name)))
    if not targets:
        print("nothing to score: no point-run predictions.json found")
        return 1

    results = [
        score_one(name, load_predictions(path), gt_by_name, source_of_name,
                  taus, args.per_source)
        for name, path in targets
    ]
    results = [r for r in results if r]

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "point_results_summary.json"
    out_path.write_text(json.dumps({"runs": results}, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
