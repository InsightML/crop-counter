"""Trunk-size report: what each backbone size costs and what it buys, on CFD-17 val.

CPU-only post-processing. It builds no model, touches no GPU and fetches
nothing: every number comes off disk — the run dirs' ``predictions.json`` and
``history.json``, the baselines' ``predictions.json`` + ``baseline_metrics.json``,
``trunk_<size>.json`` from ``measure_trunk.py``, and the calibrated taus in
``results_summary.json`` from ``evaluate_cfd17.py``. **Anything not on disk
prints as an em dash.** There is no default, no fallback and no estimate; a
report that invents a FLOP count is worse than one with a hole in it.

What it produces in ``--out-dir``:

* ``size_report.md`` — the whole thing as markdown, figures by relative path.
* ``size_report.html`` (or ``--html PATH``) — the same sections as ONE
  self-contained page: figures inlined as base64, no external reference of any
  kind, readable light and dark, safe at phone width.
* ``figures/*.png`` — Agg-rendered, 130 dpi.
* ``size_report.json`` — every number the prose quotes, machine-readable.

The statistics are scoped honestly:

* **Image-level bootstrap CIs** on AP and AP50. ``COCOeval.evaluate()`` runs
  once per system; each of the ``--bootstrap`` replicates only re-indexes that
  output (``size_report_lib.resample_evalimgs``) and re-accumulates. Replicates
  are derived from ``--seed`` alone, so every system sees the *same* resampled
  image sets and any ``--workers`` gives the same answer.
* **Paired tests across trunk sizes** on per-source AP (one pair per CFD source)
  and on per-image matched F1 at each system's own calibrated tau. Median paired
  difference and sources-won are reported beside the p-value, never instead of
  the numbers.
* The CIs are **evaluation** variance only. One seed per size means training
  variance is unmeasured, and the report says so in every rendering.

Usage::

    python size_report.py --data-root data/cfd17 --runs-dir runs \\
        --baselines-dir results --results-dir results --out-dir results/size_report \\
        --bootstrap 1000 --seed 0 --workers 16

    # smoke, on the 168-image subset the repo ships
    python size_report.py --data-root data/cfd17_smoke --runs-dir runs \\
        --baselines-dir results_smoke --results-dir results_smoke \\
        --out-dir /tmp/size_report_smoke --bootstrap 50 --workers 4
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
for _path in (str(_REPO_ROOT / "src"), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import size_report_lib as lib  # noqa: E402

#: Trunk series colours, smallest trunk first; baselines get their own family so
#: "ours" and "theirs" never share a hue on any figure.
OURS_COLOURS = ("#1f6f6b", "#2e8f89", "#57b0a8", "#89c9c1")
BASELINE_COLOURS = ("#b4532a", "#d98b4a", "#8a5a00")
#: Per-RUN colours for the training-curve panels, where the lines are separate
#: runs rather than one "ours" series and must be told apart at a glance.
RUN_COLOURS = ("#1f6f6b", "#b4532a", "#3d5a98", "#8a5a00", "#7b4173", "#4a7c3f")
FIG_DPI = 130


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="size_report.py",
        description="Trunk size vs cost vs accuracy on CFD-17 val — markdown, figures, HTML, JSON.",
    )
    parser.add_argument("--data-root", type=Path, required=True,
                        help="subset directory holding val/annotations.json")
    parser.add_argument("--runs-dir", type=Path, default=None,
                        help="directory of run dirs (config.json + predictions.json + history.json)")
    parser.add_argument("--baselines-dir", type=Path, default=None,
                        help="directory of baseline dirs (NAME/predictions.json)")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="where baseline_metrics.json, trunk_<size>.json and "
                             "results_summary.json live (default: --baselines-dir)")
    parser.add_argument("--out-dir", type=Path, required=True, help="output directory")
    parser.add_argument("--systems", nargs="*", action="extend", default=[],
                        metavar="NAME=LABEL[:trunk_size]",
                        help="explicit system list and labels; default is every run dir with "
                             "predictions.json plus every baseline dir")
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="image-level bootstrap resamples per system (default 1000)")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed (default 0)")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                        help="processes for the bootstrap (default min(8, cpu count))")
    parser.add_argument("--html", type=Path, default=None,
                        help="self-contained HTML path (default <out-dir>/size_report.html)")
    parser.add_argument("--taus-from", type=Path, default=None,
                        help="results_summary.json holding the calibrated taus and tau sweeps "
                             "(default <results-dir>/results_summary.json)")
    parser.add_argument("--match-iou", type=float, default=0.5,
                        help="IoU gate for the per-image matched F1 (default 0.5)")
    parser.add_argument("--flops-side", default="1024x576",
                        help="label for our GFLOPs column's input size (default 1024x576)")
    return parser


def parse_system_spec(spec: str) -> Tuple[str, str, Optional[str]]:
    """``NAME=LABEL[:trunk_size]`` -> ``(name, label, trunk or None)``.

    The trunk suffix is only taken when it is one of the sizes the repo builds,
    so a label may contain a colon ("Frozen base: 8 epochs") without losing it.
    """
    if "=" not in spec:
        raise SystemExit(f"--systems wants NAME=LABEL[:trunk_size], got {spec!r}")
    name, _, rest = spec.partition("=")
    trunk = None
    if ":" in rest:
        head, _, tail = rest.rpartition(":")
        if tail.strip() in lib.TRUNK_ORDER:
            rest, trunk = head, tail.strip()
    name, label = name.strip(), rest.strip()
    if not name:
        raise SystemExit(f"--systems entry has an empty NAME: {spec!r}")
    return name, label or name, trunk


def load_json(path: Optional[Path]) -> Optional[Any]:
    """Read a JSON file, or ``None`` when it is not there. Never raises on absence."""
    if path is None or not Path(path).exists():
        return None
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- #
# system discovery
# --------------------------------------------------------------------------- #


def discover_systems(args) -> List[Dict[str, Any]]:
    """Every scoreable prediction set, ours first then the baselines.

    A run dir contributes its ``predictions.json`` (the best-AP50 checkpoint);
    its trunk size comes from ``config.json``'s ``backbone`` field, never
    guessed from the run name. A baseline dir contributes its
    ``predictions.json`` and carries no trunk size.
    """
    overrides = OrderedDict()
    for spec in args.systems:
        name, label, trunk = parse_system_spec(spec)
        overrides[name] = (label, trunk)

    found: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    if args.runs_dir is not None and Path(args.runs_dir).is_dir():
        for run_dir in sorted(p for p in Path(args.runs_dir).iterdir() if p.is_dir()):
            predictions = run_dir / "predictions.json"
            if not predictions.exists():
                continue
            config = load_json(run_dir / "config.json") or {}
            found[run_dir.name] = {
                "name": run_dir.name,
                "label": run_dir.name,
                "kind": "ours",
                "trunk": config.get("backbone"),
                "backbone_trainable": config.get("backbone_trainable"),
                "predictions": predictions,
                "predictions_last": (run_dir / "predictions_last.json"
                                     if (run_dir / "predictions_last.json").exists() else None),
                "run_dir": run_dir,
                "summary_key": f"{run_dir.name}_best",
                "summary_key_last": f"{run_dir.name}_last",
                "config": config,
            }

    if args.baselines_dir is not None and Path(args.baselines_dir).is_dir():
        for sub in sorted(p for p in Path(args.baselines_dir).iterdir() if p.is_dir()):
            predictions = sub / "predictions.json"
            if not predictions.exists() or sub.name in found:
                continue
            found[sub.name] = {
                "name": sub.name,
                "label": sub.name,
                "kind": "baseline",
                "trunk": None,
                "backbone_trainable": None,
                "predictions": predictions,
                "predictions_last": None,
                "run_dir": None,
                "summary_key": sub.name,
                "summary_key_last": None,
                "config": {},
            }

    if overrides:
        missing = [n for n in overrides if n not in found]
        if missing:
            raise SystemExit(
                f"--systems names not found under --runs-dir/--baselines-dir: {', '.join(missing)}"
            )
        systems = []
        for name, (label, trunk) in overrides.items():
            system = found[name]
            system["label"] = label
            if trunk is not None:
                system["trunk"] = trunk
            systems.append(system)
        return systems

    ours = [s for s in found.values() if s["kind"] == "ours"]
    ours.sort(key=lambda s: (lib.TRUNK_ORDER.index(s["trunk"])
                             if s["trunk"] in lib.TRUNK_ORDER else len(lib.TRUNK_ORDER),
                             s["name"]))
    baselines = [s for s in found.values() if s["kind"] == "baseline"]
    return ours + baselines


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def load_ground_truth(data_root: Path):
    """``(COCO object, raw dict, image ids in COCOeval order, source -> positions)``."""
    from pycocotools.coco import COCO

    path = Path(data_root) / "val" / "annotations.json"
    if not path.exists():
        raise SystemExit(f"no val annotations at {path}")
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(path))
    return coco, raw, path


def evaluate_system(coco_gt, detections: Sequence[dict], image_ids: Sequence[Any]):
    """Run ``COCOeval`` once and hand back everything downstream needs.

    Returns ``(evaluator, stats, precision)``. ``evaluate()`` is the expensive
    half and is never repeated: the bootstrap and the per-source AP both reuse
    ``evaluator.evalImgs``.

    A ``-1`` in ``COCOeval.stats`` is pycocotools' "no eligible ground truth in
    this band" sentinel (typically ``AP small`` on a split with no small boxes).
    It comes back as ``None`` so the report prints an em dash there rather than
    a zero — a band that was never scored is not a band the model failed.
    """
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes([dict(d) for d in detections])
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.params.imgIds = list(image_ids)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = [None if float(v) < 0 else float(v) for v in evaluator.stats]
    precision = np.array(evaluator.eval["precision"], dtype=float)
    return evaluator, stats, precision


def top_k_per_image(detections: Sequence[dict], k: int = 100) -> Dict[Any, List[dict]]:
    """Group detections by image and keep the top ``k`` by score — COCOeval's own cap."""
    by_image: Dict[Any, List[dict]] = defaultdict(list)
    for det in detections:
        by_image[det["image_id"]].append(det)
    for dets in by_image.values():
        if len(dets) > k:
            dets.sort(key=lambda d: d["score"], reverse=True)
            del dets[k:]
    return by_image


def per_image_f1(
    by_image: Dict[Any, List[dict]],
    gt_boxes: Dict[Any, List[Sequence[float]]],
    image_ids: Sequence[Any],
    tau: float,
    match_iou: float,
) -> np.ndarray:
    """Matched F1 per image at one operating threshold, in ``image_ids`` order.

    ``2*tp / (2*tp + fp + fn)``. An image with no GT and no surviving detection
    scores 1.0 — the two systems agreed perfectly on an empty frame — which
    makes the paired difference exactly zero there, so those frames neither
    help nor hurt either side in the signed-rank test.
    """
    from cropcounter.boxmap import boxes_xywh_to_xyxy
    from cropcounter.det_metrics import match_boxes_iou

    out = np.zeros(len(image_ids), dtype=float)
    for position, image_id in enumerate(image_ids):
        entries = by_image.get(image_id, ())
        kept = [d["bbox"] for d in entries if d["score"] > tau]
        pred = boxes_xywh_to_xyxy(np.array(kept, dtype=np.float32))
        truth = boxes_xywh_to_xyxy(np.array(gt_boxes.get(image_id, []), dtype=np.float32))
        tp, fp, fn = match_boxes_iou(pred, truth, iou_thr=match_iou)
        denominator = 2 * tp + fp + fn
        out[position] = 1.0 if denominator == 0 else 2.0 * tp / denominator
    return out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#c8c8c8",
        "axes.labelcolor": "#22242a",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.8,
        "text.color": "#22242a",
        "xtick.color": "#55585f",
        "ytick.color": "#55585f",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "font.size": 10,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def _colour(system: Dict[str, Any], ours_seen: List[str], base_seen: List[str]) -> str:
    if system["kind"] == "ours":
        if system["name"] not in ours_seen:
            ours_seen.append(system["name"])
        return OURS_COLOURS[ours_seen.index(system["name"]) % len(OURS_COLOURS)]
    if system["name"] not in base_seen:
        base_seen.append(system["name"])
    return BASELINE_COLOURS[base_seen.index(system["name"]) % len(BASELINE_COLOURS)]


def _empty_panel(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10, color="#8a5a00",
            transform=ax.transAxes, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def figure_pareto(systems, metric: str, out_path: Path, flops_side: str) -> Optional[Path]:
    """AP (or AP50) against inference GFLOPs, log x — the decision plot.

    Ours is one series ordered by trunk size and joined by a line; the RF-DETR
    baselines are a second series with their own marker. Error bars are the
    image-level bootstrap CI, so the eye is not invited to read a gap that the
    resampling says is noise.
    """
    _style()
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    label = "AP" if metric == "ap" else "AP50"

    annotations = []
    plotted = 0
    for kind, marker, line in (("ours", "o", "-"), ("baseline", "s", "")):
        group = [s for s in systems
                 if s["kind"] == kind
                 and not lib.is_missing(s.get("gflops"))
                 and not lib.is_missing(s.get(metric))]
        group.sort(key=lambda s: s["gflops"])
        if not group:
            continue
        plotted += len(group)
        xs = [s["gflops"] for s in group]
        ys = [s[metric] for s in group]
        colour = OURS_COLOURS[0] if kind == "ours" else BASELINE_COLOURS[0]
        lows, highs = [], []
        for system, y in zip(group, ys):
            ci = (system.get("bootstrap") or {}).get(f"{metric}_ci") or [None, None]
            lows.append(0.0 if lib.is_missing(ci[0]) else max(0.0, y - ci[0]))
            highs.append(0.0 if lib.is_missing(ci[1]) else max(0.0, ci[1] - y))
        ax.errorbar(xs, ys, yerr=[lows, highs], fmt=marker + line, color=colour,
                    markersize=8, linewidth=1.6, capsize=3, elinewidth=1.1,
                    label="ours (frozen trunk + box head)" if kind == "ours"
                    else "RF-DETR (released)",
                    zorder=3)
        for system, x, y in zip(group, xs, ys):
            annotations.append((x, y, f"{system.get('trunk') or system['label']}\n{y:.3f}", colour))

    if not plotted:
        _empty_panel(ax, "no system has both a GFLOPs measurement and an "
                         f"{label} on disk — run measure_trunk.py / rfdetr_flops.py")
        fig.savefig(out_path, dpi=FIG_DPI)
        plt.close(fig)
        return out_path

    ax.set_xscale("log")
    ax.margins(x=0.28, y=0.18)
    # Annotate only once the limits are final, so a point near the right edge
    # gets its label on the inside rather than off the canvas.
    low, high = ax.get_xlim()
    midpoint = math.sqrt(low * high)
    for x, y, text, colour in annotations:
        right_half = x > midpoint
        ax.annotate(text, (x, y), textcoords="offset points",
                    xytext=(-10 if right_half else 10, -2), fontsize=9, color=colour,
                    ha="right" if right_half else "left", va="center")
    ax.set_xlabel(f"inference GFLOPs per frame (log scale) — ours at {flops_side}, "
                  "RF-DETR at its native side")
    ax.set_ylabel(label)
    ax.set_title(f"{label} against inference cost — CFD-17 val, one scorer")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=2)
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    return out_path


def figure_per_source(systems, sources: Sequence[str], out_path: Path) -> Path:
    """Per-source AP as a dot plot — 17 rows, one marker per system."""
    _style()
    height = max(4.0, 0.34 * max(1, len(sources)) + 1.6)
    fig, ax = plt.subplots(figsize=(8.6, height))
    ours_seen, base_seen = [], []
    def mean_ap(source: str) -> float:
        values = [v for v in ((s.get("per_source") or {}).get(source) for s in systems)
                  if not lib.is_missing(v)]
        return float(np.mean(values)) if values else -1.0

    order = sorted(sources, key=lambda s: -mean_ap(s))
    positions = {source: i for i, source in enumerate(order)}
    for index in range(len(order)):
        ax.axhline(index, color="#eeeeee", linewidth=6, zorder=0)
    for system in systems:
        colour = _colour(system, ours_seen, base_seen)
        marker = "o" if system["kind"] == "ours" else "s"
        xs, ys = [], []
        for source in order:
            value = (system.get("per_source") or {}).get(source)
            if lib.is_missing(value):
                continue
            xs.append(value)
            ys.append(positions[source])
        ax.scatter(xs, ys, s=44, marker=marker, color=colour, label=system["label"],
                   zorder=3, edgecolor="white", linewidth=0.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.invert_yaxis()
    ax.set_xlabel("AP (COCO, IoU .50:.95) on that source's val frames")
    ax.set_title("Per-source AP — a 17-source mean cannot hide a dead source")
    ax.legend(loc="lower right", ncols=1)
    ax.grid(axis="y", visible=False)
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    return out_path


def figure_per_size(systems, out_path: Path) -> Path:
    """COCO AP by object area — small (<32²), medium, large."""
    _style()
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    groups = ("small", "medium", "large")
    ours_seen, base_seen = [], []
    usable = [s for s in systems if s.get("per_size")]
    if not usable:
        _empty_panel(ax, "no per-size AP available")
    else:
        width = 0.8 / len(usable)
        base = np.arange(len(groups))
        for index, system in enumerate(usable):
            colour = _colour(system, ours_seen, base_seen)
            values = [system["per_size"].get(g, float("nan")) for g in groups]
            offsets = base - 0.4 + width * (index + 0.5)
            ax.bar(offsets, values, width=width * 0.92, color=colour, label=system["label"])
            for x, value in zip(offsets, values):
                if not lib.is_missing(value):
                    ax.annotate(f"{value:.3f}", (x, value), ha="center", va="bottom",
                                fontsize=8, color="#55585f", xytext=(0, 2),
                                textcoords="offset points")
        ax.set_xticks(base)
        ax.set_xticklabels([f"AP {g}" for g in groups])
        ax.set_ylabel("AP")
        ax.set_title("AP by object size — COCOeval's own area bands")
        ax.legend(loc="upper left")
        ax.grid(axis="x", visible=False)
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    return out_path


def figure_bootstrap(systems, out_path: Path) -> Path:
    """Point estimate + 95 % image-level bootstrap CI, AP and AP50."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, max(3.2, 0.5 * len(systems) + 1.8)),
                             sharey=True)
    ours_seen, base_seen = [], []
    labels = [s["label"] for s in systems]
    for ax, metric, title in zip(axes, ("ap", "ap50"), ("AP", "AP50")):
        any_ci = False
        for index, system in enumerate(systems):
            colour = _colour(system, ours_seen, base_seen)
            point = system.get(metric)
            ci = (system.get("bootstrap") or {}).get(f"{metric}_ci") or [None, None]
            if lib.is_missing(point):
                continue
            if lib.is_missing(ci[0]) or lib.is_missing(ci[1]):
                ax.scatter([point], [index], color=colour, s=46, zorder=3)
                continue
            any_ci = True
            ax.plot([ci[0], ci[1]], [index, index], color=colour, linewidth=2.4, zorder=2)
            ax.scatter([point], [index], color=colour, s=46, zorder=3,
                       edgecolor="white", linewidth=0.6)
        ax.set_title(f"{title} with 95 % image-level bootstrap CI" if any_ci else title)
        ax.set_xlabel(title)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels)
    axes[0].set_ylim(-0.7, len(labels) - 0.3)
    axes[0].invert_yaxis()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    return out_path


def figure_training_curves(runs: Sequence[Dict[str, Any]], out_path: Path) -> Path:
    """One figure, four panels, one line per run — loss, AP50, F1, LR."""
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.6))
    panels = (
        ("loss", "train / val loss", ("train_loss", "val_loss")),
        ("val_ap50", "val AP50", ("val_ap50",)),
        ("val_f1", "val F1 (at the run's config tau)", ("val_f1",)),
        ("lr", "learning rate", ("lr",)),
    )
    drawn = False
    for ax, (_, title, keys) in zip(axes.ravel(), panels):
        for index, run in enumerate(runs):
            history = run.get("history") or {}
            colour = RUN_COLOURS[index % len(RUN_COLOURS)]
            for key, style in zip(keys, ("-", "--")):
                values = history.get(key)
                if not values:
                    continue
                drawn = True
                epochs = np.arange(1, len(values) + 1)
                suffix = "" if len(keys) == 1 else f" {key.split('_')[0]}"
                # A marker, so a one-epoch run is a visible dot rather than a
                # zero-length line that silently disappears from the panel.
                ax.plot(epochs, values, style, color=colour, linewidth=1.7,
                        marker="o", markersize=3.4,
                        label=f"{run['label']}{suffix}")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        if title == "learning rate":
            ax.set_yscale("log")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=8)
    if not drawn:
        for ax in axes.ravel():
            _empty_panel(ax, "no history.json found in any run dir")
    fig.suptitle("Training curves, one line per run", y=0.99, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    return out_path


def figure_pr_curves(systems, recall_thresholds: np.ndarray, out_path: Path) -> Path:
    """Precision against recall at IoU 0.50, straight off ``COCOeval``'s array."""
    _style()
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ours_seen, base_seen = [], []
    drawn = False
    for system in systems:
        curve = system.get("pr_curve")
        if curve is None:
            continue
        curve = np.asarray(curve, dtype=float)
        mask = curve > -1
        if not mask.any():
            continue
        drawn = True
        colour = _colour(system, ours_seen, base_seen)
        style = "-" if system["kind"] == "ours" else "--"
        ax.plot(recall_thresholds[mask], curve[mask], style, color=colour,
                linewidth=1.9, label=system["label"])
    if not drawn:
        _empty_panel(ax, "no PR curve available")
    else:
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title("Precision–recall at IoU 0.50, all images, maxDets 100")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=2)
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    return out_path


def figure_tau_calibration(systems, out_path: Path) -> Path:
    """The notebook's twin-axis panel per system: count MAE (left) vs F1 (right).

    Same construction as ``4_evaluate.ipynb`` § 4 — red MAE on the left axis,
    blue F1 on the right, a dashed line at the MAE-argmin tau the headline
    quotes, and the F1-argmax tau marked when the two disagree.
    """
    _style()
    usable = [s for s in systems if s.get("sweep")]
    if not usable:
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        _empty_panel(ax, "no tau sweeps in results_summary.json (--taus-from)")
        fig.savefig(out_path, dpi=FIG_DPI)
        plt.close(fig)
        return out_path

    columns = min(2, len(usable))
    rows = math.ceil(len(usable) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(8.2 * columns, 4.3 * rows), squeeze=False)
    for ax, system in zip(axes.ravel(), usable):
        sweep = system["sweep"]
        taus = [row["tau"] for row in sweep]
        ax.plot(taus, [row["count_mae"] for row in sweep], "o-", color="#c0392b", markersize=4)
        ax.set_xlabel("tau")
        ax.set_ylabel("count MAE", color="#c0392b")
        ax.tick_params(axis="y", labelcolor="#c0392b")
        twin = ax.twinx()
        twin.plot(taus, [row["f1"] for row in sweep], "o-", color="#1f5fa8", markersize=4)
        twin.set_ylabel("F1 @ IoU 0.5", color="#1f5fa8")
        twin.tick_params(axis="y", labelcolor="#1f5fa8")
        twin.grid(False)
        tau = system.get("tau")
        if not lib.is_missing(tau):
            ax.axvline(tau, color="#555555", linestyle="--", linewidth=1.1)
        tau_f1 = system.get("tau_f1")
        if not lib.is_missing(tau_f1) and not lib.is_missing(tau) and abs(tau_f1 - tau) > 1e-9:
            ax.axvline(tau_f1, color="#1f5fa8", linestyle=":", linewidth=1.1)
        agree = "" if lib.is_missing(tau_f1) or lib.is_missing(tau) or abs(tau_f1 - tau) > 1e-9 \
            else " (F1-argmax agrees)"
        ax.set_title(f"{system['label']} — MAE-argmin tau {lib.fmt(tau, '{:.2f}')}{agree}")
    for ax in axes.ravel()[len(usable):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# report assembly
# --------------------------------------------------------------------------- #


def _ratio(numerator: Any, denominator: Any) -> Optional[float]:
    if lib.is_missing(numerator) or lib.is_missing(denominator) or not denominator:
        return None
    return float(numerator) / float(denominator)


def _delta(a: Any, b: Any) -> Optional[float]:
    if lib.is_missing(a) or lib.is_missing(b):
        return None
    return float(a) - float(b)


def recommendation_blocks(systems, flops_side: str) -> List[dict]:
    """The trade-off, with every number filled in and the conclusion left open.

    Deliberately a template: which side of the trade is worth taking is a
    business call, not something a report generator gets to decide.
    """
    ours = [s for s in systems if s["kind"] == "ours"]
    baselines = [s for s in systems if s["kind"] == "baseline"]
    reference = None
    for candidate in reversed(lib.TRUNK_ORDER):
        match = [s for s in ours if s.get("trunk") == candidate]
        if match:
            reference = match[0]
            break
    if reference is None and ours:
        reference = ours[-1]
    nano = next((s for s in baselines if "nano" in s["name"].lower()), None)
    if nano is None and baselines:
        nano = baselines[0]

    blocks = [lib.heading("Recommendation — TEMPLATE, numbers filled in, wording not", 2)]
    if reference is None:
        blocks.append(lib.paragraph(
            "No run of ours is in this report, so there is no trade-off to state."
        ))
        return blocks

    reference_name = reference["label"]
    nano_name = nano["label"] if nano else "the RF-DETR baseline"
    columns = ["System", "Trunk", "GFLOPs", "AP", "AP50",
               f"AP vs {reference_name}", f"GFLOPs x vs {reference_name}",
               f"GFLOPs x vs {nano_name}", f"AP vs {nano_name}"]
    rows = []
    for system in ours:
        rows.append([
            system["label"],
            lib.fmt(system.get("trunk") or None, "{}"),
            lib.fmt(system.get("gflops"), "{:.1f}"),
            lib.fmt(system.get("ap"), "{:.4f}"),
            lib.fmt(system.get("ap50"), "{:.4f}"),
            lib.fmt(_delta(system.get("ap"), reference.get("ap")), "{:+.4f}"),
            lib.fmt(_ratio(system.get("gflops"), reference.get("gflops")), "{:.2f}x"),
            lib.fmt(_ratio(system.get("gflops"), nano.get("gflops") if nano else None), "{:.2f}x"),
            lib.fmt(_delta(system.get("ap"), nano.get("ap") if nano else None), "{:+.4f}"),
        ])
    blocks.append(lib.table(columns, rows,
                            "Accuracy lost against FLOPs saved. Positive AP deltas are ours ahead."))

    gap = _ratio(reference.get("gflops"), nano.get("gflops") if nano else None)
    opening = (
        f"{reference_name} runs at {lib.fmt(gap, '{:.1f}')}x the inference FLOPs of {nano_name} "
        f"({lib.fmt(reference.get('gflops'), '{:.0f}')} GFLOPs at {flops_side} against "
        f"{lib.fmt(nano.get('gflops') if nano else None, '{:.0f}')} GFLOPs at "
        f"{lib.fmt((nano or {}).get('gflops_input') or None, '{}')})."
        if gap is not None else
        f"The inference-cost gap between {reference_name} and {nano_name} cannot be quoted: at "
        "least one of the two GFLOPs numbers is not on disk."
    )
    smallest = ours[0] if ours else None
    if smallest is not None and smallest["name"] != reference["name"]:
        saving = _ratio(smallest.get("gflops"), reference.get("gflops"))
        step = (
            f" Dropping to {smallest['label']} "
            f"({smallest.get('trunk') or 'trunk not recorded'}) changes inference cost to "
            f"{lib.fmt(saving, '{:.2f}x')} of {reference_name}'s and moves AP by "
            f"{lib.fmt(_delta(smallest.get('ap'), reference.get('ap')), '{:+.4f}')} "
            f"(AP50 by "
            f"{lib.fmt(_delta(smallest.get('ap50'), reference.get('ap50')), '{:+.4f}')})."
        )
    else:
        step = (
            f" There is only one trunk size in this report ({reference_name}), so there is no "
            "size-down step to price; re-run with the other sizes' run dirs present before "
            "reading this section as a choice."
        )

    blocks.append(lib.paragraph(
        f"TEMPLATE (fill the verdict, keep the numbers). {opening}"
        f"{step} Whether that accuracy is worth that cost depends on what the deployed system is "
        "paid to do, and on whether the gap survives a second seed — "
        "[Freddie's call goes here: the trade, not the arithmetic]."
    ))
    blocks.append(lib.note(lib.HONESTY_LINES[2]))
    return blocks


def build_blocks(context: Dict[str, Any]) -> List[dict]:
    """The whole document, once, as a block list both renderers consume."""
    systems = context["systems"]
    figures = context["figures"]
    blocks: List[dict] = []

    blocks.append(lib.paragraph(context["subtitle"]))
    for line in lib.HONESTY_LINES:
        blocks.append(lib.note(line))

    # 1 — headline
    blocks.append(lib.heading("1 · Headline — cost and accuracy per system", 2))
    headline = lib.headline_table(systems)
    blocks.append(lib.table(
        headline["columns"], headline["rows"],
        f"Trainable/total parameters, GFLOPs and throughput come from trunk_<size>.json "
        f"(measure_trunk.py) for ours and from baseline_metrics.json for RF-DETR; AP/AP50/AP75/"
        f"AR100 and the calibrated tau come from results_summary.json (evaluate_cfd17.py) and are "
        f"re-derived here through pycocotools as a parity check. "
        f"{lib.MISSING} means the number is not on disk — it has not been substituted."))
    if context["parity"]:
        blocks.append(lib.bullets(context["parity"]))

    # 2 — Pareto
    blocks.append(lib.heading("2 · The decision plot — accuracy against inference cost", 2))
    blocks.append(lib.paragraph(
        "Log x, because the trunk sizes are separated by multiples, not by increments. Error bars "
        "are the 95 % image-level bootstrap CI from section 4a: a gap inside two overlapping bars "
        "is not a gap this evaluation can see."))
    for name, alt, caption in (
        ("pareto_ap_gflops.png", "AP against inference GFLOPs, log x",
         "AP against measured inference GFLOPs. Ours joined by trunk size; RF-DETR a separate "
         "series at its own native input."),
        ("pareto_ap50_gflops.png", "AP50 against inference GFLOPs, log x",
         "The same plot on AP50 — the detection axis, with localisation tightness taken out."),
    ):
        if name in figures:
            blocks.append(lib.figure(figures[name], alt, caption))

    # 3 — per source / per size
    blocks.append(lib.heading("3 · Where the AP comes from", 2))
    blocks.append(lib.heading("Per source", 3))
    blocks.append(lib.table(context["per_source_table"]["columns"],
                            context["per_source_table"]["rows"],
                            "AP per CFD source, computed from the same single COCOeval pass by "
                            "restricting the accumulation to that source's images."))
    if "per_source_ap.png" in figures:
        blocks.append(lib.figure(figures["per_source_ap.png"], "Per-source AP dot plot",
                                 "One row per source, one marker per system."))
    blocks.append(lib.heading("Per object size", 3))
    blocks.append(lib.table(context["per_size_table"]["columns"],
                            context["per_size_table"]["rows"],
                            "COCOeval stats[3..5] — AP on objects under 32x32 px, 32-96 px and "
                            "above 96 px, from a COCOeval run here over the full val split."))
    if "per_size_ap.png" in figures:
        blocks.append(lib.figure(figures["per_size_ap.png"], "AP by object size",
                                 "AP small / medium / large per system."))

    # 4 — statistics
    blocks.append(lib.heading("4 · Statistics, and what they are not", 2))
    blocks.append(lib.heading("4a · Image-level bootstrap CIs", 3))
    blocks.append(lib.paragraph(context["bootstrap_text"]))
    blocks.append(lib.table(context["bootstrap_table"]["columns"],
                            context["bootstrap_table"]["rows"],
                            "2.5th and 97.5th percentiles over the resamples."))
    if "bootstrap_ci.png" in figures:
        blocks.append(lib.figure(figures["bootstrap_ci.png"], "Bootstrap CIs on AP and AP50",
                                 "Point estimate and 95 % CI per system."))
    blocks.append(lib.heading("4b · Paired comparisons across trunk sizes", 3))
    if context["paired_source_table"]["rows"]:
        blocks.append(lib.paragraph(
            "Paired over CFD sources: each source is one matched observation, so a trunk that wins "
            "by dominating a single large source does not look like a trunk that wins everywhere. "
            "Median paired difference and sources-won are the reading; the p-value is the third "
            "number, not the first."))
        blocks.append(lib.table(context["paired_source_table"]["columns"],
                                context["paired_source_table"]["rows"],
                                "Per-source AP, paired. A > B means the left system is ahead."))
    else:
        blocks.append(lib.paragraph(
            "Only one of our runs is in this report, so there is no trunk-size pair to test."))
    if context["paired_image_table"]["rows"]:
        blocks.append(lib.paragraph(
            "Paired over val images on matched F1 at each system's own calibrated tau. An image "
            "with no ground truth and no surviving detection scores F1 = 1.0 for both systems, so "
            "it contributes an exact zero to the difference and is discarded by the signed-rank "
            "test; the count of non-zero pairs is reported beside it."))
        blocks.append(lib.table(context["paired_image_table"]["columns"],
                                context["paired_image_table"]["rows"],
                                "Per-image matched F1, paired, with a bootstrap CI on the mean "
                                "difference."))
    blocks.append(lib.heading("4c · Scope", 3))
    blocks.append(lib.note(lib.HONESTY_LINES[1]))
    blocks.append(lib.paragraph(context["scope_text"]))

    # 5 — curves
    blocks.append(lib.heading("5 · Training, PR and calibration", 2))
    for name, alt, caption in (
        ("training_curves.png", "Training curves per run",
         "Train/val loss, val AP50, val F1 and the LR schedule, one line per run dir."),
        ("pr_curves.png", "Precision-recall at IoU 0.50",
         "From COCOeval's precision array at IoU 0.50, area 'all', maxDets 100."),
        ("tau_calibration.png", "Tau calibration, count MAE against F1",
         "The 4_evaluate.ipynb twin-axis panel: count MAE (red, left) against matched F1 (blue, "
         "right); dashed line at the MAE-argmin tau the headline quotes, dotted at the F1-argmax "
         "tau where the two disagree."),
    ):
        if name in figures:
            blocks.append(lib.figure(figures[name], alt, caption))
    blocks.append(lib.heading("Best epoch against last epoch", 3))
    if context["best_last_table"]["rows"]:
        blocks.append(lib.table(context["best_last_table"]["columns"],
                                context["best_last_table"]["rows"],
                                "predictions.json (the selected epoch) against predictions_last.json "
                                "(the final epoch), both rescored by the same function. Liam's "
                                "convention: report both, argue the pick, never quote the "
                                "flattering one silently."))
    else:
        blocks.append(lib.paragraph(
            "No run in this report has both a predictions.json and a predictions_last.json scored "
            "into results_summary.json."))

    # 6 — recommendation
    blocks += recommendation_blocks(systems, context["flops_side"])

    # provenance
    blocks.append(lib.heading("Provenance — what was read, and what was missing", 2))
    blocks.append(lib.bullets(context["provenance"]))
    if context["missing"]:
        blocks.append(lib.heading("Not on disk", 3))
        blocks.append(lib.bullets(context["missing"]))
    return blocks


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.results_dir is None:
        args.results_dir = args.baselines_dir
    if args.taus_from is None and args.results_dir is not None:
        args.taus_from = Path(args.results_dir) / "results_summary.json"
    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    html_path = Path(args.html) if args.html else out_dir / "size_report.html"

    started = time.perf_counter()
    coco_gt, gt_raw, annotations_path = load_ground_truth(args.data_root)
    # COCOeval sorts its imgIds through np.unique; mirror that so the evalImgs
    # image axis and every index we compute against it line up exactly.
    ordered_ids = list(np.unique(coco_gt.getImgIds()))
    position_of = {image_id: index for index, image_id in enumerate(ordered_ids)}
    n_images = len(ordered_ids)
    n_cats = len(coco_gt.getCatIds())
    known_ids = set(ordered_ids)

    gt_boxes: Dict[Any, List[Sequence[float]]] = defaultdict(list)
    for ann in gt_raw["annotations"]:
        gt_boxes[ann["image_id"]].append(ann["bbox"])
    source_of = {im["id"]: str(im.get("dataset") or "unknown") for im in gt_raw["images"]}
    positions_by_source: Dict[str, List[int]] = defaultdict(list)
    for image_id in ordered_ids:
        positions_by_source[source_of.get(image_id, "unknown")].append(position_of[image_id])
    sources = sorted(positions_by_source)

    summary = load_json(args.taus_from) or {}
    summary_models = summary.get("models", {}) if isinstance(summary, dict) else {}
    baseline_metrics = load_json(Path(args.results_dir) / "baseline_metrics.json"
                                 if args.results_dir else None) or {}

    systems = discover_systems(args)
    if not systems:
        raise SystemExit("nothing to report — pass --runs-dir and/or --baselines-dir")

    provenance = [
        f"val ground truth: {annotations_path} — {n_images} images, "
        f"{sum(len(v) for v in gt_boxes.values())} boxes, {len(sources)} sources, "
        f"{n_cats} categor{'y' if n_cats == 1 else 'ies'}",
        f"tau calibration and per-scope metrics: "
        f"{args.taus_from if args.taus_from else lib.MISSING}"
        f"{'' if summary_models else ' (absent — tau columns will be ' + lib.MISSING + ')'}",
        f"baseline cost metrics: "
        f"{(Path(args.results_dir) / 'baseline_metrics.json') if args.results_dir else lib.MISSING}"
        f"{'' if baseline_metrics else ' (absent)'}",
        f"bootstrap: {args.bootstrap} image-level resamples, seed {args.seed}, "
        f"{args.workers} worker process(es)",
    ]
    missing: List[str] = []
    parity: List[str] = []

    trunk_cache: Dict[str, Optional[dict]] = {}

    def trunk_record(size: Optional[str]) -> Optional[dict]:
        if size is None:
            return None
        if args.results_dir is None:
            if "no --results-dir" not in "".join(missing):
                missing.append("no --results-dir was given, so no trunk_<size>.json could be "
                               "read: every parameter, GFLOPs, throughput and memory cell for "
                               "our runs is " + lib.MISSING)
            return None
        if size not in trunk_cache:
            path = Path(args.results_dir) / f"trunk_{size}.json"
            record = load_json(path)
            if record is None:
                missing.append(f"trunk measurements for '{size}': no {path} "
                               "(run measure_trunk.py --size " f"{size} --out {path})")
            else:
                provenance.append(f"trunk measurements for '{size}': {path}")
            trunk_cache[size] = record
        return trunk_cache[size]

    recall_thresholds = None
    boot_seconds: Dict[str, float] = {}

    for system in systems:
        raw = json.loads(Path(system["predictions"]).read_text(encoding="utf-8"))
        detections = [d for d in raw if d["image_id"] in known_ids]
        dropped = len(raw) - len(detections)
        if dropped:
            parity.append(
                f"{system['label']}: {dropped} detection(s) on images not in this split were "
                "ignored — that prediction set was made against a different val set")
        if not detections:
            missing.append(f"{system['label']}: no detections on this split's images "
                           f"({system['predictions']})")
            system.update({"ap": None, "ap50": None, "ap75": None, "ar100": None})
            continue

        by_image = top_k_per_image(detections, k=100)
        capped = [d for dets in by_image.values() for d in dets]
        evaluator, stats, precision = evaluate_system(coco_gt, capped, ordered_ids)
        n_areas = len(evaluator._paramsEval.areaRng)
        if recall_thresholds is None:
            recall_thresholds = np.array(evaluator.params.recThrs, dtype=float)

        system["ap_recomputed"] = stats[0]
        system["ap50_recomputed"] = stats[1]
        system["per_size"] = {"small": stats[3], "medium": stats[4], "large": stats[5]}
        system["pr_curve"] = precision[0, :, 0, 0, -1].tolist()
        system["n_detections"] = len(detections)

        record = summary_models.get(system["summary_key"], {})
        scope = (record.get("scopes") or {}).get("full", {})
        for key in ("ap", "ap50", "ap75", "ar100"):
            system[key] = scope.get(key) if key in scope else stats[
                {"ap": 0, "ap50": 1, "ap75": 2, "ar100": 8}[key]]
        system["tau"] = scope.get("best_tau_mae")
        system["tau_f1"] = scope.get("best_tau_f1")
        system["f1_at_tau"] = scope.get("f1_at_best_tau")
        system["mae_at_tau"] = scope.get("mae_at_best")
        system["sweep"] = scope.get("sweep")
        if not scope:
            where = str(args.taus_from) if args.taus_from else "any results_summary.json (none given)"
            missing.append(f"{system['label']}: no '{system['summary_key']}' entry in {where} — "
                           f"tau, F1 @tau and count MAE @tau are {lib.MISSING}, and the AP columns "
                           "fall back to this script's own COCOeval")
        else:
            for key, ours_value in (("ap", stats[0]), ("ap50", stats[1])):
                if lib.is_missing(ours_value) or lib.is_missing(scope.get(key)):
                    continue
                if abs(scope[key] - ours_value) > 1e-4:
                    parity.append(
                        f"{system['label']}: results_summary {key} {scope[key]:.5f} disagrees with "
                        f"this script's COCOeval {ours_value:.5f} — one of the two is stale")

        # per-source AP: the same evaluate() output, accumulated over one source's images.
        per_source = {}
        compact = lib.compact_evalimgs(evaluator.evalImgs)
        for source in sources:
            positions = positions_by_source[source]
            subset = lib.resample_evalimgs(compact, positions, n_cats, n_areas, n_images)
            ids = [ordered_ids[p] for p in positions]
            ap_source, _ = lib.accumulate_ap(evaluator._paramsEval, subset, ids, reduced=True)
            per_source[source] = ap_source
        system["per_source"] = per_source

        # cost + throughput
        if system["kind"] == "ours":
            record = trunk_record(system.get("trunk"))
            if record:
                system["params_trainable"] = record.get("params_trainable")
                system["params_total"] = record.get("params_total")
                system["gflops"] = record.get("gflops")
                system["gflops_input"] = args.flops_side
                system["images_per_s"] = record.get("images_per_s")
                system["ms_per_image"] = record.get("ms_per_image")
                system["peak_gpu_mem_mb"] = record.get("peak_gpu_mem_mb")
        else:
            metrics = baseline_metrics.get(system["name"], {})
            system["gflops"] = metrics.get("gflops")
            side = metrics.get("side")
            system["gflops_input"] = f"{side}x{side}" if side else None
            seconds = metrics.get("seconds_per_image")
            system["images_per_s"] = (1.0 / seconds) if seconds else None
            system["ms_per_image"] = (1000.0 * seconds) if seconds else None
            system["peak_gpu_mem_mb"] = metrics.get("peak_gpu_mem_mb")
            if lib.is_missing(system["gflops"]):
                missing.append(f"{system['label']}: no 'gflops' in baseline_metrics.json "
                               "(run rfdetr_flops.py) — it cannot appear on the Pareto plot")
            if not metrics:
                missing.append(f"{system['label']}: no entry in baseline_metrics.json — "
                               "cost and throughput columns are " + lib.MISSING)

        # per-image F1 at the calibrated tau, for the paired image test
        if not lib.is_missing(system.get("tau")):
            system["image_f1"] = per_image_f1(by_image, gt_boxes, ordered_ids,
                                              float(system["tau"]), args.match_iou)

        # the bootstrap — evaluate() is NOT re-run
        tic = time.perf_counter()
        system["bootstrap"] = lib.bootstrap_ap(
            {
                "evalimgs": compact,
                "params": evaluator._paramsEval,
                "img_ids": list(ordered_ids),
                "n_images": n_images,
                "n_cats": n_cats,
                "n_areas": n_areas,
            },
            n_boot=args.bootstrap, seed=args.seed, workers=max(1, args.workers),
        )
        boot_seconds[system["name"]] = time.perf_counter() - tic
        print(f"  {system['label']}: AP {lib.fmt(system.get('ap'), '{:.4f}')} "
              f"AP50 {lib.fmt(system.get('ap50'), '{:.4f}')} | bootstrap "
              f"{args.bootstrap} resamples in {boot_seconds[system['name']]:.1f}s", flush=True)
        del compact, evaluator

    if recall_thresholds is None:
        recall_thresholds = np.linspace(0.0, 1.0, 101)

    # ---------------- history -------------------------------------------- #
    runs = []
    for system in systems:
        if system["kind"] != "ours" or system.get("run_dir") is None:
            continue
        history = load_json(system["run_dir"] / "history.json")
        if history is None:
            missing.append(f"{system['label']}: no history.json in {system['run_dir']} — "
                           "no training curve")
            continue
        system["history"] = history
        runs.append(system)

    # ---------------- tables --------------------------------------------- #
    per_source_table = {
        "columns": ["Source", "val images"] + [s["label"] for s in systems],
        "rows": [
            [source, str(len(positions_by_source[source]))]
            + [lib.fmt((s.get("per_source") or {}).get(source), "{:.4f}") for s in systems]
            for source in sources
        ],
    }
    per_size_table = {
        "columns": ["System", "AP small", "AP medium", "AP large"],
        "rows": [
            [s["label"]] + [lib.fmt((s.get("per_size") or {}).get(g), "{:.4f}")
                            for g in ("small", "medium", "large")]
            for s in systems
        ],
    }
    bootstrap_table = {
        "columns": ["System", "AP", "AP 2.5 %", "AP 97.5 %", "AP50", "AP50 2.5 %", "AP50 97.5 %",
                    "resamples", "seconds"],
        "rows": [],
    }
    for system in systems:
        boot = system.get("bootstrap") or {}
        ap_ci = boot.get("ap_ci") or [None, None]
        ap50_ci = boot.get("ap50_ci") or [None, None]
        bootstrap_table["rows"].append([
            system["label"],
            lib.fmt(system.get("ap"), "{:.4f}"),
            lib.fmt(ap_ci[0], "{:.4f}"), lib.fmt(ap_ci[1], "{:.4f}"),
            lib.fmt(system.get("ap50"), "{:.4f}"),
            lib.fmt(ap50_ci[0], "{:.4f}"), lib.fmt(ap50_ci[1], "{:.4f}"),
            lib.fmt(boot.get("n_boot"), "{:d}"),
            lib.fmt(boot_seconds.get(system["name"]), "{:.1f}"),
        ])

    # ---------------- paired tests ---------------------------------------- #
    ours = [s for s in systems if s["kind"] == "ours"]
    paired_source_table = {
        "columns": ["A vs B", "median dAP", "sources A wins", "sources B wins", "ties",
                    "test", "statistic", "p"],
        "rows": [],
    }
    paired_image_table = {
        "columns": ["A vs B", "tau A", "tau B", "mean dF1", "dF1 2.5 %", "dF1 97.5 %",
                    "images", "non-zero pairs", "test", "p"],
        "rows": [],
    }
    paired_json = {"per_source_ap": [], "per_image_f1": []}
    for index, left in enumerate(ours):
        for right in ours[index + 1:]:
            pair = f"{left['label']} vs {right['label']}"
            a = [(left.get("per_source") or {}).get(s, float("nan")) for s in sources]
            b = [(right.get("per_source") or {}).get(s, float("nan")) for s in sources]
            result = lib.paired_test(a, b, unit="source")
            result["pair"] = pair
            paired_json["per_source_ap"].append(result)
            paired_source_table["rows"].append([
                pair,
                lib.fmt(result["median_diff"], "{:+.4f}"),
                str(result["n_won"]), str(result["n_lost"]), str(result["n_tied"]),
                result["test"],
                lib.fmt(result["statistic"], "{:.1f}"),
                lib.fmt(result["p_value"], "{:.4f}"),
            ])

            if left.get("image_f1") is None or right.get("image_f1") is None:
                continue
            diff = np.asarray(left["image_f1"]) - np.asarray(right["image_f1"])
            image_result = lib.paired_test(left["image_f1"], right["image_f1"], unit="image")
            ci = lib.bootstrap_mean_diff(diff, n_boot=max(1, args.bootstrap), seed=args.seed)
            image_result["pair"] = pair
            image_result["mean_diff_ci"] = ci["ci"]
            image_result["tau_a"] = left.get("tau")
            image_result["tau_b"] = right.get("tau")
            paired_json["per_image_f1"].append(image_result)
            paired_image_table["rows"].append([
                pair,
                lib.fmt(left.get("tau"), "{:.2f}"), lib.fmt(right.get("tau"), "{:.2f}"),
                lib.fmt(ci["mean"], "{:+.4f}"),
                lib.fmt(ci["ci"][0], "{:+.4f}"), lib.fmt(ci["ci"][1], "{:+.4f}"),
                str(image_result["n_pairs"]), str(image_result["n_nonzero"]),
                image_result["test"], lib.fmt(image_result["p_value"], "{:.4g}"),
            ])

    # ---------------- best vs last ---------------------------------------- #
    best_last_table = {
        "columns": ["Run", "Checkpoint", "AP", "AP50", "AP75", "F1 @tau", "tau"],
        "rows": [],
    }
    for system in systems:
        if system["kind"] != "ours":
            continue
        for which, key in (("best (selected)", system.get("summary_key")),
                           ("last (final epoch)", system.get("summary_key_last"))):
            scope = ((summary_models.get(key) or {}).get("scopes") or {}).get("full")
            if not scope:
                continue
            best_last_table["rows"].append([
                system["label"], which,
                lib.fmt(scope.get("ap"), "{:.4f}"), lib.fmt(scope.get("ap50"), "{:.4f}"),
                lib.fmt(scope.get("ap75"), "{:.4f}"),
                lib.fmt(scope.get("f1_at_best_tau"), "{:.4f}"),
                lib.fmt(scope.get("best_tau_mae"), "{:.2f}"),
            ])

    # ---------------- figures --------------------------------------------- #
    figures: Dict[str, str] = {}

    def emit(name: str, builder) -> None:
        path = figures_dir / name
        builder(path)
        figures[name] = f"figures/{name}"

    emit("pareto_ap_gflops.png",
         lambda p: figure_pareto(systems, "ap", p, args.flops_side))
    emit("pareto_ap50_gflops.png",
         lambda p: figure_pareto(systems, "ap50", p, args.flops_side))
    emit("per_source_ap.png", lambda p: figure_per_source(systems, sources, p))
    emit("per_size_ap.png", lambda p: figure_per_size(systems, p))
    emit("bootstrap_ci.png", lambda p: figure_bootstrap(systems, p))
    emit("training_curves.png", lambda p: figure_training_curves(runs, p))
    emit("pr_curves.png", lambda p: figure_pr_curves(systems, recall_thresholds, p))
    emit("tau_calibration.png", lambda p: figure_tau_calibration(systems, p))

    # ---------------- prose ------------------------------------------------ #
    elapsed = time.perf_counter() - started
    subtitle = (
        f"CFD-17 val, {n_images} images over {len(sources)} sources · "
        f"{len(systems)} system(s) · {args.bootstrap} bootstrap resamples, seed {args.seed}, "
        f"{args.workers} worker(s) · generated in {elapsed:.1f}s by size_report.py"
    )
    bootstrap_text = (
        f"Each system's detections were scored by one COCOeval pass over all {n_images} val "
        f"images; that pass's per-image evaluation entries were then resampled with replacement "
        f"{args.bootstrap} times and re-accumulated. COCOeval.evaluate() is never re-run, so a "
        "resample costs an accumulate rather than a full evaluation, and every system is scored "
        "on the same resampled image sets (common random numbers, derived from --seed alone)."
    )
    scope_text = (
        "These intervals resample IMAGES. They measure how much of each number is an accident of "
        "which frames landed in the val split — nothing else. They do not contain initialisation, "
        "data-order or augmentation variance, because there is one training seed per trunk size. "
        "Before any size is chosen on the strength of a gap this report shows, run a second seed "
        "of the chosen size and re-read the gap against the seed-to-seed spread."
    )

    context = {
        "systems": systems,
        "figures": figures,
        "subtitle": subtitle,
        "per_source_table": per_source_table,
        "per_size_table": per_size_table,
        "bootstrap_table": bootstrap_table,
        "bootstrap_text": bootstrap_text,
        "paired_source_table": paired_source_table,
        "paired_image_table": paired_image_table,
        "best_last_table": best_last_table,
        "scope_text": scope_text,
        "provenance": provenance,
        "missing": missing,
        "parity": parity,
        "flops_side": args.flops_side,
    }
    blocks = build_blocks(context)

    title = "Trunk size vs cost vs accuracy — CFD-17 val"
    markdown_path = out_dir / "size_report.md"
    markdown_path.write_text(lib.render_markdown(blocks, title), encoding="utf-8")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    # The subtitle is already the document's first block, so it is not passed
    # again as render_html's styled meta line.
    html_path.write_text(lib.render_html(blocks, title, out_dir), encoding="utf-8")

    payload = {
        "generated_seconds": elapsed,
        "annotations": str(annotations_path),
        "n_images": n_images,
        "sources": sources,
        "honesty_lines": list(lib.HONESTY_LINES),
        "bootstrap": {"resamples": args.bootstrap, "seed": args.seed, "workers": args.workers,
                      "seconds_per_system": boot_seconds},
        "systems": [
            {
                "name": s["name"], "label": s["label"], "kind": s["kind"], "trunk": s.get("trunk"),
                "backbone_trainable": s.get("backbone_trainable"),
                "predictions": str(s["predictions"]),
                "n_detections": s.get("n_detections"),
                **{k: s.get(k) for k in (
                    "params_trainable", "params_total", "gflops", "gflops_input",
                    "images_per_s", "ms_per_image", "peak_gpu_mem_mb",
                    "ap", "ap50", "ap75", "ar100", "ap_recomputed", "ap50_recomputed",
                    "tau", "tau_f1", "f1_at_tau", "mae_at_tau", "per_source", "per_size",
                )},
                "bootstrap_ci": {
                    "ap": (s.get("bootstrap") or {}).get("ap_ci"),
                    "ap50": (s.get("bootstrap") or {}).get("ap50_ci"),
                    "n_boot": (s.get("bootstrap") or {}).get("n_boot"),
                },
            }
            for s in systems
        ],
        "paired": paired_json,
        "tables": {
            "headline": lib.headline_table(systems),
            "per_source_ap": per_source_table,
            "per_size_ap": per_size_table,
            "bootstrap": bootstrap_table,
            "paired_per_source_ap": paired_source_table,
            "paired_per_image_f1": paired_image_table,
            "best_vs_last": best_last_table,
        },
        "figures": figures,
        "provenance": provenance,
        "missing": missing,
        "parity_warnings": parity,
    }
    (out_dir / "size_report.json").write_text(
        json.dumps(payload, indent=1, default=str), encoding="utf-8")

    print()
    print("| " + " | ".join(lib.HEADLINE_COLUMNS) + " |")
    print("|" + "---|" * len(lib.HEADLINE_COLUMNS))
    for row in lib.headline_table(systems)["rows"]:
        print("| " + " | ".join(row) + " |")
    print()
    print(f"wrote {markdown_path}")
    print(f"wrote {html_path}")
    print(f"wrote {out_dir / 'size_report.json'}")
    print(f"wrote {len(figures)} figures to {figures_dir}")
    if missing:
        print(f"{len(missing)} value(s) not on disk — shown as {lib.MISSING}:")
        for line in missing:
            print(f"  - {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
