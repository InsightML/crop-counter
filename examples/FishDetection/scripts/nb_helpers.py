"""Notebook helpers for ``examples/FishDetection`` — the point task on CFD Brackish.

Everything here is figure-making, table-formatting or a thin decode wrapper: the
science lives in ``cropcounter``, and these functions only arrange its outputs so
the four notebooks stay readable. They are importable from a notebook with::

    import sys; sys.path.insert(0, 'examples/FishDetection/scripts')
    import nb_helpers as nbh

Two deliberate choices:

* **No pyplot.** Figures are built with ``matplotlib.figure.Figure`` + an explicit
  Agg canvas, exactly as :func:`cropcounter.inference.save_visualization` does, so
  a notebook that saves dozens of large panels never fills pyplot's global figure
  registry. Every plotting function writes a PNG and returns its path; the
  notebook displays that file.
* **Autocast is an argument, never a default.** :func:`decode_image_points` takes
  ``amp`` explicitly because the backbone reproduction gate has to run in fp32
  while training and evaluation run in bf16 on CUDA — the same model, different
  numerics, and the gate is only meaningful like-for-like.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from PIL import Image

#: Radius, in *image* pixels, of the ground-truth dot drawn on sample grids.
#: Drawn as a data-coordinate circle (not a scatter marker) so it stays ~10 px
#: on the frame whatever the figure size or dpi.
GT_DOT_RADIUS_PX = 10.0

#: Columns per row in the seeded data-sample grids.
SAMPLE_GRID_COLS = 4

#: Fraction of a sample grid drawn from frames that contain fish. Brackish is
#: ~60 % empty, so a uniform sample would be mostly water.
SAMPLE_POSITIVE_FRACTION = 2.0 / 3.0

_TRAIN_COLOUR = "#2b5f9e"
_VAL_COLOUR = "#c1440e"


def _new_figure(figsize: Tuple[float, float]) -> Figure:
    """A standalone Agg figure, outside pyplot's global registry."""
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    return fig


def _save(fig: Figure, out_path: Path, dpi: int = 100) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path


# --------------------------------------------------------------------------- #
# Reading the point-task data root
# --------------------------------------------------------------------------- #


def load_points_document(points_root: Path, split: str) -> Dict:
    """Load one split's COCO **keypoints** document from a ``cfd points`` root."""
    path = Path(points_root) / split / "annotations.json"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def points_by_image(document: Dict) -> Dict[int, np.ndarray]:
    """Map image id -> (N, 2) array of keypoint (x, y) in image pixels.

    Only visible keypoints (``v > 0``) are kept, matching
    :func:`cropcounter.crop_dataset.parse_coco_keypoints`. Images with no
    annotation are absent from the mapping (callers treat missing as empty).
    """
    out: Dict[int, List[List[float]]] = {}
    for ann in document.get("annotations", ()):
        keypoints = ann.get("keypoints") or []
        for i in range(0, len(keypoints) - 2, 3):
            x, y, visibility = keypoints[i], keypoints[i + 1], keypoints[i + 2]
            if not visibility:
                continue
            out.setdefault(int(ann["image_id"]), []).append([float(x), float(y)])
    return {k: np.asarray(v, dtype=np.float32).reshape(-1, 2) for k, v in out.items()}


def per_image_counts(document: Dict) -> np.ndarray:
    """Per-frame point count, in the document's image order (zeros included)."""
    by_image = points_by_image(document)
    return np.array(
        [len(by_image.get(int(im["id"]), ())) for im in document.get("images", ())],
        dtype=np.int64,
    )


# --------------------------------------------------------------------------- #
# Table 1 — the reformat table
# --------------------------------------------------------------------------- #


def table1_rows(points_root: Path, splits: Sequence[str] = ("train", "val")) -> List[Dict]:
    """Per-split rows for Table 1, read straight out of ``points_summary.json``.

    The table is *printed from artefacts*, not recomputed: if the summary and the
    annotation files ever disagree, the round-trip verify in ``1_reformat`` is the
    thing that should fail, not the table that should quietly paper over it.
    """
    with (Path(points_root) / "points_summary.json").open(encoding="utf-8") as fh:
        summary = json.load(fh)
    rows: List[Dict] = []
    for split in splits:
        if split not in summary:
            continue
        entry = summary[split]
        n_images = int(entry["n_images"])
        n_empty = int(entry["n_empty"])
        rows.append({
            "split": split,
            "images": n_images,
            "points": int(entry["n_points"]),
            "empty_images": n_empty,
            "empty_pct": (100.0 * n_empty / n_images) if n_images else 0.0,
        })
    return rows


def render_table1(rows: Sequence[Dict], title: str = "") -> str:
    """Markdown for Table 1: split · images · points · empty images · empty %."""
    lines = []
    if title:
        lines += [f"{title}", ""]
    lines += [
        "| Split | Images | Points | Empty images | Empty % |",
        "| ----- | ------ | ------ | ------------ | ------- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['split']} | {row['images']:,} | {row['points']:,} | "
            f"{row['empty_images']:,} | {row['empty_pct']:.1f}% |"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Data visualisation
# --------------------------------------------------------------------------- #


def plot_point_samples(
    points_root: Path,
    split: str,
    out_path: Path,
    n: int = 12,
    seed: int = 0,
    cols: int = SAMPLE_GRID_COLS,
) -> Path:
    """Seeded grid of frames with their ground-truth points as green dots.

    Two thirds of the sample is drawn from frames that contain fish, the rest
    from empty frames, so the grid shows both the signal and the negatives the
    counter is being asked to keep at zero.
    """
    document = load_points_document(points_root, split)
    by_image = points_by_image(document)
    images = list(document.get("images", ()))
    rng = random.Random(seed)
    positive = [im for im in images if len(by_image.get(int(im["id"]), ()))]
    negative = [im for im in images if not len(by_image.get(int(im["id"]), ()))]
    k_pos = min(int(n * SAMPLE_POSITIVE_FRACTION), len(positive))
    picks = rng.sample(positive, k_pos) + rng.sample(negative, min(n - k_pos, len(negative)))

    rows = -(-len(picks) // cols)
    fig = _new_figure((4.2 * cols, 2.7 * rows))
    axes = np.atleast_1d(fig.subplots(rows, cols)).ravel()
    images_dir = Path(points_root) / split / "images"
    for ax, image in zip(axes, picks):
        ax.imshow(Image.open(images_dir / image["file_name"]))
        points = by_image.get(int(image["id"]), np.empty((0, 2), np.float32))
        for x, y in points:
            ax.add_patch(Circle((x, y), radius=GT_DOT_RADIUS_PX, color="lime", alpha=0.55))
        sequence = image.get("cfd_sequence") or image.get("original_data_source") or ""
        ax.set_title(f"{split} · {len(points)} fish · {str(sequence)[:30]}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(picks):]:
        ax.axis("off")

    n_points = sum(len(v) for v in by_image.values())
    empty_pct = 100.0 * len(negative) / max(len(images), 1)
    fig.suptitle(
        f"Brackish {split} (point task): {len(images):,} frames · {n_points:,} points · "
        f"{empty_pct:.0f}% empty — ground-truth points as green dots "
        f"(r≈{GT_DOT_RADIUS_PX:.0f} px; 2/3 of the sample drawn from frames with fish)",
        fontsize=11,
    )
    return _save(fig, out_path)


def plot_count_distribution(
    points_root: Path,
    out_path: Path,
    splits: Sequence[str] = ("train", "val"),
) -> Path:
    """Points-per-frame histogram for each split, plus a log-count panel.

    The left panel is the distribution a counter is scored on; the right repeats
    it on a log y-axis, because the empty-frame spike otherwise flattens every
    other bar into the axis.
    """
    colours = {"train": _TRAIN_COLOUR, "val": _VAL_COLOUR}
    fig = _new_figure((12, 3.6))
    ax_linear, ax_log = fig.subplots(1, 2)
    for split in splits:
        counts = per_image_counts(load_points_document(points_root, split))
        if not len(counts):
            continue
        bins = np.arange(0, counts.max() + 2) - 0.5
        label = (
            f"{split} (n={len(counts):,}, "
            f"{100.0 * (counts == 0).mean():.0f}% empty, "
            f"mean {counts.mean():.2f})"
        )
        for ax in (ax_linear, ax_log):
            ax.hist(counts, bins=bins, alpha=0.6, color=colours.get(split), density=True,
                    label=label if ax is ax_linear else None)
    ax_linear.set(title="points per frame", xlabel="points in frame",
                  ylabel="fraction of frames")
    ax_linear.legend(fontsize=8)
    ax_log.set(title="points per frame (log y)", xlabel="points in frame", yscale="log")
    return _save(fig, out_path, dpi=110)


def plot_train_tiles_with_targets(
    dataset,
    out_path: Path,
    n: int = 8,
    seed: int = 0,
    output_stride: int = 4,
    title: Optional[str] = None,
) -> Path:
    """Augmented training tiles with their rendered Gaussian targets overlaid.

    ``dataset`` is a train-mode :class:`cropcounter.crop_dataset.CropTileDataset`,
    which yields ``(image, target, n_points)``. The target is nearest-neighbour
    upsampled by ``output_stride`` so a peak sits visibly on its fish: a hot blob
    off the animal, or an upside-down fish, is a target/augmentation bug and the
    whole reason this panel exists.
    """
    from cropcounter.dinov3_pyramid import IMAGENET_MEAN, IMAGENET_STD

    rng = np.random.default_rng(seed)
    cols = 4
    rows = -(-n // cols)
    fig = _new_figure((4.2 * cols, 4.2 * rows))
    axes = np.atleast_1d(fig.subplots(rows, cols)).ravel()
    mean, std = np.array(IMAGENET_MEAN), np.array(IMAGENET_STD)
    for ax in axes[:n]:
        image, target, n_points = dataset[int(rng.integers(len(dataset)))]
        rgb = (image.permute(1, 2, 0).numpy() * std + mean).clip(0, 1)
        ax.imshow(rgb)
        heat = target[0].numpy()
        ax.imshow(np.kron(heat, np.ones((output_stride, output_stride))),
                  cmap="hot", alpha=0.45, vmin=0, vmax=1)
        ax.set_title(f"{n_points} points in tile", fontsize=9)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(
        title or (
            f"What the head trains on — augmented tiles with their stride-{output_stride} "
            "peak-1.0 Gaussian targets (hot). A blob off its fish, or an upside-down "
            "fish, is a bug."
        ),
        fontsize=10,
    )
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Decoding + tau calibration
# --------------------------------------------------------------------------- #


def decode_image_points(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    width: int,
    height: int,
    device: torch.device,
    tau: float,
    k: int = 3,
    nms_radius: float = 1.5,
    output_stride: int = 4,
    amp: bool = True,
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Forward one padded image and decode it to in-bounds points.

    ``amp=True`` reproduces the bf16-on-CUDA path that training and
    :func:`cropcounter.inference.predict_prob` use; ``amp=False`` forces fp32,
    which is what the backbone reproduction gate needs (its committed reference
    counts were produced in fp32 on MPS). On non-CUDA devices both are fp32.

    Returns ``(points, scores, prob)`` — points (N, 2) in source pixels.
    """
    from cropcounter.inference import decode_in_bounds

    with torch.no_grad(), torch.autocast(
        device.type, torch.bfloat16, enabled=amp and device.type == "cuda"
    ):
        logits = model(image_tensor.unsqueeze(0).to(device))
    prob = torch.sigmoid(logits.float()).cpu()
    points, scores = decode_in_bounds(
        prob, width, height, tau=tau, k=k, nms_radius=nms_radius,
        output_stride=output_stride,
    )
    return points, scores, prob


def calibrate_tau(
    model: torch.nn.Module,
    val_ds,
    device: torch.device,
    taus: Sequence[float],
    k: int = 3,
    nms_radius: float = 1.5,
    output_stride: int = 4,
    match_radius_px: float = 24.0,
) -> Dict:
    """Sweep the decode threshold over val in one model pass (Liam's pattern).

    ``cropcounter.metrics.sweep_tau`` computes each image's probability map once
    and decodes it at every tau, so the whole sweep costs one forward pass.

    Returns a dict with the full ``rows``, the tau that minimises count MAE
    (``tau_mae``, the choice — same as the wheat report), the tau that maximises
    localization F1 (``tau_f1``, the sanity check), and the chosen row.
    """
    from torch.utils.data import DataLoader

    from cropcounter.crop_dataset import collate_val
    from cropcounter.metrics import sweep_tau

    loader = DataLoader(val_ds, batch_size=1, collate_fn=collate_val)
    rows = sweep_tau(
        model, loader, device, list(taus), k=k, nms_radius=nms_radius,
        output_stride=output_stride, match_radius_px=match_radius_px,
    )
    by_mae = min(rows, key=lambda r: r["count_mae"])
    by_f1 = max(rows, key=lambda r: r["f1"])
    return {
        "rows": rows,
        "k": k,
        "nms_radius": float(nms_radius),
        "tau_mae": float(by_mae["tau"]),
        "tau_f1": float(by_f1["tau"]),
        "chosen_tau": float(by_mae["tau"]),
        "chosen_by": "count_mae",
        "chosen": by_mae,
    }


def plot_tau_sweeps(
    sweeps: Sequence[Tuple[str, Dict]],
    out_path: Path,
    suptitle: Optional[str] = None,
) -> Path:
    """One twin-axis panel per sweep: count MAE (left) and F1 (right) vs tau.

    ``sweeps`` is ``[(label, calibration_dict), ...]`` as returned by
    :func:`calibrate_tau`. The chosen tau is marked; the F1 argmax is marked too
    when it differs, because a gap between them is the thing worth noticing.
    """
    fig = _new_figure((7.5 * len(sweeps), 4.4))
    axes = np.atleast_1d(fig.subplots(1, len(sweeps), squeeze=False)).ravel()
    for ax_mae, (label, calibration) in zip(axes, sweeps):
        rows = calibration["rows"]
        taus = [r["tau"] for r in rows]
        ax_mae.plot(taus, [r["count_mae"] for r in rows], "o-", color="#c0392b")
        ax_mae.set_xlabel("tau")
        ax_mae.set_ylabel("count MAE", color="#c0392b")
        ax_f1 = ax_mae.twinx()
        ax_f1.plot(taus, [r["f1"] for r in rows], "s-", color=_TRAIN_COLOUR)
        ax_f1.set_ylabel("localization F1", color=_TRAIN_COLOUR)
        ax_mae.axvline(calibration["chosen_tau"], color="gray", ls="--", lw=1)
        if calibration["tau_f1"] != calibration["chosen_tau"]:
            ax_mae.axvline(calibration["tau_f1"], color="gray", ls=":", lw=1)
        chosen = calibration["chosen"]
        ax_mae.set_title(
            f"{label}\ntau {chosen['tau']:.2f} (MAE-argmin): MAE {chosen['count_mae']:.2f}, "
            f"F1 {chosen['f1']:.3f}, bias {chosen['count_bias']:+.2f}"
            f"   [F1-argmax tau {calibration['tau_f1']:.2f}]",
            fontsize=10,
        )
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    return _save(fig, out_path, dpi=110)


# --------------------------------------------------------------------------- #
# Table 2 — the tuning table
# --------------------------------------------------------------------------- #

#: (metric key, column header, "min" or "max" is better) for Table 2.
TUNING_COLUMNS: Tuple[Tuple[str, str, str], ...] = (
    ("count_mae", "MAE", "min"),
    ("count_rmse", "RMSE", "min"),
    ("count_bias", "Bias", "abs_min"),
    ("precision", "Precision", "max"),
    ("recall", "Recall", "max"),
    ("f1", "F1", "max"),
)


def _best_index(values: Sequence[float], direction: str) -> int:
    if direction == "max":
        return int(np.argmax(values))
    if direction == "abs_min":
        return int(np.argmin(np.abs(values)))
    return int(np.argmin(values))


def format_tuning_table(entries: Sequence[Dict], k: int = 3) -> str:
    """Markdown Table 2 in the wheat report's shape.

    ``entries`` is ``[{"model": "best epoch 4", "config": "NMS 1.5", "tau": 0.25,
    "metrics": <evaluate/sweep_tau summary>}, ...]``. The best value in each
    column is bolded (lowest MAE/RMSE, lowest |bias|, highest P/R/F1) and the
    decode settings go in a footnote rather than extra columns, so the table
    stays comparable with the wheat one.
    """
    headers = ["Model", "Config"] + [label for _, label, _ in TUNING_COLUMNS]
    cells: List[List[str]] = [[e["model"], e["config"]] for e in entries]
    for key, _, direction in TUNING_COLUMNS:
        values = [float(e["metrics"][key]) for e in entries]
        best = _best_index(values, direction) if values else -1
        for i, value in enumerate(values):
            text = f"{value:+.3f}" if key == "count_bias" else f"{value:.3f}"
            cells[i].append(f"**{text}**" if i == best else text)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("-" * max(len(h), 3) for h in headers) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in cells]

    taus = [float(e["tau"]) for e in entries]
    if len(set(taus)) == 1:
        footnote = f"All use `tau` {taus[0]:.2f} | k {k}"
    else:
        per_row = ", ".join(f"{e['tau']:.2f} ({e['model']} {e['config']})" for e in entries)
        footnote = f"`tau` per row: {per_row} | k {k}"
    return "\n".join(lines) + "\n" + footnote + "\n"


# --------------------------------------------------------------------------- #
# NMS spot checks
# --------------------------------------------------------------------------- #


def densest_indices(records: Sequence, n: int = 6) -> List[int]:
    """Indices of the ``n`` records with the most ground-truth points."""
    order = sorted(range(len(records)), key=lambda i: -len(records[i].points))
    return order[:n]


def plot_nms_spot_checks(
    model: torch.nn.Module,
    val_ds,
    device: torch.device,
    indices: Sequence[int],
    out_path: Path,
    tau: float,
    k: int = 3,
    output_stride: int = 4,
    radii: Sequence[float] = (1.5, 5.0),
    amp: bool = True,
) -> Path:
    """Ground truth beside the same frame decoded at each NMS radius.

    One row per frame, one column per radius; green dots are ground truth and red
    rings are predictions, with the counts in each panel title. The question this
    figure exists to answer is whether the wider radius is removing duplicate
    peaks on one smeared fish or merging two genuinely adjacent fish — which only
    an eye on the densest frames can settle.
    """
    from cropcounter.dinov3_pyramid import IMAGENET_MEAN, IMAGENET_STD

    mean, std = np.array(IMAGENET_MEAN), np.array(IMAGENET_STD)
    n_rows, n_cols = len(indices), len(radii)
    fig = _new_figure((7.0 * n_cols, 4.2 * n_rows))
    axes = np.atleast_2d(fig.subplots(n_rows, n_cols, squeeze=False))
    for row, index in enumerate(indices):
        item = val_ds[int(index)]
        record = val_ds.records[int(index)]
        rgb = (item["image"].permute(1, 2, 0).numpy() * std + mean).clip(0, 1)
        rgb = rgb[:record.height, :record.width]
        gt = np.asarray(item["points"], dtype=np.float32).reshape(-1, 2)
        for col, radius in enumerate(radii):
            points, _, _ = decode_image_points(
                model, item["image"], record.width, record.height, device,
                tau=tau, k=k, nms_radius=radius, output_stride=output_stride, amp=amp,
            )
            ax = axes[row, col]
            ax.imshow(rgb)
            ax.scatter(gt[:, 0], gt[:, 1], s=28, c="lime", alpha=0.85)
            if len(points):
                ax.scatter(points[:, 0], points[:, 1], s=70, facecolors="none",
                           edgecolors="red", linewidths=1.1)
            ax.set_title(
                f"{record.name[:46]}\nNMS {radius} · GT {len(gt)} · pred {len(points)}",
                fontsize=9,
            )
            ax.axis("off")
    fig.suptitle(
        f"NMS spot checks on the densest val frames at tau {tau:.2f} — "
        "green = ground truth, red = decoded points",
        fontsize=11,
    )
    return _save(fig, out_path)


# =========================================================================== #
# Notebooks 3 and 4: point-in-box scoring against BOX ground truth, the
# like-for-like table, the Fig-11/Fig-12 analogues and the spot-check grids.
#
# Everything below consumes the dicts ``point_in_box`` already produces
# (``score_dataset`` summaries and rows, ``sweep_thresholds`` lists) and turns
# them into a CSV, a markdown table or a PNG. Nothing re-implements a metric:
# the matcher stays Liam's verbatim copy in ``point_in_box.match_image``.
# =========================================================================== #
import csv  # noqa: E402  (section-local stdlib imports, kept beside their users)
import math  # noqa: E402
import re  # noqa: E402
from typing import Any, Callable, Iterable, Mapping  # noqa: E402

#: The threshold grid both notebooks sweep (0.05 … 0.95 inclusive, step 0.05).
#: Built from integers rather than with ``np.arange`` so the values are exactly
#: the printed decimals — ``np.arange`` yields 0.15000000000000002, and a JSON
#: written from it differs between numpy versions.
SWEEP_THRESHOLDS: Tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))

#: One colour per system, used by EVERY figure so a colour means the same thing
#: in all of them. Ground truth is always green, the point head red, the box-head
#: centres orange and RF-DETR blue — the scheme the box memo's spot checks used,
#: so the two sets of figures can be read side by side.
SYSTEM_COLOURS: Dict[str, str] = {
    "GT": "#32cd32",
    "point head best": "#d62728",
    "point head last": "#9467bd",
    "box head best (centres)": "#ff7f0e",
    "box head last (centres)": "#8c564b",
    "RF-DETR-Nano (centres)": "#1f77b4",
}

#: GT-count buckets for the Fig-12 analogue: ``(low, high inclusive or None, label)``.
COUNT_BUCKETS: Tuple[Tuple[int, Optional[int], str], ...] = (
    (0, 0, "0"),
    (1, 1, "1"),
    (2, 2, "2"),
    (3, 5, "3-5"),
    (6, None, "6+"),
)

#: Keys whose scalar value is read as a calibrated threshold by the tau reader.
_TAU_KEYS = ("tau", "best_tau", "calibrated_tau", "tau_calibrated", "chosen_tau", "tau_mae")
_CKPT_KEYS = ("checkpoint", "ckpt", "weights")
_NMS_KEYS = ("nms_radius", "nms", "nms_rad")


# --------------------------------------------------------------------------- #
# tau_calibration.json — read defensively; notebook 2 owns its schema
# --------------------------------------------------------------------------- #


def _as_nms(value: Any) -> Optional[float]:
    """Coerce ``nms_1.5`` / ``nms5`` / ``"1.5"`` / ``5`` (key or field) to a float."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.fullmatch(r"(?:nms[_\-]?)?(\d+(?:\.\d+)?)", str(value).strip(), re.IGNORECASE)
    return float(match.group(1)) if match else None


def _as_ckpt(value: Any) -> Optional[str]:
    """Coerce ``best`` / ``"last.pt"`` / ``best_tau`` (key or field) to ``best``/``last``."""
    text = str(value).strip().lower()
    for name in ("best", "last"):
        if text == name or text.startswith((f"{name}.", f"{name}_")):
            return name
    return None


def _scan_tau(
    node: Any,
    ckpt: Optional[str],
    nms: Optional[float],
    found: List[Tuple[Optional[str], Optional[float], float]],
) -> None:
    """Walk any JSON shape collecting ``(checkpoint, nms_radius, tau)`` triples.

    Enclosing keys are hints: a ``best`` / ``last`` key sets the checkpoint for
    everything under it, an ``nms_1.5`` key sets the radius. Fields named inside
    a record (``checkpoint``, ``nms_radius``) override the hints for that record.
    """
    if isinstance(node, Mapping):
        own_ckpt = next((_as_ckpt(node[k]) for k in _CKPT_KEYS if k in node), None) or ckpt
        own_nms = next((_as_nms(node[k]) for k in _NMS_KEYS if k in node), None)
        own_nms = own_nms if own_nms is not None else nms
        for key, value in node.items():
            if key in _TAU_KEYS and isinstance(value, (int, float)) and not isinstance(value, bool):
                found.append((own_ckpt, own_nms, float(value)))
                continue
            hint_nms = _as_nms(key)
            _scan_tau(
                value,
                _as_ckpt(key) or own_ckpt,
                hint_nms if hint_nms is not None else own_nms,
                found,
            )
    elif isinstance(node, (list, tuple)):
        for value in node:
            _scan_tau(value, ckpt, nms, found)


def read_tau_calibration(
    path,
    checkpoints: Sequence[str] = ("best", "last"),
    nms_radii: Sequence[float] = (1.5, 5.0),
    default_tau: float = 0.3,
) -> Tuple[Dict[str, Dict[float, float]], List[str]]:
    """Read ``tau_calibration.json`` without assuming its exact schema.

    Notebook 2 writes that file and owns its shape, so this accepts anything that
    could carry a calibrated threshold — ``{"best": {"nms_1.5": 0.35}}``,
    ``{"last": {"tau": …}}``, ``[{"checkpoint": "best", "nms_radius": 5, "tau": …}]``,
    a bare top-level ``best_tau`` — and resolves each requested
    ``(checkpoint, nms_radius)`` from the most specific match it can find.

    Returns:
        ``(tau, notes)``. ``tau[checkpoint][nms_radius]`` is a float for every
        requested pair. ``notes`` records where each value came from — PRINT THEM:
        a fallback to ``default_tau`` is a loud warning, never a silent
        substitution, because a wrong headline tau invalidates the table.
    """
    path = Path(path)
    notes: List[str] = []
    found: List[Tuple[Optional[str], Optional[float], float]] = []
    if path.is_file():
        try:
            _scan_tau(json.loads(path.read_text(encoding="utf-8")), None, None, found)
            notes.append(
                f"read {path}: {len(found)} candidate tau value(s) -> "
                + ", ".join(f"({c}, nms={n}, tau={t})" for c, n, t in found[:12])
            )
        except (OSError, ValueError) as exc:
            notes.append(f"!! WARNING: {path} unreadable ({exc!r}); falling back to config tau")
    else:
        notes.append(
            f"!! WARNING: {path} is MISSING — notebook 2 has not written it, so every "
            f"threshold below falls back to the config tau {default_tau}."
        )

    tau: Dict[str, Dict[float, float]] = {}
    for ckpt in checkpoints:
        tau[ckpt] = {}
        for nms in nms_radii:
            exact = [t for c, n, t in found if c == ckpt and n is not None and abs(n - nms) < 1e-9]
            no_nms = [t for c, n, t in found if c == ckpt and n is None]
            any_nms = [t for c, _, t in found if c == ckpt]
            no_ckpt = [t for c, _, t in found if c is None]
            for values, why in (
                (exact, f"exact match ({ckpt}, nms {nms})"),
                (no_nms, f"{ckpt}, no nms recorded"),
                (any_nms, f"{ckpt}, a different nms"),
                (no_ckpt, "file-level tau, no checkpoint recorded"),
            ):
                if values:
                    tau[ckpt][nms] = float(values[0])
                    notes.append(f"   tau[{ckpt}][nms {nms}] = {values[0]:.3f}   ({why})")
                    break
            else:
                tau[ckpt][nms] = float(default_tau)
                notes.append(
                    f"!! WARNING: no calibrated tau for ({ckpt}, nms {nms}); using the "
                    f"config default {default_tau}"
                )
    return tau, notes


# --------------------------------------------------------------------------- #
# Joining the three id spaces: points-root int id, CFD string id, file name
# --------------------------------------------------------------------------- #


def name_maps(bbox_document: Mapping) -> Tuple[Dict[Any, str], Dict[str, Any]]:
    """``(id -> file_name, file_name -> id)`` from a fetched CFD bbox document.

    After ``cfd fetch`` the ``file_name`` is the on-disk basename, and the bbox
    document's ``id`` is the CFD **string** id that the box-head and RF-DETR COCO
    results are keyed by. These two dicts are what make all five prediction
    systems and the GT keyable identically (by file name).
    """
    id_to_name = {img["id"]: Path(str(img["file_name"])).name
                  for img in bbox_document.get("images", ())}
    name_to_id = {name: image_id for image_id, name in id_to_name.items()}
    return id_to_name, name_to_id


def sequence_by_name(bbox_document: Mapping) -> Dict[str, str]:
    """``file_name -> cfd_sequence`` — the clip a val frame belongs to."""
    return {
        Path(str(img["file_name"])).name: str(
            img.get("cfd_sequence") or img.get("original_data_source") or "unknown"
        )
        for img in bbox_document.get("images", ())
    }


def rekey_by_name(by_id: Mapping, id_to_name: Mapping[Any, str]) -> Dict[str, Any]:
    """Re-key a ``{image id: value}`` mapping to ``{file name: value}``.

    Raises:
        KeyError: if an id is not in ``id_to_name`` — a silent drop here is the
            classic way to score a detector on fewer frames than the GT has.
    """
    out: Dict[str, Any] = {}
    for image_id, value in by_id.items():
        if image_id not in id_to_name:
            raise KeyError(f"image id {image_id!r} is not in the bbox document's images")
        out[id_to_name[image_id]] = value
    return out


# --------------------------------------------------------------------------- #
# Tables and CSVs
# --------------------------------------------------------------------------- #


def write_counts_csv(rows: Iterable[Mapping], path) -> Path:
    """Write the ``image_name,predicted_count`` CSV notebook 3 ships."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_name", "predicted_count"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "image_name": row["image_name"],
                "predicted_count": int(row["predicted_count"]),
            })
    return path


def _format_cell(value: Any, spec: Optional[str]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    try:
        return format(value, spec) if spec else str(value)
    except (TypeError, ValueError):
        return str(value)


def markdown_table(
    rows: Sequence[Mapping],
    columns: Sequence[Tuple[str, str, Optional[str]]],
    footnotes: Sequence[str] = (),
) -> str:
    """Render rows as markdown. ``columns`` is ``(row key, header, format spec)``."""
    lines = [
        "| " + " | ".join(header for _, header, _ in columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(_format_cell(row.get(key), spec) for key, _, spec in columns) + " |"
        )
    if footnotes:
        lines.append("")
        lines.extend(footnotes)
    return "\n".join(lines)


def write_results_csv(
    rows: Sequence[Mapping],
    path,
    columns: Optional[Sequence[Tuple[str, str, Optional[str]]]] = None,
) -> Path:
    """Write the results table as CSV; column order follows ``columns`` when given."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        fieldnames = [key for key, _, _ in columns]
    else:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def best_by(sweep: Sequence[Mapping], key: str, largest: bool) -> Dict:
    """Pick one summary out of a ``sweep_thresholds`` list; ties go to the LOWER tau.

    Identical rule to ``scripts/rescore_boxes_as_points.py``, so the point-head
    rows and the already-published box rows are selected the same way.
    """
    ordered = sorted(sweep, key=lambda s: (-s[key] if largest else s[key], s["conf_thr"]))
    return dict(ordered[0])


# --------------------------------------------------------------------------- #
# Breakdowns — the WDA replacement, because Brackish is ONE domain
# --------------------------------------------------------------------------- #


def count_bucket(n_gt: int) -> str:
    """Bucket a GT count into one of :data:`COUNT_BUCKETS`' labels."""
    for low, high, label in COUNT_BUCKETS:
        if n_gt >= low and (high is None or n_gt <= high):
            return label
    return COUNT_BUCKETS[-1][2]


def group_accuracy(
    rows: Iterable[Mapping],
    group_of: Callable[[Mapping], str],
) -> Dict[str, Dict[str, float]]:
    """Mean per-image accuracy within each group, from ``score_dataset`` rows.

    Returns ``{group: {"n_images", "mean_accuracy", "std_accuracy", "n_gt"}}``.
    Liam's weighted domain accuracy IS this, averaged over domains; with one
    domain it collapses to the overall mean, so grouping by clip and by GT count
    is what puts a variance back underneath the single headline number.
    """
    buckets: Dict[str, List[Tuple[float, int]]] = {}
    for row in rows:
        buckets.setdefault(group_of(row), []).append(
            (float(row["accuracy"]), int(row.get("gt_count", 0)))
        )
    out: Dict[str, Dict[str, float]] = {}
    for group, values in buckets.items():
        accuracies = np.asarray([a for a, _ in values], dtype=float)
        out[group] = {
            "n_images": int(len(accuracies)),
            "mean_accuracy": float(accuracies.mean()),
            "std_accuracy": float(accuracies.std()),
            "n_gt": int(sum(g for _, g in values)),
        }
    return out


def bucket_order(groups: Iterable[str]) -> List[str]:
    """Count-bucket labels in declared order, then anything else alphabetically."""
    groups = list(groups)
    declared = [label for _, _, label in COUNT_BUCKETS]
    return ([g for g in declared if g in groups]
            + sorted(g for g in groups if g not in declared))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def plot_accuracy_vs_threshold(
    sweeps: Mapping[str, Sequence[Mapping]],
    out_path: Path,
    marked: Optional[Mapping[str, float]] = None,
    metric: str = "mean_accuracy",
    ylabel: str = "mean per-image accuracy  tp/(tp+fp+fn)",
    title: str = "Accuracy vs confidence threshold — CFD Brackish val (3,127 frames)",
) -> Path:
    """Fig-11 analogue: one accuracy-vs-tau curve per system on one axis.

    ``marked`` draws a dashed vertical at a system's headline operating point (the
    calibrated tau for the point-head rows), in that system's own colour.
    """
    fig = _new_figure((9.5, 5.2))
    ax = fig.subplots()
    for index, (label, sweep) in enumerate(sweeps.items()):
        ax.plot(
            [s["conf_thr"] for s in sweep], [s[metric] for s in sweep],
            "o-", ms=3.5, lw=1.6, label=label,
            color=SYSTEM_COLOURS.get(label, f"C{index}"),
        )
    for label, tau in (marked or {}).items():
        colour = SYSTEM_COLOURS.get(label, "gray")
        ax.axvline(tau, ls="--", lw=1.1, color=colour, alpha=0.85)
        ax.annotate(f"{label} · calibrated τ={tau:.2f}", (tau, 0.02), fontsize=7, rotation=90,
                    va="bottom", ha="right", color=colour)
    ax.set_xlabel("confidence threshold τ")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    # "best" rather than a fixed corner: five curves plus the calibrated-tau annotations leave
    # no corner reliably free, and a legend sitting on top of the curves is the one thing this
    # figure cannot afford.
    ax.legend(fontsize=8, loc="best")
    return _save(fig, out_path, dpi=120)


def plot_grouped_bars(
    groups: Sequence[str],
    series: Mapping[str, Sequence[float]],
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str = "mean per-image accuracy",
    annotations: Sequence[str] = (),
    figsize: Tuple[float, float] = (13.0, 5.2),
    hline: Optional[Tuple[float, str]] = None,
) -> Path:
    """Fig-12 analogue: grouped bars, one bar per system inside each group."""
    fig = _new_figure(figsize)
    ax = fig.subplots()
    n_series = max(len(series), 1)
    width = 0.8 / n_series
    centres = np.arange(len(groups), dtype=float)
    for index, (label, values) in enumerate(series.items()):
        ax.bar(centres + (index - (n_series - 1) / 2) * width, list(values), width * 0.92,
               label=label, color=SYSTEM_COLOURS.get(label, f"C{index}"))
    if hline is not None:
        ax.axhline(hline[0], ls="--", lw=1.1, color="black", alpha=0.6, label=hline[1])
    ax.set_xticks(centres)
    ax.set_xticklabels(
        [f"{g}\n{a}" for g, a in zip(groups, annotations)] if annotations else list(groups),
        fontsize=8, rotation=45, ha="right",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, out_path, dpi=120)


def save_overlay_grid(
    frames: Sequence[Mapping],
    out_path: Path,
    ncols: int = 2,
    panel: Tuple[float, float] = (8.4, 5.0),
    suptitle: Optional[str] = None,
) -> Path:
    """Draw several systems' points over val frames, one panel per frame.

    Each frame is a mapping with:

    * ``image_path`` — the file to ``imshow``
    * ``title`` — panel title
    * ``boxes`` — optional ``(M, 4)`` xyxy GT boxes, drawn in the GT colour
    * ``points`` — sequence of ``(label, (N, 2) xy)``; the colour comes from
      :data:`SYSTEM_COLOURS` and the marker shape cycles, so two systems that
      agree on a fish remain separately visible instead of one hiding the other.
    """
    from matplotlib.patches import Patch, Rectangle

    markers = ("o", "s", "^", "D", "v", "P")
    frames = list(frames)
    nrows = max(1, -(-len(frames) // ncols))
    fig = _new_figure((panel[0] * ncols, panel[1] * nrows))
    axes = np.atleast_1d(np.asarray(fig.subplots(nrows, ncols, squeeze=False))).ravel()
    legend_for: Dict[str, str] = {}
    for ax, frame in zip(axes, frames):
        ax.imshow(Image.open(frame["image_path"]))
        boxes = np.asarray(frame.get("boxes", np.zeros((0, 4))), dtype=float).reshape(-1, 4)
        if len(boxes):
            legend_for["GT box"] = SYSTEM_COLOURS["GT"]
        for x0, y0, x1, y1 in boxes:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor=SYSTEM_COLOURS["GT"], linewidth=1.4))
        for index, (label, points) in enumerate(frame.get("points", ())):
            points = np.asarray(points, dtype=float).reshape(-1, 2)
            colour = SYSTEM_COLOURS.get(label, f"C{index}")
            legend_for[label] = colour
            if len(points):
                ax.scatter(points[:, 0], points[:, 1], s=70 + 26 * index, facecolors="none",
                           edgecolors=colour, linewidths=1.5, marker=markers[index % len(markers)])
        ax.set_title(frame.get("title", ""), fontsize=8)
        ax.axis("off")
    for ax in axes[len(frames):]:
        ax.axis("off")
    if legend_for:
        fig.legend(
            handles=[Patch(edgecolor=colour, facecolor="none", label=label)
                     for label, colour in legend_for.items()],
            loc="lower center", ncol=min(5, len(legend_for)), fontsize=8,
        )
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    return _save(fig, out_path, dpi=110)


def pick_informative_frames(
    gt_by_image: Mapping,
    preds_by_system: Mapping[str, Mapping],
    tau: Mapping[str, float],
    n: int = 6,
    n_dense: int = 3,
) -> List[str]:
    """Pick frames worth looking at: the densest, then the biggest disagreements.

    Disagreement is the spread of kept-prediction counts across systems, each at
    its own tau — i.e. the frames where the choice of head actually changes the
    answer, which is what a spot check should be asked to explain. Fully
    deterministic (sorted, with the image key as the final tiebreak), so a re-run
    produces the same figure.
    """
    def kept(system: str, key) -> int:
        array = np.asarray(
            preds_by_system[system].get(key, np.zeros((0, 3))), dtype=float
        ).reshape(-1, 3)
        return int((array[:, 2] >= tau.get(system, 0.0)).sum()) if len(array) else 0

    keys = list(gt_by_image)
    dense = sorted(keys, key=lambda k: (-len(gt_by_image[k]), str(k)))[:n_dense]
    spread = []
    for key in keys:
        counts = [kept(system, key) for system in preds_by_system]
        spread.append((max(counts) - min(counts) if counts else 0, len(gt_by_image[key]), str(key)))
    disagree = [k for _, _, k in sorted(spread, key=lambda t: (-t[0], -t[1], t[2]))
                if k not in dense]
    return (dense + disagree)[:n]
