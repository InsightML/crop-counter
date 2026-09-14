"""Counting and localization metrics for point-decoded predictions.

Counting quality is MAE/RMSE on per-image counts; localization quality is
precision/recall/F1 from a Hungarian match between predicted and ground
truth points within a pixel radius. Both are reported because a good count
with poor localization means compensating errors, not a good model.

Multiclass: every metric is computed per class (predictions of class ``c``
are matched only against ground truth of class ``c``) and reported under
``summary["per_class"][name]``; the top-level keys are the **macro** mean
over classes. With one class the two coincide, so single-class numbers are
unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm

from .crop_dataset import WILDCARD_CLASS
from .inference import decode_classes
from .losses import penalty_reduced_focal_loss

_COUNT_KEYS = ("n_gt", "n_pred", "tp", "fp", "fn")
_METRIC_KEYS = ("count_mae", "count_rmse", "count_bias", "precision", "recall", "f1")

_UNMATCHABLE = 1e9


def match_points(
    pred: np.ndarray,
    gt: np.ndarray,
    radius_px: float,
) -> Tuple[int, int, int]:
    """Match predicted to ground-truth points within a distance gate.

    Args:
        pred: (N, 2) predicted (x, y) locations in image pixels.
        gt: (M, 2) ground-truth locations in image pixels.
        radius_px: maximum centre distance for a valid match.

    Returns:
        (tp, fp, fn) — Hungarian-optimal one-to-one matches within the
        radius count as true positives.
    """
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 2)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 2)
    if len(pred) == 0 or len(gt) == 0:
        return 0, len(pred), len(gt)

    dist = cdist(pred, gt)
    cost = np.where(dist <= radius_px, dist, _UNMATCHABLE)
    rows, cols = linear_sum_assignment(cost)
    tp = int((dist[rows, cols] <= radius_px).sum())
    return tp, len(pred) - tp, len(gt) - tp


def _class_row(
    pred: np.ndarray,
    pred_ids: np.ndarray,
    gt: np.ndarray,
    gt_ids: np.ndarray,
    class_names: Sequence[str],
    radius_px: float,
) -> Dict[str, Any]:
    """Per-class counts + matches for one image, plus their totals."""
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 2)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 2)
    pred_ids = np.asarray(pred_ids, dtype=np.int64).reshape(-1)
    gt_ids = np.asarray(gt_ids, dtype=np.int64).reshape(-1)
    per_class: Dict[str, Dict[str, int]] = {}
    for c, name in enumerate(class_names):
        p, g = pred[pred_ids == c], gt[gt_ids == c]
        tp, fp, fn = match_points(p, g, radius_px)
        per_class[name] = {"n_gt": len(g), "n_pred": len(p), "tp": tp, "fp": fp, "fn": fn}
    totals = {key: sum(v[key] for v in per_class.values()) for key in _COUNT_KEYS}
    return {**totals, "per_class": per_class}


def _summarise_counts(rows: Iterable[Dict[str, int]]) -> Dict[str, float]:
    """Counting + localization metrics over per-image count/match rows.

    Stays finite for a class with no ground truth or no predictions: the
    count errors reduce to the raw counts and precision/recall fall to 0
    through the ``max(., 1)`` guards.
    """
    rows = list(rows)
    n_gt = np.array([r["n_gt"] for r in rows], dtype=np.float64)
    n_pred = np.array([r["n_pred"] for r in rows], dtype=np.float64)
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    err = n_pred - n_gt
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "count_mae": float(np.abs(err).mean()),
        "count_rmse": float(np.sqrt((err ** 2).mean())),
        "count_bias": float(err.mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _summarise(rows: List[Dict], class_names: Sequence[str]) -> Dict[str, Any]:
    """Aggregate per-image rows into a metrics summary.

    ``per_class[name]`` holds each class's own metrics; the top-level metric
    keys are their unweighted (macro) mean over classes.
    """
    per_class = {
        name: _summarise_counts(r["per_class"][name] for r in rows) for name in class_names
    }
    summary: Dict[str, Any] = {
        key: float(np.mean([per_class[name][key] for name in class_names]))
        for key in _METRIC_KEYS
    }
    summary["n_images"] = len(rows)
    summary["per_class"] = per_class
    return summary


def _iter_prob_maps(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    focal_alpha: float = 2.0,
    focal_beta: float = 4.0,
    progress: bool = False,
    desc: str = "val",
) -> Iterable[Tuple[torch.Tensor, np.ndarray, np.ndarray, str, Optional[float]]]:
    """Yield (prob_map, gt_points, gt_class_ids, name, loss) per validation image.

    ``gt_class_ids`` is the batch's ``class_ids`` (all zeros when the loader
    carries none — a single-class dataset). ``loss`` is the penalty-reduced
    focal loss against the batch's heatmap target (tau-independent), or
    ``None`` when the loader carries no target. Set ``progress`` to show a
    per-image tqdm bar labelled ``desc``.
    """
    model.eval()
    iterator = loader
    if progress:
        try:
            total = len(loader)
        except TypeError:  # pragma: no cover - loader without __len__
            total = None
        iterator = tqdm(loader, total=total, desc=desc, leave=False)
    with torch.no_grad():
        for batch in iterator:
            image = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(image)
            logits = logits.float()
            loss: Optional[float] = None
            if batch.get("target") is not None:
                target = batch["target"].to(device, non_blocking=True)
                loss = penalty_reduced_focal_loss(
                    logits, target, alpha=focal_alpha, beta=focal_beta
                ).item()
            prob = torch.sigmoid(logits).cpu()
            gt_points = np.asarray(batch["points"], dtype=np.float32).reshape(-1, 2)
            gt_ids = batch.get("class_ids")
            if gt_ids is None:
                gt_ids = np.zeros(len(gt_points), dtype=np.int64)
            yield prob, gt_points, np.asarray(gt_ids, dtype=np.int64).reshape(-1), batch["name"], loss


def evaluate(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    tau: Union[float, Dict[str, float]],
    k: Union[int, Dict[str, int]] = 3,
    nms_radius: Union[float, Dict[str, float]] = 1.5,
    output_stride: int = 4,
    match_radius_px: float = 24.0,
    focal_alpha: float = 2.0,
    focal_beta: float = 4.0,
    progress: bool = False,
    desc: str = "val",
    class_names: Sequence[str] = (WILDCARD_CLASS,),
) -> Tuple[Dict[str, Any], List[Dict]]:
    """Evaluate counting + localization over a whole-image val loader.

    Each class channel is decoded with its own ``tau`` / ``k`` /
    ``nms_radius`` (scalar = same for all classes, dict = per class) and
    matched against the ground truth of that class. Also averages the
    penalty-reduced focal loss against the heatmap targets, a
    tau-independent measure of heatmap fidelity, reported as ``val_loss``.
    Set ``progress`` to show a per-image tqdm bar labelled ``desc``.

    Returns:
        (summary, per_image_rows). Summary keys: count_mae, count_rmse,
        count_bias, precision, recall, f1 (each the macro mean over
        classes), n_images, val_loss, and ``per_class`` — the same six
        metrics for every class name. Rows carry the per-image totals
        (n_gt, n_pred, tp, fp, fn) plus a ``per_class`` breakdown.
    """
    class_names = tuple(class_names)
    rows: List[Dict] = []
    losses: List[float] = []
    for prob, gt, gt_ids, name, loss in _iter_prob_maps(
        model, loader, device, focal_alpha=focal_alpha, focal_beta=focal_beta,
        progress=progress, desc=desc,
    ):
        pred, _, pred_ids = decode_classes(
            prob, class_names, tau=tau, k=k, nms_radius=nms_radius, output_stride=output_stride
        )
        rows.append({
            "name": name,
            **_class_row(pred, pred_ids, gt, gt_ids, class_names, match_radius_px),
        })
        if loss is not None:
            losses.append(loss)
    summary = _summarise(rows, class_names)
    summary["val_loss"] = float(np.mean(losses)) if losses else float("nan")
    return summary, rows


def sweep_tau(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    taus: Sequence[float],
    k: Union[int, Dict[str, int]] = 3,
    nms_radius: Union[float, Dict[str, float]] = 1.5,
    output_stride: int = 4,
    match_radius_px: float = 24.0,
    class_names: Sequence[str] = (WILDCARD_CLASS,),
) -> List[Dict[str, Any]]:
    """Sweep the decode threshold tau over the val set in one model pass.

    Each image's probability map is computed once and decoded at every tau,
    so the sweep costs one forward pass plus cheap decodes. Every tau is
    applied to all classes at once; because channels decode independently,
    each class's best tau can be read off its own ``per_class`` entry — no
    joint search is needed.

    Returns:
        One summary dict per tau (with a "tau" key added), same keys as
        ``evaluate`` (including ``per_class``).
    """
    class_names = tuple(class_names)
    per_tau: List[List[Dict]] = [[] for _ in taus]
    for prob, gt, gt_ids, name, _ in _iter_prob_maps(model, loader, device):
        for i, tau in enumerate(taus):
            pred, _, pred_ids = decode_classes(
                prob, class_names, tau=float(tau), k=k, nms_radius=nms_radius,
                output_stride=output_stride,
            )
            per_tau[i].append({
                "name": name,
                **_class_row(pred, pred_ids, gt, gt_ids, class_names, match_radius_px),
            })
    summaries = []
    for tau, rows in zip(taus, per_tau):
        summary = _summarise(rows, class_names)
        summary["tau"] = float(tau)
        summaries.append(summary)
    return summaries
