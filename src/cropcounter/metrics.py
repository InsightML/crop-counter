"""Counting and localization metrics for point-decoded predictions.

Counting quality is MAE/RMSE on per-image counts; localization quality is
precision/recall/F1 from a Hungarian match between predicted and ground
truth points within a pixel radius. Both are reported because a good count
with poor localization means compensating errors, not a good model.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .heatmap import decode_peaks
from .losses import penalty_reduced_focal_loss

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


def _summarise(rows: List[Dict]) -> Dict[str, float]:
    """Aggregate per-image rows into a metrics summary."""
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
        "n_images": len(rows),
    }


def _iter_prob_maps(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    focal_alpha: float = 2.0,
    focal_beta: float = 4.0,
) -> Iterable[Tuple[torch.Tensor, np.ndarray, str, Optional[float]]]:
    """Yield (prob_map, gt_points, name, loss) per validation image.

    ``loss`` is the penalty-reduced focal loss against the batch's heatmap
    target (tau-independent), or ``None`` when the loader carries no target.
    """
    model.eval()
    with torch.no_grad():
        for batch in loader:
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
            yield prob, batch["points"], batch["name"], loss


def evaluate(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    tau: float,
    k: int = 3,
    nms_radius: float = 1.5,
    output_stride: int = 4,
    match_radius_px: float = 24.0,
    focal_alpha: float = 2.0,
    focal_beta: float = 4.0,
) -> Tuple[Dict[str, float], List[Dict]]:
    """Evaluate counting + localization over a whole-image val loader.

    Also averages the penalty-reduced focal loss against the heatmap targets,
    a tau-independent measure of heatmap fidelity, reported as ``val_loss``.

    Returns:
        (summary, per_image_rows). Summary keys: count_mae, count_rmse,
        count_bias, precision, recall, f1, n_images, val_loss.
    """
    rows: List[Dict] = []
    losses: List[float] = []
    for prob, gt, name, loss in _iter_prob_maps(
        model, loader, device, focal_alpha=focal_alpha, focal_beta=focal_beta
    ):
        pred, _ = decode_peaks(prob, k=k, tau=tau, nms_radius=nms_radius, stride=output_stride)
        tp, fp, fn = match_points(pred, gt, match_radius_px)
        rows.append({
            "name": name, "n_gt": len(gt), "n_pred": len(pred),
            "tp": tp, "fp": fp, "fn": fn,
        })
        if loss is not None:
            losses.append(loss)
    summary = _summarise(rows)
    summary["val_loss"] = float(np.mean(losses)) if losses else float("nan")
    return summary, rows


def sweep_tau(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    taus: Sequence[float],
    k: int = 3,
    nms_radius: float = 1.5,
    output_stride: int = 4,
    match_radius_px: float = 24.0,
) -> List[Dict[str, float]]:
    """Sweep the decode threshold tau over the val set in one model pass.

    Each image's probability map is computed once and decoded at every tau,
    so the sweep costs one forward pass plus cheap decodes.

    Returns:
        One summary dict per tau (with a "tau" key added), same keys as
        ``evaluate``.
    """
    per_tau: List[List[Dict]] = [[] for _ in taus]
    for prob, gt, name, _ in _iter_prob_maps(model, loader, device):
        for i, tau in enumerate(taus):
            pred, _ = decode_peaks(prob, k=k, tau=tau, nms_radius=nms_radius, stride=output_stride)
            tp, fp, fn = match_points(pred, gt, match_radius_px)
            per_tau[i].append({
                "name": name, "n_gt": len(gt), "n_pred": len(pred),
                "tp": tp, "fp": fp, "fn": fn,
            })
    summaries = []
    for tau, rows in zip(taus, per_tau):
        summary = _summarise(rows)
        summary["tau"] = float(tau)
        summaries.append(summary)
    return summaries
