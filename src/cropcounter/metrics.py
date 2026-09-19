"""Counting and localization metrics for point-decoded predictions.

Counting quality is MAE/RMSE on per-image counts; localization quality is
precision/recall/F1 from a Hungarian match between predicted and ground
truth points within a pixel radius. Both are reported because a good count
with poor localization means compensating errors, not a good model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm

from .dinov3_pyramid import autocast_context
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
    progress: bool = False,
    desc: str = "val",
) -> Iterable[Tuple[torch.Tensor, np.ndarray, str, Optional[float]]]:
    """Yield (prob_map, gt_points, name, loss) per validation image.

    ``loss`` is the penalty-reduced focal loss against the batch's heatmap
    target (tau-independent), or ``None`` when the loader carries no target.
    Set ``progress`` to show a per-image tqdm bar labelled ``desc``.
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
            with autocast_context(device):
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
    progress: bool = False,
    desc: str = "val",
    detect_tau: Optional[float] = None,
) -> Tuple[Dict[str, float], List[Dict]]:
    """Evaluate counting + localization over a whole-image val loader.

    Also averages the penalty-reduced focal loss against the heatmap targets,
    a tau-independent measure of heatmap fidelity, reported as ``val_loss``.
    Set ``progress`` to show a per-image tqdm bar labelled ``desc``.

    Args:
        detect_tau: when given, each row also carries ``"detections"`` — an
            ``(N, 3)`` list of ``[x, y, score]`` decoded at THIS threshold
            rather than at ``tau``. Set it low (0.01, as the box task's
            ``ap_tau`` is) and every operating threshold above it can be swept
            offline as a mask over a detection set already on disk, which is
            what makes point scoring a CPU-only job after the box is gone.
            The metrics in ``summary`` are unaffected — they are always decoded
            at ``tau``.

    Returns:
        (summary, per_image_rows). Summary keys: count_mae, count_rmse,
        count_bias, precision, recall, f1, n_images, val_loss.
    """
    rows: List[Dict] = []
    losses: List[float] = []
    for prob, gt, name, loss in _iter_prob_maps(
        model, loader, device, focal_alpha=focal_alpha, focal_beta=focal_beta,
        progress=progress, desc=desc,
    ):
        row: Dict = {"name": name}
        if detect_tau is not None:
            # Decoded once at the low threshold; the operating-point decode
            # below is a strict subset of it, so this costs no extra forward.
            det_pts, det_scores = decode_peaks(
                prob, k=k, tau=detect_tau, nms_radius=nms_radius, stride=output_stride
            )
            row["detections"] = [
                [float(x), float(y), float(score)]
                for (x, y), score in zip(det_pts, det_scores)
            ]
        pred, _ = decode_peaks(prob, k=k, tau=tau, nms_radius=nms_radius, stride=output_stride)
        tp, fp, fn = match_points(pred, gt, match_radius_px)
        row.update({
            "n_gt": len(gt), "n_pred": len(pred), "tp": tp, "fp": fp, "fn": fn,
        })
        rows.append(row)
        if loss is not None:
            losses.append(loss)
    summary = _summarise(rows)
    summary["val_loss"] = float(np.mean(losses)) if losses else float("nan")
    return summary, rows


def write_point_results(
    rows: Sequence[Dict],
    path: Path,
    *,
    detect_tau: float,
    output_stride: int,
    nms_radius: float,
    k: int,
) -> Path:
    """Write a point run's per-image detections — the box task's results file.

    Keyed by image **file name**, not COCO image id: the point loader carries
    the name and the ids differ between a ``cfd points`` root (renumbered ints)
    and the bbox root the scoring joins against (CFD's original strings). File
    names are shared between the two and unique within a split — one flat
    ``images/`` directory per split makes them so — which is what lets a point
    run be scored against box ground truth without a second id map.

    The decode settings ride along in the header because a threshold sweep over
    this file is only honest if the reader knows the floor it was cut at.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "detect_tau": float(detect_tau),
        "output_stride": int(output_stride),
        "nms_radius": float(nms_radius),
        "k": int(k),
        "points": {
            str(row["name"]): row.get("detections", []) for row in rows
        },
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


def read_point_results(path: Path) -> Dict[str, np.ndarray]:
    """Read :func:`write_point_results` back as ``{name: (N, 3) x/y/score}``."""
    with Path(path).open(encoding="utf-8") as fh:
        payload = json.load(fh)
    return {
        name: np.asarray(points, dtype=float).reshape(-1, 3)
        for name, points in payload["points"].items()
    }


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
