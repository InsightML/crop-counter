"""Detection metrics: COCO average precision plus the counting-style P/R/F1.

Two reporting axes, deliberately:

* **COCO AP** (``pycocotools``) — the number every published detector on this
  data quotes, computed by *their* code on *our* predictions. The same
  :func:`coco_eval` scores a baseline's ``results.json`` on the same images,
  so "we beat X" means a difference in the model, not in the scorer.
* **Matched P/R/F1 + count MAE** — the axis the point head is already reported
  on (:mod:`cropcounter.metrics`). Fish and wheat then sit in one table, which
  is the only way to say whether one head is doing the other's job better.

``pycocotools`` is an optional extra (``pip install 'cropcounter[detection]'``)
and is imported inside the functions that need it, so the IoU matcher and the
summariser work without it.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .boxmap import boxes_xywh_to_xyxy, boxes_xyxy_to_xywh, clip_boxes_xyxy, decode_boxes
from .metrics import match_points

#: COCOeval's 12 summary statistics, in the order ``COCOeval.stats`` holds them.
COCO_STAT_NAMES = (
    "ap", "ap50", "ap75", "ap_small", "ap_medium", "ap_large",
    "ar1", "ar10", "ar100", "ar_small", "ar_medium", "ar_large",
)

#: The single category id this repo's single-class detectors report under.
DEFAULT_CATEGORY_ID = 1


# --------------------------------------------------------------------------- #
# COCO results I/O
# --------------------------------------------------------------------------- #


def write_coco_results(detections: Sequence[Dict[str, Any]], path: Path) -> Path:
    """Write COCO-results entries to JSON — the file ``COCOeval`` ingests."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(list(detections), fh)
    return path


def read_coco_results(path: Path) -> List[Dict[str, Any]]:
    """Read COCO-results entries back, e.g. a baseline's ``results.json``."""
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# COCO average precision
# --------------------------------------------------------------------------- #


def _sorted_ids(ids: Iterable[Any]) -> List[Any]:
    """De-duplicate and (where the ids are mutually comparable) sort them."""
    unique = list(dict.fromkeys(ids))
    try:
        return sorted(unique)
    except TypeError:  # mixed int/str ids — order does not affect the result
        return unique


def _load_coco(gt: Union[Path, str, Dict[str, Any], Any]):
    """Coerce a path / in-memory COCO dict / ``COCO`` object into a ``COCO``."""
    from pycocotools.coco import COCO

    if isinstance(gt, COCO):
        return gt
    with contextlib.redirect_stdout(io.StringIO()):
        if isinstance(gt, (str, Path)):
            return COCO(str(gt))
        coco = COCO()
        coco.dataset = dict(gt)
        coco.createIndex()
    return coco


def coco_eval(
    gt: Union[Path, str, Dict[str, Any], Any],
    detections: Sequence[Dict[str, Any]],
    image_ids: Optional[Sequence[Any]] = None,
) -> Dict[str, float]:
    """Score COCO-results ``detections`` against ``gt`` with ``pycocotools``.

    Args:
        gt: a COCO annotation JSON path, an in-memory COCO dict, or an already
            built ``pycocotools.coco.COCO``.
        detections: standard COCO results entries —
            ``{"image_id", "category_id", "bbox": [x, y, w, h], "score"}``.
        image_ids: restrict evaluation to these images; defaults to every image
            in ``gt`` (so images we predicted nothing on still count as recall
            misses, which is the honest denominator).

    Returns:
        The 12 ``COCOeval`` statistics under readable names (see
        :data:`COCO_STAT_NAMES`). An empty detection list returns all zeros
        rather than pycocotools' ``-1`` sentinels or an exception — a model that
        predicts nothing scores zero, it does not crash the epoch.
    """
    from pycocotools.cocoeval import COCOeval

    if not detections:
        return {name: 0.0 for name in COCO_STAT_NAMES}

    coco_gt = _load_coco(gt)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes([dict(d) for d in detections])
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        # Ids pass through un-cast (COCOeval handles string ids); sorted only
        # for a stable evaluation order, and only when they are comparable.
        evaluator.params.imgIds = _sorted_ids(
            image_ids if image_ids is not None else coco_gt.getImgIds()
        )
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    # COCOeval reports -1 for a statistic with no eligible ground truth (e.g.
    # ap_small on a dataset with no small boxes); surface that as 0.0.
    return {
        name: float(max(value, 0.0))
        for name, value in zip(COCO_STAT_NAMES, evaluator.stats)
    }


# --------------------------------------------------------------------------- #
# IoU / centre matching, mirroring metrics.match_points
# --------------------------------------------------------------------------- #


def box_iou_matrix(pred_xyxy: np.ndarray, gt_xyxy: np.ndarray) -> np.ndarray:
    """(N, M) pairwise IoU between two xyxy box sets."""
    pred = np.asarray(pred_xyxy, dtype=np.float64).reshape(-1, 4)
    gt = np.asarray(gt_xyxy, dtype=np.float64).reshape(-1, 4)
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)), dtype=np.float64)

    lt = np.maximum(pred[:, None, :2], gt[None, :, :2])
    rb = np.minimum(pred[:, None, 2:], gt[None, :, 2:])
    inter_wh = np.clip(rb - lt, 0.0, None)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    area_pred = np.clip(pred[:, 2] - pred[:, 0], 0, None) * np.clip(pred[:, 3] - pred[:, 1], 0, None)
    area_gt = np.clip(gt[:, 2] - gt[:, 0], 0, None) * np.clip(gt[:, 3] - gt[:, 1], 0, None)
    union = area_pred[:, None] + area_gt[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def match_boxes_iou(
    pred_xyxy: np.ndarray,
    gt_xyxy: np.ndarray,
    iou_thr: float = 0.5,
) -> Tuple[int, int, int]:
    """Hungarian-optimal one-to-one box match above an IoU gate.

    The box analogue of :func:`cropcounter.metrics.match_points`: same shape,
    same ``(tp, fp, fn)`` return, so the two heads report on one axis.

    Args:
        pred_xyxy: (N, 4) predicted boxes in xyxy pixels.
        gt_xyxy: (M, 4) ground-truth boxes in xyxy pixels.
        iou_thr: minimum IoU for a pair to count as a true positive.

    Returns:
        ``(tp, fp, fn)``.
    """
    pred = np.asarray(pred_xyxy, dtype=np.float64).reshape(-1, 4)
    gt = np.asarray(gt_xyxy, dtype=np.float64).reshape(-1, 4)
    if len(pred) == 0 or len(gt) == 0:
        return 0, len(pred), len(gt)

    iou = box_iou_matrix(pred, gt)
    # Maximise IoU; pairs below the gate are made unattractive rather than
    # forbidden, so the assignment stays feasible on any shape.
    cost = np.where(iou >= iou_thr, -iou, 0.0)
    rows, cols = linear_sum_assignment(cost)
    tp = int((iou[rows, cols] >= iou_thr).sum())
    return tp, len(pred) - tp, len(gt) - tp


def match_box_centres(
    pred_xyxy: np.ndarray,
    gt_xyxy: np.ndarray,
    radius_px: float = 24.0,
) -> Tuple[int, int, int]:
    """Match boxes by centre distance, reusing the point head's matcher.

    The like-for-like comparison against the point model: it never predicts an
    extent, so an IoU gate would score it zero by construction.
    """
    pred = np.asarray(pred_xyxy, dtype=np.float64).reshape(-1, 4)
    gt = np.asarray(gt_xyxy, dtype=np.float64).reshape(-1, 4)
    pred_c = np.stack([(pred[:, 0] + pred[:, 2]) / 2, (pred[:, 1] + pred[:, 3]) / 2], axis=1)
    gt_c = np.stack([(gt[:, 0] + gt[:, 2]) / 2, (gt[:, 1] + gt[:, 3]) / 2], axis=1)
    return match_points(pred_c, gt_c, radius_px)


def summarise_boxes(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate per-image ``{n_gt, n_pred, tp, fp, fn}`` rows.

    Same keys as :func:`cropcounter.metrics._summarise` so box and point runs
    tabulate together.
    """
    if not rows:
        return {"count_mae": 0.0, "count_rmse": 0.0, "count_bias": 0.0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0, "n_images": 0}
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


# --------------------------------------------------------------------------- #
# Whole-val-set evaluation
# --------------------------------------------------------------------------- #


def evaluate_boxes(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    gt: Union[Path, str, Dict[str, Any], Any],
    ap_tau: float = 0.01,
    tau: float = 0.3,
    k: int = 3,
    top_k: int = 100,
    box_nms_iou: Optional[float] = None,
    output_stride: int = 4,
    size_parameterisation: str = "log",
    match_iou: float = 0.5,
    category_id: Any = DEFAULT_CATEGORY_ID,
    loss_fn: Optional[Callable[[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], float]] = None,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evaluate a box-task model over a whole-image validation loader.

    One forward pass per image feeds three things at once: the COCO detection
    list (decoded at ``ap_tau``, the low threshold AP needs for a full PR
    curve), the P/R/F1 + count row set (decoded at ``tau``, the operating
    threshold), and the tau-independent validation loss.

    Args:
        gt: the val split's COCO annotation JSON (path or loaded dict).
        ap_tau: decode threshold for the AP detection list.
        tau: decode threshold for P/R/F1 and counts.
        loss_fn: ``(outputs, targets) -> float``, normally
            :func:`cropcounter.train._batch_loss` bound to the run config.
            Injected rather than imported to keep ``train`` -> ``det_metrics``
            a one-way dependency.

    Returns:
        ``(summary, per_image_rows, detections)``. Summary keys: the COCO stats,
        plus precision/recall/f1, count_mae/rmse/bias, n_images and val_loss.
    """
    model.eval()
    rows: List[Dict[str, Any]] = []
    detections: List[Dict[str, Any]] = []
    losses: List[float] = []
    seen_ids: List[int] = []

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                outputs = model(image)
            outputs = {name: value.float() for name, value in outputs.items()}

            target = batch.get("target")
            if loss_fn is not None and isinstance(target, dict):
                losses.append(float(loss_fn(
                    outputs, {n: v.to(device, non_blocking=True) for n, v in target.items()}
                )))

            prob = torch.sigmoid(outputs["heatmap"]).cpu()
            wh, off = outputs["wh"].cpu(), outputs["off"].cpu()
            width, height = int(batch["width"]), int(batch["height"])
            # Never cast: COCO image ids are as often filename strings as ints.
            image_id = batch["image_id"]
            seen_ids.append(image_id)
            gt_xyxy = boxes_xywh_to_xyxy(batch["boxes"])

            ap_boxes, ap_scores = decode_boxes(
                prob, wh, off, stride=output_stride, k=k, tau=ap_tau, top_k=top_k,
                box_nms_iou=box_nms_iou, size_parameterisation=size_parameterisation,
            )
            ap_boxes = clip_boxes_xyxy(ap_boxes, width, height)
            for box, score in zip(boxes_xyxy_to_xywh(ap_boxes), ap_scores):
                detections.append({
                    "image_id": image_id, "category_id": category_id,
                    "bbox": [float(v) for v in box], "score": float(score),
                })

            op_keep = ap_scores > tau
            op_boxes = ap_boxes[op_keep]
            tp, fp, fn = match_boxes_iou(op_boxes, gt_xyxy, iou_thr=match_iou)
            rows.append({
                "name": batch.get("name", ""), "image_id": image_id,
                "n_gt": len(gt_xyxy), "n_pred": len(op_boxes),
                "tp": tp, "fp": fp, "fn": fn,
            })

    summary = summarise_boxes(rows)
    summary.update(coco_eval(gt, detections, image_ids=seen_ids))
    summary["val_loss"] = float(np.mean(losses)) if losses else float("nan")
    return summary, rows, detections
