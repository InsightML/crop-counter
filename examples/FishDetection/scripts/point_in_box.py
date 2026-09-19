"""Point-in-box scoring: judge POINT predictions against BOX ground truth.

This is the metric that lets a point head and a box detector be compared on the
same frames. A prediction is a single (x, y) with a confidence; the ground truth
is an xyxy box; a prediction is correct when it lands inside a still-unmatched
box. Reducing a detector's boxes to their centres therefore scores it on the
point task without retraining anything — which is the whole point of this
module: it puts a number on the box-head and RF-DETR predictions we already
have, before any point-head GPU time is spent.

**Confidence is a 0-1 float everywhere in this module** — every array's third
column, every ``conf_thr``, every threshold in a sweep. The one place a 0-100
scale exists is CVAT XML: :func:`cropcounter.inference.write_cvat_xml` stores
``int(round(score * 100))`` in a ``Confidence`` attribute, and
:func:`parse_pred_points` is its exact inverse (it divides by 100). Nothing else
here rescales anything.

:func:`match_image` and :func:`image_accuracy` are VERBATIM copies of Liam's
functions in ``examples/WheatHead/notebooks/4_evaluate.ipynb`` so that the
numbers produced here are directly comparable with the wheat report. Do not
"improve" them — see their docstrings.

No torch at module level: only :func:`coco_results_to_cvat_points` touches
``cropcounter`` (for the CVAT writer), and it imports inside the function, so
this module stays importable in a bare numpy environment.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

#: CVAT 1.1 stores the point confidence as an integer percentage.
CVAT_CONFIDENCE_SCALE = 100.0


# --------------------------------------------------------------------------- #
# Liam's matcher — copied verbatim. Do not diverge.
# --------------------------------------------------------------------------- #
def match_image(points_xyc, boxes, conf_thr=0.0):
    """Greedy one-to-one point-in-box matching, predictions ranked by confidence.

    points_xyc : (N, 3) array of (x, y, confidence)
    boxes      : (M, 4) array of (x1, y1, x2, y2)
    conf_thr   : keep only predictions with confidence >= conf_thr
    Returns (TP, FP, FN).

    ---------------------------------------------------------------------------
    COPIED VERBATIM from ``examples/WheatHead/notebooks/4_evaluate.ipynb`` (the
    WheatHead evaluation notebook, Liam's). The body above the line, including
    the parameter names, is his; only this note is added. It MUST NOT DIVERGE:
    the wheat report's WDA/precision/recall numbers come from this exact code,
    and the fish numbers are only comparable with them while it stays identical.

    Two semantics worth stating out loud, both pinned by tests:

    * **Box edges are inclusive.** The containment test is
      ``x1 <= x <= x2`` and ``y1 <= y <= y2``, so a point exactly on an edge or
      corner counts as inside.
    * **The tiebreak is the nearest box CENTRE**, not the first free box and not
      the smallest box: among the still-unmatched boxes that contain the point,
      the one whose centre is nearest wins.

    Confidence is a 0-1 float here (see the module docstring); the notebook fed
    it CVAT's 0-100 integers, which is the one place the caller must convert.
    """
    pts = points_xyc[points_xyc[:, 2] >= conf_thr] if len(points_xyc) else points_xyc
    M = len(boxes)
    if M == 0:
        return 0, len(pts), 0  # no GT: every kept prediction is a FP
    if len(pts) == 0:
        return 0, 0, M         # no predictions: every GT box is a FN

    order = np.argsort(-pts[:, 2])  # highest confidence first
    box_used = np.zeros(M, dtype=bool)
    box_cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    box_cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    tp = fp = 0
    for idx in order:
        x, y = pts[idx, 0], pts[idx, 1]
        inside = (~box_used) & (boxes[:, 0] <= x) & (x <= boxes[:, 2]) & \
                 (boxes[:, 1] <= y) & (y <= boxes[:, 3])
        if inside.any():
            cand = np.where(inside)[0]
            # nearest box centre among the containing, still-free boxes
            j = cand[np.argmin((box_cx[cand] - x) ** 2 + (box_cy[cand] - y) ** 2)]
            box_used[j] = True
            tp += 1
        else:
            fp += 1
    fn = int((~box_used).sum())
    return tp, fp, fn


def image_accuracy(tp, fp, fn):
    """Per-image accuracy tp/(tp+fp+fn); 0/0 := 1.0.

    COPIED VERBATIM from the same WheatHead notebook (there it feeds WDA, the
    Global Wheat challenge's domain-averaged accuracy). An image with no GT and
    no predictions is a perfect image, not an undefined one.
    """
    denom = tp + fn + fp
    return 1.0 if denom == 0 else tp / denom  # 0/0 (empty & no preds) := perfect


# --------------------------------------------------------------------------- #
# Dataset-level scoring
# --------------------------------------------------------------------------- #
def score_dataset(
    preds_by_image: Mapping,
    gt_by_image: Mapping,
    conf_thr: float = 0.0,
) -> Tuple[Dict, List[Dict]]:
    """Score every GT image at one confidence threshold.

    Args:
        preds_by_image: image id -> (N, 3) array of (x, y, conf in 0-1).
        gt_by_image: image id -> (M, 4) array of xyxy boxes. **This mapping
            defines the dataset**: an image in it but absent from
            ``preds_by_image`` is scored as zero predictions (so a detector that
            simply skipped the empty frames is not flattered), and an image only
            in ``preds_by_image`` is ignored.
        conf_thr: keep predictions with ``conf >= conf_thr``.

    Returns:
        ``(summary, rows)``. ``summary`` holds micro-averaged precision/recall/
        F1 over all images (0/0 := 0.0), the count errors at this threshold
        (MAE, RMSE, bias = mean(pred - gt)), the mean per-image accuracy (the
        macro number, Liam's WDA without the domain grouping), and the totals.
        ``rows`` is one dict per GT image, in ``gt_by_image`` iteration order.
    """
    rows: List[Dict] = []
    tp_total = fp_total = fn_total = 0
    n_gt = n_pred = 0
    accs: List[float] = []
    diffs: List[float] = []

    for image_id, boxes in gt_by_image.items():
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        pts = np.asarray(
            preds_by_image.get(image_id, np.zeros((0, 3))), dtype=float
        ).reshape(-1, 3)
        tp, fp, fn = match_image(pts, boxes, conf_thr)
        acc = image_accuracy(tp, fp, fn)
        gt_count = len(boxes)
        pred_count = tp + fp  # predictions kept at this threshold
        rows.append({
            "image_id": image_id,
            "gt_count": gt_count,
            "pred_count": pred_count,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "accuracy": float(acc),
        })
        tp_total += tp
        fp_total += fp
        fn_total += fn
        n_gt += gt_count
        n_pred += pred_count
        accs.append(acc)
        diffs.append(float(pred_count - gt_count))

    diff = np.asarray(diffs, dtype=float)
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    summary = {
        "conf_thr": float(conf_thr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "count_mae": float(np.abs(diff).mean()) if len(diff) else 0.0,
        "count_rmse": float(np.sqrt((diff ** 2).mean())) if len(diff) else 0.0,
        "count_bias": float(diff.mean()) if len(diff) else 0.0,
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "n_images": len(rows),
        "n_images_with_pred": int(sum(1 for r in rows if r["pred_count"] > 0)),
        "n_gt": int(n_gt),
        "n_pred": int(n_pred),
        "tp": int(tp_total),
        "fp": int(fp_total),
        "fn": int(fn_total),
    }
    return summary, rows


def sweep_thresholds(
    preds_by_image: Mapping,
    gt_by_image: Mapping,
    thresholds: Sequence[float],
) -> List[Dict]:
    """Score the dataset at each threshold -> one summary per threshold.

    The list is the accuracy/F1-vs-threshold curve, and the input to picking an
    operating point (argmax F1; argmin ``count_mae`` is the counting-first
    alternative and is usually a different, lower threshold).
    """
    return [score_dataset(preds_by_image, gt_by_image, float(t))[0] for t in thresholds]


# --------------------------------------------------------------------------- #
# I/O: CVAT points, COCO results, COCO annotations
# --------------------------------------------------------------------------- #
def parse_pred_points(cvat_xml_path) -> Dict[str, np.ndarray]:
    """Read a CVAT 1.1 points XML -> {image name: (N, 3) of (x, y, conf 0-1)}.

    The exact inverse of :func:`cropcounter.inference.write_cvat_xml`, which
    writes one ``<points points="x,y">`` per prediction with a child
    ``<attribute name="Confidence">`` holding ``int(round(score * 100))``. The
    integer is divided by ``CVAT_CONFIDENCE_SCALE`` here so callers only ever
    see 0-1 — unconditionally, because that writer is the only producer we
    read; a missing attribute means "no score recorded" and becomes 1.0.

    Keys are the ``name`` attribute (a file name), not a COCO image id.
    """
    root = ET.parse(str(cvat_xml_path)).getroot()
    preds: Dict[str, np.ndarray] = {}
    for img in root.findall("image"):
        name = img.get("name")
        pts = []
        for p in img.findall("points"):
            x, y = map(float, p.get("points").split(","))
            conf_el = p.find("attribute[@name='Confidence']")
            if conf_el is None:
                conf_el = p.find("attribute")
            raw = float(conf_el.text) if conf_el is not None else CVAT_CONFIDENCE_SCALE
            pts.append((x, y, raw / CVAT_CONFIDENCE_SCALE))
        preds[name] = np.array(pts, dtype=float).reshape(-1, 3)
    return preds


def coco_results_to_points(
    results: Iterable[Mapping],
    id_map: Optional[Mapping] = None,
) -> Dict:
    """COCO detection results -> {image id: (N, 3) of (cx, cy, score)}.

    ``results`` is the standard list of ``{"image_id", "category_id", "bbox":
    [x, y, w, h], "score"}``. Each box is reduced to its centre
    ``(x + w/2, y + h/2)`` — the cheapest honest way to ask a box detector the
    point question. Order within an image is the order in ``results``;
    :func:`match_image` re-sorts by confidence anyway, so the output order never
    affects a score.

    Args:
        id_map: optional ``{old id: new id}`` remap, for joining one system's
            image ids to another's (e.g. string CFD ids to renumbered ints).
            Ids absent from the map are passed through unchanged.
    """
    by_image: Dict[object, List[Tuple[float, float, float]]] = {}
    for det in results:
        image_id = det["image_id"]
        if id_map is not None:
            image_id = id_map.get(image_id, image_id)
        x, y, w, h = (float(v) for v in det["bbox"])
        by_image.setdefault(image_id, []).append(
            (x + w / 2.0, y + h / 2.0, float(det["score"]))
        )
    return {k: np.array(v, dtype=float).reshape(-1, 3) for k, v in by_image.items()}


def coco_boxes_by_image(annotations_json_path) -> Dict:
    """COCO annotations document -> {image id: (M, 4) of xyxy}.

    ``bbox`` is COCO xywh and becomes xyxy. Every image in ``images`` gets an
    entry, so an image with no annotations is an empty ``(0, 4)`` array rather
    than a missing key — that is what makes empty frames scoreable (they are
    where a detector's false positives show up).
    """
    with open(annotations_json_path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    by_image: Dict[object, List[Tuple[float, float, float, float]]] = {
        img["id"]: [] for img in doc.get("images", [])
    }
    for ann in doc.get("annotations", []):
        x, y, w, h = (float(v) for v in ann["bbox"])
        by_image.setdefault(ann["image_id"], []).append((x, y, x + w, y + h))
    return {k: np.array(v, dtype=float).reshape(-1, 4) for k, v in by_image.items()}


def coco_results_to_cvat_points(
    results: Iterable[Mapping],
    images: Iterable[Mapping] | Mapping,
    out_path,
    label: str = "fish",
) -> Path:
    """Write COCO results as a CVAT 1.1 points XML, via cropcounter's writer.

    Going through :func:`cropcounter.inference.write_cvat_xml` means every
    prediction system — this repo's point head, a box head's centres, a
    released detector's centres — can be handed to the scorer in one file shape,
    and viewed in CVAT. Note the writer clamps points into
    ``[0, width-1] x [0, height-1]`` and rounds to 2dp and the confidence to a
    whole percent, so a round trip is lossy at the 0.005 level.

    Args:
        images: the COCO ``images`` list (dicts with ``id``, ``file_name``,
            ``width``, ``height``), or a mapping of image id -> that dict.
            Images with no prediction are written as empty image elements, so
            the XML covers the whole evaluation set.
    """
    from cropcounter.inference import write_cvat_xml  # torch-importing: keep it local

    if isinstance(images, Mapping):
        image_records = {k: dict(v) for k, v in images.items()}
        for key, rec in image_records.items():
            rec.setdefault("id", key)
    else:
        image_records = {img["id"]: dict(img) for img in images}

    points = coco_results_to_points(results)
    image_preds = []
    for image_id, rec in image_records.items():
        arr = points.get(image_id, np.zeros((0, 3)))
        image_preds.append({
            "name": rec.get("file_name", str(image_id)),
            "width": int(rec["width"]),
            "height": int(rec["height"]),
            "points": arr[:, :2],
            "scores": arr[:, 2],
            "label": label,
        })
    return write_cvat_xml(image_preds, Path(out_path), label=label)
