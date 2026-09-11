"""Unit tests for cropcounter.det_metrics — tau calibration and NMS.

Mirrors ``tests/test_metrics.py::test_sweep_tau_single_pass_reflects_threshold``
for the box head: a tiny fake ``nn.Module`` replays fixed heat/wh/off maps, so
the sweep is exercised without the DINOv3 backbone or any real dataset.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from cropcounter.det_metrics import (
    evaluate_boxes,
    match_boxes_iou,
    sweep_tau_boxes,
    sweep_tau_from_detections,
)

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _counts(summary, n_gt_total):
    """Recover the exact (tp, fp, fn) totals a summary was built from.

    ``summarise_boxes`` reports rates, not counts, so invert them: the totals
    are pinned by ``tp + fp = sum(n_pred)`` (read off ``count_bias``) and
    ``tp + fn = sum(n_gt)``.
    """
    n_images = summary["n_images"]
    n_pred_total = round(summary["count_bias"] * n_images + n_gt_total)
    tp = round(summary["precision"] * n_pred_total)
    return tp, n_pred_total - tp, n_gt_total - tp


class _FixedMapsModel(nn.Module):
    """Ignores its input; returns the same heat/wh/off triple every forward."""

    def __init__(self, heat_logits, wh, off):
        super().__init__()
        self._out = {"heatmap": heat_logits, "wh": wh, "off": off}

    def forward(self, x):  # noqa: ARG002 - the input is deliberately unused
        return {name: value.clone() for name, value in self._out.items()}


def _maps(grid_hw, peaks, *, log_wh, low=-20.0):
    """(1,1,H,W) logits with a peak per ``(x, y, logit)``, plus wh/off maps.

    ``wh`` is filled everywhere with ``log_wh`` (log-space cells) and ``off``
    with zeros, so a decoded box is centred exactly on its cell.
    """
    h, w = grid_hw
    heat = torch.full((1, 1, h, w), low)
    for x, y, logit in peaks:
        heat[0, 0, int(y), int(x)] = logit
    wh = torch.full((1, 2, h, w), float(log_wh))
    off = torch.zeros((1, 2, h, w))
    return heat, wh, off


def _batch(boxes_xywh, width, height, image_id=1, name="synthetic.jpg"):
    return {
        "image": torch.zeros(1, 3, 4, 4),  # unused by _FixedMapsModel
        "target": None,
        "boxes": np.asarray(boxes_xywh, dtype=np.float32).reshape(-1, 4),
        "image_id": image_id,
        "width": width,
        "height": height,
        "name": name,
    }


def _coco_gt_dict(boxes_xywh, image_id=1, width=64, height=64):
    return {
        "images": [{"id": image_id, "file_name": "synthetic.jpg",
                    "width": width, "height": height}],
        "categories": [{"id": 1, "name": "object", "supercategory": "object"}],
        "annotations": [
            {"id": i + 1, "image_id": image_id, "category_id": 1,
             "bbox": [float(v) for v in box], "area": float(box[2] * box[3]),
             "iscrowd": 0}
            for i, box in enumerate(np.asarray(boxes_xywh).reshape(-1, 4))
        ],
    }


# --------------------------------------------------------------------------- #
# 1. sweep_tau_from_detections — the pure helper
# --------------------------------------------------------------------------- #


def test_sweep_tau_from_detections_drops_false_positive_then_true_positive():
    # Image 1: one GT, a matching prediction at 0.9 and a distant FP at 0.2.
    # Image 2: one GT, a matching prediction at 0.4.
    gt_box = np.array([[0.0, 0.0, 10.0, 10.0]])
    per_image = [
        (np.array([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]]),
         np.array([0.9, 0.2]), gt_box),
        (np.array([[0.0, 0.0, 10.0, 10.0]]), np.array([0.4]), gt_box),
    ]

    rows = sweep_tau_from_detections(per_image, taus=[0.1, 0.3, 0.5], match_iou=0.5)
    by_tau = {r["tau"]: r for r in rows}
    assert [r["tau"] for r in rows] == [0.1, 0.3, 0.5]

    # tau 0.1 — everything survives: 2 TP, the 0.2 FP, nothing missed.
    assert _counts(by_tau[0.1], n_gt_total=2) == (2, 1, 0)
    assert by_tau[0.1]["precision"] == pytest.approx(2 / 3)
    assert by_tau[0.1]["recall"] == pytest.approx(1.0)

    # tau 0.3 — the false positive is gone. Precision up, recall unchanged.
    assert _counts(by_tau[0.3], n_gt_total=2) == (2, 0, 0)
    assert by_tau[0.3]["precision"] == pytest.approx(1.0)
    assert by_tau[0.3]["recall"] == pytest.approx(1.0)
    assert by_tau[0.3]["f1"] == pytest.approx(1.0)
    assert by_tau[0.3]["count_mae"] == pytest.approx(0.0)

    # tau 0.5 — the 0.4 true positive goes too. Recall down.
    assert _counts(by_tau[0.5], n_gt_total=2) == (1, 0, 1)
    assert by_tau[0.5]["precision"] == pytest.approx(1.0)
    assert by_tau[0.5]["recall"] == pytest.approx(0.5)
    assert by_tau[0.5]["f1"] == pytest.approx(2 * 1.0 * 0.5 / 1.5)
    assert by_tau[0.5]["count_mae"] == pytest.approx(0.5)
    assert by_tau[0.5]["count_bias"] == pytest.approx(-0.5)

    assert all(r["n_images"] == 2 for r in rows)


def test_sweep_tau_from_detections_consumes_a_generator_once():
    gt_box = np.array([[0.0, 0.0, 10.0, 10.0]])
    stream = (
        (np.array([[0.0, 0.0, 10.0, 10.0]]), np.array([0.6]), gt_box)
        for _ in range(3)
    )
    rows = sweep_tau_from_detections(stream, taus=[0.2, 0.8])
    by_tau = {r["tau"]: r for r in rows}
    # A single pass feeds every tau: 3 images scored at both thresholds.
    assert by_tau[0.2]["n_images"] == 3 and by_tau[0.8]["n_images"] == 3
    assert by_tau[0.2]["recall"] == pytest.approx(1.0)
    assert by_tau[0.8]["recall"] == pytest.approx(0.0)


def test_sweep_tau_from_detections_threshold_is_strictly_greater_than():
    gt_box = np.array([[0.0, 0.0, 10.0, 10.0]])
    per_image = [(np.array([[0.0, 0.0, 10.0, 10.0]]), np.array([0.3]), gt_box)]
    # scores > tau, matching evaluate_boxes: a score exactly at tau is dropped.
    rows = sweep_tau_from_detections(per_image, taus=[0.3])
    assert rows[0]["recall"] == pytest.approx(0.0)


def test_sweep_tau_from_detections_matches_direct_match_boxes_iou():
    gt = np.array([[0.0, 0.0, 10.0, 10.0], [40.0, 40.0, 50.0, 50.0]])
    pred = np.array([[0.5, 0.5, 10.5, 10.5], [40.0, 40.0, 50.0, 50.0]])
    scores = np.array([0.7, 0.2])
    rows = sweep_tau_from_detections([(pred, scores, gt)], taus=[0.1, 0.5])
    by_tau = {r["tau"]: r for r in rows}
    for tau in (0.1, 0.5):
        expected = match_boxes_iou(pred[scores > tau], gt, iou_thr=0.5)
        assert _counts(by_tau[tau], n_gt_total=2) == expected


# --------------------------------------------------------------------------- #
# 2. sweep_tau_boxes == evaluate_boxes at each tau
# --------------------------------------------------------------------------- #


SHARED_KEYS = ("count_mae", "count_rmse", "count_bias",
               "precision", "recall", "f1", "n_images")


def test_sweep_tau_boxes_agrees_with_evaluate_boxes_at_each_tau():
    grid, stride = (16, 16), 4
    width = height = grid[1] * stride
    # Two peaks two cells apart (distinct 3x3 local maxima): a confident one at
    # cell (2, 2) and a weak one at (2, 8). log(2.0) cells -> 8 px boxes.
    heat, wh, off = _maps(
        grid, [(2, 2, 20.0), (2, 8, 1.0)], log_wh=float(np.log(2.0))
    )
    # GT: the confident box exactly; the weak one exactly. xywh, centre = cell*stride.
    gt_boxes = np.array([[8.0 - 4, 8.0 - 4, 8.0, 8.0],
                         [8.0 - 4, 32.0 - 4, 8.0, 8.0]], dtype=np.float32)

    taus = [0.2, 0.8]
    model = _FixedMapsModel(heat, wh, off)
    swept = sweep_tau_boxes(
        model, [_batch(gt_boxes, width, height)], torch.device("cpu"), taus,
        ap_tau=0.01, k=3, top_k=100, match_iou=0.5,
        output_stride=stride, size_parameterisation="log",
    )
    by_tau = {r["tau"]: r for r in swept}

    gt_json = _coco_gt_dict(gt_boxes, width=width, height=height)
    for tau in taus:
        summary, _, _ = evaluate_boxes(
            model, [_batch(gt_boxes, width, height)], torch.device("cpu"),
            gt=gt_json, ap_tau=0.01, tau=tau, k=3, top_k=100,
            output_stride=stride, size_parameterisation="log", match_iou=0.5,
        )
        for key in SHARED_KEYS:
            assert by_tau[tau][key] == pytest.approx(summary[key]), f"{key} at tau {tau}"

    # And the sweep actually separates the two thresholds.
    assert by_tau[0.2]["recall"] == pytest.approx(1.0)   # sigmoid(1.0) ~= 0.731
    assert by_tau[0.8]["recall"] == pytest.approx(0.5)   # weak peak dropped


def test_sweep_tau_boxes_reports_only_operating_point_keys():
    grid, stride = (16, 16), 4
    heat, wh, off = _maps(grid, [(2, 2, 20.0)], log_wh=float(np.log(2.0)))
    gt_boxes = np.array([[4.0, 4.0, 8.0, 8.0]], dtype=np.float32)
    rows = sweep_tau_boxes(
        _FixedMapsModel(heat, wh, off),
        [_batch(gt_boxes, 64, 64)], torch.device("cpu"), [0.3],
        output_stride=stride,
    )
    # AP is threshold-independent, so the sweep does not pretend to report it.
    assert set(rows[0]) == set(SHARED_KEYS) | {"tau"}


# --------------------------------------------------------------------------- #
# 3. NMS on vs off
# --------------------------------------------------------------------------- #


def test_box_nms_changes_the_count_on_duplicate_boxes():
    grid, stride = (16, 16), 4
    width = height = grid[1] * stride
    # Two peaks three cells apart in x (so both survive the 3x3 local-max test)
    # decoding to 80 px boxes -> IoU ~0.74, well above a 0.5 NMS gate. One fish.
    heat, wh, off = _maps(
        grid, [(4, 8, 20.0), (7, 8, 10.0)], log_wh=float(np.log(20.0))
    )
    gt_boxes = np.array([[0.0, 0.0, 56.0, 56.0]], dtype=np.float32)

    def sweep(box_nms_iou):
        return sweep_tau_boxes(
            _FixedMapsModel(heat, wh, off),
            [_batch(gt_boxes, width, height)], torch.device("cpu"), [0.3],
            ap_tau=0.01, k=3, top_k=100, box_nms_iou=box_nms_iou,
            output_stride=stride, size_parameterisation="log",
        )[0]

    off_nms, on_nms = sweep(None), sweep(0.5)
    # NMS off: both duplicates survive -> one over-count. NMS on: one box left.
    assert _counts(off_nms, n_gt_total=1)[1] == 1   # a false positive
    assert _counts(on_nms, n_gt_total=1)[1] == 0
    assert off_nms["count_bias"] == pytest.approx(1.0)
    assert on_nms["count_bias"] == pytest.approx(0.0)
    assert on_nms["precision"] > off_nms["precision"]
    assert on_nms["recall"] == pytest.approx(off_nms["recall"])
