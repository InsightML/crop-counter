"""Unit tests for the bounding-box detection head: targets, decode, losses, data.

Pure numpy/torch/PIL on synthetic boxes; no backbone, no network. The decoder
is constructed directly (``PyramidDecoder(STAGE_CHANNELS["base"])``) so the
state-dict compatibility check costs nothing.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from cropcounter.boxmap import decode_boxes, gaussian_radius, render_box_targets
from cropcounter.crop_dataset import Box, ImageRecord, parse_coco_detection
from cropcounter.dinov3_pyramid import STAGE_CHANNELS, PyramidDecoder
from cropcounter.heatmap import render_targets
from cropcounter.losses import masked_l1_loss

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CHECKPOINT = REPO_ROOT / "weights" / "decoder_best.pt"


# --------------------------------------------------------------------------- #
# 1. gaussian_radius
# --------------------------------------------------------------------------- #


def _centernet_radius_reference(det_size, min_overlap=0.7):
    """CenterNet's gaussian_radius, re-derived here so the test is independent.

    Transcribed from CenterNet ``src/lib/utils/image.py`` — including the
    ``(b + sq) / 2`` denominators that differ from CornerNet's derivation.
    """
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


@pytest.mark.parametrize(
    "det_size",
    [(4.0, 4.0), (8.0, 8.0), (16.0, 16.0), (4.0, 11.0), (15.0, 15.0),
     (10.0, 17.5), (3.0, 3.0), (64.0, 32.0)],
)
def test_gaussian_radius_matches_independent_centernet_formula(det_size):
    assert gaussian_radius(det_size, min_overlap=0.7) == pytest.approx(
        _centernet_radius_reference(det_size, 0.7), rel=1e-12, abs=1e-12
    )


def test_gaussian_radius_monotonic_non_decreasing_in_box_size():
    sizes = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0]
    square = [gaussian_radius((s, s)) for s in sizes]
    assert all(b >= a for a, b in zip(square, square[1:]))
    # Non-square, growing one side at a time: still non-decreasing.
    widening = [gaussian_radius((10.0, w)) for w in sizes]
    assert all(b >= a for a, b in zip(widening, widening[1:]))
    heightening = [gaussian_radius((h, 10.0)) for h in sizes]
    assert all(b >= a for a, b in zip(heightening, heightening[1:]))


def test_gaussian_radius_is_scale_homogeneous():
    # Degree-1 homogeneous: doubling the box doubles the radius.
    assert gaussian_radius((20.0, 30.0)) == pytest.approx(2.0 * gaussian_radius((10.0, 15.0)))


# --------------------------------------------------------------------------- #
# 2. render_box_targets
# --------------------------------------------------------------------------- #

# (x, y, w, h) in input pixels; stride 4 -> a 64x64 grid over a 256x256 frame.
ROUND_TRIP_BOXES = np.array(
    [
        [10.0, 12.0, 20.0, 30.0],
        [70.0, 8.0, 44.0, 16.0],
        [150.0, 20.0, 60.0, 60.0],
        [14.0, 100.0, 30.0, 24.0],
        [90.0, 95.0, 18.0, 50.0],
        [160.0, 110.0, 70.0, 40.0],
        [20.0, 180.0, 40.0, 40.0],
        [100.0, 190.0, 24.0, 36.0],
        [170.0, 200.0, 50.0, 30.0],
        [60.0, 40.0, 12.0, 12.0],
    ],
    dtype=np.float32,
)


def test_render_box_targets_shapes_and_dtypes():
    hm, wh, off, mask, n_coll = render_box_targets(ROUND_TRIP_BOXES, (64, 64), stride=4)
    assert hm.shape == (64, 64) and hm.dtype == np.float32
    assert wh.shape == (2, 64, 64) and wh.dtype == np.float32
    assert off.shape == (2, 64, 64) and off.dtype == np.float32
    assert mask.shape == (64, 64) and mask.dtype == np.float32
    assert n_coll == 0


def test_render_box_targets_peak_is_exactly_one_at_every_centre_cell():
    hm, _, _, mask, _ = render_box_targets(ROUND_TRIP_BOXES, (64, 64), stride=4)
    assert mask.sum() == len(ROUND_TRIP_BOXES)
    for x, y, w, h in ROUND_TRIP_BOXES:
        cx = int(np.floor((x + w / 2) / 4))
        cy = int(np.floor((y + h / 2) / 4))
        assert hm[cy, cx] == 1.0
        assert mask[cy, cx] == 1.0
    assert hm.max() == pytest.approx(1.0)


def test_render_box_targets_empty_input():
    hm, wh, off, mask, n_coll = render_box_targets(
        np.empty((0, 4), np.float32), (16, 16), stride=4
    )
    assert hm.max() == 0.0 and mask.sum() == 0.0
    assert wh.shape == (2, 16, 16) and off.shape == (2, 16, 16)
    assert n_coll == 0


def test_render_box_targets_counts_centre_cell_collisions():
    # Two boxes whose centres floor to the same stride-4 cell (10, 10).
    boxes = np.array(
        [[38.0, 38.0, 6.0, 6.0],    # centre (41.0, 41.0) -> cell (10, 10)
         [39.0, 39.0, 6.0, 6.0]],   # centre (42.0, 42.0) -> cell (10, 10)
        dtype=np.float32,
    )
    hm, _, _, mask, n_coll = render_box_targets(boxes, (32, 32), stride=4)
    assert n_coll == 1
    assert mask.sum() == 1.0
    assert hm[10, 10] == 1.0
    # A third, well-separated box adds no collision.
    boxes3 = np.vstack([boxes, np.array([[100.0, 100.0, 8.0, 8.0]], np.float32)])
    _, _, _, mask3, n_coll3 = render_box_targets(boxes3, (32, 32), stride=4)
    assert n_coll3 == 1
    assert mask3.sum() == 2.0


def test_render_box_targets_wh_and_off_values():
    boxes = np.array([[10.0, 12.0, 20.0, 30.0]], dtype=np.float32)
    _, wh, off, mask, _ = render_box_targets(boxes, (64, 64), stride=4, size_parameterisation="log")
    cx_f, cy_f = (10.0 + 10.0) / 4, (12.0 + 15.0) / 4   # (5.0, 6.75)
    cx, cy = int(cx_f), int(cy_f)
    assert wh[0, cy, cx] == pytest.approx(math.log(20.0 / 4), rel=1e-6)
    assert wh[1, cy, cx] == pytest.approx(math.log(30.0 / 4), rel=1e-6)
    assert off[0, cy, cx] == pytest.approx(cx_f - cx, abs=1e-6)
    assert off[1, cy, cx] == pytest.approx(cy_f - cy, abs=1e-6)
    assert 0.0 <= off[0, cy, cx] < 1.0 and 0.0 <= off[1, cy, cx] < 1.0

    _, wh_lin, _, _, _ = render_box_targets(
        boxes, (64, 64), stride=4, size_parameterisation="linear"
    )
    assert wh_lin[0, cy, cx] == pytest.approx(20.0 / 4)
    assert wh_lin[1, cy, cx] == pytest.approx(30.0 / 4)


# --------------------------------------------------------------------------- #
# 3. coordinate round trip (targets -> decode -> COCO AP)
# --------------------------------------------------------------------------- #


def _coco_gt_dict(boxes_xywh, image_id=1, width=256, height=256):
    return {
        "images": [{"id": image_id, "file_name": "synthetic.jpg",
                    "width": width, "height": height}],
        "categories": [{"id": 1, "name": "object", "supercategory": "object"}],
        "annotations": [
            {"id": i + 1, "image_id": image_id, "category_id": 1,
             "bbox": [float(v) for v in box], "area": float(box[2] * box[3]),
             "iscrowd": 0}
            for i, box in enumerate(boxes_xywh)
        ],
    }


@pytest.mark.parametrize("param", ["log", "linear"])
def test_ground_truth_targets_round_trip_through_decode_boxes(param, capsys):
    hm, wh, off, _, n_coll = render_box_targets(
        ROUND_TRIP_BOXES, (64, 64), stride=4, size_parameterisation=param
    )
    assert n_coll == 0, "the round-trip fixture must not collide at stride 4"

    boxes, scores = decode_boxes(
        hm, wh, off, stride=4, k=3, tau=0.01, top_k=100, size_parameterisation=param
    )
    assert len(boxes) == len(ROUND_TRIP_BOXES)
    assert (np.diff(scores) <= 1e-6).all()

    gt_xyxy = np.stack([
        ROUND_TRIP_BOXES[:, 0], ROUND_TRIP_BOXES[:, 1],
        ROUND_TRIP_BOXES[:, 0] + ROUND_TRIP_BOXES[:, 2],
        ROUND_TRIP_BOXES[:, 1] + ROUND_TRIP_BOXES[:, 3],
    ], axis=1)
    # Every recovered box matches its source to sub-1e-3 pixels.
    for box in boxes:
        err = np.abs(gt_xyxy - box[None]).max(axis=1)
        assert err.min() < 1e-3, f"no GT box within 1e-3 px of {box} (best {err.min():.2e})"
    assert len({int(np.abs(gt_xyxy - b[None]).max(axis=1).argmin()) for b in boxes}) == len(boxes)

    from cropcounter.det_metrics import coco_eval

    detections = [
        {"image_id": 1, "category_id": 1,
         "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
         "score": float(s)}
        for b, s in zip(boxes, scores)
    ]
    stats = coco_eval(_coco_gt_dict(ROUND_TRIP_BOXES), detections)
    assert stats["ap"] == pytest.approx(1.0)
    assert stats["ap50"] == pytest.approx(1.0)

    with capsys.disabled():
        print(f"\n[stride-4 collisions] {n_coll}/{len(ROUND_TRIP_BOXES)} boxes "
              f"({param}) collided on a centre cell")


def test_decode_boxes_accepts_batched_tensors_and_numpy():
    hm, wh, off, _, _ = render_box_targets(ROUND_TRIP_BOXES, (64, 64), stride=4)
    np_boxes, _ = decode_boxes(hm, wh, off, stride=4, tau=0.01)
    t_boxes, _ = decode_boxes(
        torch.from_numpy(hm)[None, None],
        torch.from_numpy(wh)[None],
        torch.from_numpy(off)[None],
        stride=4, tau=0.01,
    )
    np.testing.assert_allclose(np_boxes, t_boxes, atol=1e-6)


def test_decode_boxes_empty_below_threshold():
    hm = np.zeros((16, 16), np.float32)
    wh = np.zeros((2, 16, 16), np.float32)
    off = np.zeros((2, 16, 16), np.float32)
    boxes, scores = decode_boxes(hm, wh, off, stride=4, tau=0.5)
    assert boxes.shape == (0, 4) and scores.shape == (0,)


# --------------------------------------------------------------------------- #
# 4. top_k truncation and box NMS
# --------------------------------------------------------------------------- #


def test_decode_boxes_truncates_to_top_k_by_score():
    hm = np.zeros((64, 64), np.float32)
    wh = np.full((2, 64, 64), math.log(4.0), np.float32)
    off = np.full((2, 64, 64), 0.5, np.float32)
    peaks = [(4 + 6 * i, 4 + 6 * j) for i in range(4) for j in range(5)]  # 20 peaks
    for n, (cy, cx) in enumerate(peaks):
        hm[cy, cx] = 0.2 + 0.03 * n
    all_boxes, all_scores = decode_boxes(hm, wh, off, stride=4, tau=0.01, top_k=100)
    assert len(all_boxes) == len(peaks)

    boxes, scores = decode_boxes(hm, wh, off, stride=4, tau=0.01, top_k=5)
    assert len(boxes) == 5
    np.testing.assert_allclose(scores, all_scores[:5], atol=1e-6)


def test_decode_boxes_nms_suppresses_overlapping_duplicate():
    hm = np.zeros((32, 32), np.float32)
    hm[10, 10] = 0.9
    hm[10, 12] = 0.8
    wh = np.full((2, 32, 32), math.log(40.0), np.float32)   # 160 px boxes
    off = np.full((2, 32, 32), 0.5, np.float32)

    keep_all, _ = decode_boxes(hm, wh, off, stride=4, tau=0.01, box_nms_iou=None)
    assert len(keep_all) == 2

    kept, scores = decode_boxes(hm, wh, off, stride=4, tau=0.01, box_nms_iou=0.5)
    assert len(kept) == 1
    assert scores[0] == pytest.approx(0.9, abs=1e-6)

    # A loose IoU threshold keeps both.
    kept_loose, _ = decode_boxes(hm, wh, off, stride=4, tau=0.01, box_nms_iou=0.95)
    assert len(kept_loose) == 2


# --------------------------------------------------------------------------- #
# 5. masked_l1_loss
# --------------------------------------------------------------------------- #


def test_masked_l1_loss_is_mean_over_masked_cells():
    torch.manual_seed(0)
    pred = torch.randn(2, 2, 8, 8)
    target = torch.randn(2, 2, 8, 8)
    mask = torch.zeros(2, 1, 8, 8)
    mask[0, 0, 1, 1] = 1.0
    mask[0, 0, 4, 5] = 1.0
    mask[1, 0, 7, 0] = 1.0

    expected = torch.stack([
        (pred[0, :, 1, 1] - target[0, :, 1, 1]).abs(),
        (pred[0, :, 4, 5] - target[0, :, 4, 5]).abs(),
        (pred[1, :, 7, 0] - target[1, :, 7, 0]).abs(),
    ]).mean()
    assert masked_l1_loss(pred, target, mask).item() == pytest.approx(expected.item(), rel=1e-6)


def test_masked_l1_loss_all_zero_mask_is_zero_and_finite():
    pred = torch.randn(2, 2, 8, 8)
    target = torch.randn(2, 2, 8, 8)
    mask = torch.zeros(2, 1, 8, 8)
    loss = masked_l1_loss(pred, target, mask)
    assert loss.item() == 0.0
    assert torch.isfinite(loss)


def test_masked_l1_loss_accepts_a_two_channel_mask():
    pred = torch.randn(1, 2, 4, 4)
    target = torch.randn(1, 2, 4, 4)
    mask1 = torch.zeros(1, 1, 4, 4)
    mask1[0, 0, 2, 2] = 1.0
    assert masked_l1_loss(pred, target, mask1).item() == pytest.approx(
        masked_l1_loss(pred, target, mask1.expand(1, 2, 4, 4)).item(), rel=1e-6
    )


def test_masked_l1_loss_gradients_flow_only_through_masked_cells():
    pred = torch.zeros(1, 2, 4, 4, requires_grad=True)
    target = torch.ones(1, 2, 4, 4)
    mask = torch.zeros(1, 1, 4, 4)
    mask[0, 0, 1, 3] = 1.0
    masked_l1_loss(pred, target, mask).backward()
    grad = pred.grad
    assert (grad[0, :, 1, 3] != 0).all()
    assert grad.abs().sum() == pytest.approx(grad[0, :, 1, 3].abs().sum())


# --------------------------------------------------------------------------- #
# 6. per-point sigma in render_targets
# --------------------------------------------------------------------------- #


def test_render_targets_per_point_sigma_matches_per_point_max_compose():
    points = np.array([[8.0, 8.0], [24.0, 10.0], [40.0, 33.0], [50.0, 50.0]])
    sigmas = np.array([0.5, 1.1666667, 2.0, 3.25])
    combined = render_targets(points, (64, 64), sigmas)
    pieces = [render_targets(p[None], (64, 64), float(s)) for p, s in zip(points, sigmas)]
    expected = np.maximum.reduce(pieces)
    np.testing.assert_array_equal(combined, expected)


def test_render_targets_scalar_sigma_behaviour_unchanged():
    points = np.array([[8.0, 8.0], [24.0, 24.0]])
    heat = render_targets(points, (64, 64), 2.0)
    # Re-derive the old implementation inline.
    radius = max(1, int(round(3.0 * 2.0)))
    size = 2 * radius + 1
    ax = np.arange(size, dtype=np.float64) - radius
    stamp = np.exp(-(ax[None, :] ** 2 + ax[:, None] ** 2) / (2.0 * 2.0 * 2.0)).astype(np.float32)
    expected = np.zeros((64, 64), np.float32)
    for x, y in points:
        cx, cy = int(x), int(y)
        expected[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1] = np.maximum(
            expected[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1], stamp
        )
    np.testing.assert_array_equal(heat, expected)
    assert heat[8, 8] == 1.0 and heat[24, 24] == 1.0


def test_render_targets_per_point_sigma_length_must_match():
    with pytest.raises(ValueError):
        render_targets(np.array([[1.0, 1.0], [2.0, 2.0]]), (16, 16), np.array([1.0]))


# --------------------------------------------------------------------------- #
# 7. parse_coco_detection
# --------------------------------------------------------------------------- #


def _write_tiny_coco(tmp_path: Path) -> Path:
    payload = {
        "images": [
            {"id": 7, "file_name": "a.jpg", "width": 800, "height": 600,
             "dataset": "brackish", "is_train": True},
            {"id": 9, "file_name": "b.jpg", "width": 640, "height": 480,
             "dataset": "cfd", "is_train": False},
        ],
        "categories": [
            {"id": 1, "name": "fish"},
            {"id": 2, "name": "jellyfish"},
        ],
        "annotations": [
            {"id": 1, "image_id": 7, "category_id": 1,
             "bbox": [10.0, 20.0, 30.0, 40.0], "area": 1200.0, "iscrowd": 0},
            {"id": 2, "image_id": 7, "category_id": 2,
             "bbox": [100.0, 200.0, 50.0, 60.0], "area": 3000.0, "iscrowd": 1},
            {"id": 3, "image_id": 7, "category_id": 1,
             "bbox": [5.0, 5.0, 0.0, 12.0], "area": 0.0, "iscrowd": 0},  # degenerate
        ],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_parse_coco_detection_round_trip(tmp_path):
    records = parse_coco_detection(_write_tiny_coco(tmp_path))
    assert len(records) == 2

    first, second = records
    assert first.name == "a.jpg" and first.width == 800 and first.height == 600
    assert first.image_id == 7
    assert first.source_dataset == "brackish"
    assert first.is_train is True
    # The zero-width box is dropped; the iscrowd box is kept.
    assert len(first.boxes) == 2
    assert [b.label for b in first.boxes] == ["fish", "jellyfish"]
    assert (first.boxes[0].x, first.boxes[0].y, first.boxes[0].w, first.boxes[0].h) == (
        10.0, 20.0, 30.0, 40.0
    )
    assert isinstance(first.boxes[0], Box)

    # The annotation-free image still yields a record.
    assert second.name == "b.jpg" and second.image_id == 9
    assert second.boxes == []
    assert second.source_dataset == "cfd" and second.is_train is False


def test_parse_coco_detection_clips_to_the_image_and_drops_outside_boxes(tmp_path):
    """Boxes entirely outside their frame (a real CFD case: x >= width) are
    dropped; boxes that overlap the edge are clipped. Albumentations validates
    before it clips and raises on a zero-width survivor."""
    import json

    doc = {
        "images": [{"id": "f", "file_name": "f.jpg", "width": 100, "height": 50}],
        "categories": [{"id": 1, "name": "fish"}],
        "annotations": [
            {"id": 1, "image_id": "f", "category_id": 1, "bbox": [100.0, 10.0, 20.0, 10.0]},
            {"id": 2, "image_id": "f", "category_id": 1, "bbox": [90.0, 45.0, 20.0, 20.0]},
            {"id": 3, "image_id": "f", "category_id": 1, "bbox": [-5.0, -5.0, 10.0, 10.0]},
            {"id": 4, "image_id": "f", "category_id": 1, "bbox": [10.0, 10.0, 5.0, 5.0]},
            {"id": 5, "image_id": "f", "category_id": 1, "bbox": [0.0, 60.0, 5.0, 5.0]},
        ],
    }
    path = tmp_path / "edge.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    (record,) = parse_coco_detection(path)
    kept = [(b.x, b.y, b.w, b.h) for b in record.boxes]
    assert kept == [
        (90.0, 45.0, 10.0, 5.0),   # clipped to the bottom-right corner
        (0.0, 0.0, 5.0, 5.0),      # clipped to the top-left corner
        (10.0, 10.0, 5.0, 5.0),    # untouched
    ]


def test_parse_coco_detection_accepts_string_ids_and_empty_markers(tmp_path):
    """Real CFD metadata: image ids are filename strings, annotation ids are
    int for some sources and str for others, and "reviewed, nothing here" is
    recorded as a bbox-less annotation in an ``empty`` category."""
    payload = {
        "images": [
            {"id": "torsi_20190716-021037.129.JPG", "file_name": "torsi_1.JPG",
             "width": 960, "height": 540, "dataset": "torsi", "is_train": False},
            {"id": "brackish_00123.jpg.rf.9ab3f1", "file_name": "brackish_00123.jpg",
             "width": 960, "height": 540, "dataset": "brackish", "is_train": True},
        ],
        "categories": [{"id": 0, "name": "empty"}, {"id": 1, "name": "fish"}],
        "annotations": [
            {"id": 1, "image_id": "torsi_20190716-021037.129.JPG", "category_id": 1,
             "bbox": [40.0, 50.0, 46.0, 30.0], "area": 1380.0, "iscrowd": 0},
            {"id": "ann-str-2", "image_id": "torsi_20190716-021037.129.JPG",
             "category_id": 1, "bbox": [400.0, 200.0, 52.0, 44.0], "area": 2288.0,
             "iscrowd": 0},
            # Empty marker: no bbox, category 0.
            {"id": "ann-str-3", "image_id": "brackish_00123.jpg.rf.9ab3f1",
             "category_id": 0, "iscrowd": 0},
        ],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    records = parse_coco_detection(path)
    assert [r.image_id for r in records] == [
        "torsi_20190716-021037.129.JPG", "brackish_00123.jpg.rf.9ab3f1"
    ]
    assert all(isinstance(r.image_id, str) for r in records)
    assert len(records[0].boxes) == 2
    assert records[0].width == 960 and records[0].height == 540
    # The empty marker does not become a box, and the frame stays a negative.
    assert records[1].boxes == []
    assert records[1].source_dataset == "brackish" and records[1].is_train is True


def test_parse_coco_detection_skips_empty_marker_even_with_a_bbox(tmp_path):
    payload = {
        "images": [{"id": "a", "file_name": "a.jpg", "width": 960, "height": 540}],
        "categories": [{"id": 0, "name": "empty"}, {"id": 1, "name": "fish"}],
        "annotations": [
            {"id": 1, "image_id": "a", "category_id": 0,
             "bbox": [0.0, 0.0, 960.0, 540.0], "iscrowd": 0},
            {"id": 2, "image_id": "a", "category_id": 1,
             "bbox": [10.0, 10.0, 46.0, 46.0], "iscrowd": 0},
        ],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    (record,) = parse_coco_detection(path)
    assert [b.label for b in record.boxes] == ["fish"]


def test_parse_coco_detection_label_filter(tmp_path):
    records = parse_coco_detection(_write_tiny_coco(tmp_path), labels=("fish",))
    assert [b.label for b in records[0].boxes] == ["fish"]


def test_parse_coco_detection_is_registered_as_coco_bbox(tmp_path):
    from cropcounter.crop_dataset import ANNOTATION_FILENAMES, LOADERS, load_records

    assert LOADERS["coco_bbox"] is parse_coco_detection
    assert "annotations.json" in ANNOTATION_FILENAMES["coco_bbox"]
    assert "instances_default.json" in ANNOTATION_FILENAMES["coco_bbox"]
    _write_tiny_coco(tmp_path)
    records = load_records(tmp_path, fmt="coco_bbox", labels=None)
    assert len(records) == 2 and len(records[0].boxes) == 2


def test_image_record_defaults_stay_backwards_compatible():
    rec = ImageRecord(name="x.jpg", width=10, height=10)
    assert rec.points == [] and rec.boxes == []
    assert rec.source_dataset is None and rec.is_train is None and rec.image_id is None


# --------------------------------------------------------------------------- #
# 8. checkpoint-compatible state dict + head init
# --------------------------------------------------------------------------- #


def _decoder(task: str) -> PyramidDecoder:
    return PyramidDecoder(STAGE_CHANNELS["base"], c_dec=192, output_stride=4, task=task)


@pytest.mark.skipif(not SHIPPED_CHECKPOINT.is_file(), reason="shipped wheat checkpoint absent")
def test_point_decoder_state_dict_matches_shipped_checkpoint_exactly():
    payload = torch.load(SHIPPED_CHECKPOINT, map_location="cpu", weights_only=False)
    shipped = set(payload["decoder"].keys())
    assert set(_decoder("point").state_dict().keys()) == shipped
    # And it loads strictly, which is the actual contract.
    _decoder("point").load_state_dict(payload["decoder"])


def test_box_decoder_adds_only_geometry_keys():
    point_keys = set(_decoder("point").state_dict().keys())
    box_keys = set(_decoder("box").state_dict().keys())
    assert point_keys <= box_keys
    extra = box_keys - point_keys
    assert extra and all(k.startswith("geometry.") for k in extra)


def test_point_decoder_has_no_geometry_branch():
    assert _decoder("point").geometry is None
    assert _decoder("point").task == "point"


def test_box_head_bias_initialisation():
    dec = _decoder("box")
    assert dec.head.bias.detach().allclose(torch.full((1,), -4.0))
    final = dec.geometry[-1]
    assert final.out_channels == 4
    assert final.bias.detach()[:2].allclose(torch.zeros(2)), "wh bias must start at 0"
    assert final.bias.detach()[2:].allclose(torch.full((2,), 0.5)), "off bias must start at 0.5"


def test_decoder_forward_return_types():
    feats = [torch.randn(1, c, 64 // (2 ** i), 64 // (2 ** i))
             for i, c in enumerate(STAGE_CHANNELS["base"])]
    out_point = _decoder("point")(feats)
    assert isinstance(out_point, torch.Tensor)
    assert out_point.shape == (1, 1, 64, 64)

    out_box = _decoder("box")(feats)
    assert isinstance(out_box, dict)
    assert set(out_box) == {"heatmap", "wh", "off"}
    assert out_box["heatmap"].shape == (1, 1, 64, 64)
    assert out_box["wh"].shape == (1, 2, 64, 64)
    assert out_box["off"].shape == (1, 2, 64, 64)


def test_invalid_task_rejected():
    with pytest.raises(ValueError):
        PyramidDecoder(STAGE_CHANNELS["base"], task="segmentation")


# --------------------------------------------------------------------------- #
# 8b. freeze_fusion linear-probe variant
# --------------------------------------------------------------------------- #


def _trainable_names(decoder: PyramidDecoder) -> set:
    return {n for n, p in decoder.named_parameters() if p.requires_grad}


def test_freeze_fusion_leaves_only_head_and_geometry_trainable():
    box = _decoder("box")
    box.freeze_fusion()
    names = _trainable_names(box)
    assert names == {n for n, _ in box.named_parameters()
                     if n.startswith("head.") or n.startswith("geometry.")}
    assert any(n.startswith("geometry.") for n in names)
    assert not any(n.startswith(("laterals.", "blocks.", "refine.")) for n in names)

    point = _decoder("point")
    point.freeze_fusion()
    assert _trainable_names(point) == {"head.weight", "head.bias"}


def test_freeze_fusion_covers_the_stride_two_refine_block():
    dec = PyramidDecoder(STAGE_CHANNELS["base"], output_stride=2, task="box")
    assert dec.refine is not None
    dec.freeze_fusion()
    assert not any(p.requires_grad for p in dec.refine.parameters())
    assert all(p.requires_grad for p in dec.geometry.parameters())


def test_decoder_defaults_are_all_trainable():
    assert all(p.requires_grad for p in _decoder("box").parameters())


# --------------------------------------------------------------------------- #
# 9. box-task batch loss
# --------------------------------------------------------------------------- #


def test_batch_loss_box_task_is_finite_and_reports_n_pos():
    from cropcounter.train import TrainConfig, _batch_loss

    torch.manual_seed(0)
    cfg = TrainConfig(task="box", wh_weight=0.5, off_weight=2.0)
    b, h, w = 2, 16, 16
    outputs = {
        "heatmap": torch.randn(b, 1, h, w, requires_grad=True),
        "wh": torch.randn(b, 2, h, w, requires_grad=True),
        "off": torch.randn(b, 2, h, w, requires_grad=True),
    }
    mask = torch.zeros(b, 1, h, w)
    mask[0, 0, 3, 4] = 1.0
    mask[1, 0, 9, 9] = 1.0
    mask[1, 0, 2, 11] = 1.0
    heat = torch.zeros(b, 1, h, w)
    heat[mask.bool()] = 1.0
    targets = {
        "heatmap": heat,
        "wh": torch.randn(b, 2, h, w),
        "off": torch.rand(b, 2, h, w),
        "mask": mask,
    }
    loss, n_pos = _batch_loss(cfg, outputs, targets)
    assert torch.isfinite(loss)
    assert n_pos == int(mask.sum().item()) == 3
    loss.backward()
    assert torch.isfinite(outputs["wh"].grad).all()
    assert outputs["wh"].grad.abs().sum() > 0
    assert outputs["off"].grad.abs().sum() > 0


def test_batch_loss_point_task_is_plain_focal():
    from cropcounter.losses import penalty_reduced_focal_loss
    from cropcounter.train import TrainConfig, _batch_loss

    torch.manual_seed(1)
    cfg = TrainConfig()
    logits = torch.randn(2, 1, 8, 8)
    targets = torch.zeros(2, 1, 8, 8)
    targets[0, 0, 3, 3] = 1.0
    loss, n_pos = _batch_loss(cfg, logits, targets)
    assert loss.item() == pytest.approx(
        penalty_reduced_focal_loss(logits, targets, alpha=cfg.focal_alpha,
                                   beta=cfg.focal_beta).item(), rel=1e-7
    )
    assert n_pos == 1


# --------------------------------------------------------------------------- #
# 10. CropTileDataset in box mode
# --------------------------------------------------------------------------- #


def _synthetic_coco_dataset(tmp_path: Path, n_empty: int = 1) -> Path:
    """Two 800x600 JPEGs: one with three boxes, one (optionally) with none."""
    import cv2

    root = tmp_path / "brackish"
    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    images, annotations = [], []
    for i in range(2):
        name = f"img{i}.jpg"
        frame = rng.integers(0, 255, (600, 800, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / name), frame)
        images.append({"id": i + 1, "file_name": name, "width": 800, "height": 600,
                       "dataset": "synthetic", "is_train": i == 0})
        if i == 0 or n_empty == 0:
            # A 4x3 lattice at 200 px pitch: any 256 px crop contains a centre,
            # so "retry until a box survives" succeeds without flaking.
            for x in (60.0, 260.0, 460.0, 660.0):
                for y in (60.0, 260.0, 460.0):
                    w, h = 80.0, 60.0
                    annotations.append({
                        "id": len(annotations) + 1, "image_id": i + 1, "category_id": 1,
                        "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
                    })
    payload = {"images": images, "categories": [{"id": 1, "name": "fish"}],
               "annotations": annotations}
    (root / "annotations.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def test_box_dataset_train_tile_shapes(tmp_path):
    from cropcounter.crop_dataset import CropTileDataset, load_records

    root = _synthetic_coco_dataset(tmp_path)
    records = load_records(root, fmt="coco_bbox", labels=None)
    torch.manual_seed(0)
    ds = CropTileDataset(
        records, root / "images", train=True, tile=256, output_stride=4,
        task="box", augment_profile="natural", negative_tile_fraction=0.0,
        tiles_per_image=2,
    )
    image, targets, n_boxes = ds[0]
    assert image.shape == (3, 256, 256)
    assert set(targets) == {"heatmap", "wh", "off", "mask"}
    assert targets["heatmap"].shape == (1, 64, 64)
    assert targets["wh"].shape == (2, 64, 64)
    assert targets["off"].shape == (2, 64, 64)
    assert targets["mask"].shape == (1, 64, 64)
    assert int(targets["mask"].sum().item()) <= n_boxes
    assert n_boxes >= 1, "negative_tile_fraction=0 must keep retrying for a positive crop"


def test_box_dataset_default_collate_stacks_target_dicts(tmp_path):
    from torch.utils.data import DataLoader

    from cropcounter.crop_dataset import CropTileDataset, load_records

    root = _synthetic_coco_dataset(tmp_path)
    records = load_records(root, fmt="coco_bbox", labels=None)
    torch.manual_seed(0)
    ds = CropTileDataset(records, root / "images", train=True, tile=256, task="box",
                         augment_profile="natural", tiles_per_image=2)
    images, targets, n_boxes = next(iter(DataLoader(ds, batch_size=2, num_workers=0)))
    assert images.shape == (2, 3, 256, 256)
    assert targets["heatmap"].shape == (2, 1, 64, 64)
    assert targets["mask"].shape == (2, 1, 64, 64)
    assert n_boxes.shape == (2,)


def test_box_dataset_negative_tile_fraction_draws_empties(tmp_path):
    from cropcounter.crop_dataset import CropTileDataset, load_records

    root = _synthetic_coco_dataset(tmp_path)
    records = load_records(root, fmt="coco_bbox", labels=None)
    torch.manual_seed(0)
    ds = CropTileDataset(records, root / "images", train=True, tile=256, task="box",
                         augment_profile="natural", negative_tile_fraction=1.0,
                         tiles_per_image=4)
    assert all(ds[i][2] == 0 for i in range(len(ds)))

    torch.manual_seed(0)
    ds_pos = CropTileDataset(records, root / "images", train=True, tile=256, task="box",
                             augment_profile="natural", negative_tile_fraction=0.0,
                             tiles_per_image=4)
    assert all(ds_pos[i][2] >= 1 for i in range(len(ds_pos)))


def test_box_dataset_sampler_is_deterministic_given_the_torch_seed(tmp_path):
    from cropcounter.crop_dataset import CropTileDataset, load_records

    root = _synthetic_coco_dataset(tmp_path)
    records = load_records(root, fmt="coco_bbox", labels=None)
    ds = CropTileDataset(records, root / "images", train=True, tile=256, task="box",
                         augment_profile="natural", negative_tile_fraction=0.5,
                         tiles_per_image=4)

    def draw():
        torch.manual_seed(1234)
        return [ds._draw_box_record() for _ in range(32)]

    first = draw()
    assert first == draw()
    # 0.5 negatives over one positive and one empty record: both get drawn.
    assert {expect for _, expect in first} == {True, False}


def test_box_dataset_val_item_carries_boxes_and_image_id(tmp_path):
    from cropcounter.crop_dataset import CropTileDataset, collate_val, load_records

    root = _synthetic_coco_dataset(tmp_path)
    records = load_records(root, fmt="coco_bbox", labels=None)
    ds = CropTileDataset(records, root / "images", train=False, task="box", output_stride=4)
    item = ds[0]
    assert item["image_id"] == 1
    assert item["boxes"].shape == (12, 4)
    assert item["width"] == 800 and item["height"] == 600
    # Padded to a multiple of 32: 800 x 600 -> 800 x 608.
    assert item["image"].shape == (3, 608, 800)
    assert item["target"]["heatmap"].shape == (1, 152, 200)
    assert item["target"]["mask"].sum().item() == 12

    batch = collate_val([item])
    assert batch["image"].shape == (1, 3, 608, 800)
    assert batch["target"]["heatmap"].shape == (1, 1, 152, 200)
    assert batch["image_id"] == 1
    assert batch["boxes"].shape == (12, 4)


def test_point_val_item_and_collate_unchanged(tmp_path):
    import cv2

    from cropcounter.crop_dataset import CropTileDataset, Point, collate_val

    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    cv2.imwrite(str(images_dir / "p.jpg"),
                np.zeros((64, 96, 3), dtype=np.uint8))
    rec = ImageRecord(name="p.jpg", width=96, height=64,
                      points=[Point(x=10.0, y=12.0, label="Wheat")])
    ds = CropTileDataset([rec], images_dir, train=False, output_stride=4)
    item = ds[0]
    assert set(item) == {"image", "target", "points", "name"}
    batch = collate_val([item])
    assert set(batch) == {"image", "target", "points", "name"}
    assert batch["image"].shape == (1, 3, 64, 96)
    assert batch["target"].shape == (1, 1, 16, 24)


def test_natural_profile_has_no_vertical_flip_or_rotate90():
    from cropcounter.crop_dataset import CropTileDataset

    natural = repr(CropTileDataset.default_transform(256, 0.25, profile="natural"))
    wheat = repr(CropTileDataset.default_transform(256, 0.25, profile="wheat"))
    assert "VerticalFlip" not in natural and "RandomRotate90" not in natural
    assert "HorizontalFlip" in natural and "RandomBrightnessContrast" in natural
    assert "VerticalFlip" in wheat and "RandomRotate90" in wheat


# --------------------------------------------------------------------------- #
# 11. det_metrics box matching + COCO scoring
# --------------------------------------------------------------------------- #


def test_match_boxes_iou_counts_tp_fp_fn():
    from cropcounter.det_metrics import match_boxes_iou

    gt = np.array([[0, 0, 10, 10], [100, 100, 120, 120]], dtype=np.float32)
    pred = np.array([[1, 1, 11, 11], [500, 500, 510, 510]], dtype=np.float32)
    tp, fp, fn = match_boxes_iou(pred, gt, iou_thr=0.5)
    assert (tp, fp, fn) == (1, 1, 1)

    assert match_boxes_iou(np.empty((0, 4)), gt) == (0, 0, 2)
    assert match_boxes_iou(pred, np.empty((0, 4))) == (0, 2, 0)
    assert match_boxes_iou(gt, gt) == (2, 0, 0)


def test_summarise_boxes_matches_the_point_summary_shape():
    from cropcounter.det_metrics import summarise_boxes

    rows = [
        {"n_gt": 3, "n_pred": 2, "tp": 2, "fp": 0, "fn": 1},
        {"n_gt": 1, "n_pred": 3, "tp": 1, "fp": 2, "fn": 0},
    ]
    summary = summarise_boxes(rows)
    assert summary["count_mae"] == pytest.approx(1.5)
    assert summary["precision"] == pytest.approx(3 / 5)
    assert summary["recall"] == pytest.approx(3 / 4)
    assert summary["f1"] == pytest.approx(2 * (3 / 5) * (3 / 4) / (3 / 5 + 3 / 4))
    assert summary["n_images"] == 2


def test_match_box_centres_reuses_point_matching():
    from cropcounter.det_metrics import match_box_centres

    gt = np.array([[0, 0, 10, 10]], dtype=np.float32)         # centre (5, 5)
    near = np.array([[2, 2, 12, 12]], dtype=np.float32)       # centre (7, 7)
    assert match_box_centres(near, gt, radius_px=5.0) == (1, 0, 0)
    assert match_box_centres(near, gt, radius_px=1.0) == (0, 1, 1)


def test_coco_eval_empty_detections_returns_zeros():
    from cropcounter.det_metrics import coco_eval

    stats = coco_eval(_coco_gt_dict(ROUND_TRIP_BOXES), [])
    assert stats["ap"] == 0.0 and stats["ap50"] == 0.0 and stats["ar100"] == 0.0
    assert set(stats) >= {"ap", "ap50", "ap75", "ap_small", "ap_medium", "ap_large",
                          "ar1", "ar10", "ar100"}


def test_coco_results_write_read_round_trip(tmp_path):
    from cropcounter.det_metrics import read_coco_results, write_coco_results

    detections = [{"image_id": 1, "category_id": 1, "bbox": [1.0, 2.0, 3.0, 4.0],
                   "score": 0.5}]
    path = write_coco_results(detections, tmp_path / "predictions.json")
    assert read_coco_results(path) == detections
