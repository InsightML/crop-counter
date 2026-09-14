"""Unit tests for multiclass support: config, targets, decode, dataset, metrics.

CPU-light and network-free like the rest of the suite: no DINOv3 backbone
(the decoder is exercised on its own with fake stage maps), no real dataset
(a tiny synthetic PNG on ``tmp_path``), no live calls. The one on-disk
artifact used is the shipped ``weights/decoder_best.pt`` decoder checkpoint,
to prove a pre-multiclass state dict still loads.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2

from cropcounter.crop_dataset import (
    COUNTED_LABELS,
    WILDCARD_CLASS,
    CropTileDataset,
    ImageRecord,
    Point,
    collate_val,
    parse_cvat_1_1,
)
from cropcounter.dinov3_pyramid import IMAGENET_MEAN, IMAGENET_STD, STAGE_CHANNELS, PyramidDecoder
from cropcounter.heatmap import decode_peaks, per_class_values, render_class_targets, render_targets
from cropcounter.inference import decode_classes, decode_in_bounds, save_visualization, write_cvat_xml
from cropcounter.metrics import evaluate, sweep_tau
from cropcounter.train import TrainConfig

REPO = Path(__file__).resolve().parents[1]
SHIPPED_DECODER = REPO / "weights" / "decoder_best.pt"

# --------------------------------------------------------------------------- #
# TrainConfig: classes, per-class decode params, and the labels deprecation
# --------------------------------------------------------------------------- #


def test_default_config_is_the_single_wheat_class():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the default path must not warn
        cfg = TrainConfig()
    assert cfg.classes == {"Wheat": COUNTED_LABELS}
    assert cfg.class_names == ("Wheat",)
    assert cfg.n_classes == 1
    assert cfg.class_map == {"Wheat": 0, "Volunteer": 0}
    # The derived legacy view still reads like before.
    assert cfg.labels == COUNTED_LABELS


def test_config_without_labels_or_classes_keys_uses_wheat_default():
    """Old checkpoints (pre-`labels`) carry neither key and must still load silently."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = TrainConfig.from_dict({"tau": 0.3, "k": 3, "epochs": 50})
    assert cfg.classes == {"Wheat": COUNTED_LABELS}
    assert cfg.n_classes == 1


def test_legacy_labels_warns_and_becomes_one_class():
    with pytest.warns(DeprecationWarning):
        cfg = TrainConfig.from_dict({"labels": ["Tassel"]})
    assert cfg.classes == {"Tassel": ("Tassel",)}
    assert cfg.class_names == ("Tassel",)
    assert cfg.labels == ("Tassel",)
    # Canonical form only on the way out.
    out = cfg.to_dict()
    assert out["classes"] == {"Tassel": ["Tassel"]}
    assert "labels" not in out


def test_legacy_labels_none_is_the_wildcard_class():
    with pytest.warns(DeprecationWarning):
        cfg = TrainConfig(labels=None)
    assert cfg.classes is None
    assert cfg.class_names == (WILDCARD_CLASS,)
    assert cfg.n_classes == 1
    assert cfg.class_map is None
    assert cfg.labels is None
    assert cfg.to_dict()["classes"] is None


def test_labels_and_classes_together_are_rejected():
    with pytest.raises(ValueError):
        TrainConfig(labels=("Wheat",), classes={"Wheat": ("Wheat",)})


def test_classes_round_trip_keeps_order_grouping_and_per_class_params(tmp_path):
    cfg = TrainConfig(
        classes={"Wheat": ["Wheat", "Volunteer"], "Beans": "Beans"},  # bare str tolerated
        tau={"Wheat": 0.35, "Beans": 0.25},
        nms_radius=1.5,
    )
    assert cfg.classes == {"Wheat": ("Wheat", "Volunteer"), "Beans": ("Beans",)}
    assert cfg.class_names == ("Wheat", "Beans")
    assert cfg.class_map == {"Wheat": 0, "Volunteer": 0, "Beans": 1}
    assert cfg.labels == ("Wheat", "Volunteer", "Beans")

    out = cfg.to_dict()
    assert out["classes"] == {"Wheat": ["Wheat", "Volunteer"], "Beans": ["Beans"]}
    assert out["tau"] == {"Wheat": 0.35, "Beans": 0.25}
    assert out["nms_radius"] == 1.5

    restored = TrainConfig.from_json(cfg.to_json(tmp_path / "cfg.json"))
    assert restored.classes == cfg.classes
    assert restored.class_names == ("Wheat", "Beans")  # JSON keeps key order
    assert restored.tau == {"Wheat": 0.35, "Beans": 0.25}
    assert restored.to_dict() == out


def test_duplicate_label_across_classes_is_rejected():
    with pytest.raises(ValueError, match="appears in classes"):
        TrainConfig(classes={"Wheat": ("Wheat",), "Other": ("Wheat", "Beans")})


def test_empty_classes_or_empty_label_list_rejected():
    with pytest.raises(ValueError):
        TrainConfig(classes={})
    with pytest.raises(ValueError):
        TrainConfig(classes={"Wheat": ()})


def test_per_class_param_keys_must_match_classes():
    with pytest.raises(ValueError, match="do not match classes"):
        TrainConfig(classes={"Wheat": ("Wheat",), "Beans": ("Beans",)}, tau={"Wheat": 0.3})
    with pytest.raises(ValueError, match="do not match classes"):
        TrainConfig(classes={"Wheat": ("Wheat",)}, k={"Wheat": 3, "Beans": 5})


def test_per_class_values_broadcasts_scalars_and_orders_dicts():
    assert per_class_values(0.3, ("a", "b")) == (0.3, 0.3)
    assert per_class_values({"b": 2, "a": 1}, ("a", "b")) == (1, 2)
    with pytest.raises(ValueError):
        per_class_values({"a": 1}, ("a", "b"))


# --------------------------------------------------------------------------- #
# render_class_targets
# --------------------------------------------------------------------------- #


def test_render_class_targets_one_channel_per_class():
    pts = np.array([[4.0, 4.0], [10.0, 10.0]])
    heat = render_class_targets(pts, [0, 1], n_classes=3, out_hw=(16, 16), sigma=1.0)
    assert heat.shape == (3, 16, 16) and heat.dtype == np.float32
    assert heat[0, 4, 4] == pytest.approx(1.0) and heat[0, 10, 10] == 0.0
    assert heat[1, 10, 10] == pytest.approx(1.0) and heat[1, 4, 4] == 0.0
    assert heat[2].max() == 0.0  # a class with no points still gets its plane


def test_render_class_targets_single_class_equals_render_targets():
    pts = np.array([[3.0, 5.0], [9.0, 2.0]])
    stacked = render_class_targets(pts, [0, 0], n_classes=1, out_hw=(12, 12), sigma=2.0)
    np.testing.assert_array_equal(stacked[0], render_targets(pts, (12, 12), 2.0))


def test_render_class_targets_validates_ids():
    with pytest.raises(ValueError):
        render_class_targets(np.zeros((2, 2)), [0], n_classes=1, out_hw=(8, 8), sigma=1.0)
    with pytest.raises(ValueError):
        render_class_targets(np.zeros((1, 2)), [1], n_classes=1, out_hw=(8, 8), sigma=1.0)


# --------------------------------------------------------------------------- #
# decode_classes / decode_in_bounds
# --------------------------------------------------------------------------- #


def _two_channel_prob(h=16, w=16):
    """Channel 0: peaks 0.9 @ (4,4) and 0.5 @ (10,4). Channel 1: peak 0.6 @ (8,8)."""
    prob = torch.zeros(1, 2, h, w)
    prob[0, 0, 4, 4] = 0.9
    prob[0, 0, 4, 10] = 0.5
    prob[0, 1, 8, 8] = 0.6
    return prob


def test_decode_classes_applies_per_class_tau_and_tags_class_ids():
    prob = _two_channel_prob()
    pts, scores, ids = decode_classes(
        prob, ("a", "b"), tau={"a": 0.7, "b": 0.5}, k=3, nms_radius=1.5, output_stride=1
    )
    assert ids.tolist() == [0, 1]
    np.testing.assert_allclose(pts, [[4.5, 4.5], [8.5, 8.5]])
    np.testing.assert_allclose(scores, [0.9, 0.6])

    # A scalar tau applies to every class.
    pts, scores, ids = decode_classes(prob, ("a", "b"), tau=0.4, output_stride=1)
    assert ids.tolist() == [0, 0, 1]
    assert (np.diff(scores[ids == 0]) <= 0).all()  # descending within a class


def test_decode_classes_accepts_3d_and_scales_by_stride():
    prob = _two_channel_prob()
    pts, _, ids = decode_classes(prob[0], ("a", "b"), tau=0.55, output_stride=4)
    np.testing.assert_allclose(pts, [[18.0, 18.0], [34.0, 34.0]])
    assert ids.tolist() == [0, 1]


def test_decode_classes_bounds_filter_drops_pad_strip_points():
    prob = _two_channel_prob()
    pts, scores, ids = decode_classes(
        prob, ("a", "b"), tau=0.4, output_stride=1, width=8, height=16
    )
    # x >= 8 removed: the (10,4) class-a peak and the (8,8) class-b peak.
    assert pts.tolist() == [[4.5, 4.5]] and ids.tolist() == [0] and len(scores) == 1


def test_decode_classes_single_class_matches_decode_peaks():
    heat = render_targets(np.array([[3.0, 3.0], [11.0, 9.0]]), (16, 16), sigma=2.0)
    ref_pts, ref_scores = decode_peaks(heat, k=3, tau=0.3, nms_radius=1.5, stride=4)
    pts, scores, ids = decode_classes(heat[None], (WILDCARD_CLASS,), tau=0.3, output_stride=4)
    np.testing.assert_allclose(pts, ref_pts)
    np.testing.assert_allclose(scores, ref_scores)
    assert (ids == 0).all()


def test_decode_classes_rejects_channel_count_mismatch_and_batches():
    with pytest.raises(ValueError):
        decode_classes(torch.zeros(1, 2, 8, 8), ("only",))
    with pytest.raises(ValueError):
        decode_classes(torch.zeros(2, 1, 8, 8), ("only",))


def test_decode_in_bounds_is_deprecated_but_unchanged():
    heat = render_targets(np.array([[3.0, 3.0], [14.0, 14.0]]), (16, 16), sigma=1.5)
    prob = torch.from_numpy(heat)[None, None]
    with pytest.warns(DeprecationWarning):
        pts, scores = decode_in_bounds(prob, width=40, height=40, tau=0.3, output_stride=4)
    new_pts, new_scores, _ = decode_classes(
        prob, (WILDCARD_CLASS,), tau=0.3, output_stride=4, width=40, height=40
    )
    np.testing.assert_allclose(pts, new_pts)
    np.testing.assert_allclose(scores, new_scores)
    assert len(pts) == 1  # (14,14)*4 = 58 px lies outside the 40 px frame


# --------------------------------------------------------------------------- #
# CropTileDataset: class ids survive augmentation, filtering, and val batches
# --------------------------------------------------------------------------- #

CLASS_MAP = {"Wheat": 0, "Volunteer": 0, "Beans": 1}


@pytest.fixture
def synthetic_images(tmp_path):
    """A 64x64 RGB image and a record with points of three labels + one unknown."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    rng = np.random.default_rng(0)
    cv2.imwrite(str(images_dir / "img.png"), rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    record = ImageRecord(
        name="img.png", width=64, height=64,
        points=[
            Point(10.0, 10.0, "Wheat"),
            Point(50.0, 50.0, "Beans"),
            Point(50.0, 10.0, "Volunteer"),
            Point(30.0, 30.0, "Weed"),  # not in the class map -> dropped
        ],
    )
    return [record], images_dir


def _deterministic_crop(x0, y0, size):
    return A.Compose(
        [
            A.Crop(x_min=x0, y_min=y0, x_max=x0 + size, y_max=y0 + size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(
            format="xy", remove_invisible=True, label_fields=["class_ids"]
        ),
    )


def test_train_tile_class_ids_stay_aligned_when_earlier_points_are_cropped(synthetic_images):
    records, images_dir = synthetic_images
    # Crop the bottom-right 32x32: only the Beans point (50,50) survives, and
    # it is the *second* point — a misaligned label list would tag it Wheat.
    ds = CropTileDataset(
        records, images_dir, train=True, tile=32, output_stride=4, sigma=1.0,
        tiles_per_image=1, class_map=CLASS_MAP, transform=_deterministic_crop(32, 32, 32),
    )
    assert ds.n_classes == 2
    assert ds.class_ids[0].tolist() == [0, 1, 0]  # Weed dropped, order kept
    image, target, n_pts = ds[0]
    assert image.shape == (3, 32, 32)
    assert target.shape == (2, 8, 8)
    assert n_pts == 1
    assert target[0].max() == 0.0                      # no Wheat in this tile
    assert target[1, 4, 4] == pytest.approx(1.0)       # (50-32)/4 = 4.5 -> cell 4


def test_train_tile_keeps_every_class_when_nothing_is_cropped(synthetic_images):
    records, images_dir = synthetic_images
    ds = CropTileDataset(
        records, images_dir, train=True, tile=64, output_stride=4, sigma=1.0,
        tiles_per_image=1, class_map=CLASS_MAP, transform=_deterministic_crop(0, 0, 64),
    )
    _, target, n_pts = ds[0]
    assert n_pts == 3 and target.shape == (2, 16, 16)
    assert target[0, 2, 2] == pytest.approx(1.0)       # Wheat (10,10)
    assert target[0, 2, 12] == pytest.approx(1.0)      # Volunteer (50,10) -> channel 0
    assert target[1, 12, 12] == pytest.approx(1.0)     # Beans (50,50)
    assert target[1, 2, 2] == 0.0


def test_default_transform_declares_class_id_label_field():
    params = CropTileDataset.default_transform(tile=32).processors["keypoints"].params
    assert "class_ids" in params.label_fields


def test_val_item_and_collate_carry_class_ids(synthetic_images):
    records, images_dir = synthetic_images
    ds = CropTileDataset(records, images_dir, train=False, output_stride=4, sigma=1.0,
                         class_map=CLASS_MAP)
    item = ds[0]
    assert item["target"].shape == (2, 16, 16)
    assert item["points"].shape == (3, 2)
    assert item["class_ids"].tolist() == [0, 1, 0]
    batch = collate_val([item])
    assert batch["target"].shape == (1, 2, 16, 16)
    assert batch["class_ids"].tolist() == [0, 1, 0]


def test_dataset_without_class_map_is_single_channel_wildcard(synthetic_images):
    records, images_dir = synthetic_images
    ds = CropTileDataset(records, images_dir, train=False, output_stride=4)
    assert ds.n_classes == 1
    assert ds.class_ids[0].tolist() == [0, 0, 0, 0]  # every label, channel 0
    assert ds[0]["target"].shape == (1, 16, 16)


def test_dataset_n_classes_covers_a_class_absent_from_the_split(synthetic_images):
    records, images_dir = synthetic_images
    ds = CropTileDataset(records, images_dir, train=False, output_stride=4,
                         class_map={"Wheat": 0}, n_classes=3)
    assert ds[0]["target"].shape == (3, 16, 16)
    with pytest.raises(ValueError):
        CropTileDataset(records, images_dir, train=False, class_map={"Wheat": 5}, n_classes=2)


# --------------------------------------------------------------------------- #
# PyramidDecoder: head width, and backward compatibility of old checkpoints
# --------------------------------------------------------------------------- #


def _fake_stage_maps(size="base", hw=64):
    return [
        torch.zeros(1, c, hw // s, hw // s)
        for c, s in zip(STAGE_CHANNELS[size], (4, 8, 16, 32))
    ]


def test_decoder_emits_one_channel_per_class_with_focal_prior():
    dec = PyramidDecoder(STAGE_CHANNELS["base"], n_classes=3)
    assert dec.head.weight.shape == (3, 192, 1, 1)
    assert torch.all(dec.head.bias == -4.0)
    out = dec(_fake_stage_maps())
    assert out.shape == (1, 3, 16, 16)
    with pytest.raises(ValueError):
        PyramidDecoder(STAGE_CHANNELS["base"], n_classes=0)


@pytest.mark.skipif(not SHIPPED_DECODER.exists(), reason="shipped decoder checkpoint absent")
def test_shipped_single_class_decoder_loads_into_default_decoder():
    payload = torch.load(SHIPPED_DECODER, map_location="cpu", weights_only=False)
    cfg = TrainConfig.from_dict(dict(payload["config"]))  # has neither labels nor classes
    assert cfg.n_classes == 1
    dec = PyramidDecoder(STAGE_CHANNELS[cfg.backbone], c_dec=cfg.c_dec,
                         output_stride=cfg.output_stride, n_classes=cfg.n_classes)
    dec.load_state_dict(payload["decoder"])  # strict: every tensor, same shapes
    assert dec.head.weight.shape == (1, 192, 1, 1)
    # ... and a 2-class decoder refuses it loudly rather than silently.
    with pytest.raises(RuntimeError):
        PyramidDecoder(STAGE_CHANNELS["base"], n_classes=2).load_state_dict(payload["decoder"])


# --------------------------------------------------------------------------- #
# metrics: per-class matching, macro summary, finite empty classes
# --------------------------------------------------------------------------- #


class _ReplayLogitsModel(nn.Module):
    def __init__(self, logits_sequence):
        super().__init__()
        self._logits = list(logits_sequence)
        self._i = 0

    def forward(self, x):
        out = self._logits[self._i]
        self._i += 1
        return out


def _spikes(grid_hw, per_class_points, n_classes, high=20.0, low=-20.0):
    """(1, C, H, W) logits with `high` at each class's (x, y) cells."""
    h, w = grid_hw
    logits = torch.full((1, n_classes, h, w), low)
    for c, pts in per_class_points.items():
        for x, y in pts:
            logits[0, c, int(y), int(x)] = high
    return logits


def _batch(name, gt_points, gt_ids):
    return {
        "image": torch.zeros(1, 3, 4, 4),
        "target": None,
        "points": np.asarray(gt_points, dtype=np.float32).reshape(-1, 2),
        "class_ids": np.asarray(gt_ids, dtype=np.int64),
        "name": name,
    }


def test_evaluate_reports_per_class_and_macro_metrics():
    grid = (16, 16)
    logits = _spikes(grid, {0: [(4, 4), (10, 10)], 1: [(2, 8)]}, n_classes=2)
    model = _ReplayLogitsModel([logits])
    # Class a: both found. Class b: 2 GT, 1 found -> one false negative.
    loader = [_batch("img", [(4.5, 4.5), (10.5, 10.5), (2.5, 8.5), (12.5, 12.5)], [0, 0, 1, 1])]

    summary, rows = evaluate(
        model, loader, torch.device("cpu"), tau=0.3, k=3, nms_radius=1.5,
        output_stride=1, match_radius_px=2.0, class_names=("a", "b"),
    )
    a, b = summary["per_class"]["a"], summary["per_class"]["b"]
    assert a["count_mae"] == 0 and a["f1"] == pytest.approx(1.0)
    assert b["count_mae"] == 1 and b["precision"] == 1.0 and b["recall"] == 0.5
    assert b["f1"] == pytest.approx(2 * 1.0 * 0.5 / 1.5)
    # Top-level keys are the macro mean over classes.
    assert summary["count_mae"] == pytest.approx(0.5)
    assert summary["f1"] == pytest.approx((1.0 + 2 * 0.5 / 1.5) / 2)
    assert summary["n_images"] == 1
    # Rows keep the per-image totals plus the breakdown.
    row = rows[0]
    assert (row["n_gt"], row["n_pred"], row["tp"], row["fp"], row["fn"]) == (4, 3, 3, 0, 1)
    assert row["per_class"]["b"] == {"n_gt": 2, "n_pred": 1, "tp": 1, "fp": 0, "fn": 1}


def test_evaluate_never_matches_across_classes():
    grid = (16, 16)
    # The only prediction is class b, sitting exactly on the class-a ground truth.
    logits = _spikes(grid, {1: [(4, 4)]}, n_classes=2)
    summary, _ = evaluate(
        _ReplayLogitsModel([logits]), [_batch("img", [(4.5, 4.5)], [0])],
        torch.device("cpu"), tau=0.3, output_stride=1, match_radius_px=2.0,
        class_names=("a", "b"),
    )
    assert summary["per_class"]["a"]["recall"] == 0.0     # missed
    assert summary["per_class"]["b"]["precision"] == 0.0  # a false positive


def test_evaluate_stays_finite_for_a_class_with_no_ground_truth_or_predictions():
    grid = (16, 16)
    logits = _spikes(grid, {0: [(4, 4)]}, n_classes=2)  # class b silent, no GT either
    summary, _ = evaluate(
        _ReplayLogitsModel([logits]), [_batch("img", [(4.5, 4.5)], [0])],
        torch.device("cpu"), tau=0.3, output_stride=1, match_radius_px=2.0,
        class_names=("a", "b"),
    )
    b = summary["per_class"]["b"]
    assert all(np.isfinite(v) for v in b.values())
    assert b["count_mae"] == 0.0 and b["f1"] == 0.0
    assert np.isfinite(summary["f1"]) and summary["f1"] == pytest.approx(0.5)


def test_evaluate_honours_per_class_tau_dict():
    grid = (16, 16)
    logits = torch.full((1, 2, *grid), -20.0)
    logits[0, 0, 4, 4] = 1.0   # class a weak peak, sigmoid ~ 0.73
    logits[0, 1, 8, 8] = 1.0   # class b weak peak, sigmoid ~ 0.73
    loader = [_batch("img", [(4.5, 4.5), (8.5, 8.5)], [0, 1])]
    summary, _ = evaluate(
        _ReplayLogitsModel([logits]), loader, torch.device("cpu"),
        tau={"a": 0.3, "b": 0.9}, output_stride=1, match_radius_px=2.0,
        class_names=("a", "b"),
    )
    assert summary["per_class"]["a"]["recall"] == 1.0
    assert summary["per_class"]["b"]["recall"] == 0.0


def test_single_class_evaluate_without_class_ids_is_unchanged():
    """A loader with no class_ids (pre-multiclass) evaluates exactly as before."""
    grid = (16, 16)
    logits = _spikes(grid, {0: [(4, 4)]}, n_classes=1)
    batch = _batch("img", [(4.5, 4.5), (10.5, 10.5)], [0, 0])
    del batch["class_ids"]
    summary, rows = evaluate(
        _ReplayLogitsModel([logits]), [batch], torch.device("cpu"),
        tau=0.3, output_stride=1, match_radius_px=2.0,
    )
    assert summary["per_class"][WILDCARD_CLASS]["recall"] == 0.5
    assert summary["recall"] == 0.5 and summary["count_mae"] == 1.0
    assert (rows[0]["n_gt"], rows[0]["n_pred"], rows[0]["tp"]) == (2, 1, 1)


def test_sweep_tau_exposes_per_class_optima_from_one_pass():
    grid = (16, 16)
    logits = torch.full((1, 2, *grid), -20.0)
    logits[0, 0, 4, 4] = 20.0   # class a: strong
    logits[0, 0, 10, 10] = 1.0  # class a: weak (~0.73) with no GT -> a false positive
    logits[0, 1, 8, 8] = 1.0    # class b: weak (~0.73) with GT
    loader = [_batch("img", [(4.5, 4.5), (8.5, 8.5)], [0, 1])]
    rows = sweep_tau(
        _ReplayLogitsModel([logits]), loader, torch.device("cpu"), taus=[0.3, 0.8],
        output_stride=1, match_radius_px=2.0, class_names=("a", "b"),
    )
    by_tau = {r["tau"]: r["per_class"] for r in rows}
    # Class a is best at the high tau (kills its false positive)...
    assert by_tau[0.8]["a"]["f1"] > by_tau[0.3]["a"]["f1"]
    # ...while class b needs the low one — separable per class.
    assert by_tau[0.3]["b"]["f1"] > by_tau[0.8]["b"]["f1"]


# --------------------------------------------------------------------------- #
# inference exports
# --------------------------------------------------------------------------- #


def test_write_cvat_xml_per_point_labels_round_trip(tmp_path):
    preds = [{
        "name": "a.jpg", "width": 100, "height": 100,
        "points": np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]),
        "scores": np.array([0.9, 0.8, 0.7]),
        "labels": ["Wheat", "Beans", "Wheat"],
    }]
    path = write_cvat_xml(preds, tmp_path / "pred.xml")
    (rec,) = parse_cvat_1_1(path, labels=None)
    assert [p.label for p in rec.points] == ["Wheat", "Beans", "Wheat"]
    assert [p.confidence for p in rec.points] == [90, 80, 70]


def test_write_cvat_xml_rejects_mismatched_labels(tmp_path):
    preds = [{"name": "a.jpg", "width": 10, "height": 10,
              "points": np.zeros((2, 2)), "scores": np.zeros(2), "labels": ["x"]}]
    with pytest.raises(ValueError):
        write_cvat_xml(preds, tmp_path / "pred.xml")


def test_save_visualization_colours_points_by_class(tmp_path):
    image = torch.zeros(3, 64, 64)
    prob = torch.zeros(1, 2, 16, 16)
    prob[0, 0, 2, 2] = 0.9
    prob[0, 1, 8, 8] = 0.7
    pts = np.array([[10.0, 10.0], [34.0, 34.0]])
    out = save_visualization(
        image, prob, pts, 64, 64, tmp_path / "viz.png", output_stride=4,
        class_ids=np.array([0, 1]), class_names=("Wheat", "Beans"),
    )
    assert out.exists() and out.stat().st_size > 0
