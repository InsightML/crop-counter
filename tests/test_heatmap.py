"""Unit tests for cropcounter.heatmap — target rendering, NMS, and peak decode.

Pure numpy/torch on synthetic points; no backbone, no dataset, no network.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.spatial.distance import cdist

from cropcounter.heatmap import decode_peaks, point_nms, render_targets

# --------------------------------------------------------------------------- #
# render_targets
# --------------------------------------------------------------------------- #


def test_render_targets_empty_points_is_all_zero():
    heat = render_targets(np.empty((0, 2)), (16, 16), sigma=2.0)
    assert heat.shape == (16, 16)
    assert heat.dtype == np.float32
    assert heat.max() == 0.0


def test_render_targets_peak_is_exactly_one():
    heat = render_targets(np.array([[8.0, 8.0]]), (16, 16), sigma=2.0)
    assert heat[8, 8] == pytest.approx(1.0)
    assert heat.max() == pytest.approx(1.0)


def test_render_targets_max_compose_keeps_distinct_peaks():
    # Two points 4 cells apart, sigma=2: both peaks stay exactly 1.0 and the
    # saddle between them dips below 1.0 (never summed to something taller).
    heat = render_targets(np.array([[4.0, 8.0], [8.0, 8.0]]), (16, 16), sigma=2.0)
    assert heat[8, 4] == pytest.approx(1.0)
    assert heat[8, 8] == pytest.approx(1.0)
    assert heat[8, 6] < 1.0
    assert heat.max() <= 1.0 + 1e-6


def test_render_targets_clips_points_to_grid_bounds():
    # A point outside the grid still contributes a (clipped) stamp rather
    # than raising or silently vanishing.
    heat = render_targets(np.array([[-5.0, 20.0]]), (16, 16), sigma=2.0)
    assert heat.max() == pytest.approx(1.0)
    assert heat[15, 0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# point_nms
# --------------------------------------------------------------------------- #


def test_point_nms_empty_input():
    keep = point_nms(np.empty((0, 2)), np.empty(0), radius=1.0)
    assert keep.shape == (0,)


def test_point_nms_drops_near_duplicates_keeps_higher_score():
    pts = np.array([[0.0, 0.0], [0.5, 0.5], [10.0, 10.0]])
    scores = np.array([0.9, 0.8, 0.95])
    keep = point_nms(pts, scores, radius=1.5)
    assert len(keep) == 2
    assert 0 in keep  # beats index 1 (lower score, within radius)
    assert 2 in keep
    assert 1 not in keep


def test_point_nms_keeps_points_beyond_radius():
    pts = np.array([[0.0, 0.0], [5.0, 0.0]])
    scores = np.array([0.5, 0.9])
    keep = point_nms(pts, scores, radius=1.5)
    assert set(keep.tolist()) == {0, 1}


# --------------------------------------------------------------------------- #
# decode_peaks
# --------------------------------------------------------------------------- #


def test_decode_peaks_all_below_threshold_returns_empty():
    heat = np.zeros((16, 16), dtype=np.float32)
    pts, scores = decode_peaks(heat, tau=0.3)
    assert pts.shape == (0, 2)
    assert scores.shape == (0,)


def test_decode_peaks_accepts_2d_array_and_batched_tensor():
    heat_np = render_targets(np.array([[8.0, 8.0]]), (16, 16), sigma=2.0)
    pts_from_2d, _ = decode_peaks(heat_np, tau=0.3)
    heat_t = torch.from_numpy(heat_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    pts_from_4d, _ = decode_peaks(heat_t, tau=0.3)
    assert len(pts_from_2d) == 1
    np.testing.assert_allclose(pts_from_2d, pts_from_4d)


def test_decode_peaks_rejects_multi_image_batch():
    heat = torch.zeros(2, 1, 16, 16)
    with pytest.raises(ValueError):
        decode_peaks(heat, tau=0.3)


def test_decode_peaks_round_trip_recovers_synthetic_points():
    """render_targets -> decode_peaks should recover every well-separated
    point within about one grid cell, with no spurious extras — mirroring
    the target/decode round-trip check this repo runs on the real dataset."""
    grid = (64, 64)
    gt_points = np.array(
        [
            [8, 8], [24, 8], [40, 8],
            [8, 24], [24, 24], [40, 24],
            [16, 48], [48, 48],
        ],
        dtype=np.float32,
    )
    heat = render_targets(gt_points, grid, sigma=2.0)
    pred_points, scores = decode_peaks(heat, k=3, tau=0.3, nms_radius=1.5, stride=1)

    assert len(pred_points) == len(gt_points)
    nearest = cdist(pred_points, gt_points).min(axis=1)
    assert (nearest < 1.5).all()
    # decode_peaks promises descending scores.
    assert (np.diff(scores) <= 1e-6).all()


def test_decode_peaks_tau_and_k_gate_detections():
    # Two points 6 cells apart: k=3's separation floor and a mid threshold
    # both resolve them; a very tight local-max window doesn't lose either.
    grid = (32, 32)
    points = np.array([[10.0, 16.0], [16.0, 16.0]])
    heat = render_targets(points, grid, sigma=1.0)

    pts, scores = decode_peaks(heat, k=3, tau=0.9, nms_radius=1.5, stride=1)
    assert len(pts) == 2
    assert (scores > 0.9).all()

    # Raising tau above the achievable peak value drops everything.
    pts_none, _ = decode_peaks(heat, k=3, tau=1.5, nms_radius=1.5, stride=1)
    assert len(pts_none) == 0
