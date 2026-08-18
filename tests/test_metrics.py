"""Unit tests for cropcounter.metrics — point matching and evaluate()/sweep_tau().

``evaluate``/``sweep_tau`` are exercised with a tiny fake nn.Module that
replays pre-baked logits instead of a real backbone, so this stays CPU-light,
network-free, and independent of the DINOv3 backbone and any real dataset.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from cropcounter.metrics import evaluate, match_points, sweep_tau

# --------------------------------------------------------------------------- #
# match_points
# --------------------------------------------------------------------------- #


def test_match_points_perfect_match():
    pts = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    tp, fp, fn = match_points(pts, pts, radius_px=1.0)
    assert (tp, fp, fn) == (3, 0, 0)


def test_match_points_no_predictions_all_false_negatives():
    gt = np.array([[0.0, 0.0], [10.0, 10.0]])
    tp, fp, fn = match_points(np.empty((0, 2)), gt, radius_px=1.0)
    assert (tp, fp, fn) == (0, 0, 2)


def test_match_points_no_ground_truth_all_false_positives():
    pred = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    tp, fp, fn = match_points(pred, np.empty((0, 2)), radius_px=1.0)
    assert (tp, fp, fn) == (0, 3, 0)


def test_match_points_both_empty():
    tp, fp, fn = match_points(np.empty((0, 2)), np.empty((0, 2)), radius_px=1.0)
    assert (tp, fp, fn) == (0, 0, 0)


def test_match_points_respects_radius_gate():
    pred = np.array([[0.0, 0.0]])
    gt = np.array([[5.0, 0.0]])  # distance exactly 5
    tp, fp, fn = match_points(pred, gt, radius_px=4.0)
    assert (tp, fp, fn) == (0, 1, 1)  # too far to match
    tp, fp, fn = match_points(pred, gt, radius_px=6.0)
    assert (tp, fp, fn) == (1, 0, 0)  # within radius


def test_match_points_is_one_to_one():
    # Two predictions both close to a single ground-truth point: only one
    # can claim it; the other counts as an unmatched false positive.
    pred = np.array([[0.0, 0.0], [0.1, 0.0]])
    gt = np.array([[0.0, 0.0]])
    tp, fp, fn = match_points(pred, gt, radius_px=1.0)
    assert (tp, fp, fn) == (1, 1, 0)


def test_match_points_needs_global_optimum_not_greedy_nearest_edge():
    # A is only within radius of X; B is within radius of BOTH X and Y.
    # A greedy "take the closest edge first" walk assigns B-X (dist 0.4,
    # the single closest edge overall) before A gets a turn, stranding A
    # (its only in-radius option, X, is gone) -> tp=1. The globally optimal
    # (Hungarian) assignment matches A-X and B-Y instead -> tp=2. This is
    # exactly the failure mode one-to-one Hungarian matching is for.
    gt = np.array([[0.0, 0.0], [0.8, 0.0]])       # X, Y
    pred = np.array([[-0.9, 0.0], [0.4, 0.0]])    # A, B
    tp, fp, fn = match_points(pred, gt, radius_px=1.0)
    assert (tp, fp, fn) == (2, 0, 0)


# --------------------------------------------------------------------------- #
# evaluate() / sweep_tau() — synthetic model + loader, no backbone/dataset
# --------------------------------------------------------------------------- #


class _ReplayLogitsModel(nn.Module):
    """Ignores its input; yields one pre-baked logits tensor per forward call.

    Stands in for CropCounter so evaluate()/sweep_tau() can be exercised
    without the DINOv3 backbone or any trained decoder.
    """

    def __init__(self, logits_sequence):
        super().__init__()
        self._logits = list(logits_sequence)
        self._i = 0

    def forward(self, x):
        out = self._logits[self._i]
        self._i += 1
        return out


def _spike_logits(grid_hw, points_xy, high=20.0, low=-20.0):
    """(1, 1, H, W) logits: `high` at each (x, y) cell, `low` everywhere else."""
    h, w = grid_hw
    logits = torch.full((1, 1, h, w), low)
    for x, y in points_xy:
        logits[0, 0, int(y), int(x)] = high
    return logits


def _batch(name, gt_points_xy):
    return {
        "image": torch.zeros(1, 3, 4, 4),  # unused by _ReplayLogitsModel
        "target": None,
        "points": np.asarray(gt_points_xy, dtype=np.float32),
        "name": name,
    }


def test_evaluate_on_toy_two_image_batch():
    grid = (16, 16)
    # Image 1: both ground-truth points detected exactly.
    img1_logits = _spike_logits(grid, [(4, 4), (10, 10)])
    # Image 2: 3 ground truth, only 2 detected (one missed -> a false negative).
    img2_logits = _spike_logits(grid, [(2, 2), (8, 8)])

    model = _ReplayLogitsModel([img1_logits, img2_logits])
    loader = [
        _batch("img1", [(4.5, 4.5), (10.5, 10.5)]),
        _batch("img2", [(2.5, 2.5), (8.5, 8.5), (13.5, 13.5)]),
    ]

    summary, rows = evaluate(
        model, loader, torch.device("cpu"),
        tau=0.3, k=3, nms_radius=1.5, output_stride=1, match_radius_px=2.0,
    )

    assert summary["n_images"] == 2
    assert rows[0] == {"name": "img1", "n_gt": 2, "n_pred": 2, "tp": 2, "fp": 0, "fn": 0}
    assert rows[1] == {"name": "img2", "n_gt": 3, "n_pred": 2, "tp": 2, "fp": 0, "fn": 1}

    # counts: [2, 3] gt vs [2, 2] pred -> err = [0, -1]
    assert summary["count_mae"] == pytest.approx(0.5)
    assert summary["count_rmse"] == pytest.approx(math.sqrt(0.5))
    assert summary["count_bias"] == pytest.approx(-0.5)

    # tp=4, fp=0, fn=1
    assert summary["precision"] == pytest.approx(1.0)
    assert summary["recall"] == pytest.approx(0.8)
    assert summary["f1"] == pytest.approx(2 * 1.0 * 0.8 / 1.8)

    # No `target` in either batch -> loss is never computed.
    assert math.isnan(summary["val_loss"])


def test_sweep_tau_single_pass_reflects_threshold():
    grid = (16, 16)
    h, w = grid
    logits = torch.full((1, 1, h, w), -20.0)
    logits[0, 0, 4, 4] = 20.0   # strong peak: sigmoid ~= 1.0
    logits[0, 0, 10, 10] = 1.0  # weak peak: sigmoid(1.0) ~= 0.731

    model = _ReplayLogitsModel([logits])
    loader = [_batch("only", [(4.5, 4.5), (10.5, 10.5)])]

    summaries = sweep_tau(
        model, loader, torch.device("cpu"), taus=[0.3, 0.8],
        k=3, nms_radius=1.5, output_stride=1, match_radius_px=2.0,
    )
    by_tau = {s["tau"]: s for s in summaries}

    # Low threshold: both peaks clear it.
    assert by_tau[0.3]["f1"] == pytest.approx(1.0)
    assert by_tau[0.3]["count_mae"] == pytest.approx(0.0)

    # High threshold: only the strong peak survives -> one point missed.
    assert by_tau[0.8]["count_mae"] == pytest.approx(1.0)
    assert by_tau[0.8]["recall"] == pytest.approx(0.5)
    assert by_tau[0.8]["f1"] == pytest.approx(2 * 1.0 * 0.5 / 1.5)
