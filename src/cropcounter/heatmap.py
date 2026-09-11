"""Heatmap target rendering and peak decoding for point-based counting.

Targets are CenterNet-style peak-normalised Gaussians (peak value 1.0)
composed with element-wise max, so nearby points keep distinct maxima.
Decoding is a max-pool local-max test, a score threshold ``tau``, and a
greedy point-NMS that removes plateau ties and duplicate peaks.

All coordinates here are (x, y). "Grid coordinates" means the output-stride
grid of the model head; ``decode_peaks`` converts back to input pixels.
"""
from __future__ import annotations

import functools
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

#: Decimal places the Gaussian-stamp cache rounds ``sigma`` to. Per-object
#: sigmas come from a small integer radius ((2r+1)/6), so a handful of distinct
#: values recur over and over; rounding makes them hashable cache keys without
#: moving the stamp by anything a float32 target can hold.
_SIGMA_CACHE_DP = 6


@functools.lru_cache(maxsize=256)
def _gaussian_stamp(radius: int, sigma: float) -> np.ndarray:
    """A (2r+1, 2r+1) float32 Gaussian stamp with peak exactly 1.0 at its centre.

    Cached because per-object sigmas (box targets) repeat heavily within an
    epoch. The returned array is shared — treat it as read-only.
    """
    ax = np.arange(2 * radius + 1, dtype=np.float64) - radius
    stamp = np.exp(-(ax[None, :] ** 2 + ax[:, None] ** 2) / (2.0 * sigma * sigma))
    stamp = stamp.astype(np.float32)
    stamp.flags.writeable = False
    return stamp


def _stamp_radius(sigma: float) -> int:
    """Stamp half-width for a sigma: 3 sigma, never smaller than one cell."""
    return max(1, int(round(3.0 * float(sigma))))


def render_targets(
    points: np.ndarray,
    out_hw: Tuple[int, int],
    sigma: Union[float, np.ndarray],
) -> np.ndarray:
    """Render a peak-normalised Gaussian heatmap target.

    Args:
        points: (N, 2) array of (x, y) locations in output-grid coordinates.
        out_hw: (height, width) of the output grid.
        sigma: Gaussian spread in output-grid cells. Either a scalar shared by
            every point (the point-counting case) or an ``(N,)`` array of
            per-point spreads (the box case, where CenterNet derives sigma from
            each box's size — see :func:`cropcounter.boxmap.render_box_targets`).

    Returns:
        (H, W) float32 heatmap in [0, 1]. Each point contributes a Gaussian
        with peak exactly 1.0 at its (floored) cell; overlaps are composed
        with element-wise max so adjacent points keep distinct maxima. The
        exact-1.0 peak is what the focal loss uses to identify positives.
    """
    out_h, out_w = out_hw
    heat = np.zeros((out_h, out_w), dtype=np.float32)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return heat

    sigmas = np.asarray(sigma, dtype=np.float64).reshape(-1)
    if sigmas.size == 1:
        sigmas = np.repeat(sigmas, len(points))
    elif sigmas.size != len(points):
        raise ValueError(
            f"sigma must be a scalar or one value per point, got {sigmas.size} "
            f"for {len(points)} point(s)"
        )

    for (x, y), point_sigma in zip(points, sigmas):
        radius = _stamp_radius(point_sigma)
        stamp = _gaussian_stamp(radius, round(float(point_sigma), _SIGMA_CACHE_DP))
        cx = min(max(int(x), 0), out_w - 1)
        cy = min(max(int(y), 0), out_h - 1)
        x0, x1 = max(0, cx - radius), min(out_w, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(out_h, cy + radius + 1)
        sx0, sy0 = x0 - (cx - radius), y0 - (cy - radius)
        window = heat[y0:y1, x0:x1]
        np.maximum(
            window,
            stamp[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)],
            out=window,
        )
    return heat


def point_nms(points: np.ndarray, scores: np.ndarray, radius: float) -> np.ndarray:
    """Greedy point non-maximum suppression.

    Keeps points in descending score order, dropping any point strictly
    within ``radius`` of an already-kept point.

    Args:
        points: (N, 2) candidate locations.
        scores: (N,) candidate scores.
        radius: suppression radius, same units as ``points``.

    Returns:
        Indices of kept points, ordered by descending score.
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(points) == 0:
        return np.empty(0, dtype=np.int64)

    order = np.argsort(-scores)
    kept: list[int] = []
    kept_pts = np.empty((0, 2), dtype=np.float32)
    r2 = float(radius) * float(radius)
    for i in order:
        p = points[i]
        if len(kept) and (((kept_pts - p) ** 2).sum(axis=1) <= r2).any():
            continue
        kept.append(int(i))
        kept_pts = np.vstack([kept_pts, p[None]])
    return np.asarray(kept, dtype=np.int64)


def decode_peaks(
    heatmap: Union[torch.Tensor, np.ndarray],
    k: int = 3,
    tau: float = 0.3,
    nms_radius: float = 1.5,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode a probability heatmap into point detections.

    Args:
        heatmap: (H, W) or (1, 1, H, W) probabilities in [0, 1]. Apply
            sigmoid to model logits before calling.
        k: max-pool kernel; sets the local-max window. Minimum resolvable
            separation is k // 2 + 1 cells.
        tau: score threshold applied after the local-max test.
        nms_radius: greedy point-NMS radius in grid cells. The default 1.5
            kills plateau ties in adjacent/diagonal cells (d <= sqrt(2))
            while keeping genuine peaks 2 cells apart — consistent with the
            k=3 separation floor.
        stride: output stride; detections are scaled back to input pixels.

    Returns:
        (points, scores): points (N, 2) float32 (x, y) at cell centres in
        input-pixel coordinates, scores (N,) descending.
    """
    if isinstance(heatmap, np.ndarray):
        heat = torch.from_numpy(np.ascontiguousarray(heatmap))
    else:
        heat = heatmap
    heat = heat.detach().float()
    while heat.dim() < 4:
        heat = heat.unsqueeze(0)
    if heat.shape[0] != 1 or heat.shape[1] != 1:
        raise ValueError(f"decode_peaks expects a single-image heatmap, got {tuple(heat.shape)}")

    pooled = F.max_pool2d(heat, kernel_size=k, stride=1, padding=k // 2)
    mask = (heat == pooled) & (heat > tau)
    ys, xs = mask[0, 0].nonzero(as_tuple=True)
    if len(ys) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.float32)

    scores = heat[0, 0, ys, xs].cpu().numpy()
    pts = torch.stack([xs, ys], dim=1).float().cpu().numpy()
    keep = point_nms(pts, scores, nms_radius)
    pts, scores = pts[keep], scores[keep]
    return (pts + 0.5) * float(stride), scores
