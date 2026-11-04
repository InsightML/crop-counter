"""Heatmap target rendering and peak decoding for point-based counting.

Targets are CenterNet-style peak-normalised Gaussians (peak value 1.0)
composed with element-wise max, so nearby points keep distinct maxima.
Decoding is a max-pool local-max test, a score threshold ``tau``, and a
greedy point-NMS that removes plateau ties and duplicate peaks.

All coordinates here are (x, y). "Grid coordinates" means the output-stride
grid of the model head; ``decode_peaks`` converts back to input pixels.

Everything here is single-channel: one class, one heatmap. Multiclass models
stack one such heatmap per class along the channel axis —
``render_class_targets`` renders that stack, ``per_class_values`` resolves a
scalar-or-dict decode parameter to one value per class, and
``cropcounter.inference.decode_classes`` runs ``decode_peaks`` per channel.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def render_targets(
    points: np.ndarray,
    out_hw: Tuple[int, int],
    sigma: float,
) -> np.ndarray:
    """Render a peak-normalised Gaussian heatmap target.

    Args:
        points: (N, 2) array of (x, y) locations in output-grid coordinates.
        out_hw: (height, width) of the output grid.
        sigma: Gaussian spread in output-grid cells.

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

    radius = max(1, int(round(3.0 * sigma)))
    size = 2 * radius + 1
    ax = np.arange(size, dtype=np.float64) - radius
    stamp = np.exp(-(ax[None, :] ** 2 + ax[:, None] ** 2) / (2.0 * sigma * sigma))
    stamp = stamp.astype(np.float32)

    for x, y in points:
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


def render_class_targets(
    points: np.ndarray,
    class_ids: np.ndarray,
    n_classes: int,
    out_hw: Tuple[int, int],
    sigma: float,
) -> np.ndarray:
    """Render one peak-normalised Gaussian heatmap per class.

    Args:
        points: (N, 2) array of (x, y) locations in output-grid coordinates.
        class_ids: (N,) integer channel index of each point, in
            ``[0, n_classes)``.
        n_classes: number of output channels C.
        out_hw: (height, width) of the output grid.
        sigma: Gaussian spread in output-grid cells.

    Returns:
        (C, H, W) float32. Channel ``c`` is :func:`render_targets` over the
        points whose class id is ``c`` (all zero when a class has no points
        in this tile). With ``n_classes=1`` this is ``render_targets(...)``
        with a leading channel axis, so the single-class target is unchanged.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    class_ids = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    if len(class_ids) != len(points):
        raise ValueError(
            f"class_ids has {len(class_ids)} entries for {len(points)} points"
        )
    if n_classes < 1:
        raise ValueError(f"n_classes must be >= 1, got {n_classes}")
    if len(class_ids) and (class_ids.min() < 0 or class_ids.max() >= n_classes):
        raise ValueError(
            f"class ids must lie in [0, {n_classes}), got "
            f"[{class_ids.min()}, {class_ids.max()}]"
        )
    return np.stack(
        [render_targets(points[class_ids == c], out_hw, sigma) for c in range(n_classes)],
        axis=0,
    )


def per_class_values(value: Any, class_names: Sequence[str]) -> Tuple[Any, ...]:
    """Resolve a scalar-or-dict decode parameter to one value per class.

    ``tau`` / ``k`` / ``nms_radius`` may be a single number, broadcast to every
    class, or a ``{class_name: number}`` mapping whose keys must be exactly
    ``class_names``.

    Returns:
        A tuple in ``class_names`` order.
    """
    class_names = tuple(class_names)
    if isinstance(value, Mapping):
        missing = [name for name in class_names if name not in value]
        extra = [key for key in value if key not in class_names]
        if missing or extra:
            raise ValueError(
                f"per-class value keys {sorted(value)} do not match classes "
                f"{list(class_names)} (missing {missing}, unexpected {extra})"
            )
        return tuple(value[name] for name in class_names)
    return tuple(value for _ in class_names)


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
