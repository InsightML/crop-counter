"""Box-target rendering and box decoding for the CenterNet-style detection head.

The point head predicts *where*; the box head adds *how big* (``wh``) and
*where inside the cell* (``off``). Targets are written only at each box's
integer centre cell, exactly as CenterNet does, so the geometry branch is
supervised on a handful of cells per image and the heatmap branch carries the
rest of the signal.

Coordinates. Boxes enter and leave this module as **COCO xywh input pixels**
(the Albumentations ``format="coco"`` convention), so nothing converts on the
way in from an annotation file. Internally everything is grid units: the
centre of a box is ``c = (x + w/2) / stride`` and the cell holding it is
``floor(c)``; ``off = c - floor(c)`` recovers the sub-cell remainder that
``decode_boxes`` adds back. Outputs of :func:`decode_boxes` are xyxy pixels,
the format ``torchvision.ops.nms`` and IoU matching want.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

#: Size parameterisations accepted by the target renderer and the decoder.
SIZE_PARAMETERISATIONS = ("log", "linear")

ArrayLike = Union[torch.Tensor, np.ndarray]


def gaussian_radius(det_size: Sequence[float], min_overlap: float = 0.7) -> float:
    """CenterNet's Gaussian radius for a box of ``det_size = (height, width)``.

    The radius is the largest centre displacement that still leaves a box of
    this size overlapping the ground truth by ``min_overlap`` IoU, under three
    displacement regimes (both corners out, both in, one each); the smallest of
    the three roots wins.

    This is transcribed **verbatim from CenterNet** (``src/lib/utils/image.py``),
    including its well-known departure from CornerNet's derivation: cases 2 and
    3 divide by 2 rather than by ``2 * a``, and all three take the ``+ sqrt``
    root. That makes the returned radius differ from the algebraically correct
    one by a sub-pixel-to-few-pixel term at typical sizes. It is kept exactly as
    CenterNet has it on purpose — every CenterNet-family number this repo will
    be compared against was produced with these constants, and comparability
    with the family beats a more correct radius that silently reshapes the
    target distribution.

    Args:
        det_size: ``(height, width)`` of the box, in output-grid cells.
        min_overlap: the IoU the displaced box must still achieve.

    Returns:
        The radius in output-grid cells (a float; CenterNet floors it to an int
        before use — see :func:`render_box_targets`).
    """
    height, width = float(det_size[0]), float(det_size[1])

    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0.0))
    r1 = (b1 + sq1) / 2

    a2 = 4.0
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(max(b2 ** 2 - 4 * a2 * c2, 0.0))
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(max(b3 ** 2 - 4 * a3 * c3, 0.0))
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def _check_parameterisation(name: str) -> str:
    if name not in SIZE_PARAMETERISATIONS:
        raise ValueError(
            f"size_parameterisation must be one of {SIZE_PARAMETERISATIONS}, got {name!r}"
        )
    return name


def render_box_targets(
    boxes_xywh_px: np.ndarray,
    out_hw: Tuple[int, int],
    stride: int,
    size_parameterisation: str = "log",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Render CenterNet heatmap + size + offset targets for one image.

    Args:
        boxes_xywh_px: (N, 4) boxes as COCO ``[x, y, w, h]`` in input pixels.
        out_hw: ``(height, width)`` of the output grid.
        stride: output stride mapping input pixels to grid cells.
        size_parameterisation: ``"log"`` stores ``log(size / stride)`` (scale
            invariant, and what makes a 20 px and a 200 px fish contribute
            comparable L1 gradients); ``"linear"`` stores ``size / stride``,
            CenterNet's own choice — which is why CenterNet needs ``wh_weight``
            down at 0.1.

    Returns:
        ``(hm, wh, off, mask, n_collisions)``. ``hm`` is (H, W) with a
        peak-normalised Gaussian per box whose sigma comes from the box's own
        size; ``wh`` and ``off`` are (2, H, W), written **only** at each box's
        integer centre cell; ``mask`` is (H, W) with 1.0 at those cells.
        ``n_collisions`` is the number of boxes whose centre cell was already
        taken — the empirical cost of the output stride, and the number to look
        at before arguing for stride 2. All arrays are float32.
    """
    from .heatmap import render_targets  # local: heatmap imports nothing from here

    _check_parameterisation(size_parameterisation)
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    wh = np.zeros((2, out_h, out_w), dtype=np.float32)
    off = np.zeros((2, out_h, out_w), dtype=np.float32)
    mask = np.zeros((out_h, out_w), dtype=np.float32)

    boxes = np.asarray(boxes_xywh_px, dtype=np.float64).reshape(-1, 4)
    if len(boxes) == 0:
        return np.zeros((out_h, out_w), dtype=np.float32), wh, off, mask, 0

    stride_f = float(stride)
    w_cells = boxes[:, 2] / stride_f
    h_cells = boxes[:, 3] / stride_f
    centres = np.stack(
        [(boxes[:, 0] + boxes[:, 2] / 2.0) / stride_f,
         (boxes[:, 1] + boxes[:, 3] / 2.0) / stride_f],
        axis=1,
    )
    cells = np.stack(
        [np.clip(np.floor(centres[:, 0]), 0, out_w - 1),
         np.clip(np.floor(centres[:, 1]), 0, out_h - 1)],
        axis=1,
    ).astype(np.int64)

    # CenterNet: radius floored to an int (and never negative), sigma = (2r+1)/6
    # so the stamp's 3-sigma support is exactly the radius window.
    sigmas = np.array(
        [(2 * max(0, int(gaussian_radius((h, w)))) + 1) / 6.0
         for h, w in zip(h_cells, w_cells)],
        dtype=np.float64,
    )
    hm = render_targets(centres, (out_h, out_w), sigmas)

    n_collisions = 0
    for i in range(len(boxes)):
        cx, cy = int(cells[i, 0]), int(cells[i, 1])
        if mask[cy, cx] == 1.0:
            n_collisions += 1
        if size_parameterisation == "log":
            wh[0, cy, cx] = math.log(max(w_cells[i], 1e-6))
            wh[1, cy, cx] = math.log(max(h_cells[i], 1e-6))
        else:
            wh[0, cy, cx] = w_cells[i]
            wh[1, cy, cx] = h_cells[i]
        off[0, cy, cx] = centres[i, 0] - cx
        off[1, cy, cx] = centres[i, 1] - cy
        mask[cy, cx] = 1.0

    return hm, wh, off, mask, n_collisions


def _as_tensor(array: ArrayLike) -> torch.Tensor:
    if isinstance(array, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(array))
    return array


def decode_boxes(
    hm: ArrayLike,
    wh: ArrayLike,
    off: ArrayLike,
    stride: int,
    k: int = 3,
    tau: float = 0.01,
    top_k: int = 100,
    box_nms_iou: Optional[float] = None,
    size_parameterisation: str = "log",
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode a heatmap + size + offset triple into boxes.

    The peak test is identical to :func:`cropcounter.heatmap.decode_peaks`: a
    ``k``-wide max-pool local-max, then a score threshold. Suppression is where
    the two part company — boxes are suppressed by IoU, never by centre
    distance. ``point_nms`` would merge two fish a single cell apart however
    different their extents, and its greedy ``np.vstack`` loop is O(N^2) in
    both time and allocation, which a 100-detection budget per image cannot
    afford.

    Args:
        hm: (H, W) or (1, 1, H, W) **probabilities** (sigmoid already applied).
        wh: (2, H, W) or (1, 2, H, W) size map.
        off: (2, H, W) or (1, 2, H, W) sub-cell offset map.
        stride: output stride; boxes come back in input pixels.
        k: local-max pooling kernel.
        tau: score threshold. Default 0.01, not the point head's 0.3 — average
            precision wants a long, low-confidence tail to trace out the PR
            curve; use the F1 threshold only for P/R/F1.
        top_k: keep at most this many detections, by score, **before** any
            suppression (so NMS cost is bounded whatever the heatmap does).
        box_nms_iou: IoU threshold for ``torchvision.ops.nms``; ``None``
            disables it (the local-max test alone already deduplicates cleanly
            when boxes do not overlap much).
        size_parameterisation: must match the one the targets were rendered with.

    Returns:
        ``(boxes_xyxy_px (N, 4) float32, scores (N,) float32)``, scores descending.
    """
    _check_parameterisation(size_parameterisation)

    heat = _as_tensor(hm).detach().float()
    while heat.dim() < 4:
        heat = heat.unsqueeze(0)
    if heat.shape[0] != 1 or heat.shape[1] != 1:
        raise ValueError(f"decode_boxes expects a single-image heatmap, got {tuple(heat.shape)}")

    size_map = _as_tensor(wh).detach().float()
    off_map = _as_tensor(off).detach().float()
    for name, tensor in (("wh", size_map), ("off", off_map)):
        if tensor.dim() == 4:
            if tensor.shape[0] != 1:
                raise ValueError(f"decode_boxes expects a single-image {name} map")
        elif tensor.dim() != 3:
            raise ValueError(f"{name} must be (2, H, W) or (1, 2, H, W), got {tuple(tensor.shape)}")
    size_map = size_map.reshape(2, *heat.shape[-2:])
    off_map = off_map.reshape(2, *heat.shape[-2:])

    pooled = F.max_pool2d(heat, kernel_size=k, stride=1, padding=k // 2)
    peaks = (heat == pooled) & (heat > tau)
    ys, xs = peaks[0, 0].nonzero(as_tuple=True)
    empty = (np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32))
    if len(ys) == 0:
        return empty

    scores = heat[0, 0, ys, xs]
    order = torch.argsort(scores, descending=True)
    if top_k is not None and len(order) > top_k:
        order = order[:top_k]
    ys, xs, scores = ys[order], xs[order], scores[order]

    cx = (xs.float() + off_map[0, ys, xs]) * float(stride)
    cy = (ys.float() + off_map[1, ys, xs]) * float(stride)
    if size_parameterisation == "log":
        box_w = torch.exp(size_map[0, ys, xs]) * float(stride)
        box_h = torch.exp(size_map[1, ys, xs]) * float(stride)
    else:
        box_w = size_map[0, ys, xs] * float(stride)
        box_h = size_map[1, ys, xs] * float(stride)

    boxes = torch.stack(
        [cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2], dim=1
    )

    if box_nms_iou is not None and len(boxes):
        from torchvision.ops import nms  # already a core dependency

        keep = nms(boxes, scores, float(box_nms_iou))
        boxes, scores = boxes[keep], scores[keep]

    return (boxes.cpu().numpy().astype(np.float32),
            scores.cpu().numpy().astype(np.float32))


def boxes_xyxy_to_xywh(boxes_xyxy: np.ndarray) -> np.ndarray:
    """xyxy -> COCO xywh, the format detections are reported in."""
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    return np.stack(
        [boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]],
        axis=1,
    )


def boxes_xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    """COCO xywh -> xyxy, the format IoU matching and NMS want."""
    boxes = np.asarray(boxes_xywh, dtype=np.float32).reshape(-1, 4)
    return np.stack(
        [boxes[:, 0], boxes[:, 1], boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]],
        axis=1,
    )


def clip_boxes_xyxy(boxes_xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clamp boxes to the ``[0, width] x [0, height]`` frame (drops nothing)."""
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4).copy()
    boxes[:, 0] = boxes[:, 0].clip(0, width)
    boxes[:, 2] = boxes[:, 2].clip(0, width)
    boxes[:, 1] = boxes[:, 1].clip(0, height)
    boxes[:, 3] = boxes[:, 3].clip(0, height)
    return boxes
