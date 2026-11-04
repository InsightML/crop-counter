"""Batch inference helpers: folder -> probability map -> points -> exports.

Inference mirrors the validation path in :mod:`cropcounter.crop_dataset`:
whole image, ImageNet normalisation, bottom-right pad to a multiple of 32,
sigmoid, then ``heatmap.decode_peaks`` on every class channel
(:func:`decode_classes`). Source images are never modified.

The pieces are deliberately small and separate so a notebook can run them one
at a time::

    recs  = records_from_folder("images/")
    ds    = CropTileDataset(recs, "images/", train=False, output_stride=cfg.output_stride)
    prob  = predict_prob(model, ds[0]["image"], device)          # (1, C, h, w)
    pts, scores, class_ids = decode_classes(
        prob, cfg.class_names, tau=0.35, k=cfg.k, nms_radius=cfg.nms_radius,
        output_stride=cfg.output_stride, width=recs[0].width, height=recs[0].height,
    )
"""
from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

from .crop_dataset import WILDCARD_CLASS, ImageRecord
from .dinov3_pyramid import IMAGENET_MEAN, IMAGENET_STD
from .heatmap import decode_peaks, per_class_values

#: Image extensions enumerated by :func:`records_from_folder`.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

#: Longest side of the left panel in :func:`save_visualization`, in pixels.
VIZ_MAX_SIDE = 1400


def records_from_folder(
    images_dir: Path,
    max_images: Optional[int] = None,
    extensions: Sequence[str] = IMAGE_EXTENSIONS,
) -> List[ImageRecord]:
    """Enumerate a flat images folder into label-free ImageRecords.

    Dimensions come from the image header only (fast, no decode). Points are
    empty: the validation-mode dataset needs the record just for the filename
    and to render a (here unused) zero target.
    """
    images_dir = Path(images_dir)
    suffixes = {ext.lower() for ext in extensions}
    paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in suffixes and not p.name.startswith(".")
    )
    if max_images is not None:
        paths = paths[:max_images]

    records: List[ImageRecord] = []
    for path in paths:
        with Image.open(path) as im:
            width, height = im.size
        records.append(ImageRecord(name=path.name, width=width, height=height, points=[]))
    return records


def predict_prob(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Forward one padded image tensor (3, H, W) -> sigmoid prob map (1, C, h, w).

    C is the model's class count (1 for a single-class model). Runs under
    ``torch.no_grad``; bf16 autocast is used on CUDA only. The map comes back
    on the CPU as float32, ready for :func:`decode_classes`.
    """
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    with torch.no_grad(), torch.autocast(
        device.type, torch.bfloat16, enabled=device.type == "cuda"
    ):
        logits = model(image_tensor.unsqueeze(0).to(device))
    return torch.sigmoid(logits.float()).cpu()


def decode_classes(
    prob: Union[torch.Tensor, np.ndarray],
    class_names: Sequence[str],
    tau: Union[float, Dict[str, float]] = 0.3,
    k: Union[int, Dict[str, int]] = 3,
    nms_radius: Union[float, Dict[str, float]] = 1.5,
    output_stride: int = 4,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a multi-channel probability map into class-tagged points.

    Runs :func:`cropcounter.heatmap.decode_peaks` independently on every
    channel with that class's ``tau`` / ``k`` / ``nms_radius`` (a scalar
    applies to all classes; a ``{class_name: value}`` dict sets them per
    class — see :func:`cropcounter.heatmap.per_class_values`). There is no
    cross-class suppression: a peak in two channels yields two points.

    Args:
        prob: (1, C, h, w) or (C, h, w) probabilities, C = ``len(class_names)``.
        class_names: channel names in channel order (``cfg.class_names``).
        output_stride: grid-to-pixel scale of the map.
        width, height: when given, drop points outside the original image
            frame. Padding is bottom-right, so decoded coordinates are
            already in the source frame; this only removes stray peaks in
            the pad strip.

    Returns:
        ``(points, scores, class_ids)`` — points (N, 2) float32 (x, y) in
        source pixels, scores (N,) float32, class_ids (N,) int64 channel
        indices into ``class_names``. Points are grouped by class, scores
        descending within each class.
    """
    if isinstance(prob, np.ndarray):
        heat = torch.from_numpy(np.ascontiguousarray(prob))
    else:
        heat = prob
    heat = heat.detach().float()
    if heat.dim() == 3:
        heat = heat.unsqueeze(0)
    if heat.dim() != 4 or heat.shape[0] != 1:
        raise ValueError(
            f"decode_classes expects a single-image (1, C, h, w) map, got {tuple(heat.shape)}"
        )
    class_names = tuple(class_names)
    if heat.shape[1] != len(class_names):
        raise ValueError(
            f"heatmap has {heat.shape[1]} channel(s) but {len(class_names)} class name(s) "
            f"{list(class_names)} were given"
        )
    taus = per_class_values(tau, class_names)
    ks = per_class_values(k, class_names)
    radii = per_class_values(nms_radius, class_names)

    points_per_class: List[np.ndarray] = []
    scores_per_class: List[np.ndarray] = []
    ids_per_class: List[np.ndarray] = []
    for c in range(len(class_names)):
        pts, sc = decode_peaks(
            heat[:, c:c + 1], k=ks[c], tau=taus[c], nms_radius=radii[c], stride=output_stride
        )
        points_per_class.append(pts)
        scores_per_class.append(sc)
        ids_per_class.append(np.full(len(pts), c, dtype=np.int64))
    points = np.concatenate(points_per_class).reshape(-1, 2).astype(np.float32)
    scores = np.concatenate(scores_per_class).astype(np.float32)
    class_ids = np.concatenate(ids_per_class).astype(np.int64)

    if len(points) and (width is not None or height is not None):
        keep = np.ones(len(points), dtype=bool)
        if width is not None:
            keep &= points[:, 0] < width
        if height is not None:
            keep &= points[:, 1] < height
        points, scores, class_ids = points[keep], scores[keep], class_ids[keep]
    return points, scores, class_ids


def decode_in_bounds(
    prob: torch.Tensor,
    width: int,
    height: int,
    tau: float = 0.3,
    k: int = 3,
    nms_radius: float = 1.5,
    output_stride: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode a single-channel map and keep only points inside (width, height).

    .. deprecated:: 0.2.0
        Use :func:`decode_classes`, which handles any number of class
        channels and also returns each point's class id. This wrapper decodes
        a 1-channel map exactly as before and will be removed in a future
        release.

    Returns:
        ``(points, scores)`` — points (N, 2) float32 (x, y) in source pixels.
    """
    warnings.warn(
        "decode_in_bounds is deprecated since cropcounter 0.2.0; use "
        "decode_classes(prob, class_names, ...) which also returns class ids",
        DeprecationWarning,
        stacklevel=2,
    )
    points, scores, _ = decode_classes(
        prob, (WILDCARD_CLASS,), tau=tau, k=k, nms_radius=nms_radius,
        output_stride=output_stride, width=width, height=height,
    )
    return points, scores


def grid_hw(width: int, height: int, output_stride: int = 4) -> Tuple[int, int]:
    """Output-grid (rows, cols) covering an original (width, height), ceil-divided."""
    return -(-height // output_stride), -(-width // output_stride)


def save_visualization(
    image_tensor: torch.Tensor,
    prob: torch.Tensor,
    points: np.ndarray,
    width: int,
    height: int,
    out_path: Path,
    output_stride: int = 4,
    max_side: int = VIZ_MAX_SIDE,
    class_ids: Optional[np.ndarray] = None,
    class_names: Optional[Sequence[str]] = None,
) -> Path:
    """Save a 2-panel figure (points overlay | predicted heatmap) via the OO API.

    Built with ``matplotlib.figure.Figure`` directly (NOT ``plt.*``) so batch
    figures never enter pyplot's global registry — the source of the RAM
    balloon when saving thousands of large-image figures in a loop. The left
    panel is decimated to ``max_side`` so multi-megapixel images stay small in
    memory. No ``close()`` is needed: the figure owns its own Agg canvas.

    Multiclass: pass the ``class_ids`` from :func:`decode_classes` (and
    ``class_names`` for the legend) to colour the overlay per class; the
    heatmap panel shows the per-cell maximum over class channels. Without
    ``class_ids`` the overlay is a single colour, as for a 1-class model.
    """
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = image_tensor.permute(1, 2, 0).numpy() * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    img = img[:height, :width].clip(0, 1)
    step = max(1, int(round(max(height, width) / max_side)))
    disp = img[::step, ::step]
    grid_h, grid_w = grid_hw(width, height, output_stride)
    n_channels = int(prob.shape[1])
    heat = prob[0, :, :grid_h, :grid_w].max(dim=0).values.numpy()

    fig = Figure(figsize=(15, 7))
    FigureCanvasAgg(fig)
    ax0, ax1 = fig.subplots(1, 2)
    ax0.imshow(disp)
    if len(points) and class_ids is None:
        ax0.scatter(points[:, 0] / step, points[:, 1] / step, s=45, facecolors="none",
                    edgecolors="red", linewidths=1.0)
    elif len(points):
        ids = np.asarray(class_ids, dtype=np.int64).reshape(-1)
        names = list(class_names) if class_names is not None else [
            str(c) for c in range(int(ids.max()) + 1)
        ]
        palette = colormaps["tab10"]
        for c, name in enumerate(names):
            mask = ids == c
            if not mask.any():
                continue
            ax0.scatter(points[mask, 0] / step, points[mask, 1] / step, s=45,
                        facecolors="none", edgecolors=[palette(c % 10)], linewidths=1.0,
                        label=f"{name}: {int(mask.sum())}")
        ax0.legend(loc="upper right", fontsize=8)
    ax0.set_title(f"{out_path.stem[:60]}\npredicted count: {len(points)}", fontsize=9)
    ax1.imshow(heat, cmap="hot", vmin=0, vmax=1)
    ax1.set_title("predicted heatmap" + (" (max over classes)" if n_channels > 1 else ""))
    for ax in (ax0, ax1):
        ax.axis("off")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return out_path


def write_cvat_xml(
    image_preds: Iterable[Dict],
    out_path: Path,
    label: str = "Wheat",
) -> Path:
    """Write predictions as "CVAT for images 1.1" point annotations.

    Args:
        image_preds: dicts with ``name``, ``width``, ``height``, ``points``
            (N, 2) and ``scores`` (N,). An optional ``labels`` key gives one
            label per point (e.g. ``[cfg.class_names[c] for c in class_ids]``
            for a multiclass model); an optional ``label`` key overrides the
            default for every point of that image.
        out_path: XML file to write.
        label: point label written when a record carries neither ``labels``
            nor ``label``.

    The result re-imports into CVAT and round-trips through
    :func:`cropcounter.parse_cvat_1_1`: scores are written as an integer
    ``Confidence`` attribute (score * 100).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    for i, rec in enumerate(image_preds):
        img_el = ET.SubElement(root, "image", id=str(i), name=str(rec["name"]),
                               width=str(rec["width"]), height=str(rec["height"]))
        point_labels: Optional[Sequence[Any]] = rec.get("labels")
        if point_labels is not None and len(point_labels) != len(rec["points"]):
            raise ValueError(
                f"{rec['name']}: {len(point_labels)} labels for {len(rec['points'])} points"
            )
        for j, ((x, y), s) in enumerate(zip(rec["points"], rec["scores"])):
            xc = min(max(float(x), 0.0), rec["width"] - 1)
            yc = min(max(float(y), 0.0), rec["height"] - 1)
            point_label = point_labels[j] if point_labels is not None else rec.get("label", label)
            pts_el = ET.SubElement(img_el, "points", label=str(point_label),
                                   occluded="0", source="auto", points=f"{xc:.2f},{yc:.2f}")
            ET.SubElement(pts_el, "attribute", name="Confidence").text = str(
                int(round(float(s) * 100))
            )
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(str(out_path), encoding="utf-8", xml_declaration=True)
    return out_path
