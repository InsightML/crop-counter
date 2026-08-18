"""Batch inference helpers: folder -> probability map -> points -> exports.

Inference mirrors the validation path in :mod:`cropcounter.crop_dataset`:
whole image, ImageNet normalisation, bottom-right pad to a multiple of 32,
sigmoid, then ``heatmap.decode_peaks``. Source images are never modified.

The pieces are deliberately small and separate so a notebook can run them one
at a time::

    recs  = records_from_folder("images/")
    ds    = CropTileDataset(recs, "images/", train=False, output_stride=cfg.output_stride)
    prob  = predict_prob(model, ds[0]["image"], device)
    pts, scores = decode_in_bounds(prob, recs[0].width, recs[0].height, tau=0.35,
                                   k=cfg.k, nms_radius=cfg.nms_radius,
                                   output_stride=cfg.output_stride)
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .crop_dataset import ImageRecord
from .dinov3_pyramid import IMAGENET_MEAN, IMAGENET_STD
from .heatmap import decode_peaks

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
    """Forward one padded image tensor (C, H, W) -> sigmoid prob map (1, 1, h, w).

    Runs under ``torch.no_grad``; bf16 autocast is used on CUDA only. The map
    comes back on the CPU as float32, ready for :func:`decode_in_bounds`.
    """
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    with torch.no_grad(), torch.autocast(
        device.type, torch.bfloat16, enabled=device.type == "cuda"
    ):
        logits = model(image_tensor.unsqueeze(0).to(device))
    return torch.sigmoid(logits.float()).cpu()


def decode_in_bounds(
    prob: torch.Tensor,
    width: int,
    height: int,
    tau: float = 0.3,
    k: int = 3,
    nms_radius: float = 1.5,
    output_stride: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode peaks and keep only points inside the original (width, height) frame.

    Padding is bottom-right, so decoded pixel coordinates are already in the
    original image frame; this just drops any stray peak in the pad strip.

    Returns:
        ``(points, scores)`` — points (N, 2) float32 (x, y) in source pixels.
    """
    points, scores = decode_peaks(
        prob, k=k, tau=tau, nms_radius=nms_radius, stride=output_stride
    )
    if len(points):
        keep = (points[:, 0] < width) & (points[:, 1] < height)
        points, scores = points[keep], scores[keep]
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
) -> Path:
    """Save a 2-panel figure (points overlay | predicted heatmap) via the OO API.

    Built with ``matplotlib.figure.Figure`` directly (NOT ``plt.*``) so batch
    figures never enter pyplot's global registry — the source of the RAM
    balloon when saving thousands of large-image figures in a loop. The left
    panel is decimated to ``max_side`` so multi-megapixel images stay small in
    memory. No ``close()`` is needed: the figure owns its own Agg canvas.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = image_tensor.permute(1, 2, 0).numpy() * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    img = img[:height, :width].clip(0, 1)
    step = max(1, int(round(max(height, width) / max_side)))
    disp = img[::step, ::step]
    grid_h, grid_w = grid_hw(width, height, output_stride)
    heat = prob[0, 0, :grid_h, :grid_w].numpy()

    fig = Figure(figsize=(15, 7))
    FigureCanvasAgg(fig)
    ax0, ax1 = fig.subplots(1, 2)
    ax0.imshow(disp)
    if len(points):
        ax0.scatter(points[:, 0] / step, points[:, 1] / step, s=45, facecolors="none",
                    edgecolors="red", linewidths=1.0)
    ax0.set_title(f"{out_path.stem[:60]}\npredicted count: {len(points)}", fontsize=9)
    ax1.imshow(heat, cmap="hot", vmin=0, vmax=1)
    ax1.set_title("predicted heatmap")
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
            (N, 2) and ``scores`` (N,); an optional ``label`` key overrides the
            default for that image.
        out_path: XML file to write.
        label: point label written when a record does not carry its own.

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
        for (x, y), s in zip(rec["points"], rec["scores"]):
            xc = min(max(float(x), 0.0), rec["width"] - 1)
            yc = min(max(float(y), 0.0), rec["height"] - 1)
            pts_el = ET.SubElement(img_el, "points", label=str(rec.get("label", label)),
                                   occluded="0", source="auto", points=f"{xc:.2f},{yc:.2f}")
            ET.SubElement(pts_el, "attribute", name="Confidence").text = str(
                int(round(float(s) * 100))
            )
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(str(out_path), encoding="utf-8", xml_declaration=True)
    return out_path
