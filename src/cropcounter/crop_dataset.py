"""Dataset pipeline for the DINOv3 pyramid-decoder crop emergence counter.

Covers: point-annotation parsing behind a small format registry (CVAT for
images 1.1, COCO keypoints, and an optional Datumaro adapter), filesystem
train/val splits, and a torch Dataset producing native-resolution training
tiles with Gaussian heatmap targets or whole padded validation images.

Splitting is a **filesystem** concern, as in most detection libraries::

    dataset/
    ├─ train/
    │  ├─ annotations.xml
    │  └─ images/
    └─ val/
       ├─ annotations.xml
       └─ images/

``load_splits(root)`` reads that layout; ``load_records(path)`` reads a single
annotation file (or a single flat ``annotations.xml`` + ``images/`` folder).
How the split was drawn — random, by field, by farm, by date — is up to
whoever exports the data, and can encode grouping the parser cannot see.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from .dinov3_pyramid import IMAGENET_MEAN, IMAGENET_STD
from .heatmap import render_targets

# Point labels kept by default. These are the labels of the shipped wheat
# example: volunteer wheat is counted together with drilled wheat (matching
# previous model iterations). Pass ``labels=None`` to keep every point label,
# or your own tuple for another crop.
COUNTED_LABELS = ("Wheat", "Volunteer")

#: Sub-directory holding the image files inside a dataset (or split) folder.
IMAGES_DIRNAME = "images"

#: Split folder names ``load_splits`` expects under the dataset root.
SPLIT_NAMES = ("train", "val")


@dataclass
class Point:
    """One point annotation with its QC metadata."""

    x: float
    y: float
    label: str
    confidence: Optional[int] = None
    label_status: Optional[str] = None
    occluded: bool = False


@dataclass
class ImageRecord:
    """One annotated image: dimensions and its point annotations."""

    name: str
    width: int
    height: int
    points: List[Point] = field(default_factory=list)


def _keep(label: str, labels: Optional[Sequence[str]]) -> bool:
    """Whether a point label survives the label filter (``None`` keeps all)."""
    return labels is None or label in labels


def _confidence(value) -> Optional[int]:
    """Coerce a raw Confidence attribute to int, or None when absent/blank."""
    if value is None or value == "":
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Format loaders
# --------------------------------------------------------------------------- #


def parse_cvat_1_1(
    xml_path: Path,
    labels: Optional[Sequence[str]] = COUNTED_LABELS,
) -> List[ImageRecord]:
    """Parse a "CVAT for images 1.1" annotation file into ImageRecords.

    Only ``<points>`` elements whose label passes ``labels`` are kept
    (polylines, polygons and tags are ignored). Per-point ``Confidence``
    and ``Label Status`` attributes are preserved for later filtering and
    stratified evaluation.
    """
    root = ET.parse(str(xml_path)).getroot()
    records: List[ImageRecord] = []
    for image_el in root.iter("image"):
        points: List[Point] = []
        for pts_el in image_el.findall("points"):
            label = pts_el.get("label", "")
            if not _keep(label, labels):
                continue
            attrs: Dict[str, Optional[str]] = {
                a.get("name", ""): a.text for a in pts_el.findall("attribute")
            }
            confidence = _confidence(attrs.get("Confidence"))
            status = attrs.get("Label Status")
            occluded = pts_el.get("occluded", "0") == "1"
            for pair in pts_el.get("points", "").split(";"):
                if not pair:
                    continue
                x_str, y_str = pair.split(",")
                points.append(Point(
                    x=float(x_str), y=float(y_str), label=label,
                    confidence=confidence, label_status=status, occluded=occluded,
                ))
        records.append(ImageRecord(
            name=image_el.get("name", ""),
            width=int(image_el.get("width", 0)),
            height=int(image_el.get("height", 0)),
            points=points,
        ))
    return records


def parse_coco_keypoints(
    json_path: Path,
    labels: Optional[Sequence[str]] = COUNTED_LABELS,
) -> List[ImageRecord]:
    """Parse a COCO **keypoints** JSON into the same ImageRecords as CVAT.

    Plain-json parsing, no pycocotools. Every visible keypoint triple
    ``(x, y, v)`` with ``v > 0`` becomes one :class:`Point`; the label is the
    annotation's category name, ``occluded`` is ``v == 1`` (COCO's "labelled
    but not visible"), and ``Confidence`` / ``Label Status`` are read from the
    annotation's ``attributes`` dict when the exporter wrote one (CVAT does).
    Images with no surviving annotation still yield an empty record, so image
    counts match the CVAT loader on the same data.
    """
    with Path(json_path).open(encoding="utf-8") as fh:
        payload = json.load(fh)

    category_names = {
        int(cat["id"]): str(cat.get("name", "")) for cat in payload.get("categories", [])
    }
    records: Dict[int, ImageRecord] = {}
    order: List[int] = []
    for image in payload.get("images", []):
        image_id = int(image["id"])
        records[image_id] = ImageRecord(
            name=str(image.get("file_name", "")),
            width=int(image.get("width", 0)),
            height=int(image.get("height", 0)),
            points=[],
        )
        order.append(image_id)

    for ann in payload.get("annotations", []):
        image_id = int(ann.get("image_id", -1))
        record = records.get(image_id)
        if record is None:
            continue
        label = category_names.get(int(ann.get("category_id", -1)), "")
        if not _keep(label, labels):
            continue
        attributes = ann.get("attributes") or {}
        confidence = _confidence(attributes.get("Confidence"))
        if confidence is None and ann.get("score") is not None:
            # CVAT writes Confidence as 0-100; a COCO score is 0-1. Same scale out.
            score = float(ann["score"])
            confidence = _confidence(score * 100 if score <= 1.0 else score)
        status = attributes.get("Label Status")
        keypoints = ann.get("keypoints") or []
        for i in range(0, len(keypoints) - 2, 3):
            x, y, visibility = keypoints[i], keypoints[i + 1], keypoints[i + 2]
            if not visibility:  # v == 0: not labelled
                continue
            record.points.append(Point(
                x=float(x), y=float(y), label=label,
                confidence=confidence, label_status=status,
                occluded=bool(attributes.get("occluded", int(visibility) == 1)),
            ))

    return [records[i] for i in order]


def parse_datumaro(
    path: Path,
    labels: Optional[Sequence[str]] = COUNTED_LABELS,
    dataset_format: Optional[str] = None,
) -> List[ImageRecord]:
    """Parse anything Datumaro can read into the same ImageRecords.

    Optional: install with ``pip install 'cropcounter[formats]'``. Point
    annotations are flattened the same way as the built-in loaders, so a
    Datumaro-converted export produces identical records. ``dataset_format``
    is passed straight to Datumaro; ``None`` lets it auto-detect.
    """
    try:
        import datumaro as dm
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "fmt='datumaro' needs the optional Datumaro dependency: "
            "pip install 'cropcounter[formats]'"
        ) from exc

    dataset = (
        dm.Dataset.import_from(str(path), dataset_format)
        if dataset_format else dm.Dataset.import_from(str(path))
    )
    label_categories = dataset.categories().get(dm.AnnotationType.label)

    def label_name(index: Optional[int]) -> str:
        if index is None or label_categories is None:
            return ""
        try:
            return str(label_categories[index].name)
        except (IndexError, TypeError):  # pragma: no cover - defensive
            return ""

    records: List[ImageRecord] = []
    for item in dataset:
        media = getattr(item, "media", None)
        size = getattr(media, "size", None) or (0, 0)
        height, width = int(size[0]), int(size[1])
        name = getattr(media, "path", None) or item.id
        points: List[Point] = []
        for ann in item.annotations:
            if getattr(ann, "type", None) != dm.AnnotationType.points:
                continue
            label = label_name(getattr(ann, "label", None))
            if not _keep(label, labels):
                continue
            attributes = getattr(ann, "attributes", None) or {}
            confidence = _confidence(attributes.get("Confidence"))
            status = attributes.get("Label Status")
            coords = list(getattr(ann, "points", []) or [])
            visibility = list(getattr(ann, "visibility", []) or [])
            for i in range(0, len(coords) - 1, 2):
                if visibility and i // 2 < len(visibility) and not visibility[i // 2]:
                    continue
                points.append(Point(
                    x=float(coords[i]), y=float(coords[i + 1]), label=label,
                    confidence=confidence, label_status=status,
                    occluded=bool(attributes.get("occluded", False)),
                ))
        records.append(ImageRecord(
            name=Path(str(name)).name, width=width, height=height, points=points
        ))
    return records


#: Annotation format -> loader. Every loader emits identical ImageRecord/Point.
LOADERS: Dict[str, Callable[..., List[ImageRecord]]] = {
    "cvat": parse_cvat_1_1,
    "coco": parse_coco_keypoints,
    "datumaro": parse_datumaro,
}

#: Filenames looked for when ``load_records`` is handed a directory.
ANNOTATION_FILENAMES: Dict[str, Tuple[str, ...]] = {
    "cvat": ("annotations.xml",),
    "coco": (
        "annotations.json",
        "person_keypoints_default.json",
        "instances_default.json",
    ),
}


def resolve_annotations(path: Path, fmt: str = "cvat") -> Path:
    """Return the annotation file for ``path``, which may be a file or a folder.

    A folder is searched for the format's conventional filename, both directly
    and inside an ``annotations/`` sub-folder (CVAT's own export layout).
    Datumaro takes a project/dataset directory, so the path passes through.
    """
    path = Path(path)
    if fmt == "datumaro":
        return path
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"annotation path does not exist: {path}")

    candidates = [
        directory / name
        for directory in (path, path / "annotations")
        for name in ANNOTATION_FILENAMES.get(fmt, ())
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    looked = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"no {fmt} annotation file found under {path} (looked for: {looked})")


def load_records(
    path: Path,
    fmt: str = "cvat",
    labels: Optional[Sequence[str]] = COUNTED_LABELS,
) -> List[ImageRecord]:
    """Load point annotations from a file or dataset folder.

    Args:
        path: annotation file, or a folder holding one (e.g. a flat
            ``dataset/`` with ``annotations.xml`` + ``images/``, or one split
            of a ``train/`` + ``val/`` layout).
        fmt: ``"cvat"`` (CVAT for images 1.1), ``"coco"`` (COCO keypoints), or
            ``"datumaro"`` (optional extra).
        labels: point labels to keep; ``None`` keeps every label.

    Returns:
        One :class:`ImageRecord` per image in the annotation file, in file
        order, whatever the format.
    """
    if fmt not in LOADERS:
        raise ValueError(f"unknown annotation format {fmt!r}; expected one of {sorted(LOADERS)}")
    return LOADERS[fmt](resolve_annotations(path, fmt), labels=labels)


def load_splits(
    root: Path,
    fmt: str = "cvat",
    labels: Optional[Sequence[str]] = COUNTED_LABELS,
    splits: Sequence[str] = SPLIT_NAMES,
) -> Tuple[List[ImageRecord], List[ImageRecord]]:
    """Load the ``train`` and ``val`` records of a split-on-disk dataset.

    Expects ``root/train/`` and ``root/val/``, each holding the annotation file
    and an ``images/`` folder. ``splits`` renames those two folders; there are
    always exactly two.

    Returns:
        ``(train_records, val_records)``.
    """
    root = Path(root)
    if len(splits) != 2:
        raise ValueError(f"load_splits returns exactly two splits, got {tuple(splits)}")
    missing = [name for name in splits if not (root / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"missing split folder(s) {missing} under {root.absolute()}; expected "
            f"{'/ '.join(splits)} sub-folders, each with an annotation file and {IMAGES_DIRNAME}/"
        )
    train_recs, val_recs = (
        load_records(root / name, fmt=fmt, labels=labels) for name in splits
    )
    return train_recs, val_recs


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


class CropTileDataset(Dataset):
    """Training tiles or whole validation images with point targets.

    Train mode: each __getitem__ draws a fresh random ``tile`` x ``tile``
    crop at native resolution (``tiles_per_image`` draws per image per
    epoch), applies geometric + photometric augmentation with
    point-consistent keypoints, and renders the Gaussian heatmap target at
    ``output_stride``. Images smaller than the tile are constant-padded —
    never reflected, which would create mirror-image plants with no labels.

    Val mode: the whole image, padded bottom-right to a multiple of 32,
    with the raw ground-truth points for decode-based metrics. Use
    ``collate_val`` and batch_size=1.

    Pass ``transform`` to replace the default training augmentation pipeline;
    it must be an albumentations Compose with ``KeypointParams(format="xy")``
    and end in ``Normalize`` + ``ToTensorV2``.
    """

    def __init__(
        self,
        records: Sequence[ImageRecord],
        images_dir: Path,
        train: bool,
        tile: int = 768,
        output_stride: int = 4,
        sigma: float = 2.0,
        tiles_per_image: int = 4,
        scale_jitter: float = 0.25,
        exclude_label_statuses: Sequence[str] = (),
        transform: Optional[A.Compose] = None,
    ) -> None:
        self.records = list(records)
        self.images_dir = Path(images_dir)
        self.train = train
        self.tile = tile
        self.output_stride = output_stride
        self.sigma = sigma
        self.tiles_per_image = tiles_per_image

        excluded = set(exclude_label_statuses)
        self.points_px: List[np.ndarray] = [
            np.array(
                [[p.x, p.y] for p in rec.points if p.label_status not in excluded],
                dtype=np.float32,
            ).reshape(-1, 2)
            for rec in self.records
        ]

        if train:
            self.transform = transform or self.default_transform(tile, scale_jitter)
        else:
            self.transform = None

    @staticmethod
    def default_transform(tile: int = 768, scale_jitter: float = 0.25) -> A.Compose:
        """The default training augmentation pipeline."""
        return A.Compose(
            [
                A.RandomScale(scale_limit=scale_jitter, p=0.8),
                A.RandomCrop(
                    height=tile, width=tile, pad_if_needed=True,
                    border_mode=cv2.BORDER_CONSTANT, fill=0,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.75),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=10, p=0.3
                ),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=True),
        )

    def __len__(self) -> int:
        return len(self.records) * (self.tiles_per_image if self.train else 1)

    def _load_image(self, record: ImageRecord) -> np.ndarray:
        path = self.images_dir / record.name
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"failed to read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index: int):
        if self.train:
            return self._get_train_tile(index % len(self.records))
        return self._get_val_image(index)

    def _get_train_tile(self, rec_idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        record = self.records[rec_idx]
        image = self._load_image(record)
        out = self.transform(image=image, keypoints=self.points_px[rec_idx])
        keypoints = np.asarray(out["keypoints"], dtype=np.float32).reshape(-1, 2)

        out_size = self.tile // self.output_stride
        target = render_targets(
            keypoints / self.output_stride, (out_size, out_size), self.sigma
        )
        return out["image"], torch.from_numpy(target).unsqueeze(0), len(keypoints)

    def _get_val_image(self, index: int) -> Dict:
        record = self.records[index]
        image = self._load_image(record).astype(np.float32) / 255.0
        image = (image - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)

        h, w = image.shape[:2]
        pad_h, pad_w = (-h) % 32, (-w) % 32
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

        # Heatmap target over the padded output grid, for a decode-free
        # (tau-independent) validation loss. Padding is bottom-right so the
        # point coordinates need no shift; the padded strip is all-background.
        out_h = image.shape[0] // self.output_stride
        out_w = image.shape[1] // self.output_stride
        target = render_targets(
            self.points_px[index] / self.output_stride, (out_h, out_w), self.sigma
        )

        return {
            "image": torch.from_numpy(image.transpose(2, 0, 1).copy()),
            "target": torch.from_numpy(target).unsqueeze(0),
            "points": self.points_px[index],
            "name": record.name,
        }


def collate_val(batch: List[Dict]) -> Dict:
    """Batch-size-1 collate keeping variable-length point arrays intact."""
    if len(batch) != 1:
        raise ValueError("validation loader must use batch_size=1")
    item = batch[0]
    return {
        "image": item["image"].unsqueeze(0),
        "target": item["target"].unsqueeze(0),
        "points": item["points"],
        "name": item["name"],
    }
