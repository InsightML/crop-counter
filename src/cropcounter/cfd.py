"""Subset the Community Fish Detection Dataset (CFD, LILA) into trainable slices.

CFD harmonises ~17 marine/freshwater source datasets into ONE COCO json:
~1.9M image records, ~935k boxes, a single ``fish`` category. The master file
is ~1.1 GB uncompressed, so every pass here is **streamed** with ijson — the
whole document is never held in memory.

Three subcommands, all deterministic under ``--seed``::

    python -m cropcounter.cfd manifest --metadata cfd.json.zip --out reports/
    python -m cropcounter.cfd subset   --metadata cfd.json.zip --out data/brackish \\
        --sources brackish --train-cap 20000 --val-cap 4000
    python -m cropcounter.cfd fetch    --subset data/brackish --max-side 1024

``manifest`` profiles every source (box sizes, emptiness, train/val balance,
licence, and the stride-4 centre-cell collision rate a point/centre head would
actually see). ``subset`` writes standard COCO for the sources you pick.
``fetch`` pulls the pixels, resizes on write, and rescales the annotations to
match.

The output layout is the repo's usual filesystem split (see
:mod:`cropcounter.crop_dataset`)::

    <out>/
    ├─ train/
    │  ├─ annotations.json          # scaled to the fetched pixels
    │  ├─ annotations.native.json    # as published, before any resize
    │  └─ images/
    ├─ val/                          # same shape
    ├─ download_list.txt
    └─ subset_summary.json

Notes on the real file, verified against the published metadata:

* Image ``id`` is a **string** (``"torsi_20190716-021037.129.JPG"``), and
  annotation ``id`` is a string for some sources and an int for others. Ids are
  copied through verbatim — never renumbered — so a subset joins back to the
  master.
* The master carries **two** categories: ``fish`` (id 1) and ``empty`` (id 0).
  Empty images carry a bbox-less ``category_id: 0`` annotation. Those are
  dropped on write (an empty image in standard COCO simply has no annotation)
  but are what ``n_empty_images`` counts.
* Per-image non-standard fields ``dataset``, ``original_data_source`` and
  ``is_train`` are retained on every written image record.

Imports stay cheap: ijson, PIL and tqdm are imported inside the functions that
need them, so ``import cropcounter.cfd`` costs nothing.
"""
from __future__ import annotations

import csv
import io
import json
import math
import random
import re
import sys
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Sub-directory holding image files inside a split, matching ``crop_dataset``.
IMAGES_DIRNAME = "images"

#: Split folder names written by :func:`run_subset`.
SPLIT_NAMES = ("train", "val")

#: Blob prefixes serving the image files. Same bytes, three clouds.
MIRRORS: Dict[str, str] = {
    "azure": "https://lilawildlife.blob.core.windows.net/lila-wildlife/community-fish-detection-dataset/",
    "gcs": "https://storage.googleapis.com/public-datasets-lila/community-fish-detection-dataset/",
    "aws": "http://us-west-2.opendata.source.coop.s3.amazonaws.com/agentmorris/lila-wildlife/community-fish-detection-dataset/",
}

#: The only category written out. The master also has ``{"id": 0, "name": "empty"}``.
FISH_CATEGORY = {"id": 1, "name": "fish"}

#: ``category_id`` of the master's bbox-less "this image is empty" annotation.
EMPTY_CATEGORY_ID = 0

#: Long side the collision statistic virtually resizes to, and the head stride
#: it then quantises centres onto. Mirrors the training resize in ``fetch``.
COLLISION_LONG_SIDE = 1024
COLLISION_STRIDE = 4

#: Licence per source, keyed by the master's ``dataset`` field. The 17 keys are
#: the exact strings observed in the published metadata; licences are
#: transcribed from the LILA dataset page. Anything not listed reports
#: ``unknown`` and is therefore never permissive.
#:
#: ``marine_detect`` mixes Roboflow (CC-BY-4.0) and GBIF (CC-BY-NC-4.0) parts;
#: the whole source is treated as NC, which is the safe reading.
LICENCES: Dict[str, str] = {
    "brackish_dataset": "CC-BY-SA-4.0",
    "coralscapes": "Apache-2.0",
    "deep_vision": "CC-BY-4.0",
    "deepfish": "MIT",
    "f4k": "unstated",
    "fathomnet": "CC-BY-ND-4.0",
    "fishclef": "unstated",
    "kakadu": "CC-BY-4.0",
    "marine_detect": "CC-BY-NC-4.0",
    "mit_river_herring": "CDLA-Permissive-1.0",
    "noaa_puget": "CDLA-Permissive-1.0",
    "project_natick": "unstated",
    "roboflow_fish": "CC0-1.0",
    "salmon_computer_vision": "CC-BY-NC-SA-4.0",
    "torsi": "CC-BY-NC-SA-4.0",
    "viame_fishtrack": "CC-BY-4.0",
    "zebrafish": "CC-BY-4.0",
}

#: Friendlier names accepted on ``--sources`` and mapped onto :data:`LICENCES`.
LICENCE_ALIASES: Dict[str, str] = {
    "aau_zebrafish": "zebrafish",
    "brackish": "brackish_dataset",
    "fish4knowledge": "f4k",
    "fishclef_2015": "fishclef",
    "fishtrack23": "viame_fishtrack",
    "mit_sea_grant_river_herring": "mit_river_herring",
    "ozfish": "kakadu",
    "puget_sound_nearshore": "noaa_puget",
    "salmon": "salmon_computer_vision",
}

#: Licence strings that permit redistribution and derivative/commercial use.
#: Anything else — ND, NC, ``unstated``, ``unknown`` — is not permissive.
PERMISSIVE_LICENCES = frozenset({
    "CC-BY-4.0", "CC-BY-SA-4.0", "CC0-1.0", "CDLA-Permissive-1.0",
    "Apache-2.0", "MIT",
})

UNKNOWN_LICENCE = "unknown"

#: Percentiles reported for box size.
PERCENTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def licence_for(dataset: str) -> str:
    """Licence string for a master ``dataset`` value, or ``"unknown"``."""
    key = str(dataset).strip().lower()
    key = LICENCE_ALIASES.get(key, key)
    return LICENCES.get(key, UNKNOWN_LICENCE)


def is_permissive(licence: str) -> bool:
    """Whether a licence allows redistribution + derivative commercial use."""
    return licence in PERMISSIVE_LICENCES


# --------------------------------------------------------------------------- #
# Streaming the master JSON
# --------------------------------------------------------------------------- #


@contextmanager
def open_master(path: Path) -> Iterator[BinaryIO]:
    """Open the master metadata as a binary stream.

    Accepts the ``.json`` directly or the published ``.json.zip``, whose single
    JSON member is opened in place — the 1.1 GB file is never extracted to disk.
    """
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [
                info for info in archive.infolist()
                if not info.is_dir() and info.filename.lower().endswith(".json")
            ]
            if not members:
                raise ValueError(f"no .json member inside {path}")
            with archive.open(members[0]) as handle:
                yield handle
    else:
        with path.open("rb") as handle:
            yield handle


def _attach(container: Any, key: Optional[str], value: Any) -> None:
    """Append to a list container, or set ``key`` on a dict container."""
    if isinstance(container, list):
        container.append(value)
    else:
        container[key] = value


def stream_master(
    handle: BinaryIO,
    sections: Sequence[str] = ("images", "annotations"),
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield ``(section, object)`` for every element of the named top-level arrays.

    One forward pass over the document, re-assembling objects from ijson's
    event stream rather than materialising any array. ``images`` precedes
    ``annotations`` in the published file, so a caller can build an image index
    and still see every annotation in the same pass.

    Nested objects and arrays (``bbox``) are rebuilt faithfully; sections not
    listed are skipped without being built.
    """
    import ijson

    wanted = {f"{name}.item": name for name in sections}
    section: Optional[str] = None
    stack: List[Any] = []
    key: Optional[str] = None

    for prefix, event, value in ijson.parse(handle, use_float=True):
        if section is None:
            if event == "start_map" and prefix in wanted:
                section = wanted[prefix]
                stack = [{}]
                key = None
            continue
        if event == "map_key":
            key = value
        elif event in ("start_map", "start_array"):
            child: Any = {} if event == "start_map" else []
            _attach(stack[-1], key, child)
            stack.append(child)
            key = None
        elif event in ("end_map", "end_array"):
            done = stack.pop()
            if not stack:
                yield section, done
                section = None
        else:
            _attach(stack[-1], key, value)


# --------------------------------------------------------------------------- #
# Sequence keys
# --------------------------------------------------------------------------- #

#: A trailing file extension (``.jpg``, ``.JPG``, ``.png`` ...).
_EXTENSION = re.compile(r"\.[A-Za-z0-9]{1,5}$")

#: A trailing frame counter: an optional separator run, an optional ``frame``
#: token, and a final run of digits. Deliberately conservative — it strips ONE
#: trailing number and nothing else, so ``3759335805_train`` keeps its digits
#: (they are not at the end) and ``clip_12_part_3`` only loses the ``_3``.
_FRAME_TAIL = re.compile(r"[-_. ]*(?:frames?|frm|f)?[-_. ]*\d+$", re.IGNORECASE)

#: Extensions that mean ``original_data_source`` names the *clip*, not a frame:
#: ``f4k`` and ``viame_fishtrack`` record ``gt_124.flv`` / ``Clip3.mp4`` for
#: every frame of that video. Stripping a counter off those would glue whole
#: videos together (``gt_124`` and ``gt_125`` -> ``gt``), so they are used whole.
_VIDEO_EXTENSIONS = frozenset({
    "flv", "mp4", "avi", "mov", "mkv", "mpg", "mpeg", "wmv", "webm", "m4v", "ts",
})

#: Roboflow's export suffix ``<stem>_<ext>.rf.<32 hex>[_split]``, seen on
#: ``brackish_dataset``, ``roboflow_fish`` and ``marine_detect``. The hash is
#: per-image, so leaving it on gives one "sequence" per frame and grouping
#: becomes a no-op; stripping it recovers the underlying clip name.
_ROBOFLOW_TAIL = re.compile(
    r"_(?:jpg|jpeg|png|bmp|tif|tiff)\.rf\.[0-9a-f]{6,}(?:_(?:train|valid|val|test))?$",
    re.IGNORECASE,
)


def sequence_key(
    original_data_source: Optional[str] = None,
    file_name: Optional[str] = None,
) -> str:
    """Derive a sequence (video/burst) key from an image's original filename.

    ``video_007_frame_000123.jpg`` -> ``video_007``. Prefers
    ``original_data_source`` — the name inside the source dataset, which is
    where the frame counter lives — and falls back to the basename of
    ``file_name``. Any directory component of ``original_data_source`` is kept:
    two sources that both number frames from zero inside different folders must
    not collapse into one key.

    Two source-shaped exceptions, both forced by the real metadata: when
    ``original_data_source`` is a **video** file (``gt_124.flv``) it already
    names the clip and is used whole, and Roboflow's per-image export hash
    (``..._jpg.rf.<32 hex>``) is removed before the counter so those sources
    group at all.

    When nothing strips (no trailing digits) the stem is returned unchanged, so
    a still-image dataset degrades to one "sequence" per image rather than
    silently gluing unrelated images together.
    """
    raw = (original_data_source or "").strip()
    if not raw:
        raw = (file_name or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    raw = raw.replace("\\", "/").strip()
    if not raw:
        return ""
    extension = raw.rsplit(".", 1)[-1].lower() if "." in raw else ""
    stem = _EXTENSION.sub("", raw)
    if extension in _VIDEO_EXTENSIONS:
        return stem or raw  # already one clip per name
    unhashed = _ROBOFLOW_TAIL.sub("", stem, count=1)
    stripped = _FRAME_TAIL.sub("", unhashed, count=1).rstrip("-_. /")
    return stripped or unhashed or stem


def _prefix_boundary(prefix: str) -> str:
    """Cut a raw common prefix back to its last ``/`` or ``_`` boundary.

    The longest common prefix of a source's filenames runs on into the data
    (``JPEGImages/torsi_2019071``); this trims it to the shortname that actually
    identifies the source (``JPEGImages/torsi_``). Sources whose files carry no
    shortname at all — ``deep_vision``, ``viame_fishtrack`` — correctly report
    just the folder.
    """
    cut = max(prefix.rfind("/"), prefix.rfind("_")) + 1
    return prefix[:cut] if cut > 0 else prefix


def _common_prefix(a: Optional[str], b: str) -> str:
    """Longest common prefix of two strings (``None`` seeds with ``b``)."""
    if a is None:
        return b
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return a[:i]


# --------------------------------------------------------------------------- #
# Small numeric helpers (stdlib only — this module must import cheaply)
# --------------------------------------------------------------------------- #


def _percentile(sorted_values: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile of an already-sorted sequence."""
    n = len(sorted_values)
    if n == 0:
        return None
    if n == 1:
        return float(sorted_values[0])
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _percentiles(values: Iterable[float], qs: Sequence[float] = PERCENTILES) -> List[Optional[float]]:
    """Percentiles of an unsorted iterable, sorted once."""
    ordered = sorted(values)
    return [_percentile(ordered, q) for q in qs]


def _round(value: Optional[float], digits: int = 2) -> Optional[float]:
    """Round, passing ``None`` through."""
    return None if value is None else round(float(value), digits)


# Packed (dataset index, width, height) + a "has at least one box" flag, kept as
# one Python int per image. The image index is ~1.9M entries; a tuple per image
# costs several hundred MB, a single int well under a hundred.
_DS_BITS, _W_BITS = 12, 21
_HAS_BOX = 1 << 62


def _pack_dims(ds_index: int, width: int, height: int) -> int:
    """Pack a source index and image dimensions into one int."""
    width = max(0, min(int(width), (1 << _W_BITS) - 1))
    height = max(0, int(height))
    return (ds_index & ((1 << _DS_BITS) - 1)) | (width << _DS_BITS) | (height << (_DS_BITS + _W_BITS))


def _unpack_dims(packed: int) -> Tuple[int, int, int]:
    """Inverse of :func:`_pack_dims` (the has-box flag is masked off)."""
    packed &= ~_HAS_BOX
    return (
        packed & ((1 << _DS_BITS) - 1),
        (packed >> _DS_BITS) & ((1 << _W_BITS) - 1),
        packed >> (_DS_BITS + _W_BITS),
    )


def collision_cell(
    bbox: Sequence[float],
    width: int,
    height: int,
    long_side: int = COLLISION_LONG_SIDE,
    stride: int = COLLISION_STRIDE,
) -> Tuple[int, int]:
    """Integer head cell a box centre lands in, after the training resize.

    The image is virtually resized so its long side is ``long_side`` (never
    upscaled, matching :func:`fetch_subset`), then the box centre is quantised
    onto a ``stride``-pixel grid. Two boxes sharing a cell cannot both be
    represented by a one-peak-per-cell centre head.
    """
    scale = _resize_scale(width, height, long_side)
    x, y, w, h = (float(v) for v in bbox[:4])
    cx = (x + w / 2.0) * scale
    cy = (y + h / 2.0) * scale
    return int(cx // stride), int(cy // stride)


def _resize_scale(width: int, height: int, max_side: int) -> float:
    """Scale factor putting the long side at ``max_side``, never upscaling."""
    longest = max(int(width), int(height))
    if longest <= 0 or max_side <= 0:
        return 1.0
    return min(1.0, max_side / longest)


def resized_size(width: int, height: int, max_side: int) -> Tuple[int, int, float]:
    """``(new_width, new_height, scale)`` for a long-side resize that never upscales."""
    scale = _resize_scale(width, height, max_side)
    if scale >= 1.0:
        return int(width), int(height), 1.0
    return max(1, round(width * scale)), max(1, round(height * scale)), scale


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #


@dataclass
class SourceStats:
    """Accumulator for one source dataset during the manifest pass."""

    dataset: str
    index: int
    n_images: int = 0
    n_images_with_boxes: int = 0
    n_boxes: int = 0
    n_collisions: int = 0
    n_train: int = 0
    n_val: int = 0
    n_missing_split: int = 0
    prefix: Optional[str] = None
    widths: List[int] = field(default_factory=list)
    heights: List[int] = field(default_factory=list)
    box_px: List[float] = field(default_factory=list)
    box_rel: List[float] = field(default_factory=list)
    sequences: Set[str] = field(default_factory=set)

    def row(self) -> Dict[str, Any]:
        """Flatten into the manifest record written to csv/json."""
        licence = licence_for(self.dataset)
        n_empty = self.n_images - self.n_images_with_boxes
        px = _percentiles(self.box_px)
        rel = _percentiles(self.box_rel)
        row: Dict[str, Any] = {
            "source": self.dataset,
            "prefix": _prefix_boundary(self.prefix or ""),
            "licence": licence,
            "permissive": is_permissive(licence),
            "n_images": self.n_images,
            "n_boxes": self.n_boxes,
            "n_empty_images": n_empty,
            "empty_fraction": _round(n_empty / self.n_images, 4) if self.n_images else None,
            "boxes_per_image": _round(self.n_boxes / self.n_images, 3) if self.n_images else None,
            "is_train_true": self.n_train,
            "is_train_false": self.n_val,
            "is_train_missing": self.n_missing_split,
            "sequences": len(self.sequences),
            "median_width": int(_percentile(sorted(self.widths), 0.5) or 0),
            "median_height": int(_percentile(sorted(self.heights), 0.5) or 0),
            "collision_boxes": self.n_collisions,
            "collision_rate": _round(self.n_collisions / self.n_boxes, 4) if self.n_boxes else None,
        }
        for label, value in zip(("p5", "p25", "p50", "p75", "p95"), px):
            row[f"box_px_{label}"] = _round(value, 1)
        for label, value in zip(("p5", "p25", "p50", "p75", "p95"), rel):
            row[f"box_rel_{label}"] = _round(value, 4)
        return row


#: Manifest columns, in written order.
MANIFEST_COLUMNS = (
    "source", "prefix", "licence", "permissive",
    "n_images", "n_boxes", "n_empty_images", "empty_fraction", "boxes_per_image",
    "is_train_true", "is_train_false", "is_train_missing", "sequences",
    "median_width", "median_height", "collision_boxes", "collision_rate",
    "box_px_p5", "box_px_p25", "box_px_p50", "box_px_p75", "box_px_p95",
    "box_rel_p5", "box_rel_p25", "box_rel_p50", "box_rel_p75", "box_rel_p95",
)

#: The compact subset shown in the stdout / markdown table.
TABLE_COLUMNS = (
    ("source", "source"),
    ("licence", "licence"),
    ("permissive", "perm"),
    ("n_images", "images"),
    ("n_boxes", "boxes"),
    ("empty_fraction", "empty"),
    ("is_train_true", "train"),
    ("is_train_false", "val"),
    ("is_train_missing", "no-split"),
    ("sequences", "seqs"),
    ("median_width", "med W"),
    ("median_height", "med H"),
    ("box_px_p50", "box p50 px"),
    ("box_rel_p50", "box p50 rel"),
    ("collision_rate", "collide@4"),
)


def build_manifest(metadata: Path, progress: bool = True) -> List[Dict[str, Any]]:
    """Profile every source in the master metadata in one streaming pass.

    Returns one row per source plus an ``ALL`` total row, sorted by image count
    descending. Peak memory is dominated by the image index (one packed int per
    image) and the per-image sequence keys.
    """
    stats: Dict[str, SourceStats] = {}
    by_index: List[SourceStats] = []
    dims: Dict[Any, int] = {}
    seen_cells: Set[Tuple[Any, int, int]] = set()
    ticker = _ticker(progress, "manifest")

    with open_master(metadata) as handle:
        for section, obj in stream_master(handle):
            if section == "images":
                _manifest_image(obj, stats, by_index, dims)
            else:
                _manifest_annotation(obj, by_index, dims, seen_cells)
            ticker()
    ticker(final=True)

    rows = [entry.row() for entry in by_index]
    rows.sort(key=lambda r: (-r["n_images"], r["source"]))
    rows.append(_total_row(rows, stats))
    return rows


def _manifest_image(
    image: Dict[str, Any],
    stats: Dict[str, SourceStats],
    by_index: List[SourceStats],
    dims: Dict[Any, int],
) -> None:
    """Fold one image record into the per-source accumulators."""
    name = str(image.get("dataset", "<missing>"))
    entry = stats.get(name)
    if entry is None:
        entry = stats[name] = SourceStats(dataset=name, index=len(by_index))
        by_index.append(entry)

    width = int(image.get("width", 0) or 0)
    height = int(image.get("height", 0) or 0)
    entry.n_images += 1
    entry.widths.append(width)
    entry.heights.append(height)
    entry.prefix = _common_prefix(entry.prefix, str(image.get("file_name", "")))
    entry.sequences.add(
        sequence_key(image.get("original_data_source"), image.get("file_name"))
    )

    split = image.get("is_train")
    if split is True:
        entry.n_train += 1
    elif split is False:
        entry.n_val += 1
    else:
        entry.n_missing_split += 1

    dims[image.get("id")] = _pack_dims(entry.index, width, height)


def _manifest_annotation(
    ann: Dict[str, Any],
    by_index: List[SourceStats],
    dims: Dict[Any, int],
    seen_cells: Set[Tuple[Any, int, int]],
) -> None:
    """Fold one annotation into box-size, emptiness and collision statistics."""
    bbox = ann.get("bbox")
    if ann.get("category_id") == EMPTY_CATEGORY_ID or not bbox or len(bbox) < 4:
        return  # the master's bbox-less "empty image" marker
    image_id = ann.get("image_id")
    packed = dims.get(image_id)
    if packed is None:
        return  # annotation for an image not in the file; nothing to attribute
    ds_index, width, height = _unpack_dims(packed)
    entry = by_index[ds_index]

    entry.n_boxes += 1
    if not packed & _HAS_BOX:
        dims[image_id] = packed | _HAS_BOX
        entry.n_images_with_boxes += 1

    box_w, box_h = float(bbox[2]), float(bbox[3])
    size = math.sqrt(max(box_w, 0.0) * max(box_h, 0.0))
    entry.box_px.append(size)
    longest = max(width, height)
    if longest:
        entry.box_rel.append(size / longest)

    cell = (image_id,) + collision_cell(bbox, width, height)
    if cell in seen_cells:
        entry.n_collisions += 1
    else:
        seen_cells.add(cell)


def _total_row(rows: List[Dict[str, Any]], stats: Dict[str, SourceStats]) -> Dict[str, Any]:
    """An ``ALL`` row summing the per-source rows, with pooled percentiles."""
    total: Dict[str, Any] = {c: None for c in MANIFEST_COLUMNS}
    total["source"] = "ALL"
    total["prefix"] = ""
    total["licence"] = ""
    total["permissive"] = ""
    for key in (
        "n_images", "n_boxes", "n_empty_images", "is_train_true", "is_train_false",
        "is_train_missing", "sequences", "collision_boxes",
    ):
        total[key] = sum(r[key] for r in rows)
    if total["n_images"]:
        total["empty_fraction"] = round(total["n_empty_images"] / total["n_images"], 4)
        total["boxes_per_image"] = round(total["n_boxes"] / total["n_images"], 3)
    if total["n_boxes"]:
        total["collision_rate"] = round(total["collision_boxes"] / total["n_boxes"], 4)
    pooled_px: List[float] = []
    pooled_rel: List[float] = []
    widths: List[int] = []
    heights: List[int] = []
    for entry in stats.values():
        pooled_px.extend(entry.box_px)
        pooled_rel.extend(entry.box_rel)
        widths.extend(entry.widths)
        heights.extend(entry.heights)
    total["median_width"] = int(_percentile(sorted(widths), 0.5) or 0)
    total["median_height"] = int(_percentile(sorted(heights), 0.5) or 0)
    for label, value in zip(("p5", "p25", "p50", "p75", "p95"), _percentiles(pooled_px)):
        total[f"box_px_{label}"] = _round(value, 1)
    for label, value in zip(("p5", "p25", "p50", "p75", "p95"), _percentiles(pooled_rel)):
        total[f"box_rel_{label}"] = _round(value, 4)
    return total


def _format_cell(column: str, value: Any) -> str:
    """Render one table cell: counts grouped, fractions as percentages."""
    if value is None or value == "":
        return "-" if value is None else ""
    if column in ("empty_fraction", "collision_rate"):
        return f"{float(value) * 100:.1f}%"
    if column in ("box_rel_p50",):
        return f"{float(value):.3f}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.1f}"
    return str(value)


def render_table(rows: Sequence[Dict[str, Any]], markdown: bool = False) -> str:
    """Render manifest rows as a fixed-width or markdown table."""
    headers = [label for _, label in TABLE_COLUMNS]
    body = [[_format_cell(col, row.get(col)) for col, _ in TABLE_COLUMNS] for row in rows]
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in body)) if body else len(headers[i])
        for i in range(len(headers))
    ]

    def line(cells: Sequence[str]) -> str:
        padded = [c.ljust(widths[i]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(cells)]
        return ("| " + " | ".join(padded) + " |") if markdown else "  ".join(padded)

    out = [line(headers)]
    if markdown:
        out.append(
            "|" + "|".join(
                (":" + "-" * (w + 1)) if i == 0 else ("-" * (w + 1) + ":")
                for i, w in enumerate(widths)
            ) + "|"
        )
    else:
        out.append("-" * len(out[0]))
    out.extend(line(r) for r in body)
    return "\n".join(out)


def write_manifest(rows: Sequence[Dict[str, Any]], out_dir: Path) -> Dict[str, Path]:
    """Write ``manifest.csv`` / ``.json`` / ``.md`` into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": out_dir / "manifest.csv",
        "json": out_dir / "manifest.json",
        "md": out_dir / "manifest.md",
    }
    with paths["csv"].open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(MANIFEST_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    paths["json"].write_text(json.dumps(list(rows), indent=2), encoding="utf-8")
    paths["md"].write_text(
        "# Community Fish Detection Dataset — per-source manifest\n\n"
        f"{len(rows) - 1} source datasets. `collide@4` is the fraction of boxes whose "
        f"stride-{COLLISION_STRIDE} centre cell is already taken by another box in the same "
        f"image, after a virtual resize to long side {COLLISION_LONG_SIDE}px.\n\n"
        + render_table(rows, markdown=True) + "\n",
        encoding="utf-8",
    )
    return paths


def run_manifest(metadata: Path, out_dir: Path, progress: bool = True) -> List[Dict[str, Any]]:
    """``manifest`` subcommand: profile every source, write the three files."""
    rows = build_manifest(metadata, progress=progress)
    paths = write_manifest(rows, out_dir)
    print(render_table(rows))
    print()
    for kind, path in paths.items():
        print(f"wrote {kind}: {path}")
    return rows


# --------------------------------------------------------------------------- #
# subset
# --------------------------------------------------------------------------- #


def _split_of(image: Dict[str, Any]) -> Optional[str]:
    """``"train"``/``"val"`` from ``is_train``, or ``None`` when it is missing."""
    value = image.get("is_train")
    if value is True:
        return "train"
    if value is False:
        return "val"
    return None


def _trim_uniform(items: Sequence[Any], keep: int) -> List[Any]:
    """Keep ``keep`` items spread evenly across ``items``, order preserved.

    Used only on the sequence that overshoots a cap: a video is decimated in
    time rather than truncated, so the kept frames still span the whole clip.
    """
    n = len(items)
    if keep >= n:
        return list(items)
    if keep <= 0:
        return []
    return [items[(i * n) // keep] for i in range(keep)]


def select_images(
    sequences: Dict[str, List[Dict[str, Any]]],
    cap: Optional[int],
    seed: Any,
    group_by_sequence: bool,
) -> List[Dict[str, Any]]:
    """Sample up to ``cap`` images from one source/split, in master order.

    With grouping on, whole sequences are drawn in a seeded shuffle until the
    cap is reached; only the sequence that overshoots is trimmed, uniformly.
    With grouping off, images are shuffled and sliced individually.
    """
    rng = random.Random(seed)
    if not group_by_sequence:
        flat = [img for group in sequences.values() for img in group]
        flat.sort(key=lambda img: img["_order"])
        if cap is not None and cap < len(flat):
            flat = sorted(rng.sample(flat, cap), key=lambda img: img["_order"])
        return flat

    keys = list(sequences)
    rng.shuffle(keys)
    chosen: List[Dict[str, Any]] = []
    for key in keys:
        group = sequences[key]
        if cap is None:
            chosen.extend(group)
            continue
        remaining = cap - len(chosen)
        if remaining <= 0:
            break
        chosen.extend(group if len(group) <= remaining else _trim_uniform(group, remaining))
    chosen.sort(key=lambda img: img["_order"])
    return chosen


def run_subset(
    metadata: Path,
    out_dir: Path,
    sources: Sequence[str] = ("all",),
    train_cap: Optional[int] = None,
    val_cap: Optional[int] = None,
    permissive_only: bool = False,
    group_by_sequence: bool = True,
    seed: int = 0,
    progress: bool = True,
) -> Dict[str, Any]:
    """``subset`` subcommand: one streaming pass to COCO splits on disk.

    ``is_train`` is honoured verbatim — the LILA split already keeps a location
    in train or val, never both, and re-splitting here would leak it. Images
    with no ``is_train`` are excluded and counted.
    """
    out_dir = Path(out_dir)
    wanted = {LICENCE_ALIASES.get(s.strip().lower(), s.strip()) for s in sources if s.strip()}
    keep_all = "all" in wanted
    caps = {"train": train_cap, "val": val_cap}

    # source -> split -> sequence key -> [image records, in master order]
    pools: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}
    available: Dict[str, Dict[str, int]] = {}
    skipped_no_split: Dict[str, int] = {}
    skipped_licence: Dict[str, int] = {}
    seen_sources: Set[str] = set()
    selection: Optional[Dict[str, List[Dict[str, Any]]]] = None
    keep_ids: Set[Any] = set()
    kept_anns: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": []}
    split_of_id: Dict[Any, str] = {}
    order = 0
    ticker = _ticker(progress, "subset")

    with open_master(metadata) as handle:
        for section, obj in stream_master(handle):
            ticker()
            if section == "images":
                name = str(obj.get("dataset", "<missing>"))
                seen_sources.add(name)
                if not (keep_all or name in wanted):
                    continue
                if permissive_only and not is_permissive(licence_for(name)):
                    skipped_licence[name] = skipped_licence.get(name, 0) + 1
                    continue
                split = _split_of(obj)
                if split is None:
                    skipped_no_split[name] = skipped_no_split.get(name, 0) + 1
                    continue
                obj["_order"] = order
                order += 1
                key = sequence_key(obj.get("original_data_source"), obj.get("file_name"))
                pools.setdefault(name, {}).setdefault(split, {}).setdefault(key, []).append(obj)
                available.setdefault(name, {"train": 0, "val": 0})[split] += 1
                continue

            # First annotation: the images array is complete, so choose now.
            if selection is None:
                selection, keep_ids, split_of_id = _choose(
                    pools, caps, seed, group_by_sequence
                )
            image_id = obj.get("image_id")
            if image_id not in keep_ids:
                continue
            if obj.get("category_id") == EMPTY_CATEGORY_ID or not obj.get("bbox"):
                continue
            kept_anns[split_of_id[image_id]].append(_coco_annotation(obj))

    if selection is None:  # a metadata file with no annotations at all
        selection, keep_ids, split_of_id = _choose(pools, caps, seed, group_by_sequence)
    ticker(final=True)

    unknown = wanted - seen_sources - {"all"}
    if unknown:
        raise ValueError(
            f"unknown source(s) {sorted(unknown)}; the file has {sorted(seen_sources)}"
        )

    summary = _write_subset(
        out_dir, selection, kept_anns, available, skipped_no_split, skipped_licence,
        caps, seed, group_by_sequence, permissive_only, sources,
    )
    print(json.dumps(summary, indent=2))
    return summary


def _choose(
    pools: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]],
    caps: Dict[str, Optional[int]],
    seed: int,
    group_by_sequence: bool,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Set[Any], Dict[Any, str]]:
    """Apply the per-source caps, returning the images kept per split."""
    selection: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": []}
    keep_ids: Set[Any] = set()
    split_of_id: Dict[Any, str] = {}
    for source in sorted(pools):
        for split in SPLIT_NAMES:
            sequences = pools[source].get(split)
            if not sequences:
                continue
            # Seeded per (source, split) so adding a source never reshuffles
            # the images another source already selected.
            chosen = select_images(
                sequences, caps[split], f"{seed}:{source}:{split}", group_by_sequence
            )
            selection[split].extend(chosen)
            for image in chosen:
                keep_ids.add(image.get("id"))
                split_of_id[image.get("id")] = split
    for split in SPLIT_NAMES:
        selection[split].sort(key=lambda img: img["_order"])
    return selection, keep_ids, split_of_id


def _coco_annotation(ann: Dict[str, Any]) -> Dict[str, Any]:
    """Master annotation -> standard COCO, with ``area`` and ``iscrowd`` filled."""
    bbox = [float(v) for v in ann["bbox"][:4]]
    return {
        "id": ann.get("id"),
        "image_id": ann.get("image_id"),
        "category_id": int(ann.get("category_id", FISH_CATEGORY["id"])),
        "bbox": bbox,
        "area": round(bbox[2] * bbox[3], 2),
        "iscrowd": 0,
    }


def _coco_image(image: Dict[str, Any]) -> Dict[str, Any]:
    """Master image -> COCO image record, keeping CFD's extra fields."""
    record = {
        "id": image.get("id"),
        "file_name": image.get("file_name"),
        "width": int(image.get("width", 0) or 0),
        "height": int(image.get("height", 0) or 0),
        "dataset": image.get("dataset"),
        "original_data_source": image.get("original_data_source"),
        "is_train": image.get("is_train"),
        "cfd_sequence": sequence_key(
            image.get("original_data_source"), image.get("file_name")
        ),
    }
    return record


def coco_document(images: Sequence[Dict[str, Any]], annotations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Assemble a standard single-category COCO document."""
    return {
        "images": [_coco_image(img) for img in images],
        "annotations": list(annotations),
        "categories": [dict(FISH_CATEGORY)],
        "info": {
            "description": "Community Fish Detection Dataset subset",
            "source": "https://lila.science/datasets/community-fish-detection-dataset/",
        },
    }


def _write_subset(
    out_dir: Path,
    selection: Dict[str, List[Dict[str, Any]]],
    annotations: Dict[str, List[Dict[str, Any]]],
    available: Dict[str, Dict[str, int]],
    skipped_no_split: Dict[str, int],
    skipped_licence: Dict[str, int],
    caps: Dict[str, Optional[int]],
    seed: int,
    group_by_sequence: bool,
    permissive_only: bool,
    sources: Sequence[str],
) -> Dict[str, Any]:
    """Write both splits, ``download_list.txt`` and ``subset_summary.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    download: List[str] = []
    documents: Dict[str, Dict[str, Any]] = {}

    for split in SPLIT_NAMES:
        images = selection[split]
        split_dir = out_dir / split
        (split_dir / IMAGES_DIRNAME).mkdir(parents=True, exist_ok=True)
        documents[split] = coco_document(images, annotations[split])
        (split_dir / "annotations.json").write_text(
            json.dumps(documents[split], indent=1), encoding="utf-8"
        )
        download.extend(str(img.get("file_name", "")) for img in images)

    names = set(available) | set(skipped_no_split) | set(skipped_licence)
    per_source: Dict[str, Dict[str, Any]] = {}
    for name in sorted(names):
        licence = licence_for(name)
        kept = {
            split: sum(1 for i in documents[split]["images"] if i["dataset"] == name)
            for split in SPLIT_NAMES
        }
        per_source[name] = {
            "licence": licence,
            "permissive": is_permissive(licence),
            "available": available.get(name, {"train": 0, "val": 0}),
            "kept": kept,
            "sequences_kept": {
                split: len({
                    i["cfd_sequence"] for i in documents[split]["images"]
                    if i["dataset"] == name
                })
                for split in SPLIT_NAMES
            },
            "excluded_missing_is_train": skipped_no_split.get(name, 0),
            "excluded_licence": skipped_licence.get(name, 0),
        }

    (out_dir / "download_list.txt").write_text(
        "".join(f"{n}\n" for n in download), encoding="utf-8"
    )
    summary = {
        "sources_requested": list(sources),
        "train_cap": caps["train"],
        "val_cap": caps["val"],
        "seed": seed,
        "group_by_sequence": group_by_sequence,
        "permissive_only": permissive_only,
        "n_images": {s: len(selection[s]) for s in SPLIT_NAMES},
        "n_boxes": {s: len(annotations[s]) for s in SPLIT_NAMES},
        "n_empty_images": {
            s: len(selection[s]) - len({a["image_id"] for a in annotations[s]})
            for s in SPLIT_NAMES
        },
        "n_download": len(download),
        "per_source": per_source,
    }
    (out_dir / "subset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# --------------------------------------------------------------------------- #
# fetch
# --------------------------------------------------------------------------- #


def image_url(file_name: str, mirror: str = "azure") -> str:
    """Public URL of one CFD image on the chosen mirror.

    The path is percent-encoded: ``fishclef`` filenames contain ``#`` (which
    would otherwise start a URL fragment and fetch the wrong object) and
    ``viame_fishtrack`` filenames contain ``:``.
    """
    from urllib.parse import quote

    if mirror not in MIRRORS:
        raise ValueError(f"unknown mirror {mirror!r}; expected one of {sorted(MIRRORS)}")
    return MIRRORS[mirror] + quote(str(file_name).lstrip("/"), safe="/:")


def _download_bytes(url: str, retries: int = 3, timeout: float = 30.0) -> bytes:
    """GET a URL into memory, retrying with linear backoff.

    The single seam the tests monkeypatch: everything downstream (resize,
    encode, annotation rescale) runs for real against generated pixels.
    """
    import time
    import urllib.request

    last: Optional[BaseException] = None
    for attempt in range(max(1, retries)):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001 - any transport error is retryable
            last = exc
            if attempt + 1 < max(1, retries):
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"failed to download {url} after {retries} attempts: {last}")


def _write_resized(data: bytes, dest: Path, max_side: int, jpeg_quality: int) -> Tuple[int, int]:
    """Decode, long-side resize (never upscaling), and save. Returns ``(w, h)``."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        im.load()
        fmt = (im.format or "JPEG").upper()
        width, height = im.size
        new_w, new_h, scale = resized_size(width, height, max_side)
        if scale >= 1.0:
            dest.write_bytes(data)  # no upscale: keep the published bytes
            return width, height
        resized = im.resize((new_w, new_h), Image.LANCZOS)

    if fmt in ("JPEG", "JPG", "MPO"):
        if resized.mode not in ("RGB", "L"):
            resized = resized.convert("RGB")
        resized.save(dest, format="JPEG", quality=jpeg_quality, optimize=True)
    else:
        try:
            resized.save(dest, format=fmt)
        except (KeyError, ValueError, OSError):
            resized.convert("RGB").save(dest, format="JPEG", quality=jpeg_quality)
    return new_w, new_h


def _image_size(path: Path) -> Optional[Tuple[int, int]]:
    """``(width, height)`` from a file header, or ``None`` if unreadable."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            return im.size
    except Exception:  # noqa: BLE001 - a truncated resume artefact re-downloads
        return None


def fetch_one(
    file_name: str,
    dest: Path,
    mirror: str = "azure",
    max_side: int = 1024,
    jpeg_quality: int = 90,
    retries: int = 3,
    expected: Optional[Tuple[int, int]] = None,
) -> Tuple[bool, Optional[Tuple[int, int]]]:
    """Download one image if needed. Returns ``(downloaded, (width, height))``.

    Resume: an existing file whose size already matches ``expected`` is left
    alone and reported as not downloaded.
    """
    if dest.exists() and dest.stat().st_size > 0:
        size = _image_size(dest)
        if size is not None and (expected is None or size == expected):
            return False, size
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = _download_bytes(image_url(file_name, mirror), retries=retries)
    return True, _write_resized(data, dest, max_side, jpeg_quality)


def rescale_annotations(document: Dict[str, Any], scales: Dict[Any, Tuple[int, int, float]]) -> Dict[str, Any]:
    """Return a copy of a COCO document scaled to the fetched pixel sizes.

    ``scales`` maps image id -> ``(width, height, scale)``. Each image gains a
    ``cfd_scale`` field and its ``file_name`` becomes the on-disk basename (the
    master path is kept as ``cfd_file_name``); every bbox and area is multiplied
    by the same factor.
    Images absent from ``scales`` (download failed, or the file is missing from
    every mirror -- a handful of coralscapes PNGs are) are LEFT OUT, with their
    annotations: ``annotations.json`` describes what is on disk, and the
    training loader raises on a listed image it cannot open. The native
    document keeps them, so a later fetch retries and re-admits them.
    """
    out = dict(document)
    images = []
    for image in document.get("images", []):
        if image.get("id") not in scales:
            continue
        record = dict(image)
        width, height, scale = scales.get(
            image.get("id"), (image.get("width"), image.get("height"), 1.0)
        )
        record["width"], record["height"], record["cfd_scale"] = width, height, round(scale, 6)
        # ``annotations.json`` describes what is on disk: fetch writes every image
        # as ``images/<basename>``, so ``file_name`` must be the bare basename or
        # the training loader looks for a ``JPEGImages/`` sub-folder that never
        # existed. The master's path survives as ``cfd_file_name``.
        raw_name = str(image.get("file_name", "")).replace("\\", "/")
        record["cfd_file_name"] = raw_name
        record["file_name"] = raw_name.rsplit("/", 1)[-1]
        images.append(record)
    by_id = {img["id"]: img["cfd_scale"] for img in images}

    annotations = []
    for ann in document.get("annotations", []):
        if ann.get("image_id") not in by_id:
            continue
        record = dict(ann)
        scale = by_id[ann["image_id"]]
        if scale != 1.0:
            record["bbox"] = [round(float(v) * scale, 2) for v in ann["bbox"][:4]]
            record["area"] = round(record["bbox"][2] * record["bbox"][3], 2)
        annotations.append(record)

    out["images"] = images
    out["annotations"] = annotations
    return out


def fetch_subset(
    subset_dir: Path,
    max_side: int = 1024,
    workers: int = 16,
    mirror: str = "azure",
    jpeg_quality: int = 90,
    retries: int = 3,
    progress: bool = True,
) -> Dict[str, Any]:
    """Download a subset's pixels and rescale its annotations to match.

    Library entry point — a notebook can call this directly::

        from cropcounter.cfd import fetch_subset
        fetch_subset("data/brackish", max_side=1024, workers=32)

    Images land in ``<subset>/{train,val}/images/<basename>`` (the
    ``JPEGImages/`` prefix is stripped). The published geometry is preserved as
    ``annotations.native.json`` and ``annotations.json`` is rewritten against the
    files actually on disk, so re-running with a different ``--max-side`` is
    safe and resumable.
    """
    from concurrent.futures import ThreadPoolExecutor

    subset_dir = Path(subset_dir)
    totals = {"downloaded": 0, "skipped": 0, "failed": 0}
    per_split: Dict[str, Dict[str, int]] = {}
    errors: List[str] = []

    for split in SPLIT_NAMES:
        split_dir = subset_dir / split
        native_path = split_dir / "annotations.native.json"
        live_path = split_dir / "annotations.json"
        if not native_path.exists():
            if not live_path.exists():
                continue
            native_path.write_bytes(live_path.read_bytes())
        document = json.loads(native_path.read_text(encoding="utf-8"))
        images = document.get("images", [])
        images_dir = split_dir / IMAGES_DIRNAME
        images_dir.mkdir(parents=True, exist_ok=True)

        jobs = []
        for image in images:
            file_name = str(image.get("file_name", ""))
            dest = images_dir / file_name.replace("\\", "/").rsplit("/", 1)[-1]
            expected = resized_size(
                int(image.get("width", 0) or 0), int(image.get("height", 0) or 0), max_side
            )[:2]
            jobs.append((image.get("id"), file_name, dest, expected))

        scales: Dict[Any, Tuple[int, int, float]] = {}
        counts = {"downloaded": 0, "skipped": 0, "failed": 0, "images": len(images)}
        ticker = _ticker(progress, f"fetch {split}", total=len(images))

        def run(job):
            image_id, file_name, dest, expected = job
            try:
                downloaded, size = fetch_one(
                    file_name, dest, mirror=mirror, max_side=max_side,
                    jpeg_quality=jpeg_quality, retries=retries, expected=expected,
                )
                return image_id, downloaded, size, None
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the pull
                return image_id, False, None, f"{file_name}: {exc}"

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for image_id, downloaded, size, error in pool.map(run, jobs):
                ticker()
                if error is not None:
                    counts["failed"] += 1
                    if len(errors) < 20:
                        errors.append(error)
                    continue
                counts["downloaded" if downloaded else "skipped"] += 1
                scales[image_id] = (size[0], size[1], 1.0)
        ticker(final=True)

        # Resolve the scale factor from native vs on-disk width, so a resumed
        # run and a fresh one agree exactly.
        for image in images:
            entry = scales.get(image.get("id"))
            if entry is None:
                continue
            native_w = int(image.get("width", 0) or 0)
            scale = (entry[0] / native_w) if native_w else 1.0
            scales[image["id"]] = (entry[0], entry[1], scale)

        live_path.write_text(
            json.dumps(rescale_annotations(document, scales), indent=1), encoding="utf-8"
        )
        per_split[split] = counts
        for key in ("downloaded", "skipped", "failed"):
            totals[key] += counts[key]

    result = {"max_side": max_side, "mirror": mirror, "totals": totals,
              "per_split": per_split, "errors": errors}
    for split, counts in per_split.items():
        print(
            f"{split}: {counts['images']:,} images "
            f"({counts['downloaded']:,} downloaded, {counts['skipped']:,} already present, "
            f"{counts['failed']:,} failed -> left out of annotations.json)"
        )
    print(
        f"total: {sum(c['images'] for c in per_split.values()):,} images, "
        f"{totals['downloaded']:,} downloaded, {totals['failed']:,} failed, "
        f"long side <= {max_side}px"
    )
    for error in errors:
        print(f"  ! {error}", file=sys.stderr)
    return result


# --------------------------------------------------------------------------- #
# Progress + CLI
# --------------------------------------------------------------------------- #


def _ticker(enabled: bool, label: str, total: Optional[int] = None):
    """A tqdm-backed counter degrading to a no-op (and to plain counting)."""
    if not enabled:
        return lambda final=False: None
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - tqdm is a hard dependency in practice
        return lambda final=False: None
    bar = tqdm(total=total, desc=label, unit="rec", unit_scale=True, leave=False)

    def tick(final: bool = False) -> None:
        if final:
            bar.close()
        else:
            bar.update(1)

    return tick


def _parse_sources(value: str) -> List[str]:
    """``"a,b"`` -> ``["a", "b"]``; ``"all"`` stays a single sentinel."""
    return [part.strip() for part in str(value).split(",") if part.strip()]


def build_parser():
    """The ``python -m cropcounter.cfd`` argument parser."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m cropcounter.cfd",
        description="Profile, subset and fetch the Community Fish Detection Dataset.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_metadata(p):
        p.add_argument(
            "--metadata", type=Path, required=True,
            help="community_fish_detection_dataset.json or .json.zip",
        )
        p.add_argument("--seed", type=int, default=0, help="sampling seed (default 0)")
        p.add_argument("--no-progress", dest="progress", action="store_false",
                       help="suppress the progress bar")

    p_manifest = sub.add_parser("manifest", help="per-source statistics table")
    add_metadata(p_manifest)
    p_manifest.add_argument("--out", type=Path, required=True, help="output directory")

    p_subset = sub.add_parser("subset", help="write COCO splits for chosen sources")
    add_metadata(p_subset)
    p_subset.add_argument("--out", type=Path, required=True, help="output directory")
    p_subset.add_argument("--sources", default="all",
                          help="comma-separated dataset names, or 'all'")
    p_subset.add_argument("--train-cap", type=int, default=None,
                          help="max train images per source (default: no cap)")
    p_subset.add_argument("--val-cap", type=int, default=None,
                          help="max val images per source (default: no cap)")
    p_subset.add_argument("--permissive-only", action="store_true",
                          help="drop sources whose licence is ND, NC, unstated or unknown")
    group = p_subset.add_mutually_exclusive_group()
    group.add_argument("--group-by-sequence", dest="group_by_sequence",
                       action="store_true", default=True,
                       help="cap whole sequences, not frames (default)")
    group.add_argument("--no-group-by-sequence", dest="group_by_sequence",
                       action="store_false", help="cap individual images")

    p_fetch = sub.add_parser("fetch", help="download a subset's images and rescale")
    p_fetch.add_argument("--subset", type=Path, required=True, help="a subset output directory")
    p_fetch.add_argument("--max-side", type=int, default=1024)
    p_fetch.add_argument("--workers", type=int, default=16)
    p_fetch.add_argument("--mirror", choices=sorted(MIRRORS), default="azure")
    p_fetch.add_argument("--jpeg-quality", type=int, default=90)
    p_fetch.add_argument("--retries", type=int, default=3)
    p_fetch.add_argument("--no-progress", dest="progress", action="store_false")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    if args.command == "manifest":
        run_manifest(args.metadata, args.out, progress=args.progress)
    elif args.command == "subset":
        run_subset(
            args.metadata, args.out, sources=_parse_sources(args.sources),
            train_cap=args.train_cap, val_cap=args.val_cap,
            permissive_only=args.permissive_only,
            group_by_sequence=args.group_by_sequence,
            seed=args.seed, progress=args.progress,
        )
    else:
        fetch_subset(
            args.subset, max_side=args.max_side, workers=args.workers,
            mirror=args.mirror, jpeg_quality=args.jpeg_quality,
            retries=args.retries, progress=args.progress,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
