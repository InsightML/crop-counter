"""Offline tests for the CFD subsetting tool.

Everything runs against a synthetic master JSON built in ``tmp_path`` with
hand-counted expectations (40 images, 3 sources, deliberate empties, a missing
``is_train``, and two kinds of stride-4 centre collision). No network: the one
download seam, ``cfd._download_bytes``, is monkeypatched to return a generated
JPEG so the resize and annotation-rescale paths still run for real.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from cropcounter import cfd

# --------------------------------------------------------------------------- #
# Synthetic master
# --------------------------------------------------------------------------- #
#
# brackish_dataset  permissive  12 images  3 clips x 4 frames  1 empty   22 boxes  2 collisions
# fathomnet         ND          16 images  16 singletons       4 empty   13 boxes  1 collision
# torsi             NC-SA       12 images  2 clips x 6 frames  0 empty   12 boxes  0 collisions
#
# The fathomnet collision only appears once the virtual resize to long side
# 1024 is applied: its two centres are 6px apart at native 2048px width, and
# 3px apart (same stride-4 cell) after the 0.5x scale.

BRACKISH_W, BRACKISH_H = 1000, 500
FATHOMNET_W, FATHOMNET_H = 2048, 1024
TORSI_W, TORSI_H = 640, 480

EXPECTED = {
    "brackish_dataset": {
        "n_images": 12, "n_boxes": 22, "n_empty_images": 1,
        "is_train_true": 8, "is_train_false": 4, "is_train_missing": 0,
        "sequences": 3, "collision_boxes": 2,
    },
    "fathomnet": {
        "n_images": 16, "n_boxes": 13, "n_empty_images": 4,
        "is_train_true": 8, "is_train_false": 4, "is_train_missing": 4,
        "sequences": 16, "collision_boxes": 1,
    },
    "torsi": {
        "n_images": 12, "n_boxes": 12, "n_empty_images": 0,
        "is_train_true": 6, "is_train_false": 6, "is_train_missing": 0,
        "sequences": 2, "collision_boxes": 0,
    },
}

#: Every image's native (width, height), filled while building the master.
NATIVE_SIZES: dict = {}


def _image(image_id, file_name, width, height, dataset, original, is_train):
    """One master image record; ``is_train=None`` omits the field entirely."""
    record = {
        "id": image_id, "file_name": file_name, "width": width, "height": height,
        "dataset": dataset, "original_data_source": original,
    }
    if is_train is not None:
        record["is_train"] = is_train
    NATIVE_SIZES[file_name.rsplit("/", 1)[-1]] = (width, height)
    return record


def build_master() -> dict:
    """The synthetic master COCO document the whole module tests against."""
    images, annotations = [], []
    ann_id = 0

    def box(image_id, x, y, w, h):
        nonlocal ann_id
        annotations.append({
            "id": ann_id, "image_id": image_id, "category_id": 1,
            "bbox": [float(x), float(y), float(w), float(h)],
        })
        ann_id += 1

    def empty(image_id):
        nonlocal ann_id
        annotations.append({"id": ann_id, "image_id": image_id, "category_id": 0})
        ann_id += 1

    # --- brackish_dataset: three Roboflow-exported clips of four frames ------
    hashes = iter(f"{i:032x}" for i in range(100))
    for clip, is_train in (("clipA", True), ("clipB", True), ("clipC", False)):
        for frame in range(4):
            original = f"{clip}-{frame:04d}_jpg.rf.{next(hashes)}.jpg"
            image_id = f"brackish_dataset_{original}"
            images.append(_image(
                image_id, f"JPEGImages/{image_id}", BRACKISH_W, BRACKISH_H,
                "brackish_dataset", original, is_train,
            ))
            if clip == "clipC" and frame == 3:
                empty(image_id)                     # the one empty brackish image
            elif (clip, frame) in (("clipA", 0), ("clipB", 2)):
                box(image_id, 40, 40, 10, 10)       # centre (45, 45) -> cell (11, 11)
                box(image_id, 42, 42, 10, 10)       # centre (47, 47) -> cell (11, 11)
            else:
                box(image_id, 0, 0, 20, 20)         # cell (2, 2)
                box(image_id, 400, 200, 20, 20)     # cell (102, 52)

    # --- fathomnet: 16 counter-free names, 4 empty, 4 without is_train -------
    words = [
        "Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel",
        "India", "Juliett", "Kilo", "Lima", "Mike", "November", "Oscar", "Papa",
    ]
    for i, word in enumerate(words):
        is_train = True if i < 8 else (False if i < 12 else None)
        original = f"object{word}.png"
        image_id = f"fathomnet_{original}"
        images.append(_image(
            image_id, f"JPEGImages/{image_id}", FATHOMNET_W, FATHOMNET_H,
            "fathomnet", original, is_train,
        ))
        if word in ("Mike", "November", "Oscar", "Papa"):
            empty(image_id)
        elif word == "Alpha":
            box(image_id, 100, 100, 8, 8)           # native centre (104, 104)
            box(image_id, 106, 100, 8, 8)           # native centre (110, 104)
        else:
            box(image_id, 500, 500, 40, 40)

    # --- torsi: two six-frame clips, one per split --------------------------
    for clip, is_train in (("torsiclip1", True), ("torsiclip2", False)):
        for frame in range(6):
            original = f"{clip}_frame_{frame:03d}.jpg"
            image_id = f"torsi_{original}"
            images.append(_image(
                image_id, f"JPEGImages/{image_id}", TORSI_W, TORSI_H,
                "torsi", original, is_train,
            ))
            box(image_id, 10 * frame, 20, 30, 30)

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "fish"}, {"id": 0, "name": "empty"}],
        "info": {"version": "test", "description": "synthetic CFD"},
    }


@pytest.fixture(scope="module")
def master() -> dict:
    return build_master()


@pytest.fixture(scope="module")
def master_json(tmp_path_factory, master) -> Path:
    path = tmp_path_factory.mktemp("cfd") / "master.json"
    path.write_text(json.dumps(master), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def master_zip(tmp_path_factory, master) -> Path:
    path = tmp_path_factory.mktemp("cfd_zip") / "master.json.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("community_fish_detection_dataset.json", json.dumps(master))
    return path


def rows_by_source(rows):
    return {row["source"]: row for row in rows}


# --------------------------------------------------------------------------- #
# Streaming + sequence keys
# --------------------------------------------------------------------------- #


def test_stream_master_round_trips_every_record(master_json, master):
    with cfd.open_master(master_json) as handle:
        streamed = list(cfd.stream_master(handle))
    images = [obj for section, obj in streamed if section == "images"]
    annotations = [obj for section, obj in streamed if section == "annotations"]
    assert images == master["images"]
    assert annotations == master["annotations"]


@pytest.mark.parametrize("name,expected", [
    # the spec's worked example
    ("video_007_frame_000123.jpg", "video_007"),
    # separators and frame tokens
    ("clip_12_part_3.jpg", "clip_12_part"),
    ("Coonamessett_1086961_images_default_frame_000000.PNG",
     "Coonamessett_1086961_images_default"),
    ("20180602_23-41-08-0730-1430_frame_835.jpg", "20180602_23-41-08-0730-1430"),
    ("Vid1_0001.png", "Vid1"),
    ("7623_F2_f000280.jpg", "7623_F2"),
    ("test/07-15-2020_18-48-46_m_left_bank_underwater_frame_000000.jpg",
     "test/07-15-2020_18-48-46_m_left_bank_underwater"),
    # nothing strips -> the stem itself
    ("site4_000001_000200_leftImg8bit.png", "site4_000001_000200_leftImg8bit"),
    ("objectAlpha.png", "objectAlpha"),
    # all-digits: stripping would empty it, so fall back
    ("4.jpg", "4"),
    ("0001.png", "0001"),
    # a video filename names the clip already — never strip its counter
    ("gt_124.flv", "gt_124"),
    ("CDFW-LakeCam-Misc-SpiderBlocks3.mp4", "CDFW-LakeCam-Misc-SpiderBlocks3"),
    # Roboflow's per-image export hash is removed before the frame counter
    ("2019-03-19_17-01-06to2019-03-19_17-01-19_1-0113_jpg.rf."
     "004b062faf5b7af1a8e3f9c514b99a81.jpg",
     "2019-03-19_17-01-06to2019-03-19_17-01-19_1"),
    # only ONE trailing run goes: leading digits survive
    ("3759335805_train.jpg", "3759335805_train"),
])
def test_sequence_key_table(name, expected):
    assert cfd.sequence_key(name) == expected


def test_sequence_key_falls_back_to_file_name():
    assert cfd.sequence_key(None, "JPEGImages/torsi_clip_frame_004.jpg") == "torsi_clip"
    assert cfd.sequence_key("", "") == ""


def test_licence_table_marks_permissive_correctly():
    assert cfd.licence_for("brackish_dataset") == "CC-BY-SA-4.0"
    assert cfd.is_permissive(cfd.licence_for("brackish_dataset"))
    # ND and NC are not permissive, nor is an unlisted source
    assert not cfd.is_permissive(cfd.licence_for("fathomnet"))
    assert not cfd.is_permissive(cfd.licence_for("salmon_computer_vision"))
    assert cfd.licence_for("no_such_source") == cfd.UNKNOWN_LICENCE
    assert not cfd.is_permissive(cfd.UNKNOWN_LICENCE)


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #


def test_manifest_counts_are_exact(master_json):
    rows = rows_by_source(cfd.build_manifest(master_json, progress=False))
    assert set(rows) == set(EXPECTED) | {"ALL"}
    for source, expected in EXPECTED.items():
        row = rows[source]
        for key, value in expected.items():
            assert row[key] == value, f"{source}.{key}"


def test_manifest_empty_fraction_and_collision_rate(master_json):
    rows = rows_by_source(cfd.build_manifest(master_json, progress=False))
    assert rows["brackish_dataset"]["empty_fraction"] == pytest.approx(1 / 12, abs=5e-5)
    assert rows["brackish_dataset"]["collision_rate"] == pytest.approx(2 / 22, abs=5e-5)
    assert rows["fathomnet"]["empty_fraction"] == 0.25
    assert rows["fathomnet"]["collision_rate"] == pytest.approx(1 / 13, abs=5e-5)
    assert rows["torsi"]["collision_rate"] == 0.0
    assert rows["torsi"]["empty_fraction"] == 0.0


def test_collision_cell_uses_the_virtual_resize():
    # 6px apart natively on a 2048px-wide image -> 3px apart at long side 1024
    a = cfd.collision_cell([100, 100, 8, 8], FATHOMNET_W, FATHOMNET_H)
    b = cfd.collision_cell([106, 100, 8, 8], FATHOMNET_W, FATHOMNET_H)
    assert a == b == (13, 13)
    # the same boxes on a small image are NOT resized, so they separate
    assert cfd.collision_cell([100, 100, 8, 8], 640, 480) != \
        cfd.collision_cell([106, 100, 8, 8], 640, 480)


def test_manifest_total_row_sums_the_sources(master_json):
    rows = cfd.build_manifest(master_json, progress=False)
    total = rows[-1]
    assert total["source"] == "ALL"
    assert total["n_images"] == 40
    assert total["n_boxes"] == 22 + 13 + 12
    assert total["n_empty_images"] == 5
    assert total["is_train_missing"] == 4


def test_manifest_box_percentiles_and_medians(master_json):
    rows = rows_by_source(cfd.build_manifest(master_json, progress=False))
    torsi = rows["torsi"]
    assert torsi["median_width"] == TORSI_W and torsi["median_height"] == TORSI_H
    # every torsi box is 30x30 -> sqrt(w*h) == 30 at every percentile
    assert torsi["box_px_p5"] == torsi["box_px_p95"] == 30.0
    assert torsi["box_rel_p50"] == pytest.approx(30 / TORSI_W, abs=5e-5)


def test_manifest_reports_the_file_name_prefix(master_json):
    rows = rows_by_source(cfd.build_manifest(master_json, progress=False))
    assert rows["torsi"]["prefix"] == "JPEGImages/torsi_"
    assert rows["fathomnet"]["prefix"] == "JPEGImages/fathomnet_"


def test_manifest_writes_csv_json_and_markdown(master_json, tmp_path):
    out = tmp_path / "manifest"
    rows = cfd.run_manifest(master_json, out, progress=False)
    assert json.loads((out / "manifest.json").read_text()) == rows
    csv_text = (out / "manifest.csv").read_text()
    assert csv_text.splitlines()[0].split(",") == list(cfd.MANIFEST_COLUMNS)
    assert len(csv_text.strip().splitlines()) == len(rows) + 1
    markdown = (out / "manifest.md").read_text()
    assert markdown.startswith("# Community Fish Detection Dataset")
    assert markdown.count("|") > 0 and "brackish_dataset" in markdown


def test_manifest_from_zip_matches_plain_json(master_json, master_zip):
    assert cfd.build_manifest(master_zip, progress=False) == \
        cfd.build_manifest(master_json, progress=False)


# --------------------------------------------------------------------------- #
# subset
# --------------------------------------------------------------------------- #


def run_subset(master, out, **kwargs):
    kwargs.setdefault("progress", False)
    return cfd.run_subset(master, out, **kwargs)


def read_split(out: Path, split: str) -> dict:
    return json.loads((out / split / "annotations.json").read_text())


def test_subset_renumbers_annotation_ids_as_ints(master, tmp_path):
    """pycocotools stores annotation ids in a float array while matching, so a
    source whose master ids are strings (the salmon-camera clips are) crashes
    COCOeval unless the subset writer renumbers. The master id survives as
    ``cfd_id``."""
    doc = json.loads(json.dumps(master))
    for ann in doc["annotations"]:
        if str(ann["image_id"]).startswith("torsi"):
            ann["id"] = f"{ann['image_id']}_{ann['id']}"
    path = tmp_path / "master_str_ids.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    out = tmp_path / "ints"
    cfd.run_subset(path, out, sources=["torsi", "brackish_dataset"], progress=False)
    for split in ("train", "val"):
        anns = read_split(out, split)["annotations"]
        ids = [a["id"] for a in anns]
        assert all(isinstance(i, int) for i in ids)
        assert len(set(ids)) == len(ids)
        assert any(isinstance(a["cfd_id"], str) for a in anns)   # the torsi originals
        assert all("cfd_id" in a for a in anns)


def test_subset_honours_is_train_and_never_resplits(master_json, tmp_path):
    out = tmp_path / "all"
    summary = run_subset(master_json, out, sources=["all"])
    train, val = read_split(out, "train"), read_split(out, "val")
    assert all(img["is_train"] is True for img in train["images"])
    assert all(img["is_train"] is False for img in val["images"])
    # 8 + 8 + 6 train, 4 + 4 + 6 val; the 4 fathomnet images with no is_train
    # are excluded outright and counted, never quietly assigned to a split.
    assert summary["n_images"] == {"train": 22, "val": 14}
    assert summary["per_source"]["fathomnet"]["excluded_missing_is_train"] == 4
    ids = {img["id"] for img in train["images"]} | {img["id"] for img in val["images"]}
    assert not any("Papa" in i or "Oscar" in i for i in ids)


def test_subset_writes_valid_single_category_coco(master_json, tmp_path):
    out = tmp_path / "coco"
    run_subset(master_json, out, sources=["all"])
    for split in ("train", "val"):
        document = read_split(out, split)
        assert document["categories"] == [{"id": 1, "name": "fish"}]
        image_ids = {img["id"] for img in document["images"]}
        for ann in document["annotations"]:
            assert ann["image_id"] in image_ids
            assert ann["category_id"] == 1
            assert len(ann["bbox"]) == 4 and ann["iscrowd"] == 0
            assert ann["area"] == pytest.approx(ann["bbox"][2] * ann["bbox"][3])
        # the master's bbox-less category-0 markers never survive
        assert all(ann["category_id"] != 0 for ann in document["annotations"])
        # CFD's extra per-image fields are retained verbatim
        for img in document["images"]:
            assert set(img) >= {"dataset", "original_data_source", "is_train", "file_name"}


def test_subset_ids_are_preserved_verbatim(master_json, tmp_path, master):
    out = tmp_path / "ids"
    run_subset(master_json, out, sources=["torsi"])
    written = {img["id"] for img in read_split(out, "train")["images"]}
    original = {i["id"] for i in master["images"]
                if i["dataset"] == "torsi" and i.get("is_train") is True}
    assert written == original


def test_subset_cap_takes_whole_sequences(master_json, tmp_path):
    out = tmp_path / "whole"
    summary = run_subset(
        master_json, out, sources=["brackish_dataset"], train_cap=4, val_cap=100,
    )
    train = read_split(out, "train")["images"]
    assert len(train) == 4
    # exactly one complete clip, never four frames taken across two clips
    assert len({img["cfd_sequence"] for img in train}) == 1
    assert summary["per_source"]["brackish_dataset"]["available"]["train"] == 8
    assert len(read_split(out, "val")["images"]) == 4


def test_subset_trims_only_the_overshooting_sequence(master_json, tmp_path):
    out = tmp_path / "trim"
    run_subset(master_json, out, sources=["brackish_dataset"], train_cap=6)
    train = read_split(out, "train")["images"]
    assert len(train) == 6
    per_sequence = {}
    for img in train:
        per_sequence[img["cfd_sequence"]] = per_sequence.get(img["cfd_sequence"], 0) + 1
    # one clip whole (4), one trimmed to the remaining budget (2)
    assert sorted(per_sequence.values()) == [2, 4]


def test_trim_uniform_spreads_across_the_clip():
    assert cfd._trim_uniform(list(range(8)), 4) == [0, 2, 4, 6]
    assert cfd._trim_uniform(list(range(4)), 2) == [0, 2]
    assert cfd._trim_uniform(list(range(3)), 9) == [0, 1, 2]
    assert cfd._trim_uniform(list(range(3)), 0) == []


def test_subset_without_grouping_caps_individual_images(master_json, tmp_path):
    out = tmp_path / "nogroup"
    run_subset(
        master_json, out, sources=["brackish_dataset"], train_cap=6,
        group_by_sequence=False, seed=0,
    )
    train = read_split(out, "train")["images"]
    assert len(train) == 6
    # uniform image sampling straddles clips; grouping is what prevents that
    assert len({img["cfd_sequence"] for img in train}) >= 2


def test_subset_is_deterministic_under_a_seed(master_json, tmp_path):
    def pick(directory, seed):
        out = tmp_path / directory
        run_subset(master_json, out, sources=["all"], train_cap=5, seed=seed)
        return (out / "download_list.txt").read_text()

    assert pick("seed0a", 0) == pick("seed0b", 0)
    assert pick("seed0a", 0) != "" and pick("seed7", 7) != ""


def test_subset_permissive_only_drops_nd_and_nc_sources(master_json, tmp_path):
    out = tmp_path / "permissive"
    summary = run_subset(master_json, out, sources=["all"], permissive_only=True)
    kept = {img["dataset"] for img in read_split(out, "train")["images"]}
    kept |= {img["dataset"] for img in read_split(out, "val")["images"]}
    assert kept == {"brackish_dataset"}
    assert summary["per_source"]["fathomnet"]["excluded_licence"] == 16
    assert summary["per_source"]["torsi"]["excluded_licence"] == 12


def test_subset_download_list_covers_both_splits(master_json, tmp_path):
    out = tmp_path / "dl"
    summary = run_subset(master_json, out, sources=["all"])
    lines = (out / "download_list.txt").read_text().splitlines()
    assert len(lines) == summary["n_download"] == 36
    assert all(line.startswith("JPEGImages/") for line in lines)
    assert len(set(lines)) == len(lines)


def test_subset_rejects_an_unknown_source(master_json, tmp_path):
    with pytest.raises(ValueError, match="unknown source"):
        run_subset(master_json, tmp_path / "bad", sources=["not_a_dataset"])


def test_subset_from_zip_matches_plain_json(master_zip, master_json, tmp_path):
    a, b = tmp_path / "fromzip", tmp_path / "fromjson"
    run_subset(master_zip, a, sources=["all"], train_cap=5)
    run_subset(master_json, b, sources=["all"], train_cap=5)
    assert (a / "download_list.txt").read_text() == (b / "download_list.txt").read_text()
    assert read_split(a, "train") == read_split(b, "train")


# --------------------------------------------------------------------------- #
# fetch
# --------------------------------------------------------------------------- #


def jpeg_bytes(width: int, height: int) -> bytes:
    """A real JPEG of the given size, so PIL decodes a genuine image."""
    from PIL import Image

    image = Image.new("RGB", (width, height))
    for x in range(0, width, 16):  # some structure, so it is not a flat file
        for y in range(0, height, 16):
            image.putpixel((x, y), ((x * 7) % 256, (y * 5) % 256, 128))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


@pytest.fixture
def fake_download(monkeypatch):
    """Serve the synthetic images from memory, at their native sizes."""
    from urllib.parse import unquote

    calls = []

    def fake(url, retries=3, timeout=30.0):
        name = unquote(url.rsplit("/", 1)[-1])
        calls.append(name)
        return jpeg_bytes(*NATIVE_SIZES[name])

    monkeypatch.setattr(cfd, "_download_bytes", fake)
    return calls


@pytest.fixture
def fetched(master_json, tmp_path, fake_download):
    """A brackish+torsi subset with its pixels fetched at max_side=512."""
    out = tmp_path / "fetchme"
    cfd.run_subset(
        master_json, out, sources=["brackish_dataset", "torsi"], progress=False
    )
    result = cfd.fetch_subset(out, max_side=512, workers=4, progress=False)
    return out, result, fake_download


def test_fetch_downloads_every_image_by_basename(fetched):
    out, result, calls = fetched
    assert result["totals"]["failed"] == 0
    assert result["totals"]["downloaded"] == 24
    assert result["per_split"]["train"]["images"] == 14   # 8 brackish + 6 torsi
    assert result["per_split"]["val"]["images"] == 10     # 4 brackish + 6 torsi
    files = list((out / "train" / cfd.IMAGES_DIRNAME).iterdir())
    assert len(files) == 14
    # the JPEGImages/ prefix is stripped; only the basename lands on disk
    assert all("/" not in f.name and not f.name.startswith("JPEGImages") for f in files)


def test_fetch_rescales_width_height_bbox_and_area(fetched):
    out, _, _ = fetched
    native = {i["id"]: i for i in
              json.loads((out / "train" / "annotations.native.json").read_text())["images"]}
    document = read_split(out, "train")
    scale = 512 / BRACKISH_W  # 1000px long side -> 512

    brackish = [i for i in document["images"] if i["dataset"] == "brackish_dataset"]
    assert brackish
    for image in brackish:
        assert image["cfd_scale"] == pytest.approx(scale)
        assert (image["width"], image["height"]) == (512, 256)
        assert (native[image["id"]]["width"], native[image["id"]]["height"]) == \
            (BRACKISH_W, BRACKISH_H)

    native_anns = {
        a["id"]: a for a in
        json.loads((out / "train" / "annotations.native.json").read_text())["annotations"]
    }
    scaled = [a for a in document["annotations"]
              if a["image_id"] in {i["id"] for i in brackish}]
    assert scaled
    for ann in scaled:
        before = native_anns[ann["id"]]["bbox"]
        assert ann["bbox"] == pytest.approx([v * scale for v in before], abs=0.01)
        assert ann["area"] == pytest.approx(ann["bbox"][2] * ann["bbox"][3], abs=0.01)


def test_fetch_never_upscales(fetched):
    out, _, _ = fetched
    from PIL import Image

    document = read_split(out, "train")
    torsi = [i for i in document["images"] if i["dataset"] == "torsi"]
    assert torsi
    for image in torsi:  # native long side 640 < max_side 512? no: 640 > 512
        assert (image["width"], image["height"]) == (512, 384)

    # a max_side above every native size must leave the geometry untouched
    assert cfd.resized_size(TORSI_W, TORSI_H, 4096) == (TORSI_W, TORSI_H, 1.0)
    assert cfd.resized_size(BRACKISH_W, BRACKISH_H, 1024) == (BRACKISH_W, BRACKISH_H, 1.0)
    assert cfd.resized_size(0, 0, 512) == (0, 0, 1.0)
    for path in (out / "train" / cfd.IMAGES_DIRNAME).iterdir():
        with Image.open(path) as im:
            assert max(im.size) <= 512


def test_fetch_at_a_larger_max_side_keeps_native_geometry(
    master_json, tmp_path, fake_download
):
    out = tmp_path / "native"
    cfd.run_subset(master_json, out, sources=["torsi"], progress=False)
    cfd.fetch_subset(out, max_side=4096, workers=2, progress=False)
    document = read_split(out, "train")
    native = json.loads((out / "train" / "annotations.native.json").read_text())
    for image in document["images"]:
        assert image["cfd_scale"] == 1.0
        assert (image["width"], image["height"]) == (TORSI_W, TORSI_H)
    assert [a["bbox"] for a in document["annotations"]] == \
        [a["bbox"] for a in native["annotations"]]


def test_fetch_resumes_without_redownloading(fetched):
    out, _, calls = fetched
    calls.clear()
    again = cfd.fetch_subset(out, max_side=512, workers=4, progress=False)
    assert calls == []
    assert again["totals"]["downloaded"] == 0
    assert again["totals"]["skipped"] == 24
    # the rewrite is idempotent: a resumed run reproduces the same annotations
    assert again["totals"]["failed"] == 0


def test_fetch_survives_a_failing_download(master_json, tmp_path, monkeypatch):
    out = tmp_path / "broken"
    cfd.run_subset(master_json, out, sources=["torsi"], progress=False)

    def boom(url, retries=3, timeout=30.0):
        raise RuntimeError("nope")

    monkeypatch.setattr(cfd, "_download_bytes", boom)
    result = cfd.fetch_subset(out, max_side=512, workers=2, progress=False)
    assert result["totals"]["failed"] == 12 and result["totals"]["downloaded"] == 0
    # Failed images are left out of the live annotations (with their boxes):
    # the loader raises on a listed image it cannot open, and a few CFD files
    # are 404 on every mirror. The native document keeps them for a retry.
    for split in ("train", "val"):
        live = read_split(out, split)
        assert live["images"] == [] and live["annotations"] == []
        native = json.loads((out / split / "annotations.native.json").read_text())
        assert len(native["images"]) > 0
        for image in native["images"]:
            assert (image["width"], image["height"]) == (TORSI_W, TORSI_H)


def test_fetch_drops_only_the_failed_images(master_json, tmp_path, monkeypatch, fake_download):
    """One 404 removes that image and its boxes; every other image is intact."""
    out = tmp_path / "one_bad"
    cfd.run_subset(master_json, out, sources=["torsi"], progress=False)
    native = json.loads((out / "train" / "annotations.native.json").read_text()) \
        if (out / "train" / "annotations.native.json").exists() \
        else read_split(out, "train")
    victim = native["images"][0]
    real = cfd._download_bytes

    def flaky(url, retries=3, timeout=30.0):
        if victim["file_name"].rsplit("/", 1)[-1] in url:
            raise RuntimeError("HTTP Error 404: Not Found")
        return real(url, retries=retries, timeout=timeout)

    monkeypatch.setattr(cfd, "_download_bytes", flaky)
    result = cfd.fetch_subset(out, max_side=512, workers=2, progress=False)
    assert result["totals"]["failed"] == 1
    live = read_split(out, "train")
    assert victim["id"] not in {img["id"] for img in live["images"]}
    assert all(ann["image_id"] != victim["id"] for ann in live["annotations"])
    assert len(live["images"]) == len(native["images"]) - 1
    n_victim_boxes = sum(1 for a in native["annotations"] if a["image_id"] == victim["id"])
    assert len(live["annotations"]) == len(native["annotations"]) - n_victim_boxes


def test_image_url_percent_encodes_awkward_filenames():
    url = cfd.image_url("JPEGImages/fishclef_sub_abc#201106111440_0_frame_27.jpg")
    assert "#" not in url and "%23" in url
    assert url.startswith(cfd.MIRRORS["azure"])
    # a colon is legal in a URL path and stays readable
    assert ":" in cfd.image_url("JPEGImages/CDFW_00:00:16.200000.jpg").rsplit("/", 1)[-1]
    for mirror in cfd.MIRRORS:
        assert cfd.image_url("JPEGImages/a.jpg", mirror).endswith("JPEGImages/a.jpg")
    with pytest.raises(ValueError, match="unknown mirror"):
        cfd.image_url("a.jpg", "dropbox")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_runs_manifest_and_subset(master_zip, tmp_path, capsys):
    assert cfd.main([
        "manifest", "--metadata", str(master_zip),
        "--out", str(tmp_path / "m"), "--no-progress",
    ]) == 0
    assert (tmp_path / "m" / "manifest.md").exists()
    assert "brackish_dataset" in capsys.readouterr().out

    assert cfd.main([
        "subset", "--metadata", str(master_zip), "--out", str(tmp_path / "s"),
        "--sources", "brackish_dataset,torsi", "--train-cap", "4",
        "--val-cap", "4", "--seed", "3", "--no-group-by-sequence", "--no-progress",
    ]) == 0
    summary = json.loads((tmp_path / "s" / "subset_summary.json").read_text())
    assert summary["group_by_sequence"] is False and summary["seed"] == 3
    assert summary["n_images"] == {"train": 8, "val": 8}


def test_fetch_rewrites_file_name_to_the_on_disk_basename(fetched):
    """The loader resolves ``images_dir / file_name``; after fetch the images sit
    at ``images/<basename>``, so ``file_name`` must be the bare basename (the
    smoke run on real Brackish frames failed on ``JPEGImages/...``)."""
    out, _result, _calls = fetched
    for split in ("train", "val"):
        split_dir = Path(out) / split
        document = json.loads((split_dir / "annotations.json").read_text(encoding="utf-8"))
        native = json.loads((split_dir / "annotations.native.json").read_text(encoding="utf-8"))
        native_names = {img["id"]: img["file_name"] for img in native["images"]}
        for image in document["images"]:
            assert "/" not in image["file_name"]
            assert (split_dir / "images" / image["file_name"]).exists()
            assert image["cfd_file_name"] == native_names[image["id"]]


# --------------------------------------------------------------------------- #
# points
# --------------------------------------------------------------------------- #
#
# The fetched brackish+torsi subset, hand-counted:
#   train  14 images  22 boxes  0 empty   (clipA 8 + clipB 8 + torsiclip1 6)
#   val    10 images  12 boxes  1 empty   (clipC 6 + torsiclip2 6; clipC frame 3)

POINTS_EXPECTED = {
    "train": {"n_images": 14, "n_boxes": 22, "n_empty": 0},
    "val": {"n_images": 10, "n_boxes": 12, "n_empty": 1},
}


def _centre(bbox):
    x, y, w, h = (float(v) for v in bbox[:4])
    return x + w / 2.0, y + h / 2.0


def test_points_keypoints_are_exact_box_centres(fetched):
    out, _result, _calls = fetched
    for split, expected in POINTS_EXPECTED.items():
        document = read_split(out, split)
        points, _id_map = cfd.bbox_centres_to_keypoints(document)

        assert points["categories"] == [
            {"id": 1, "name": "fish", "keypoints": ["fish"], "skeleton": []}
        ]
        assert len(points["annotations"]) == expected["n_boxes"]
        by_box = {tuple(a["bbox"]): a for a in points["annotations"]}
        for ann in document["annotations"]:
            keyed = by_box[tuple(ann["bbox"])]
            cx, cy = _centre(ann["bbox"])
            assert keyed["keypoints"] == [cx, cy, 2]
            assert keyed["num_keypoints"] == 1
            assert keyed["category_id"] == 1
            # bbox/area ride along untouched
            assert keyed["bbox"] == ann["bbox"] and keyed["area"] == ann["area"]


def test_points_renumbers_ids_to_ints_and_the_map_round_trips(fetched):
    out, _result, _calls = fetched
    document = read_split(out, "train")
    points, id_map = cfd.bbox_centres_to_keypoints(document)

    image_ids = [img["id"] for img in points["images"]]
    assert image_ids == list(range(1, len(document["images"]) + 1))
    assert all(isinstance(i, int) and not isinstance(i, bool) for i in image_ids)

    ann_ids = [a["id"] for a in points["annotations"]]
    assert ann_ids == list(range(1, len(points["annotations"]) + 1))
    assert all(isinstance(i, int) for i in ann_ids)

    # the string id survives on the record, and the map is its inverse
    for original, new in id_map.items():
        assert isinstance(original, str) and isinstance(new, int)
    assert {img["cfd_image_id"]: img["id"] for img in points["images"]} == id_map
    assert id_map == {img["id"]: n + 1 for n, img in enumerate(document["images"])}

    # every extra CFD per-image field is carried through verbatim
    source = {img["id"]: img for img in document["images"]}
    for image in points["images"]:
        original = source[image["cfd_image_id"]]
        for key in ("file_name", "width", "height", "dataset", "is_train",
                    "cfd_sequence", "cfd_scale", "cfd_file_name",
                    "original_data_source"):
            assert image[key] == original[key], key

    # annotation image_ids point at the new ints, never the old strings
    assert {a["image_id"] for a in points["annotations"]} <= set(image_ids)


def test_points_document_reads_back_through_parse_coco_keypoints(fetched, tmp_path):
    from cropcounter.crop_dataset import parse_coco_keypoints

    out, _result, _calls = fetched
    for split, expected in POINTS_EXPECTED.items():
        document = read_split(out, split)
        points, _id_map = cfd.bbox_centres_to_keypoints(document)
        path = tmp_path / f"{split}_points.json"
        path.write_text(json.dumps(points), encoding="utf-8")

        records = parse_coco_keypoints(path, labels=["fish"])
        # empty images ARE kept as records with zero points (parse_coco_keypoints
        # creates one record per entry in "images" and only then attaches points)
        assert len(records) == expected["n_images"]
        assert sum(len(r.points) for r in records) == expected["n_boxes"]
        assert sum(1 for r in records if not r.points) == expected["n_empty"]
        for record, image in zip(records, points["images"]):
            assert record.name == image["file_name"]
            assert (record.width, record.height) == (image["width"], image["height"])
        assert all(p.label == "fish" for r in records for p in r.points)

        # the default labels=("Wheat", "Volunteer") silently drop every fish
        assert sum(len(r.points) for r in parse_coco_keypoints(path)) == 0


def test_raw_bbox_subset_is_unreadable_as_points(fetched, tmp_path):
    """Negative control for the two reasons conversion must happen here.

    CFD image ids are strings, so ``parse_coco_keypoints``' ``int(image["id"])``
    raises; and even with int ids a bbox-only document loads silently EMPTY
    because it carries no ``keypoints``.
    """
    from cropcounter.crop_dataset import parse_coco_keypoints

    out, _result, _calls = fetched
    with pytest.raises(ValueError):
        parse_coco_keypoints(Path(out) / "train" / "annotations.json", labels=["fish"])

    document = read_split(out, "train")
    renumbered = {img["id"]: n + 1 for n, img in enumerate(document["images"])}
    document["images"] = [dict(i, id=renumbered[i["id"]]) for i in document["images"]]
    document["annotations"] = [
        dict(a, image_id=renumbered[a["image_id"]]) for a in document["annotations"]
    ]
    int_ids = tmp_path / "int_ids_boxes.json"
    int_ids.write_text(json.dumps(document), encoding="utf-8")
    records = parse_coco_keypoints(int_ids, labels=["fish"])
    assert len(records) == POINTS_EXPECTED["train"]["n_images"]
    assert sum(len(r.points) for r in records) == 0  # silently empty


def test_points_drop_empty_removes_exactly_the_empty_frames(fetched):
    out, _result, _calls = fetched
    document = read_split(out, "val")
    kept, kept_map = cfd.bbox_centres_to_keypoints(document)
    dropped, dropped_map = cfd.bbox_centres_to_keypoints(document, drop_empty=True)

    n_empty = POINTS_EXPECTED["val"]["n_empty"]
    assert len(kept["images"]) == POINTS_EXPECTED["val"]["n_images"]
    assert len(dropped["images"]) == len(kept["images"]) - n_empty
    assert len(dropped["annotations"]) == len(kept["annotations"])

    boxed = {a["image_id"] for a in document["annotations"]}
    gone = {img["id"] for img in document["images"]} - boxed
    assert len(gone) == n_empty
    assert set(dropped_map) == set(kept_map) - gone
    # ids stay consecutive after the drop
    assert [i["id"] for i in dropped["images"]] == list(range(1, len(dropped["images"]) + 1))


def test_points_refuses_an_unfetched_subset(master_json, tmp_path):
    out = tmp_path / "unfetched"
    cfd.run_subset(master_json, out, sources=["torsi"], progress=False)
    document = read_split(out, "train")
    assert all("cfd_scale" not in img for img in document["images"])
    with pytest.raises(ValueError, match="cfd_scale"):
        cfd.bbox_centres_to_keypoints(document)
    with pytest.raises(ValueError, match="cfd_scale"):
        cfd.write_points_root(out, tmp_path / "unfetched_points")


def test_cli_points_writes_both_splits_the_id_map_and_the_summary(fetched, tmp_path):
    out, _result, _calls = fetched
    points_root = tmp_path / "points_root"
    assert cfd.main([
        "points", "--subset", str(out), "--out", str(points_root),
    ]) == 0

    id_map = json.loads((points_root / "cfd_id_map.json").read_text())
    summary = json.loads((points_root / "points_summary.json").read_text())
    for split, expected in POINTS_EXPECTED.items():
        document = json.loads(
            (points_root / split / "annotations.json").read_text(encoding="utf-8")
        )
        assert len(document["images"]) == expected["n_images"]
        assert len(document["annotations"]) == expected["n_boxes"]
        assert len(id_map[split]) == expected["n_images"]
        assert summary[split] == {
            "n_images": expected["n_images"],
            "n_points": expected["n_boxes"],
            "n_empty": expected["n_empty"],
            "dropped_empty": 0,
            # This fixture's boxes sit well inside their frames; the real
            # subset drops 48 across both splits.
            "n_dropped_outside_frame": 0,
            "category": "fish",
            "subset": str(Path(out).resolve() / split),
        }
        # images/ is a symlink into the bbox subset, not a second copy
        images = points_root / split / cfd.IMAGES_DIRNAME
        assert images.is_symlink()
        assert images.resolve() == (Path(out) / split / cfd.IMAGES_DIRNAME).resolve()
        for image in document["images"]:
            assert (images / image["file_name"]).is_file()

    # the separate root matters: resolve_annotations picks annotations.json first,
    # so a points file sitting beside the bbox one could never win.
    from cropcounter.crop_dataset import resolve_annotations
    assert resolve_annotations(points_root / "train", "coco") == \
        points_root / "train" / "annotations.json"


def test_points_copy_images_writes_real_files(fetched, tmp_path):
    out, _result, _calls = fetched
    copied = tmp_path / "points_copied"
    cfd.write_points_root(out, copied, copy_images=True)
    images = copied / "train" / cfd.IMAGES_DIRNAME
    assert not images.is_symlink() and images.is_dir()
    assert len(list(images.iterdir())) == POINTS_EXPECTED["train"]["n_images"]


# --------------------------------------------------------------------------- #
# Out-of-frame box centres
#
# CFD carries boxes whose centre falls outside the image — 47 of 222,152 train
# boxes (0.021%) and 1 of 52,038 val boxes, the worst 454px outside a 1024x683
# frame. The box task never noticed: albumentations silently drops a box below
# min_visibility. The point task DIES, because KeypointParams raises on an
# out-of-range keypoint, and it took down a whole 8-epoch run 33 minutes in.
#
# They are dropped, not clipped: a centre outside the frame has no cell in the
# heatmap to live in, and clipping it to the edge would train the model to fire
# at image borders where nothing is.
# --------------------------------------------------------------------------- #


def _doc(width, height, boxes):
    """A one-image fetched document with the given boxes."""
    return {
        "images": [{"id": "img_a", "file_name": "a.jpg", "width": width,
                    "height": height, "cfd_scale": 1.0}],
        "annotations": [
            {"id": i + 1, "image_id": "img_a", "category_id": 1, "bbox": list(b)}
            for i, b in enumerate(boxes)
        ],
        "categories": [{"id": 1, "name": "fish"}],
    }


def test_a_centre_outside_the_frame_is_dropped():
    """The exact failure: x=1247.4 in a 1024-wide image."""
    document = _doc(1024, 768, [
        [1200.0, 650.0, 94.81, 52.01],   # centre (1247.4, 676.0) — x out of range
        [100.0, 100.0, 50.0, 50.0],      # centre (125, 125) — fine
    ])
    points, _ = cfd.bbox_centres_to_keypoints(document)
    assert len(points["annotations"]) == 1
    assert points["annotations"][0]["keypoints"][:2] == [125.0, 125.0]


def test_every_surviving_centre_is_inside_its_frame():
    document = _doc(1024, 768, [
        [1200.0, 650.0, 94.0, 52.0],     # x outside
        [900.0, 800.0, 40.0, 61.0],      # y outside  (centre y = 830.5)
        [-80.0, 100.0, 40.0, 40.0],      # centre x = -60, negative
        [10.0, 10.0, 20.0, 20.0],        # good
    ])
    points, _ = cfd.bbox_centres_to_keypoints(document)
    assert len(points["annotations"]) == 1
    for ann in points["annotations"]:
        x, y, _v = ann["keypoints"]
        assert 0 <= x < 1024 and 0 <= y < 768


def test_a_box_clipping_the_edge_keeps_its_centre():
    """Only the CENTRE decides. A half-off box is still a findable object."""
    document = _doc(1024, 768, [[-20.0, 100.0, 100.0, 50.0]])   # centre (30, 125)
    points, _ = cfd.bbox_centres_to_keypoints(document)
    assert len(points["annotations"]) == 1
    assert points["annotations"][0]["keypoints"][:2] == [30.0, 125.0]


def test_the_drop_count_is_reported_not_silent():
    """Silently losing annotations is how a data bug becomes a model result."""
    document = _doc(1024, 768, [[1200.0, 650.0, 94.0, 52.0], [10.0, 10.0, 20.0, 20.0]])
    points, _ = cfd.bbox_centres_to_keypoints(document)
    assert points["n_dropped_outside_frame"] == 1


def test_an_image_without_a_size_keeps_its_points():
    """No width/height means no bound to test — keep, never silently drop all."""
    document = _doc(1024, 768, [[1200.0, 650.0, 94.0, 52.0]])
    del document["images"][0]["width"]
    del document["images"][0]["height"]
    points, _ = cfd.bbox_centres_to_keypoints(document)
    assert len(points["annotations"]) == 1


def test_write_points_root_surfaces_the_drop_count(fetched, tmp_path):
    summary = cfd.write_points_root(fetched[0], tmp_path / "pts")
    for split in ("train", "val"):
        assert "n_dropped_outside_frame" in summary[split]
