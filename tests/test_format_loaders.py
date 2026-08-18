"""Unit tests for cropcounter.crop_dataset's annotation-format loaders.

Tiny inline CVAT-1.1 XML and COCO-keypoints JSON fixtures describing the
same two images, points, and QC attributes -- parse_cvat_1_1 and
parse_coco_keypoints must produce identical ImageRecord/Point sequences.
No real dataset, no network.
"""
from __future__ import annotations

import json

import pytest

from cropcounter.crop_dataset import (
    ImageRecord,
    Point,
    load_records,
    parse_coco_keypoints,
    parse_cvat_1_1,
)

CVAT_XML = """<?xml version='1.0' encoding='utf-8'?>
<annotations>
  <version>1.1</version>
  <image id="0" name="img001.jpg" width="100" height="200">
    <points label="Wheat" occluded="0" points="10.5,20.5;30.0,40.0" z_order="0">
      <attribute name="Confidence">85</attribute>
      <attribute name="Label Status">Correct</attribute>
    </points>
    <points label="Volunteer" occluded="1" points="50.0,60.0" z_order="0">
      <attribute name="Confidence">40</attribute>
    </points>
  </image>
  <image id="1" name="img002.jpg" width="150" height="250">
  </image>
</annotations>
"""

# Same two images/points/attributes as CVAT_XML above, in COCO-keypoints shape.
COCO_JSON = {
    "images": [
        {"id": 1, "file_name": "img001.jpg", "width": 100, "height": 200},
        {"id": 2, "file_name": "img002.jpg", "width": 150, "height": 250},
    ],
    "categories": [
        {"id": 1, "name": "Wheat"},
        {"id": 2, "name": "Volunteer"},
    ],
    "annotations": [
        {
            "image_id": 1,
            "category_id": 1,
            # v=2: visible -> occluded False, matching the CVAT group's occluded="0".
            "keypoints": [10.5, 20.5, 2, 30.0, 40.0, 2],
            "attributes": {"Confidence": 85, "Label Status": "Correct"},
        },
        {
            "image_id": 1,
            "category_id": 2,
            # v=1: labelled-but-not-visible -> occluded True, matching occluded="1".
            "keypoints": [50.0, 60.0, 1],
            "attributes": {"Confidence": 40},
        },
    ],
}


def _records_as_tuples(records):
    """Flatten ImageRecords into plain, order-preserving comparable tuples."""
    return [
        (
            r.name, r.width, r.height,
            [(p.x, p.y, p.label, p.confidence, p.label_status, p.occluded) for p in r.points],
        )
        for r in records
    ]


@pytest.fixture
def cvat_path(tmp_path):
    path = tmp_path / "annotations.xml"
    path.write_text(CVAT_XML, encoding="utf-8")
    return path


@pytest.fixture
def coco_path(tmp_path):
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(COCO_JSON), encoding="utf-8")
    return path


def test_cvat_parses_expected_records(cvat_path):
    records = parse_cvat_1_1(cvat_path)
    assert len(records) == 2

    img1, img2 = records
    assert img1 == ImageRecord(
        name="img001.jpg", width=100, height=200,
        points=[
            Point(10.5, 20.5, "Wheat", confidence=85, label_status="Correct", occluded=False),
            Point(30.0, 40.0, "Wheat", confidence=85, label_status="Correct", occluded=False),
            Point(50.0, 60.0, "Volunteer", confidence=40, label_status=None, occluded=True),
        ],
    )
    # Images with no <points> at all still yield an (empty) record.
    assert img2 == ImageRecord(name="img002.jpg", width=150, height=250, points=[])


def test_coco_parses_expected_records(coco_path):
    records = parse_coco_keypoints(coco_path)
    assert len(records) == 2

    img1, img2 = records
    assert img1 == ImageRecord(
        name="img001.jpg", width=100, height=200,
        points=[
            Point(10.5, 20.5, "Wheat", confidence=85, label_status="Correct", occluded=False),
            Point(30.0, 40.0, "Wheat", confidence=85, label_status="Correct", occluded=False),
            Point(50.0, 60.0, "Volunteer", confidence=40, label_status=None, occluded=True),
        ],
    )
    assert img2 == ImageRecord(name="img002.jpg", width=150, height=250, points=[])


def test_cvat_and_coco_loaders_agree_on_identical_data(cvat_path, coco_path):
    """The whole point of a format registry: every loader emits the same
    ImageRecord/Point shape for the same underlying annotations."""
    cvat_records = load_records(cvat_path, fmt="cvat")
    coco_records = load_records(coco_path, fmt="coco")
    assert _records_as_tuples(cvat_records) == _records_as_tuples(coco_records)


def test_label_filter_drops_unlisted_labels(cvat_path):
    records = load_records(cvat_path, fmt="cvat", labels=("Wheat",))
    img1 = records[0]
    assert {p.label for p in img1.points} == {"Wheat"}
    assert len(img1.points) == 2


def test_labels_none_keeps_every_label(cvat_path):
    records = load_records(cvat_path, fmt="cvat", labels=None)
    img1 = records[0]
    assert {p.label for p in img1.points} == {"Wheat", "Volunteer"}


def test_load_records_unknown_format_raises(tmp_path):
    with pytest.raises(ValueError):
        load_records(tmp_path, fmt="bogus")


def test_load_records_missing_annotation_file_raises(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_records(empty_dir, fmt="cvat")
