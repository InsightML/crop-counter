"""Hand-counted unit tests for the point-in-box scorer.

The scorer lives in ``examples/FishDetection/scripts/`` (a worked example, not
an installed package), so it is imported by path. CI collects ``tests/`` only,
and this repo has no conftest.py — the path insert therefore lives here.

Every expected value below is computed by hand in a comment. The matcher is a
verbatim copy of Liam's ``match_image`` from
``examples/WheatHead/notebooks/4_evaluate.ipynb``; these tests pin its
semantics (inclusive box edges, confidence-descending greedy order,
nearest-box-centre tiebreak) so a later edit cannot silently change the
numbers the wheat report was built on.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "examples" / "FishDetection" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import point_in_box as pib  # noqa: E402


def _pts(*rows):
    return np.array(rows, dtype=float).reshape(-1, 3)


def _boxes(*rows):
    return np.array(rows, dtype=float).reshape(-1, 4)


def test_perfect_match():
    """3 boxes, one point at each centre -> every point inside its own box."""
    boxes = _boxes((0, 0, 10, 10), (20, 20, 30, 30), (40, 0, 60, 20))
    # centres: (5, 5), (25, 25), (50, 10)
    points = _pts((5, 5, 0.9), (25, 25, 0.8), (50, 10, 0.7))
    # hand count: TP=3 (each point inside a distinct free box), FP=0, FN=0
    assert pib.match_image(points, boxes, 0.0) == (3, 0, 0)


def test_two_points_in_one_box():
    """One box, two points inside it: greedy one-to-one keeps one."""
    boxes = _boxes((0, 0, 10, 10))
    points = _pts((2, 2, 0.9), (8, 8, 0.4))
    # hand count: 0.9 point matches the only box -> TP=1; 0.4 point finds the
    # box already used and is inside no other box -> FP=1; no box left -> FN=0
    assert pib.match_image(points, boxes, 0.0) == (1, 1, 0)


def test_nearest_centre_tiebreak():
    """A point inside two free boxes takes the one whose CENTRE is nearer.

    Box order is deliberately the reverse of the answer, so a 'first free
    containing box' implementation would fail this test.
    """
    #        index 0 = B, centre (14, 5)        index 1 = A, centre (5, 5)
    boxes = _boxes((4, 0, 24, 10), (0, 0, 10, 10))
    p1 = (6, 5, 0.9)  # inside B (4<=6<=24) and A (0<=6<=10); |A centre|=1, |B centre|=8 -> A
    p2 = (2, 5, 0.5)  # inside A only (4<=2 is False) -> A already used -> FP
    points = _pts(p1, p2)
    # hand count: TP=1 (p1 -> A), FP=1 (p2, A taken), FN=1 (B never matched)
    assert pib.match_image(points, boxes, 0.0) == (1, 1, 1)
    # sanity on the tiebreak direction: had p1 taken B, p2 would have matched A -> (2, 0, 0)


def test_confidence_ordering_changes_assignment():
    """Processing order is confidence-descending, and it changes the outcome."""
    #        A centre (5, 5)                B centre (11, 5)
    boxes = _boxes((0, 0, 10, 10), (6, 0, 16, 10))
    inner = (7, 5)  # inside A and B; nearer A's centre (2 vs 4)
    outer = (2, 5)  # inside A only
    # inner first (higher conf): inner -> A (nearer centre); outer finds A used
    # and is in no other box -> FP; B unmatched -> FN.  (1, 1, 1)
    assert pib.match_image(_pts((*inner, 0.9), (*outer, 0.1)), boxes, 0.0) == (1, 1, 1)
    # outer first (higher conf): outer -> A; inner then finds A used but B free
    # and containing -> TP.  (2, 0, 0)
    assert pib.match_image(_pts((*inner, 0.1), (*outer, 0.9)), boxes, 0.0) == (2, 0, 0)
    # and with one box + two inside points, the threshold keeps the higher-conf
    # one and it is the TP: 0.2 < 0.5 <= 0.9  -> (1, 0, 0)
    assert pib.match_image(_pts((2, 2, 0.9), (8, 8, 0.2)), _boxes((0, 0, 10, 10)), 0.5) == (1, 0, 0)


def test_conf_thr_turns_tp_into_fn():
    """The same point is a TP below its confidence and absent above it."""
    boxes = _boxes((0, 0, 10, 10))
    points = _pts((5, 5, 0.3))
    assert pib.match_image(points, boxes, 0.0) == (1, 0, 0)  # kept: 0.3 >= 0.0
    assert pib.match_image(points, boxes, 0.3) == (1, 0, 0)  # kept: >= is inclusive
    assert pib.match_image(points, boxes, 0.5) == (0, 0, 1)  # dropped -> box is a FN
    # the sweep sees the same cliff: F1 = 2*1/(2*1+0+0) = 1.0 then 0/(0+0+1) = 0.0
    sweep = pib.sweep_thresholds({"im": points}, {"im": boxes}, [0.0, 0.5])
    assert [s["conf_thr"] for s in sweep] == [0.0, 0.5]
    assert sweep[0]["f1"] == 1.0
    assert sweep[1]["f1"] == 0.0
    assert sweep[1]["recall"] == 0.0


def test_empty_image_no_predictions_is_perfect():
    """No GT and no predictions -> 0/0 := 1.0 (a correctly-empty image)."""
    tp, fp, fn = pib.match_image(_pts(), _boxes(), 0.0)
    assert (tp, fp, fn) == (0, 0, 0)
    assert pib.image_accuracy(tp, fp, fn) == 1.0
    # and through the dataset scorer, where the image is absent from preds entirely
    summary, rows = pib.score_dataset({}, {"empty": _boxes()}, 0.0)
    assert rows == [
        {"image_id": "empty", "gt_count": 0, "pred_count": 0,
         "tp": 0, "fp": 0, "fn": 0, "accuracy": 1.0}
    ]
    assert summary["mean_accuracy"] == 1.0
    assert (summary["n_images"], summary["n_gt"], summary["n_pred"]) == (1, 0, 0)
    # precision/recall/f1 are 0/0 on an all-empty dataset -> defined as 0.0
    assert (summary["precision"], summary["recall"], summary["f1"]) == (0.0, 0.0, 0.0)


def test_empty_image_with_predictions_is_zero():
    """No GT, two predictions -> both FP, accuracy 0/(0+2+0) = 0.0."""
    tp, fp, fn = pib.match_image(_pts((1, 1, 0.9), (2, 2, 0.8)), _boxes(), 0.0)
    assert (tp, fp, fn) == (0, 2, 0)
    assert pib.image_accuracy(tp, fp, fn) == 0.0
    summary, rows = pib.score_dataset(
        {"empty": _pts((1, 1, 0.9), (2, 2, 0.8))}, {"empty": _boxes()}, 0.0
    )
    assert rows[0]["pred_count"] == 2 and rows[0]["accuracy"] == 0.0
    # count error: pred 2 vs gt 0 -> MAE 2, RMSE 2, bias +2
    assert (summary["count_mae"], summary["count_rmse"], summary["count_bias"]) == (2.0, 2.0, 2.0)


def test_no_predictions_all_boxes_are_fn():
    """M boxes, zero predictions -> (0, 0, M); accuracy 0/(0+0+3) = 0.0."""
    boxes = _boxes((0, 0, 1, 1), (2, 2, 3, 3), (4, 4, 5, 5))
    tp, fp, fn = pib.match_image(_pts(), boxes, 0.0)
    assert (tp, fp, fn) == (0, 0, 3)
    assert pib.image_accuracy(tp, fp, fn) == 0.0


def test_coco_results_to_points_centres_and_id_map():
    """xywh -> centre is (x + w/2, y + h/2); score passes through; ids remap."""
    results = [
        {"image_id": "a", "category_id": 1, "bbox": [10.0, 20.0, 30.0, 40.0], "score": 0.25},
        {"image_id": "a", "category_id": 1, "bbox": [0.0, 0.0, 4.0, 6.0], "score": 0.75},
        {"image_id": "b", "category_id": 1, "bbox": [100.0, 200.0, 10.0, 10.0], "score": 0.5},
    ]
    got = pib.coco_results_to_points(results)
    assert sorted(got) == ["a", "b"]
    # (10 + 15, 20 + 20) = (25, 40); (0 + 2, 0 + 3) = (2, 3)
    assert np.allclose(got["a"], [[25.0, 40.0, 0.25], [2.0, 3.0, 0.75]])
    # (100 + 5, 200 + 5) = (105, 205)
    assert np.allclose(got["b"], [[105.0, 205.0, 0.5]])
    remapped = pib.coco_results_to_points(results, id_map={"a": 7, "b": 9})
    assert sorted(remapped) == [7, 9]
    assert np.allclose(remapped[7], got["a"])
    # an id missing from id_map is left untouched
    partial = pib.coco_results_to_points(results, id_map={"a": 7})
    assert sorted(partial, key=str) == [7, "b"]


def test_cvat_round_trip(tmp_path):
    """results -> CVAT points XML -> parsed points reproduces x, y and conf."""
    results = [
        {"image_id": 1, "category_id": 1, "bbox": [10.0, 20.0, 31.0, 41.0], "score": 0.37},
        {"image_id": 1, "category_id": 1, "bbox": [100.5, 200.25, 9.0, 11.0], "score": 0.91},
        {"image_id": 2, "category_id": 1, "bbox": [0.0, 0.0, 7.0, 7.0], "score": 0.04},
    ]
    images = [
        {"id": 1, "file_name": "one.jpg", "width": 960, "height": 540},
        {"id": 2, "file_name": "two.jpg", "width": 960, "height": 540},
    ]
    xml_path = tmp_path / "preds.xml"
    pib.coco_results_to_cvat_points(results, images, xml_path, label="fish")
    parsed = pib.parse_pred_points(xml_path)
    assert sorted(parsed) == ["one.jpg", "two.jpg"]
    # centres: (10 + 15.5, 20 + 20.5) = (25.5, 40.5); (100.5 + 4.5, 200.25 + 5.5) = (105, 205.75)
    expected_one = np.array([[25.5, 40.5, 0.37], [105.0, 205.75, 0.91]])
    assert np.allclose(parsed["one.jpg"], expected_one, atol=0.005)
    # (0 + 3.5, 0 + 3.5) = (3.5, 3.5); conf 0.04 -> written as int 4 -> read back 0.04
    assert np.allclose(parsed["two.jpg"], [[3.5, 3.5, 0.04]], atol=0.005)
    # the same file shape also scores: a COCO annotations doc gives the GT boxes,
    # and an image with no annotations still appears with an empty (0, 4) array.
    ann_path = tmp_path / "annotations.json"
    ann_path.write_text(json.dumps({
        "images": [{"id": "one.jpg", "file_name": "one.jpg", "width": 960, "height": 540},
                   {"id": "two.jpg", "file_name": "two.jpg", "width": 960, "height": 540}],
        "annotations": [{"id": 1, "image_id": "one.jpg", "category_id": 1,
                         "bbox": [10.0, 20.0, 31.0, 41.0], "area": 1271.0, "iscrowd": 0}],
        "categories": [{"id": 1, "name": "fish"}],
    }))
    gt = pib.coco_boxes_by_image(ann_path)
    assert np.allclose(gt["one.jpg"], [[10.0, 20.0, 41.0, 61.0]])  # xywh -> xyxy
    assert gt["two.jpg"].shape == (0, 4)


def test_point_on_box_edge_is_inside():
    """Liam's inequalities are INCLUSIVE (<=), so edges and corners count."""
    boxes = _boxes((0, 0, 10, 10))
    for point in [(0.0, 0.0), (10.0, 10.0), (0.0, 5.0), (10.0, 5.0), (5.0, 0.0), (5.0, 10.0)]:
        assert pib.match_image(_pts((*point, 0.9)), boxes, 0.0) == (1, 0, 0), point
    # one pixel outside the edge is a FP, and the box is then a FN
    assert pib.match_image(_pts((10.001, 5.0, 0.9)), boxes, 0.0) == (0, 1, 1)
