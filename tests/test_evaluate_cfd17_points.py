"""The CPU-only point scorer for CFD-17.

Hand-built fixtures with counted expectations: three val frames from two
sources, one of them deliberately empty. The scorer lives in
``examples/FishDetection/scripts/`` (a worked example, not an installed
package), so it is imported by path, as ``test_point_in_box.py`` does.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "examples" / "FishDetection" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import evaluate_cfd17_points as ev  # noqa: E402


@pytest.fixture
def bbox_root(tmp_path):
    """A 3-image val split: 2 boxes in brackish_a, 1 in torsi_a, 1 empty frame."""
    root = tmp_path / "cfd17"
    (root / "val").mkdir(parents=True)
    document = {
        "images": [
            {"id": "brackish_a.jpg", "file_name": "brackish_a.jpg",
             "width": 100, "height": 100, "dataset": "brackish"},
            {"id": "torsi_a.jpg", "file_name": "torsi_a.jpg",
             "width": 100, "height": 100, "dataset": "torsi"},
            {"id": "torsi_empty.jpg", "file_name": "torsi_empty.jpg",
             "width": 100, "height": 100, "dataset": "torsi"},
        ],
        "annotations": [
            {"id": 1, "image_id": "brackish_a.jpg", "category_id": 1,
             "bbox": [10, 10, 20, 20]},
            {"id": 2, "image_id": "brackish_a.jpg", "category_id": 1,
             "bbox": [50, 50, 20, 20]},
            {"id": 3, "image_id": "torsi_a.jpg", "category_id": 1,
             "bbox": [0, 0, 40, 40]},
        ],
        "categories": [{"id": 1, "name": "fish"}],
    }
    (root / "val" / "annotations.json").write_text(json.dumps(document), encoding="utf-8")
    return root


def _write_points(path: Path, points: dict, detect_tau: float = 0.01) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "detect_tau": detect_tau, "output_stride": 4, "nms_radius": 1.5, "k": 3,
        "points": points,
    }), encoding="utf-8")
    return path


def test_parse_taus_includes_the_stop():
    assert ev.parse_taus("0.1:0.3:0.1") == [0.1, 0.2, 0.3]
    assert ev.parse_taus("0.5:0.5:0.1") == [0.5]


def test_parse_taus_rejects_a_non_positive_step():
    with pytest.raises(ValueError, match="step must be positive"):
        ev.parse_taus("0.1:0.3:0")


def test_ground_truth_keeps_empty_frames_and_their_source(bbox_root):
    gt, sources = ev.load_ground_truth(bbox_root)
    assert set(gt) == {"brackish_a.jpg", "torsi_a.jpg", "torsi_empty.jpg"}
    assert len(gt["brackish_a.jpg"]) == 2
    assert len(gt["torsi_empty.jpg"]) == 0, "an empty frame must survive as (0, 4)"
    assert sources == {
        "brackish_a.jpg": "brackish", "torsi_a.jpg": "torsi", "torsi_empty.jpg": "torsi",
    }
    # COCO xywh -> xyxy
    assert list(gt["torsi_a.jpg"][0]) == [0.0, 0.0, 40.0, 40.0]


def test_a_box_runs_predictions_file_is_refused(tmp_path):
    """A COCO-results list has no 'points' key; say so instead of scoring 0.0."""
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps([{"image_id": "a", "bbox": [0, 0, 1, 1], "score": 0.9}]))
    with pytest.raises(ValueError, match="BOX run"):
        ev.load_predictions(path)


def test_discover_runs_ignores_box_runs(tmp_path):
    runs = tmp_path / "runs"
    _write_points(runs / "points_run" / "predictions.json", {"a.jpg": []})
    (runs / "box_run").mkdir(parents=True)
    (runs / "box_run" / "predictions.json").write_text(json.dumps([{"image_id": "a"}]))
    assert [name for name, _ in ev.discover_runs(runs)] == ["points_run"]


def test_a_perfect_point_run_scores_one(bbox_root, tmp_path, capsys):
    """One confident point at each box centre -> P = R = F1 = 1, MAE 0."""
    runs = tmp_path / "runs"
    _write_points(runs / "perfect" / "predictions.json", {
        "brackish_a.jpg": [[20.0, 20.0, 0.9], [60.0, 60.0, 0.9]],
        "torsi_a.jpg": [[20.0, 20.0, 0.9]],
        "torsi_empty.jpg": [],
    })
    out = tmp_path / "results"
    assert ev.main(["--data-root", str(bbox_root), "--runs-dir", str(runs),
                    "--out", str(out), "--taus", "0.5:0.5:0.1"]) == 0

    summary = json.loads((out / "point_results_summary.json").read_text())
    run = summary["runs"][0]
    assert run["name"] == "perfect"
    assert run["best_f1"]["f1"] == 1.0
    assert run["best_f1"]["precision"] == 1.0
    assert run["best_f1"]["recall"] == 1.0
    assert run["best_count_mae"]["count_mae"] == 0.0
    assert run["n_unpredicted_images"] == 0
    # Per source: brackish perfect on 1 image, torsi perfect over 2 (one empty).
    sources = run["per_source"]["sources"]
    assert sources["brackish"]["n_images"] == 1 and sources["brackish"]["f1"] == 1.0
    assert sources["torsi"]["n_images"] == 2 and sources["torsi"]["f1"] == 1.0


def test_false_positives_on_the_empty_frame_are_counted(bbox_root, tmp_path):
    """The empty frame is where a detector's false positives must show up."""
    runs = tmp_path / "runs"
    _write_points(runs / "noisy" / "predictions.json", {
        "brackish_a.jpg": [[20.0, 20.0, 0.9], [60.0, 60.0, 0.9]],
        "torsi_a.jpg": [[20.0, 20.0, 0.9]],
        "torsi_empty.jpg": [[5.0, 5.0, 0.9], [7.0, 7.0, 0.9]],
    })
    out = tmp_path / "results"
    ev.main(["--data-root", str(bbox_root), "--runs-dir", str(runs),
             "--out", str(out), "--taus", "0.5:0.5:0.1"])
    run = json.loads((out / "point_results_summary.json").read_text())["runs"][0]
    # 3 TP, 2 FP, 0 FN -> P = 3/5, R = 1.0
    assert run["best_f1"]["tp"] == 3
    assert run["best_f1"]["fp"] == 2
    assert run["best_f1"]["fn"] == 0
    assert run["best_f1"]["precision"] == pytest.approx(0.6)
    assert run["best_f1"]["recall"] == 1.0
    # Count error lands on the empty frame alone: |2-0| over 3 images.
    assert run["best_count_mae"]["count_mae"] == pytest.approx(2 / 3)


def test_an_image_missing_from_the_predictions_scores_as_zero_predictions(
    bbox_root, tmp_path
):
    """A run that simply skipped a frame must not be flattered for it."""
    runs = tmp_path / "runs"
    _write_points(runs / "partial" / "predictions.json", {
        "brackish_a.jpg": [[20.0, 20.0, 0.9], [60.0, 60.0, 0.9]],
    })
    out = tmp_path / "results"
    ev.main(["--data-root", str(bbox_root), "--runs-dir", str(runs),
             "--out", str(out), "--taus", "0.5:0.5:0.1"])
    run = json.loads((out / "point_results_summary.json").read_text())["runs"][0]
    assert run["n_unpredicted_images"] == 2
    assert run["best_f1"]["fn"] == 1, "torsi_a's box is a miss, not an absence"
    assert run["best_f1"]["n_images"] == 3


def test_the_threshold_sweep_moves_the_operating_point(bbox_root, tmp_path, capsys):
    """A low-confidence false positive is removed by raising tau."""
    runs = tmp_path / "runs"
    _write_points(runs / "sweepable" / "predictions.json", {
        "brackish_a.jpg": [[20.0, 20.0, 0.9], [60.0, 60.0, 0.9]],
        "torsi_a.jpg": [[20.0, 20.0, 0.9]],
        "torsi_empty.jpg": [[5.0, 5.0, 0.2]],
    })
    out = tmp_path / "results"
    ev.main(["--data-root", str(bbox_root), "--runs-dir", str(runs),
             "--out", str(out), "--taus", "0.1:0.5:0.2"])
    run = json.loads((out / "point_results_summary.json").read_text())["runs"][0]
    low = next(r for r in run["sweep"] if r["conf_thr"] == pytest.approx(0.1))
    high = next(r for r in run["sweep"] if r["conf_thr"] == pytest.approx(0.5))
    assert low["fp"] == 1 and high["fp"] == 0
    assert low["f1"] < high["f1"], "the false positive must cost F1 at tau 0.1"
    # best_by takes the FIRST maximum, so the chosen operating point is the
    # LOWEST threshold that reaches the best F1 — 0.3, not 0.5. That is the
    # right tie-break: it keeps the most recall for the same F1.
    assert run["best_f1"]["conf_thr"] == pytest.approx(0.3)
    assert run["best_f1"]["f1"] == 1.0


def test_an_empty_name_join_is_reported_not_scored(bbox_root, tmp_path, capsys):
    """Pointed at the wrong root, the scorer must say so, not report zeros."""
    runs = tmp_path / "runs"
    _write_points(runs / "mismatched" / "predictions.json", {
        "1.jpg": [[20.0, 20.0, 0.9]], "2.jpg": [],
    })
    out = tmp_path / "results"
    ev.main(["--data-root", str(bbox_root), "--runs-dir", str(runs),
             "--out", str(out), "--taus", "0.5:0.5:0.1"])
    assert "SKIP" in capsys.readouterr().out
    assert json.loads((out / "point_results_summary.json").read_text())["runs"] == []


def test_nothing_to_score_is_a_non_zero_exit(bbox_root, tmp_path):
    assert ev.main(["--data-root", str(bbox_root), "--runs-dir", str(tmp_path / "empty"),
                    "--out", str(tmp_path / "results")]) == 1
