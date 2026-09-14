"""Unit tests for ``examples/FishDetection/scripts/size_report.py`` and its helper lib.

The load-bearing claim of the report is the image-level bootstrap: that
resampling ``COCOeval.evalImgs`` and re-accumulating gives the *same* number as
re-running the whole evaluation on a val set where the sampled images have been
physically duplicated. That is checked here against brute force rather than
asserted in a docstring — if the flat-list indexing were off by an area range,
these tests would fail and the CIs in every report would be silently wrong.

Nothing here touches the network, a GPU, or a real dataset; the COCO fixture is
three 100x100 images built in memory.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "examples" / "FishDetection" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_spec = importlib.util.spec_from_file_location("size_report", _SCRIPTS / "size_report.py")
size_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(size_report)
lib = size_report.lib


# --------------------------------------------------------------------------- #
# a three-image COCO fixture
# --------------------------------------------------------------------------- #


def _dataset(image_ids, boxes_per_image):
    """A minimal COCO dict: one category, ``image_ids`` images, given GT boxes."""
    images = [{"id": i, "file_name": f"{i}.jpg", "width": 100, "height": 100}
              for i in image_ids]
    annotations = []
    next_id = 1
    for image_id in image_ids:
        for box in boxes_per_image[image_id]:
            annotations.append({
                "id": next_id, "image_id": image_id, "category_id": 1,
                "bbox": [float(v) for v in box], "area": float(box[2] * box[3]),
                "iscrowd": 0,
            })
            next_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "fish"}],
    }


#: GT boxes per image. Deliberately a mix: two objects, one object, one object
#: that nothing is predicted near (so recall misses are in play).
GT_BOXES = {
    "a": [[10, 10, 20, 20], [60, 60, 24, 24]],
    "b": [[30, 30, 18, 18]],
    "c": [[5, 70, 30, 12]],
}
#: Detections per image, every score distinct so no cross-image ordering tie can
#: make the comparison depend on concatenation order.
DETECTIONS = {
    "a": [([11, 11, 19, 19], 0.91), ([62, 59, 22, 26], 0.53), ([80, 10, 10, 10], 0.22)],
    "b": [([31, 29, 17, 19], 0.83), ([70, 70, 12, 12], 0.31)],
    "c": [([40, 40, 10, 10], 0.66)],
}


def _detection_records(image_ids, source_of=None):
    """COCO-results entries for ``image_ids``; ``source_of`` remaps a clone's boxes."""
    source_of = source_of or {}
    out = []
    for image_id in image_ids:
        source = source_of.get(image_id, image_id)
        for box, score in DETECTIONS[source]:
            out.append({
                "image_id": image_id, "category_id": 1,
                "bbox": [float(v) for v in box], "score": float(score),
            })
    return out


def _evaluate(dataset, detections):
    """``COCOeval`` run through ``evaluate`` only — the state the bootstrap reuses."""
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = dataset
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes([dict(d) for d in detections])
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.evaluate()
    return evaluator


# --------------------------------------------------------------------------- #
# 1 — the resampling indexer, against brute force
# --------------------------------------------------------------------------- #


def test_resample_evalimgs_matches_a_physically_duplicated_dataset():
    """AP of the resample ``[a, a, b]`` == AP of a val set holding a twice.

    The reference side builds a genuinely different COCO dataset — image ``a``
    appears under two ids, with its own copies of the annotations and of the
    detections — and scores it with the repo's own ``coco_eval``. If the flat
    ``evalImgs`` list were indexed as ``(image, area)`` instead of
    ``(area, image)``, or if the duplicate were dropped rather than repeated,
    the two sides would not meet.

    Exact equality is expected and asserted: every detection score in the
    fixture is distinct, so the stable sort inside ``accumulate`` has no tie to
    break, and the duplicated image's two entry groups are element-wise
    identical, so their order between themselves cannot matter either.
    """
    from cropcounter.det_metrics import coco_eval

    ids = ["a", "b", "c"]
    evaluator = _evaluate(_dataset(ids, GT_BOXES), _detection_records(ids))
    ordered = list(np.unique(evaluator.params.imgIds))
    assert ordered == ["a", "b", "c"]
    n_images, n_areas = len(ordered), len(evaluator._paramsEval.areaRng)
    n_cats = len(evaluator._paramsEval.catIds)
    assert len(evaluator.evalImgs) == n_cats * n_areas * n_images

    compact = lib.compact_evalimgs(evaluator.evalImgs)
    selection = [0, 0, 1]  # a, a, b
    resampled = lib.resample_evalimgs(compact, selection, n_cats, n_areas, n_images)
    assert len(resampled) == n_cats * n_areas * len(selection)
    ours_ap, ours_ap50 = lib.accumulate_ap(
        evaluator._paramsEval, resampled, [ordered[i] for i in selection], reduced=True,
    )

    # brute force: a real dataset carrying image "a" twice
    clone_ids = ["a", "a_clone", "b"]
    clone_gt = dict(GT_BOXES)
    clone_gt["a_clone"] = GT_BOXES["a"]
    reference = coco_eval(
        _dataset(clone_ids, clone_gt),
        _detection_records(clone_ids, source_of={"a_clone": "a"}),
    )

    assert ours_ap == pytest.approx(reference["ap"], abs=1e-12)
    assert ours_ap50 == pytest.approx(reference["ap50"], abs=1e-12)


def test_resample_of_every_image_once_reproduces_the_unresampled_score():
    """The identity resample is a no-op: selection ``[0..n-1]`` == the plain eval."""
    from cropcounter.det_metrics import coco_eval

    ids = ["a", "b", "c"]
    dataset, detections = _dataset(ids, GT_BOXES), _detection_records(ids)
    evaluator = _evaluate(dataset, detections)
    ordered = list(np.unique(evaluator.params.imgIds))
    compact = lib.compact_evalimgs(evaluator.evalImgs)
    selection = list(range(len(ordered)))
    resampled = lib.resample_evalimgs(
        compact, selection, len(evaluator._paramsEval.catIds),
        len(evaluator._paramsEval.areaRng), len(ordered),
    )
    ap, ap50 = lib.accumulate_ap(evaluator._paramsEval, resampled, ordered, reduced=True)
    reference = coco_eval(dataset, detections)
    assert ap == pytest.approx(reference["ap"], abs=1e-12)
    assert ap50 == pytest.approx(reference["ap50"], abs=1e-12)


def test_reduced_accumulate_equals_full_summarize():
    """``reduced=True`` reads the same slice ``summarize()`` puts in stats[0]/[1]."""
    ids = ["a", "b", "c"]
    evaluator = _evaluate(_dataset(ids, GT_BOXES), _detection_records(ids))
    ordered = list(np.unique(evaluator.params.imgIds))
    compact = lib.compact_evalimgs(evaluator.evalImgs)
    selection = [2, 0, 2, 1]
    resampled = lib.resample_evalimgs(
        compact, selection, len(evaluator._paramsEval.catIds),
        len(evaluator._paramsEval.areaRng), len(ordered),
    )
    ids_resampled = [ordered[i] for i in selection]
    fast = lib.accumulate_ap(evaluator._paramsEval, resampled, ids_resampled, reduced=True)
    full = lib.accumulate_ap(evaluator._paramsEval, resampled, ids_resampled, reduced=False)
    assert fast[0] == pytest.approx(full[0], abs=1e-12)
    assert fast[1] == pytest.approx(full[1], abs=1e-12)


def test_resample_evalimgs_rejects_a_mis_shaped_list_and_a_bad_index():
    entries = [{"x": i} for i in range(2 * 4 * 3)]
    with pytest.raises(ValueError, match="expected"):
        lib.resample_evalimgs(entries, [0], n_cats=1, n_areas=4, n_images=3)
    with pytest.raises(IndexError):
        lib.resample_evalimgs(entries, [3], n_cats=2, n_areas=4, n_images=3)


def test_resample_evalimgs_takes_the_same_image_out_of_every_area_block():
    """Structural check on the layout, independent of pycocotools."""
    n_cats, n_areas, n_images = 2, 4, 3
    entries = [(k, a, i) for k in range(n_cats) for a in range(n_areas) for i in range(n_images)]
    out = lib.resample_evalimgs(entries, [2, 2, 0], n_cats, n_areas, n_images)
    assert len(out) == n_cats * n_areas * 3
    # block (k=1, a=2) must be the entries for images 2, 2, 0 of that same block
    block = out[(1 * n_areas + 2) * 3: (1 * n_areas + 2) * 3 + 3]
    assert block == [(1, 2, 2), (1, 2, 2), (1, 2, 0)]


def test_bootstrap_ap_is_deterministic_and_worker_count_does_not_change_it():
    ids = ["a", "b", "c"]
    evaluator = _evaluate(_dataset(ids, GT_BOXES), _detection_records(ids))
    ordered = list(np.unique(evaluator.params.imgIds))
    payload = {
        "evalimgs": lib.compact_evalimgs(evaluator.evalImgs),
        "params": evaluator._paramsEval,
        "img_ids": ordered,
        "n_images": len(ordered),
        "n_cats": len(evaluator._paramsEval.catIds),
        "n_areas": len(evaluator._paramsEval.areaRng),
    }
    first = lib.bootstrap_ap(payload, n_boot=12, seed=7, workers=1)
    second = lib.bootstrap_ap(payload, n_boot=12, seed=7, workers=1)
    assert first["ap"] == second["ap"]
    assert first["ap_ci"][0] <= first["ap_ci"][1]
    assert len(first["ap50"]) == 12
    # the same replicate index gives the same image selection, every time, and a
    # different one gives a different selection (checked at a width where an
    # accidental collision is not a realistic outcome)
    assert lib.resample_indices(3, 7, 4).tolist() == lib.resample_indices(3, 7, 4).tolist()
    assert lib.resample_indices(200, 7, 4).tolist() != lib.resample_indices(200, 7, 5).tolist()
    assert lib.resample_indices(200, 7, 4).tolist() != lib.resample_indices(200, 8, 4).tolist()
    # every drawn position is a valid image position, and repeats do occur
    draw = lib.resample_indices(200, 0, 0)
    assert draw.min() >= 0 and draw.max() < 200 and len(set(draw.tolist())) < 200


# --------------------------------------------------------------------------- #
# 2 — the paired test helper
# --------------------------------------------------------------------------- #


def test_paired_test_counts_wins_ties_and_the_median_difference():
    a = [0.5, 0.4, 0.3, 0.9, 0.2]
    b = [0.1, 0.4, 0.8, 0.5, 0.2]
    result = lib.paired_test(a, b, unit="source")
    assert result["n_pairs"] == 5
    assert result["n_won"] == 2       # 0.5>0.1, 0.9>0.5
    assert result["n_lost"] == 1      # 0.3<0.8
    assert result["n_tied"] == 2
    assert result["n_nonzero"] == 3
    assert result["median_diff"] == pytest.approx(0.0)
    assert result["mean_diff"] == pytest.approx(np.mean(np.array(a) - np.array(b)))
    assert result["unit"] == "source"
    assert result["test"] == "wilcoxon-signed-rank"


def test_paired_test_drops_nan_pairs_and_says_how_many():
    result = lib.paired_test([1.0, float("nan"), 0.2], [0.5, 0.1, float("nan")])
    assert result["n_dropped"] == 2
    assert result["n_pairs"] == 1
    assert result["n_won"] == 1


def test_paired_test_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="differ in shape"):
        lib.paired_test([1.0, 2.0], [1.0])


def test_paired_test_with_no_differences_reports_no_test():
    result = lib.paired_test([0.3, 0.3], [0.3, 0.3])
    assert result["n_nonzero"] == 0
    assert result["test"] == "none"
    assert np.isnan(result["p_value"])


def test_sign_test_fallback_when_scipy_is_unavailable(monkeypatch):
    """Without SciPy the helper falls back to an EXACT sign test, not a normal one."""
    monkeypatch.setitem(sys.modules, "scipy.stats", None)
    result = lib.paired_test([1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0])
    assert result["test"] == "exact-sign-test"
    # five wins out of five nonzero pairs: 2 * (1/32)
    assert result["p_value"] == pytest.approx(2 * 1 / 32)
    assert result["n_won"] == 5


def test_sign_test_p_values_are_the_exact_binomial_ones():
    assert lib._sign_test_p(5, 5) == pytest.approx(0.0625)
    assert lib._sign_test_p(3, 6) == pytest.approx(1.0)
    assert np.isnan(lib._sign_test_p(0, 0))


def test_bootstrap_mean_diff_brackets_the_mean():
    rng = np.random.default_rng(0)
    diff = rng.normal(0.2, 0.05, size=400)
    result = lib.bootstrap_mean_diff(diff, n_boot=200, seed=3)
    low, high = result["ci"]
    assert low < result["mean"] < high
    assert result["mean"] == pytest.approx(float(diff.mean()))


# --------------------------------------------------------------------------- #
# 3 — headline table assembly with holes in it
# --------------------------------------------------------------------------- #


def test_headline_row_prints_an_em_dash_for_everything_absent():
    """A system with nothing measured must produce dashes, never zeros."""
    row = lib.headline_row({"label": "frozen tiny", "trunk": "tiny"})
    assert row[0] == "frozen tiny"
    assert row[1] == "tiny"
    assert set(row[2:]) == {lib.MISSING}
    assert len(row) == len(lib.HEADLINE_COLUMNS)


def test_headline_row_formats_what_is_present_and_dashes_the_rest():
    row = lib.headline_row({
        "label": "frozen base", "trunk": "base",
        "params_trainable": 3_580_000, "params_total": 91_100_000,
        "gflops": 477.0, "gflops_input": "1024x576",
        "ap": 0.2910, "ap50": 0.6740,
        "tau": 0.35, "f1_at_tau": 0.6979,
        # deliberately absent: total throughput, peak memory, ap75, ar100, count MAE
    })
    columns = list(lib.HEADLINE_COLUMNS)
    cell = dict(zip(columns, row))
    assert cell["Trainable M"] == "3.58"
    assert cell["Total M"] == "91.10"
    assert cell["GFLOPs"] == "477.0"
    assert cell["FLOPs input"] == "1024x576"
    assert cell["AP"] == "0.2910"
    assert cell["tau"] == "0.35"
    assert cell["img/s"] == lib.MISSING
    assert cell["ms/img"] == lib.MISSING
    assert cell["Peak GPU MB"] == lib.MISSING
    assert cell["AP75"] == lib.MISSING
    assert cell["AR100"] == lib.MISSING
    assert cell["count MAE @tau"] == lib.MISSING


def test_headline_row_treats_nan_as_missing_not_as_a_number():
    row = lib.headline_row({"label": "x", "ap": float("nan"), "gflops": float("nan")})
    cell = dict(zip(lib.HEADLINE_COLUMNS, row))
    assert cell["AP"] == lib.MISSING
    assert cell["GFLOPs"] == lib.MISSING


def test_headline_table_rows_line_up_with_the_columns():
    built = lib.headline_table([{"label": "a"}, {"label": "b", "ap": 0.5}])
    assert built["columns"] == list(lib.HEADLINE_COLUMNS)
    assert all(len(row) == len(built["columns"]) for row in built["rows"])


# --------------------------------------------------------------------------- #
# 4 — the HTML is genuinely self-contained
# --------------------------------------------------------------------------- #


def _example_blocks():
    blocks = [lib.paragraph("subtitle")]
    blocks += [lib.note(line) for line in lib.HONESTY_LINES]
    blocks += [
        lib.heading("1 · Headline", 2),
        lib.table(["System", "AP"], [["frozen base", "0.2910"], ["nano", lib.MISSING]], "caption"),
        lib.bullets(["one", "two"]),
        lib.figure("figures/demo.png", "a demo figure", "what it shows"),
        lib.heading("Escaping <b>matters</b> & so do ampersands", 3),
    ]
    return blocks


def _write_png(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, ax = plt.subplots(figsize=(2, 1.2))
    ax.plot([0, 1], [0, 1])
    figure.savefig(path, dpi=60)
    plt.close(figure)


def test_html_has_no_external_reference_of_any_kind(tmp_path):
    _write_png(tmp_path / "figures" / "demo.png")
    html = lib.render_html(_example_blocks(), "Report", tmp_path, subtitle="sub")

    lowered = html.lower()
    for forbidden in ("http://", "https://", "//cdn", "<link", "<script", "@import",
                      "url(", "srcset", "<iframe", "<object", "<embed"):
        assert forbidden not in lowered, f"HTML reaches outside itself: {forbidden!r}"

    # every src= is an inlined data: URI, and the figure really was inlined
    sources = [part.split('"', 1)[0] for part in html.split('src="')[1:]]
    assert sources, "no image was inlined"
    assert all(src.startswith("data:image/png;base64,") for src in sources)
    assert len(sources[0]) > 200


def test_html_carries_the_three_honesty_lines_verbatim(tmp_path):
    html = lib.render_html(_example_blocks(), "Report", tmp_path)
    for line in lib.HONESTY_LINES:
        assert line in html


def test_markdown_carries_the_three_honesty_lines_verbatim():
    markdown = lib.render_markdown(_example_blocks(), "Report")
    for line in lib.HONESTY_LINES:
        assert line in markdown


def test_html_escapes_markup_in_content():
    html = lib.render_html([lib.paragraph("a < b & c > d")], "T", Path("."))
    assert "a &lt; b &amp; c &gt; d" in html
    assert "<b>" not in html


def test_html_is_readable_in_both_themes_and_at_phone_width():
    html = lib.render_html([lib.paragraph("x")], "T", Path("."))
    assert "prefers-color-scheme: dark" in html
    assert '[data-theme="dark"]' in html
    assert "max-width: 640px" in html          # the phone breakpoint
    assert "overflow-x: auto" in html          # wide tables scroll, the page does not
    # a complete document: without the viewport meta the breakpoint never fires
    assert html.startswith("<!doctype html>")
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in html
    assert '<meta charset="utf-8">' in html
    assert html.rstrip().endswith("</html>")


def test_missing_figure_degrades_to_a_note_rather_than_a_broken_link(tmp_path):
    html = lib.render_html([lib.figure("figures/nope.png", "gone")], "T", tmp_path)
    assert "missing figure" in html
    assert "src=" not in html


# --------------------------------------------------------------------------- #
# 5 — the full document, assembled
# --------------------------------------------------------------------------- #


def _minimal_context():
    empty = {"columns": ["a"], "rows": []}
    return {
        "systems": [
            {"name": "frozen_base", "label": "frozen base", "kind": "ours", "trunk": "base",
             "ap": 0.29, "ap50": 0.67, "gflops": 477.0, "gflops_input": "1024x576"},
            {"name": "rfdetr_nano_640", "label": "nano", "kind": "baseline", "trunk": None,
             "ap": 0.49, "ap50": 0.79, "gflops": 97.3, "gflops_input": "640x640"},
        ],
        "figures": {},
        "subtitle": "sub",
        "per_source_table": empty,
        "per_size_table": empty,
        "bootstrap_table": empty,
        "bootstrap_text": "bootstrap text",
        "paired_source_table": {"columns": ["a"], "rows": []},
        "paired_image_table": {"columns": ["a"], "rows": []},
        "best_last_table": {"columns": ["a"], "rows": []},
        "scope_text": "scope text",
        "provenance": ["read something"],
        "missing": ["a thing that is not on disk"],
        "parity": [],
        "flops_side": "1024x576",
    }


def test_build_blocks_emits_the_honesty_lines_and_the_template_recommendation():
    blocks = size_report.build_blocks(_minimal_context())
    markdown = lib.render_markdown(blocks, "Report")
    for line in lib.HONESTY_LINES:
        assert line in markdown
    assert "TEMPLATE" in markdown
    assert "[Freddie's call goes here" in markdown
    # the arithmetic is filled in, the verdict is not
    assert "4.9x the inference FLOPs of nano" in markdown


def test_recommendation_reports_the_gap_as_unquotable_when_flops_are_absent():
    systems = [
        {"name": "frozen_base", "label": "frozen base", "kind": "ours", "trunk": "base",
         "ap": 0.29, "ap50": 0.67},
        {"name": "rfdetr_nano_640", "label": "nano", "kind": "baseline", "ap": 0.49},
    ]
    markdown = lib.render_markdown(
        size_report.recommendation_blocks(systems, "1024x576"), "R")
    assert "cannot be quoted" in markdown
    assert "TEMPLATE" in markdown


# --------------------------------------------------------------------------- #
# 6 — CLI surface
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("spec", "expected"), [
    ("run=Label", ("run", "Label", None)),
    ("run=Label:tiny", ("run", "Label", "tiny")),
    ("run=Frozen base: 8 epochs", ("run", "Frozen base: 8 epochs", None)),
    ("run=Frozen: 8 epochs:base", ("run", "Frozen: 8 epochs", "base")),
    ("run=", ("run", "run", None)),
])
def test_parse_system_spec(spec, expected):
    assert size_report.parse_system_spec(spec) == expected


def test_parse_system_spec_rejects_a_spec_without_a_name():
    with pytest.raises(SystemExit):
        size_report.parse_system_spec("nolabel")
    with pytest.raises(SystemExit):
        size_report.parse_system_spec("=label")


def test_discover_systems_reads_the_trunk_size_out_of_config_json(tmp_path):
    runs = tmp_path / "runs"
    (runs / "r_tiny").mkdir(parents=True)
    (runs / "r_tiny" / "predictions.json").write_text("[]")
    (runs / "r_tiny" / "config.json").write_text(json.dumps({"backbone": "tiny"}))
    (runs / "r_base").mkdir()
    (runs / "r_base" / "predictions.json").write_text("[]")
    (runs / "r_base" / "config.json").write_text(json.dumps({"backbone": "base"}))
    (runs / "no_predictions").mkdir()
    baselines = tmp_path / "results"
    (baselines / "rfdetr_nano_640").mkdir(parents=True)
    (baselines / "rfdetr_nano_640" / "predictions.json").write_text("[]")

    args = size_report.build_parser().parse_args([
        "--data-root", str(tmp_path), "--out-dir", str(tmp_path / "out"),
        "--runs-dir", str(runs), "--baselines-dir", str(baselines),
    ])
    systems = size_report.discover_systems(args)
    assert [s["name"] for s in systems] == ["r_tiny", "r_base", "rfdetr_nano_640"]
    assert [s["trunk"] for s in systems] == ["tiny", "base", None]
    assert [s["kind"] for s in systems] == ["ours", "ours", "baseline"]
    assert systems[0]["summary_key"] == "r_tiny_best"
    assert systems[0]["summary_key_last"] == "r_tiny_last"

    args.systems = ["r_base=Big one:small"]
    picked = size_report.discover_systems(args)
    assert [(s["name"], s["label"], s["trunk"]) for s in picked] == [("r_base", "Big one", "small")]

    args.systems = ["does_not_exist=x"]
    with pytest.raises(SystemExit, match="not found"):
        size_report.discover_systems(args)


def test_per_image_f1_scores_an_agreed_empty_frame_as_one():
    """The convention the report states: nothing there, nothing predicted, F1 = 1."""
    by_image = {"a": [{"bbox": [10, 10, 20, 20], "score": 0.9}]}
    gt_boxes = {"a": [[10, 10, 20, 20]]}
    scores = size_report.per_image_f1(by_image, gt_boxes, ["a", "empty"], tau=0.5, match_iou=0.5)
    assert scores[0] == pytest.approx(1.0)   # matched
    assert scores[1] == pytest.approx(1.0)   # agreed empty
    # raising tau past the detection's score turns the hit into a pure miss
    missed = size_report.per_image_f1(by_image, gt_boxes, ["a"], tau=0.95, match_iou=0.5)
    assert missed[0] == pytest.approx(0.0)


def test_top_k_per_image_keeps_the_highest_scores():
    detections = [{"image_id": "a", "score": s, "bbox": [0, 0, 1, 1]} for s in (0.1, 0.9, 0.5)]
    kept = size_report.top_k_per_image(detections, k=2)
    assert sorted(d["score"] for d in kept["a"]) == [0.5, 0.9]
