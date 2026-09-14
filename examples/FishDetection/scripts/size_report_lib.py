"""Pure helpers for ``size_report.py`` — no I/O side effects, no GPU, no network.

Everything here is either arithmetic on things already on disk or a renderer.
It lives beside the script rather than inside it so the statistics that decide a
trunk size can be unit-tested without running the report.

Three pieces carry the weight:

* :func:`resample_evalimgs` — the image-level bootstrap's indexer. ``COCOeval``
  stores ``evalImgs`` as a flat list over ``(category, area range, image)``
  built by ``[f(i, k, a) for k in catIds for a in areaRng for i in imgIds]``,
  so the entry for ``(k, a, i)`` sits at ``k*A*I + a*I + i``. Resampling images
  means taking *the same* image position out of every ``(k, a)`` block, with
  multiplicity — a duplicated image duplicates its detections *and* its
  ``gtIgnore`` row, which is exactly what a bootstrap replicate means.
* :func:`accumulate_ap` — feeds a resampled list back through the real
  ``COCOeval.accumulate``. ``evaluate()`` is never re-run; that is the whole
  point (it is the expensive half, and it is resample-independent).
* :func:`paired_test` — Wilcoxon signed-rank when SciPy is importable, an exact
  sign test when it is not, and in both cases the median paired difference and
  the win count beside the p-value, because 17 sources cannot support a
  p-value read on its own.

``accumulate_ap(..., reduced=True)`` passes ``accumulate`` a params object
restricted to the ``areaRng='all'`` / ``maxDets=100`` slice — the exact slice
``summarize()`` reads for ``stats[0]`` (AP) and ``stats[1]`` (AP50). It is ~4x
cheaper than the full accumulate and provably identical on those two numbers;
``tests/test_size_report.py`` asserts that identity against ``summarize()``.
"""
from __future__ import annotations

import base64
import copy
import contextlib
import io
import math
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: What a missing number prints as. Never a zero, never a guess.
MISSING = "—"

#: Verbatim, non-negotiable. These three lines appear in every rendering.
HONESTY_LINES = (
    "Baseline contamination favours RF-DETR: it was trained on the full CFD train split; "
    "our val frames are CFD's published is_train=false frames, but the clip-level split is "
    "unpublished.",
    "One seed per size: differences inside the bootstrap CI are evaluation noise; differences "
    "outside it are still single-seed reads.",
    "Trainable parameters and total FLOPs are both reported; a frozen trunk saves training "
    "compute, never inference compute.",
)

#: Trunk sizes the repo builds, smallest first — the Pareto x-order.
TRUNK_ORDER = ("tiny", "small", "base", "large")


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #


def is_missing(value: Any) -> bool:
    """True for ``None`` and for any float NaN — the two shapes of "not on disk"."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def fmt(value: Any, spec: str = "{:.3f}") -> str:
    """Format a number, or :data:`MISSING` if it is absent.

    Absent means absent: a value that is not on disk is never silently replaced
    by 0, by a default, or by the value of a neighbouring column.
    """
    if is_missing(value):
        return MISSING
    if isinstance(value, str):
        return value
    try:
        return spec.format(value)
    except (TypeError, ValueError):
        return str(value)


def fmt_millions(value: Any) -> str:
    """Parameter counts in millions, two decimals — ``3.58`` rather than ``3580224``."""
    if is_missing(value):
        return MISSING
    return f"{float(value) / 1e6:.2f}"


# --------------------------------------------------------------------------- #
# COCOeval bootstrap
# --------------------------------------------------------------------------- #


def compact_evalimgs(evalimgs: Sequence[Optional[dict]]) -> List[Optional[dict]]:
    """Trim ``evalImgs`` to the four fields ``accumulate`` actually reads.

    ``evaluateImg`` returns eleven keys per entry; ``accumulate`` touches
    ``dtScores``, ``dtMatches``, ``dtIgnore`` and ``gtIgnore`` and nothing else.
    Dropping the rest — and storing the two match masks as ``bool`` rather than
    ``float64`` — takes a 36k-image val split from ~1.2 GB to ~0.4 GB, which is
    what makes shipping the list to worker processes affordable.

    The ``bool`` cast is exact, not an approximation: ``accumulate`` only ever
    asks ``dtMatches`` for its truthiness (``np.logical_and(dtm, ...)``), and a
    match is recorded as a ground-truth id, which is falsy only when it is
    absent — the same convention ``pycocotools`` itself relies on.
    """
    out: List[Optional[dict]] = []
    for entry in evalimgs:
        if entry is None:
            out.append(None)
            continue
        out.append({
            "dtScores": np.asarray(entry["dtScores"], dtype=np.float64),
            "dtMatches": np.asarray(entry["dtMatches"]).astype(bool),
            "dtIgnore": np.asarray(entry["dtIgnore"]).astype(bool),
            "gtIgnore": np.asarray(entry["gtIgnore"]),
        })
    return out


def resample_evalimgs(
    evalimgs: Sequence[Optional[dict]],
    selection: Sequence[int],
    n_cats: int,
    n_areas: int,
    n_images: int,
) -> List[Optional[dict]]:
    """Rebuild a flat ``evalImgs`` list over a resampled set of image positions.

    Args:
        evalimgs: the flat list ``COCOeval.evaluate()`` produced, length
            ``n_cats * n_areas * n_images``.
        selection: image positions (indices into ``_paramsEval.imgIds``), with
            replacement. Repeats are honoured — an image drawn twice appears
            twice in every ``(category, area range)`` block, which is what gives
            it double weight in the accumulated PR curve.
        n_cats, n_areas, n_images: the three axis lengths of the original list.

    Returns:
        A new flat list of length ``n_cats * n_areas * len(selection)``, laid
        out in the same ``(category, area range, image)`` order so
        ``accumulate`` can index it with ``I0 = len(selection)``.

    Every area range is kept. Dropping the three non-``all`` blocks would shift
    every subsequent index and silently score the wrong slice.
    """
    expected = n_cats * n_areas * n_images
    if len(evalimgs) != expected:
        raise ValueError(
            f"evalImgs has {len(evalimgs)} entries, expected {expected} "
            f"= {n_cats} cats x {n_areas} area ranges x {n_images} images"
        )
    selection = list(selection)
    for index in selection:
        if not 0 <= index < n_images:
            raise IndexError(f"image position {index} outside 0..{n_images - 1}")
    out: List[Optional[dict]] = []
    for cat in range(n_cats):
        cat_base = cat * n_areas * n_images
        for area in range(n_areas):
            base = cat_base + area * n_images
            out.extend(evalimgs[base + index] for index in selection)
    return out


def _mean_valid(precision: np.ndarray) -> float:
    """``_summarize``'s own reduction: mean over the entries that are not -1."""
    valid = precision[precision > -1]
    return float(valid.mean()) if valid.size else float("nan")


def _shell(params, evalimgs: Sequence[Optional[dict]], img_ids: Sequence[Any]):
    """A ``COCOeval`` carrying nothing but the state ``accumulate`` needs.

    No ``COCO`` objects, no annotations, no detections — ``accumulate`` reads
    ``self.evalImgs``, ``self.params`` and ``self._paramsEval`` and nothing
    else, so a shell is enough and is cheap to build per resample.
    """
    from pycocotools.cocoeval import COCOeval

    evaluator = COCOeval(iouType="bbox")
    own = copy.copy(params)
    own.imgIds = list(img_ids)
    evaluated = copy.copy(params)
    evaluated.imgIds = list(img_ids)
    evaluator.params = own
    evaluator._paramsEval = evaluated
    evaluator.evalImgs = list(evalimgs)
    return evaluator


def accumulate_ap(
    params,
    evalimgs: Sequence[Optional[dict]],
    img_ids: Sequence[Any],
    reduced: bool = True,
) -> Tuple[float, float]:
    """``(AP, AP50)`` for one already-evaluated (possibly resampled) image set.

    Args:
        params: the ``Params`` ``evaluate()`` left in ``_paramsEval`` (its
            ``imgIds`` is replaced here; everything else is reused as-is).
        evalimgs: a flat ``evalImgs`` list, already resampled if wanted.
        img_ids: the image ids in the same order as ``evalimgs``' image axis.
            Duplicates are required for a resample — ``accumulate`` derives
            ``I0`` from this list's length.
        reduced: restrict the accumulation to the ``areaRng='all'`` /
            ``maxDets=100`` slice, i.e. exactly what ``stats[0]`` and
            ``stats[1]`` read. ``False`` runs the full 4x3 grid and reads
            ``summarize()``'s ``stats`` instead — identical on these two
            numbers, ~4x slower, and the reference the tests check against.
    """
    evaluator = _shell(params, evalimgs, img_ids)
    with contextlib.redirect_stdout(io.StringIO()):
        if reduced:
            slim = copy.copy(evaluator.params)
            slim.areaRng = [evaluator.params.areaRng[0]]
            slim.areaRngLbl = [evaluator.params.areaRngLbl[0]]
            slim.maxDets = [evaluator.params.maxDets[-1]]
            evaluator.accumulate(slim)
            precision = evaluator.eval["precision"]
            return _mean_valid(precision), _mean_valid(precision[0])
        evaluator.accumulate()
        evaluator.summarize()
    return float(evaluator.stats[0]), float(evaluator.stats[1])


def resample_indices(n_images: int, seed: int, replicate: int) -> np.ndarray:
    """The image positions for one bootstrap replicate.

    Derived from ``(seed, replicate)`` alone, so the same ``--seed`` gives the
    same resamples at any ``--workers``, and every system is scored on the
    *same* replicates (common random numbers — it makes the per-system CIs
    comparable rather than merely individually correct).
    """
    rng = np.random.default_rng([int(seed), int(replicate)])
    return rng.integers(0, n_images, size=n_images)


# worker-process state: set once per pool by the initialiser, read by every task.
_PAYLOAD: Optional[dict] = None


def _init_worker(path: str) -> None:
    global _PAYLOAD
    with open(path, "rb") as handle:
        _PAYLOAD = pickle.load(handle)


def _boot_chunk(task: Tuple[int, Sequence[int]]) -> List[Tuple[int, float, float]]:
    """Score a chunk of bootstrap replicates against the worker's payload."""
    seed, replicates = task
    payload = _PAYLOAD
    if payload is None:  # pragma: no cover - guarded by the caller
        raise RuntimeError("bootstrap worker has no payload")
    out = []
    for replicate in replicates:
        selection = resample_indices(payload["n_images"], seed, replicate)
        resampled = resample_evalimgs(
            payload["evalimgs"], selection,
            payload["n_cats"], payload["n_areas"], payload["n_images"],
        )
        ids = [payload["img_ids"][i] for i in selection]
        ap, ap50 = accumulate_ap(payload["params"], resampled, ids, reduced=True)
        out.append((int(replicate), ap, ap50))
    return out


def _chunks(total: int, n_chunks: int) -> List[List[int]]:
    n_chunks = max(1, min(n_chunks, total))
    bounds = np.linspace(0, total, n_chunks + 1).astype(int)
    return [list(range(bounds[i], bounds[i + 1])) for i in range(n_chunks) if bounds[i + 1] > bounds[i]]


def bootstrap_ap(
    payload: dict,
    n_boot: int,
    seed: int = 0,
    workers: int = 1,
) -> Dict[str, Any]:
    """Image-level bootstrap CIs on AP and AP50 for one already-evaluated system.

    ``COCOeval.evaluate()`` runs **once**, before this is called; each replicate
    only re-indexes its output and re-accumulates. Resamples are spread over
    ``workers`` processes, which receive the payload once via a temporary pickle
    (``spawn`` is used rather than ``fork`` so the same code path works on macOS
    and beside a threaded BLAS).

    Returns the 2.5/97.5 percentiles, the point estimate over the unresampled
    set, and the raw replicate arrays.
    """
    global _PAYLOAD

    if n_boot <= 0:
        return {"n_boot": 0, "ap": [], "ap50": []}

    replicates = list(range(n_boot))
    if workers <= 1:
        previous = _PAYLOAD
        _PAYLOAD = payload
        try:
            results = _boot_chunk((seed, replicates))
        finally:
            _PAYLOAD = previous
    else:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context

        results = []
        handle = tempfile.NamedTemporaryFile(prefix="size_report_boot_", suffix=".pkl", delete=False)
        try:
            with handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            tasks = [(seed, chunk) for chunk in _chunks(n_boot, workers * 4)]
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=get_context("spawn"),
                initializer=_init_worker,
                initargs=(handle.name,),
            ) as pool:
                for part in pool.map(_boot_chunk, tasks):
                    results.extend(part)
        finally:
            Path(handle.name).unlink(missing_ok=True)

    results.sort()
    ap = np.array([r[1] for r in results], dtype=float)
    ap50 = np.array([r[2] for r in results], dtype=float)
    return {
        "n_boot": int(n_boot),
        "seed": int(seed),
        "ap": ap.tolist(),
        "ap50": ap50.tolist(),
        "ap_ci": percentile_ci(ap),
        "ap50_ci": percentile_ci(ap50),
    }


def percentile_ci(values: Sequence[float], lo: float = 2.5, hi: float = 97.5) -> List[Optional[float]]:
    """``[2.5th, 97.5th]`` percentile of a replicate array, NaNs dropped."""
    array = np.asarray(values, dtype=float)
    array = array[~np.isnan(array)]
    if array.size == 0:
        return [None, None]
    return [float(np.percentile(array, lo)), float(np.percentile(array, hi))]


# --------------------------------------------------------------------------- #
# paired statistics
# --------------------------------------------------------------------------- #


def _sign_test_p(n_positive: int, n_nonzero: int) -> float:
    """Two-sided exact sign test — the SciPy-free fallback.

    ``P(X <= k)`` and ``P(X >= k)`` under ``X ~ Binomial(n, 0.5)``, doubled and
    clipped at 1. Exact, not normal-approximated: with 17 sources the normal
    approximation is the wrong tool.
    """
    if n_nonzero == 0:
        return float("nan")
    total = 2.0 ** n_nonzero
    lower = sum(math.comb(n_nonzero, i) for i in range(0, n_positive + 1)) / total
    upper = sum(math.comb(n_nonzero, i) for i in range(n_positive, n_nonzero + 1)) / total
    return float(min(1.0, 2.0 * min(lower, upper)))


def paired_test(
    a: Sequence[float],
    b: Sequence[float],
    unit: str = "source",
) -> Dict[str, Any]:
    """Paired comparison of two systems over matched units.

    Reports the median paired difference and the win count **before** the
    p-value, because over 17 sources a p-value is the least informative of the
    three. Pairs where either side is NaN are dropped and counted.

    Args:
        a, b: matched measurements (``a - b`` is the reported difference).
        unit: what one pair is — "source" or "image"; echoed into the result.
    """
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.shape != right.shape:
        raise ValueError(f"paired arrays differ in shape: {left.shape} vs {right.shape}")
    finite = ~(np.isnan(left) | np.isnan(right))
    n_dropped = int((~finite).sum())
    left, right = left[finite], right[finite]
    diff = left - right
    nonzero = diff[diff != 0]
    n_won = int((diff > 0).sum())
    n_lost = int((diff < 0).sum())

    test, statistic, p_value = "none", None, float("nan")
    if nonzero.size:
        try:
            from scipy.stats import wilcoxon

            result = wilcoxon(nonzero)
            test = "wilcoxon-signed-rank"
            statistic, p_value = float(result.statistic), float(result.pvalue)
        except ImportError:
            test = "exact-sign-test"
            p_value = _sign_test_p(int((nonzero > 0).sum()), int(nonzero.size))
    return {
        "unit": unit,
        "n_pairs": int(diff.size),
        "n_dropped": n_dropped,
        "n_nonzero": int(nonzero.size),
        "n_won": n_won,
        "n_lost": n_lost,
        "n_tied": int(diff.size - n_won - n_lost),
        "mean_diff": float(diff.mean()) if diff.size else float("nan"),
        "median_diff": float(np.median(diff)) if diff.size else float("nan"),
        "median_diff_nonzero": float(np.median(nonzero)) if nonzero.size else float("nan"),
        "test": test,
        "statistic": statistic,
        "p_value": p_value,
    }


def bootstrap_mean_diff(
    diff: Sequence[float],
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Percentile CI of the mean of a paired difference vector (resampling pairs)."""
    values = np.asarray(diff, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0 or n_boot <= 0:
        return {"mean": float("nan"), "ci": [None, None], "n_boot": int(max(0, n_boot))}
    rng = np.random.default_rng([int(seed), 0xB007])
    # Chunked so a 36k-image paired vector never allocates an (n_boot, n) int64
    # draw matrix in one go.
    per_chunk = max(1, min(int(n_boot), max(1, 4_000_000 // max(1, values.size))))
    means: List[float] = []
    remaining = int(n_boot)
    while remaining > 0:
        take = min(per_chunk, remaining)
        draws = rng.integers(0, values.size, size=(take, values.size))
        means.extend(values[draws].mean(axis=1).tolist())
        remaining -= take
    return {"mean": float(values.mean()), "ci": percentile_ci(means), "n_boot": int(n_boot)}


# --------------------------------------------------------------------------- #
# headline table
# --------------------------------------------------------------------------- #

HEADLINE_COLUMNS = (
    "System", "Trunk", "Trainable M", "Total M", "GFLOPs", "FLOPs input",
    "img/s", "ms/img", "Peak GPU MB", "AP", "AP50", "AP75", "AR100",
    "tau", "F1 @tau", "count MAE @tau",
)


def headline_row(system: Dict[str, Any]) -> List[str]:
    """One formatted headline row. Anything absent prints :data:`MISSING`."""
    return [
        str(system.get("label") or system.get("name") or MISSING),
        fmt(system.get("trunk") or None, "{}"),
        fmt_millions(system.get("params_trainable")),
        fmt_millions(system.get("params_total")),
        fmt(system.get("gflops"), "{:.1f}"),
        fmt(system.get("gflops_input") or None, "{}"),
        fmt(system.get("images_per_s"), "{:.2f}"),
        fmt(system.get("ms_per_image"), "{:.1f}"),
        fmt(system.get("peak_gpu_mem_mb"), "{:.0f}"),
        fmt(system.get("ap"), "{:.4f}"),
        fmt(system.get("ap50"), "{:.4f}"),
        fmt(system.get("ap75"), "{:.4f}"),
        fmt(system.get("ar100"), "{:.4f}"),
        fmt(system.get("tau"), "{:.2f}"),
        fmt(system.get("f1_at_tau"), "{:.4f}"),
        fmt(system.get("mae_at_tau"), "{:.3f}"),
    ]


def headline_table(systems: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """``{"columns": [...], "rows": [[...], ...]}`` — every cell already a string."""
    return {
        "columns": list(HEADLINE_COLUMNS),
        "rows": [headline_row(system) for system in systems],
    }


# --------------------------------------------------------------------------- #
# document model + renderers
# --------------------------------------------------------------------------- #


def heading(text: str, level: int = 2) -> dict:
    return {"type": "h", "level": level, "text": text}


def paragraph(text: str) -> dict:
    return {"type": "p", "text": text}


def note(text: str) -> dict:
    return {"type": "note", "text": text}


def table(columns: Sequence[str], rows: Sequence[Sequence[str]], caption: str = "") -> dict:
    return {
        "type": "table",
        "columns": [str(c) for c in columns],
        "rows": [[str(cell) for cell in row] for row in rows],
        "caption": caption,
    }


def figure(path: str, alt: str, caption: str = "") -> dict:
    return {"type": "figure", "path": str(path), "alt": alt, "caption": caption}


def bullets(items: Sequence[str]) -> dict:
    return {"type": "ul", "items": [str(i) for i in items]}


def _md_cell(text: str) -> str:
    return str(text).replace("|", "\\|")


def render_markdown(blocks: Iterable[dict], title: str) -> str:
    """The document model as GitHub-flavoured markdown."""
    out: List[str] = [f"# {title}", ""]
    for block in blocks:
        kind = block["type"]
        if kind == "h":
            out += [f"{'#' * int(block['level'])} {block['text']}", ""]
        elif kind == "p":
            out += [block["text"], ""]
        elif kind == "note":
            out += [f"> {block['text']}", ""]
        elif kind == "ul":
            out += [f"- {item}" for item in block["items"]] + [""]
        elif kind == "pre":
            out += ["```", block["text"], "```", ""]
        elif kind == "table":
            out.append("| " + " | ".join(_md_cell(c) for c in block["columns"]) + " |")
            out.append("|" + "---|" * len(block["columns"]))
            for row in block["rows"]:
                out.append("| " + " | ".join(_md_cell(c) for c in row) + " |")
            out.append("")
            if block.get("caption"):
                out += [f"*{block['caption']}*", ""]
        elif kind == "figure":
            out += [f"![{block['alt']}]({block['path']})", ""]
            if block.get("caption"):
                out += [f"*{block['caption']}*", ""]
    return "\n".join(out).rstrip() + "\n"


def _escape(text: str) -> str:
    return (
        str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


#: Inline stylesheet. Light palette on bare :root, dark redefined under both the
#: media query and an explicit attribute, so neither theme borrows the host's.
_CSS = """
:root {
  --bg: #fbfaf8; --fg: #1b1b1f; --muted: #5d5f66; --rule: #e2e0db;
  --panel: #ffffff; --accent: #1f6f6b; --accent2: #b4532a; --warn: #8a5a00;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #16181c; --fg: #e8e6e1; --muted: #a0a3aa; --rule: #2e3138;
    --panel: #1e2126; --accent: #58c3bc; --accent2: #f0895c; --warn: #d9a43c;
  }
}
:root[data-theme="dark"] {
  --bg: #16181c; --fg: #e8e6e1; --muted: #a0a3aa; --rule: #2e3138;
  --panel: #1e2126; --accent: #58c3bc; --accent2: #f0895c; --warn: #d9a43c;
}
html { color-scheme: light dark; }
body {
  background: var(--bg); color: var(--fg); margin: 0;
  font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.wrap { max-width: 1180px; margin: 0 auto; padding-block: 28px; padding-left: 20px; padding-right: 20px; }
h1 { font-size: 1.72rem; line-height: 1.2; margin: 0 0 6px; letter-spacing: -0.01em; }
h2 { font-size: 1.2rem; margin: 34px 0 10px; padding-bottom: 6px; border-bottom: 1px solid var(--rule); }
h3 { font-size: 1.02rem; margin: 22px 0 8px; color: var(--muted); text-transform: uppercase;
     letter-spacing: 0.06em; }
p { margin: 0 0 12px; }
ul { margin: 0 0 14px; padding-left: 20px; }
li { margin-bottom: 5px; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.86em; }
pre { background: var(--panel); border: 1px solid var(--rule); border-radius: 8px;
      padding: 12px; overflow-x: auto; }
.note { border-left: 3px solid var(--accent); background: var(--panel); padding: 10px 14px;
        margin: 0 0 14px; border-radius: 0 8px 8px 0; color: var(--fg); }
.honesty { border-left-color: var(--warn); }
.tw { overflow-x: auto; margin: 0 0 8px; border: 1px solid var(--rule); border-radius: 8px;
      background: var(--panel); }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { text-align: right; padding: 7px 11px; white-space: nowrap;
         border-bottom: 1px solid var(--rule); }
th:first-child, td:first-child { text-align: left; white-space: normal; min-width: 150px; }
thead th { position: sticky; top: 0; background: var(--panel); font-weight: 600;
           color: var(--muted); font-size: 0.8rem; text-transform: uppercase;
           letter-spacing: 0.04em; }
tbody tr:last-child td { border-bottom: none; }
figure { margin: 0 0 20px; }
figure img { display: block; width: 100%; height: auto; background: #ffffff;
             border: 1px solid var(--rule); border-radius: 8px; }
figcaption, .caption { color: var(--muted); font-size: 0.84rem; margin-top: 6px; }
.meta { color: var(--muted); font-size: 0.86rem; margin-bottom: 22px; }
@media (max-width: 640px) {
  .wrap { padding-block: 18px; padding-left: 16px; padding-right: 16px; }
  h1 { font-size: 1.4rem; }
  th, td { padding: 6px 8px; font-size: 0.82rem; }
}
"""


def render_html(blocks: Iterable[dict], title: str, base_dir: Path, subtitle: str = "") -> str:
    """The same document model as ONE self-contained HTML file.

    Figures are inlined as base64 ``data:`` URIs; the stylesheet is inline;
    there is no script and no external reference of any kind. That is asserted
    by ``tests/test_size_report.py``, not merely intended.
    """
    base_dir = Path(base_dir)
    body: List[str] = []
    for block in blocks:
        kind = block["type"]
        if kind == "h":
            level = min(6, max(2, int(block["level"])))
            body.append(f"<h{level}>{_escape(block['text'])}</h{level}>")
        elif kind == "p":
            body.append(f"<p>{_escape(block['text'])}</p>")
        elif kind == "note":
            css = "note honesty" if block["text"] in HONESTY_LINES else "note"
            body.append(f'<div class="{css}">{_escape(block["text"])}</div>')
        elif kind == "ul":
            items = "".join(f"<li>{_escape(i)}</li>" for i in block["items"])
            body.append(f"<ul>{items}</ul>")
        elif kind == "pre":
            body.append(f"<pre>{_escape(block['text'])}</pre>")
        elif kind == "table":
            head = "".join(f"<th>{_escape(c)}</th>" for c in block["columns"])
            rows = "".join(
                "<tr>" + "".join(f"<td>{_escape(c)}</td>" for c in row) + "</tr>"
                for row in block["rows"]
            )
            body.append(
                f'<div class="tw"><table><thead><tr>{head}</tr></thead>'
                f"<tbody>{rows}</tbody></table></div>"
            )
            if block.get("caption"):
                body.append(f'<p class="caption">{_escape(block["caption"])}</p>')
        elif kind == "figure":
            path = base_dir / block["path"]
            if path.exists():
                data = base64.b64encode(path.read_bytes()).decode("ascii")
                img = f'<img alt="{_escape(block["alt"])}" src="data:image/png;base64,{data}">'
            else:
                img = f'<p class="caption">[missing figure: {_escape(block["path"])}]</p>'
            caption = (
                f"<figcaption>{_escape(block['caption'])}</figcaption>"
                if block.get("caption") else ""
            )
            body.append(f"<figure>{img}{caption}</figure>")
    meta = f'<p class="meta">{_escape(subtitle)}</p>' if subtitle else ""
    # A complete document, not a fragment: without the viewport meta a phone
    # browser lays the page out at 980 px and the phone breakpoint never fires.
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_escape(title)}</title>\n"
        f"<style>{_CSS}</style>\n"
        "</head>\n<body>\n"
        f'<div class="wrap">\n<h1>{_escape(title)}</h1>\n{meta}\n'
        + "\n".join(body)
        + "\n</div>\n</body>\n</html>\n"
    )
