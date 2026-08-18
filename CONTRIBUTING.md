# Contributing to cropcounter

## Dev setup

```bash
conda create -n crop-counter python=3.12
conda activate crop-counter
pip install -e ".[portal,dev]"
```

`dev` pulls in `pytest`, `ruff`, and `nbstripout`. `portal` is only needed if
you're touching `cropcounter.portal` or its tests. If you also need the
`fmt="datumaro"` loader, add `[formats]`.

The DINOv3 backbone weights are **not** part of this install — they're a
gated download from Meta. See the README's "DINOv3 backbone weights"
section, or run `python -m cropcounter.weights` for the exact instructions.
You don't need them to run the unit tests below.

## nbstripout hook

The notebooks under `notebooks/` should never carry cell outputs or
execution counts in a commit — they bloat diffs and can leak whatever was in
the output (including real farmer imagery from local runs). Activate the
repo's pre-commit hook once per clone:

```bash
git config core.hooksPath .githooks
```

The hook also needs `nbstripout` **on PATH** (`pip install -e ".[dev]"`
covers this). If it isn't found, the hook prints a warning and lets the
commit through rather than blocking your workflow — so a missing
`nbstripout` is a silent no-op, not a hard stop. Run `nbstripout --verify
notebooks/*.ipynb` yourself before pushing if you're not sure it's active.

## Running tests

```bash
conda run -n crop-counter python -m pytest tests/ -q
```

The test suite is CPU-light and network-free: no DINOv3 backbone, no
dataset, no live portal calls. It should pass in any environment with the
package installed, gated weights or not.

## PR expectations

- Keep unit tests passing; add tests for new behaviour (heatmap/metrics/format
  loaders can be exercised with tiny synthetic fixtures — no real dataset
  needed).
- Run `ruff check .` before opening a PR.
- No real training data, farmer-identifiable imagery, API keys, or `.env`
  files in commits or notebook outputs — this is a public repo.
- Describe what changed and why in the PR body; keep commits scoped and the
  message concise.
