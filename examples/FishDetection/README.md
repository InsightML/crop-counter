# examples/FishDetection — the point task on CFD Brackish

`cropcounter`'s **native point head**, trained on fish instead of wheat.

The data is the [Community Fish Detection Dataset][cfd] (CFD, LILA) restricted
to its `brackish_dataset` source — fixed-camera underwater footage from a
Danish brackish-water harbour, 14,674 frames (11,547 train / 3,127 val),
CC-BY-SA-4.0, published at 960×540. CFD ships **boxes**; this package counts
**points**, so box centres are converted to COCO keypoints before training
(see [`docs/cfd.md`](../../docs/cfd.md) → `points`).

Why Brackish, from the `manifest` numbers (it is *not* the biggest permissive
source — `mit_river_herring`, `noaa_puget`, `viame_fishtrack` and `kakadu` are
all larger):

- **0.0% stride-4 centre-cell collisions** — no two boxes in a frame share a
  head cell, so a one-peak-per-cell point head can represent every fish. Compare
  `torsi` at 15.6%, which this head shape simply cannot learn at this stride.
- **60.2% of frames contain no fish.** Real negatives, which a counter needs and
  a crop dataset almost never has.
- Permissive (CC-BY-SA-4.0), one fixed camera, 89 sequences — so the
  sequence-grouped cap is real leakage protection rather than a no-op.
- Small objects: median box 45.7px, 0.048 of the long side.

It is also the source the parked box-head run used, so the point numbers are
comparable with it frame for frame.

## Three commands

```bash
# 1. a COCO bbox subset of one source, honouring LILA's own is_train split
python -m cropcounter.cfd subset --metadata data/cfd/community_fish_detection_dataset.json.zip \
    --out data/brackish --sources brackish_dataset --seed 0 \
    --train-cap 20000 --val-cap 4000

# 2. the pixels (resize on write, annotations rescaled to match)
python -m cropcounter.cfd fetch  --subset data/brackish --max-side 1024

# 3. box centres -> COCO keypoints, in a SEPARATE data root
python -m cropcounter.cfd points --subset data/brackish --out data/brackish_points

python -m cropcounter.train --config examples/FishDetection/config_points_8ep.json
```

Step 3 is not optional and not cosmetic. `parse_coco_keypoints` does
`int(image["id"])` and CFD ids are *strings*, so the bbox subset raises
`ValueError` on load; and a bbox-only document has no `keypoints` key at all, so
even with int ids it would load **silently empty**. The output is a separate
root because `resolve_annotations` picks `annotations.json` first — a points
file written beside the bbox one could never be the file the loader opens. Its
`images/` is a symlink back to the subset, so the pixels are not duplicated.

`--max-side 1024` is a no-op for Brackish (960×540 never upscales); it matters
for the other CFD sources.

## Three traps, all of them silent

| Trap | What happens | The fix |
| :-- | :-- | :-- |
| `labels` defaults to `("Wheat", "Volunteer")` | every `fish` point is filtered out and you train on an empty dataset with a finite, falling loss | `"labels": ["fish"]` |
| `augment_profile` defaults to `"wheat"` | `VerticalFlip` + `RandomRotate90` — fine for nadir crop photos, wrong for underwater footage, which has an up | `"augment_profile": "natural"` |
| `annotation_format` defaults to `"cvat"` | looks for `annotations.xml` and raises | `"annotation_format": "coco"` |

Check the first two landed by printing the dataset's transform (no
`VerticalFlip` / `RandomRotate90`) and the loaded point count (it must equal the
number of boxes in the bbox subset).

## Configs

- **[`config_points_8ep.json`](config_points_8ep.json)** — the real run, written
  for Colab (`/content/data/brackish_points`, checkpoints onto Drive). Mirrors
  the parked box run's recipe exactly, minus every box-only field: frozen `base`
  backbone, `c_dec` 192, stride 4, `sigma` 2.0, 768 tiles at 1 tile/image,
  batch 8, 8 epochs, lr 1e-3 with 2 warmup epochs, seed 0.
- **[`config_points_smoke.json`](config_points_smoke.json)** — the same recipe
  at 1 epoch, `num_workers` 0, local paths. For proving the seams before paying
  for a GPU.

## Smoke recipe

A 36-frame end-to-end run on **real** frames, finishing in well under a minute
on an M-series MPS device. Do this before any cloud run:

```bash
python -m cropcounter.cfd subset --metadata data/cfd/community_fish_detection_dataset.json.zip \
    --out data/brackish_smoke --sources brackish_dataset --seed 0 --train-cap 24 --val-cap 12
python -m cropcounter.cfd fetch  --subset data/brackish_smoke --max-side 1024
python -m cropcounter.cfd points --subset data/brackish_smoke --out data/brackish_points_smoke
python -m cropcounter.train --config examples/FishDetection/config_points_smoke.json --device mps
```

What to check, in this order:

1. `points_summary.json` — `n_points` per split equals the number of `bbox`
   annotations in `data/brackish_smoke/<split>/annotations.json`.
2. The loader agrees: `sum(len(r.points) for r in train_recs)` equals that same
   number, and `sum(1 for r in train_recs if not r.points)` equals
   `n_empty` — the negatives survived.
3. `runs/brackish_points_smoke/` holds `history.json`, `curves.png`, `best.pt`,
   `last.pt`, and its `config.json` shows `labels: ["fish"]` /
   `augment_profile: "natural"`.
4. Train and val loss are finite. One epoch with two warmup epochs decodes
   nothing yet, so P/R/F1 = 0 and count MAE ≈ the mean points-per-image — that
   is the expected shape of a working smoke, not a failure.

`data/` is gitignored, so nothing here is committed; `weights/` must hold the
gated DINOv3 backbone (see the root [README](../../README.md)).

[cfd]: https://lila.science/datasets/community-fish-detection-dataset/
