# Community Fish Detection Dataset (CFD) subsetting

`cropcounter.cfd` turns LILA's [Community Fish Detection Dataset][lila] — ~1.9M
images, ~935k boxes, 17 source datasets harmonised into one 1.1 GB COCO json —
into a slice small enough to train on.

```bash
pip install ijson   # the only extra dependency this module needs

python -m cropcounter.cfd manifest --metadata cfd.json.zip --out reports/
python -m cropcounter.cfd subset   --metadata cfd.json.zip --out data/brackish \
    --sources brackish_dataset --train-cap 20000 --val-cap 4000
python -m cropcounter.cfd fetch    --subset data/brackish --max-side 1024
python -m cropcounter.cfd points   --subset data/brackish --out data/brackish_points
```

Every pass streams the master with `ijson` — the document is never loaded whole
— and `--metadata` accepts the published `.json.zip` directly.

## `manifest`

One streaming pass, one row per source, written as `manifest.csv`,
`manifest.json` and `manifest.md` (paste-ready) plus a table on stdout:

| column | meaning |
| :-- | :-- |
| `n_images`, `n_boxes`, `n_empty_images`, `empty_fraction` | an "empty" image is one with no fish box |
| `is_train_true/false/missing` | LILA's own split, never recomputed here |
| `sequences` | distinct sequence keys (see below) |
| `box_px_p5…p95` | percentiles of `sqrt(w*h)` in native pixels |
| `box_rel_p5…p95` | the same, over the image's long side |
| `median_width`, `median_height` | native image size |
| `collision_rate` | **the number to look at** |
| `licence`, `permissive`, `prefix` | per-source licence, and the `file_name` shortname |

`collision_rate` is the fraction of boxes whose integer centre cell at stride 4
is already taken by another box in the same image, measured **after** a virtual
resize to long side 1024 — i.e. what a one-peak-per-cell centre/point head
actually sees at train time. A high rate means the source cannot be learned by
this head shape at this stride without either upsampling the head or dropping
the crowded images.

## `subset`

A second streaming pass writing standard COCO:

```
<out>/
├─ train/annotations.json + images/
├─ val/annotations.json   + images/
├─ download_list.txt        # one file_name per line, both splits
└─ subset_summary.json      # kept vs available per source, caps, seed, filters
```

- **`is_train` is honoured, never recomputed.** LILA built it with location
  information where it had any, so a location sits in train or val but not
  both. Re-splitting would leak it. Images with no `is_train` are excluded and
  counted in `subset_summary.json`.
- **Caps apply per source**, at the sequence level when
  `--group-by-sequence` (the default). Whole sequences are drawn in a seeded
  shuffle until the cap is reached; only the sequence that overshoots is
  trimmed, and it is decimated uniformly rather than truncated, so the kept
  frames still span the whole clip. `--no-group-by-sequence` samples images.
- **`--permissive-only`** drops any source whose licence is ND, NC, unstated or
  unknown. Permissive = CC-BY, CC-BY-SA, CC0, CDLA-Permissive, Apache, MIT.
- Image and annotation ids are copied through **verbatim**, so a subset joins
  back to the master. Note that CFD ids are *strings*, not ints.
- The master's bbox-less `category_id: 0` "empty image" markers are dropped;
  the image record stays, which is how standard COCO says "nothing here".

### Sequence keys

`cfd_sequence` is derived from `original_data_source` by stripping the file
extension and then **one** trailing run of digits, with optional separators and
a `frame`/`f` token: `video_007_frame_000123.jpg` → `video_007`. It is
deliberately conservative — when nothing strips, the key is the stem, so a
still-image source degrades to one sequence per image rather than silently
gluing unrelated images together.

Two exceptions are forced by the real metadata:

- If `original_data_source` is a **video** file (`gt_124.flv`,
  `Clip3.mp4` — `f4k` and `viame_fishtrack` do this), it already names the clip
  and is used whole. Stripping its counter would collapse `gt_124` and `gt_125`
  into one key.
- Roboflow's per-image export hash (`..._jpg.rf.<32 hex>[_valid]`, on
  `brackish_dataset`, `roboflow_fish` and `marine_detect`) is removed first.
  Leaving it on makes every frame its own sequence and grouping a no-op.

Grouping is only as good as the filenames, and it fails in both directions —
always read the `sequences` column against `n_images` before trusting a cap to
be group-aware:

- **No grouping at all** (one sequence per image): `noaa_puget`, `kakadu`
  (bare integers), `fathomnet` (UUIDs), `coralscapes` (Cityscapes-style
  `_leftImg8bit` names) and `torsi` (per-second timestamps). A cap there is
  just uniform image sampling.
- **Over-grouping** (everything collapses into one or two keys):
  `project_natick` gives 1 sequence for 1,072 images and `zebrafish` 2 for
  2,224. Harmless — the single sequence overshoots the cap and is decimated
  uniformly, which is again uniform sampling — but it is not the leakage
  protection it looks like.

`noaa_puget`, `mit_river_herring` and `coralscapes` also carry a `location`
field in the master, which is a better grouping key than the filename.

## `fetch`

Downloads `download_list.txt` into `<out>/{train,val}/images/` (basename only,
the `JPEGImages/` prefix is stripped) with a thread pool, 3 retries and resume.

The long side is resized to `--max-side` **on write, never upscaling**, and
`annotations.json` is rewritten with `width`, `height`, `bbox` and `area`
scaled by the same factor and the factor itself recorded per image as
`cfd_scale`. The published geometry is kept as `annotations.native.json`, and
the rewrite always runs from that file — so re-running at a different
`--max-side` is safe, and a resumed run and a fresh one produce identical
annotations.

`--mirror azure|gcs|aws` picks the blob store. URLs are percent-encoded:
`fishclef` filenames contain `#` (which would otherwise be read as a URL
fragment) and `viame_fishtrack` filenames contain `:`.

It is also a library call, for a notebook that pulls on Colab rather than
locally:

```python
from cropcounter.cfd import fetch_subset
fetch_subset("data/brackish", max_side=1024, workers=32, mirror="gcs")
```

## `points`

CFD is a **bbox** dataset; this package's head is a **point** head. `points`
converts one into the other — box centres become COCO keypoints — so the native
counter can train on a fetched subset:

```bash
python -m cropcounter.cfd points --subset data/brackish --out data/brackish_points
```

```
<out>/
├─ train/annotations.json          # COCO keypoints, int ids
├─ train/images -> <subset>/train/images     # absolute symlink
├─ val/ …
├─ cfd_id_map.json                 # per split: original CFD id -> new int id
└─ points_summary.json             # per split: n_images, n_points, n_empty, …
```

One annotation per box, `keypoints = [x + w/2, y + h/2, 2]` and
`num_keypoints = 1`; `bbox` and `area` ride along untouched (harmless to the
point loader, handy for later box work). `--category` names the keypoint
category (default `fish`).

**Why the ids are renumbered.** `crop_dataset.parse_coco_keypoints` does
`int(image["id"])`, and CFD ids are *strings* — so the subset's own ids raise
`ValueError` on load. Image and annotation ids therefore become consecutive
1-based ints in file order. Nothing is lost: the original id stays on every
image as `cfd_image_id` (and on every annotation as `cfd_annotation_id`), and
`cfd_id_map.json` maps original → int per split. Every other CFD per-image field
(`dataset`, `is_train`, `cfd_sequence`, `cfd_scale`, `cfd_file_name`,
`original_data_source`) is carried through verbatim.

**Why a separate data root**, never a sibling file in `data/brackish/`:
`crop_dataset.resolve_annotations` looks for `annotations.json` *first*, so a
points document written beside the bbox one could never be the file the loader
opens. The points root is a real root with its own `annotations.json`; its
`images/` is a symlink back to the subset, so the pixels are not duplicated
(`--copy-images` makes real copies instead — for a Drive or zip hand-off, where
symlinks do not travel).

**Empty frames are kept** by default, as image records with no annotation. That
is what standard COCO means by "nothing here", and they are the negatives the
heatmap head needs — a Brackish subset with the empties removed teaches the
model that every frame contains a fish. `--drop-empty` removes them, for a
crowd-only ablation; `points_summary.json` reports `n_empty` and
`dropped_empty` either way.

A subset that has not been through `fetch` is **refused** (`ValueError`): its
images carry no `cfd_scale`, so the geometry is still in published pixels and
`file_name` still carries the master's `JPEGImages/` prefix — the points would
not match any file on disk, and there would be no file on disk to match.

Two traps on the training side, neither of which this command can fix for you:

- `TrainConfig.labels` defaults to `("Wheat", "Volunteer")`, so the config must
  set `"labels": ["fish"]` (or the category you passed) or every point is
  silently filtered out and you train on an empty dataset.
- `"augment_profile": "natural"` — underwater footage has an up, so
  `VerticalFlip` and `RandomRotate90` do not belong.

It is also a library call:

```python
from cropcounter.cfd import bbox_centres_to_keypoints, write_points_root

write_points_root("data/brackish", "data/brackish_points")
points_doc, id_map = bbox_centres_to_keypoints(bbox_doc)   # in memory
```

## Cost of a real run

Against the published 47 MB zip (1.1 GB of JSON, 1,903,035 images, 935,049
boxes) on a laptop: `manifest` takes ~22s and peaks at ~1.0 GB RSS — it holds
one packed int per image plus one `(image_id, cell)` tuple per box. `subset`
for a single source is ~11s and ~130 MB, since only that source's images are
buffered; `--sources all` is closer to `manifest`.

## Licences

Encoded in `cfd.LICENCES`, keyed by the master's `dataset` string, from the
LILA page. `marine_detect` mixes Roboflow (CC-BY) and GBIF (CC-BY-NC) parts and
is treated as NC throughout, which is the safe reading. Anything not in the
table reports `unknown` and is therefore never permissive. The table is a
convenience, **not legal advice** — check the source datasets before you
redistribute anything.

[lila]: https://lila.science/datasets/community-fish-detection-dataset/
