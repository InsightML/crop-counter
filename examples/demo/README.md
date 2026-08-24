# examples/data — sample dataset

SLIMERS project field imagery, autumn 2024, anonymised; published with permission of the
BOFIN SLIMERS programme.

20 wheat-field images (14 train / 6 val) with point annotations, laid out for
`cropcounter.load_splits`:

```
examples/data/
├── train/
│   ├── annotations.xml
│   └── images/           # q0001.jpg … q0014.jpg
└── val/
    ├── annotations.xml
    └── images/           # q0015.jpg … q0020.jpg
```

Point density is mixed on purpose — both splits span sparse (1 point) to dense (114 train /
89 val) images, so a quick run over this folder exercises the tile sampler and the padded
val path at realistic extremes, not just the median case.

```python
from cropcounter import load_splits

train_records, val_records = load_splits("examples/data")
```

## Anonymisation

The source export's filenames and CVAT metadata identify individual farmers (name embedded
in the image filename, plus task/assignee/source fields and a free-text "Farmer Notes"
attribute in the full annotation set). None of that survives here:

- Images are renamed `q0001.jpg` … `q0020.jpg` and re-encoded (re-decoded and re-saved),
  which drops any embedded EXIF/metadata along with the original filename.
- Each split's `annotations.xml` is rewritten from scratch and carries only `<image name=
  width= height=>` elements containing `<points label= points=>` for `Wheat` and `Volunteer`
  — the two labels `cropcounter` counts. Everything else in the source export (task/job
  metadata, `WheatLine` polylines, the `Remove` and `has farmer notes` tags, and the
  per-point `Confidence` / `Label Status` / `Farmer Notes` attributes) is dropped.

See the main README's "CVAT for images 1.1" section for the full annotation schema this is
a minimal instance of.

## Using your own data

Point `load_splits` at any folder with the same `{train,val}/{images/,annotations.xml}`
layout — export your own CVAT-for-images-1.1 project (or COCO keypoints, via `fmt="coco"`)
into that shape and it works the same way:

```python
train_records, val_records = load_splits("/path/to/your/dataset")
```

`load_records("/path/to/one/split")` reads a single flat `annotations.xml` + `images/`
folder if you don't need a train/val split at all.
