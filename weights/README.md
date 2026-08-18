# `weights/`

Two different kinds of checkpoint live here, and only one of them is in git.

| File | In git? | What it is |
|---|---|---|
| `decoder_best.pt` | **yes** (~13 MB) | the trained **pyramid decoder** — the part this project trains |
| `dinov3_convnext_*_pretrain_lvd1689m-*.pth` | no (gitignored) | Meta's **frozen DINOv3 backbone** — a gated download you fetch yourself |

You need both to run the model. `python -m cropcounter.weights` prints where
the backbone is expected and how to get it if it is missing.

## `decoder_best.pt` — the shipped checkpoint

Decoder state dict (28 tensors, 3.36 M parameters) plus the `TrainConfig` it
was trained with. No backbone weights, no optimiser state, no image data.

### Provenance

| | |
|---|---|
| Run | `20260813_113733` |
| Trained by | InsightML, 13 Aug 2026 |
| Selection | lowest validation loss (epoch 24 of 50) |
| Task | wheat emergence counting from proximal RGB quadrat photos |
| Data | ~1,150 point-annotated field images (CVAT 1.1, labels `Wheat` + `Volunteer`) |
| Split | grower-grouped — 1,031 train / **122 validation** images; no grower appears in both |

### Config

The hyperparameters carried inside the checkpoint, which are also
`examples/config_13ep.json` with `epochs` changed:

| | | | |
|---|---|---|---|
| backbone | `base` | tile | 768 |
| `c_dec` | 192 | tiles per image | 4 |
| output stride | 4 | scale jitter | 0.25 |
| sigma | 2.0 | batch size | 8 |
| `k` | 3 | lr | 1e-3 (2-epoch warmup, then cosine) |
| nms radius | 1.5 | weight decay | 1e-4 |
| `tau` | 0.3 (training-time val only — see below) | grad clip | 1.0 |
| match radius | 24 px | focal alpha / beta | 2.0 / 4.0 |
| epochs | 50 | seed | 42 |

### Validation metrics

122 held-out images. `tau` is a decode-time threshold, not a trained
parameter, so the same weights give different numbers at different `tau` —
always compare swept against swept.

| Checkpoint | Epochs | `tau` | count-MAE | F1 |
|---|---|---|---|---|
| `decoder_best.pt` (shipped) | 50 | swept → 0.35 | **8.59** | **0.729** |
| `decoder_best.pt` (shipped) | 50 | 0.3 (training default) | 13.83 | 0.718 |
| reproduction check `20260818_200723` | 13 | swept → 0.35 | 8.64 | 0.726 |
| reproduction check `20260818_200723` | 13 | 0.3 (training default) | 12.27 | 0.721 |

The **reproduction check** is this package, at this commit, retraining the
same recipe on the same split for 13 of the 50 epochs
(`examples/config_13ep.json`, ~3 h on an M-series MPS device). It is here to
show the packaged pipeline reproduces the original training notebook, not to
argue 13 epochs is equivalent to 50.

Re-measuring the shipped checkpoint on that split with this package gives MAE
8.48 / F1 0.729 at `tau` = 0.35 — the ~0.1 MAE gap to the 8.59 above is
fp32-on-MPS versus bf16-autocast-on-CUDA numerics, not a different model.

### Sanitisation

The checkpoint's stored config was rewritten before it was committed, so a
public artifact carries no trace of the machine it was trained on:

| Field | Before | After |
|---|---|---|
| `data_root` | a Windows-relative path into the training machine's private dataset tree | `data` |
| `val_frac` | `0.15` | *removed* — the splitter it belonged to is gone; splits are filesystem-level now |

The decoder tensors are byte-for-byte the originals; only `payload["config"]`
changed.

## Loading it

```python
import torch
from cropcounter import load_checkpoint

model, cfg = load_checkpoint(
    "weights/decoder_best.pt",
    torch.device("cpu"),          # or "cuda" / "mps"
    weights_dir="weights",        # where the DINOv3 backbone .pth lives
)
model.eval()
```

`weights_dir=` matters: the path stored in the checkpoint is relative to
wherever training ran, so loading from any other working directory needs the
override (or `$CROPCOUNTER_WEIGHTS_DIR`, or a `weights/` folder in the cwd).
Without it you get `BackboneWeightsNotFound` with download instructions.

`notebooks/inference.ipynb` runs this checkpoint over a folder of images and
writes counts, overlays and a CVAT XML export.

## Licence

The decoder checkpoint is released under this repository's licence
(AGPL-3.0-only). The DINOv3 backbone weights are **not** ours to
redistribute — they are covered by Meta's DINOv3 licence, which you accept
when you request the gated download.
