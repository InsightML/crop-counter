# Available Trained Models

Trained **pyramid decoders** for tasks beyond the one shipped in git, hosted
off-repo on Google Drive. Each Drive folder holds one or more **zipped runs**,
and every run contains the same four things:

- the `TrainConfig` it was trained with,
- the training history (curves / metrics),
- `best.pt` lowest validation loss,
- `last.pt` the final epoch.

These are decoder checkpoints only, the same kind of artifact as
[`weights/decoder_best.pt`](../decoder_best.pt). They carry **no** DINOv3
backbone weights; you still fetch Meta's frozen backbone yourself (see
[`weights/README.md`](../README.md), which also covers loading, the
`weights_dir=` override, and sanitisation).

## Models

| Task | Dataset | Source | Tutorial | Weights |
|---|---|---|---|---|
| Wheat emergence | private | --- | [`examples/demo`](../../examples/demo) | [Drive](https://drive.google.com/drive/folders/1CBlMdOc7Kmh_ZUbLIOWIGVCnSEgGxqpD?usp=sharing) |
| Wheat head | GWHD_2021 | [Zenodo](https://zenodo.org/records/5092309) | [`examples/WheatHead`](../../examples/WheatHead) | [Drive](https://drive.google.com/drive/folders/1r2IfbiIKhwAp4d07RgrCw6UTsWPAKRxI?usp=sharing) |
| Maize tassel | MTC-UAV | [GitHub](https://github.com/poppinace/mtc-uav) | [`examples/MaizeTassel`](../../examples/MaizeTassel) | [Drive](https://drive.google.com/drive/folders/1ErISwSfwszWipR8RjeK6HBA9ML0JKdCU?usp=sharing) |

---

### Wheat emergence

Counting emerged wheat plants from proximal RGB quadrat photos. This is the
task whose best decoder is also the one committed to git as
[`weights/decoder_best.pt`](../decoder_best.pt) — see
[`weights/README.md`](../README.md) for its full provenance, config, metrics,
and validation numbers. The Drive folder additionally provides the packaged
run(s) with `decoder_last.pt` and training history.

- **Data:** private (~1,150 point-annotated field images, CVAT 1.1).
- **Tutorial:** [`examples/demo`](../../examples/demo) plus
  `notebooks/inference.ipynb`.

### Wheat head GWHD_2021

Detecting wheat heads on the Global Wheat Head Dataset 2021 (>6,000 images from
11 countries, >300,000 labelled heads).

- **Data:** [Zenodo record 5092309](https://zenodo.org/records/5092309); reformat
  to points with `examples/WheatHead/notebooks/1_reformat.ipynb`.
- **Tutorial / report:**
  [`examples/WheatHead/notebooks`](../../examples/WheatHead/notebooks) and its
  [`docs/report.md`](../../examples/WheatHead/notebooks/docs/report.md).
- Includes the `Base` backbone model trained in the tutorial. Additionally includes a `Tiny` backbone model.

### Maize tassel MTC-UAV

Counting maize tassels in the MTC-UAV drone dataset (306 images at
5472×3648 px, ~70,870 annotated tassels).

- **Data:** [poppinace/mtc-uav](https://github.com/poppinace/mtc-uav); reformat
  with `examples/MaizeTassel/notebooks/1_reformat.ipynb`.
- **Tutorial / report:**
  [`examples/MaizeTassel/notebooks`](../../examples/MaizeTassel/notebooks) and its
  [`docs/report.md`](../../examples/MaizeTassel/notebooks/docs/report.md).
- Includes the `Base` backbone model trained in the tutorial. Additionally includes `Tiny` and `Large` backbone models.


## Loading

Identical to the shipped checkpoint, download and unzip a run, then point
`load_checkpoint` at its `decoder_best.pt` (or `decoder_last.pt`):

```python
import torch
from cropcounter import load_checkpoint

model, cfg = load_checkpoint(
    "weights/trained-models/<run>/decoder_best.pt",
    torch.device("cpu"),          # or "cuda" / "mps"
    weights_dir="weights",        # where the DINOv3 backbone .pth lives
)
model.eval()
```

See the **Loading it** section of [`weights/README.md`](../README.md) for why
`weights_dir=` matters and how to supply the gated DINOv3 backbone.

## Licence

The decoder checkpoints are released under this repository's licence (**AGPL-3.0-only**). 
The DINOv3 backbone weights are **not** ours to redistribute, they are covered by Meta's DINOv3 licence, which you accept when you request the gated download. 
Dataset imagery and annotations remain under their respective sources' terms.
