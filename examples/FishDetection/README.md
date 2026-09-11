# examples/FishDetection — fish detection on the Community Fish Detection Dataset

Applies `crop-counter`'s `task="box"` head to the **Brackish** source of the open
Community Fish Detection Dataset (CFD — LILA; Varini, Morris et al.), and scores it
against the released RF-DETR baselines through one shared `pycocotools` scorer on the
identical val images. It is the frozen-encoder thesis tested outside wheat.

**Colab-first.** The four notebooks in `notebooks/` run in order in one VM session:
`1_reformat` (CFD metadata → manifest → Brackish subset on the published `is_train`
split → fetch to the VM disk at long side 1024) → `2_training` (backbone check,
throughput probe, the 8 ep × 1 tile frozen run, the linear probe) → `3_inference`
(released RF-DETR over the same val images) → `4_evaluate` (one table, params, GFLOPs,
spot checks). Images stay on the VM disk; **weights, run dirs and results go to Google
Drive `frozen-trunk-detection/`** (`weights/`, `runs/<name>/`, `results/`).

Configs: `config_8ep.json` is the go/no-go run; `config_smoke_ep.json` is a 1-epoch
smoke on a handful of frames. Method, rules-before-numbers and results:
[`notebooks/docs/report.md`](notebooks/docs/report.md).
