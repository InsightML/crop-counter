# Frozen-trunk detection head on open marine data — Stage 0 benchmark memo

**Date:** 12 Sep 2026 (Stage 0a 11 Sep; Stage 0c run overnight 11–12 Sep on Colab, A100) · **Model:** `crop-counter` `poc/detection-head` — frozen DINOv3 ConvNeXt-B (~89M, `no_grad`) + the existing pyramid decoder (3.36M) + a new 4-channel geometry stem (wh/off, ~0.22M), i.e. CenterNet with its two missing branches added · **Question:** does a small head on *frozen* DINOv3 features detect fish competitively on the Brackish source of the Community Fish Detection Dataset (CFD), scored by the same `pycocotools` harness as the released RF-DETR baselines on the *identical* val images — and is that evidence for or against the frozen-encoder thesis behind wheat V2?

> **Status: Stage 0c complete (one seed, one source).** Rules and setup were written *before* any number existed (§ Rules written before the numbers) and are unchanged. Every number below comes from the run's artefacts (`results.csv`, `tau_calibration.json`, `tuning_table.md`, `baseline_metrics.json`, the three `history_*.json`, all alongside this memo and on Drive `frozen-trunk-detection/`), not retyped from memory; the notebooks that produced them are `examples/FishDetection/notebooks/1_reformat → 2_training → 3_inference → 4_evaluate` on `InsightML/crop-counter` PR #6.

## Prior work — this is not novel, and the memo says so

Frozen-backbone detection heads are well-trodden: ViTDet (frozen/plain ViT backbones), frozen-backbone DETR variants, Detic. The head itself is CenterNet (*Objects as Points*, 2019). The closest prior work is the DINOv2 and DINOv3 papers themselves, which probe frozen dense features for detection. What is defensible here is empirical only: (a) one independent visual domain (turbid underwater video) for a trunk trained on nadir wheat quadrats, (b) a like-for-like comparison against released open baselines on identical images through one scorer, (c) later, leave-one-source-out and label-efficiency curves against the *same architecture unfrozen*.

## Setup

- **Data.** CFD (LILA; Varini, Morris et al. 2025): >1.9M images / >935k boxes, 17 sources, one COCO archive, single class `fish`, per-image `dataset` / `original_data_source` / `is_train`. Stage 0c uses **only the Brackish source** (89 short clips, 14,674 frames in CFD; CC-BY-SA 4.0) — small, turbid, genuinely hard, cleanly licensed. Split = the **published `is_train`** field, never our own. Images fetched per-source from the LILA GCS mirror straight to the Colab VM and **resized to long side 1024 on write** (boxes rescaled in the COCO json); the master JSON is streamed, never `json.load`-ed. Per-source manifest (counts, empty-image fraction, box-size percentiles, stride-4 centre-cell collision rate, licence): `manifest/manifest.md`.
- **What the data looks like** (`results/viz/data_samples_{train,val}.png`, `data_stats.png`, generated from the fetched slice): turbid green-brown fixed-camera frames; 60 % of train and 56 % of val frames are empty, most non-empty frames hold 1–2 fish with a train tail to ~19. Box size √(w·h) runs 10–400 px. **The published clip split is not i.i.d.:** val comes from different deployments (a calibration board in view, larger fish — mode ≈ 70 px vs ≈ 40 px in train), so Brackish in-domain AP is a conservative number, not an inflated one. The augmented-tile panel (`train_tiles_targets.png`) shows peak-1.0 Gaussians on the fish and boxes decoded back from the wh/off targets wrapping them — Stage 0a's round trip, on real frames.
- **Model.** Backbone frozen, eval-mode, `no_grad`; the fusion trunk untouched; `task="box"` adds one `Conv3x3(192→128)-GN-GELU-Conv1x1(128→4)` stem beside the unchanged 1×1 heat head (offset bias initialised 0.5 — the existing decoder's `+0.5` cell-centre prior). Point path has zero new parameters: the shipped wheat `decoder_best.pt` still loads `strict=True` and reproduces MAE 8.59 / F1 0.729 (verification § below).
- **Targets / loss / decode.** CenterNet radius (`min_overlap=0.7`, its formula verbatim), per-object σ, peak exactly 1.0; penalty-reduced focal (bit-identical to the wheat path) + masked L1 on `log(w/4), log(h/4)` and on sub-cell offsets; decode = 3×3 max-pool local max → top-100 → boxes; optional IoU NMS. Empty-image hazard handled in the **sampler** (`negative_tile_fraction=0.2`, positives re-cropped until a box survives), not the loss.
- **Augmentation.** `natural` profile: scale jitter, random crop, horizontal flip, photometric only — **no vertical flip / 90° rotation** (gravity prior underwater; the wheat default would be wrong here).
- **Baseline.** Released `cfd-rf-detr-nano-640` (Apache) run over the same val images at its native 640, threshold 0.001, scored by the same `det_metrics.coco_eval`. RF-DETR-Small-1024 optional.
- **Compute.** Colab **free tier** (T4, session capped at ~5 h) — so the Stage 0c go/no-go run was trimmed at launch from the config's 12 epochs × 2 tiles/frame to **8 epochs × 1 tile/frame** (~11.5k tiles/epoch on 960×540 frames), leaving room for the linear probe and the baseline in one session; T4/L4 for the frozen path (expected CPU-bound on JPEG decode + augmentation — measured in notebook cell 8, `results/throughput_probe.json`). Data on the VM disk; weights, checkpoints, results on Drive.
- **Code provenance.** The Stage 0c Colab run executes from `poc/detection-head` @ `30dc0b6`. `origin/main` (Liam's PRs #1–#5: `labels` knob, `val_freq`, tiny backbone, `examples/demo/` move) was merged into the branch at `fe33845` while the run was in flight; the merge changes nothing numerical on the box-task path (`det_metrics` gained a progress bar only; `val_freq` defaults to every epoch; the box loader still takes all classes), so the run is reported as-is rather than restarted.
- **Working dir.** `output/frozen-trunk-detection/` — this memo, `colab_stage0c_brackish.ipynb`, `manifest/`, and (once run) `results.csv`, `predictions.json`, `viz/`. Code: `InsightML/crop-counter` branch `poc/detection-head`.

## Rules written before the numbers

1. **Never cite our AP against the README's `.609 / .596`.** Different, unpublished split. We re-run their weights on our val images instead.
2. **Baseline contamination cuts one way.** Their checkpoints were trained on CFD and very likely *saw* our val images. That biases in *their* favour. So: **a win for the frozen head is conservative; a loss is ambiguous and is reported as ambiguous, not spun.**
3. **Parameter framing.** "3.6M vs 30M" is dishonest — the frozen 89M ConvNeXt-B runs on every forward. Report **trainable params and total inference GFLOPs, both** (notebook cell 13, measured with `torch.utils.flop_counter`).
4. **Licence.** Brackish is CC-BY-SA — fine. Later stages: FathomNet is CC-BY-ND, Marine Detect (GBIF parts) / Salmon CV / TORSI are NC — a research benchmark, not anything InsightML ships; hence the permissive-only variant in Stage 1.
5. **LILA's own `is_train` caveat** (location info not always available; same backgrounds may straddle splits) inflates every in-domain number, ours and theirs. Protocol B (leave-one-source-out) is the trustworthy headline, not this.
6. **Seeds.** ≥2 seeds on the headline config before any claim; Stage 0c is one seed and is treated as a go/no-go, not a result.
7. **Kill criterion (fixed now):** frozen head **AP50 < ~0.5 while RF-DETR-Nano > ~0.8** on Brackish val ⇒ the frozen-trunk premise fails for underwater imagery → rescope to partial unfreeze (last ConvNeXt stage) or to the label-efficiency claim alone.
8. **Linear-probe reading (fixed now):** non-trivial AP with the fuse trunk frozen too ⇒ DINOv3 features are near-linearly box-decodable (thesis strong). Near-zero while the full decoder works ⇒ the story is the *decoder*, not the *features*. **Caveat found at run time (2026-09-12):** the first probe (`brackish_linearprobe_s0`) froze the fuse trunk at its *random* initialisation — the box config builds a fresh decoder and nothing loaded the wheat-trained trunk — so its AP50 of 0.006 is uninformative and is reported as such, not as evidence against the features. The informative variant initialises laterals/blocks/head from the wheat `decoder_best.pt` and then freezes the trunk (`config_linearprobe_wheatinit.json`).
9. Any throughput/speed claim is **measured** in the notebook or not made.

## Results

Brackish val = 3,127 frames / 1,965 GT boxes, native 960×540 (no resize applied). Ours: A100, bf16, 8 epochs × 1 tile/frame, ~5 min/epoch. All rows scored by the same `det_metrics.coco_eval` on the same images.

| Model | Input | Trainable M | Total M | GFLOPs | AP | AP50 | AP75 | AR100 | F1 @IoU.5 (τ = calibrated, see Tuning) |
|---|---|---|---|---|---|---|---|---|---|
| Frozen DINOv3-B + decoder + box head — **last epoch (8)** (`brackish_frozen_s0`) | 960×540 | 3.58 | 91.1 | **477** (at 1024×576) | **0.291** | **0.674** | 0.170 | 0.398 | **0.698** @τ 0.35 (0.714 with NMS 0.5) |
| same — best-by-val-loss epoch (4), the house convention | 960×540 | 3.58 | 91.1 | 477 | 0.227 | 0.586 | 0.109 | 0.337 | 0.651 @τ 0.35 (0.673 with NMS 0.5) |
| Linear probe, trunk frozen at **random** init, 2 ep (`brackish_linearprobe_s0`) — ⚠️ uninformative, see rule 8 | 960×540 | 0.22 | ~92 | — | 0.006 | 0.029 | 0.001 | 0.124 | 0.004 |
| Linear probe, trunk initialised from the **wheat** decoder then frozen (28 tensors loaded, geometry stem fresh), 2 ep (`brackish_linearprobe_wheatinit_s0`) | 960×540 | 0.22 | ~92 | — | 0.000 | 0.002 | 0.000 | 0.090 | 0.000 |
| RF-DETR-Nano, released (⚠️ trained on all of CFD, likely saw these frames) | 640 | — | ~30 | **97.3** (at 640², measured) | **0.492** | **0.790** | 0.525 | 0.645 | — |

**Compute, per rule 3:** "3.6M trainable vs ~30M" flatters us; per frame the frozen ConvNeXt-B trunk makes ours **~5× the inference FLOPs of Nano** (477 vs 97 GFLOPs, both measured with `torch.utils.flop_counter` on this GPU; Nano at its native 640², ours at 1024×576). Cheap to *train*, not cheap to *run*.

Per-epoch trajectory of the frozen head (val, τ=0.3 for P/R/F1): AP50 0.47 → 0.53 → 0.60 → 0.59 → 0.62 → 0.64 → 0.65 → **0.67**; AP75 0.02 → 0.04 → 0.07 → 0.11 → 0.09 → 0.19 → 0.11 → 0.17; val loss bottomed at epoch 4 (1.294) and rose to 1.436 by epoch 8 while every detection metric kept improving — the loss is dominated by the 56 % empty frames and the L1 terms, not by ranking quality, which is why the last epoch is the headline row and the best-by-val-loss row sits beside it (Liam's best-vs-last convention, argued rather than assumed). RF-DETR-Nano by object size: AP_small 0.014 / AP_medium 0.483 / AP_large 0.698 — it barely detects small fish on this source; ours by size lands from `4_evaluate`.

**Stage 0a — green (2026-09-11).** `tests/test_boxmap.py`: 57 tests; GT `(hm, wh, off)` → `decode_boxes` recovers all boxes to < 1e-3 px in both `log` and `linear` parameterisations and `coco_eval` scores AP = AP50 = 1.0; peak exactly 1.0; radius monotonic and equal to CenterNet's formula; `top_k` truncation and IoU-NMS verified; `parse_coco_detection` round-trips string ids and skips `empty` markers. Whole suite **169 passed**, ruff clean. Simulated CFD-like frames (960×540, median 46 px boxes): stride-4 centre-cell collisions 0.01 % at 10 fish/frame, 0.08 % at 50 — the empirical case against a stride-2 decoder.
**Stage 0c manifest facts (real metadata, `manifest/manifest.md`, streamed in 22 s):** Brackish in CFD = **14,674 frames / 14,120 boxes / 89 clips**, native **960×540** (so the 1024 long-side cap does not resize it), median box **45.7 px** (0.048 of the long side), **60.2 % of frames empty**, stride-4 centre-cell collision rate **0.0 %**. Published split: train 11,547 frames (71 clips, 12,155 boxes) / val 3,127 frames (18 clips, 1,965 boxes, 1,740 empty) — **clip-clean, 0 sequence overlap**, which independently validates both LILA's split and our sequence key. Whole-CFD context: 1,903,035 images / 935,049 boxes (matches LILA exactly), **64.8 % empty overall**, `is_train` never missing; TORSI collides at 15.6 % at stride 4 (every other source ≤ 0.8 %) — it needs a finer head or dropping in Stage 2.
**Throughput probe (T4, fp16, 768² tiles, forward only):** 14.4 img/s at 87 % GPU util with 2 workers, 14.6 img/s at 93 % with 4 — **GPU-bound, not CPU-bound**, which corrects the plan's expectation for this GPU (the frozen ConvNeXt-B forward dominates; Colab free tier has 2 vCPUs). Implication: ~11.5k tiles/epoch ≈ 15–20 min/epoch incl. validation on 3,127 frames, so 8 epochs + linear probe + baseline ≈ 3–3.5 h of a ~5 h free session. **A100 (bf16, same probe): 73–79 img/s** for 2–12 workers, i.e. ~5× the T4 and ~5 min per epoch including validation — Freddie moved the run to Colab Pro compute units on 2026-09-12 for exactly this reason. Buying the A100 pays for the frozen path too; the unfrozen comparator will need it.

## Tuning

`tau` is the decode confidence threshold; every P/R/F1 the training log printed used the config's untuned 0.3. Calibrated the way the WheatHead example does (its report § 5): one forward pass over Brackish val **per checkpoint**, decoded once at `ap_tau` 0.01, then every `tau` from 0.05 to 0.75 in steps of 0.05 applied as a mask over that one decoded set (`det_metrics.sweep_tau_boxes`; § 4 of `4_evaluate.ipynb`). **AP / AP50 / AP75 / AR100 did not move and could not have** — they integrate over the score axis — so only the operating-point columns (P/R/F1, counts) are recomputed at the calibrated `tau`.

**Both checkpoints are calibrated and the last epoch is the headline.** `train.py` writes `best.pt` on lowest val loss; on this run that selector stopped at epoch 4 while AP, AP50, AP75 and F1 all improved through epoch 8. Selecting on val loss and then reporting the epoch it did not pick is said out loud, not papered over: the val-loss rule is the house convention and it is the wrong proxy for this head on this data (the loss is dominated by the 56 % empty frames and the L1 terms). Stage 1 selects on val AP50 instead. `last.pt` has no `predictions.json` from training, so § 4 decodes `predictions_last.json` through the identical `evaluate_boxes` path (same `ap_tau`, top-k, NMS setting, clipping); its rescored AP equals the history value (0.291), the parity check.

![Tau sweep on Brackish val, one panel per checkpoint: count MAE (red, left axis) against localisation F1 (blue, right axis), dashed line at each panel's chosen tau.](images/tau_sweep.png)

Figure T1. Left `best.pt` (epoch 4), right `last.pt` (epoch 8, headline), each at its own MAE-argmin `tau`.

**Choice of `tau`.** MAE-argmin gives **0.35 for both checkpoints**. The F1-argmax agrees for `best.pt` (0.35, F1 0.651) and sits one grid step higher for `last.pt` (0.40, F1 0.700 vs 0.698 at 0.35) — a flat top, not a disagreement of substance; at 0.40 the trade is +0.06 precision for −0.04 recall and bias moves from −0.07 to −0.14. Both checkpoints are reported at 0.35. (Source: `tau_calibration.json` → `best`/`last` blocks; `best_tau` = the headline's.)

**NMS.** `box_nms_iou` was `null` for the run (the 3×3 local-max test alone; Brackish's 0.0 % centre-cell collision rate suggested boxes rarely overlap). Spot checks (Figure T2) showed stacked duplicate boxes on single fish in the dense clusters, so IoU-NMS was compared at each checkpoint's `tau`, suppression inside the decode before thresholding:

| Checkpoint | τ | NMS IoU | Precision | Recall | F1 | count MAE | bias |
|---|---|---|---|---|---|---|---|
| last epoch (headline) | 0.35 | none | 0.741 | 0.660 | 0.698 | 0.251 | −0.07 |
| last epoch (headline) | 0.35 | **0.5** | **0.780** | 0.658 | **0.714** | **0.231** | −0.10 |
| last epoch (headline) | 0.35 | 0.6 | 0.777 | 0.659 | 0.713 | 0.232 | −0.10 |
| best epoch (val-loss) | 0.35 | none | 0.710 | 0.601 | 0.651 | 0.238 | −0.10 |
| best epoch (val-loss) | 0.35 | 0.5 | 0.767 | 0.600 | 0.673 | 0.213 | −0.14 |

Table T1. **Decision: NMS IoU 0.5 on.** +0.04 precision at flat recall on the headline checkpoint, F1 0.698 → 0.714, count MAE −0.02; 0.6 is indistinguishable from 0.5; the direction is the same on `best.pt`. Figure T2 is the evidence: the doubled boxes go, no fish is lost.

![NMS spot check on the last-epoch checkpoint: four val frames with at least three GT fish, NMS off on the left and NMS 0.5 on the right, GT green and ours red.](images/nms_compare.png)

Figure T2. Four val frames with ≥ 3 GT fish, `last.pt` at `tau` 0.35, NMS off (left) vs 0.5 (right).

| Model | Config | MAE | RMSE | Bias | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| best epoch (val-loss) | NMS off @τ=0.35 | 0.238 | 0.547 | −0.097 | 0.7104 | 0.6005 | 0.6509 |
| best epoch (val-loss) | NMS 0.5 @τ=0.35 | 0.213 | 0.523 | −0.138 | 0.7674 | 0.5995 | 0.6731 |
| last epoch | NMS off @τ=0.35 | 0.251 | 0.544 | −0.069 | 0.7410 | 0.6595 | 0.6979 |
| **last epoch** | **NMS 0.5 @τ=0.35** | 0.231 | 0.520 | −0.098 | **0.7799** | 0.6580 | **0.7138** |

Table T2. Validation metrics at per-row `tau` | IoU 0.5 | top-k 100 (`tuning_table.md`, pasted verbatim). Best epoch = lowest val loss, not best AP; headline = last epoch, for the reason above.

## Visual spot checks

`results/viz/spot_checks.png` (six val frames across the GT-count range; GT green · ours red · RF-DETR blue) and `nms_compare.png` (four dense frames, NMS off vs IoU 0.5, at τ = 0.35):

- **Empties are clean.** On the three GT-empty frames neither model fires — no turbid-background false positives from either, which is the main thing the negative-tile sampler was there to buy.
- **Single mid-size fish: both agree.** One-fish frames show the red and blue boxes on top of each other and on the fish.
- **The small fish is missed by both.** The one-GT frame with a small, low-contrast fish gets zero detections from ours *and* from Nano — the per-size table's story in one image.
- **Dense clusters are where the two differ.** On the 9-fish frame ours returns 8 and Nano 12; ours under-counts the pile of overlapping fish, Nano over-splits it. With NMS off ours draws visibly stacked duplicate boxes on single fish in these clusters; IoU-0.5 NMS removes them (10 → 9 on one frame) with no lost fish, which is what the +0.06 precision at flat recall in the tuning table looks like.
- **Box extents:** ours are the right size on isolated fish and loose or merged in clusters — consistent with the AP75 gap being localisation rather than detection.

## Interpretation

Rules applied before narrative: rule 7 — the kill criterion **does not fire** (ours AP50 0.674 ≥ 0.5; Nano 0.790 < 0.8); rule 2 — Nano's row is biased in *its* favour (trained on all of CFD, very likely on these frames), so the gap below is an upper bound on the true gap and is reported as ambiguous, not as a clean loss; rule 8 — both linear probes are read as trunk results, not feature results (below).

1. **The frozen trunk finds fish nearly as well as the released detector and sizes them badly.** Of the AP gap to RF-DETR-Nano (0.492 vs 0.291), AP50 accounts for 0.116 and AP75 for 0.355. A 3.6M-parameter CenterNet head on frozen DINOv3 features, 8 epochs, one source, reaches 85 % of Nano's AP50 and a third of its AP75. The heat branch learned the task; the wh/off branch, trained from scratch on 12k boxes with log-space L1, has not converged on tight extents (AP75 was still rising and noisy: 0.11 → 0.19 → 0.11 → 0.17 over the last four epochs).
2. **No small-object edge — the hypothesis that a stride-4 frozen trunk would beat Nano on small fish is dead.** Per-size COCO AP, scored locally from the saved predictions against a val split regenerated from the CFD metadata (which reproduces both models' headline AP to four decimals — the scorer and the split are deterministic across machines): ours (epoch 8, headline) small **0.007** / medium 0.310 / large 0.298 (epoch 4: 0.010 / 0.245 / 0.223); Nano small 0.014 / medium 0.483 / large 0.698. Both collapse on COCO-small objects (< 32² px) and Nano's small-object recall is twice ours (AR_small 0.378 vs 0.180). Ours is also flat between medium and large where Nano climbs (0.31 vs 0.30 against 0.48 vs 0.70) — consistent with a size branch that has not learned large extents. For wheat V2, whose objects are small, this is the finding to carry: the frozen trunk does not rescue small-object recall by itself.
3. **Val loss is the wrong model selector for this head.** It bottomed at epoch 4 while AP50 rose for four more epochs; the loss is dominated by the 56 % empty frames and by the L1 terms. Selection should use AP50 (or F1 at the calibrated τ) on a held-out split — a one-line change to `train.py` worth making before Stage 1.
4. **The linear probes say the decoder does the domain work.** A trunk frozen at random init (uninformative by construction) and a trunk frozen at its *wheat*-trained weights both give AP50 ≈ 0.002–0.03 with a trainable readout, while training the same trunk fresh on the same frozen features gives 0.67. The fusion blocks are where fish-vs-wheat specialisation lives; DINOv3 supplies a shared substrate that neither a linear readout nor a wheat-specialised readout can use directly. **Not tested:** a linear probe straight off the raw backbone pyramid — the informative "are the features linearly decodable" experiment — which is a ~20-line config away and should precede any claim about the features themselves.
5. **Calibration moved the operating point, not the ranking.** Both checkpoints calibrate to τ = 0.35 by MAE-argmin (the F1-argmax for the headline sits one grid step higher, a flat top), and IoU-0.5 NMS removes real duplicate boxes: on the headline checkpoint F1 0.683 at the untuned 0.3 → 0.698 at 0.35 → 0.714 with NMS, with precision 0.68 → 0.78 at essentially unchanged recall. Details in § Tuning.
6. **One seed, one source, one split.** Epoch-to-epoch swings of ±0.08 AP75 on 1,965 val boxes are visible in the trajectory; nothing here is a result until ≥ 2 seeds and Protocol B.

## Implications

- **For the foundation-model transfer-learning thesis (InsightML wiki) thesis / wheat V2:** first data point outside wheat, and it supports the *weak* form of the thesis — frozen DINOv3 + a small trained decoder detects competently on an unrelated visual domain after 40 minutes on one GPU — and undercuts the *strong* form (that a frozen encoder plus a thin, transferable head suffices): the head that transfers is the fusion trunk retrained per domain, not a linear readout. For V2's few-shot/FiLM ambition that is a real constraint: whatever FiLM conditions has to be the fusion trunk, not a head on top of it.
- **For Stage 1+:** go. Protocol A on the published split with baselines re-scored + the permissive-only variant; Protocol B LOSO with the *unfrozen* twin as comparator (GPU spend goes here — the A100 probe read 73–79 img/s frozen, so the unfrozen run is affordable); Protocol C label efficiency. Before any of them: AP50-based checkpoint selection, a second seed of this run, and the raw-backbone linear probe.
- **Licence:** ⚠️ the DINOv3 commercial-licence question on the framework page is still open — settle it before wheat V2 depends on the answer.
- **Not building:** backbone feature caching (16.9 MiB fp16 per 768² tile → ~177 GB for 10k tiles, and augmentation invalidates it).

## Verdict

**Go, with the claim sized to the evidence.** On Brackish the frozen-DINOv3 trunk with a 3.6M-parameter CenterNet head reaches AP 0.291 / AP50 0.674 after 8 epochs (~40 min on an A100), against 0.492 / 0.790 for the released RF-DETR-Nano scored on the identical frames by the identical function — a baseline that trained on all 17 CFD sources and almost certainly saw these frames, so the true gap is at most what is shown. The head finds fish nearly as well as Nano and localises them far less tightly; the linear probes show the fusion trunk, not a linear readout, carries the domain adaptation. The frozen-trunk premise survives its first test outside wheat; it does not yet earn a "competitive with trained-end-to-end detectors" sentence, and one seed on one source earns nothing about generality. The next spend is Protocol B with the unfrozen twin, not more epochs here.

## Verification ledger

| Check | Result |
|---|---|
| Existing test suite green after the change | ✅ 169 passed (66 pre-existing unchanged + 57 boxmap + 46 cfd), ruff clean |
| `decoder_best.pt` loads `strict=True` into the point model | ✅ point path has zero new parameters; state-dict key set asserted equal in a test |
| Wheat val (122 imgs, τ=0.35, MPS) reproduced before AND after | ✅ identical to every printed digit: MAE 8.4836 · RMSE 11.8726 · P 0.7413 · R 0.7168 · **F1 0.7288** · val loss 1.0775. (The 8.59 MAE quoted in the scale-test memo came from a different eval setting; the check here is the before/after identity.) |
| Stage 0a round-trip tests green | ✅ see § Results |
| End-to-end on REAL Brackish frames (local, MPS): `cfd subset` 24+12 frames → `fetch` → 1-epoch `train --task box` → `predictions.json` + COCO eval | ✅ ran clean after one fix it caught (fetch left `JPEGImages/` in `file_name`; now basename + `cfd_file_name`). 14/24 train frames empty → sampler active, `n_pos` 6–16 per batch, loss finite, val eval + artefacts written. AP after 1 epoch on 24 images is 0 by construction, not a result. |
| Converted backbone reproduces committed example counts in the cloud (notebook cell 4, Colab T4) | ✅ strict load OK; under the T4's **fp16** autocast 5 of 6 images reproduce the fp32/MPS reference counts exactly (0, 0, 28, 55, 64) and the densest (92 plants) gives 89 — τ=0.35 threshold noise on a dense image, not a backbone mismatch. **A100 follow-up:** in CUDA **fp32** the same image gives 89, in bf16 90 — so the delta is CUDA-vs-MPS conv numerics (TF32 by default on Ampere), not mixed precision. Gate is now fp32 with TF32 off at max(1, 5 %) per image; a wrong backbone would wreck all six, cross-device noise moves a threshold-straddling peak or two on the densest one. |
| Baseline re-scored through our `coco_eval` (metric parity) | ✅ RF-DETR-Nano's detections and ours are scored by the same function on the same `val/annotations.json`; regenerating the val split on the laptop from the CFD metadata and rescoring both saved prediction files reproduces AP 0.4925 (Nano), 0.2274 (ours, epoch 4) and 0.2910 (ours, epoch 8) to four decimals — scorer and split are deterministic across machines |
| End to end on Colab: `runs/<name>/` with `history.json`, `curves.png`, `best.pt`/`last.pt`, `predictions.json` on Drive; Stage 0c AP in the table | ✅ three runs (`brackish_frozen_s0`, `brackish_linearprobe_s0`, `brackish_linearprobe_wheatinit_s0`) |
| τ calibrated per checkpoint, NMS compared, best-vs-last argued (Liam's tuning convention) | ✅ § Tuning |
| Inference FLOPs measured for both models | ✅ 477 (ours, 1024×576) / 97.3 (Nano, 640²) GFLOPs |
| Branch is `poc/detection-head`, merged with `origin/main` (Liam's PRs #1–#5) at `fe33845`; deliverable = PR #6 | ✅ |
