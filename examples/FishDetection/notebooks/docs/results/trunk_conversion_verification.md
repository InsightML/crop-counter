# DINOv3 ConvNeXt Tiny / Small trunks, converted from timm — and proved

Meta gates the DINOv3 downloads. timm re-hosts the *same tensors* ungated on the
Hugging Face Hub, under timm's own module names. `examples/FishDetection/scripts/convert_timm_dinov3.py`
renames them into Meta's layout and writes a plain state dict that Meta's own
hub constructor loads with `strict=True`.

Strictness is a weak guarantee on its own. ConvNeXt stages are a stack of
identically-shaped blocks, so a permuted or partially-dropped mapping loads
strictly, matches every shape, and silently returns wrong features. Everything
below exists to rule that out.

Reproduce with:

```bash
python examples/FishDetection/scripts/convert_timm_dinov3.py --size tiny  --check --verify
python examples/FishDetection/scripts/convert_timm_dinov3.py --size small --check --verify
```

## What was converted

| size | HF file | timm tensors | Meta tensors | written | hub strict load |
|---|---|---|---|---|---|
| tiny | `timm/convnext_tiny.dinov3_lvd1689m` → `model.safetensors` (fp32) | 180 | 182 | `weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth`, 111.3 MB | OK, 27.82 M params |
| small | `timm/convnext_small.dinov3_lvd1689m` → `model.safetensors` (fp32) | 342 | 344 | `weights/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth`, 197.9 MB | OK, 49.45 M params |

The +2 is not slack: Meta's `ConvNeXt` registers its final LayerNorm as
`self.norm` and then aliases the *same module* into `self.norms[3]`, so the
checkpoint carries that one norm's weight and bias twice. `head.norm.*` is the
map's only one-to-many entry; every other key is 1:1.

```
stem.{0,1}.*                 -> downsample_layers.0.{0,1}.*
stages.i.downsample.{0,1}.*  -> downsample_layers.i.{0,1}.*   (i >= 1)
stages.i.blocks.j.conv_dw.*  -> stages.i.j.dwconv.*
stages.i.blocks.j.mlp.fc1.*  -> stages.i.j.pwconv1.*
stages.i.blocks.j.mlp.fc2.*  -> stages.i.j.pwconv2.*
stages.i.blocks.j.norm.*     -> stages.i.j.norm.*
stages.i.blocks.j.gamma      -> stages.i.j.gamma
head.norm.*                  -> norm.*  AND  norms.3.*
```

Anything outside that list raises rather than being skipped, and two timm keys
claiming one Meta key is an error rather than a last-write-wins. Unit-tested on
a synthetic key list in `tests/test_convert_timm_dinov3.py`.

## Proof 1 — the map reproduces Meta's own `base` file, bit for bit

The one size whose gated Meta checkpoint is already on disk is `base`. Running
the *same* mapping over timm's `base` re-host and diffing tensor-by-tensor
against Meta's file settles the question completely: if renaming timm's tensors
yields Meta's checkpoint exactly, this is the map Meta used, not a map that
happens to work.

```
[base] tensors  : timm 342 -> Meta 344 (+2 norm alias)
[base] vs weights/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth:
[base]   keys 344 vs 344, missing 0, extra 0
[base]   bit-identical tensors 344/344, worst max-abs 0.000e+00 (stages.3.2.pwconv2.weight)
```

344 of 344 tensors bit-identical, zero missing, zero extra, worst absolute
difference exactly 0. Tiny and Small go through the identical code path, and
the checks below confirm it end-to-end for those two specifically.

```bash
python examples/FishDetection/scripts/convert_timm_dinov3.py --size base \
  --out /tmp/base_out --compare-meta weights/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth
```

## Proof 2 — activations, with a negative control

Four real images (three CFD frames from `data/cfd17_smoke/val/images/`, one
wheat plot from `data/val/images/`) resized to 256x384 — a multiple of 32 and
deliberately non-square, so a transposed feature map cannot slip through — and
ImageNet-normalised. fp32, CPU, `torch.no_grad()`.

Two models see the same batch:

* **reference** — `timm.create_model("convnext_<size>.dinov3_lvd1689m", pretrained=True)`,
  via `forward_intermediates(indices=[0,1,2,3], norm=False, intermediates_only=True)`.
* **ours** — the real hub constructor against the converted `.pth`, via
  `get_intermediate_layers(x, n=[0,1,2,3], reshape=True, norm=False)`.

Stage 3 is also compared *with* the final norm applied: our
`get_intermediate_layers(..., norm=True)` (which runs `norms[3]`, Identity for
stages 0–2) against timm's stage-3 map pushed through `timm_model.head.norm`.
That row is the only one that exercises the `head.norm -> norm / norms.3`
duplication, so it is reported separately rather than folded in.

`relative` = max-abs difference / the reference map's own max-abs. The bar is
**≤ 1e-4 on every stage**; the shipped conversions come in three orders of
magnitude under it.

Then the same check is run twice more on *deliberately corrupted* mappings.
Both corruptions keep the key set and every shape identical, so both still load
with `strict=True` — which is exactly the point. A check that a wrong mapping
also passes is not a check.

* `control:swap-stage2-blocks` — blocks `stages.2.1` and `stages.2.2` have
  their tensors swapped.
* `control:drop-gamma` — every LayerScale `gamma` left at the constructor's
  `1e-6`, i.e. what "I forgot to map `.gamma`" looks like.

### tiny — `convnext_tiny.dinov3_lvd1689m`

Images (4, resized to 256x384, ImageNet-normalised): `08-30-2020_14-27-03_m_salmon_camera_frame_000000.jpg`, `08-30-2020_14-27-03_m_salmon_camera_frame_000067.jpg`, `08-30-2020_14-27-03_m_salmon_camera_frame_000143.jpg`, `01_11_2024_12_54_Adam_Hayward_B_5_PICTURE_SLIMERS.jpg`

| variant | stage | max abs diff | mean abs diff | ref max abs | relative | <= 1e-4 |
|---|---|---|---|---|---|---|
| `converted` | stage0 (s4) | 2.682e-06 | 1.215e-07 | 1.491e+00 | 1.799e-06 | yes |
| `converted` | stage1 (s8) | 5.960e-06 | 1.088e-07 | 3.041e+00 | 1.960e-06 | yes |
| `converted` | stage2 (s16) | 2.747e-04 | 1.371e-06 | 1.188e+02 | 2.312e-06 | yes |
| `converted` | stage3 (s32) | 4.101e-05 | 9.823e-07 | 1.590e+01 | 2.579e-06 | yes |
| `converted` | stage3 + final norm | 3.552e-05 | 3.736e-06 | 1.381e+01 | 2.572e-06 | yes |
| `control:swap-stage2-blocks` | stage0 (s4) | 2.682e-06 | 1.215e-07 | 1.491e+00 | 1.799e-06 | yes |
| `control:swap-stage2-blocks` | stage1 (s8) | 5.960e-06 | 1.088e-07 | 3.041e+00 | 1.960e-06 | yes |
| `control:swap-stage2-blocks` | stage2 (s16) | 1.081e+02 | 5.259e-01 | 1.188e+02 | 9.098e-01 | **NO** |
| `control:swap-stage2-blocks` | stage3 (s32) | 1.487e+01 | 3.752e-01 | 1.590e+01 | 9.352e-01 | **NO** |
| `control:swap-stage2-blocks` | stage3 + final norm | 1.611e+01 | 1.634e+00 | 1.381e+01 | 1.166e+00 | **NO** |
| `control:drop-gamma` | stage0 (s4) | 1.492e+00 | 1.197e-01 | 1.491e+00 | 1.001e+00 | **NO** |
| `control:drop-gamma` | stage1 (s8) | 2.974e+00 | 7.448e-02 | 3.041e+00 | 9.779e-01 | **NO** |
| `control:drop-gamma` | stage2 (s16) | 1.187e+02 | 6.989e-01 | 1.188e+02 | 9.989e-01 | **NO** |
| `control:drop-gamma` | stage3 (s32) | 1.607e+01 | 3.965e-01 | 1.590e+01 | 1.011e+00 | **NO** |
| `control:drop-gamma` | stage3 + final norm | 1.048e+02 | 3.126e+00 | 1.381e+01 | 7.591e+00 | **NO** |

- `converted`: **PASS**
- `control:swap-stage2-blocks`: **FAIL**
- `control:drop-gamma`: **FAIL**

### small — `convnext_small.dinov3_lvd1689m`

Images (4, resized to 256x384, ImageNet-normalised): `08-30-2020_14-27-03_m_salmon_camera_frame_000000.jpg`, `08-30-2020_14-27-03_m_salmon_camera_frame_000067.jpg`, `08-30-2020_14-27-03_m_salmon_camera_frame_000143.jpg`, `01_11_2024_12_54_Adam_Hayward_B_5_PICTURE_SLIMERS.jpg`

| variant | stage | max abs diff | mean abs diff | ref max abs | relative | <= 1e-4 |
|---|---|---|---|---|---|---|
| `converted` | stage0 (s4) | 1.244e-06 | 6.577e-08 | 7.379e-01 | 1.686e-06 | yes |
| `converted` | stage1 (s8) | 1.907e-06 | 6.971e-08 | 1.518e+00 | 1.256e-06 | yes |
| `converted` | stage2 (s16) | 4.120e-04 | 1.781e-06 | 1.979e+02 | 2.082e-06 | yes |
| `converted` | stage3 (s32) | 1.049e-05 | 2.564e-07 | 3.345e+00 | 3.136e-06 | yes |
| `converted` | stage3 + final norm | 3.636e-05 | 2.977e-06 | 1.308e+01 | 2.780e-06 | yes |
| `control:swap-stage2-blocks` | stage0 (s4) | 1.244e-06 | 6.577e-08 | 7.379e-01 | 1.686e-06 | yes |
| `control:swap-stage2-blocks` | stage1 (s8) | 1.907e-06 | 6.971e-08 | 1.518e+00 | 1.256e-06 | yes |
| `control:swap-stage2-blocks` | stage2 (s16) | 8.879e+01 | 3.613e-01 | 1.979e+02 | 4.486e-01 | **NO** |
| `control:swap-stage2-blocks` | stage3 (s32) | 3.026e+00 | 5.150e-02 | 3.345e+00 | 9.047e-01 | **NO** |
| `control:swap-stage2-blocks` | stage3 + final norm | 5.524e+00 | 6.120e-01 | 1.308e+01 | 4.223e-01 | **NO** |
| `control:drop-gamma` | stage0 (s4) | 7.371e-01 | 4.975e-02 | 7.379e-01 | 9.989e-01 | **NO** |
| `control:drop-gamma` | stage1 (s8) | 1.365e+00 | 4.389e-02 | 1.518e+00 | 8.990e-01 | **NO** |
| `control:drop-gamma` | stage2 (s16) | 1.977e+02 | 8.595e-01 | 1.979e+02 | 9.989e-01 | **NO** |
| `control:drop-gamma` | stage3 (s32) | 3.346e+00 | 1.639e-01 | 3.345e+00 | 1.000e+00 | **NO** |
| `control:drop-gamma` | stage3 + final norm | 4.503e+01 | 2.499e+00 | 1.308e+01 | 3.442e+00 | **NO** |

- `converted`: **PASS**
- `control:swap-stage2-blocks`: **FAIL**
- `control:drop-gamma`: **FAIL**

Two things worth reading off the control rows. First, both controls leave
stages 0 and 1 at float noise — as they must, since neither corruption touches
those stages — which says the check is localising the damage rather than
failing everything indiscriminately. Second, `drop-gamma` moves the *relative*
difference to ~1.0 at every stage: zeroing LayerScale turns every block into a
pass-through, so the residual stream is all that survives. A shape check sees
none of this.

## Smoke: `CropCounter` builds and runs from the converted weights

`CropCounter(backbone_size=..., task="box")` resolves its checkpoint through
`resolve_backbone_weights`, so this also confirms the filenames are the ones
`src/cropcounter/weights.py` expects. One 768x768 forward, `torch.no_grad()`,
on **MPS and CPU** — both produce finite `{heatmap, wh, off}` at
`(1, C, 192, 192)`, i.e. the stride-4 output.

| size | backbone params | decoder (trainable) params | total |
|---|---|---|---|
| tiny | 27,820,128 | 3,487,813 | 31,307,941 |
| small | 49,454,688 | 3,487,813 | 52,942,501 |
| base | 87,566,464 | 3,579,973 | 91,146,437 |

The decoder is 92,160 parameters smaller for tiny/small because their stage-3
width is 768 rather than base's 1024, which shrinks the top lateral 1x1.
Tiny and Small share a decoder exactly — they have identical stage widths
`(96, 192, 384, 768)` and differ only in stage-2 depth (9 blocks vs 27).

## Cost: `measure_trunk.py`, MPS, 1024x576, batch 1

Full `CropCounter` (frozen trunk + box head), eval mode, 5 warm-up iterations
then 50 timed, `torch.mps.synchronize()` around the timed window. `amp_dtype`
returns `None` off CUDA, so autocast is off and this is fp32 throughout.
GFLOPs are one forward, counted by `torch.utils.flop_counter.FlopCounterMode`
**on MPS directly** — it did not misbehave there, so no CPU fallback was needed
(the script keeps one anyway and records `flops_device`). `peak_gpu_mem_mb` is
null: `torch.cuda.max_memory_allocated` has no MPS equivalent.

```bash
python examples/FishDetection/scripts/measure_trunk.py --size tiny --device mps \
  --side 1024x576 --batch 1 --iters 50
```

| size | total params | trainable | GFLOPs @ 1024x576 | img/s | ms/img | vs base |
|---|---|---|---|---|---|---|
| tiny | 31,307,941 | 3,487,813 | 220.0 | 22.58 | 44.29 | 0.46x FLOPs, 2.07x faster |
| small | 52,942,501 | 3,487,813 | 319.4 | 15.72 | 63.60 | 0.67x FLOPs, 1.44x faster |
| base | 91,146,437 | 3,579,973 | 477.0 | 10.89 | 91.82 | — |

Machine: macOS 26.1, arm64, torch 2.13.0, MPS. These are laptop numbers for
relative sizing, not a throughput claim — a CUDA run under bf16 autocast will
land somewhere else entirely, and the FLOP ratios (0.46 / 0.67 / 1.00) are the
part that transfers.

## Environment

* `timm` 1.0.29 (installed into the `crop-counter-unfrozen` conda env for this
  work), with `huggingface_hub` 1.31.0 and `safetensors` 0.8.0 pulled in with it.
* torch 2.13.0, Python 3.12.13.
* DINOv3 hub repo read from the local cache at
  `~/.cache/torch/hub/facebookresearch_dinov3_main`, so `--check` and `--verify`
  touch no network beyond the timm download.

## Gotcha worth keeping

Meta's `_make_dinov3_convnext` turns `weights=<path>` into a `file://` URL and
hands it to `load_state_dict_from_url`, which **caches by basename** under
`~/.cache/torch/hub/checkpoints/`. Re-converting to the same filename and
re-loading would therefore silently pick up the *previous* conversion.
`build_hub_model` deletes the stale cached copy before loading. Anyone loading
these checkpoints by hand after a re-convert needs to do the same.
