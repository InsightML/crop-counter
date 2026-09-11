"""Frozen DINOv3 ConvNeXt backbone + pyramid decoder for counting and detection.

The backbone stays frozen and permanently in eval mode; only the decoder
trains.

Two tasks share one fusion trunk, selected by ``task``:

* ``"point"`` — single-channel peak logits at ``output_stride`` (4 by
  default). Apply sigmoid + ``heatmap.decode_peaks`` to get points.
* ``"box"`` — the same peak logits plus a ``geometry`` branch emitting
  ``wh`` and ``off``. Apply sigmoid to ``"heatmap"`` and feed the triple to
  ``boxmap.decode_boxes``.

``task`` is a parameter, not a subclass, for one reason: the point decoder's
**state-dict key set must not move**. The shipped wheat checkpoint loads with
``strict=True``, so the box branch has to be absent — not merely unused — when
``task == "point"``. A subclass or an always-constructed-then-ignored branch
would add keys and break that load.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import ContextManager, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .weights import WEIGHT_FILES, resolve_backbone_weights

STAGE_CHANNELS = {
    "small": (96, 192, 384, 768),
    "base": (128, 256, 512, 1024),
    "large": (192, 384, 768, 1536),
}
STAGE_STRIDES = (4, 8, 16, 32)

# ImageNet statistics, as the DINOv3 web-pretrained checkpoints expect.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Hidden width of the box task's geometry branch.
GEOMETRY_WIDTH = 128

#: Initial bias of the two offset channels. ``decode_peaks``/``decode_boxes``
#: place a zero-offset detection at the cell CENTRE, so 0.5 — not 0.0 — is the
#: "no sub-cell correction" prior.
OFFSET_BIAS_INIT = 0.5

#: Compute capability at which CUDA gains native bfloat16 tensor cores (Ampere,
#: sm_80). Below it — Turing T4 (sm_75), the free Colab GPU — bf16 is emulated:
#: cuDNN convolutions can refuse it outright with CUDNN_STATUS_NOT_SUPPORTED,
#: and where they do run they are slower than fp16.
BF16_MIN_CUDA_MAJOR = 8


def amp_dtype(device: torch.device) -> Optional[torch.dtype]:
    """The mixed-precision dtype to autocast to on ``device``, or None for fp32.

    * CUDA, compute capability >= 8.0 (Ampere and later): ``torch.bfloat16``.
    * CUDA, below that (Turing/Volta/Pascal, e.g. the Colab T4): ``torch.float16``
      — which needs a :class:`torch.amp.GradScaler` when training.
    * Anything else (MPS, CPU): ``None``, i.e. autocast stays off.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return None
    major, _minor = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= BF16_MIN_CUDA_MAJOR else torch.float16


def autocast_context(device: torch.device) -> ContextManager[None]:
    """``torch.autocast`` in :func:`amp_dtype`'s dtype, or a no-op off CUDA.

    Wrap every forward pass in this rather than hard-coding a dtype, so one
    device check serves training, validation and inference alike.
    """
    dtype = amp_dtype(device)
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(torch.device(device).type, dtype=dtype)


class DinoV3Backbone(nn.Module):
    """Frozen DINOv3 ConvNeXt feature extractor exposing the 4 stage maps.

    Never pass ``patch_size`` to the hub constructor: it makes
    ``get_intermediate_layers`` bilinearly resample every stage to a
    ViT-like stride-16 grid, destroying the stride-4 skip the decoder
    relies on. The hub default (None) keeps native stage resolutions.
    """

    def __init__(self, size: str = "base", weights_dir: Optional[Path] = None) -> None:
        super().__init__()
        if size not in WEIGHT_FILES:
            raise ValueError(f"backbone size must be one of {sorted(WEIGHT_FILES)}, got {size!r}")
        # Raises BackboneWeightsNotFound with Meta's gated-download instructions.
        weights_path = resolve_backbone_weights(size, weights_dir)

        self.size = size
        self.stage_channels: Tuple[int, ...] = STAGE_CHANNELS[size]
        self.model = torch.hub.load(
            repo_or_dir="facebookresearch/dinov3",
            source="github",
            model=f"dinov3_convnext_{size}",
            weights=str(weights_path),
            # Meta's hub code is fetched from GitHub on first load; without this,
            # torch prompts on stdin for confirmation and blocks CI/nohup runs.
            trust_repo=True,
        )
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True) -> "DinoV3Backbone":
        """Keep the frozen backbone in eval mode regardless of parent state."""
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return the four stage maps (NCHW) at strides 4/8/16/32."""
        with torch.no_grad():
            feats = self.model.get_intermediate_layers(x, n=[0, 1, 2, 3], reshape=True)
        return list(feats)


class _ConvBlock(nn.Sequential):
    """Two 3x3 Conv -> GroupNorm -> GELU units."""

    def __init__(self, in_ch: int, out_ch: int, groups: int = 32) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
        )


class PyramidDecoder(nn.Module):
    """U-Net-style fusion decoder over an FPN-style lateral pyramid -> 1-channel logits.

    A hybrid: 1x1 lateral projections bring every backbone stage to a uniform
    ``c_dec`` width (the FPN trait), then a bilinear x2 top-down ladder from
    stride 32 to stride 4 *concatenates* each same-stride lateral and applies a
    double conv block (the U-Net trait). A single head emits fine-scale logits
    rather than per-level predictions. With ``output_stride=2`` one extra
    skip-less upsample+conv block is added (ConvNeXt has no stride-2 features
    to fuse).

    ``task="box"`` adds a second branch off the same fused map::

        x (c_dec, stride 4)
        |-- head     : Conv1x1(c_dec -> 1)                          peak logits
        `-- geometry : Conv3x3(c_dec -> 128) -> GN(32) -> GELU
                       -> Conv1x1(128 -> 4)   channels 0:2 = wh, 2:4 = off

    ``head`` keeps its name, module and -4.0 bias in both tasks, so the point
    checkpoint's keys are a strict subset of the box decoder's.
    """

    #: The two prediction tasks. ``"point"`` must stay free of extra parameters.
    TASKS = ("point", "box")

    def __init__(
        self,
        stage_channels: Sequence[int],
        c_dec: int = 192,
        output_stride: int = 4,
        task: str = "point",
    ) -> None:
        super().__init__()
        if output_stride not in (2, 4):
            raise ValueError(f"output_stride must be 2 or 4, got {output_stride}")
        if task not in self.TASKS:
            raise ValueError(f"task must be one of {self.TASKS}, got {task!r}")
        self.output_stride = output_stride
        self.task = task

        self.laterals = nn.ModuleList(
            [nn.Conv2d(c, c_dec, kernel_size=1) for c in stage_channels]
        )
        # One fuse block per ladder step: stride 16, 8, 4.
        self.blocks = nn.ModuleList([_ConvBlock(2 * c_dec, c_dec) for _ in range(3)])
        self.refine = _ConvBlock(c_dec, c_dec) if output_stride == 2 else None
        self.head = nn.Conv2d(c_dec, 1, kernel_size=1)
        # Focal-style prior: start predicting p ~ 0.02 everywhere so the
        # dominant negatives don't swamp early training.
        nn.init.constant_(self.head.bias, -4.0)

        # Constructed only for the box task — `None` registers no parameters,
        # so the point state-dict key set is exactly what it always was.
        self.geometry: Optional[nn.Sequential] = None
        if task == "box":
            self.geometry = nn.Sequential(
                nn.Conv2d(c_dec, GEOMETRY_WIDTH, kernel_size=3, padding=1),
                nn.GroupNorm(32, GEOMETRY_WIDTH),
                nn.GELU(),
                nn.Conv2d(GEOMETRY_WIDTH, 4, kernel_size=1),
            )
            with torch.no_grad():
                # wh starts at 0 (= a 1-cell box under the log parameterisation);
                # off starts at 0.5, the zero-offset prior. decode_peaks already
                # places a point at the CELL CENTRE ((pts + 0.5) * stride), so a
                # 0.0 offset bias would start the box head predicting every
                # centre at its cell's top-left corner — half a cell of bias to
                # unlearn on every object.
                self.geometry[-1].bias.zero_()
                self.geometry[-1].bias[2:].fill_(OFFSET_BIAS_INIT)

    def freeze_fusion(self) -> "PyramidDecoder":
        """Freeze the fusion trunk, leaving only ``head`` (+ ``geometry``) trainable.

        The linear-probe variant: are the frozen DINOv3 features near-linearly
        box-decodable through a *fixed* fuse trunk? Answering that needs the
        laterals, the ladder blocks and the stride-2 refine block held still
        while the prediction branches train.
        """
        for module in (self.laterals, self.blocks, self.refine):
            if module is not None:
                module.requires_grad_(False)
        return self

    def forward(
        self, feats: Sequence[torch.Tensor]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Fuse the pyramid and predict.

        Returns:
            ``task="point"``: (B, 1, H, W) peak logits.
            ``task="box"``: ``{"heatmap": (B, 1, H, W), "wh": (B, 2, H, W),
            "off": (B, 2, H, W)}``.
        """
        laterals = [lat(f) for lat, f in zip(self.laterals, feats)]
        x = laterals[3]
        for skip_idx, block in zip((2, 1, 0), self.blocks):
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            x = block(torch.cat([x, laterals[skip_idx]], dim=1))
        if self.refine is not None:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            x = self.refine(x)
        logits = self.head(x)
        if self.geometry is None:
            return logits
        geometry = self.geometry(x)
        return {"heatmap": logits, "wh": geometry[:, :2], "off": geometry[:, 2:]}


class CropCounter(nn.Module):
    """Frozen DINOv3 ConvNeXt backbone + trainable pyramid decoder.

    forward(x): (B, 3, H, W) normalised RGB, H and W divisible by 32 ->
    (B, 1, H/s, W/s) logits for ``task="point"``, or a
    ``{"heatmap", "wh", "off"}`` dict at the same resolution for
    ``task="box"``. s = ``output_stride``.
    """

    def __init__(
        self,
        backbone_size: str = "base",
        weights_dir: Optional[Path] = None,
        c_dec: int = 192,
        output_stride: int = 4,
        task: str = "point",
    ) -> None:
        super().__init__()
        self.backbone = DinoV3Backbone(backbone_size, weights_dir)
        self.decoder = PyramidDecoder(
            self.backbone.stage_channels, c_dec=c_dec, output_stride=output_stride,
            task=task,
        )
        self.output_stride = output_stride
        self.task = task

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.decoder(self.backbone(x))

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Decoder parameters — the only ones the optimizer should see."""
        return [p for p in self.parameters() if p.requires_grad]
