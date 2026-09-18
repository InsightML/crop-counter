"""Frozen DINOv3 ConvNeXt backbone + pyramid decoder for peak-heatmap counting.

The backbone stays frozen and permanently in eval mode; only the decoder
trains. The model outputs single-channel logits at ``output_stride`` (4 by
default) — apply sigmoid + ``heatmap.decode_peaks`` to get points.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .weights import WEIGHT_FILES, resolve_backbone_weights

STAGE_CHANNELS = {
    "tiny": (96, 192, 384, 768),
    "small": (96, 192, 384, 768),
    "base": (128, 256, 512, 1024),
    "large": (192, 384, 768, 1536),
}
STAGE_STRIDES = (4, 8, 16, 32)

# ImageNet statistics, as the DINOv3 web-pretrained checkpoints expect.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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

    A hybrid: 1x1 lateral projections bring every backbone stage to the
    decoder's width (the FPN trait), then a bilinear x2 top-down ladder from
    stride 32 to stride 4 *concatenates* each same-stride lateral and applies a
    double conv block (the U-Net trait). A single head emits fine-scale logits
    rather than per-level predictions. With ``output_stride=2`` one extra
    skip-less upsample+conv block is added (ConvNeXt has no stride-2 features
    to fuse).

    **Width can vary per ladder step.** ``c_dec`` takes either one int (every
    step that width — the original behaviour) or three ints ordered coarse to
    fine: ``(stride 16, stride 8, stride 4)``.

    The reason is that cost is wildly uneven along the ladder, because each
    step runs at 4x the pixels of the one above it. Measured on ConvNeXt-Tiny
    at 1024x576 with a uniform ``c_dec=192``, of the decoder's 98.9 GFLOPs:

    =====================  =======  =======
    ladder step            GFLOPs   share
    =====================  =======  =======
    fuse block, stride 16      4.6    4.6%
    fuse block, stride 8      18.3   18.6%
    fuse block, stride 4      73.4   74.2%
    laterals + head            2.6    2.6%
    =====================  =======  =======

    So widening the whole ladder spends roughly three quarters of the extra
    compute on the one step where compute is dearest. A taper — wide where the
    maps are small, narrow where they are large — buys most of the parameters
    for a fraction of the FLOPs. Whether that trade helps accuracy is an
    empirical question; this only makes it askable.

    Every width must divide by the ``_ConvBlock`` GroupNorm group count (32),
    and a single int reproduces the previous module exactly — same submodules,
    same ``state_dict`` keys and shapes — so existing checkpoints load unchanged.
    """

    #: GroupNorm groups inside :class:`_ConvBlock`; every width must divide by it.
    NORM_GROUPS = 32

    def __init__(
        self,
        stage_channels: Sequence[int],
        c_dec: Union[int, Sequence[int]] = 192,
        output_stride: int = 4,
    ) -> None:
        super().__init__()
        if output_stride not in (2, 4):
            raise ValueError(f"output_stride must be 2 or 4, got {output_stride}")
        self.output_stride = output_stride

        #: Decoder width at each ladder step, coarse to fine: stride 16, 8, 4.
        self.level_widths = self._resolve_widths(c_dec)
        w16, w8, w4 = self.level_widths

        # A lateral is projected to the width of whatever consumes it: stage 3
        # seeds the ladder and stage 2 is concatenated into the first block, so
        # both feed the stride-16 step; stage 1 feeds stride 8 and stage 0
        # feeds stride 4. One uniform width collapses this to the original
        # all-``c_dec`` projection.
        lateral_widths = (w4, w8, w16, w16)
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, w, kernel_size=1)
            for c, w in zip(stage_channels, lateral_widths)
        ])
        # One fuse block per ladder step: stride 16, 8, 4. Each consumes the
        # upsampled map from the step above, concatenated with its own lateral.
        self.blocks = nn.ModuleList([
            _ConvBlock(w16 + w16, w16, groups=self.NORM_GROUPS),
            _ConvBlock(w16 + w8, w8, groups=self.NORM_GROUPS),
            _ConvBlock(w8 + w4, w4, groups=self.NORM_GROUPS),
        ])
        self.refine = (
            _ConvBlock(w4, w4, groups=self.NORM_GROUPS) if output_stride == 2 else None
        )
        self.head = nn.Conv2d(w4, 1, kernel_size=1)
        # Focal-style prior: start predicting p ~ 0.02 everywhere so the
        # dominant negatives don't swamp early training.
        nn.init.constant_(self.head.bias, -4.0)

    @classmethod
    def _resolve_widths(cls, c_dec: Union[int, Sequence[int]]) -> Tuple[int, int, int]:
        """Normalise ``c_dec`` to three per-step widths, coarse to fine."""
        # bool is an int subclass and would silently become a width of 0 or 1.
        if isinstance(c_dec, bool):
            raise TypeError("c_dec must be an int or a sequence of 3 ints, got a bool")
        if isinstance(c_dec, int):
            widths: Tuple[int, ...] = (c_dec, c_dec, c_dec)
        else:
            widths = tuple(int(w) for w in c_dec)
            if len(widths) != 3:
                raise ValueError(
                    "c_dec as a sequence gives one width per ladder step "
                    f"(stride 16, 8, 4) — expected 3, got {len(widths)}"
                )
        for w in widths:
            if w <= 0:
                raise ValueError(f"decoder widths must be positive, got {widths}")
            if w % cls.NORM_GROUPS:
                raise ValueError(
                    f"every decoder width must divide by {cls.NORM_GROUPS} "
                    f"(the _ConvBlock GroupNorm group count), got {widths}"
                )
        return widths[0], widths[1], widths[2]

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        laterals = [lat(f) for lat, f in zip(self.laterals, feats)]
        x = laterals[3]
        for skip_idx, block in zip((2, 1, 0), self.blocks):
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            x = block(torch.cat([x, laterals[skip_idx]], dim=1))
        if self.refine is not None:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            x = self.refine(x)
        return self.head(x)


class CropCounter(nn.Module):
    """Frozen DINOv3 ConvNeXt backbone + trainable pyramid decoder.

    forward(x): (B, 3, H, W) normalised RGB, H and W divisible by 32 ->
    (B, 1, H/s, W/s) logits, s = ``output_stride``.
    """

    def __init__(
        self,
        backbone_size: str = "base",
        weights_dir: Optional[Path] = None,
        c_dec: Union[int, Sequence[int]] = 192,
        output_stride: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = DinoV3Backbone(backbone_size, weights_dir)
        self.decoder = PyramidDecoder(
            self.backbone.stage_channels, c_dec=c_dec, output_stride=output_stride
        )
        self.output_stride = output_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.backbone(x))

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Decoder parameters — the only ones the optimizer should see."""
        return [p for p in self.parameters() if p.requires_grad]
