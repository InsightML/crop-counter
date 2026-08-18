"""Frozen DINOv3 ConvNeXt backbone + pyramid decoder for peak-heatmap counting.

The backbone stays frozen and permanently in eval mode; only the decoder
trains. The model outputs single-channel logits at ``output_stride`` (4 by
default) — apply sigmoid + ``heatmap.decode_peaks`` to get points.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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
    """

    def __init__(
        self,
        stage_channels: Sequence[int],
        c_dec: int = 192,
        output_stride: int = 4,
    ) -> None:
        super().__init__()
        if output_stride not in (2, 4):
            raise ValueError(f"output_stride must be 2 or 4, got {output_stride}")
        self.output_stride = output_stride

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
        c_dec: int = 192,
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
