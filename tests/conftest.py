"""Shared pytest fixtures.

The only one here is :func:`stub_backbone`, which lets a test build a real
:class:`~cropcounter.dinov3_pyramid.CropCounter` without the 354 MB DINOv3
checkpoint, a network round-trip to the torch hub, or a GPU.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from cropcounter import dinov3_pyramid
from cropcounter.dinov3_pyramid import STAGE_CHANNELS


class _StubBlock(nn.Module):
    """One ConvNeXt-ish residual block: a depthwise conv scaled by a 1-D gamma.

    ``gamma`` stands in for the real trunk's LayerScale parameter. It matters
    for two reasons: it is the 1-D tensor that must land in a *no weight decay*
    parameter group, and its name (``stages.2.0.gamma``) is the shape the depth
    mapping has to parse.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.gamma = nn.Parameter(torch.full((channels,), 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma.view(1, -1, 1, 1) * self.dwconv(x)


def _downsample(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
    """A stride-reducing conv + norm pair, named ``.0`` / ``.1`` like the hub's.

    Grouped so the widest stage costs a few hundred parameters rather than
    millions: the stub only has to have the hub's module *names* and stage
    *shapes*, never its capacity.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 1, stride=stride, groups=math.gcd(in_ch, out_ch)),
        nn.GroupNorm(1, out_ch),
    )


class _StubConvNeXt(nn.Module):
    """A ~30k-parameter stand-in for the DINOv3 ConvNeXt-Base hub trunk.

    It copies the three things the library actually depends on: the child
    module NAMES (``downsample_layers``, ``stages``, ``norm``, and the
    ``norms`` ModuleList whose last entry is the *same object* as ``norm``),
    the four stage widths at strides 4/8/16/32, and the
    ``get_intermediate_layers`` entry point. Every parameter is used in the
    forward pass, so "did the whole trunk get a gradient?" is a real question
    to ask of it.
    """

    def __init__(self, channels: tuple = STAGE_CHANNELS["base"]) -> None:
        super().__init__()
        self.downsample_layers = nn.ModuleList(
            [_downsample(3, channels[0], 4)]
            + [_downsample(channels[i - 1], channels[i], 2) for i in range(1, 4)]
        )
        self.stages = nn.ModuleList(
            [nn.Sequential(_StubBlock(c)) for c in channels]
        )
        self.norm = nn.LayerNorm(channels[-1])
        # The hub model aliases its final norm into a ModuleList as well, so a
        # naive module walk counts it twice. Reproduced here on purpose.
        self.norms = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        self.norms.append(self.norm)

    def get_intermediate_layers(self, x, n=None, reshape: bool = True):
        """The four stage maps (NCHW) at strides 4/8/16/32, finest first."""
        feats = []
        for down, stage in zip(self.downsample_layers, self.stages):
            x = stage(down(x))
            feats.append(x)
        # The real trunk's final LayerNorm is channels-last; permute around it.
        feats[-1] = self.norms[3](feats[-1].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return tuple(feats)


@pytest.fixture
def stub_backbone(monkeypatch):
    """Make ``DinoV3Backbone`` build a :class:`_StubConvNeXt` instead of the hub.

    Patches the two outward-facing calls in
    ``cropcounter.dinov3_pyramid``: the weights resolver (so no gated 354 MB
    ``.pth`` has to exist) and ``torch.hub.load`` (so no GitHub fetch happens).
    Yields the stub class, for tests that want to build one directly.
    """
    monkeypatch.setattr(
        dinov3_pyramid, "resolve_backbone_weights",
        lambda size, weights_dir=None: Path("stub-weights.pth"),
    )
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: _StubConvNeXt())
    return _StubConvNeXt
