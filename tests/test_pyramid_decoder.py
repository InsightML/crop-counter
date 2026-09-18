"""PyramidDecoder width handling: uniform, tapered, and what must not change.

The decoder is the only trainable part of the model, so its ``state_dict``
shape is a compatibility surface: every checkpoint ever trained has to keep
loading. The first test here is the one that matters — a single int must build
exactly the module it always did.

These run on CPU with synthetic feature maps; no backbone and no weights.
"""
from __future__ import annotations

import pytest
import torch

from cropcounter.dinov3_pyramid import STAGE_CHANNELS, PyramidDecoder

TINY = STAGE_CHANNELS["tiny"]
BASE = STAGE_CHANNELS["base"]


def feats(stage_channels, height=64, width=64, batch=1):
    """Four synthetic stage maps at strides 4, 8, 16, 32."""
    return [
        torch.randn(batch, c, height // s, width // s)
        for c, s in zip(stage_channels, (4, 8, 16, 32))
    ]


def test_int_width_is_uniform_across_the_ladder():
    dec = PyramidDecoder(TINY, c_dec=192)
    assert dec.level_widths == (192, 192, 192)


def test_int_and_equivalent_sequence_build_identical_modules():
    """The compatibility guarantee: c_dec=192 and [192,192,192] are one module.

    Same keys, same shapes — so a checkpoint trained under either loads under
    the other, and old checkpoints keep working after this change.
    """
    uniform = PyramidDecoder(TINY, c_dec=192).state_dict()
    explicit = PyramidDecoder(TINY, c_dec=[192, 192, 192]).state_dict()
    assert uniform.keys() == explicit.keys()
    for key in uniform:
        assert uniform[key].shape == explicit[key].shape, key


def test_uniform_decoder_state_dict_is_unchanged_by_this_feature():
    """Pin the pre-change parameter shapes for the default configuration.

    Hard-coded rather than compared against a rebuilt module, so the test still
    fails if a later refactor silently changes the default architecture.
    """
    dec = PyramidDecoder(TINY, c_dec=192, output_stride=4)
    shapes = {k: tuple(v.shape) for k, v in dec.state_dict().items()}
    # Laterals project each stage (96, 192, 384, 768) to 192 channels.
    assert shapes["laterals.0.weight"] == (192, 96, 1, 1)
    assert shapes["laterals.1.weight"] == (192, 192, 1, 1)
    assert shapes["laterals.2.weight"] == (192, 384, 1, 1)
    assert shapes["laterals.3.weight"] == (192, 768, 1, 1)
    # Every fuse block takes 2*192 in, emits 192.
    for i in range(3):
        assert shapes[f"blocks.{i}.0.weight"] == (192, 384, 3, 3)
        assert shapes[f"blocks.{i}.3.weight"] == (192, 192, 3, 3)
    assert shapes["head.weight"] == (1, 192, 1, 1)
    assert "refine.0.weight" not in shapes


def test_tapered_widths_size_each_level_independently():
    dec = PyramidDecoder(TINY, c_dec=[384, 256, 128])
    assert dec.level_widths == (384, 256, 128)
    shapes = {k: tuple(v.shape) for k, v in dec.state_dict().items()}
    # Laterals follow whatever consumes them: stage 0 -> stride 4 (128),
    # stage 1 -> stride 8 (256), stages 2 and 3 -> stride 16 (384).
    assert shapes["laterals.0.weight"] == (128, 96, 1, 1)
    assert shapes["laterals.1.weight"] == (256, 192, 1, 1)
    assert shapes["laterals.2.weight"] == (384, 384, 1, 1)
    assert shapes["laterals.3.weight"] == (384, 768, 1, 1)
    # Block inputs are "upsampled map from above" + "own lateral".
    assert shapes["blocks.0.0.weight"] == (384, 384 + 384, 3, 3)
    assert shapes["blocks.1.0.weight"] == (256, 384 + 256, 3, 3)
    assert shapes["blocks.2.0.weight"] == (128, 256 + 128, 3, 3)
    assert shapes["head.weight"] == (1, 128, 1, 1)


@pytest.mark.parametrize("c_dec", [192, [384, 256, 128], [128, 128, 320]])
@pytest.mark.parametrize("output_stride", [4, 2])
def test_forward_shape_matches_output_stride(c_dec, output_stride):
    dec = PyramidDecoder(TINY, c_dec=c_dec, output_stride=output_stride)
    out = dec(feats(TINY, 64, 96))
    assert out.shape == (1, 1, 64 // output_stride, 96 // output_stride)


def test_refine_block_follows_the_finest_width():
    dec = PyramidDecoder(TINY, c_dec=[384, 256, 128], output_stride=2)
    shapes = {k: tuple(v.shape) for k, v in dec.state_dict().items()}
    assert shapes["refine.0.weight"] == (128, 128, 3, 3)


def test_tapering_moves_parameters_off_the_expensive_level():
    """A taper should buy parameters without buying the stride-4 block's cost.

    Not a FLOPs assertion — just that the wide-at-the-top arrangement has more
    parameters than uniform while its finest (and by far most expensive) block
    stays no wider.
    """
    uniform = PyramidDecoder(TINY, c_dec=192)
    tapered = PyramidDecoder(TINY, c_dec=[384, 256, 192])
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert n(tapered) > n(uniform)
    assert tapered.level_widths[2] == uniform.level_widths[2]


def test_head_bias_keeps_the_focal_prior():
    for c_dec in (192, [384, 256, 128]):
        dec = PyramidDecoder(TINY, c_dec=c_dec)
        assert torch.allclose(dec.head.bias, torch.full_like(dec.head.bias, -4.0))


def test_widths_work_for_every_backbone_size():
    for stage_channels in STAGE_CHANNELS.values():
        dec = PyramidDecoder(stage_channels, c_dec=[256, 192, 128])
        out = dec(feats(stage_channels, 32, 32))
        assert out.shape == (1, 1, 8, 8)


@pytest.mark.parametrize(
    "bad, match",
    [
        ([192, 192], "expected 3"),
        ([192, 192, 192, 192], "expected 3"),
        ([192, 192, 100], "divide by 32"),
        (100, "divide by 32"),
        ([192, 0, 192], "positive"),
        ([192, -64, 192], "positive"),
    ],
)
def test_invalid_widths_are_rejected(bad, match):
    with pytest.raises(ValueError, match=match):
        PyramidDecoder(TINY, c_dec=bad)


def test_bool_is_not_accepted_as_a_width():
    """``bool`` subclasses ``int``; without a guard True would mean width 1."""
    with pytest.raises(TypeError, match="bool"):
        PyramidDecoder(TINY, c_dec=True)


def test_output_stride_is_still_validated():
    with pytest.raises(ValueError, match="output_stride must be 2 or 4"):
        PyramidDecoder(TINY, c_dec=192, output_stride=8)


def test_gradients_reach_every_level_of_a_tapered_ladder():
    dec = PyramidDecoder(BASE, c_dec=[256, 192, 128])
    dec(feats(BASE, 32, 32)).sum().backward()
    for name, p in dec.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
