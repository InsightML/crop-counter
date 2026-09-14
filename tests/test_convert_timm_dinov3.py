"""The timm -> Meta name map, on a synthetic key list.

Only the pure string mapping is exercised here: no download, no torch.hub, no
checkpoint. The expensive end of the conversion — do the renamed tensors
actually reproduce timm's own features — is
``convert_timm_dinov3.py --verify``, whose recorded output lives in
``examples/FishDetection/notebooks/docs/results/trunk_conversion_verification.md``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples" / "FishDetection" / "scripts" / "convert_timm_dinov3.py"
)
_spec = importlib.util.spec_from_file_location("convert_timm_dinov3", _SCRIPT)
convert = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(convert)


def synthetic_timm_keys(depths=(1, 1, 2, 1)) -> list[str]:
    """A miniature but structurally complete timm ConvNeXt key list.

    Every shape the real 180/342-key lists contain appears once: the stem, the
    three inter-stage downsamples, all four per-block tensor families, and the
    ``head.norm`` pair that Meta stores twice.
    """
    keys = ["stem.0.weight", "stem.0.bias", "stem.1.weight", "stem.1.bias"]
    for stage, depth in enumerate(depths):
        if stage > 0:
            keys += [f"stages.{stage}.downsample.{i}.{p}" for i in (0, 1) for p in ("weight", "bias")]
        for block in range(depth):
            prefix = f"stages.{stage}.blocks.{block}"
            keys.append(f"{prefix}.gamma")
            keys += [f"{prefix}.conv_dw.{p}" for p in ("weight", "bias")]
            keys += [f"{prefix}.norm.{p}" for p in ("weight", "bias")]
            keys += [f"{prefix}.mlp.fc1.{p}" for p in ("weight", "bias")]
            keys += [f"{prefix}.mlp.fc2.{p}" for p in ("weight", "bias")]
    keys += ["head.norm.weight", "head.norm.bias"]
    return keys


@pytest.mark.parametrize(
    ("timm_key", "expected"),
    [
        ("stem.0.weight", ["downsample_layers.0.0.weight"]),
        ("stem.1.bias", ["downsample_layers.0.1.bias"]),
        ("stages.1.downsample.0.weight", ["downsample_layers.1.0.weight"]),
        ("stages.3.downsample.1.bias", ["downsample_layers.3.1.bias"]),
        ("stages.0.blocks.0.conv_dw.weight", ["stages.0.0.dwconv.weight"]),
        ("stages.2.blocks.11.conv_dw.bias", ["stages.2.11.dwconv.bias"]),
        ("stages.2.blocks.5.mlp.fc1.weight", ["stages.2.5.pwconv1.weight"]),
        ("stages.2.blocks.5.mlp.fc2.bias", ["stages.2.5.pwconv2.bias"]),
        ("stages.3.blocks.2.norm.weight", ["stages.3.2.norm.weight"]),
        ("stages.3.blocks.2.gamma", ["stages.3.2.gamma"]),
    ],
)
def test_map_timm_key_one_to_one(timm_key, expected):
    assert convert.map_timm_key(timm_key) == expected


def test_head_norm_lands_on_both_meta_names():
    """Meta aliases its final LayerNorm as ``norm`` and ``norms.3``.

    That duplication is the whole reason Meta's state dict has two more
    tensors than timm's, so it is asserted directly rather than inferred from
    a count.
    """
    assert convert.map_timm_key("head.norm.weight") == ["norm.weight", "norms.3.weight"]
    assert convert.map_timm_key("head.norm.bias") == ["norm.bias", "norms.3.bias"]


def test_full_key_list_maps_totally_and_gains_exactly_two():
    keys = synthetic_timm_keys()
    mapping = convert.map_timm_keys(keys)
    assert set(mapping) == set(keys)  # nothing silently dropped
    meta_keys = [target for targets in mapping.values() for target in targets]
    assert len(meta_keys) == len(set(meta_keys))  # no collisions
    assert len(meta_keys) == len(keys) + 2  # exactly the norm alias, nothing else


def test_real_shaped_counts():
    """The synthetic list at the real depths reproduces the real key counts."""
    tiny = convert.map_timm_keys(synthetic_timm_keys((3, 3, 9, 3)))
    assert len(tiny) == 180
    assert sum(len(v) for v in tiny.values()) == 182
    base = convert.map_timm_keys(synthetic_timm_keys((3, 3, 27, 3)))
    assert len(base) == 342
    assert sum(len(v) for v in base.values()) == 344


def test_every_mapped_name_is_a_meta_name():
    """No mapped key may keep a timm-ism — that would fail the strict load."""
    forbidden = ("stem.", ".blocks.", "conv_dw", "mlp.fc", "head.")
    for targets in convert.map_timm_keys(synthetic_timm_keys()).values():
        for target in targets:
            assert not any(token in target for token in forbidden), target


@pytest.mark.parametrize(
    "bad_key",
    [
        "head.fc.weight",              # a classifier head we must not silently drop
        "stages.0.downsample.0.weight",  # would collide with the stem
        "stages.0.blocks.0.mlp.fc3.weight",
        "stages.0.blocks.0.unknown.weight",
        "norm_pre.weight",
        "stages.x.blocks.0.gamma",
        "stages.0.blocks.0",
    ],
)
def test_unmapped_keys_raise(bad_key):
    with pytest.raises(KeyError):
        convert.map_timm_key(bad_key)


def test_collision_raises():
    """Two timm keys claiming one Meta key must be an error, not a last-write-wins."""
    with pytest.raises(ValueError, match="claimed by both"):
        convert.map_timm_keys(["head.norm.weight", "head.norm.weight"])


def test_convert_state_dict_shares_the_norm_tensor():
    import torch

    timm_sd = {
        "stem.0.weight": torch.zeros(2, 3, 4, 4),
        "head.norm.weight": torch.ones(2),
        "head.norm.bias": torch.zeros(2),
    }
    meta_sd = convert.convert_state_dict(timm_sd)
    assert set(meta_sd) == {
        "downsample_layers.0.0.weight", "norm.weight", "norm.bias",
        "norms.3.weight", "norms.3.bias",
    }
    assert meta_sd["norm.weight"] is meta_sd["norms.3.weight"]
