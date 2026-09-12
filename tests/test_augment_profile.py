"""Unit tests for the ``augment_profile`` training-augmentation knob.

The point path trains on natural (underwater) imagery, which has an up:
``VerticalFlip`` and ``RandomRotate90`` are label-preserving on a nadir crop
photo but not on a reef. ``"wheat"`` must stay byte-for-byte today's pipeline
(the class sequence below was captured from the pre-change ``default_transform``),
``"natural"`` drops exactly those two. CPU-light, no dataset, no model, no
network -- pure pipeline construction and config coercion, matching
``test_train_config.py``.
"""
from __future__ import annotations

import pytest

from cropcounter.crop_dataset import CropTileDataset
from cropcounter.train import TrainConfig

#: The pre-change default pipeline, captured before ``augment_profile`` existed.
WHEAT_TRANSFORMS = [
    "RandomScale",
    "RandomCrop",
    "HorizontalFlip",
    "VerticalFlip",
    "RandomRotate90",
    "RandomBrightnessContrast",
    "HueSaturationValue",
    "Normalize",
    "ToTensorV2",
]
#: Same list, minus the two orientation-destroying geometric transforms.
NATURAL_TRANSFORMS = [t for t in WHEAT_TRANSFORMS if t not in ("VerticalFlip", "RandomRotate90")]


def _names(compose):
    return [type(t).__name__ for t in compose.transforms]


def test_default_profile_is_unchanged_wheat_pipeline():
    """No-argument and explicit-wheat calls both rebuild the pre-change pipeline."""
    implicit = CropTileDataset.default_transform()
    explicit = CropTileDataset.default_transform(profile="wheat")
    assert _names(implicit) == WHEAT_TRANSFORMS
    assert repr(implicit) == repr(explicit)
    # Params the wheat recipe depends on, byte-for-byte.
    by_name = dict(zip(_names(implicit), implicit.transforms))
    assert by_name["HorizontalFlip"].p == 0.5
    assert by_name["VerticalFlip"].p == 0.5
    assert by_name["RandomRotate90"].p == 0.75
    assert by_name["RandomCrop"].height == 768
    assert by_name["RandomCrop"].width == 768
    assert implicit.processors["keypoints"].params.format == "xy"
    assert implicit.processors["keypoints"].params.remove_invisible is True


def test_natural_profile_drops_vertical_flip_and_rotate90():
    """"natural" = the wheat list minus VerticalFlip/RandomRotate90, same order."""
    names = _names(CropTileDataset.default_transform(profile="natural"))
    assert "VerticalFlip" not in names
    assert "RandomRotate90" not in names
    assert names == NATURAL_TRANSFORMS


def test_unknown_profile_raises_value_error():
    """A typo'd profile fails loudly, naming the valid profiles."""
    with pytest.raises(ValueError) as excinfo:
        CropTileDataset.default_transform(profile="underwater")
    message = str(excinfo.value)
    assert "wheat" in message and "natural" in message


def test_train_config_augment_profile_round_trip(capsys):
    """Absent key -> "wheat"; explicit "natural" survives; neither warns."""
    assert TrainConfig().augment_profile == "wheat"
    assert TrainConfig.from_dict({"epochs": 5}).augment_profile == "wheat"
    assert TrainConfig.from_dict({"augment_profile": "natural"}).augment_profile == "natural"
    assert "ignoring unknown field" not in capsys.readouterr().out
    restored = TrainConfig.from_dict(TrainConfig(augment_profile="natural").to_dict())
    assert restored.augment_profile == "natural"
