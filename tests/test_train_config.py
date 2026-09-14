"""Unit tests for cropcounter.train.TrainConfig (de)serialisation.

Focused on the legacy ``labels`` field (deprecated since 0.2.0 in favour of
``classes``): it must still default to the shipped wheat labels, survive a
JSON round-trip as a tuple, and honour an explicit ``None`` (keep every
label) -- while warning, and while ``to_dict`` writes the ``classes`` form.
CPU-light, no dataset, no network -- pure config coercion. The ``classes``
field itself is covered in ``test_multiclass.py``.
"""
from __future__ import annotations

import pytest

from cropcounter.crop_dataset import COUNTED_LABELS
from cropcounter.train import TrainConfig


def test_labels_default_is_counted_labels():
    """A default config counts the shipped wheat labels."""
    assert TrainConfig().labels == COUNTED_LABELS


def test_labels_absent_from_json_uses_default():
    """An old config with no ``labels`` key still loads with the wheat default."""
    cfg = TrainConfig.from_dict({"epochs": 5})
    assert cfg.labels == COUNTED_LABELS


def test_labels_json_list_coerced_to_tuple():
    """JSON stores labels as a list; from_dict must coerce it back to a tuple."""
    with pytest.warns(DeprecationWarning):
        cfg = TrainConfig.from_dict({"labels": ["Tassel"]})
    assert cfg.labels == ("Tassel",)
    assert isinstance(cfg.labels, tuple)


def test_labels_none_kept():
    """``labels: null`` means keep every annotated label; it must stay None."""
    with pytest.warns(DeprecationWarning):
        cfg = TrainConfig.from_dict({"labels": None})
    assert cfg.labels is None


def test_labels_round_trip_through_dict():
    """to_dict -> from_dict preserves a custom single-class label set.

    Since 0.2.0 the dict carries the canonical ``classes`` form instead of
    echoing the deprecated ``labels`` key, and re-loading it no longer warns.
    """
    with pytest.warns(DeprecationWarning):
        original = TrainConfig(labels=("Tassel",))
    payload = original.to_dict()
    assert "labels" not in payload
    assert payload["classes"] == {"Tassel": ["Tassel"]}
    restored = TrainConfig.from_dict(payload)
    assert restored.labels == ("Tassel",)
    assert restored.classes == {"Tassel": ("Tassel",)}
