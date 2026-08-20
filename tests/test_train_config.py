"""Unit tests for cropcounter.train.TrainConfig (de)serialisation.

Focused on the ``labels`` field added for non-wheat datasets: it must default
to the shipped wheat labels (so old configs keep working), survive a
JSON round-trip as a tuple, and honour an explicit ``None`` (keep every
label). CPU-light, no dataset, no network -- pure config coercion.
"""
from __future__ import annotations

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
    cfg = TrainConfig.from_dict({"labels": ["Tassel"]})
    assert cfg.labels == ("Tassel",)
    assert isinstance(cfg.labels, tuple)


def test_labels_none_kept():
    """``labels: null`` means keep every annotated label; it must stay None."""
    cfg = TrainConfig.from_dict({"labels": None})
    assert cfg.labels is None


def test_labels_round_trip_through_dict():
    """to_dict -> from_dict preserves a custom single-class label set."""
    original = TrainConfig(labels=("Tassel",))
    restored = TrainConfig.from_dict(original.to_dict())
    assert restored.labels == ("Tassel",)
    # to_dict is JSON-ready, so the tuple is emitted as a list.
    assert original.to_dict()["labels"] == ["Tassel"]
