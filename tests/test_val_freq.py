"""Unit tests for the ``val_freq`` validation-frequency knob.

Covers the pure ``should_validate`` schedule and the ``TrainConfig.val_freq``
round-trip. CPU-light, no dataset, no model, no network -- pure logic and
config coercion, matching ``test_train_config.py``.
"""
from __future__ import annotations

from cropcounter.train import TrainConfig, should_validate


def test_val_freq_default_is_one():
    """A default config validates every epoch."""
    assert TrainConfig().val_freq == 1


def test_val_freq_one_validates_every_epoch():
    """val_freq=1 preserves today's behaviour: validate every epoch."""
    assert all(should_validate(e, 10, 1) for e in range(1, 11))


def test_val_freq_schedule_hits_multiples_plus_first_and_last():
    """val_freq=5 validates epoch 1, every 5th, and the final epoch."""
    validated = [e for e in range(1, 101) if should_validate(e, 100, 5)]
    assert validated[0] == 1  # baseline
    assert validated[-1] == 100  # final epoch
    assert all(e in validated for e in (5, 10, 50, 95, 100))
    assert all(e not in validated for e in (2, 3, 4, 6, 99))


def test_val_freq_final_epoch_always_validates_off_multiple():
    """The final epoch validates even when it is not a val_freq multiple."""
    assert should_validate(100, 100, 7)  # 100 % 7 != 0
    assert not should_validate(99, 100, 7)


def test_val_freq_zero_treated_as_one():
    """A 0 (or negative) val_freq must not crash and validates every epoch."""
    assert all(should_validate(e, 5, 0) for e in range(1, 6))
    assert all(should_validate(e, 5, -3) for e in range(1, 6))


def test_val_freq_round_trip_through_dict():
    """to_dict -> from_dict preserves a custom val_freq."""
    restored = TrainConfig.from_dict(TrainConfig(val_freq=5).to_dict())
    assert restored.val_freq == 5
    assert TrainConfig(val_freq=5).to_dict()["val_freq"] == 5


def test_val_freq_absent_from_json_uses_default():
    """An old config with no val_freq key still loads with the default 1."""
    assert TrainConfig.from_dict({"epochs": 5}).val_freq == 1
