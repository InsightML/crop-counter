"""Unit tests for the ``val_freq`` validation-frequency knob.

Covers the pure ``should_validate`` schedule and the ``TrainConfig.val_freq``
round-trip. CPU-light, no dataset, no model, no network -- pure logic and
config coercion, matching ``test_train_config.py``.
"""
from __future__ import annotations

import math

import pytest

from cropcounter.train import HISTORY_KEYS, TrainConfig, _append_history, should_validate


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


# --------------------------------------------------------------------------- #
# The skipped-epoch history row, for both task key sets
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("task", ["point", "box"])
def test_append_history_pads_a_skipped_epoch_with_nan(task):
    """summary=None NaN-pads every val series, so history.json stays rectangular.

    plot_history masks the NaNs out; json.dump renders them as ``NaN``. The box
    key set has to pad too, or a val_freq > 1 box run desynchronises its arrays
    from the epoch index.
    """
    history = {key: [] for key in HISTORY_KEYS[task]}
    _append_history(history, task, train_loss=1.5, lr=1e-3)

    assert history["train_loss"] == [1.5] and history["lr"] == [1e-3]
    val_keys = [k for k, v in HISTORY_KEYS[task].items() if v]
    assert val_keys, "every task has at least one val series"
    for key in val_keys:
        assert len(history[key]) == 1 and math.isnan(history[key][0])
    assert len({len(v) for v in history.values()}) == 1, "history must stay rectangular"


@pytest.mark.parametrize("task", ["point", "box"])
def test_append_history_with_a_summary_is_unchanged(task):
    """The validated-epoch path still copies the summary straight through."""
    summary = {v: 0.25 for v in HISTORY_KEYS[task].values() if v}
    history = {key: [] for key in HISTORY_KEYS[task]}
    _append_history(history, task, train_loss=2.0, lr=5e-4, summary=summary)

    assert history["train_loss"] == [2.0]
    for key, summary_key in HISTORY_KEYS[task].items():
        if summary_key:
            assert history[key] == [0.25]
