"""The run monitor, tested against REAL log lines.

The fixture in ``tests/fixtures/run_all_frozen_excerpt.log`` is copied verbatim
out of the 66 MB ``run_all.log`` the frozen CFD-17 run actually produced, so the
parser is pinned to the format tqdm and ``train.py`` really emit rather than to
a format invented here.

The numbers that matter are known independently: that run measured **25.6
min/epoch** training and **12.5 min** per validation pass (recorded in the epic
and in `claude-memory/reference_cfd_benchmark_infra.md`). A monitor whose whole
job is re-costing has to reproduce them.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "examples" / "FishDetection" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import watch_run as wr  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "run_all_frozen_excerpt.log"
REAL = FIXTURE.read_text()

#: A point run's epoch line — the other format train.py emits.
POINT_EPOCH = (
    "epoch   1 | loss 3.3697 val 3.8464 | val MAE 2.18 RMSE 4.71 bias -2.17 | "
    "P 0.812 R 0.034 F1 0.065 | lr 2.00e-03  <- best"
)


# --- parsing ---------------------------------------------------------------- #

def test_parses_the_real_log_excerpt():
    state = wr.parse_log(REAL)
    assert state.run_name == "cfd17_frozen_s0"
    assert state.backbone == "base"
    assert state.decoder_params_m == 3.58
    assert state.select_on == "ap50"
    assert state.started_at == datetime(2026, 9, 13, 16, 1, 55, tzinfo=timezone.utc)
    assert [e.epoch for e in state.epochs] == [1, 2]


def test_box_epoch_metrics_come_through():
    epoch = wr.parse_log(REAL).epochs[0]
    assert epoch.train_loss == 1.8458
    assert epoch.val_loss == 1.4914
    assert epoch.ap50 == 0.615
    assert epoch.f1 == 0.628
    assert epoch.mae == 0.70
    assert epoch.best is True


def test_point_epoch_line_parses_and_has_no_ap50():
    state = wr.parse_log(POINT_EPOCH)
    epoch = state.epochs[0]
    assert epoch.f1 == 0.065
    assert epoch.mae == 2.18
    assert epoch.ap50 is None
    assert state.is_point_run, "no AP50 is what identifies a point run"


def test_a_box_run_is_not_mistaken_for_a_point_run():
    assert not wr.parse_log(REAL).is_point_run


def test_both_phases_of_bar_are_remembered_separately():
    """A poll landing in validation must still know the training rate."""
    state = wr.parse_log(REAL)
    assert state.progress.phase == "epoch" and state.progress.epoch == 2
    assert state.last_train.total == 20238
    assert state.last_val.total == 36632


def test_seconds_per_iteration_bars_are_inverted():
    state = wr.parse_log("epoch 1/8:   1%|          | 1/126 [00:04<10:21,  4.97s/it]")
    assert state.progress.rate_it_s == pytest.approx(1 / 4.97)


@pytest.mark.parametrize("text, expected", [("25:25", 1525), ("1:02:03", 3723), ("00:07", 7)])
def test_parse_hms(text, expected):
    assert wr.parse_hms(text) == expected


def test_unknown_eta_is_none():
    assert wr.parse_hms("?") is None


# --- the measurement that the whole tool exists for ------------------------- #

def test_full_phase_reproduces_the_measured_epoch_time():
    """25.5 min from the bar against 25.6 min actually measured."""
    train = wr.parse_log(REAL).last_train
    assert train.full_phase_s / 60 == pytest.approx(25.6, abs=0.2)


def test_full_phase_reproduces_the_measured_validation_time():
    """12.4 min from the bar against 12.5 min actually measured."""
    val = wr.parse_log(REAL).last_val
    assert val.full_phase_s / 60 == pytest.approx(12.5, abs=0.2)


def test_elapsed_extrapolation_beats_the_smoothed_rate_on_a_near_complete_pass():
    """tqdm's rate is a recent-window average; elapsed/fraction is the truth.

    The gap only opens once a pass is well under way, because the rate has by
    then forgotten the slow early steps. On the REAL epoch-1 bar (99.5% done,
    25:25 elapsed, 13.43 it/s) the rate says 25.1 min and elapsed says 25.5 —
    against 25.6 actually measured.

    Early in a pass the two agree, which is why this test uses the epoch-1 bar
    specifically rather than whichever bar happens to be last.
    """
    epoch_one_bar = next(l for l in REAL.splitlines() if l.startswith("epoch 1/8: 100%"))
    train = wr.parse_log(epoch_one_bar).last_train
    from_rate = train.total / train.rate_it_s / 60
    from_elapsed = train.full_phase_s / 60
    assert from_rate == pytest.approx(25.1, abs=0.1)
    assert from_elapsed == pytest.approx(25.5, abs=0.1)
    assert abs(from_elapsed - 25.6) < abs(from_rate - 25.6)


def test_early_in_a_pass_the_two_methods_agree():
    """The counterpart: no false confidence that one is always better."""
    train = wr.parse_log(REAL).last_train          # epoch 2, 10% in
    from_rate = train.total / train.rate_it_s / 60
    assert from_rate == pytest.approx(train.full_phase_s / 60, abs=0.2)


# --- projection ------------------------------------------------------------- #

def _projection(text=REAL, val_freq=1, minutes_in=78):
    state = wr.parse_log(text)
    now = state.started_at.replace(tzinfo=timezone.utc)
    now = now + (datetime(1, 1, 1, 1, 18) - datetime(1, 1, 1, 0, 0)) if minutes_in == 78 else now
    return state, wr.project(state, val_freq=val_freq, usd_per_hour=2.55, now=now)


def test_projection_reproduces_the_runs_own_arithmetic():
    """At 10% into epoch 2 of 8: 6.9 epochs train + 7 val passes still to come."""
    state, projection = _projection()
    assert projection.train_min_per_epoch == pytest.approx(25.6, abs=0.2)
    assert projection.val_min_per_pass == pytest.approx(12.5, abs=0.2)
    # 6.9 * 25.55 + 7 * 12.36  ~= 176 + 87 = 263 min
    assert projection.remaining_min == pytest.approx(263, abs=8)
    assert projection.total_min == pytest.approx(78 + 263, abs=8)


def test_val_freq_two_halves_the_validation_passes():
    _s, every = _projection(val_freq=1)
    _s, other = _projection(val_freq=2)
    assert other.remaining_min < every.remaining_min
    # 7 passes vs 4 (epochs 2,4,6,8), a saving of ~3 passes.
    assert every.remaining_min - other.remaining_min == pytest.approx(3 * 12.4, abs=2)


def test_cost_follows_the_hourly_rate():
    _s, projection = _projection()
    assert projection.projected_usd == pytest.approx(projection.total_min / 60 * 2.55, rel=1e-6)
    assert projection.on_demand_usd > projection.projected_usd


def test_a_poll_during_validation_still_projects():
    """The regression that made the tool useless a third of the time."""
    during_val = REAL[: REAL.index("epoch 2/8:")]
    state = wr.parse_log(during_val)
    assert state.progress.phase == "val"
    projection = wr.project(state, val_freq=1, usd_per_hour=2.55, now=state.started_at)
    assert projection.train_min_per_epoch == pytest.approx(25.6, abs=0.2)
    assert projection.remaining_min is not None


def test_no_training_bar_yet_is_reported_not_guessed():
    state = wr.parse_log("start 2026-09-13T16:01:55Z\n[run ] fetch\n")
    projection = wr.project(state, val_freq=2, usd_per_hour=2.55)
    assert projection.train_min_per_epoch is None
    assert projection.projected_usd is None


def test_a_relaunch_resets_the_previous_attempt():
    """The real log holds a FAILED attempt before the run that finished."""
    text = REAL + "\nFAILED rc=1 wall=2135s (35m) — see above\nstart 2026-09-13T16:53:36Z\n"
    state = wr.parse_log(text)
    assert state.finished is None
    assert state.epochs == []
    assert state.started_at == datetime(2026, 9, 13, 16, 53, 36, tzinfo=timezone.utc)


def test_all_done_and_failed_are_read():
    done = wr.parse_log("ALL DONE wall=59607s (993m) runs=/x results=/y")
    assert done.finished == "ALL DONE" and done.wall_s == 59607
    failed = wr.parse_log("FAILED rc=1 wall=2135s (35m) — see above")
    assert failed.finished == "FAILED" and failed.rc == 1


# --- alarms ----------------------------------------------------------------- #

def test_a_healthy_run_raises_nothing():
    state, projection = _projection()
    assert wr.alarms(state, projection, budget_usd=100.0) == []


def test_over_budget_is_raised():
    state, projection = _projection()
    raised = wr.alarms(state, projection, budget_usd=1.0)
    assert any("exceeds" in a for a in raised)


def test_a_point_run_stuck_at_zero_f1_is_raised():
    """The degenerate-selection case: scores below the fixed tau."""
    text = "\n".join([
        "epoch   1 | loss 3.0 val 3.8 | val MAE 2.1 | P 0.0 R 0.0 F1 0.000 | lr 2e-03",
        "epoch   2 | loss 2.9 val 2.5 | val MAE 2.2 | P 0.0 R 0.0 F1 0.000 | lr 1e-03",
    ])
    state = wr.parse_log(text)
    raised = wr.alarms(state, wr.Projection(), budget_usd=19.0)
    assert any("F1 is 0.000" in a and "lower it" in a for a in raised)


def test_a_point_run_that_has_started_scoring_is_not_raised():
    text = "\n".join([
        "epoch   1 | loss 3.0 val 3.8 | val MAE 2.1 | P 0.0 R 0.0 F1 0.000 | lr 2e-03",
        "epoch   2 | loss 2.9 val 2.5 | val MAE 2.2 | P 0.5 R 0.3 F1 0.375 | lr 1e-03",
    ])
    state = wr.parse_log(text)
    assert not any("F1 is 0.000" in a
                   for a in wr.alarms(state, wr.Projection(), budget_usd=19.0))


def test_a_failed_run_is_raised():
    state = wr.parse_log("FAILED rc=1 wall=2135s (35m)")
    assert any("FAILED" in a for a in wr.alarms(state, wr.Projection(), budget_usd=19.0))


# --- windowing + CLI -------------------------------------------------------- #

def test_local_window_keeps_both_ends(tmp_path):
    """A tail alone loses `start`; the head window is what brings it back."""
    path = tmp_path / "big.log"
    path.write_text("start 2026-09-13T16:01:55Z\n" + ("x" * 5000) + "\nALL DONE wall=10s\n")
    text = wr.read_local_window(path, head_bytes=64, tail_bytes=64)
    assert "start 2026-09-13" in text
    assert "ALL DONE" in text
    assert wr.WINDOW_JOIN in text


def test_a_small_file_is_not_windowed(tmp_path):
    path = tmp_path / "small.log"
    path.write_text(REAL)
    assert wr.read_local_window(path, 1_000_000, 1_000_000) == REAL


def test_cli_reports_and_exits_zero_on_a_healthy_run(tmp_path, capsys):
    log = tmp_path / "run_all.log"
    log.write_text(REAL)
    code = wr.main(["--log", str(log), "--val-freq", "1", "--budget", "100",
                    "--now", "2026-09-13T17:20:00Z"])
    out = capsys.readouterr().out
    assert code == 0
    assert "cfd17_frozen_s0" in out
    assert "25.6 min/epoch" in out
    assert "nothing to act on" in out


def test_cli_exits_one_when_an_alarm_fires(tmp_path, capsys):
    log = tmp_path / "run_all.log"
    log.write_text(REAL)
    code = wr.main(["--log", str(log), "--val-freq", "1", "--budget", "1",
                    "--now", "2026-09-13T17:20:00Z"])
    assert code == 1
    assert "!!" in capsys.readouterr().out


def test_cli_exits_two_on_a_failed_run(tmp_path):
    log = tmp_path / "run_all.log"
    log.write_text("start 2026-09-13T16:01:55Z\nFAILED rc=1 wall=2135s (35m)\n")
    assert wr.main(["--log", str(log)]) == 2


def test_a_relaunch_also_forgets_the_previous_attempts_throughput():
    """Found live: a second `start` in the window left a rate with no projection.

    The S3 log is appended across launches, so a poll's window really can hold
    one attempt's bars and a later attempt's `start`. Measurements from the old
    attempt do not describe the new one.
    """
    state = wr.parse_log(REAL + "\nstart 2026-09-14T18:55:00Z\n")
    assert state.last_train is None and state.last_val is None
    projection = wr.project(state, val_freq=2, usd_per_hour=2.55)
    assert projection.train_min_per_epoch is None
    assert projection.projected_usd is None


def test_measured_and_projected_are_never_inconsistent():
    """Either both are present or the report says it cannot cost the run."""
    for text in (REAL, REAL + "\nstart 2026-09-14T18:55:00Z\n",
                 "start 2026-09-13T16:01:55Z\n[run ] fetch\n"):
        state = wr.parse_log(text)
        projection = wr.project(state, val_freq=1, usd_per_hour=2.55,
                                now=state.started_at)
        if projection.train_min_per_epoch is not None:
            assert projection.total_min is not None, text[:40]
