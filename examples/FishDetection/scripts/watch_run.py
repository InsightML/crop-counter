"""Watch a CFD-17 benchmark run and re-cost it from measured throughput.

``status.sh`` answers "is the box alive?" over SSM. This answers the question
that actually decides whether to let a run continue: **given what it is really
doing per second, what will it cost, and is anything going wrong?**

It is pure log analysis — it never touches the GPU box. The log can be a local
file or the copy ``run_all.sh`` syncs to S3 every 600 s, which is read by RANGE
request (the last run's log was 121 MB; polling that whole object would cost
more than the instance).

Three things it reports:

* **Progress** — pipeline stage, current epoch, and the live tqdm rate.
* **Re-cost** — measured it/s -> min/epoch -> projected wall -> projected $,
  against the budget. Plan step 5 calls for exactly this after epoch 1, because
  the estimate was scaled from a different trunk by a FLOPs model that assumes
  compute-bound scaling, and the trunk's depthwise convs are memory-bound.
* **Alarms** — the things worth killing a run over, each of which has actually
  happened on this benchmark:
  - a point run whose F1 stays 0.000 (selection metric is measured at the fixed
    ``tau``; if scores sit below it, ``best.pt`` is being chosen at random);
  - the selection metric not improving for several validations;
  - the projection running past budget.

Usage::

    python watch_run.py --log runs/run_all.log
    python watch_run.py --s3 s3://insightml-cfd-benchmark/runs/run_all.log
    python watch_run.py --s3 ... --follow 300     # re-read every 5 min

Exit code: 0 healthy or finished, 1 alarms raised, 2 the run FAILED.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

#: Measured p5.4xlarge spot in us-east-2, 2026-09: $2.4857-$2.6295/h.
DEFAULT_SPOT_USD_PER_HOUR = 2.55
#: On-demand, for the "if spot dies" line.
ON_DEMAND_USD_PER_HOUR = 6.88

#: Run B's budget: expected $14-15, +30% for one reclaim.
DEFAULT_BUDGET_USD = 19.0

#: How many validations may pass without the selection metric improving before
#: that is worth saying out loud.
STALL_VALIDATIONS = 3

_TS = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"

# `epoch   3 | loss 1.4966 val 1.4576 | ... | P 0.742 R 0.603 F1 0.666 MAE 0.66 | lr 9.33e-04  <- best`
# `epoch   1 | loss 3.3697 val 3.8464 | val MAE 2.18 ... | P 0.812 R 0.034 F1 0.065 | lr 2.00e-03  <- best`
EPOCH_RE = re.compile(
    r"epoch\s+(?P<epoch>\d+)\s*\|\s*loss\s+(?P<train_loss>[\d.]+)\s+"
    r"val\s+(?P<val_loss>[\d.]+)\s*\|(?P<body>.*?)\|\s*lr\s+(?P<lr>[\d.e+-]+)"
    r"(?P<best>\s*<- best)?\s*$"
)
F1_RE = re.compile(r"\bF1\s+(?P<f1>[\d.]+)")
MAE_RE = re.compile(r"\bMAE\s+(?P<mae>[\d.]+)")
AP50_RE = re.compile(r"\bAP50\s+(?P<ap50>[\d.]+)")

# `epoch 1/8: 100%|...| 20137/20238 [25:25<00:07, 13.43it/s, loss=...]`
BAR_RE = re.compile(
    r"(?P<phase>epoch|val)\s+(?P<epoch>\d+)/(?P<epochs>\d+):\s*\d+%\|[^|]*\|\s*"
    r"(?P<done>\d+)/(?P<total>\d+)\s*\[(?P<elapsed>[\d:]+)<(?P<eta>[\d:?]+),\s*"
    r"(?P<rate>[\d.]+)(?P<unit>it/s|s/it)"
)

MARKER_RE = re.compile(r"^\[(run |skip|info|gate|sync|warn)\s*\]\s*(?P<text>.*)$")
START_RE = re.compile(rf"^start\s+(?P<ts>{_TS})")
DONE_RE = re.compile(r"^ALL DONE wall=(?P<wall>\d+)s")
FAILED_RE = re.compile(r"^FAILED rc=(?P<rc>\d+) wall=(?P<wall>\d+)s")
TRAIN_RE = re.compile(r"^\[run \]\s*train\s+(?P<name>\S+)")
CONFIG_RE = re.compile(
    r"backbone\s+(?P<backbone>\S+)\s+(?P<frozen>frozen|trainable)[^|]*\|\s*"
    r"decoder params:\s*(?P<params>[\d.]+)M\s*\|\s*select on\s+(?P<select_on>\S+)"
)


def parse_hms(text: str) -> Optional[float]:
    """``"25:25"`` or ``"1:02:03"`` -> seconds; ``"?"`` -> None."""
    if "?" in text:
        return None
    parts = [int(p) for p in text.split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


@dataclass
class Epoch:
    epoch: int
    train_loss: float
    val_loss: float
    f1: Optional[float] = None
    mae: Optional[float] = None
    ap50: Optional[float] = None
    best: bool = False


@dataclass
class Progress:
    phase: str          # "epoch" (training) or "val"
    epoch: int
    epochs: int
    done: int
    total: int
    elapsed_s: Optional[float]
    eta_s: Optional[float]
    rate_it_s: float

    @property
    def full_phase_s(self) -> Optional[float]:
        """Seconds for the WHOLE pass, extrapolated from elapsed — not 1/rate.

        tqdm's rate is a smoothed recent average, so it quietly excludes the
        loader stalls and checkpoint writes a real epoch pays for.
        Elapsed-over-fraction includes everything that has actually happened.
        Checked against the last run: epoch 1's bar gives 25.5 min against a
        measured 25.6, and val gives 12.4 against 12.5.
        """
        if self.elapsed_s and self.done and self.total:
            return self.elapsed_s / (self.done / self.total)
        if self.rate_it_s and self.total:
            return self.total / self.rate_it_s
        return None


@dataclass
class RunState:
    started_at: Optional[datetime] = None
    finished: Optional[str] = None          # "ALL DONE" | "FAILED"
    wall_s: Optional[int] = None
    rc: Optional[int] = None
    stages: List[str] = field(default_factory=list)
    run_name: Optional[str] = None
    backbone: Optional[str] = None
    decoder_params_m: Optional[float] = None
    select_on: Optional[str] = None
    epochs: List[Epoch] = field(default_factory=list)
    progress: Optional[Progress] = None
    #: Most recent bar of each phase. A poll that lands during validation still
    #: needs the training rate, and vice versa — keeping only the latest bar
    #: made the tool blind for the ~30% of wall clock spent in validation.
    last_train: Optional[Progress] = None
    last_val: Optional[Progress] = None

    @property
    def is_point_run(self) -> bool:
        """A point run reports no AP50 — that is the only box-only metric here."""
        return bool(self.epochs) and all(e.ap50 is None for e in self.epochs)


def parse_log(text: str) -> RunState:
    """Parse a ``run_all.log`` (whole file or a tail) into a :class:`RunState`.

    tqdm redraws with carriage returns, so the text is split on both; a tail
    read can begin mid-line, which only ever costs the first record.
    """
    state = RunState()
    for raw in re.split(r"[\r\n]+", text):
        line = raw.rstrip()
        if not line:
            continue

        match = START_RE.match(line)
        if match:
            state.started_at = datetime.fromisoformat(
                match.group("ts").replace("Z", "+00:00")
            )
            # A relaunch restarts the pipeline: drop the previous attempt
            # entirely, measurements included — a previous attempt's throughput
            # does not describe this one (different trunk, different batch,
            # possibly a different instance). Forgetting `progress` but keeping
            # `last_train` left a state that reported a rate it would not project
            # from.
            state.stages, state.epochs, state.progress = [], [], None
            state.last_train = state.last_val = None
            state.finished = state.wall_s = state.rc = None
            continue

        match = DONE_RE.match(line)
        if match:
            state.finished, state.wall_s = "ALL DONE", int(match.group("wall"))
            continue

        match = FAILED_RE.match(line)
        if match:
            state.finished = "FAILED"
            state.wall_s, state.rc = int(match.group("wall")), int(match.group("rc"))
            continue

        match = MARKER_RE.match(line)
        if match:
            if not line.startswith("[sync"):
                state.stages.append(line)
            train = TRAIN_RE.match(line)
            if train:
                state.run_name = train.group("name")
                state.epochs = []          # a new run's epochs are its own
            continue

        match = CONFIG_RE.search(line)
        if match:
            state.backbone = match.group("backbone")
            state.decoder_params_m = float(match.group("params"))
            state.select_on = match.group("select_on")
            continue

        match = EPOCH_RE.search(line)
        if match:
            body = match.group("body")
            f1 = F1_RE.search(body)
            mae = MAE_RE.search(body)
            ap50 = AP50_RE.search(body)
            epoch = Epoch(
                epoch=int(match.group("epoch")),
                train_loss=float(match.group("train_loss")),
                val_loss=float(match.group("val_loss")),
                f1=float(f1.group("f1")) if f1 else None,
                mae=float(mae.group("mae")) if mae else None,
                ap50=float(ap50.group("ap50")) if ap50 else None,
                best=bool(match.group("best")),
            )
            # The same epoch can be logged twice across a resume; last wins.
            state.epochs = [e for e in state.epochs if e.epoch != epoch.epoch]
            state.epochs.append(epoch)
            state.epochs.sort(key=lambda e: e.epoch)
            continue

        match = BAR_RE.search(line)
        if match:
            rate = float(match.group("rate"))
            if match.group("unit") == "s/it":     # slower than 1 it/s
                rate = 1.0 / rate if rate else 0.0
            bar = Progress(
                phase=match.group("phase"),
                epoch=int(match.group("epoch")),
                epochs=int(match.group("epochs")),
                done=int(match.group("done")),
                total=int(match.group("total")),
                elapsed_s=parse_hms(match.group("elapsed")),
                eta_s=parse_hms(match.group("eta")),
                rate_it_s=rate,
            )
            state.progress = bar
            if bar.phase == "epoch":
                state.last_train = bar
            else:
                state.last_val = bar
    return state


@dataclass
class Projection:
    train_min_per_epoch: Optional[float] = None
    val_min_per_pass: Optional[float] = None
    remaining_min: Optional[float] = None
    total_min: Optional[float] = None
    projected_usd: Optional[float] = None
    on_demand_usd: Optional[float] = None
    elapsed_min: Optional[float] = None


def project(
    state: RunState,
    *,
    val_freq: int,
    usd_per_hour: float,
    now: Optional[datetime] = None,
) -> Projection:
    """Re-cost the run from whatever throughput has actually been measured.

    Training time comes from the live tqdm rate (or a completed epoch's own
    elapsed time, which is better because it includes the loader stalls a
    momentary rate does not). Validation is costed per pass and multiplied by
    how many passes ``val_freq`` implies, plus the final full pass.
    """
    out = Projection()
    now = now or datetime.now(timezone.utc)
    if state.started_at:
        out.elapsed_min = (now - state.started_at).total_seconds() / 60.0

    if state.last_train and state.last_train.full_phase_s:
        out.train_min_per_epoch = state.last_train.full_phase_s / 60.0
    if state.last_val and state.last_val.full_phase_s:
        out.val_min_per_pass = state.last_val.full_phase_s / 60.0

    progress = state.progress
    # Everything below needs a training rate; a run that has not trained a
    # single measured step yet is reported as progress only, not re-costed.
    if progress is None or out.train_min_per_epoch is None:
        return out

    epochs_total = progress.epochs
    # How much of the CURRENT epoch remains. During validation that epoch's
    # training is already paid for, so it counts as done.
    if progress.phase == "epoch":
        epochs_done = progress.epoch - 1
        fraction = progress.done / progress.total if progress.total else 0.0
    else:
        epochs_done = progress.epoch
        fraction = 0.0
    epochs_left = max(epochs_total - epochs_done - fraction, 0.0)

    # Validation passes still to come: every val_freq-th epoch, plus the last.
    def validates(epoch: int) -> bool:
        return epoch % val_freq == 0 or epoch == epochs_total

    first_pending = progress.epoch if progress.phase == "epoch" else progress.epoch + 1
    remaining_vals = sum(1 for e in range(first_pending, epochs_total + 1) if validates(e))
    val_min = out.val_min_per_pass if out.val_min_per_pass is not None else 0.0

    out.remaining_min = epochs_left * out.train_min_per_epoch + remaining_vals * val_min
    if out.elapsed_min is not None:
        out.total_min = out.elapsed_min + out.remaining_min
        out.projected_usd = out.total_min / 60.0 * usd_per_hour
        out.on_demand_usd = out.total_min / 60.0 * ON_DEMAND_USD_PER_HOUR
    return out


def alarms(state: RunState, projection: Projection, *, budget_usd: float) -> List[str]:
    """The things worth interrupting a run for. Empty list = nothing to do."""
    out: List[str] = []

    if state.finished == "FAILED":
        out.append(f"run FAILED (rc={state.rc}) — read the log tail above")

    if state.is_point_run and state.epochs:
        zero = [e.epoch for e in state.epochs if e.f1 == 0.0]
        if len(zero) >= 2 and len(zero) == len(state.epochs):
            out.append(
                f"F1 is 0.000 at every validation so far (epochs {zero}) — the "
                "selection metric is measured at the FIXED tau, so best.pt is "
                "being chosen on noise. Scores are sitting below tau: lower it "
                "rather than adding epochs."
            )

    if state.select_on and len(state.epochs) > STALL_VALIDATIONS:
        key = {"f1": "f1", "ap50": "ap50", "val_loss": "val_loss"}.get(state.select_on)
        values = [getattr(e, key, None) for e in state.epochs] if key else []
        values = [v for v in values if v is not None]
        if len(values) > STALL_VALIDATIONS:
            recent, earlier = values[-STALL_VALIDATIONS:], values[:-STALL_VALIDATIONS]
            better = min if state.select_on == "val_loss" else max
            if earlier and better(recent + earlier) not in recent:
                out.append(
                    f"{state.select_on} has not improved in {STALL_VALIDATIONS} "
                    f"validations (best is epoch {state.epochs[values.index(better(values))].epoch})"
                )

    if projection.projected_usd is not None and projection.projected_usd > budget_usd:
        out.append(
            f"projected ${projection.projected_usd:.2f} exceeds the "
            f"${budget_usd:.2f} budget"
        )
    return out


def _fmt_min(minutes: Optional[float]) -> str:
    if minutes is None:
        return "—"
    hours, mins = divmod(int(minutes), 60)
    return f"{hours}h {mins:02d}m" if hours else f"{mins}m"


def render(state: RunState, projection: Projection, *, budget_usd: float,
           usd_per_hour: float, val_freq: int) -> str:
    """The whole report, as one block of text."""
    lines: List[str] = []
    head = state.run_name or "(no training run started yet)"
    lines.append(f"run        {head}")
    if state.backbone:
        lines.append(
            f"model      {state.backbone} | decoder {state.decoder_params_m}M "
            f"| select on {state.select_on}"
        )
    if state.started_at:
        lines.append(f"started    {state.started_at:%Y-%m-%d %H:%M}Z "
                     f"(elapsed {_fmt_min(projection.elapsed_min)})")
    if state.stages:
        lines.append(f"stage      {state.stages[-1]}")

    progress = state.progress
    if state.finished:
        lines.append(f"status     {state.finished} after {_fmt_min((state.wall_s or 0) / 60)}")
    elif progress:
        pct = 100.0 * progress.done / progress.total if progress.total else 0.0
        label = "train" if progress.phase == "epoch" else "val  "
        lines.append(
            f"progress   {label} epoch {progress.epoch}/{progress.epochs} "
            f"{progress.done:,}/{progress.total:,} ({pct:.0f}%) "
            f"at {progress.rate_it_s:.2f} it/s, eta {_fmt_min((progress.eta_s or 0) / 60)}"
        )
    else:
        lines.append("progress   no training bar in this window yet")

    if state.epochs:
        lines.append("")
        lines.append("  epoch   train    val      F1      MAE    ")
        for e in state.epochs:
            f1 = f"{e.f1:.3f}" if e.f1 is not None else "  —  "
            mae = f"{e.mae:.2f}" if e.mae is not None else "  —  "
            flag = "  <- best" if e.best else ""
            lines.append(
                f"  {e.epoch:>5}   {e.train_loss:.4f}  {e.val_loss:.4f}  "
                f"{f1}   {mae}{flag}"
            )

    lines.append("")
    if projection.train_min_per_epoch is not None:
        lines.append(f"measured   {projection.train_min_per_epoch:.1f} min/epoch train"
                     + (f", {projection.val_min_per_pass:.1f} min/val pass"
                        if projection.val_min_per_pass is not None else ""))
    if projection.total_min is not None:
        lines.append(
            f"projected  {_fmt_min(projection.total_min)} total "
            f"({_fmt_min(projection.remaining_min)} left, val_freq {val_freq}) "
            f"— to the end of THIS training run, excluding post-training scoring"
        )
        lines.append(
            f"cost       ${projection.projected_usd:.2f} at ${usd_per_hour:.2f}/h spot "
            f"| ${projection.on_demand_usd:.2f} if forced to on-demand "
            f"| budget ${budget_usd:.2f}"
        )
    else:
        lines.append("projected  not enough measured throughput yet to re-cost")

    raised = alarms(state, projection, budget_usd=budget_usd)
    lines.append("")
    if raised:
        for alarm in raised:
            lines.append(f"!! {alarm}")
    else:
        lines.append("ok         nothing to act on")
    return "\n".join(lines)


#: Marks where a head window was joined to a tail window, so a reader of the
#: raw text can see that the middle is missing.
WINDOW_JOIN = "\n[watch_run] ---- log middle skipped ----\n"


def _s3_range(uri: str, spec: str) -> str:
    """One ranged GET against an S3 object, as text."""
    bucket, _, key = uri[len("s3://"):].partition("/")
    result = subprocess.run(
        ["aws", "s3api", "get-object", "--bucket", bucket, "--key", key,
         "--range", f"bytes={spec}", "/dev/stdout"],
        capture_output=True, check=True,
    )
    text = result.stdout.decode("utf-8", errors="replace")
    # get-object appends its JSON metadata to stdout after the body.
    return re.sub(r"\n?\{[^{}]*\"ContentLength\".*\}\s*$", "", text, flags=re.S)


def read_window(uri: str, head_bytes: int, tail_bytes: int) -> str:
    """The log's HEAD plus its TAIL, which is what a live poll actually needs.

    Two range requests, not one, because the two ends answer different
    questions and everything between them is tqdm redraw. The head carries
    ``start``, the pipeline markers and the model line; the tail carries the
    current bar and the most recent epochs. A tail alone cannot tell you when
    the run started — and on the last run, 2 MB of tail was about half an epoch
    of a 66 MB log, so "just read more" is not an answer.
    """
    head = _s3_range(uri, f"0-{head_bytes - 1}")
    tail = _s3_range(uri, f"-{tail_bytes}")
    return head + WINDOW_JOIN + tail


def read_local_window(path: Path, head_bytes: int, tail_bytes: int) -> str:
    """:func:`read_window` for a file on disk."""
    data = path.read_bytes()
    if len(data) <= head_bytes + tail_bytes:
        return data.decode("utf-8", errors="replace")
    head = data[:head_bytes].decode("utf-8", errors="replace")
    tail = data[-tail_bytes:].decode("utf-8", errors="replace")
    return head + WINDOW_JOIN + tail


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--log", type=Path, help="a local run_all.log")
    source.add_argument("--s3", help="s3://bucket/key of run_all.log")
    parser.add_argument("--tail-bytes", type=int, default=2_000_000,
                        help="how much of the log END to read (default 2 MB)")
    parser.add_argument("--head-bytes", type=int, default=262_144,
                        help="how much of the log START to read (default 256 KB) — "
                             "it holds `start`, the stage markers and the model line")
    parser.add_argument("--val-freq", type=int, default=2,
                        help="the run's val_freq, for the projection (default 2)")
    parser.add_argument("--usd-per-hour", type=float, default=DEFAULT_SPOT_USD_PER_HOUR)
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD)
    parser.add_argument("--now", default=None,
                        help="ISO timestamp to treat as 'now' (for replaying an old log)")
    parser.add_argument("--follow", type=int, metavar="SECONDS", default=0,
                        help="re-read and reprint every SECONDS until the run ends")
    return parser


def read_source(args) -> str:
    if args.log:
        return read_local_window(args.log, args.head_bytes, args.tail_bytes)
    return read_window(args.s3, args.head_bytes, args.tail_bytes)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    while True:
        now = (datetime.fromisoformat(args.now.replace("Z", "+00:00"))
               if args.now else None)
        state = parse_log(read_source(args))
        projection = project(state, val_freq=args.val_freq,
                             usd_per_hour=args.usd_per_hour, now=now)
        print(f"--- {datetime.now(timezone.utc):%H:%M:%S}Z "
              f"{'-' * 50}")
        print(render(state, projection, budget_usd=args.budget,
                     usd_per_hour=args.usd_per_hour, val_freq=args.val_freq))
        raised = alarms(state, projection, budget_usd=args.budget)
        if state.finished == "FAILED":
            return 2
        if state.finished or not args.follow:
            return 1 if raised else 0
        time.sleep(args.follow)


if __name__ == "__main__":
    raise SystemExit(main())
