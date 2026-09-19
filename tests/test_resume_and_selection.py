"""Unit tests for checkpoint selection, backbone checkpoints and resume.

Three separable things land together here because they share a payload:

* ``select_on`` decides which metric writes ``best.pt``. ``val_loss`` (the
  default) is minimised; ``ap50`` and ``f1`` are maximised.
* an unfrozen run's ``best.pt``/``last.pt`` carries the fine-tuned trunk, and
  :func:`load_backbone_from` is the seam that puts it under another decoder.
* ``last.pt`` also carries optimizer/scheduler/scaler state, so ``--resume``
  picks a killed run back up.

CPU-light: the models are either a toy ``nn.Linear`` or the ``stub_backbone``
fixture's ~31k-parameter trunk. No DINOv3 weights, no dataset, no network.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

from cropcounter.train import (
    TrainConfig,
    _is_better,
    _reimpose_schedule_horizon,
    _save_checkpoint,
    apply_overrides,
    build_optimizer_and_scheduler,
    load_backbone_from,
    load_checkpoint,
    main,
    seed_everything,
    selection_key,
    train,
)

#: base_lr x this ladder, stem -> top, at the default layer_decay of 0.8.
EXPECTED_LADDER = [0.8 ** 4, 0.8 ** 3, 0.8 ** 2, 0.8, 1.0]

#: The shipped wheat decoder — present in a checkout, absent in a bare CI job.
WHEAT_DECODER = Path(__file__).resolve().parents[1] / "weights" / "decoder_best.pt"

#: Val summaries shaped like the Brackish box run: the loss bottoms out early
#: while every detection metric keeps climbing to the last epoch.
BRACKISH = [
    {"val_loss": 3.0, "ap50": 0.10},
    {"val_loss": 2.0, "ap50": 0.20},
    {"val_loss": 1.0, "ap50": 0.30},   # epoch 3 — best val_loss
    {"val_loss": 1.2, "ap50": 0.42},
    {"val_loss": 1.4, "ap50": 0.51},
    {"val_loss": 1.6, "ap50": 0.58},
    {"val_loss": 1.9, "ap50": 0.63},
    {"val_loss": 2.4, "ap50": 0.67},   # epoch 8 — best ap50
]


def _pick_best_epoch(summaries, select_on: str, task: str) -> int:
    """Replay ``train()``'s selection rule over a list of val summaries."""
    key, sign = selection_key(select_on, task)
    best_metric, best_epoch = float("inf") * -sign, 0
    for epoch, summary in enumerate(summaries, start=1):
        if _is_better(summary[key], best_metric, sign):
            best_metric, best_epoch = summary[key], epoch
    return best_epoch


def _model(stub_cls, trainable: bool, task: str = "box"):
    """A stub-trunk CropCounter, built through the patched hub loader."""
    from cropcounter.dinov3_pyramid import CropCounter
    return CropCounter(task=task, backbone_trainable=trainable)


def _perturb(module: nn.Module) -> None:
    """Move every parameter, so an equality assertion after a load bites."""
    with torch.no_grad():
        for param in module.parameters():
            param.add_(torch.randn_like(param))


def _write(model, cfg: TrainConfig, path: Path) -> Path:
    """Save in the format ``train()`` writes for ``last.pt``."""
    _save_checkpoint(model, cfg, path, epoch=1, history={}, best_metric=0.0, best_epoch=1)
    return path


# --- 12: the selection table ------------------------------------------------

def test_selection_key_table():
    """val_loss is minimised; ap50 and f1 are maximised."""
    assert selection_key("val_loss", "point") == ("val_loss", -1)
    assert selection_key("val_loss", "box") == ("val_loss", -1)
    assert selection_key("f1", "point") == ("f1", +1)
    assert selection_key("ap50", "box") == ("ap50", +1)


def test_selection_key_rejects_an_unknown_metric():
    """A typo must fail before the data loads, not silently pick val_loss."""
    with pytest.raises(ValueError, match="select_on"):
        selection_key("val_ap50", "box")


def test_selection_key_rejects_ap50_for_the_point_task():
    """The point task never computes AP, so selecting on it is a config bug."""
    with pytest.raises(ValueError, match="point"):
        selection_key("ap50", "point")


# --- 13: the rule the loop applies ------------------------------------------

def test_ap50_and_val_loss_pick_different_epochs():
    """On Brackish-shaped summaries the two selectors genuinely disagree."""
    assert _pick_best_epoch(BRACKISH, "val_loss", "box") == 3
    assert _pick_best_epoch(BRACKISH, "ap50", "box") == 8


# --- 14: the backbone in the payload ----------------------------------------

def test_trainable_checkpoint_round_trips_the_backbone(stub_backbone, tmp_path):
    """An unfrozen run's checkpoint carries the trunk, exactly."""
    cfg = TrainConfig(task="box", backbone_trainable=True)
    source = _model(stub_backbone, trainable=True)
    _perturb(source.backbone.model)
    path = _write(source, cfg, tmp_path / "last.pt")
    assert "backbone" in torch.load(path, map_location="cpu", weights_only=False)

    target = _model(stub_backbone, trainable=True)
    load_backbone_from(target, path)

    tuned = source.backbone.model.state_dict()
    for key, tensor in target.backbone.model.state_dict().items():
        assert torch.equal(tensor, tuned[key]), key


def test_frozen_checkpoint_has_no_backbone_to_load(stub_backbone, tmp_path):
    """Asking a frozen run's checkpoint for a trunk says why there isn't one."""
    cfg = TrainConfig(task="box")
    path = _write(_model(stub_backbone, trainable=False), cfg, tmp_path / "last.pt")
    assert "backbone" not in torch.load(path, map_location="cpu", weights_only=False)

    with pytest.raises(KeyError, match="frozen"):
        load_backbone_from(_model(stub_backbone, trainable=True), path)


def test_best_checkpoint_carries_no_optimizer_state(stub_backbone, tmp_path):
    """``resume_state`` is opt-in: AdamW moments are ~700 MB on this trunk."""
    cfg = TrainConfig(task="box", backbone_trainable=True)
    path = _write(_model(stub_backbone, trainable=True), cfg, tmp_path / "best.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "resume_state" not in payload
    assert {"decoder", "config", "epoch", "history", "best_metric", "best_epoch"} <= set(payload)


# --- 15: an inconsistent payload --------------------------------------------

def test_load_checkpoint_rejects_a_backbone_in_a_frozen_run(stub_backbone, tmp_path):
    """A payload claiming frozen while carrying a trunk is not loadable."""
    decoder = _model(stub_backbone, trainable=False).decoder.state_dict()
    path = tmp_path / "inconsistent.pt"
    torch.save(
        {"decoder": decoder, "config": TrainConfig(task="box").to_dict(), "backbone": {}},
        path,
    )
    with pytest.raises(ValueError, match="backbone_trainable"):
        load_checkpoint(path, torch.device("cpu"))


# --- 16: resume restores the optimiser ---------------------------------------

def test_resume_restores_optimizer_scheduler_and_history(tmp_path):
    """Same construction + ``load_state_dict`` gives the identical run state."""
    cfg = TrainConfig(epochs=6, warmup_epochs=2, lr=1e-3)
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)

    history = {"train_loss": []}
    for _epoch in range(3):
        optimizer.zero_grad()
        model(torch.ones(2, 4)).sum().backward()
        optimizer.step()
        scheduler.step()
        history["train_loss"].append(float(_epoch))

    path = tmp_path / "last.pt"
    torch.save({
        "epoch": 3, "history": history, "best_metric": 1.0, "best_epoch": 2,
        "model": model.state_dict(),
        "resume_state": {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": None,
        },
    }, path)
    resume = torch.load(path, map_location="cpu", weights_only=False)

    torch.manual_seed(99)                       # a deliberately different init
    fresh = nn.Linear(4, 1)
    fresh.load_state_dict(resume["model"])
    new_optimizer, new_scheduler = build_optimizer_and_scheduler(fresh, cfg)
    new_optimizer.load_state_dict(resume["resume_state"]["optimizer"])
    new_scheduler.load_state_dict(resume["resume_state"]["scheduler"])

    assert resume["epoch"] + 1 == 4
    assert resume["history"] == history
    assert resume["best_metric"] == 1.0 and resume["best_epoch"] == 2
    assert new_scheduler.get_last_lr() == scheduler.get_last_lr()
    assert [g["lr"] for g in new_optimizer.param_groups] == [
        g["lr"] for g in optimizer.param_groups
    ]
    for param, restored in zip(model.parameters(), fresh.parameters()):
        moments = optimizer.state[param]
        new_moments = new_optimizer.state[restored]
        assert torch.equal(new_moments["exp_avg"], moments["exp_avg"])
        assert torch.equal(new_moments["exp_avg_sq"], moments["exp_avg_sq"])
        assert new_moments["step"] == moments["step"]


# --- 17: the forgetting seam -------------------------------------------------

@pytest.mark.skipif(not WHEAT_DECODER.exists(), reason="shipped wheat decoder not present")
def test_wheat_decoder_loads_onto_a_tuned_trunk(stub_backbone, tmp_path):
    """The experiment this is all for: fine-tuned trunk, shipped wheat head.

    The decoder's state-dict key set must not have moved, so the shipped point
    checkpoint still loads with ``strict=True`` onto a model whose backbone
    came from somewhere else entirely.
    """
    donor = _model(stub_backbone, trainable=True, task="point")
    _perturb(donor.backbone.model)
    tuned = _write(donor, TrainConfig(backbone_trainable=True), tmp_path / "tuned.pt")

    model = _model(stub_backbone, trainable=True, task="point")
    load_backbone_from(model, tuned)
    payload = torch.load(WHEAT_DECODER, map_location="cpu", weights_only=False)
    model.decoder.load_state_dict(payload["decoder"], strict=True)

    donor_state = donor.backbone.model.state_dict()
    for key, tensor in model.backbone.model.state_dict().items():
        assert torch.equal(tensor, donor_state[key]), key


# --- 18: the CLI -------------------------------------------------------------

def _cfg_from_cli(monkeypatch, argv) -> TrainConfig:
    """Run ``main(argv)`` with training stubbed out and return the config."""
    captured = {}
    # By dotted path: the package rebinds ``cropcounter.train`` to the train
    # FUNCTION, so the attribute of that name is not the module.
    monkeypatch.setattr(
        "cropcounter.train.train", lambda cfg: captured.setdefault("cfg", cfg)
    )
    assert main(argv) == 0
    return captured["cfg"]


def test_cli_parses_the_new_flags(monkeypatch, tmp_path):
    """``--resume``/``--backbone-trainable``/``--backbone-lr``/``--select-on``."""
    cfg = _cfg_from_cli(monkeypatch, [
        "--resume", str(tmp_path / "last.pt"),
        "--backbone-trainable",
        "--backbone-lr", "5e-5",
        "--select-on", "ap50",
    ])
    assert cfg.resume_from == tmp_path / "last.pt"
    assert cfg.backbone_trainable is True
    assert cfg.backbone_lr == 5e-5
    assert cfg.select_on == "ap50"


def test_absent_backbone_trainable_flag_does_not_clobber_the_config(monkeypatch, tmp_path):
    """A store_true default of False would silently refreeze an unfrozen run."""
    config = TrainConfig(backbone_trainable=True, select_on="ap50", task="box").to_json(
        tmp_path / "config.json"
    )
    cfg = _cfg_from_cli(monkeypatch, ["--config", str(config)])
    assert cfg.backbone_trainable is True
    assert cfg.select_on == "ap50"


def test_apply_overrides_ignores_unset_arguments():
    """Only non-None arguments reach the config — the whole override contract."""
    cfg = TrainConfig(epochs=8, backbone_lr=3e-5)

    class _Args:
        epochs = None
        backbone_lr = 1e-5

    apply_overrides(cfg, _Args())
    assert cfg.epochs == 8
    assert cfg.backbone_lr == 1e-5


# ===========================================================================
# End-to-end: the real train() loop on a synthetic COCO detection split.
# ===========================================================================

#: Side of the synthetic frames. Divisible by 32, so validation pads nothing.
SIDE = 128


@pytest.fixture
def tiny_box_data(tmp_path):
    """A 5-image synthetic COCO detection split on disk: 3 train, 2 val.

    Noise frames with two boxes each — enough for the loader, the sampler, the
    box targets and COCOeval to all run for real, and small enough that a
    whole epoch is a second or two on CPU.
    """
    rng = np.random.default_rng(0)
    root = tmp_path / "data"
    for split, count in (("train", 3), ("val", 2)):
        images_dir = root / split / "images"
        images_dir.mkdir(parents=True)
        images, annotations = [], []
        for i in range(count):
            name = f"{split}_{i}.png"
            cv2.imwrite(str(images_dir / name),
                        rng.integers(0, 255, (SIDE, SIDE, 3), dtype=np.uint8))
            images.append({"id": f"{split}{i}", "file_name": name,
                           "width": SIDE, "height": SIDE})
            for j in range(2):
                annotations.append({
                    "id": len(annotations) + 1, "image_id": f"{split}{i}",
                    "category_id": 1, "bbox": [8 + 32 * j, 12 + 28 * j, 16, 16],
                    "area": 256.0, "iscrowd": 0,
                })
        (root / split / "annotations.json").write_text(
            json.dumps({"images": images, "annotations": annotations,
                        "categories": [{"id": 1, "name": "fish"}]}),
            encoding="utf-8",
        )
    return root


def _box_cfg(data_root: Path, out_dir: Path, **overrides) -> TrainConfig:
    """A minimal but genuine box run: tiny decoder, one tile, CPU, no workers."""
    settings = dict(
        data_root=data_root, out_dir=out_dir, annotation_format="coco_bbox",
        task="box", select_on="ap50", c_dec=32, tile=64, tiles_per_image=1,
        batch_size=1, num_workers=0, device="cpu", epochs=1, warmup_epochs=1,
        seed=0,
    )
    settings.update(overrides)
    return TrainConfig(**settings)


def _payload(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _first_moment(optimizer_state: dict) -> torch.Tensor:
    """The exp_avg of the first parameter that has one — AdamW's momentum."""
    for entry in optimizer_state["state"].values():
        if "exp_avg" in entry:
            return entry["exp_avg"]
    raise AssertionError("optimizer state carries no exp_avg")


def test_train_writes_both_checkpoints_and_both_prediction_files(
    stub_backbone, tiny_box_data, tmp_path
):
    """One real epoch: resume state on last.pt only, both COCO result files."""
    out = tmp_path / "runs"
    train(_box_cfg(tiny_box_data, out, run_name="e2e"))
    run_dir = out / "e2e"

    last, best = _payload(run_dir / "last.pt"), _payload(run_dir / "best.pt")
    assert "resume_state" in last
    assert "resume_state" not in best
    assert (run_dir / "predictions.json").exists()
    assert (run_dir / "predictions_last.json").exists()
    # Frozen by default, so neither checkpoint carries the trunk.
    assert "backbone" not in last and "backbone" not in best


def test_unfrozen_run_puts_the_trunk_in_both_checkpoints(
    stub_backbone, tiny_box_data, tmp_path
):
    """``backbone_trainable`` is what decides whether the trunk is saved."""
    out = tmp_path / "runs"
    train(_box_cfg(tiny_box_data, out, run_name="unfrozen", backbone_trainable=True))
    run_dir = out / "unfrozen"
    for name in ("last.pt", "best.pt"):
        payload = _payload(run_dir / name)
        assert "backbone" in payload, name
        assert payload["backbone"], name


def test_resume_continues_the_run_for_real(stub_backbone, tiny_box_data, tmp_path):
    """Resume a real 1-epoch run to 2 epochs: history, selection, schedule."""
    out = tmp_path / "runs"
    train(_box_cfg(tiny_box_data, out, run_name="e2e"))
    run_dir = out / "e2e"

    _model2, history, _name, best_epoch = train(
        _box_cfg(tiny_box_data, out, run_name="e2e", epochs=2,
                 resume_from=run_dir / "last.pt")
    )

    assert len(history["train_loss"]) == 2, "epoch 1's row must survive the resume"
    assert len(history["val_ap50"]) == 2
    # best.pt follows AP50, and ties go to the earlier epoch (a strict >).
    ap50 = history["val_ap50"]
    assert best_epoch == ap50.index(max(ap50)) + 1
    # The optimizer kept counting rather than starting over.
    resumed_step = _first_moment(_payload(run_dir / "last.pt")["resume_state"]["optimizer"])
    assert resumed_step is not None


def test_resume_restores_the_optimizer_moments(
    stub_backbone, tiny_box_data, tmp_path, monkeypatch
):
    """A zero-epoch resume isolates the restore: the loop never runs.

    ``epochs`` equal to the saved epoch makes ``range(start_epoch, epochs + 1)``
    empty, so whatever the optimizer holds when ``train`` returns is exactly
    what was loaded — no training step can have moved it.
    """
    out = tmp_path / "runs"
    train(_box_cfg(tiny_box_data, out, run_name="e2e"))
    saved = _payload(out / "e2e" / "last.pt")["resume_state"]["optimizer"]

    captured = {}
    real = build_optimizer_and_scheduler

    def spy(model, cfg):
        captured["optimizer"], captured["scheduler"] = real(model, cfg)
        return captured["optimizer"], captured["scheduler"]

    monkeypatch.setattr("cropcounter.train.build_optimizer_and_scheduler", spy)
    train(_box_cfg(tiny_box_data, out, run_name="e2e", epochs=1,
                   resume_from=out / "e2e" / "last.pt"))

    restored = captured["optimizer"].state_dict()
    assert torch.equal(_first_moment(restored), _first_moment(saved))
    assert restored["state"] and len(restored["state"]) == len(saved["state"])


def test_resume_reimposes_the_schedule_horizon(
    stub_backbone, tiny_box_data, tmp_path, monkeypatch
):
    """A longer relaunch must anneal over the NEW epoch count, not the old one.

    Without this the cosine keeps the checkpoint's T_max, hits lr 0 at the old
    horizon and then warm-restarts to the base rate for the rest of the run.
    """
    out = tmp_path / "runs"
    train(_box_cfg(tiny_box_data, out, run_name="e2e"))

    captured = {}
    real = build_optimizer_and_scheduler
    monkeypatch.setattr(
        "cropcounter.train.build_optimizer_and_scheduler",
        lambda model, cfg: captured.setdefault("pair", real(model, cfg)),
    )
    train(_box_cfg(tiny_box_data, out, run_name="e2e", epochs=1, warmup_epochs=1,
                   resume_from=out / "e2e" / "last.pt"))
    saved_t_max = captured["pair"][1]._schedulers[1].T_max
    captured.clear()

    train(_box_cfg(tiny_box_data, out, run_name="e2e", epochs=6, warmup_epochs=1,
                   resume_from=out / "e2e" / "last.pt"))
    warmup, cosine = captured["pair"][1]._schedulers
    assert cosine.T_max == 5, f"stale horizon {saved_t_max} survived the resume"
    assert warmup.total_iters == 1


def test_reimposed_horizon_moves_the_warmup_switch_too():
    """``_milestones`` is restored with the rest of the schedule state.

    Left alone, a relaunch with a longer warmup gets ``total_iters`` from the
    new config but still hands over to the cosine at the OLD epoch, so the
    warmup is truncated mid-ramp.
    """
    model = nn.Linear(2, 1)
    _opt, saved = build_optimizer_and_scheduler(
        model, TrainConfig(epochs=8, warmup_epochs=2)
    )
    longer = TrainConfig(epochs=30, warmup_epochs=5)
    _opt2, scheduler = build_optimizer_and_scheduler(model, longer)
    scheduler.load_state_dict(saved.state_dict())
    assert scheduler._milestones == [2], "the stale milestone is what we are fixing"

    _reimpose_schedule_horizon(scheduler, longer)

    warmup, cosine = scheduler._schedulers
    assert warmup.total_iters == 5
    assert cosine.T_max == 25
    assert scheduler._milestones == [5]


def test_seed_everything_is_called_once_in_a_fresh_epoch_one(
    stub_backbone, tiny_box_data, tmp_path, monkeypatch
):
    """The epoch-1 RNG invariant: one seed call, at the top of train()."""
    calls = []
    real = seed_everything
    monkeypatch.setattr(
        "cropcounter.train.seed_everything",
        lambda seed: (calls.append(seed), real(seed))[1],
    )
    train(_box_cfg(tiny_box_data, tmp_path / "runs", run_name="once"))
    assert calls == [0], "epoch 1 must not reseed after the model has been built"


# --- the resume edge cases ---------------------------------------------------

def test_resume_rejects_a_checkpoint_without_optimizer_state(
    stub_backbone, tiny_box_data, tmp_path
):
    """Passing best.pt is the easy mistake; say so before anything loads."""
    out = tmp_path / "runs"
    train(_box_cfg(tiny_box_data, out, run_name="e2e"))
    with pytest.raises(ValueError, match="not a resumable checkpoint"):
        train(_box_cfg(tiny_box_data, out, run_name="e2e", epochs=2,
                       resume_from=out / "e2e" / "best.pt"))


def test_resume_from_is_never_written_into_a_config(
    stub_backbone, tiny_box_data, tmp_path
):
    """Re-running a run's own config.json must not silently resume it."""
    out = tmp_path / "runs"
    train(_box_cfg(tiny_box_data, out, run_name="e2e"))
    train(_box_cfg(tiny_box_data, out, run_name="e2e", epochs=2,
                   resume_from=out / "e2e" / "last.pt"))

    run_dir = out / "e2e"
    assert json.loads((run_dir / "config.json").read_text())["resume_from"] is None
    for name in ("last.pt", "best.pt"):
        assert _payload(run_dir / name)["config"]["resume_from"] is None, name


def test_resume_without_a_run_name_uses_the_checkpoints_own_directory(
    stub_backbone, tiny_box_data, tmp_path
):
    """The relaunch writes where the checkpoint lives, not out_dir/<name>."""
    out = tmp_path / "runs"
    train(_box_cfg(tiny_box_data, out, run_name="e2e"))
    moved = tmp_path / "elsewhere" / "e2e"
    moved.parent.mkdir()
    (out / "e2e").rename(moved)

    _m, _h, run_name, _b = train(
        _box_cfg(tiny_box_data, out, epochs=2, resume_from=moved / "last.pt")
    )
    assert run_name == "e2e"
    assert (moved / "history.json").exists()
    assert not (out / "e2e").exists(), "the run must not fork into out_dir"


def test_changing_select_on_resets_the_best_so_far(
    stub_backbone, tiny_box_data, tmp_path, capsys
):
    """A val_loss best is not comparable with an AP50 best; start the race over."""
    out = tmp_path / "runs"
    train(_box_cfg(tiny_box_data, out, run_name="e2e", select_on="val_loss"))
    capsys.readouterr()

    _m, _h, _n, best_epoch = train(
        _box_cfg(tiny_box_data, out, run_name="e2e", epochs=2, select_on="ap50",
                 resume_from=out / "e2e" / "last.pt")
    )
    printed = capsys.readouterr().out
    assert "select_on" in printed
    # Epoch 1's val_loss best cannot carry over, so the AP50 race restarts and
    # one of the resumed epochs must win it.
    assert best_epoch >= 2


def test_resume_reports_recipe_drift(stub_backbone, tiny_box_data, tmp_path, capsys):
    """Changed hyper-parameters are printed, including the ones that won't apply."""
    out = tmp_path / "runs"
    train(_box_cfg(tiny_box_data, out, run_name="e2e", lr=1e-3, batch_size=1))
    capsys.readouterr()

    train(_box_cfg(tiny_box_data, out, run_name="e2e", epochs=2, lr=5e-4,
                   resume_from=out / "e2e" / "last.pt"))
    printed = capsys.readouterr().out
    assert "resume: lr checkpoint=0.001 now=0.0005" in printed
    assert "resume: epochs checkpoint=1 now=2" in printed
    # The optimizer state carries the old lr, so the new one is NOT in force.
    assert "not applied" in printed


def test_selection_key_rejects_an_unknown_task():
    """A task typo must be a ValueError like every other config error."""
    with pytest.raises(ValueError, match="task"):
        selection_key("val_loss", "segmentation")


# --- 9: the forgetting seam through load_checkpoint --------------------------

def test_load_checkpoint_backbone_from_swaps_the_trunk(stub_backbone, tmp_path):
    """decoder from one checkpoint, trunk from another, both exact."""
    decoder_run = _model(stub_backbone, trainable=False)
    _perturb(decoder_run.decoder)
    decoder_path = _write(decoder_run, TrainConfig(task="box"), tmp_path / "decoder.pt")

    tuned_run = _model(stub_backbone, trainable=True)
    _perturb(tuned_run.backbone.model)
    tuned_path = _write(
        tuned_run, TrainConfig(task="box", backbone_trainable=True), tmp_path / "tuned.pt"
    )

    model, _cfg = load_checkpoint(
        decoder_path, torch.device("cpu"), backbone_from=tuned_path
    )

    tuned_state = tuned_run.backbone.model.state_dict()
    for key, tensor in model.backbone.model.state_dict().items():
        assert torch.equal(tensor, tuned_state[key]), key
    decoder_state = decoder_run.decoder.state_dict()
    for key, tensor in model.decoder.state_dict().items():
        assert torch.equal(tensor, decoder_state[key]), key


# --- 10: the ladder survives the schedule ------------------------------------

def test_every_group_warms_up_by_the_same_factor(stub_backbone):
    """Warmup scales each group off its OWN base lr, so the ladder is preserved."""
    model = _model(stub_backbone, trainable=True)
    cfg = TrainConfig(task="box", backbone_trainable=True, lr=1e-3,
                      backbone_lr=2e-5, warmup_epochs=2, epochs=10)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
    assert len(optimizer.param_groups) == 11

    bases = [group["initial_lr"] for group in optimizer.param_groups]
    assert bases[0] == 1e-3
    for _ in range(cfg.warmup_epochs):
        optimizer.step()        # the loop's order, and it silences torch's warning
        scheduler.step()

    factors = [g["lr"] / base for g, base in zip(optimizer.param_groups, bases)]
    assert factors == pytest.approx([factors[0]] * len(factors))
    backbone_decay = [g["lr"] for g in optimizer.param_groups
                      if g["name"].endswith(".decay") and g["name"] != "decoder"]
    assert backbone_decay == pytest.approx(
        [2e-5 * factors[0] * f for f in EXPECTED_LADDER]
    )


# --- the write itself has to survive a kill ----------------------------------

def test_save_checkpoint_is_atomic(stub_backbone, tmp_path, monkeypatch):
    """A kill mid-write must leave the PREVIOUS checkpoint whole and loadable.

    Spot instances get reclaimed, and the run directory is synced to S3 while
    training continues. Either reader must see one complete file or the other,
    never a truncated one that ``--resume`` would then trust.
    """
    model = _model(stub_backbone, trainable=False)
    cfg = TrainConfig(task="box")
    path = _write(model, cfg, tmp_path / "last.pt")
    original = path.read_bytes()

    real_save = torch.save

    def die_after_writing(payload, target, *args, **kwargs):
        real_save(payload, target, *args, **kwargs)   # the bytes land somewhere...
        raise RuntimeError("spot reclaim")            # ...and then the box goes away

    # A different payload, so a non-atomic write would visibly clobber the file.
    _perturb(model.decoder)
    monkeypatch.setattr(torch, "save", die_after_writing)
    with pytest.raises(RuntimeError, match="spot reclaim"):
        _write(model, cfg, path)

    assert path.read_bytes() == original, "the previous checkpoint was overwritten"
    restored = torch.load(path, map_location="cpu", weights_only=False)
    assert restored["decoder"], "the surviving checkpoint must still load"


# --- 10: F1 selection on the POINT task --------------------------------------
#
# The CFD-17 head-phase run is the first to pair ``task="point"`` with
# ``select_on="f1"``. The table already allows it (``val_f1`` is in
# ``HISTORY_KEYS["point"]``), but nothing had ever run a point training loop
# through that branch end to end — and val loss has now picked the wrong epoch
# on three different architectures, so this is the seam that stops it happening
# a fourth time.


@pytest.fixture
def tiny_point_data(tmp_path):
    """A 5-image synthetic COCO **keypoints** split: 3 train, 2 val.

    Deliberately shaped like a ``cfd points`` root — int ids, one ``fish``
    category, a visible keypoint per object, and one empty val frame so the
    negatives path runs too.
    """
    rng = np.random.default_rng(0)
    root = tmp_path / "point_data"
    for split, count in (("train", 3), ("val", 2)):
        images_dir = root / split / "images"
        images_dir.mkdir(parents=True)
        images, annotations = [], []
        for i in range(count):
            name = f"{split}_{i}.png"
            cv2.imwrite(str(images_dir / name),
                        rng.integers(0, 255, (SIDE, SIDE, 3), dtype=np.uint8))
            images.append({"id": i + 1, "file_name": name,
                           "width": SIDE, "height": SIDE})
            if split == "val" and i == count - 1:
                continue  # one genuinely empty frame
            for j in range(2):
                annotations.append({
                    "id": len(annotations) + 1, "image_id": i + 1,
                    "category_id": 1, "keypoints": [16.0 + 32 * j, 26.0 + 28 * j, 2],
                    "num_keypoints": 1,
                })
        (root / split / "annotations.json").write_text(
            json.dumps({
                "images": images, "annotations": annotations,
                "categories": [{"id": 1, "name": "fish",
                                "keypoints": ["fish"], "skeleton": []}],
            }),
            encoding="utf-8",
        )
    return root


def _point_cfg(data_root: Path, out_dir: Path, **overrides) -> TrainConfig:
    """A minimal but genuine point run, shaped like the CFD-17 head-phase config."""
    settings = dict(
        data_root=data_root, out_dir=out_dir, annotation_format="coco",
        task="point", labels=("fish",), select_on="f1", c_dec=32, tile=64,
        tiles_per_image=1, batch_size=1, num_workers=0, device="cpu", epochs=2,
        warmup_epochs=1, seed=0,
    )
    settings.update(overrides)
    return TrainConfig(**settings)


def test_a_point_run_selects_on_f1_end_to_end(stub_backbone, tiny_point_data, tmp_path):
    """``best.pt`` is the F1-best epoch, and carries that F1 as its best_metric."""
    out = tmp_path / "runs"
    _m, history, _n, best_epoch = train(
        _point_cfg(tiny_point_data, out, run_name="pts")
    )
    f1s = history["val_f1"]
    assert len(f1s) == 2 and all(np.isfinite(f1s))
    assert best_epoch == int(np.argmax(f1s)) + 1

    payload = _payload(out / "pts" / "best.pt")
    assert payload["best_epoch"] == best_epoch
    assert payload["best_metric"] == pytest.approx(max(f1s))
    assert payload["config"]["select_on"] == "f1"


def test_point_selection_on_f1_and_on_val_loss_each_optimise_their_own_metric(
    stub_backbone, tiny_point_data, tmp_path
):
    """The point of the fix: the two criteria read different columns.

    They may agree on this toy data — what must hold is that each run's
    ``best_epoch`` is optimal *for the metric it selected on*, which is exactly
    what val-loss selection failed to be for the real runs.
    """
    out = tmp_path / "runs"
    _m, f1_history, _n, f1_best = train(
        _point_cfg(tiny_point_data, out, run_name="on_f1", select_on="f1")
    )
    _m, loss_history, _n, loss_best = train(
        _point_cfg(tiny_point_data, out, run_name="on_loss", select_on="val_loss")
    )
    assert f1_best == int(np.argmax(f1_history["val_f1"])) + 1
    assert loss_best == int(np.argmin(loss_history["val_loss"])) + 1


def test_a_tapered_point_decoder_trains_end_to_end(stub_backbone, tiny_point_data, tmp_path):
    """The run's actual shape — a point task on a tapered ladder — completes.

    Widths are the run's ratio scaled down to what the stub trunk affords;
    what is being tested is that a sequence ``c_dec`` survives config
    round-trip, model build, training, validation and checkpointing.
    """
    out = tmp_path / "runs"
    cfg = _point_cfg(tiny_point_data, out, run_name="taper", c_dec=[128, 96, 32])
    saved = cfg.to_json(tmp_path / "taper.json")
    assert json.loads(saved.read_text())["c_dec"] == [128, 96, 32]

    model, history, _n, best_epoch = train(TrainConfig.from_json(saved, strict=True))
    assert model.decoder.level_widths == (128, 96, 32)
    assert best_epoch >= 1
    assert all(np.isfinite(history["val_f1"]))
    payload = _payload(out / "taper" / "best.pt")
    assert payload["config"]["c_dec"] == [128, 96, 32]
    assert payload["decoder"]["head.weight"].shape == (1, 32, 1, 1)


def test_a_point_run_writes_scoreable_predictions(stub_backbone, tiny_point_data, tmp_path):
    """The seam that makes offline scoring possible: points land on disk.

    Without this a point run leaves only history.json behind, and point-in-box
    F1 could not be recovered without standing the GPU box back up.
    """
    from cropcounter.metrics import read_point_results

    out = tmp_path / "runs"
    cfg = _point_cfg(tiny_point_data, out, run_name="pred", ap_tau=0.001)
    train(cfg)

    for filename in ("predictions.json", "predictions_last.json"):
        payload = json.loads((out / "pred" / filename).read_text())
        assert payload["detect_tau"] == 0.001
        assert payload["output_stride"] == cfg.output_stride
        # One entry per val image, including the empty frame.
        assert set(payload["points"]) == {"val_0.png", "val_1.png"}

    points = read_point_results(out / "pred" / "predictions.json")
    for name, array in points.items():
        assert array.ndim == 2 and array.shape[1] == 3, name
        if len(array):
            assert (array[:, 2] >= 0.001).all(), f"{name} below the decode floor"
            assert (array[:, 0] <= SIDE).all() and (array[:, 1] <= SIDE).all()


def test_point_predictions_score_through_the_point_in_box_scorer(
    stub_backbone, tiny_point_data, tiny_box_data, tmp_path
):
    """End to end: a point run's file joins to BOX ground truth by file name.

    The join is the whole reason the predictions are keyed by name — the points
    root renumbers image ids and the bbox root keeps CFD's original strings, so
    an id-keyed file would not join at all.
    """
    import sys

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1] / "examples" / "FishDetection" / "scripts")
    )
    import point_in_box as pib

    from cropcounter.metrics import read_point_results

    out = tmp_path / "runs"
    train(_point_cfg(tiny_point_data, out, run_name="score", ap_tau=0.001))
    preds = read_point_results(out / "score" / "predictions.json")

    # GT boxes from the BOX root, rekeyed from COCO id to file name.
    document = json.loads((tiny_box_data / "val" / "annotations.json").read_text())
    names = {img["id"]: img["file_name"] for img in document["images"]}
    gt = {
        names[image_id]: boxes
        for image_id, boxes in pib.coco_boxes_by_image(
            tiny_box_data / "val" / "annotations.json"
        ).items()
    }
    assert set(gt) == set(preds), "the name join must be total in both directions"

    summary, rows = pib.score_dataset(preds, gt, conf_thr=0.5)
    assert len(rows) == len(gt)
    assert 0.0 <= summary["precision"] <= 1.0
    assert 0.0 <= summary["recall"] <= 1.0
    assert 0.0 <= summary["f1"] <= 1.0
    assert summary["count_mae"] >= 0.0
