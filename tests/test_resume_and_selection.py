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

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from cropcounter.train import (
    TrainConfig,
    _is_better,
    _save_checkpoint,
    apply_overrides,
    build_optimizer_and_scheduler,
    load_backbone_from,
    load_checkpoint,
    main,
    selection_key,
)

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
