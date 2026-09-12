"""Unit tests for the optionally-trainable DINOv3 trunk.

The default is unchanged: the backbone is frozen, in eval mode, and produces
no gradients. ``backbone_trainable=True`` unfreezes the whole trunk and hands
the optimizer a layer-wise-decayed ladder of parameter groups — one group per
(depth, decay-or-not) pair, so early layers move slowly and no 1-D tensor is
weight-decayed.

CPU-light: every model here is built on the ``stub_backbone`` fixture, so no
DINOv3 checkpoint, no torch.hub fetch and no dataset are involved.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from cropcounter.dinov3_pyramid import (
    BACKBONE_DEPTHS,
    CropCounter,
    convnext_param_depth,
)
from cropcounter.losses import penalty_reduced_focal_loss
from cropcounter.train import SELECTION, TrainConfig, build_param_groups

#: One 64x64 RGB tile is enough: stride 32 still leaves a 2x2 map to fuse.
TILE = 64

#: base_lr x this ladder, stem -> top, at the default layer_decay of 0.8.
EXPECTED_LADDER = [0.8 ** 4, 0.8 ** 3, 0.8 ** 2, 0.8, 1.0]


def _model(trainable: bool) -> CropCounter:
    """A stub-trunk point counter, frozen or unfrozen."""
    return CropCounter(task="point", backbone_trainable=trainable)


def _one_training_step(model: CropCounter) -> None:
    """Forward + focal loss + backward on one random tile, so grads exist."""
    torch.manual_seed(0)
    images = torch.randn(1, 3, TILE, TILE)
    targets = torch.zeros(1, 1, TILE // 4, TILE // 4)
    targets[0, 0, 4, 4] = 1.0
    loss = penalty_reduced_focal_loss(model(images).float(), targets)
    loss.backward()


# --- 3, 4: the frozen default ------------------------------------------------

def test_frozen_backbone_has_no_trainable_parameters(stub_backbone):
    """The default build freezes the trunk and pins it to eval mode."""
    model = _model(trainable=False)
    assert not any(p.requires_grad for p in model.backbone.model.parameters())
    assert model.backbone.model.training is False

    model.train()
    assert model.backbone.model.training is False, "train() must not wake the trunk"
    assert model.decoder.training is True


def test_frozen_backbone_gets_no_gradients(stub_backbone):
    """A backward pass reaches the decoder and stops at the trunk."""
    model = _model(trainable=False)
    model.train()
    _one_training_step(model)

    assert all(p.grad is None for p in model.backbone.model.parameters())
    assert any(p.grad is not None for p in model.decoder.parameters())


# --- 5, 6, 7: the unfrozen trunk --------------------------------------------

def test_trainable_backbone_gets_gradients_everywhere(stub_backbone):
    """Every trunk tensor receives a gradient, and they are not all zero."""
    model = _model(trainable=True)
    model.train()
    _one_training_step(model)

    grads = [(name, p.grad) for name, p in model.backbone.model.named_parameters()]
    assert grads
    for name, grad in grads:
        assert grad is not None, name
    assert any(grad.abs().sum() > 0 for _name, grad in grads)


def test_trainable_backbone_forward_under_no_grad_builds_no_graph(stub_backbone):
    """Validation stays graph-free even with the trunk unfrozen."""
    model = _model(trainable=True)
    model.eval()
    with torch.no_grad():
        feats = model.backbone(torch.randn(1, 3, TILE, TILE))
        logits = model(torch.randn(1, 3, TILE, TILE))
    assert all(f.grad_fn is None for f in feats)
    assert logits.grad_fn is None


def test_train_mode_propagates_only_when_trainable(stub_backbone):
    """``train()``/``eval()`` reach the trunk only when it is unfrozen."""
    unfrozen = _model(trainable=True)
    unfrozen.train()
    assert unfrozen.backbone.model.training is True
    unfrozen.eval()
    assert unfrozen.backbone.model.training is False

    frozen = _model(trainable=False)
    frozen.train()
    assert frozen.backbone.model.training is False
    frozen.eval()
    assert frozen.backbone.model.training is False


# --- 8, 9, 10: the layer-wise parameter groups ------------------------------

def test_param_groups_lr_ladder(stub_backbone):
    """Decay groups climb base_lr x 0.8**(depth from the top), stem to norm."""
    model = _model(trainable=True)
    groups = model.backbone.param_groups(1e-4)

    decay_lrs = [g["lr"] for g in groups if g["name"].endswith(".decay")]
    assert decay_lrs == pytest.approx([1e-4 * f for f in EXPECTED_LADDER])
    assert [g["name"] for g in groups[:2]] == ["backbone.d0.decay", "backbone.d0.no_decay"]
    assert len(groups) == 2 * BACKBONE_DEPTHS


def test_layer_decay_one_flattens_the_ladder(stub_backbone):
    """``layer_decay=1.0`` is the no-decay escape hatch: one LR for the trunk."""
    model = _model(trainable=True)
    groups = model.backbone.param_groups(3e-5, layer_decay=1.0)
    assert {g["lr"] for g in groups} == {3e-5}


def test_param_groups_cover_every_tensor_exactly_once(stub_backbone):
    """The alias guard: ``norm`` is also ``norms[3]``, and AdamW rejects dupes."""
    model = _model(trainable=True)
    groups = model.backbone.param_groups(1e-4)

    grouped = [id(p) for g in groups for p in g["params"]]
    expected = [id(p) for p in model.backbone.model.parameters()]
    assert len(grouped) == len(set(grouped)), "a parameter landed in two groups"
    assert set(grouped) == set(expected)
    # The real failure mode this protects against: AdamW raises "some
    # parameters appear in more than one parameter group".
    torch.optim.AdamW(groups)


def test_one_dimensional_tensors_are_never_weight_decayed(stub_backbone):
    """Norm weights, biases and LayerScale gammas get weight_decay 0.0."""
    model = _model(trainable=True)
    for group in model.backbone.param_groups(1e-4, weight_decay=0.05):
        for param in group["params"]:
            assert (group["weight_decay"] == 0.0) == (param.ndim <= 1)


def test_param_groups_refuses_a_frozen_backbone(stub_backbone):
    """Asking a frozen trunk for optimizer groups is a bug, not a no-op."""
    model = _model(trainable=False)
    with pytest.raises(RuntimeError, match="frozen"):
        model.backbone.param_groups(1e-4)


def test_convnext_param_depth_maps_the_hub_layout():
    """The stem is 0; every downsample rides with the stage it feeds."""
    assert convnext_param_depth("downsample_layers.0.0.weight") == 0
    assert convnext_param_depth("stages.0.0.dwconv.weight") == 1
    assert convnext_param_depth("downsample_layers.2.1.weight") == 3
    assert convnext_param_depth("stages.2.0.gamma") == 3
    assert convnext_param_depth("stages.3.0.gamma") == 4
    assert convnext_param_depth("norm.weight") == 4
    assert convnext_param_depth("norms.3.bias") == 4


@pytest.mark.parametrize(
    "name", ["cls_token", "blocks.0.attn.qkv.weight", "stages.9.0.gamma", "downsample_layers"]
)
def test_convnext_param_depth_rejects_an_unknown_layout(name):
    """A hub layout change must fail loudly, not land in the wrong LR bucket."""
    with pytest.raises(KeyError):
        convnext_param_depth(name)


# --- 1, 2: the config fields ------------------------------------------------

def test_new_config_fields_default_to_todays_behaviour():
    """A default config is exactly the frozen, val-loss-selected run we ship."""
    cfg = TrainConfig()
    assert cfg.backbone_trainable is False
    assert cfg.backbone_lr == 2e-5
    assert cfg.backbone_layer_decay == 0.8
    assert cfg.backbone_weight_decay == 0.05
    assert cfg.select_on == "val_loss"
    assert cfg.resume_from is None


def test_resume_from_round_trips_as_a_path():
    """JSON stores the path as a string; from_dict must coerce it back to Path."""
    as_dict = TrainConfig(resume_from=Path("runs/cfd17/last.pt")).to_dict()
    assert as_dict["resume_from"] == "runs/cfd17/last.pt"

    restored = TrainConfig.from_dict(as_dict)
    assert restored.resume_from == Path("runs/cfd17/last.pt")
    assert isinstance(restored.resume_from, Path)


def test_shipped_configs_load_strictly_and_keep_the_defaults():
    """No shipped config silently changed behaviour when these fields landed.

    A config that names the new keys is an unfrozen/selection experiment and
    is allowed its own values; one that does not must come back at the
    frozen, val-loss defaults.
    """
    root = Path(__file__).resolve().parents[1]
    configs = sorted(root.glob("examples/*/*.json"))
    assert configs, "no example configs found"

    for path in configs:
        raw = TrainConfig.from_json(path, strict=True)
        assert raw.select_on in SELECTION, path
        assert isinstance(raw.backbone_trainable, bool), path
        with path.open(encoding="utf-8") as fh:
            keys = set(json.load(fh))
        if "backbone_trainable" not in keys:
            assert raw.backbone_trainable is False, path
        if "select_on" not in keys:
            assert raw.select_on == "val_loss", path


# --- 11: the optimizer's view of a frozen run -------------------------------

def test_frozen_build_param_groups_is_one_group_of_trainable_parameters(stub_backbone):
    """The frozen default hands AdamW exactly what it handed it before."""
    model = _model(trainable=False)
    groups = build_param_groups(model, TrainConfig())

    assert len(groups) == 1
    assert groups[0]["name"] == "decoder"
    assert [id(p) for p in groups[0]["params"]] == [
        id(p) for p in model.trainable_parameters()
    ]


def test_trainable_build_param_groups_adds_the_backbone_ladder(stub_backbone):
    """Unfrozen: one decoder group at cfg.lr, then the trunk's ten."""
    model = _model(trainable=True)
    cfg = TrainConfig(backbone_trainable=True, lr=1e-3, backbone_lr=2e-5)
    groups = build_param_groups(model, cfg)

    assert len(groups) == 1 + 2 * BACKBONE_DEPTHS
    assert groups[0]["lr"] == 1e-3
    assert groups[-1]["lr"] == pytest.approx(2e-5)
    torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
