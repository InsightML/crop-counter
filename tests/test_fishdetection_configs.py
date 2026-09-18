"""The shipped ``examples/FishDetection`` configs must stay loadable and sane.

These are the files a GPU box is launched with; a typo in one of them is not
found until the box is up and the money is spent. Two failure modes are worth a
cheap test each:

* an unknown or renamed field — ``strict=True`` turns that into an error rather
  than a silently ignored key;
* a **point** config that forgets ``labels``. ``TrainConfig.labels`` defaults to
  ``COUNTED_LABELS`` — the WHEAT labels — so a fish point config without it
  filters every annotation away and trains, uncomplaining, on zero targets.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cropcounter.crop_dataset import COUNTED_LABELS
from cropcounter.dinov3_pyramid import STAGE_CHANNELS, PyramidDecoder
from cropcounter.train import TrainConfig, selection_key

CONFIG_DIR = Path(__file__).resolve().parents[1] / "examples" / "FishDetection"
CONFIGS = sorted(CONFIG_DIR.glob("config_*.json"))
RUN_B = CONFIG_DIR / "config_cfd17_points_tiny_taper_8ep.json"


def test_there_are_configs_to_check():
    assert CONFIGS, f"no config_*.json under {CONFIG_DIR}"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_every_shipped_config_loads_strictly(path):
    """No unknown fields, and every value coerces to its declared type."""
    cfg = TrainConfig.from_json(path, strict=True)
    assert cfg.task in ("point", "box")
    assert cfg.epochs >= 1 and cfg.batch_size >= 1
    # A bad select_on must fail here, not an epoch into the run.
    selection_key(cfg.select_on, cfg.task)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_a_fish_point_config_names_its_labels(path):
    """The silent-empty trap: wheat labels filtering a fish dataset to nothing."""
    cfg = TrainConfig.from_json(path, strict=True)
    if cfg.task != "point":
        return
    assert cfg.labels != COUNTED_LABELS, (
        f"{path.name} is a point config still carrying the default wheat labels "
        f"{COUNTED_LABELS} — every fish annotation would be filtered away"
    )
    assert cfg.labels is None or "fish" in cfg.labels


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_point_configs_read_keypoints_and_box_configs_read_boxes(path):
    """``coco`` is the keypoints loader; ``coco_bbox`` the detection one.

    Crossing the two is the failure that loads without error and trains on
    nothing (a bbox document carries no ``keypoints`` key at all).
    """
    cfg = TrainConfig.from_json(path, strict=True)
    if cfg.annotation_format == "coco_bbox":
        assert cfg.task == "box"
    elif cfg.annotation_format == "coco":
        assert cfg.task == "point"


def test_run_b_is_the_configuration_the_plan_authorised():
    """Pin run B: tapered point head on frozen Tiny, F1-selected, 8 epochs."""
    raw = json.loads(RUN_B.read_text())
    cfg = TrainConfig.from_json(RUN_B, strict=True)

    assert cfg.task == "point"
    assert cfg.annotation_format == "coco"
    assert cfg.labels == ("fish",)
    assert cfg.backbone == "tiny"
    assert cfg.backbone_trainable is False
    assert raw["c_dec"] == [512, 384, 192]
    assert cfg.output_stride == 4
    assert cfg.augment_profile == "natural"
    assert cfg.epochs == 8
    assert cfg.seed == 0
    assert cfg.select_on == "f1"
    assert selection_key(cfg.select_on, cfg.task) == ("f1", +1)

    # The agreed head-phase recipe: batch 32 with the LR scaled by sqrt(32/8),
    # validation every other epoch, workers to the p5.4xlarge's 16 vCPUs.
    assert cfg.batch_size == 32
    assert cfg.lr == pytest.approx(2e-3)
    assert cfg.val_freq == 2
    assert cfg.num_workers == 16
    # A low decode floor is what makes the offline tau sweep possible at all.
    assert cfg.ap_tau == pytest.approx(0.01)


def test_run_b_builds_the_decoder_the_budget_was_costed_from():
    cfg = TrainConfig.from_json(RUN_B, strict=True)
    decoder = PyramidDecoder(
        STAGE_CHANNELS[cfg.backbone], c_dec=cfg.c_dec,
        output_stride=cfg.output_stride, task=cfg.task,
    )
    assert decoder.level_widths == (512, 384, 192)
    assert decoder.geometry is None, "a point run must carry no geometry branch"
    params = sum(p.numel() for p in decoder.parameters())
    assert round(params / 1e6, 2) == 13.52
