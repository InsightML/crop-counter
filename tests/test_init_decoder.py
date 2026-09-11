"""Unit tests for ``init_decoder_from``: seeding a decoder from a checkpoint.

The setting exists for the linear probe. ``freeze_fusion`` only measures
whether frozen DINOv3 features are near-linearly box-decodable if the fusion
trunk being held still is a *trained* one; freezing a randomly initialised
trunk measures nothing. So the contract under test is: name-and-shape matched
tensors are taken from the checkpoint, everything else is left at its init
without raising, and the load happens before any freezing.

CPU-light: the helper is exercised on a bare :class:`PyramidDecoder`, so no
backbone, no DINOv3 weights, no dataset and no network are involved.
"""
from __future__ import annotations

from pathlib import Path

import torch

from cropcounter.dinov3_pyramid import STAGE_CHANNELS, PyramidDecoder
from cropcounter.train import TrainConfig, init_decoder_from_checkpoint

#: The wheat checkpoint's geometry: ConvNeXt-Base stages, c_dec 192, stride 4.
BASE_STAGES = STAGE_CHANNELS["base"]

#: The modules ``freeze_fusion`` holds still (``refine`` exists at stride 2 only).
FUSION_PREFIXES = ("laterals.", "blocks.", "refine.")


def _decoder(task: str, c_dec: int = 192, seed: int = 0) -> PyramidDecoder:
    """A decoder with distinctive weights, so equality assertions mean something."""
    torch.manual_seed(seed)
    decoder = PyramidDecoder(BASE_STAGES, c_dec=c_dec, output_stride=4, task=task)
    with torch.no_grad():
        for param in decoder.parameters():
            # Constant inits (head.bias = -4.0, the offset bias) would otherwise
            # compare equal between source and target by coincidence.
            param.add_(torch.randn_like(param))
    return decoder


def _write_checkpoint(decoder: PyramidDecoder, path: Path, **cfg_kwargs) -> Path:
    """Save in exactly the format ``train._save_checkpoint`` writes."""
    torch.save(
        {"decoder": decoder.state_dict(), "config": TrainConfig(**cfg_kwargs).to_dict()},
        path,
    )
    return path


def test_point_checkpoint_seeds_box_trunk(tmp_path):
    """A point checkpoint fills a box decoder's trunk and head, exactly."""
    source = _decoder("point", seed=1)
    ckpt = _write_checkpoint(source, tmp_path / "decoder_best.pt", task="point")

    target = _decoder("box", seed=2)
    geometry_before = {k: v.clone() for k, v in target.state_dict().items()
                       if k.startswith("geometry.")}

    report = init_decoder_from_checkpoint(target, ckpt)

    source_state = source.state_dict()
    assert report["loaded"] == sorted(source_state)
    assert report["shape_mismatch"] == []
    assert report["unexpected"] == []
    # The box branch has no counterpart in a point checkpoint, so it is
    # reported as left at init rather than silently zeroed.
    assert report["missing"] == sorted(geometry_before)
    assert all(key.startswith("geometry.") for key in report["missing"])

    loaded_state = target.state_dict()
    for key, tensor in source_state.items():
        assert torch.equal(loaded_state[key], tensor), key
    for key, tensor in geometry_before.items():
        assert torch.equal(loaded_state[key], tensor), key


def test_shape_mismatch_is_skipped_not_raised(tmp_path):
    """A c_dec-96 checkpoint contributes no weights to a c_dec-192 decoder."""
    source = _decoder("point", c_dec=96, seed=3)
    ckpt = _write_checkpoint(source, tmp_path / "narrow.pt", task="point", c_dec=96)

    target = _decoder("box", c_dec=192, seed=4)
    before = {k: v.clone() for k, v in target.state_dict().items()}

    report = init_decoder_from_checkpoint(target, ckpt)  # must not raise

    # ``head.bias`` is (1,) at any width, so it is the one genuinely
    # shape-compatible tensor; nothing that carries a c_dec dimension loads.
    assert set(report["loaded"]) <= {"head.bias"}
    assert set(report["shape_mismatch"]) == set(source.state_dict()) - {"head.bias"}
    after = target.state_dict()
    for key, tensor in before.items():
        if key in report["loaded"]:
            continue
        assert torch.equal(after[key], tensor), key


def test_init_then_freeze_trains_only_head_and_geometry(tmp_path):
    """build_model's order: seed the trunk, then freeze it — head + geometry train."""
    source = _decoder("point", seed=5)
    ckpt = _write_checkpoint(source, tmp_path / "decoder_best.pt", task="point")

    target = _decoder("box", seed=6)
    init_decoder_from_checkpoint(target, ckpt)
    target.freeze_fusion()

    trainable = {name for name, param in target.named_parameters() if param.requires_grad}
    expected = {name for name, _ in target.named_parameters()
                if name.startswith(("head.", "geometry."))}
    assert trainable == expected
    assert trainable, "a probe with nothing to train is not a probe"

    # The frozen tensors are the checkpoint's, which is the whole point: a
    # frozen RANDOM trunk is what made the first probe run uninformative.
    frozen_state = target.state_dict()
    source_state = source.state_dict()
    frozen = [name for name, param in target.named_parameters() if not param.requires_grad]
    assert frozen, "freeze_fusion froze nothing"
    for name in frozen:
        assert name.startswith(FUSION_PREFIXES), name
        assert torch.equal(frozen_state[name], source_state[name]), name


def test_bare_state_dict_checkpoint_also_works(tmp_path):
    """A payload that is just a state_dict loads too, not only the run format."""
    source = _decoder("point", seed=7)
    path = tmp_path / "bare.pt"
    torch.save(source.state_dict(), path)

    target = _decoder("box", seed=8)
    report = init_decoder_from_checkpoint(target, path)
    assert report["loaded"] == sorted(source.state_dict())


def test_init_decoder_from_defaults_to_none():
    """Nothing changes for runs that do not ask for it."""
    assert TrainConfig().init_decoder_from is None
    assert TrainConfig().to_dict()["init_decoder_from"] is None


def test_init_decoder_from_round_trips_as_a_path():
    """JSON stores the path as a string; from_dict must coerce it back to Path."""
    original = TrainConfig(init_decoder_from=Path("weights/decoder_best.pt"))
    as_dict = original.to_dict()
    assert as_dict["init_decoder_from"] == "weights/decoder_best.pt"

    restored = TrainConfig.from_dict(as_dict)
    assert restored.init_decoder_from == Path("weights/decoder_best.pt")
    assert isinstance(restored.init_decoder_from, Path)


def test_json_round_trip_through_a_file(tmp_path):
    """to_json -> from_json keeps the new path field intact."""
    path = TrainConfig(
        task="box", freeze_fusion=True, init_decoder_from=Path("weights/decoder_best.pt")
    ).to_json(tmp_path / "config.json")
    restored = TrainConfig.from_json(path, strict=True)
    assert restored.init_decoder_from == Path("weights/decoder_best.pt")
    assert restored.freeze_fusion is True


def test_shipped_configs_load_without_unknown_field_warning(capsys):
    """Every example config still loads strictly — no field was renamed."""
    examples = sorted(Path(__file__).resolve().parents[1].glob("examples/*/*.json"))
    assert examples, "no example configs found"
    for path in examples:
        TrainConfig.from_json(path, strict=True)
    assert capsys.readouterr().out == ""
