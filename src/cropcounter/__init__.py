"""PyTorch library for the DINOv3 ConvNeXt pyramid-decoder crop emergence counter."""
from importlib import import_module

__version__ = "0.1.0"

#: Public attribute -> the submodule that defines it. Resolved lazily (PEP 562)
#: rather than imported here, for two reasons:
#:
#: * ``python -m cropcounter.train`` / ``python -m cropcounter.weights`` import
#:   the package first and then execute the submodule as ``__main__``. If the
#:   package had already imported that submodule, runpy warns
#:   ("found in sys.modules ... prior to execution"), once per process — and
#:   again in every DataLoader worker that re-imports the main module.
#: * ``import cropcounter`` stays cheap: torch is only pulled in by the first
#:   attribute that actually needs it.
#:
#: ``from cropcounter import CropCounter, train, ...`` works exactly as before.
_LAZY_ATTRS = {
    # data
    "COUNTED_LABELS": "crop_dataset",
    "CropTileDataset": "crop_dataset",
    "ImageRecord": "crop_dataset",
    "Point": "crop_dataset",
    "collate_val": "crop_dataset",
    "load_records": "crop_dataset",
    "load_splits": "crop_dataset",
    "parse_coco_keypoints": "crop_dataset",
    "parse_cvat_1_1": "crop_dataset",
    "parse_datumaro": "crop_dataset",
    # model
    "CropCounter": "dinov3_pyramid",
    "DinoV3Backbone": "dinov3_pyramid",
    "PyramidDecoder": "dinov3_pyramid",
    # heatmap + loss + metrics
    "decode_peaks": "heatmap",
    "point_nms": "heatmap",
    "render_targets": "heatmap",
    "penalty_reduced_focal_loss": "losses",
    "evaluate": "metrics",
    "match_points": "metrics",
    "sweep_tau": "metrics",
    # inference
    "decode_in_bounds": "inference",
    "predict_prob": "inference",
    "records_from_folder": "inference",
    "save_visualization": "inference",
    "write_cvat_xml": "inference",
    # training
    "TrainConfig": "train",
    "build_loaders": "train",
    "build_model": "train",
    "load_checkpoint": "train",
    "resolve_device": "train",
    "train": "train",
    # backbone weights
    "DINOV3_DOWNLOAD_URL": "weights",
    "WEIGHT_FILES": "weights",
    "BackboneWeightsNotFound": "weights",
    "resolve_backbone_dir": "weights",
    "resolve_backbone_weights": "weights",
}


def __getattr__(name: str):
    """Import the submodule that owns ``name`` on first access (PEP 562)."""
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    # Cache as a package attribute. This also re-binds ``cropcounter.train`` to
    # the train *function*, which importing the same-named submodule shadows.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS))


try:
    # Optional: needs `pip install cropcounter[portal]` (requests + python-dotenv).
    # Guarded (and eager, unlike the rest) so `__all__` reflects whether the extra
    # is actually installed. Pulls in requests/PIL, never torch.
    from .portal import (  # noqa: F401 - re-exported below via __all__
        AuthError,
        MarkerDetectionError,
        PortalClient,
        PortalError,
        RateLimitError,
        WheatCountResult,
    )
    _HAVE_PORTAL = True
except ImportError:  # pragma: no cover - exercised by not installing the extra
    _HAVE_PORTAL = False

__all__ = [
    # model
    "DinoV3Backbone", "PyramidDecoder", "CropCounter",
    # heatmap + loss + metrics
    "decode_peaks", "point_nms", "render_targets",
    "penalty_reduced_focal_loss",
    "evaluate", "match_points", "sweep_tau",
    # training
    "TrainConfig", "build_loaders", "build_model", "load_checkpoint",
    "resolve_device", "train",
    # data
    "COUNTED_LABELS", "ImageRecord", "Point", "CropTileDataset", "collate_val",
    "load_records", "load_splits",
    "parse_cvat_1_1", "parse_coco_keypoints", "parse_datumaro",
    # inference
    "records_from_folder", "predict_prob", "decode_in_bounds",
    "save_visualization", "write_cvat_xml",
    # backbone weights
    "WEIGHT_FILES", "DINOV3_DOWNLOAD_URL", "BackboneWeightsNotFound",
    "resolve_backbone_dir", "resolve_backbone_weights",
    "__version__",
]

if _HAVE_PORTAL:
    __all__.extend([
        "PortalClient", "WheatCountResult",
        "PortalError", "AuthError", "RateLimitError", "MarkerDetectionError",
    ])
