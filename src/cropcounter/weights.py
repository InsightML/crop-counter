"""Locating the DINOv3 ConvNeXt backbone checkpoints on disk.

The backbone weights are a **gated** download from Meta: they are not on PyPI,
not in this repository, and this package will never fetch them for you. This
module only resolves where they already are, and — when they are missing —
raises an error that says exactly what to download and where to put it.

Stdlib only, so ``dinov3_pyramid`` can import it without a cycle.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

#: Backbone size -> the filename Meta ships for the LVD-1689M pretrained ConvNeXt.
WEIGHT_FILES = {
    "tiny": "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
    "small": "dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth",
    "base": "dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth",
    "large": "dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth",
}

#: Meta's gated download page — request access, then download the ConvNeXt checkpoints.
DINOV3_DOWNLOAD_URL = "https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/"

#: Environment variable checked when no ``weights_dir`` is passed explicitly.
WEIGHTS_DIR_ENV = "CROPCOUNTER_WEIGHTS_DIR"

#: Default directory, relative to the current working directory.
DEFAULT_WEIGHTS_DIR = Path("weights")


class BackboneWeightsNotFound(FileNotFoundError):
    """Raised when the DINOv3 backbone checkpoint is not on disk."""


def backbone_filename(size: str) -> str:
    """Return the expected checkpoint filename for a backbone size."""
    if size not in WEIGHT_FILES:
        raise ValueError(f"backbone size must be one of {sorted(WEIGHT_FILES)}, got {size!r}")
    return WEIGHT_FILES[size]


def candidate_dirs(weights_dir: Optional[Path] = None) -> list[Path]:
    """Directories searched for backbone checkpoints, in priority order.

    Explicit argument first, then ``$CROPCOUNTER_WEIGHTS_DIR``, then ``./weights``.
    """
    candidates: list[Path] = []
    if weights_dir is not None:
        candidates.append(Path(weights_dir))
    env_dir = os.environ.get(WEIGHTS_DIR_ENV)
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(DEFAULT_WEIGHTS_DIR)

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = Path(path).expanduser()
        key = resolved.absolute()
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return unique


def download_instructions(size: str, searched: list[Path]) -> str:
    """The message shown when a backbone checkpoint cannot be found."""
    filename = backbone_filename(size)
    looked = "\n".join(f"  - {p.absolute()}" for p in searched)
    return (
        f"DINOv3 backbone weights not found for size {size!r}.\n\n"
        f"Expected file: {filename}\n"
        f"Searched:\n{looked}\n\n"
        "The DINOv3 weights are a gated download from Meta and are deliberately not\n"
        "bundled with this package. To get them:\n"
        f"  1. Request access and download the ConvNeXt checkpoints from\n"
        f"     {DINOV3_DOWNLOAD_URL}\n"
        f"  2. Put the .pth file in a 'weights/' directory next to where you run from,\n"
        f"     or point {WEIGHTS_DIR_ENV} at the directory holding it, or pass\n"
        f"     weights_dir=... to CropCounter/DinoV3Backbone.\n\n"
        "Expected filenames:\n"
        + "\n".join(f"  {name:<6} {fn}" for name, fn in WEIGHT_FILES.items())
    )


def resolve_backbone_dir(size: str = "base", weights_dir: Optional[Path] = None) -> Path:
    """Return the first directory that actually holds the backbone checkpoint.

    Args:
        size: backbone size whose checkpoint must be present.
        weights_dir: explicit directory to try first; ``None`` falls back to
            ``$CROPCOUNTER_WEIGHTS_DIR`` then ``./weights``.

    Raises:
        BackboneWeightsNotFound: with download instructions, if no candidate
            directory contains the file. Nothing is ever downloaded.
    """
    filename = backbone_filename(size)
    searched = candidate_dirs(weights_dir)
    for directory in searched:
        if (directory / filename).is_file():
            return directory
    raise BackboneWeightsNotFound(download_instructions(size, searched))


def resolve_backbone_weights(size: str = "base", weights_dir: Optional[Path] = None) -> Path:
    """Return the full path to the backbone checkpoint for ``size``.

    Same ``(size, weights_dir)`` argument order as :func:`resolve_backbone_dir`.
    """
    return resolve_backbone_dir(size, weights_dir) / backbone_filename(size)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI: report where the backbone weights are, or how to get them.

    ``python -m cropcounter.weights [--size base] [--weights-dir DIR]``
    """
    parser = argparse.ArgumentParser(
        prog="python -m cropcounter.weights",
        description="Locate the gated DINOv3 ConvNeXt backbone checkpoints.",
    )
    parser.add_argument("--size", default="base", choices=sorted(WEIGHT_FILES))
    parser.add_argument("--weights-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        path = resolve_backbone_weights(args.size, args.weights_dir)
    except BackboneWeightsNotFound as exc:
        print(exc, file=sys.stderr)
        return 1
    print(path.absolute())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
