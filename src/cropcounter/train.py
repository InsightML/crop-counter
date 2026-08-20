"""Training loop for the DINOv3 pyramid-decoder crop emergence counter.

Plain-logging first pass: console output via tqdm, history saved as JSON
with matplotlib curves, best/last checkpoints per run directory.

Run a config file straight from the command line::

    python -m cropcounter.train --config examples/config_13ep.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .crop_dataset import (
    COUNTED_LABELS,
    IMAGES_DIRNAME,
    CropTileDataset,
    collate_val,
    load_splits,
)
from .dinov3_pyramid import CropCounter
from .losses import penalty_reduced_focal_loss
from .metrics import evaluate

#: TrainConfig fields that are paths on the dataclass but strings in JSON.
_PATH_FIELDS = ("data_root", "weights_dir", "out_dir")
#: TrainConfig fields that must be tuples, not the lists JSON round-trips to.
_TUPLE_FIELDS = ("exclude_label_statuses", "labels")


@dataclass
class TrainConfig:
    """All knobs for one training run. Defaults follow the NN-distance analysis."""

    # Paths (relative to the working directory the run starts from)
    data_root: Path = Path("data")
    weights_dir: Path = Path("weights")
    out_dir: Path = Path("runs")
    run_name: Optional[str] = None

    # Data format: "cvat" (CVAT for images 1.1), "coco" (keypoints), "datumaro"
    annotation_format: str = "cvat"

    # Model
    backbone: str = "base"
    c_dec: int = 192
    output_stride: int = 4

    # Targets & decoding
    sigma: float = 2.0          # output-grid cells
    k: int = 3                  # local-max kernel
    nms_radius: float = 1.5     # cells; kills plateau ties, keeps d>=2
    tau: float = 0.3            # decode threshold used during epoch val
    match_radius_px: float = 24.0

    # Data
    tile: int = 768
    tiles_per_image: int = 4
    scale_jitter: float = 0.25
    exclude_label_statuses: Tuple[str, ...] = ()
    num_workers: int = 4

    labels: Optional[Tuple[str, ...]] = COUNTED_LABELS  # Point labels to count

    # Optimisation
    batch_size: int = 8
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    grad_clip: float = 1.0
    focal_alpha: float = 2.0
    focal_beta: float = 4.0
    seed: int = 42

    # Runtime
    #: torch device string ("cuda", "mps", "cpu"); None auto-detects — see
    #: :func:`resolve_device`.
    device: Optional[str] = None

    # --- Split layout: data_root/{train,val}/{annotations file, images/} ---

    @property
    def train_dir(self) -> Path:
        return Path(self.data_root) / "train"

    @property
    def val_dir(self) -> Path:
        return Path(self.data_root) / "val"

    @property
    def train_images_dir(self) -> Path:
        return self.train_dir / IMAGES_DIRNAME

    @property
    def val_images_dir(self) -> Path:
        return self.val_dir / IMAGES_DIRNAME

    # --- (de)serialisation -------------------------------------------------

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], strict: bool = False) -> "TrainConfig":
        """Build a config from a plain dict (JSON, or a checkpoint payload).

        Coerces path and tuple fields back from their JSON forms. Unknown keys
        are dropped with a warning unless ``strict``, so configs saved by an
        older version of the package still load.
        """
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            if strict:
                raise TypeError(f"unknown TrainConfig field(s): {unknown}")
            print(f"TrainConfig: ignoring unknown field(s) {unknown}")
        clean = {k: v for k, v in raw.items() if k in known}
        for name in _PATH_FIELDS:
            if clean.get(name) is not None:
                clean[name] = Path(clean[name])
        for name in _TUPLE_FIELDS:
            if clean.get(name) is not None:
                clean[name] = tuple(clean[name])
        return cls(**clean)

    @classmethod
    def from_json(cls, path: Path, strict: bool = False) -> "TrainConfig":
        """Load a config from a JSON file, so configs live outside the code."""
        with Path(path).open(encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh), strict=strict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict: Paths as strings, tuples as lists."""
        out: Dict[str, Any] = {}
        for key, value in asdict(self).items():
            if isinstance(value, Path):
                out[key] = str(value)
            elif isinstance(value, tuple):
                out[key] = list(value)
            else:
                out[key] = value
        return out

    def to_json(self, path: Path) -> Path:
        """Write the config to a JSON file and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.write("\n")
        return path


def resolve_device(prefer: Optional[str] = None) -> torch.device:
    """Pick the compute device: an explicit override, else cuda > mps > cpu.

    Apple Silicon is a first-class target here — the backbone is frozen, so only
    the ~3M-parameter decoder trains — so MPS must be preferred over CPU rather
    than silently fallen back past.

    Args:
        prefer: an explicit torch device string; returned as-is when given.
    """
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(
    cfg: TrainConfig,
    device: Optional[torch.device] = None,
) -> Tuple[DataLoader, DataLoader, List, List]:
    """Read the on-disk train/val split and build the loaders.

    Args:
        cfg: the run config.
        device: the device batches will be copied to. Only CUDA supports pinned
            host memory; MPS warns on every loader it is requested for, so the
            flag is gated here. ``None`` resolves the device exactly as
            :func:`train` does.
    """
    device = device or resolve_device(cfg.device)
    pin = device.type == "cuda"
    train_recs, val_recs = load_splits(
        cfg.data_root, fmt=cfg.annotation_format, labels=cfg.labels
    )

    train_ds = CropTileDataset(
        train_recs, cfg.train_images_dir, train=True, tile=cfg.tile,
        output_stride=cfg.output_stride, sigma=cfg.sigma,
        tiles_per_image=cfg.tiles_per_image, scale_jitter=cfg.scale_jitter,
        exclude_label_statuses=cfg.exclude_label_statuses,
    )
    val_ds = CropTileDataset(
        val_recs, cfg.val_images_dir, train=False, output_stride=cfg.output_stride,
        sigma=cfg.sigma, exclude_label_statuses=cfg.exclude_label_statuses,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin, drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, collate_fn=collate_val,
        num_workers=min(cfg.num_workers, 2), pin_memory=pin,
        persistent_workers=cfg.num_workers > 0,
    )
    return train_loader, val_loader, train_recs, val_recs


def build_model(cfg: TrainConfig, device: torch.device) -> CropCounter:
    """Construct the model on the target device."""
    model = CropCounter(
        backbone_size=cfg.backbone, weights_dir=cfg.weights_dir,
        c_dec=cfg.c_dec, output_stride=cfg.output_stride,
    )
    return model.to(device)


def _save_checkpoint(model: CropCounter, cfg: TrainConfig, path: Path) -> None:
    torch.save({"decoder": model.decoder.state_dict(), "config": cfg.to_dict()}, path)


def load_checkpoint(
    path: Path,
    device: torch.device,
    weights_dir: Optional[Path] = None,
) -> Tuple[CropCounter, TrainConfig]:
    """Rebuild a CropCounter from a saved decoder checkpoint.

    Args:
        path: a ``best.pt`` / ``last.pt`` written by :func:`train`.
        device: device to build the model on.
        weights_dir: overrides the DINOv3 ``weights_dir`` stored in the
            checkpoint. The stored value is relative to wherever training ran,
            so loading from a different working directory needs this (or
            ``$CROPCOUNTER_WEIGHTS_DIR``, or a ``weights/`` folder in the cwd).

    Returns:
        ``(model, config)``. The model is on ``device``, decoder weights loaded;
        call ``.eval()`` before inference.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TrainConfig.from_dict(dict(payload["config"]))
    if weights_dir is not None:
        cfg.weights_dir = Path(weights_dir)
    model = build_model(cfg, device)
    model.decoder.load_state_dict(payload["decoder"])
    return model, cfg


def plot_history(history: Dict[str, List[float]], path: Path) -> None:
    """Save loss / MAE / F1 / LR curves as a single PNG."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Panel 0: train vs val focal loss together (val is tau-independent).
    axes[0].plot(epochs, history["train_loss"], color="#2b5f9e", label="train")
    axes[0].plot(epochs, history["val_loss"], color="#c1440e", label="val")
    axes[0].set_title("focal loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    panels = [
        ("val_count_mae", "val count MAE"),
        ("val_f1", "val localization F1"),
        ("lr", "learning rate"),
    ]
    for ax, (key, title) in zip(axes[1:], panels):
        ax.plot(epochs, history[key], color="#2b5f9e")
        ax.set_title(title)
        ax.set_xlabel("epoch")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def train(cfg: TrainConfig) -> Tuple[CropCounter, Dict[str, List[float]], str, int]:
    """Run a full training session; returns the model (best weights NOT
    auto-restored), the metric history, the run name, and the best epoch.
    Checkpoints and curves land in ``cfg.out_dir / run_name``."""
    device = resolve_device(cfg.device)
    seed_everything(cfg.seed)

    run_name = cfg.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(run_dir / "config.json")

    train_loader, val_loader, train_recs, val_recs = build_loaders(cfg, device)
    print(f"train: {len(train_recs)} images | val: {len(val_recs)} images "
          f"from {Path(cfg.data_root).absolute()}")

    model = build_model(cfg, device)
    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    print(f"backbone {cfg.backbone} frozen; decoder params: {n_trainable / 1e6:.2f}M "
          f"| device {device}")

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=max(cfg.warmup_epochs, 1)
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.epochs - cfg.warmup_epochs, 1)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[max(cfg.warmup_epochs, 1)]
    )

    use_amp = device.type == "cuda"
    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "val_count_mae": [], "val_count_rmse": [],
        "val_count_bias": [], "val_precision": [], "val_recall": [],
        "val_f1": [], "lr": [],
    }
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False)
        for images, targets, _ in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(images)
            loss = penalty_reduced_focal_loss(
                logits.float(), targets, alpha=cfg.focal_alpha, beta=cfg.focal_beta
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{epoch_loss / n_batches:.4f}")

        scheduler.step()
        train_loss = epoch_loss / max(n_batches, 1)

        summary, _ = evaluate(
            model, val_loader, device, tau=cfg.tau, k=cfg.k,
            nms_radius=cfg.nms_radius, output_stride=cfg.output_stride,
            match_radius_px=cfg.match_radius_px,
            focal_alpha=cfg.focal_alpha, focal_beta=cfg.focal_beta,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(summary["val_loss"])
        history["val_count_mae"].append(summary["count_mae"])
        history["val_count_rmse"].append(summary["count_rmse"])
        history["val_count_bias"].append(summary["count_bias"])
        history["val_precision"].append(summary["precision"])
        history["val_recall"].append(summary["recall"])
        history["val_f1"].append(summary["f1"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        marker = ""
        if summary["val_loss"] < best_val_loss:
            best_val_loss = summary["val_loss"]
            best_epoch = epoch
            _save_checkpoint(model, cfg, run_dir / "best.pt")
            marker = "  <- best"
        _save_checkpoint(model, cfg, run_dir / "last.pt")

        print(f"epoch {epoch:3d} | loss {train_loss:.4f} val {summary['val_loss']:.4f} | "
              f"val MAE {summary['count_mae']:.2f} RMSE {summary['count_rmse']:.2f} "
              f"bias {summary['count_bias']:+.2f} | "
              f"P {summary['precision']:.3f} R {summary['recall']:.3f} "
              f"F1 {summary['f1']:.3f} | lr {history['lr'][-1]:.2e}{marker}")

        with open(run_dir / "history.json", "w") as fh:
            json.dump(history, fh, indent=2)
        plot_history(history, run_dir / "curves.png")

    print(f"done. best val loss {best_val_loss:.4f} (epoch {best_epoch}) | "
          f"artifacts in {run_dir.resolve()}")
    return model, history, run_name, best_epoch


def main(argv: Optional[List[str]] = None) -> int:
    """CLI: ``python -m cropcounter.train --config config.json``."""
    parser = argparse.ArgumentParser(
        prog="python -m cropcounter.train",
        description="Train the pyramid decoder on top of a frozen DINOv3 backbone.",
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="JSON TrainConfig; omit to train with the defaults")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="override the config's data_root")
    parser.add_argument("--weights-dir", type=Path, default=None,
                        help="override the config's DINOv3 weights_dir")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="override the config's out_dir")
    parser.add_argument("--run-name", default=None, help="override the run name")
    parser.add_argument("--epochs", type=int, default=None, help="override the epoch count")
    parser.add_argument("--device", default=None,
                        help="force a torch device (cuda/mps/cpu); default auto-detects")
    args = parser.parse_args(argv)

    cfg = TrainConfig.from_json(args.config) if args.config else TrainConfig()
    for name in ("data_root", "weights_dir", "out_dir", "run_name", "epochs", "device"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)

    train(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
