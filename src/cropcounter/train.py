"""Training loop for the DINOv3 pyramid-decoder crop emergence counter.

Plain-logging first pass: console output via tqdm, history saved as JSON
with matplotlib curves, best/last checkpoints per run directory.

Run a config file straight from the command line::

    python -m cropcounter.train --config examples/config_13ep.json
    python -m cropcounter.train --config examples/config_box_brackish.json

``cfg.task`` picks the head: ``"point"`` trains the peak heatmap alone (every
number here is unchanged from before the detection head existed), ``"box"``
adds the size and offset branches and reports COCO AP alongside the same
P/R/F1 and count MAE the point task uses.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
    resolve_annotations,
)
from .det_metrics import evaluate_boxes, write_coco_results
from .dinov3_pyramid import CropCounter
from .losses import masked_l1_loss, penalty_reduced_focal_loss
from .metrics import evaluate

#: TrainConfig fields that are paths on the dataclass but strings in JSON.
_PATH_FIELDS = ("data_root", "weights_dir", "out_dir")
#: TrainConfig fields that must be tuples, not the lists JSON round-trips to.
_TUPLE_FIELDS = ("exclude_label_statuses",)


@dataclass
class TrainConfig:
    """All knobs for one training run. Defaults follow the NN-distance analysis."""

    # Paths (relative to the working directory the run starts from)
    data_root: Path = Path("data")
    weights_dir: Path = Path("weights")
    out_dir: Path = Path("runs")
    run_name: Optional[str] = None

    # Data format: "cvat" (CVAT 1.1), "coco" (keypoints), "coco_bbox" (boxes),
    # "datumaro"
    annotation_format: str = "cvat"

    #: "point" (peak heatmap only) or "box" (heatmap + wh + off).
    task: str = "point"

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

    # --- box task only ---
    #: Minimum fraction of a box that must survive a crop for it to be kept.
    min_bbox_visibility: float = 0.25
    #: "wheat" (any orientation) or "natural" (gravity prior: no vflip/rot90).
    augment_profile: str = "wheat"
    #: "log" stores log(size/stride); "linear" stores size/stride as CenterNet
    #: does — which is why CenterNet needs wh_weight down at 0.1.
    size_parameterisation: str = "log"
    wh_weight: float = 1.0
    off_weight: float = 1.0
    #: Detections kept per image, by score, before any suppression.
    top_k: int = 100
    #: IoU threshold for torchvision NMS on decoded boxes; None disables it.
    box_nms_iou: Optional[float] = None
    #: Probability a training draw takes an annotation-free image. Empty tiles
    #: are the point of a detection set, but an all-empty batch divides the
    #: focal loss by clamp(n_pos, 1) = 1 and spikes it — so their share is
    #: controlled here rather than left to the shuffle.
    negative_tile_fraction: float = 0.2
    #: Decode threshold for the AP detection list. Much lower than cfg.tau: AP
    #: wants a long low-confidence tail to trace the PR curve, while cfg.tau
    #: stays the operating threshold that P/R/F1 and the counts are read at.
    ap_tau: float = 0.01
    #: IoU a detection needs to count as a true positive in P/R/F1.
    match_iou: float = 0.5

    # Data
    tile: int = 768
    tiles_per_image: int = 4
    scale_jitter: float = 0.25
    exclude_label_statuses: Tuple[str, ...] = ()
    num_workers: int = 4

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
    #: Linear-probe variant: freeze the fusion trunk (laterals, ladder blocks,
    #: the stride-2 refine block) and train only head + geometry. Answers
    #: whether frozen DINOv3 features are near-linearly box-decodable through a
    #: fixed fuse trunk.
    freeze_fusion: bool = False

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
    # The box loaders keep every class; the point loaders keep COUNTED_LABELS.
    labels = None if cfg.task == "box" else COUNTED_LABELS
    train_recs, val_recs = load_splits(
        cfg.data_root, fmt=cfg.annotation_format, labels=labels
    )

    box_kwargs = dict(
        task=cfg.task, size_parameterisation=cfg.size_parameterisation,
        augment_profile=cfg.augment_profile,
        min_bbox_visibility=cfg.min_bbox_visibility,
        negative_tile_fraction=cfg.negative_tile_fraction,
    )
    train_ds = CropTileDataset(
        train_recs, cfg.train_images_dir, train=True, tile=cfg.tile,
        output_stride=cfg.output_stride, sigma=cfg.sigma,
        tiles_per_image=cfg.tiles_per_image, scale_jitter=cfg.scale_jitter,
        exclude_label_statuses=cfg.exclude_label_statuses,
        **box_kwargs,
    )
    val_ds = CropTileDataset(
        val_recs, cfg.val_images_dir, train=False, output_stride=cfg.output_stride,
        sigma=cfg.sigma, exclude_label_statuses=cfg.exclude_label_statuses,
        **box_kwargs,
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
        c_dec=cfg.c_dec, output_stride=cfg.output_stride, task=cfg.task,
    )
    if cfg.freeze_fusion:
        model.decoder.freeze_fusion()
    return model.to(device)


def _batch_loss(
    cfg: TrainConfig,
    outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
    targets: Union[torch.Tensor, Dict[str, torch.Tensor]],
) -> Tuple[torch.Tensor, int]:
    """One batch's loss, for either task, so the epoch loop body is shared.

    * ``"point"``: the penalty-reduced focal loss alone — numerically identical
      to what the point trainer computed before the box head existed.
    * ``"box"``: focal on the heatmap plus weighted masked-L1 on ``wh`` and
      ``off``, supervised only at the annotated centre cells.

    Returns:
        ``(loss, n_pos)`` — ``n_pos`` is the number of supervised centre cells
        in the batch. It is the diagnostic for the empty-image hazard: the
        focal loss divides by ``clamp(n_pos, 1)``, so a batch where it reads 0
        is a batch whose loss is on a different scale from its neighbours.
    """
    if cfg.task == "box":
        heatmap = outputs["heatmap"].float()
        mask = targets["mask"]
        loss = penalty_reduced_focal_loss(
            heatmap, targets["heatmap"], alpha=cfg.focal_alpha, beta=cfg.focal_beta
        )
        loss = loss + cfg.wh_weight * masked_l1_loss(
            outputs["wh"].float(), targets["wh"], mask
        )
        loss = loss + cfg.off_weight * masked_l1_loss(
            outputs["off"].float(), targets["off"], mask
        )
        return loss, int(mask.sum().item())

    logits = outputs.float()
    loss = penalty_reduced_focal_loss(
        logits, targets, alpha=cfg.focal_alpha, beta=cfg.focal_beta
    )
    return loss, int((targets == 1.0).sum().item())


def _targets_to_device(targets, device: torch.device):
    """Move a target tensor, or a box task's dict of them, to the device."""
    if isinstance(targets, dict):
        return {name: value.to(device, non_blocking=True) for name, value in targets.items()}
    return targets.to(device, non_blocking=True)


#: History keys written per epoch, by task, mapped to their summary key.
#: ``plot_history`` picks its panels off whichever set is present.
HISTORY_KEYS: Dict[str, Dict[str, str]] = {
    "point": {
        "train_loss": "", "val_loss": "val_loss", "val_count_mae": "count_mae",
        "val_count_rmse": "count_rmse", "val_count_bias": "count_bias",
        "val_precision": "precision", "val_recall": "recall", "val_f1": "f1",
        "lr": "",
    },
    "box": {
        "train_loss": "", "val_loss": "val_loss", "val_ap": "ap", "val_ap50": "ap50",
        "val_ap75": "ap75", "val_ar100": "ar100", "val_precision": "precision",
        "val_recall": "recall", "val_f1": "f1", "val_count_mae": "count_mae",
        "lr": "",
    },
}


def _append_history(
    history: Dict[str, List[float]],
    task: str,
    train_loss: float,
    lr: float,
    summary: Dict[str, float],
) -> None:
    """Append one epoch's row, in whichever key set the task uses."""
    history["train_loss"].append(float(train_loss))
    history["lr"].append(float(lr))
    for key, summary_key in HISTORY_KEYS[task].items():
        if summary_key:
            history[key].append(float(summary[summary_key]))


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
    """Save loss / quality / F1 / LR curves as a single PNG.

    Four panels for either task: the box run swaps the count-MAE panel for
    AP50, which is the number that decides whether the detection head is worth
    keeping. Panel selection is by key presence, so an old point history plots
    exactly as it always did.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Panel 0: train vs val loss together (val is tau-independent).
    axes[0].plot(epochs, history["train_loss"], color="#2b5f9e", label="train")
    axes[0].plot(epochs, history["val_loss"], color="#c1440e", label="val")
    axes[0].set_title("loss" if "val_ap50" in history else "focal loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    second = (("val_ap50", "val AP50") if "val_ap50" in history
              else ("val_count_mae", "val count MAE"))
    panels = [
        second,
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
    val_annotations: Optional[Path] = None
    if cfg.task == "box":
        # COCOeval scores against the val split's own annotation file, so our
        # AP and a published baseline's AP come out of the same scorer.
        val_annotations = resolve_annotations(cfg.val_dir, cfg.annotation_format)
        n_empty = sum(1 for r in train_recs if not r.boxes)
        print(f"box task | {n_empty}/{len(train_recs)} train images are empty; "
              f"sampling {cfg.negative_tile_fraction:.0%} negative tiles "
              f"| val annotations {val_annotations}")

    model = build_model(cfg, device)
    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    frozen_note = " (fusion trunk frozen)" if cfg.freeze_fusion else ""
    print(f"backbone {cfg.backbone} frozen; decoder params: {n_trainable / 1e6:.2f}M"
          f"{frozen_note} | device {device}")

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
    history: Dict[str, List[float]] = {key: [] for key in HISTORY_KEYS[cfg.task]}
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss, n_batches, epoch_pos = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False)
        for images, targets, _ in pbar:
            images = images.to(device, non_blocking=True)
            targets = _targets_to_device(targets, device)

            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(images)
            loss, n_pos = _batch_loss(cfg, outputs, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_pos += n_pos
            n_batches += 1
            postfix = {"loss": f"{epoch_loss / n_batches:.4f}"}
            if cfg.task == "box":
                # Objects per batch: if this trends to 0 the sampler, not the
                # model, is what to look at.
                postfix["n_pos"] = f"{epoch_pos / n_batches:.1f}"
            pbar.set_postfix(**postfix)

        scheduler.step()
        train_loss = epoch_loss / max(n_batches, 1)

        detections: List[Dict[str, Any]] = []
        if cfg.task == "box":
            summary, _, detections = evaluate_boxes(
                model, val_loader, device, gt=val_annotations,
                ap_tau=cfg.ap_tau, tau=cfg.tau, k=cfg.k, top_k=cfg.top_k,
                box_nms_iou=cfg.box_nms_iou, output_stride=cfg.output_stride,
                size_parameterisation=cfg.size_parameterisation,
                match_iou=cfg.match_iou,
                loss_fn=lambda out, tgt: _batch_loss(cfg, out, tgt)[0].item(),
            )
        else:
            summary, _ = evaluate(
                model, val_loader, device, tau=cfg.tau, k=cfg.k,
                nms_radius=cfg.nms_radius, output_stride=cfg.output_stride,
                match_radius_px=cfg.match_radius_px,
                focal_alpha=cfg.focal_alpha, focal_beta=cfg.focal_beta,
            )

        _append_history(history, cfg.task, train_loss,
                        optimizer.param_groups[0]["lr"], summary)

        marker = ""
        if summary["val_loss"] < best_val_loss:
            best_val_loss = summary["val_loss"]
            best_epoch = epoch
            _save_checkpoint(model, cfg, run_dir / "best.pt")
            if cfg.task == "box":
                write_coco_results(detections, run_dir / "predictions.json")
            marker = "  <- best"
        _save_checkpoint(model, cfg, run_dir / "last.pt")

        if cfg.task == "box":
            print(f"epoch {epoch:3d} | loss {train_loss:.4f} val {summary['val_loss']:.4f} | "
                  f"AP {summary['ap']:.3f} AP50 {summary['ap50']:.3f} "
                  f"AP75 {summary['ap75']:.3f} AR100 {summary['ar100']:.3f} | "
                  f"P {summary['precision']:.3f} R {summary['recall']:.3f} "
                  f"F1 {summary['f1']:.3f} MAE {summary['count_mae']:.2f} | "
                  f"lr {history['lr'][-1]:.2e}{marker}")
        else:
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
