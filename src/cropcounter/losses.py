"""Penalty-reduced focal loss (CenterNet variant) for peak heatmaps."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def penalty_reduced_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """Focal loss with penalty-reduced negatives, computed on logits.

    Positives are cells where the target is exactly 1.0 (the Gaussian
    peaks); every other cell is a negative down-weighted by
    ``(1 - target)^beta`` so cells near a peak are barely penalised.

    Args:
        logits: (B, 1, H, W) raw model outputs (pre-sigmoid).
        targets: (B, 1, H, W) max-composed Gaussian targets in [0, 1].
        alpha: focal exponent on the probability term.
        beta: penalty-reduction exponent on the negative weighting.

    Returns:
        Scalar loss, summed over the batch and normalised by the number of
        peaks (clamped to at least 1 so empty tiles remain well-defined).
    """
    prob = torch.sigmoid(logits)
    log_p = F.logsigmoid(logits)
    log_not_p = F.logsigmoid(-logits)

    pos_mask = targets == 1.0
    pos_loss = -((1.0 - prob) ** alpha) * log_p
    neg_loss = -((1.0 - targets) ** beta) * (prob ** alpha) * log_not_p

    loss = torch.where(pos_mask, pos_loss, neg_loss).sum()
    n_pos = pos_mask.sum().clamp(min=1)
    return loss / n_pos
