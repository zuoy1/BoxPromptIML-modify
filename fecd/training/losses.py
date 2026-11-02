"""Imagewise class-balanced masked hard-target BCE for controlled baselines."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def class_balanced_masked_bce(logits, target, selection):
    """Normalize each selected class per image, then average fixed .5/.5 sums.

    A missing selected class contributes zero with no redistribution of its
    weight. Details' pos_mean/neg_mean include those zeros in the image mean.
    Empty selections retain a differentiable zero and zero ignored gradients.
    """
    if (logits.ndim != 4 or logits.shape[1] != 1 or logits.numel() == 0
            or logits.shape != target.shape or logits.shape != selection.shape):
        raise ValueError('Inputs must share nonempty [B,1,H,W] shape')
    if (logits.device != target.device or logits.device != selection.device
            or not logits.is_floating_point()):
        raise ValueError('Inputs must share device and logits must be floating point')
    if selection.dtype != torch.bool:
        raise ValueError('Selection mask must be boolean')
    if (not torch.isfinite(logits).all() or not torch.isfinite(target).all()
            or not torch.all((target == 0) | (target == 1))):
        raise ValueError('Logits must be finite and hard targets finite binary')
    target_float = target.to(dtype=torch.float32)
    pixel_loss = F.binary_cross_entropy_with_logits(logits.float(), target_float, reduction='none')
    positive = selection & (target == 1)
    negative = selection & (target == 0)
    axes = (1,2,3)
    pos_count = positive.sum(dim=axes)
    neg_count = negative.sum(dim=axes)
    pos_per_image = (pixel_loss * positive).sum(dim=axes) / pos_count.clamp_min(1)
    neg_per_image = (pixel_loss * negative).sum(dim=axes) / neg_count.clamp_min(1)
    pos_mean = pos_per_image.mean()
    neg_mean = neg_per_image.mean()
    pos_contribution = .5 * pos_mean
    neg_contribution = .5 * neg_mean
    loss = pos_contribution + neg_contribution
    details = {
        'pos_mean':pos_mean.detach(), 'neg_mean':neg_mean.detach(),
        'pos_contribution':pos_contribution.detach(),
        'neg_contribution':neg_contribution.detach(),
        'selected_pos':pos_count.sum().detach(), 'selected_neg':neg_count.sum().detach(),
        'selected_pos_per_image':pos_count.detach(),
        'selected_neg_per_image':neg_count.detach(),
        'valid_pos_images':(pos_count > 0).sum().detach(),
        'valid_neg_images':(neg_count > 0).sum().detach(),
    }
    return loss, details
