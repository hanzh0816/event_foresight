"""Feature regression and active-patch alignment in the frozen DINO space."""

import torch
from torch.nn import functional as F


def event_alignment_loss(events, targets, activity, *, beta=1.0, cosine_weight=0.2):
    if events.shape != targets.shape or activity.shape != targets.shape[:-1]:
        raise ValueError("Event alignment requires matching [B,T,N,D] features and [B,T,N] activity")
    targets = targets.detach().float()
    error = F.smooth_l1_loss(events.float(), targets, beta=beta, reduction="none").mean(-1)
    error = error + cosine_weight * (1 - F.cosine_similarity(events.float(), targets, dim=-1))
    mask = activity.to(error.dtype)
    # Each sample/time contributes equally; an empty active set contributes zero.
    return ((error * mask).sum(-1) / mask.sum(-1).clamp_min(1)).mean()


def foreground_mask(targets, grid_size, device):
    height, width = grid_size
    y, x = torch.meshgrid((torch.arange(height, device=device) + 0.5) / height,
                          (torch.arange(width, device=device) + 0.5) / width, indexing="ij")
    centers = torch.stack((x.flatten(), y.flatten()), -1)
    masks = []
    for target in targets:
        boxes = target["boxes"].to(device)
        lo, hi = boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2
        masks.append(((centers[:, None] >= lo) & (centers[:, None] <= hi)).all(-1).any(-1))
    return torch.stack(masks)


def latent_region_totals(prediction, target, targets, grid_size):
    """Return error sums/counts so unequal or empty foreground sets aggregate exactly."""
    error = (prediction.float() - target.float()).abs().mean(-1)
    cosine = F.cosine_similarity(prediction.float(), target.float(), dim=-1)
    fg = foreground_mask(targets, grid_size, prediction.device)
    result = {}
    for name, mask in (("all", torch.ones_like(fg)), ("foreground", fg), ("background", ~fg)):
        result[name] = torch.stack(((error * mask).sum(), (cosine * mask).sum(), mask.sum())).double()
    return result
