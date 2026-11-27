"""Patch-region statistics shared by training validation and standalone evaluation."""

import torch
from torch.nn import functional as F


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
