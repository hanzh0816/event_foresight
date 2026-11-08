"""Box geometry shared by matching, losses and postprocessing."""

import torch


def box_cxcywh_to_xyxy(boxes):
    center, size = boxes.split(2, dim=-1)
    return torch.cat((center - size / 2, center + size / 2), dim=-1)


def box_xyxy_to_cxcywh(boxes):
    low, high = boxes.split(2, dim=-1)
    return torch.cat(((low + high) / 2, high - low), dim=-1)


def generalized_box_iou(boxes1, boxes2):
    """Pairwise GIoU, including optional leading batch dimensions."""
    area1 = (boxes1[..., 2:] - boxes1[..., :2]).clamp_min(0).prod(-1)
    area2 = (boxes2[..., 2:] - boxes2[..., :2]).clamp_min(0).prod(-1)
    low = torch.maximum(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
    high = torch.minimum(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])
    intersection = (high - low).clamp_min(0).prod(-1)
    union = area1[..., :, None] + area2[..., None, :] - intersection
    low = torch.minimum(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
    high = torch.maximum(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])
    enclosing = (high - low).clamp_min(0).prod(-1)
    return intersection / union.clamp_min(1e-7) - (enclosing - union) / enclosing.clamp_min(1e-7)


def aligned_generalized_box_iou(boxes1, boxes2):
    """GIoU of corresponding pairs in O(N), without an N by N intermediate."""
    area1 = (boxes1[..., 2:] - boxes1[..., :2]).clamp_min(0).prod(-1)
    area2 = (boxes2[..., 2:] - boxes2[..., :2]).clamp_min(0).prod(-1)
    low = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    high = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    intersection = (high - low).clamp_min(0).prod(-1)
    union = area1 + area2 - intersection
    low = torch.minimum(boxes1[..., :2], boxes2[..., :2])
    high = torch.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclosing = (high - low).clamp_min(0).prod(-1)
    return intersection / union.clamp_min(1e-7) - (enclosing - union) / enclosing.clamp_min(1e-7)
