"""Hungarian matching and focal/L1/GIoU set losses.

Adapted from fundamentalvision/Deformable-DETR (Apache-2.0).
Copyright (c) 2020 SenseTime. All Rights Reserved.
See APACHE-LICENSE in this package.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import distributed as dist, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from .boxes import aligned_generalized_box_iou, box_cxcywh_to_xyxy, generalized_box_iou


class DetectionCriterion(nn.Module):
    def __init__(self, class_weight=2.0, bbox_weight=5.0, giou_weight=2.0, alpha=0.25, gamma=2.0):
        super().__init__()
        self.weights = {"loss_class": class_weight, "loss_bbox": bbox_weight, "loss_giou": giou_weight}
        self.alpha, self.gamma = alpha, gamma

    @torch.no_grad()
    def match(self, outputs, targets, packed=None):
        labels, target_boxes, sizes = packed if packed is not None else self.pack_targets(targets)
        logits, boxes = outputs["pred_logits"].float(), outputs["pred_boxes"].float()
        probability = logits.sigmoid()
        negative = (1 - self.alpha) * probability.pow(self.gamma) * -(1 - probability + 1e-8).log()
        positive = self.alpha * (1 - probability).pow(self.gamma) * -(probability + 1e-8).log()
        class_cost = (positive - negative).gather(2, labels[:, None].expand(-1, boxes.shape[1], -1))
        bbox_cost = torch.cdist(boxes, target_boxes, p=1)
        giou_cost = -generalized_box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(target_boxes))
        cost = (self.weights["loss_class"] * class_cost + self.weights["loss_bbox"] * bbox_cost
                + self.weights["loss_giou"] * giou_cost)
        # One device-to-host synchronization per layer, rather than per image.
        costs = cost.cpu().numpy()
        assignments = [linear_sum_assignment(matrix[:, :size]) for matrix, size in zip(costs, sizes)]
        lengths = [len(rows) for rows, _ in assignments]
        indices = torch.as_tensor(np.concatenate([np.stack(pair) for pair in assignments], axis=1),
                                  device=boxes.device)
        return [(pair[0], pair[1]) for pair in indices.split(lengths, dim=1)]

    @staticmethod
    def pack_targets(targets):
        return (pad_sequence([target["labels"] for target in targets], batch_first=True),
                pad_sequence([target["boxes"].float() for target in targets], batch_first=True),
                [len(target["labels"]) for target in targets])

    def layer_loss(self, outputs, targets, normalizer, packed=None):
        packed = self.pack_targets(targets) if packed is None else packed
        matches = self.match(outputs, targets, packed)
        logits, boxes = outputs["pred_logits"].float(), outputs["pred_boxes"].float()
        class_targets = torch.zeros_like(logits)
        batch_indices = torch.tensor([index for index, (source, _) in enumerate(matches) for _ in range(len(source))],
                                     device=boxes.device, dtype=torch.long)
        source_indices = torch.cat([source for source, _ in matches])
        target_indices = torch.cat([destination for _, destination in matches])
        labels, target_boxes, _ = packed
        class_targets[batch_indices, source_indices, labels[batch_indices, target_indices]] = 1
        probability = logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logits, class_targets, reduction="none")
        pt = probability * class_targets + (1 - probability) * (1 - class_targets)
        alpha = self.alpha * class_targets + (1 - self.alpha) * (1 - class_targets)
        class_loss = (ce * (1 - pt).pow(self.gamma) * alpha).sum() / normalizer
        source = boxes[batch_indices, source_indices]
        destination = target_boxes[batch_indices, target_indices]
        bbox_loss = F.l1_loss(source, destination, reduction="sum") / normalizer
        giou = aligned_generalized_box_iou(box_cxcywh_to_xyxy(source), box_cxcywh_to_xyxy(destination))
        giou_loss = (1 - giou).sum() / normalizer
        return {"loss_class": class_loss, "loss_bbox": bbox_loss, "loss_giou": giou_loss}

    def forward(self, outputs, targets, sync_num_boxes=False):
        with torch.autocast(device_type=outputs["pred_logits"].device.type, enabled=False):
            normalizer = outputs["pred_boxes"].new_tensor(float(sum(len(t["labels"]) for t in targets)))
            if sync_num_boxes and dist.is_available() and dist.is_initialized():
                dist.all_reduce(normalizer)
                normalizer /= dist.get_world_size()
            normalizer = normalizer.clamp_min(1)
            packed = self.pack_targets(targets)
            losses = self.layer_loss(outputs, targets, normalizer, packed)
            total = sum(self.weights[key] * value for key, value in losses.items())
            for index, auxiliary in enumerate(outputs.get("aux_outputs", [])):
                layer_losses = self.layer_loss(auxiliary, targets, normalizer, packed)
                total = total + sum(self.weights[key] * value for key, value in layer_losses.items())
                losses.update({f"{key}_{index}": value for key, value in layer_losses.items()})
            return dict(losses, loss=total)
