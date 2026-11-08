"""One-stage Deformable DETR on a predicted feature grid, without an RGB backbone.

Adapted from fundamentalvision/Deformable-DETR (Apache-2.0).
Copyright (c) 2020 SenseTime. All Rights Reserved.
See APACHE-LICENSE in this package.
"""

import math

import torch
from torch import nn

from .geometry import normalize_hw

from .attention import MultiScaleDeformableAttention
from .boxes import box_cxcywh_to_xyxy


DETECTOR_FORMAT = "event-foresight-detector-v1"


def load_detector(checkpoint):
    """Load the exported RGB detector; no training framework or data required."""
    from .forecaster import read_weights
    saved = read_weights(checkpoint, DETECTOR_FORMAT)
    architecture = saved.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("Detector export has no architecture")
    expected = {"feature_dim", "grid_size", "hidden_dim", "num_classes", "num_queries",
                "num_levels", "encoder_layers", "decoder_layers", "heads", "points", "mlp_dim", "dropout"}
    if set(architecture) != expected:
        raise ValueError("Detector export architecture fields are incomplete or unexpected")
    if architecture["feature_dim"] != 3072 or tuple(architecture["grid_size"]) != (32, 64) or architecture["num_classes"] != 8:
        raise ValueError("Detector export does not match the RGB DSEC baseline feature/class contract")
    with torch.device("meta"):
        model = FutureDetector(**architecture)
    model.load_state_dict(saved["state_dict"], strict=True, assign=True)
    if any(value.is_meta for value in model.state_dict().values()):
        raise RuntimeError("Detector export did not materialize every tensor")
    model.architecture = architecture
    model.input_spec = saved.get("input_spec", {})
    model.provenance = saved.get("provenance", {})
    return model


def inverse_sigmoid(x):
    return torch.logit(x.float().clamp(1e-5, 1 - 1e-5))


def positions(height, width, dim, device):
    y, x = torch.meshgrid((torch.arange(height, device=device) + 0.5) / height,
                          (torch.arange(width, device=device) + 0.5) / width, indexing="ij")
    frequencies = 10000 ** (2 * (torch.arange(dim // 2, device=device) // 2) / (dim // 2))
    def encode(coordinates):
        phase = coordinates.flatten()[:, None] * (2 * math.pi) / frequencies
        return torch.stack((phase[:, 0::2].sin(), phase[:, 1::2].cos()), -1).flatten(1)
    return torch.cat((encode(y), encode(x)), -1), torch.stack((x.flatten(), y.flatten()), -1)


class EncoderLayer(nn.Module):
    def __init__(self, dim, heads, levels, points, mlp_dim, dropout):
        super().__init__()
        self.attention = MultiScaleDeformableAttention(dim, heads, levels, points)
        self.ffn = nn.Sequential(nn.Linear(dim, mlp_dim), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(mlp_dim, dim))
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, memory, position, reference, shapes):
        memory = self.norm1(memory + self.dropout(self.attention(memory + position, reference, memory, shapes)))
        return self.norm2(memory + self.dropout(self.ffn(memory)))


class DecoderLayer(nn.Module):
    def __init__(self, dim, heads, levels, points, mlp_dim, dropout):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_attention = MultiScaleDeformableAttention(dim, heads, levels, points)
        self.ffn = nn.Sequential(nn.Linear(dim, mlp_dim), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(mlp_dim, dim))
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(dim) for _ in range(3))
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, position, reference, memory, shapes):
        q = query + position
        query = self.norm1(query + self.dropout(self.self_attention(q, q, query, need_weights=False)[0]))
        query = self.norm2(query + self.dropout(
            self.cross_attention(query + position, reference, memory, shapes)))
        return self.norm3(query + self.dropout(self.ffn(query)))


class FutureDetector(nn.Module):
    def __init__(self, feature_dim=1024, grid_size=16, hidden_dim=256, num_classes=8,
                 num_queries=300, num_levels=4, encoder_layers=6, decoder_layers=6,
                 heads=8, points=4, mlp_dim=1024, dropout=0.1):
        super().__init__()
        self.grid_size, self.num_levels = grid_size, num_levels
        self.grid_hw = normalize_hw(grid_size, "grid_size")
        groups = math.gcd(32, hidden_dim)
        self.input_proj = nn.Sequential(nn.Conv2d(feature_dim, hidden_dim, 1), nn.GroupNorm(groups, hidden_dim))
        self.downsample = nn.ModuleList([
            nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
                          nn.GroupNorm(groups, hidden_dim)) for _ in range(num_levels - 1)
        ])
        self.level_embedding = nn.Parameter(torch.randn(num_levels, hidden_dim))
        layer_args = (hidden_dim, heads, num_levels, points, mlp_dim, dropout)
        self.encoder = nn.ModuleList([EncoderLayer(*layer_args) for _ in range(encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(*layer_args) for _ in range(decoder_layers)])
        self.query_embedding = nn.Embedding(num_queries, hidden_dim * 2)
        self.reference_points = nn.Linear(hidden_dim, 2)
        self.class_heads = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in self.decoder])
        self.box_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                          nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 4))
            for _ in self.decoder
        ])
        for head in self.class_heads:
            nn.init.constant_(head.bias, -math.log(99))
        for head in self.box_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        with torch.no_grad():
            self.box_heads[0][-1].bias[2:].fill_(-2)

    def forward(self, features):
        """[B,N,D] future features -> logits, normalized cxcywh boxes, auxiliary outputs."""
        if features.ndim != 3 or features.shape[1] != self.grid_hw[0] * self.grid_hw[1]:
            raise ValueError(f"features must have shape [B,{self.grid_hw[0] * self.grid_hw[1]},D] "
                             f"for grid_size {self.grid_hw}, got {tuple(features.shape)}")
        batch = features.shape[0]
        feature = self.input_proj(features.transpose(1, 2).reshape(batch, -1, *self.grid_hw))
        levels = [feature]
        for downsample in self.downsample:
            levels.append(downsample(levels[-1]))
        shapes = [level.shape[-2:] for level in levels]
        memory, position, centers = [], [], []
        for index, level in enumerate(levels):
            pos, center = positions(*level.shape[-2:], level.shape[1], level.device)
            memory.append(level.flatten(2).transpose(1, 2))
            position.append(pos + self.level_embedding[index])
            centers.append(center)
        memory = torch.cat(memory, 1)
        position = torch.cat(position)[None].to(memory.dtype)
        reference = torch.cat(centers)[None, :, None].expand(batch, -1, self.num_levels, -1)
        for layer in self.encoder:
            memory = layer(memory, position, reference, shapes)
        query_position, query = self.query_embedding.weight.chunk(2, -1)
        query_position = query_position[None].expand(batch, -1, -1)
        query = query[None].expand(batch, -1, -1)
        reference = self.reference_points(query_position).float().sigmoid()
        outputs = []
        for layer, class_head, box_head in zip(self.decoder, self.class_heads, self.box_heads):
            query = layer(query, query_position, reference[:, :, None].expand(-1, -1, self.num_levels, -1),
                          memory, shapes)
            delta = box_head(query).float()
            if reference.shape[-1] == 2:
                delta = torch.cat((delta[..., :2] + inverse_sigmoid(reference), delta[..., 2:]), -1)
            else:
                delta = delta + inverse_sigmoid(reference)
            boxes = delta.sigmoid()
            outputs.append({"pred_logits": class_head(query), "pred_boxes": boxes})
            reference = boxes.detach()
        return dict(outputs[-1], aux_outputs=outputs[:-1])


@torch.no_grad()
def postprocess(outputs, image_sizes=None, max_dets=100):
    """Return per-image boxes (original-pixel xyxy), labels and scores; no NMS/threshold.

    image_sizes is [B,2] in (height, width) order; defaults to DSEC's (480,640).
    """
    logits, boxes = outputs["pred_logits"].float(), outputs["pred_boxes"].float()
    scores, indices = logits.sigmoid().flatten(1).topk(min(max_dets, logits.shape[1] * logits.shape[2]), dim=1)
    labels, queries = indices % logits.shape[-1], indices // logits.shape[-1]
    boxes = box_cxcywh_to_xyxy(boxes).gather(1, queries[..., None].expand(-1, -1, 4))
    if image_sizes is None:
        image_sizes = boxes.new_tensor([[480, 640]]).expand(len(boxes), -1)
    image_sizes = torch.as_tensor(image_sizes, device=boxes.device)
    results = []
    for box, label, score, (height, width) in zip(boxes, labels, scores, image_sizes):
        box = box * torch.stack((width, height, width, height))
        box[:, 0::2] = box[:, 0::2].clamp(0, width - 1)
        box[:, 1::2] = box[:, 1::2].clamp(0, height - 1)
        keep = (box[:, 2:] > box[:, :2]).all(-1)
        results.append({"boxes": box[keep], "labels": label[keep], "scores": score[keep]})
    return results
