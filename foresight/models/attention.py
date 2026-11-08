"""Pure PyTorch multi-scale deformable attention.

Adapted from fundamentalvision/Deformable-DETR (Apache-2.0).
Copyright (c) 2020 SenseTime. All Rights Reserved.
See APACHE-LICENSE in this package.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def deformable_attention(value, spatial_shapes, sampling_locations, attention_weights):
    """[B,S,heads,dim] values sampled at [B,Q,heads,levels,points,2] locations."""
    batch, _, heads, dim = value.shape
    queries = sampling_locations.shape[1]
    # grid_sample needs FP32 on CPU and this also avoids BF16 coordinate rounding.
    with torch.autocast(device_type=value.device.type, enabled=False):
        values = value.float().split([h * w for h, w in spatial_shapes], dim=1)
        grids = sampling_locations.float() * 2 - 1
        samples = []
        for level, (height, width) in enumerate(spatial_shapes):
            feature = values[level].flatten(2).transpose(1, 2).reshape(batch * heads, dim, height, width)
            grid = grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
            samples.append(F.grid_sample(feature, grid, mode="bilinear", padding_mode="zeros",
                                         align_corners=False))
        weights = attention_weights.float().transpose(1, 2).reshape(batch * heads, 1, queries, -1)
        output = (torch.stack(samples, dim=-2).flatten(-2) * weights).sum(-1)
        return output.view(batch, heads * dim, queries).transpose(1, 2).contiguous()


class MultiScaleDeformableAttention(nn.Module):
    def __init__(self, dim=256, heads=8, levels=4, points=4):
        super().__init__()
        self.heads, self.levels, self.points = heads, levels, points
        self.value_proj = nn.Linear(dim, dim)
        self.sampling_offsets = nn.Linear(dim, heads * levels * points * 2)
        self.attention_weights = nn.Linear(dim, heads * levels * points)
        self.output_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.sampling_offsets.weight)
        angles = torch.arange(heads).float() * (2 * math.pi / heads)
        directions = torch.stack((angles.cos(), angles.sin()), -1)
        directions = directions / directions.abs().amax(-1, keepdim=True)
        offsets = directions[:, None, None, :].repeat(1, levels, points, 1)
        offsets *= torch.arange(1, points + 1)[None, None, :, None]
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(offsets.flatten())
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        for layer in (self.value_proj, self.output_proj):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, query, reference_points, memory, spatial_shapes):
        batch, queries, _ = query.shape
        value = self.value_proj(memory).unflatten(-1, (self.heads, -1))
        # Compute offsets/weights and coordinates in FP32 under mixed precision too.
        with torch.autocast(device_type=query.device.type, enabled=False):
            offsets = self.sampling_offsets(query.float()).view(
                batch, queries, self.heads, self.levels, self.points, 2)
            weights = self.attention_weights(query.float()).view(batch, queries, self.heads, -1)
            weights = weights.softmax(-1).view(batch, queries, self.heads, self.levels, self.points)
            reference_points = reference_points.float()
            if reference_points.shape[-1] == 2:
                normalizer = query.new_tensor([(w, h) for h, w in spatial_shapes], dtype=torch.float32)
                locations = reference_points[:, :, None, :, None, :] + offsets / normalizer[None, None, None, :, None]
            else:
                locations = (reference_points[:, :, None, :, None, :2]
                             + offsets / self.points * reference_points[:, :, None, :, None, 2:] * 0.5)
            output = deformable_attention(value, spatial_shapes, locations, weights)
        return self.output_proj(output)
