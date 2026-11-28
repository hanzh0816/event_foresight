"""Causal RGB/Event cross-attention, per-frame spatial attention and 3D RoPE."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def rope_3d_tables(coordinates):
    """Build FP32 cos/sin tables once for all layers using [N,t/y/x] positions.

    Time is measured in seconds, space in RGB patch coordinates. Each axis uses
    angular periods logarithmically spaced from 1e-2 to 1e2. Trigonometry is FP32.
    Spatial attention supplies zero time coordinates, leaving its first 16
    channels unchanged while rotating the same 24+24 spatial channels.
    """
    angles = []
    with torch.autocast(device_type=coordinates.device.type, enabled=False):
        for axis, width in enumerate((16, 24, 24)):
            periods = torch.logspace(-2, 2, width // 2, device=coordinates.device, dtype=torch.float32)
            angles.append(coordinates[:, axis].float()[:, None] * (2 * math.pi / periods))
        angles = torch.cat(angles, dim=-1)
        return angles.cos(), angles.sin()


def apply_rope_3d(tokens, tables):
    """Rotate Q/K with shared tables; feature rotations remain layer-specific."""
    cosine, sine = tables
    with torch.autocast(device_type=tokens.device.type, enabled=False):
        real, imaginary = tokens.float().reshape(*tokens.shape[:-1], 32, 2).unbind(-1)
        cosine = cosine.reshape(1, 1, tokens.shape[-2], 32)
        sine = sine.reshape(1, 1, tokens.shape[-2], 32)
        result = torch.stack((real * cosine - imaginary * sine,
                              real * sine + imaginary * cosine), dim=-1).flatten(-2)
    return result.to(tokens.dtype)


class DropPath(nn.Module):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, value):
        if not self.training or not self.probability:
            return value
        keep = 1 - self.probability
        mask = value.new_empty((value.shape[0],) + (1,) * (value.ndim - 1)).bernoulli_(keep)
        return value * mask / keep


class CausalCrossAttention(nn.Module):
    """Independent query/context normalization and one K/V projection per call."""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.query_norm, self.context_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.kv = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, query, context, query_rope, context_rope, prefixes, chunk_size):
        """Attend to CPU-planned (query start, query end, visible KV count) runs."""
        batch, count, dim = query.shape
        q = self.q(self.query_norm(query)).reshape(batch, count, self.heads, 64).transpose(1, 2)
        k, v = self.kv(self.context_norm(context)).reshape(batch, -1, 2, self.heads, 64).unbind(2)
        q = apply_rope_3d(q, query_rope)
        k = apply_rope_3d(k.transpose(1, 2), context_rope)
        v = v.transpose(1, 2)
        pieces = []
        for group_start, group_end, visible in prefixes:
            for start in range(group_start, group_end, chunk_size or count):
                end = min(start + (chunk_size or count), group_end)
                if visible:
                    value = F.scaled_dot_product_attention(
                        q[:, :, start:end], k[:, :, :visible], v[:, :, :visible],
                        dropout_p=self.dropout if self.training else 0.0)
                    value = self.proj(value.transpose(1, 2).reshape(batch, end - start, dim))
                else:
                    # An empty Event prefix contributes zero, including proj.bias.
                    # Empty sums keep K/V and their normalization in autograd so
                    # DDP still receives zero gradients for an unavailable branch.
                    value = q[:, :, start:end].transpose(1, 2).reshape(batch, end - start, dim) * 0
                    value = self.proj(value + k[:, :, :0].sum() + v[:, :, :0].sum()) * 0
                pieces.append(value)
        return F.dropout(torch.cat(pieces, dim=1), p=self.dropout, training=self.training)


class SpatialSelfAttention(nn.Module):
    """Each target frame is an independent batch item for spatial attention."""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.norm = nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)

    def forward(self, query, spatial_rope, chunk_size):
        batch, count, dim = query.shape
        spatial = len(spatial_rope[0])
        frames = count // spatial
        value = self.norm(query).reshape(batch * frames, spatial, dim)
        q, k, v = self.qkv(value).reshape(batch * frames, spatial, 3, self.heads, 64).unbind(2)
        q = apply_rope_3d(q.transpose(1, 2), spatial_rope)
        k = apply_rope_3d(k.transpose(1, 2), spatial_rope)
        pieces = []
        for start in range(0, spatial, chunk_size or spatial):
            end = min(start + (chunk_size or spatial), spatial)
            pieces.append(F.scaled_dot_product_attention(
                q[:, :, start:end], k, v.transpose(1, 2),
                dropout_p=self.dropout if self.training else 0.0))
        output = torch.cat(pieces, dim=2).transpose(1, 2).reshape(batch, count, dim)
        return F.dropout(self.proj(output), p=self.dropout, training=self.training)


class CrossAttentionBlock(nn.Module):
    """RGB cross-attention, gated Event cross-attention, spatial attention, FFN."""

    def __init__(self, dim, heads, mlp_ratio, dropout, event_enabled=False,
                 gate_init=0.1, drop_path=0.0):
        super().__init__()
        self.rgb_attention = CausalCrossAttention(dim, heads, dropout)
        self.event_attention = CausalCrossAttention(dim, heads, dropout) if event_enabled else None
        self.gamma = nn.Parameter(torch.tensor(float(gate_init))) if event_enabled else None
        self.spatial_attention = SpatialSelfAttention(dim, heads, dropout)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(int(dim * mlp_ratio), dim),
                                 nn.Dropout(dropout))
        self.drop_path = DropPath(drop_path)

    def forward(self, query, context, query_rope, context_rope, rgb_prefixes,
                spatial_rope, chunk_size, event_context=None, event_rope=None,
                event_prefixes=None):
        query = query + self.drop_path(self.rgb_attention(
            query, context, query_rope, context_rope, rgb_prefixes, chunk_size))
        if self.event_attention is not None and event_context is not None:
            query = query + self.drop_path(self.gamma.tanh() * self.event_attention(
                query, event_context, query_rope, event_rope, event_prefixes, chunk_size))
        query = query + self.drop_path(self.spatial_attention(query, spatial_rope, chunk_size))
        return query + self.drop_path(self.mlp(self.norm(query)))
