"""Causal RGB/Event cross-attention, per-frame spatial attention and 3D RoPE."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def apply_rope_3d(tokens, coordinates):
    """Rotate 64 channels as time=16, y=24, x=24; coordinates are [N,t/y/x].

    Time is measured in seconds, space in RGB patch coordinates. Each axis uses
    angular periods logarithmically spaced from 1e-2 to 1e2. Trigonometry is FP32.
    Spatial attention supplies zero time coordinates, leaving its first 16
    channels unchanged while rotating the same 24+24 spatial channels.
    """
    if tokens.ndim != 4 or tokens.shape[-1] != 64:
        raise ValueError("3D RoPE requires [B,heads,N,64]")
    if tuple(coordinates.shape) != (tokens.shape[-2], 3):
        raise ValueError("RoPE coordinates must have shape [N,3]")
    pieces, start = [], 0
    with torch.autocast(device_type=tokens.device.type, enabled=False):
        for axis, width in enumerate((16, 24, 24)):
            periods = torch.logspace(-2, 2, width // 2, device=tokens.device, dtype=torch.float32)
            angles = coordinates[:, axis].float()[:, None] * (2 * math.pi / periods)
            cosine, sine = angles.cos()[None, None], angles.sin()[None, None]
            pairs = tokens[..., start:start + width].float().reshape(*tokens.shape[:-1], width // 2, 2)
            real, imaginary = pairs.unbind(-1)
            pieces.append(torch.stack((real * cosine - imaginary * sine,
                                       real * sine + imaginary * cosine), dim=-1).flatten(-2))
            start += width
    return torch.cat(pieces, dim=-1).to(tokens.dtype)


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

    def forward(self, query, context, query_coordinates, context_coordinates,
                observation_times, context_times, chunk_size):
        batch, count, dim = query.shape
        q = self.q(self.query_norm(query)).reshape(batch, count, self.heads, 64).transpose(1, 2)
        k, v = self.kv(self.context_norm(context)).reshape(batch, -1, 2, self.heads, 64).unbind(2)
        q = apply_rope_3d(q, query_coordinates)
        k = apply_rope_3d(k.transpose(1, 2), context_coordinates)
        pieces = []
        for start in range(0, count, chunk_size or count):
            end = min(start + (chunk_size or count), count)
            # Availability is bounded by the input cutoff, never by the future
            # query timestamp. This also handles event endpoints between RGBs.
            allowed = context_times[None] <= observation_times[start:end, None]
            value = F.scaled_dot_product_attention(
                q[:, :, start:end], k, v.transpose(1, 2), attn_mask=allowed,
                dropout_p=self.dropout if self.training else 0.0)
            value = self.proj(value.transpose(1, 2).reshape(batch, end - start, dim))
            # If no event has arrived at a query's cutoff, the branch contributes
            # zero including the output projection bias.
            value = value * allowed.any(-1)[None, :, None]
            pieces.append(value)
        return F.dropout(torch.cat(pieces, dim=1), p=self.dropout, training=self.training)


class SpatialSelfAttention(nn.Module):
    """Each target frame is an independent batch item for spatial attention."""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.norm = nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)

    def forward(self, query, spatial_coordinates, chunk_size):
        batch, count, dim = query.shape
        spatial = len(spatial_coordinates)
        frames = count // spatial
        value = self.norm(query).reshape(batch * frames, spatial, dim)
        q, k, v = self.qkv(value).reshape(batch * frames, spatial, 3, self.heads, 64).unbind(2)
        q = apply_rope_3d(q.transpose(1, 2), spatial_coordinates)
        k = apply_rope_3d(k.transpose(1, 2), spatial_coordinates)
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

    def forward(self, query, context, query_coordinates, context_coordinates,
                observation_times, context_times, spatial_coordinates, chunk_size,
                event_context=None, event_coordinates=None, event_times=None):
        query = query + self.drop_path(self.rgb_attention(
            query, context, query_coordinates, context_coordinates,
            observation_times, context_times, chunk_size))
        if self.event_attention is not None and event_context is not None:
            query = query + self.drop_path(self.gamma.tanh() * self.event_attention(
                query, event_context, query_coordinates, event_coordinates,
                observation_times, event_times, chunk_size))
        query = query + self.drop_path(self.spatial_attention(query, spatial_coordinates, chunk_size))
        return query + self.drop_path(self.mlp(self.norm(query)))
