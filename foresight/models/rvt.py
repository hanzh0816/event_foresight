"""RVT event backbone and the v3.1 multi-scale DINO feature neck.

Independent implementation of the recurrent, partitioned-attention architecture
described by Gehrig and Scaramuzza (CVPR 2023). Architecture settings were checked
against https://github.com/uzh-rpg/RVT (MIT); no upstream source is vendored.
The FPN and feature projection follow the project's research-v3.1.md.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


LSTMState = tuple[Tensor, Tensor]
RVTState = tuple[LSTMState, ...]


def partition_tokens(x: Tensor, size: tuple[int, int], grid: bool) -> Tensor:
    """Group NHWC features into contiguous blocks or dilated global grids."""
    batch, height, width, channels = x.shape
    ph, pw = size
    if height % ph or width % pw:
        raise ValueError(f"Feature grid {(height, width)} is not divisible by {size}")
    if grid:
        grouped = x.reshape(batch, ph, height // ph, pw, width // pw, channels)
        grouped = grouped.permute(0, 2, 4, 1, 3, 5)
    else:
        grouped = x.reshape(batch, height // ph, ph, width // pw, pw, channels)
        grouped = grouped.permute(0, 1, 3, 2, 4, 5)
    return grouped.reshape(-1, ph * pw, channels)


def restore_tokens(
    tokens: Tensor, size: tuple[int, int], image_size: tuple[int, int], grid: bool,
) -> Tensor:
    """Invert partition_tokens without joining independent batch elements."""
    ph, pw = size
    height, width = image_size
    channels = tokens.shape[-1]
    grouped = tokens.reshape(-1, height // ph, width // pw, ph, pw, channels)
    order = (0, 3, 1, 4, 2, 5) if grid else (0, 1, 3, 2, 4, 5)
    return grouped.permute(order).reshape(-1, height, width, channels)


class PartitionAttention(nn.Module):
    """Pre-norm attention and MLP within one RVT spatial partition."""

    def __init__(
        self, dim: int, heads: int, partition_size: tuple[int, int], *,
        grid: bool, mlp_ratio: float = 4.0, layer_scale: float = 1e-5,
        attention_dropout: float = 0.0, attention_batch_chunk_size: int = 16384,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("RVT channel count must be divisible by attention heads")
        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")
        if not 0 < attention_batch_chunk_size <= 65535:
            raise ValueError("attention_batch_chunk_size must be between 1 and 65535")
        self.heads = heads
        self.head_dim = dim // heads
        self.partition_size = partition_size
        self.grid = grid
        self.attention_dropout = attention_dropout
        self.attention_batch_chunk_size = attention_batch_chunk_size
        self.norm_attn = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.scale_attn = nn.Parameter(torch.full((dim,), layer_scale))
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.scale_mlp = nn.Parameter(torch.full((dim,), layer_scale))

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[1:3]
        tokens = partition_tokens(self.norm_attn(x), self.partition_size, self.grid)
        groups, length, dim = tokens.shape
        qkv = self.qkv(tokens).reshape(groups, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        dropout = self.attention_dropout if self.training else 0.0
        # Grouping can create more than 65,535 independent windows. Bound only
        # that independent axis; each window still attends to every own token.
        outputs = []
        for first in range(0, groups, self.attention_batch_chunk_size):
            last = first + self.attention_batch_chunk_size
            outputs.append(F.scaled_dot_product_attention(
                q[first:last], k[first:last], v[first:last], dropout_p=dropout,
            ))
        attended = torch.cat(outputs, dim=0).transpose(1, 2).reshape(groups, length, dim)
        attended = self.proj(attended)
        attended = restore_tokens(attended, self.partition_size, (height, width), self.grid)
        x = x + attended * self.scale_attn.to(attended.dtype)
        update = self.mlp(self.norm_mlp(x))
        return x + update * self.scale_mlp.to(update.dtype)


class PixelLSTM(nn.Module):
    """Shared 1x1 LSTM gates, with a separate hidden/cell pair per pixel."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gates = nn.Conv2d(2 * channels, 4 * channels, kernel_size=1)

    def forward(self, x: Tensor, state: LSTMState | None = None) -> LSTMState:
        if state is None:
            hidden = torch.zeros_like(x)
            cell = torch.zeros_like(x, dtype=torch.float32)
        else:
            if not isinstance(state, (tuple, list)) or len(state) != 2:
                raise ValueError("Each RVT stage state must be a (hidden, cell) pair")
            hidden, cell = state
            if hidden.shape != x.shape or cell.shape != x.shape:
                raise ValueError("RVT state shape must match this clip's batch and stage grid")
            if hidden.device != x.device or cell.device != x.device:
                raise ValueError("RVT state and input must be on the same device")
            hidden = hidden.to(x.dtype)
        forget, write, read, candidate = self.gates(torch.cat((x, hidden), dim=1)).float().chunk(4, dim=1)
        # Keep the additive memory in FP32 under bf16 autocast.
        cell = torch.sigmoid(forget) * cell.float() + torch.sigmoid(write) * torch.tanh(candidate)
        hidden = (torch.sigmoid(read) * torch.tanh(cell)).to(x.dtype)
        return hidden, cell


class RVTStage(nn.Module):
    def __init__(
        self, in_channels: int, channels: int, stride: int, heads: int,
        partition_size: tuple[int, int], num_blocks: int, **attention_kwargs,
    ) -> None:
        super().__init__()
        kernel_size = 7 if stride == 4 else 3
        self.downsample = nn.Conv2d(
            in_channels, channels, kernel_size=kernel_size, stride=stride,
            padding=kernel_size // 2, bias=False,
        )
        self.norm = nn.LayerNorm(channels)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                PartitionAttention(channels, heads, partition_size, grid=False, **attention_kwargs),
                PartitionAttention(channels, heads, partition_size, grid=True, **attention_kwargs),
            ) for _ in range(num_blocks)
        ])
        # The downsample already normalizes the first block's attention input.
        self.blocks[0][0].norm_attn = nn.Identity()
        self.lstm = PixelLSTM(channels)

    def forward(self, x: Tensor, state: LSTMState | None) -> tuple[Tensor, LSTMState]:
        x = self.norm(self.downsample(x).permute(0, 2, 3, 1))
        for block in self.blocks:
            x = block(x)
        state = self.lstm(x.permute(0, 3, 1, 2), state)
        return state[0], state


class RVTEventEncoder(nn.Module):
    """Encode [B,T,2*bins,H,W] clips as [B,T,N,feature_dim] features.

    Callers pass and receive recurrent state explicitly when streaming. With the
    default state=None every batch element starts from zero, independently of
    earlier calls. Returned state retains its graph for clip-wise backpropagation.
    """

    def __init__(
        self, in_channels: int = 20, feature_dim: int = 768,
        input_size: tuple[int, int] = (512, 1024),
        channels: Sequence[int] = (48, 96, 192, 384), output_stride: int = 16,
        fpn_channels: int = 256, heads: Sequence[int] | None = None,
        partition_size: tuple[int, int] = (8, 16),
        num_blocks: Sequence[int] = (1, 1, 1, 1), mlp_ratio: float = 4.0,
        layer_scale: float = 1e-5, attention_dropout: float = 0.0,
        attention_batch_chunk_size: int = 16384,
    ) -> None:
        super().__init__()
        input_size = tuple(int(x) for x in input_size)
        channels = tuple(int(x) for x in channels)
        num_blocks = tuple(int(x) for x in num_blocks)
        partition_size = tuple(int(x) for x in partition_size)
        if len(input_size) != 2 or min(input_size) < 32 or any(x % 32 for x in input_size):
            raise ValueError("RVT input_size must contain two positive multiples of 32")
        if len(channels) != 4 or min(channels) < 1 or len(num_blocks) != 4 or min(num_blocks) < 1:
            raise ValueError("RVT needs four positive channel counts and block depths")
        if len(partition_size) != 2 or min(partition_size) < 1:
            raise ValueError("partition_size must contain two positive values")
        if output_stride not in (16, 32):
            raise ValueError("RVT supports output_stride=16 (FPN) or 32 (compact S4)")
        if min(in_channels, feature_dim, fpn_channels) < 1 or mlp_ratio <= 0 or layer_scale <= 0:
            raise ValueError("RVT dimensions, MLP ratio, and layer scale must be positive")
        # RVT-S uses 24 channels per head. Tiny test widths use a divisor of 24.
        if heads is None:
            heads = tuple(c // math.gcd(c, 24) for c in channels)
        heads = tuple(int(x) for x in heads)
        if len(heads) != 4 or min(heads) < 1 or any(c % h for c, h in zip(channels, heads)):
            raise ValueError("heads must contain four positive divisors of stage channels")
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.input_size = input_size
        self.channels = channels
        self.output_stride = output_stride
        self.output_grid = tuple(x // output_stride for x in input_size)
        self.stages = nn.ModuleList()
        previous_channels = in_channels
        for stage, (width, head_count, depth) in enumerate(zip(channels, heads, num_blocks)):
            factor = 4 * (2 ** stage)
            stage_grid = tuple(x // factor for x in input_size)
            size = tuple(min(p, s) for p, s in zip(partition_size, stage_grid))
            if any(s % p for s, p in zip(stage_grid, size)):
                raise ValueError(f"Stage {stage + 1} grid {stage_grid} is not divisible by partition {size}")
            self.stages.append(RVTStage(
                previous_channels, width, 4 if stage == 0 else 2,
                head_count, size, depth, mlp_ratio=mlp_ratio, layer_scale=layer_scale,
                attention_dropout=attention_dropout,
                attention_batch_chunk_size=attention_batch_chunk_size,
            ))
            previous_channels = width
        if output_stride == 16:
            self.s2_down = nn.Conv2d(channels[1], fpn_channels, 3, stride=2, padding=1)
            self.s3_lateral = nn.Conv2d(channels[2], fpn_channels, 1)
            self.s4_lateral = nn.Conv2d(channels[3], fpn_channels, 1)
            self.fpn_refine = nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                nn.GroupNorm(math.gcd(32, fpn_channels), fpn_channels), nn.GELU(),
            )
            projection_dim = fpn_channels
        else:
            projection_dim = channels[3]
        self.projection = nn.Sequential(nn.Linear(projection_dim, feature_dim), nn.LayerNorm(feature_dim))
        self.architecture = {
            "in_channels": in_channels, "feature_dim": feature_dim, "input_size": input_size,
            "channels": channels, "output_stride": output_stride, "fpn_channels": fpn_channels,
            "heads": heads, "partition_size": partition_size, "num_blocks": num_blocks,
            "mlp_ratio": mlp_ratio, "layer_scale": layer_scale,
            "attention_dropout": attention_dropout,
            "attention_batch_chunk_size": attention_batch_chunk_size,
        }

    def _neck(self, stages: Sequence[Tensor]) -> Tensor:
        if self.output_stride == 32:
            features = stages[3]
        else:
            s2, s3, s4 = stages[1:]
            s4 = F.interpolate(s4, size=s3.shape[-2:], mode="bilinear", align_corners=False)
            features = self.fpn_refine(self.s2_down(s2) + self.s3_lateral(s3) + self.s4_lateral(s4))
        return self.projection(features.flatten(2).transpose(1, 2))

    def forward(
        self, events: Tensor, state: RVTState | None = None, *, return_state: bool = False,
    ) -> Tensor | tuple[Tensor, RVTState]:
        if events.ndim != 5:
            raise ValueError("RVT events must have shape [B,T,C,H,W]")
        batch, steps, channels, height, width = events.shape
        if not batch or not steps:
            raise ValueError("RVT needs nonempty batches and event histories")
        if channels != self.in_channels or (height, width) != self.input_size:
            raise ValueError(f"Expected [B,T,{self.in_channels},{self.input_size[0]},{self.input_size[1]}]")
        if not events.is_floating_point():
            raise ValueError("RVT expects floating point log-count event voxels")
        if state is not None and len(state) != len(self.stages):
            raise ValueError("RVT state must contain exactly four stage states")
        previous = state if state is not None else (None,) * len(self.stages)
        outputs = []
        for step in range(steps):
            x = events[:, step]
            stage_features, next_states = [], []
            for stage, old_state in zip(self.stages, previous):
                x, new_state = stage(x, old_state)
                stage_features.append(x)
                next_states.append(new_state)
            previous = tuple(next_states)
            outputs.append(self._neck(stage_features))
        features = torch.stack(outputs, dim=1)
        return (features, previous) if return_state else features
