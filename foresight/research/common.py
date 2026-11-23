"""Small spatial/temporal components independent of DINO and its checkpoints."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def normalize_grid(grid_size: Sequence[int]) -> tuple[int, int]:
    if len(grid_size) != 2 or any(int(x) <= 0 for x in grid_size):
        raise ValueError("grid_size must contain two positive integers")
    return int(grid_size[0]), int(grid_size[1])


def attention_heads(width: int) -> int:
    return next(h for h in (4, 2, 1) if width % h == 0)


def groups(width: int) -> int:
    return next(g for g in (8, 4, 2, 1) if width % g == 0)


def check_scene(scene: Tensor, grid_size: tuple[int, int], scene_dim: int,
                *, history_frames: int | None = None) -> None:
    expected_ndim = 4 if history_frames is not None else 3
    if scene.ndim != expected_ndim:
        raise ValueError(f"scene must have {expected_ndim} dimensions, got {tuple(scene.shape)}")
    if scene.shape[-2:] != (grid_size[0] * grid_size[1], scene_dim):
        raise ValueError("scene token count/dimension does not match model grid_size/scene_dim")
    if history_frames is not None and scene.shape[1] != history_frames:
        raise ValueError(f"expected {history_frames} history frames, got {scene.shape[1]}")


def prediction_tensor(prediction: Tensor | dict[str, Tensor]) -> Tensor:
    """Normalize the existing RGB predictor and research model output APIs."""
    if isinstance(prediction, dict):
        prediction = prediction["scene"]
    if prediction.ndim == 4:
        if prediction.shape[1] != 1:
            raise ValueError("base predictor must return exactly one future frame")
        prediction = prediction[:, 0]
    if prediction.ndim != 3:
        raise ValueError("base prediction must be [B,N,D] or [B,1,N,D]")
    return prediction


def parameter_counts(module: nn.Module) -> dict[str, int]:
    return {
        "total": sum(p.numel() for p in module.parameters()),
        "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad),
    }


class TemporalMixer(nn.Module):
    """Attention over a short explicit time axis independently at each site."""

    # Event encoding folds B * intervals * spatial_sites into the attention
    # batch. Some CUDA SDPA kernels fail at 65,536 batch entries (grid limit):
    # https://github.com/pytorch/pytorch/issues/146704
    # Bound every call in the shared mixer, including HistoryFeatures. This is
    # an execution limit, not a model parameter or a checkpoint/config change.
    MAX_BATCH_SITES = 16_384

    def __init__(self, width: int, length: int):
        super().__init__()
        self.length = length
        self.position = nn.Parameter(torch.empty(1, length, width))
        nn.init.normal_(self.position, std=0.02)
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, attention_heads(width), batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2),
                                 nn.GELU(), nn.Linear(width * 2, width))

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[1] != self.length:
            raise ValueError("temporal mixer requires [batch_sites, configured_length, width]")
        if tokens.shape[0] <= self.MAX_BATCH_SITES:
            return self._forward_chunk(tokens)
        # Sites are independent; retain the entire time axis in every chunk.
        # Autograd sums parameter gradients across chunks as in the full batch.
        return torch.cat([self._forward_chunk(chunk)
                          for chunk in tokens.split(self.MAX_BATCH_SITES, dim=0)], dim=0)

    def _forward_chunk(self, tokens: Tensor) -> Tensor:
        x = tokens + self.position
        q = self.norm(x)
        x = x + self.attention(q, q, q, need_weights=False)[0]
        return x + self.ffn(x)


class EventMotionEncoder(nn.Module):
    """Shared spatial stem followed by explicit subwindow temporal mixing.

    Ordered and pooled variants have identical parameters and tensor compute
    shapes. Pooling removes order before positional encoding and attention;
    consequently the pooled variant is invariant to subwindow permutations.
    No RGB feature alignment objective is imposed by this encoder.
    """

    def __init__(self, grid_size=(32, 64), motion_dim=128, hidden_dim=128,
                 bins=5, temporal_mode="ordered"):
        super().__init__()
        self.grid_size = normalize_grid(grid_size)
        self.motion_dim = int(motion_dim)
        self.hidden_dim = int(hidden_dim)
        self.bins = int(bins)
        if min(self.motion_dim, self.hidden_dim, self.bins) <= 0:
            raise ValueError("encoder dimensions and bins must be positive")
        if temporal_mode not in {"ordered", "pooled"}:
            raise ValueError("temporal_mode must be ordered or pooled")
        self.temporal_mode = temporal_mode
        stem_width = max(4, hidden_dim // 4)
        self.stem = nn.Sequential(
            nn.Conv2d(2, stem_width, 5, stride=2, padding=2),
            nn.GroupNorm(groups(stem_width), stem_width), nn.GELU(),
            nn.Conv2d(stem_width, hidden_dim, 3, stride=2, padding=1),
            nn.GroupNorm(groups(hidden_dim), hidden_dim), nn.GELU(),
            nn.AdaptiveAvgPool2d(self.grid_size),
        )
        self.temporal = TemporalMixer(hidden_dim, bins)
        self.spatial = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.GELU(), nn.Conv2d(hidden_dim, motion_dim, 1),
        )

    def forward(self, events: Tensor) -> Tensor:
        if events.ndim != 6:
            raise ValueError("events must be [B,intervals,bins,2,H,W]")
        b, intervals, bins, polarity, h, w = events.shape
        if intervals < 1 or bins != self.bins or polarity != 2:
            raise ValueError("events require positive intervals, configured bins, and two polarities")
        if not events.is_floating_point():
            raise ValueError("events must be floating point voxel tensors")
        grid_h, grid_w = self.grid_size
        x = self.stem(events.reshape(b * intervals * bins, 2, h, w))
        x = x.reshape(b * intervals, bins, self.hidden_dim, grid_h * grid_w)
        x = x.permute(0, 3, 1, 2).reshape(-1, bins, self.hidden_dim)
        if self.temporal_mode == "pooled":
            x = x.mean(dim=1, keepdim=True).expand(-1, bins, -1)
        x = self.temporal(x).mean(dim=1)
        x = x.reshape(b * intervals, grid_h, grid_w, self.hidden_dim).permute(0, 3, 1, 2)
        x = self.spatial(x).flatten(2).transpose(1, 2)
        return x.reshape(b, intervals, grid_h * grid_w, self.motion_dim)


class SceneResidual(nn.Module):
    """Spatial scene residual conditioned on a separately parameterized signal."""

    def __init__(self, grid_size, scene_dim, condition_dim, hidden_dim, *, zero_init=True):
        super().__init__()
        self.grid_size = normalize_grid(grid_size)
        self.scene_dim = scene_dim
        self.condition_dim = condition_dim
        self.scene = nn.Sequential(nn.LayerNorm(scene_dim), nn.Linear(scene_dim, hidden_dim))
        self.condition = nn.Linear(condition_dim, hidden_dim)
        self.spatial = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                                     nn.GELU(), nn.Conv2d(hidden_dim, hidden_dim, 1), nn.GELU())
        self.output = nn.Linear(hidden_dim, scene_dim)
        if zero_init:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, scene: Tensor, condition: Tensor) -> Tensor:
        check_scene(scene, self.grid_size, self.scene_dim)
        if condition.shape != (*scene.shape[:-1], self.condition_dim):
            raise ValueError("condition must have the same batch and spatial tokens as scene")
        h, w = self.grid_size
        x = self.scene(scene) + self.condition(condition)
        x = x.transpose(1, 2).reshape(scene.shape[0], -1, h, w)
        x = self.spatial(x).flatten(2).transpose(1, 2)
        return scene + self.output(x)


class HistoryFeatures(nn.Module):
    """Fuse scene history and interval motion without any future observations."""

    def __init__(self, scene_dim, motion_dim, hidden_dim, history_frames):
        super().__init__()
        if history_frames < 2:
            raise ValueError("at least two history frames are required")
        self.history_frames = history_frames
        self.hidden_dim = hidden_dim
        self.scene = nn.Sequential(nn.LayerNorm(scene_dim), nn.Linear(scene_dim, hidden_dim))
        self.motion = nn.Linear(motion_dim, hidden_dim)
        self.temporal = TemporalMixer(hidden_dim, history_frames)

    def forward(self, scene_history: Tensor, motion_history: Tensor) -> Tensor:
        b, c, n, _ = scene_history.shape
        if c != self.history_frames or motion_history.shape[:3] != (b, c - 1, n):
            raise ValueError("history motion intervals must connect consecutive history frames")
        z = self.scene(scene_history)
        m = self.motion(motion_history)
        m = torch.cat((torch.zeros_like(m[:, :1]), m), dim=1)
        x = (z + m).permute(0, 2, 1, 3).reshape(b * n, c, self.hidden_dim)
        return self.temporal(x)[:, -1].reshape(b, n, self.hidden_dim)
