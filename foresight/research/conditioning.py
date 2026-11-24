"""Historical Event conditions inside the complete pretrained DINO predictor."""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor, nn

from .common import check_scene, groups, normalize_grid, prediction_tensor


def _positive_int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class ConditioningWorldModel(nn.Module):
    """Read ordered Event subwindows with the predicted future hidden query.

    Every interval/bin retains a separate spatial feature map. With four RGB
    history frames and five bins, each spatial site attends to 3 * 5 Event
    tokens. ``event_pooled`` averages bin features *within each interval*, then
    repeats them before adding the same interval/bin positions. It removes
    within-interval order while preserving the three-interval history and the
    ordered variant's parameter count and attention shapes.

    The pretrained predictor is passed to forward, never registered twice.
    Only the last condition projection is zero initialized. Thus initialization
    preserves the RGB predictor while allowing that projection to learn first
    and subsequently transmit scene-loss gradients into the Event branch.
    """

    MODES = {"event_ordered", "event_pooled", "rgb"}
    MAX_BATCH_SITES = 16_384

    def __init__(self, grid_size, scene_dim, predictor_dim, motion_dim=128,
                 hidden_dim=128, bins=5, history_frames=4, injection_layer=6,
                 attention_heads=4, mode="event_ordered"):
        super().__init__()
        self.grid_size = normalize_grid(grid_size)
        self.scene_dim = _positive_int(scene_dim, "scene_dim")
        self.predictor_dim = _positive_int(predictor_dim, "predictor_dim")
        self.motion_dim = _positive_int(motion_dim, "motion_dim")
        self.hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        self.bins = _positive_int(bins, "bins")
        self.history_frames = _positive_int(history_frames, "history_frames", 2)
        self.injection_layer = _positive_int(injection_layer, "injection_layer")
        self.attention_heads = _positive_int(attention_heads, "attention_heads")
        if hidden_dim % attention_heads:
            raise ValueError("attention_heads must divide hidden_dim")
        if mode not in self.MODES:
            raise ValueError("Unknown conditioning mode")
        self.mode = mode
        if mode == "rgb":
            return

        stem_width = max(4, hidden_dim // 4)
        self.event_stem = nn.Sequential(
            nn.Conv2d(2, stem_width, 5, stride=2, padding=2),
            nn.GroupNorm(groups(stem_width), stem_width), nn.GELU(),
            nn.Conv2d(stem_width, hidden_dim, 3, stride=2, padding=1),
            nn.GroupNorm(groups(hidden_dim), hidden_dim), nn.GELU(),
            nn.AdaptiveAvgPool2d(self.grid_size),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.GELU(), nn.Conv2d(hidden_dim, hidden_dim, 1),
        )
        self.interval_position = nn.Parameter(torch.empty(1, history_frames - 1, 1, hidden_dim))
        self.bin_position = nn.Parameter(torch.empty(1, 1, bins, hidden_dim))
        nn.init.normal_(self.interval_position, std=0.02)
        nn.init.normal_(self.bin_position, std=0.02)
        self.query = nn.Sequential(nn.LayerNorm(predictor_dim), nn.Linear(predictor_dim, hidden_dim))
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, attention_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 2 * hidden_dim),
                                 nn.GELU(), nn.Linear(2 * hidden_dim, hidden_dim))
        self.motion_head = nn.Linear(hidden_dim, motion_dim)
        self.output = nn.Linear(motion_dim, predictor_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def model_spec(self):
        return {"kind": "conditioning", "kwargs": {
            "grid_size": list(self.grid_size), "scene_dim": self.scene_dim,
            "predictor_dim": self.predictor_dim, "motion_dim": self.motion_dim,
            "hidden_dim": self.hidden_dim, "bins": self.bins,
            "history_frames": self.history_frames, "injection_layer": self.injection_layer,
            "attention_heads": self.attention_heads, "mode": self.mode}}

    def encode_events(self, events: Tensor) -> Tensor:
        """[B, intervals, bins, 2, H, W] -> [B, spatial, intervals*bins, hidden]."""
        return self._encode_events(events)

    def _encode_events(self, events: Tensor, diagnostics: dict[str, Tensor] | None = None) -> Tensor:
        if self.mode == "rgb":
            raise ValueError("RGB mode has no Event encoder")
        if not isinstance(events, Tensor) or events.ndim != 6:
            raise ValueError("events must be [B,history_frames-1,bins,2,H,W]")
        b, intervals, bins, polarity, height, width = events.shape
        if (b < 1 or intervals != self.history_frames - 1 or bins != self.bins or
                polarity != 2 or min(height, width) < 1):
            raise ValueError("events must contain only the configured historical intervals and bins")
        if not events.is_floating_point():
            raise ValueError("events must be floating point voxel tensors")
        spatial = self.grid_size[0] * self.grid_size[1]
        encoded = self.event_stem(events.reshape(b * intervals * bins, 2, height, width))
        encoded = encoded.flatten(2).transpose(1, 2).reshape(b, intervals, bins, spatial, self.hidden_dim)
        if self.mode == "event_pooled":
            encoded = encoded.mean(dim=2, keepdim=True).expand(-1, -1, bins, -1, -1)
        if diagnostics is not None:
            diagnostics["encoded_events"] = encoded.permute(0, 3, 1, 2, 4).reshape(
                b, spatial, intervals * bins, self.hidden_dim)
        positions = self.interval_position + self.bin_position
        encoded = encoded + positions.unsqueeze(3)
        memory = encoded.permute(0, 3, 1, 2, 4).reshape(b, spatial, intervals * bins, self.hidden_dim)
        if diagnostics is not None:
            diagnostics["memory"] = memory
        return memory

    def _condition(self, future_hidden: Tensor, memory: Tensor, *, condition_mode="full",
                   diagnostics: dict[str, Tensor] | None = None) -> tuple[Tensor, Tensor]:
        b, spatial, length, _ = memory.shape
        if future_hidden.shape != (b, spatial, self.predictor_dim):
            raise ValueError("future hidden map differs from configured predictor_dim/grid")
        query = self.query(future_hidden).reshape(b * spatial, 1, self.hidden_dim)
        memory = self.memory_norm(memory).reshape(b * spatial, length, self.hidden_dim)
        parts = []
        attention_parts = [] if diagnostics is not None else None
        # Bound the spatial-site batch for CUDA SDPA, preserving the entire
        # historical time axis in each call and the gradients of all sites.
        for q, kv in zip(query.split(self.MAX_BATCH_SITES), memory.split(self.MAX_BATCH_SITES), strict=True):
            attention = (self.cross_attention(q, kv, kv, need_weights=False)[0]
                         if condition_mode == "full" else torch.zeros_like(q))
            value = q + attention
            parts.append(value + self.ffn(value))
            if attention_parts is not None:
                attention_parts.append(attention)
        features = torch.cat(parts).reshape(b, spatial, self.hidden_dim)
        motion = self.motion_head(features)
        if diagnostics is not None:
            diagnostics["query"] = query.reshape(b, spatial, self.hidden_dim)
            diagnostics["attention"] = torch.cat(attention_parts).reshape(b, spatial, self.hidden_dim)
            diagnostics["hidden"] = future_hidden
            diagnostics["motion"] = motion
        return self.output(motion), motion

    def forward(self, z_history: Tensor, events_history: Tensor | None, base_predictor) -> dict[str, Tensor]:
        return self._forward_impl(z_history, events_history, base_predictor)

    def forward_diagnostic(self, z_history: Tensor, events_history: Tensor, base_predictor, *,
                           condition_mode="full", condition_scale=1.0) -> dict[str, Tensor | dict[str, Tensor]]:
        """Return a forecast and intermediate tensors without changing checkpoint keys.

        ``full`` preserves ordinary forward arithmetic at scale 1. ``query_only``
        removes the cross-attention contribution while retaining the RGB query
        residual, FFN and projections. Scale applies to the final hidden update,
        including its bias. Use the predictor directly for a complete bypass.

        Diagnostic tensors retain their graph; callers select no-grad, detach,
        or gradient probes. No activations are stored on the module. The event
        tensors are [B,N,intervals*bins,hidden], query/attention are [B,N,hidden],
        hidden/update are [B,N,predictor_dim], and motion is [B,N,motion_dim].
        """
        if self.mode == "rgb":
            raise ValueError("forward_diagnostic requires an Event conditioning model")
        if condition_mode not in {"full", "query_only"}:
            raise ValueError("condition_mode must be 'full' or 'query_only'")
        if (isinstance(condition_scale, bool) or not isinstance(condition_scale, Real)
                or not math.isfinite(condition_scale)):
            raise ValueError("condition_scale must be a finite real number")
        diagnostics: dict[str, Tensor] = {}
        result = self._forward_impl(z_history, events_history, base_predictor,
                                    condition_mode=condition_mode, condition_scale=float(condition_scale),
                                    diagnostics=diagnostics)
        return {**result, "diagnostics": diagnostics}

    def _forward_impl(self, z_history: Tensor, events_history: Tensor | None, base_predictor, *,
                      condition_mode="full", condition_scale=1.0,
                      diagnostics: dict[str, Tensor] | None = None) -> dict[str, Tensor]:
        check_scene(z_history, self.grid_size, self.scene_dim, history_frames=self.history_frames)
        if not z_history.is_floating_point():
            raise ValueError("scene history must be floating point PCA tokens")
        if self.mode == "rgb":
            scene = prediction_tensor(base_predictor(z_history))
            check_scene(scene, self.grid_size, self.scene_dim)
            return {"scene": scene.unsqueeze(1)}
        if not callable(getattr(base_predictor, "forward_conditioned", None)):
            raise TypeError("Event conditioning requires the pretrained predictor's forward_conditioned API")
        if not isinstance(events_history, Tensor) or events_history.ndim != 6:
            raise ValueError("Event conditioning requires historical events [B,intervals,bins,2,H,W]")
        if events_history.shape[0] != z_history.shape[0] or events_history.device != z_history.device:
            raise ValueError("RGB and Event histories must have the same batch and device")
        memory = self._encode_events(events_history, diagnostics)
        predicted_motion = []

        def conditioner(future_hidden):
            update, motion = self._condition(future_hidden, memory, condition_mode=condition_mode,
                                              diagnostics=diagnostics)
            if condition_scale != 1.0:
                update = update * condition_scale
            if diagnostics is not None:
                diagnostics["update"] = update
            predicted_motion.append(motion)
            return update

        scene = base_predictor.forward_conditioned(z_history, conditioner, self.injection_layer)
        check_scene(scene, self.grid_size, self.scene_dim)
        if len(predicted_motion) != 1:
            raise RuntimeError("Predictor must call its condition exactly once per forecast")
        return {"scene": scene.unsqueeze(1), "motion": predicted_motion[0].unsqueeze(1)}
