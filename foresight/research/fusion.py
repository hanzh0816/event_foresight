"""Stage 1: causal historical Event information probes."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .common import (EventMotionEncoder, HistoryFeatures, SceneResidual,
                     check_scene, normalize_grid, prediction_tensor)


class FusionWorldModel(nn.Module):
    MODES = {"rgb", "rgb_control", "event_pooled", "event_ordered"}

    def __init__(self, grid_size=(32, 64), scene_dim=1152, motion_dim=128,
                 hidden_dim=128, bins=5, history_frames=4, mode="event_ordered"):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown fusion mode: {mode}")
        self.grid_size = normalize_grid(grid_size)
        self.scene_dim = scene_dim
        self.motion_dim = motion_dim
        self.hidden_dim = hidden_dim
        self.bins = bins
        self.history_frames = history_frames
        self.mode = mode
        if mode == "rgb":
            return
        if mode.startswith("event_"):
            self.event_encoder = EventMotionEncoder(
                self.grid_size, motion_dim, hidden_dim, bins, mode.removeprefix("event_"))
        else:
            self.rgb_motion = nn.Sequential(nn.LayerNorm(scene_dim),
                                            nn.Linear(scene_dim, hidden_dim), nn.GELU(),
                                            nn.Linear(hidden_dim, motion_dim))
        self.history = HistoryFeatures(scene_dim, motion_dim, hidden_dim, history_frames)
        self.adapter = SceneResidual(self.grid_size, scene_dim, hidden_dim, hidden_dim)

    def model_spec(self):
        return {"kind": "fusion", "kwargs": {
            "grid_size": list(self.grid_size), "scene_dim": self.scene_dim,
            "motion_dim": self.motion_dim, "hidden_dim": self.hidden_dim,
            "bins": self.bins, "history_frames": self.history_frames, "mode": self.mode}}

    def forward(self, z_history: Tensor, events_history: Tensor | None,
                base_prediction: Tensor | dict[str, Tensor]) -> dict[str, Tensor]:
        check_scene(z_history, self.grid_size, self.scene_dim, history_frames=self.history_frames)
        base = prediction_tensor(base_prediction)
        check_scene(base, self.grid_size, self.scene_dim)
        if base.shape[0] != z_history.shape[0]:
            raise ValueError("base prediction batch does not match scene history")
        if self.mode == "rgb":
            return {"scene": base.unsqueeze(1)}
        if self.mode.startswith("event_"):
            if events_history is None:
                raise ValueError("event mode requires historical events")
            motion = self.event_encoder(events_history)
        else:
            motion = self.rgb_motion(z_history[:, 1:] - z_history[:, :-1])
        features = self.history(z_history, motion)
        return {"scene": self.adapter(base, features).unsqueeze(1)}
