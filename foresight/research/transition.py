"""Stage 2: observed-interval transition representation, not forecasting."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .common import EventMotionEncoder, SceneResidual, check_scene, normalize_grid


class TransitionWorldModel(nn.Module):
    def __init__(self, grid_size=(32, 64), scene_dim=1152, motion_dim=128,
                 hidden_dim=128, bins=5, temporal_mode="ordered", transition_mode="motion"):
        super().__init__()
        if transition_mode not in {"motion", "no_motion"}:
            raise ValueError("transition_mode must be motion or no_motion")
        self.grid_size = normalize_grid(grid_size)
        self.scene_dim = scene_dim
        self.motion_dim = motion_dim
        self.hidden_dim = hidden_dim
        self.bins = bins
        self.temporal_mode = temporal_mode
        self.transition_mode = transition_mode
        self.event_encoder = EventMotionEncoder(self.grid_size, motion_dim, hidden_dim,
                                                bins, temporal_mode)
        if transition_mode == "no_motion":
            self.event_encoder.requires_grad_(False)
        self.scene_transition = SceneResidual(self.grid_size, scene_dim, motion_dim, hidden_dim)

    def model_spec(self):
        return {"kind": "transition", "kwargs": {
            "grid_size": list(self.grid_size), "scene_dim": self.scene_dim,
            "motion_dim": self.motion_dim, "hidden_dim": self.hidden_dim, "bins": self.bins,
            "temporal_mode": self.temporal_mode, "transition_mode": self.transition_mode}}

    def encode_motion(self, events: Tensor) -> Tensor:
        return self.event_encoder(events)

    def transition(self, scene: Tensor, motion: Tensor) -> Tensor:
        if self.transition_mode == "no_motion":
            motion = torch.zeros_like(motion)
        return self.scene_transition(scene, motion)

    def forward(self, start_scene: Tensor, interval_events: Tensor) -> dict[str, Tensor]:
        check_scene(start_scene, self.grid_size, self.scene_dim)
        motion = self.encode_motion(interval_events)
        if motion.shape[0] != start_scene.shape[0]:
            raise ValueError("events and starting scene have different batch sizes")
        scenes = []
        z = start_scene
        for m in motion.unbind(dim=1):
            z = self.transition(z, m)
            scenes.append(z)
        return {"scene": torch.stack(scenes, dim=1), "motion": motion}
