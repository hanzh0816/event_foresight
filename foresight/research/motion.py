"""Stages 3–5: predict motion from history, then roll out future scene states."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import torch
from torch import Tensor, nn

from .common import HistoryFeatures, check_scene
from .transition import TransitionWorldModel


class MotionWorldModel(nn.Module):
    def __init__(self, teacher: TransitionWorldModel, history_frames=4,
                 mode="explicit", motion_input="event", copy_last_motion=False,
                 train_transition=False):
        super().__init__()
        if mode not in {"explicit", "auxiliary", "direct"}:
            raise ValueError("motion mode must be explicit, auxiliary, or direct")
        if motion_input not in {"event", "rgb"}:
            raise ValueError("motion_input must be event or rgb")
        if copy_last_motion and (motion_input != "event" or mode != "explicit"):
            raise ValueError("copy_last_motion is an explicit Event-history control")
        if train_transition and mode != "explicit":
            raise ValueError("train_transition only applies to explicit motion rollout")
        if teacher.transition_mode != "motion":
            raise ValueError("motion prediction requires a teacher trained with motion conditioning")
        self.teacher = deepcopy(teacher).requires_grad_(False).eval()
        self.history_frames = history_frames
        self.mode = mode
        self.motion_input = motion_input
        self.copy_last_motion = copy_last_motion
        self.train_transition = train_transition
        self.grid_size = teacher.grid_size
        self.scene_dim = teacher.scene_dim
        self.motion_dim = teacher.motion_dim
        self.hidden_dim = teacher.hidden_dim
        self.history = HistoryFeatures(self.scene_dim, self.motion_dim,
                                       self.hidden_dim, history_frames)
        self.motion_head = nn.Linear(self.hidden_dim, self.motion_dim)
        self.scene_condition_head = nn.Linear(self.hidden_dim, self.motion_dim)
        for head in (self.motion_head, self.scene_condition_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        if motion_input == "rgb":
            self.rgb_motion = nn.Sequential(nn.LayerNorm(self.scene_dim),
                                            nn.Linear(self.scene_dim, self.motion_dim))
        if train_transition:
            self.adapted_transition = deepcopy(teacher.scene_transition).requires_grad_(True)
        if copy_last_motion:
            self.history.requires_grad_(False)
            self.motion_head.requires_grad_(False)
            self.scene_condition_head.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def model_spec(self):
        return {"kind": "motion", "teacher": self.teacher.model_spec(), "kwargs": {
            "history_frames": self.history_frames, "mode": self.mode,
            "motion_input": self.motion_input, "copy_last_motion": self.copy_last_motion,
            "train_transition": self.train_transition}}

    @torch.no_grad()
    def encode_target_motion(self, events: Tensor) -> Tensor:
        return self.teacher.encode_motion(events)

    def _transition(self, scene: Tensor, motion: Tensor) -> Tensor:
        if self.train_transition:
            if self.teacher.transition_mode == "no_motion":
                motion = torch.zeros_like(motion)
            return self.adapted_transition(scene, motion)
        # Frozen parameters still transmit input gradients to the motion head.
        return self.teacher.transition(scene, motion)

    def forward(self, z_history: Tensor, events_history: Tensor | None,
                steps: int = 1,
                base_predictor: Callable[[Tensor], Tensor | dict[str, Tensor]] | None = None
                ) -> dict[str, Tensor]:
        # Retained for caller compatibility. All modes intentionally use the
        # same learned transition and starting scene, with no RGB predictor
        # bypass that would confound the intermediate-representation test.
        del base_predictor
        check_scene(z_history, self.grid_size, self.scene_dim, history_frames=self.history_frames)
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if self.motion_input == "event":
            if events_history is None:
                raise ValueError("Event motion input requires historical events")
            motion_history = self.encode_target_motion(events_history)
        else:
            motion_history = self.rgb_motion(z_history[:, 1:] - z_history[:, :-1])
        if motion_history.shape[:3] != (z_history.shape[0], self.history_frames - 1, z_history.shape[2]):
            raise ValueError("events must cover exactly the consecutive history intervals")
        scenes, motions = [], []
        z = z_history
        m = motion_history
        for _ in range(steps):
            if self.copy_last_motion:
                predicted_motion = m[:, -1]
            else:
                features = self.history(z, m)
                predicted_motion = m[:, -1] + self.motion_head(features)
            if self.mode == "explicit":
                scene_condition = predicted_motion
            else:
                scene_condition = m[:, -1] + self.scene_condition_head(features)
            predicted_scene = self._transition(z[:, -1], scene_condition)
            scenes.append(predicted_scene)
            motions.append(predicted_motion)
            z = torch.cat((z[:, 1:], predicted_scene.unsqueeze(1)), dim=1)
            # Direct/auxiliary roll out their freely learned transition code;
            # their supervised motion head never enters the scene path, even
            # indirectly through the condition history of later steps.
            m = torch.cat((m[:, 1:], scene_condition.unsqueeze(1)), dim=1)
        return {"scene": torch.stack(scenes, dim=1), "motion": torch.stack(motions, dim=1)}
