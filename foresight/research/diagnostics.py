"""Streaming diagnostics for observed-interval transition teachers.

Zeroing a learned teacher's motion input is a sensitivity probe. It is not a
replacement for separately training the no-motion transition control.
"""

from __future__ import annotations

import torch
from torch import Tensor


class TransitionDiagnostics:
    """Accumulate true-start, one-step transitions without retaining batches.

    Motion moments aggregate over batch, interval, and spatial token axes for
    each channel. A second set uses spatially averaged interval embeddings,
    measuring variation between intervals rather than fixed spatial patterns.
    Only scalar totals and CPU vectors of length motion_dim are retained.
    """

    def __init__(self):
        self.sample_count = 0
        self.interval_count = 0
        self.motion_token_count = 0
        self.scene_element_count = 0
        self.motion_sum = None
        self.motion_square_sum = None
        self.interval_mean_sum = None
        self.interval_mean_square_sum = None
        self.error_sums = {key: 0.0 for key in (
            "teacher_one_step_l1", "persistence_l1", "zero_motion_l1",
            "real_vs_zero_prediction_l1")}

    @torch.no_grad()
    def update(self, model, starts: Tensor, interval_events: Tensor, targets: Tensor):
        if starts.ndim != 4 or targets.shape != starts.shape:
            raise ValueError("starts and targets must be matching [B,S,N,Dz] tensors")
        b, steps, tokens, _ = starts.shape
        if min(starts.shape) < 1:
            raise ValueError("diagnostic batches must have nonempty axes")
        if interval_events.ndim != 6 or interval_events.shape[:2] != (b, steps):
            raise ValueError("interval_events must be [B,S,K,2,H,W] with matching B,S")
        was_training = model.training
        model.eval()
        try:
            motion = model.encode_motion(interval_events)
            if motion.ndim != 4 or motion.shape[:3] != (b, steps, tokens):
                raise ValueError("motion encoding must have shape [B,S,N,Dm]")
            start = starts.flatten(0, 1)
            m = motion.flatten(0, 1)
            real = model.transition(start, m).reshape_as(starts)
            zero = model.transition(start, torch.zeros_like(m)).reshape_as(starts)
            if not all(torch.isfinite(x).all().item() for x in (motion, starts, targets, real, zero)):
                raise FloatingPointError("non-finite tensor in transition diagnostics")
            # Compute reductions in float64; only channel vectors survive update.
            values = motion.double()
            sums = values.sum(dim=(0, 1, 2)).cpu()
            squares = values.square().sum(dim=(0, 1, 2)).cpu()
            interval_means = values.mean(dim=2)
            mean_sums = interval_means.sum(dim=(0, 1)).cpu()
            mean_squares = interval_means.square().sum(dim=(0, 1)).cpu()
            if self.motion_sum is not None and sums.shape != self.motion_sum.shape:
                raise ValueError("motion dimension changed between diagnostic batches")
            errors = {
                "teacher_one_step_l1": (real.double() - targets.double()).abs().sum().item(),
                "persistence_l1": (starts.double() - targets.double()).abs().sum().item(),
                "zero_motion_l1": (zero.double() - targets.double()).abs().sum().item(),
                "real_vs_zero_prediction_l1": (real.double() - zero.double()).abs().sum().item(),
            }
            if self.motion_sum is None:
                self.motion_sum, self.motion_square_sum = sums, squares
                self.interval_mean_sum, self.interval_mean_square_sum = mean_sums, mean_squares
            else:
                self.motion_sum += sums
                self.motion_square_sum += squares
                self.interval_mean_sum += mean_sums
                self.interval_mean_square_sum += mean_squares
            for key, value in errors.items():
                self.error_sums[key] += value
            self.sample_count += b
            self.interval_count += b * steps
            self.motion_token_count += b * steps * tokens
            self.scene_element_count += targets.numel()
        finally:
            model.train(was_training)

    @staticmethod
    def _moments(total, square_total, count):
        if not count:
            return {"mean_per_channel": None, "variance_per_channel": None,
                    "std_per_channel": None, "mean_channel_std": None}
        mean = total / count
        variance = (square_total / count - mean.square()).clamp_min(0)
        std = variance.sqrt()
        return {"mean_per_channel": mean.tolist(), "variance_per_channel": variance.tolist(),
                "std_per_channel": std.tolist(), "mean_channel_std": std.mean().item()}

    def compute(self):
        return {
            "kind": "observed_transition_teacher_diagnostic",
            "scene_protocol": "one step from each true interval start; no recursive rollout",
            "zero_motion_protocol": "same trained transition with zero motion; sensitivity probe, not a trained no-motion control",
            "sample_count": self.sample_count,
            "interval_count": self.interval_count,
            "motion_token_count": self.motion_token_count,
            "scene_element_count": self.scene_element_count,
            "motion_token_statistics": {
                "aggregation_axes": "batch, interval, spatial_token; independently per motion channel",
                "variance_estimator": "population",
                **self._moments(self.motion_sum, self.motion_square_sum, self.motion_token_count),
            },
            "motion_interval_mean_statistics": {
                "aggregation_axes": "spatial mean first, then batch and interval; independently per motion channel",
                "variance_estimator": "population",
                **self._moments(self.interval_mean_sum, self.interval_mean_square_sum, self.interval_count),
            },
            **{key: value / self.scene_element_count if self.scene_element_count else None
               for key, value in self.error_sums.items()},
        }
