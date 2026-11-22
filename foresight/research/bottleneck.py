"""Paired motion/transition diagnostics with an unchanged trained checkpoint.

The future-observation routes are diagnostics, never history-only forecasts.
All transition routes use the *same active transition* in the loaded motion
model, including its adapted copy when ``train_transition`` is enabled.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .common import check_scene
from .motion import MotionWorldModel


def _batch_tensor(batch, key, dimensions):
    value = batch.get(key)
    if not isinstance(value, Tensor) or value.ndim != dimensions:
        raise ValueError(f"{key} must be a {dimensions}-dimensional tensor")
    return value


def _validate_inputs(task, batch, steps):
    model = task.model
    if task.stage != "motion" or not isinstance(model, MotionWorldModel):
        raise ValueError("Bottleneck diagnostics require a motion-stage checkpoint")
    if model.mode != "explicit" or model.motion_input != "event" or model.copy_last_motion:
        raise ValueError("Require explicit, Event-input, learned motion prediction (not direct/auxiliary/copy-last)")
    history_rgb = _batch_tensor(batch, "history_rgb", 5)
    history_events = _batch_tensor(batch, "history_events", 6)
    future_rgb = _batch_tensor(batch, "future_rgb_target", 5)
    future_events = _batch_tensor(batch, "future_events_target", 6)
    steps = future_rgb.shape[1] if steps is None else steps
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    batch_size = history_rgb.shape[0]
    if batch_size < 1 or history_rgb.shape[1:3] != (model.history_frames, 3):
        raise ValueError("history_rgb must be [B,history_frames,3,H,W] with B > 0")
    if future_rgb.shape[0] != batch_size or future_rgb.shape[2:] != history_rgb.shape[2:]:
        raise ValueError("future_rgb_target must share the history batch, channels and image size")
    if future_rgb.shape[1] < steps or future_events.shape[1] < steps:
        raise ValueError("Future RGB and Event targets must both contain at least steps intervals")
    expected = (batch_size, model.history_frames - 1, model.teacher.bins, 2)
    if history_events.shape[:4] != expected:
        raise ValueError("history_events must be [B,history_frames-1,bins,2,H,W]")
    if future_events.shape[0] != batch_size or future_events.shape[2:] != history_events.shape[2:]:
        raise ValueError("Future Event targets must share the history batch, bins, polarity and spatial size")
    if any(value.device != history_rgb.device for value in (history_events, future_rgb, future_events)):
        raise ValueError("All diagnostic observation tensors must use the same device")
    return model, history_rgb, history_events, future_rgb[:, :steps], future_events[:, :steps], steps


@torch.inference_mode()
def diagnostic_predictions(task, batch, steps=None):
    """Return paired predictions and common targets without changing weights.

    Returns ``{'routes': {name: {'scene': [B,S,N,Dz], 'motion': optional}},
    'target_scene': [B,S,N,Dz], 'target_motion': [B,S,N,Dm]}``.

    ``rgb_predictor``, ``scene_persistence``, ``last_motion`` and
    ``predicted_motion`` only consume historical observations. The other routes
    use future observations and must be labelled as diagnostics by the caller.
    For S > 1, ``true_future_event`` recursively consumes its own scene state;
    ``true_state_true_event`` instead starts every step at the encoded true
    preceding scene. They coincide at step 1.

    The requested horizon can exceed the checkpoint's training horizon. Neither
    ``task.config`` nor the model architecture is changed. RGB history and RGB
    targets are each encoded exactly once. Module training flags are restored
    after inference, including when an error occurs.
    """
    model, _, history_events, future_rgb, future_events, steps = _validate_inputs(task, batch, steps)
    previous_modes = [(module, module.training) for module in task.modules()]
    task.eval()
    try:
        history = task.encode_history(batch)
        check_scene(history, model.grid_size, model.scene_dim, history_frames=model.history_frames)

        # This is the original model forward and receives no future tensors.
        predicted = model(history, history_events, steps=steps)
        history_motion = model.encode_target_motion(history_events)
        held_motion = history_motion[:, -1:, :, :].expand(-1, steps, -1, -1)

        rgb_scenes, z = [], history
        for _ in range(steps):
            next_z = task.forecaster.predictor(z)
            check_scene(next_z, model.grid_size, model.scene_dim)
            rgb_scenes.append(next_z)
            z = torch.cat((z[:, 1:], next_z[:, None]), dim=1)

        def transition_rollout(motions):
            state, scenes = history[:, -1], []
            for motion in motions.unbind(dim=1):
                state = model._transition(state, motion)
                scenes.append(state)
            return {"scene": torch.stack(scenes, dim=1), "motion": motions}

        # All history-only paths are computed before future observations enter
        # any encoder. The observed-event route changes only motion conditions.
        routes = {
            "rgb_predictor": {"scene": torch.stack(rgb_scenes, dim=1)},
            "scene_persistence": {"scene": history[:, -1:].expand(-1, steps, -1, -1)},
            "last_motion": transition_rollout(held_motion),
            "predicted_motion": predicted,
        }
        batch_size = future_rgb.shape[0]
        target_scene = task.forecaster.encode_latents(future_rgb.flatten(0, 1)).reshape(
            batch_size, steps, -1, model.scene_dim)
        check_scene(target_scene, model.grid_size, model.scene_dim, history_frames=steps)
        target_motion = model.encode_target_motion(future_events)
        expected_motion = (batch_size, steps, target_scene.shape[2], model.motion_dim)
        if target_motion.shape != expected_motion:
            raise ValueError(f"Future motion has shape {tuple(target_motion.shape)}, expected {expected_motion}")
        routes["true_future_rgb"] = {"scene": target_scene}
        routes["true_future_event"] = transition_rollout(target_motion)
        if steps > 1:
            starts = torch.cat((history[:, -1:], target_scene[:, :-1]), dim=1)
            scenes = [model._transition(starts[:, index], target_motion[:, index])
                      for index in range(steps)]
            routes["true_state_true_event"] = {"scene": torch.stack(scenes, dim=1), "motion": target_motion}
        return {"routes": routes, "target_scene": target_scene, "target_motion": target_motion}
    finally:
        # Restore heterogeneous flags (e.g. a frozen/eval teacher inside a task
        # whose trainable predictor was in training mode) without train hooks.
        for module, was_training in previous_modes:
            module.training = was_training
