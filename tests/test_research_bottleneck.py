from copy import deepcopy

import pytest
import torch
from torch import nn

from foresight.research.bottleneck import diagnostic_predictions
from foresight.research.motion import MotionWorldModel
from foresight.research.transition import TransitionWorldModel


class TinyForecaster(nn.Module):
    pca_dim = 6

    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor(0.7))
        self.encode_calls = []

    def encode_latents(self, images):
        self.encode_calls.append(tuple(images.shape))
        pixels = images.flatten(2).transpose(1, 2)
        return torch.cat((pixels, pixels * self.scale), dim=-1)

    def predictor(self, history):
        return history[:, -1] + 0.3 * (history[:, -1] - history[:, -2])


class TinyTask(nn.Module):
    stage = "motion"

    def __init__(self, **kwargs):
        super().__init__()
        self.forecaster = TinyForecaster()
        teacher = TransitionWorldModel((2, 2), 6, 4, 8, 2)
        nn.init.normal_(teacher.scene_transition.output.weight, std=0.15)
        self.model = MotionWorldModel(teacher, history_frames=3, **kwargs)
        nn.init.normal_(self.model.motion_head.weight, std=0.1)
        self.config = {"data": {"rollout_steps": 1, "context_frames": 3}}
        self.history_encode_calls = 0

    def encode_history(self, batch):
        self.history_encode_calls += 1
        rgb = batch["history_rgb"]
        return self.forecaster.encode_latents(rgb.flatten(0, 1)).reshape(rgb.shape[0], 3, 4, 6)


@pytest.fixture
def example():
    torch.manual_seed(71)
    task = TinyTask()
    batch = {
        "history_rgb": torch.randn(2, 3, 3, 2, 2),
        "history_events": torch.randn(2, 2, 2, 2, 8, 12),
        "future_rgb_target": torch.randn(2, 3, 3, 2, 2),
        "future_events_target": torch.randn(2, 3, 2, 2, 8, 12),
    }
    return task, batch


def test_single_step_shared_encoding_restores_modes_and_weights(example):
    task, batch = example
    before_config = deepcopy(task.config)
    before_weights = {name: value.clone() for name, value in task.state_dict().items()}
    before_modes = {name: module.training for name, module in task.named_modules()}
    result = diagnostic_predictions(task, batch, steps=1)
    assert set(result) == {"routes", "target_scene", "target_motion"}
    assert set(result["routes"]) == {
        "rgb_predictor", "scene_persistence", "true_future_rgb", "true_future_event", "last_motion", "predicted_motion"}
    assert task.history_encode_calls == 1
    assert task.forecaster.encode_calls == [(6, 3, 2, 2), (2, 3, 2, 2)]
    assert result["target_scene"].shape == (2, 1, 4, 6)
    assert result["target_motion"].shape == (2, 1, 4, 4)
    for route in result["routes"].values():
        assert route["scene"].shape == (2, 1, 4, 6)
        assert not route["scene"].requires_grad
    assert task.config == before_config
    assert {name: module.training for name, module in task.named_modules()} == before_modes
    for name, value in task.state_dict().items():
        torch.testing.assert_close(value, before_weights[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in task.parameters())


def test_forecast_routes_cannot_depend_on_future_observations(example):
    task, batch = example
    original = diagnostic_predictions(task, batch)
    altered = dict(batch)
    altered["future_rgb_target"] = torch.randn_like(batch["future_rgb_target"]) * 9
    altered["future_events_target"] = torch.randn_like(batch["future_events_target"]) * 9
    changed = diagnostic_predictions(task, altered)
    for key in ("rgb_predictor", "scene_persistence", "last_motion", "predicted_motion"):
        torch.testing.assert_close(changed["routes"][key]["scene"], original["routes"][key]["scene"], rtol=0, atol=0)
    assert not torch.equal(changed["target_scene"], original["target_scene"])
    assert not torch.equal(changed["routes"]["true_future_event"]["scene"], original["routes"]["true_future_event"]["scene"])


def test_observed_event_rollout_does_not_consume_true_future_rgb(example):
    task, batch = example
    original = diagnostic_predictions(task, batch)
    altered = dict(batch, future_rgb_target=batch["future_rgb_target"] + 5)
    changed = diagnostic_predictions(task, altered)
    torch.testing.assert_close(original["routes"]["true_future_event"]["scene"],
                               changed["routes"]["true_future_event"]["scene"], rtol=0, atol=0)
    assert not torch.equal(original["routes"]["true_state_true_event"]["scene"][:, 1:],
                           changed["routes"]["true_state_true_event"]["scene"][:, 1:])


def test_true_future_event_is_exact_g_then_active_t_and_forecast_is_original_forward(example):
    task, batch = example
    task.eval()
    result = diagnostic_predictions(task, batch)
    with torch.inference_mode():
        history = task.encode_history(batch)
        target_motion = task.model.teacher.encode_motion(batch["future_events_target"])
        state, expected = history[:, -1], []
        for motion in target_motion.unbind(1):
            state = task.model._transition(state, motion)
            expected.append(state)
        torch.testing.assert_close(result["routes"]["true_future_event"]["scene"], torch.stack(expected, 1))
        torch.testing.assert_close(result["target_motion"], target_motion)
        predicted = task.model(history, batch["history_events"], steps=3)
        torch.testing.assert_close(result["routes"]["predicted_motion"]["scene"], predicted["scene"])
        torch.testing.assert_close(result["routes"]["predicted_motion"]["motion"], predicted["motion"])
        held = task.model.encode_target_motion(batch["history_events"])[:, -1:]
        torch.testing.assert_close(result["routes"]["last_motion"]["motion"], held.expand(-1, 3, -1, -1))


def test_multistep_true_state_separates_state_feedback_from_motion_error(example):
    task, batch = example
    result = diagnostic_predictions(task, batch)
    routes = result["routes"]
    recursive = routes["true_future_event"]["scene"]
    true_state = routes["true_state_true_event"]["scene"]
    torch.testing.assert_close(recursive[:, 0], true_state[:, 0])
    assert not torch.equal(recursive[:, 1:], true_state[:, 1:])
    with torch.inference_mode():
        expected_second = task.model._transition(result["target_scene"][:, 0], result["target_motion"][:, 1])
    torch.testing.assert_close(true_state[:, 1], expected_second)
    # Evaluation can exceed the checkpoint's single-step training horizon.
    assert recursive.shape[1] == 3
    assert task.config["data"]["rollout_steps"] == 1


def test_all_transition_routes_use_adapted_t_without_changing_target_g(example):
    _, batch = example
    task = TinyTask(train_transition=True)
    original = diagnostic_predictions(task, batch)
    # Changing the inactive teacher T must not affect any route. Frozen G stays
    # untouched and is the target representation for both checkpoint variants.
    with torch.no_grad():
        task.model.teacher.scene_transition.output.bias.add_(7)
    inactive_changed = diagnostic_predictions(task, batch)
    for name in original["routes"]:
        torch.testing.assert_close(original["routes"][name]["scene"], inactive_changed["routes"][name]["scene"], rtol=0, atol=0)
    with torch.no_grad():
        task.model.adapted_transition.output.bias.add_(2)
    active_changed = diagnostic_predictions(task, batch)
    for name in ("true_future_event", "last_motion", "predicted_motion", "true_state_true_event"):
        assert not torch.equal(original["routes"][name]["scene"], active_changed["routes"][name]["scene"])
        # At the first step each route changes by exactly the same T bias.
        torch.testing.assert_close(active_changed["routes"][name]["scene"][:, 0] - original["routes"][name]["scene"][:, 0], torch.full((2, 4, 6), 2.0))
    torch.testing.assert_close(original["target_motion"], active_changed["target_motion"], rtol=0, atol=0)


@pytest.mark.parametrize("options", [
    {"mode": "direct"}, {"mode": "auxiliary"}, {"motion_input": "rgb"}, {"copy_last_motion": True},
])
def test_rejects_nonexplicit_or_nonlearned_event_models(example, options):
    _, batch = example
    with pytest.raises(ValueError, match="Require explicit"):
        diagnostic_predictions(TinyTask(**options), batch)


@pytest.mark.parametrize("steps", [0, -1, True, 1.5, 4])
def test_rejects_invalid_or_unavailable_horizon(example, steps):
    task, batch = example
    with pytest.raises(ValueError, match="steps"):
        diagnostic_predictions(task, batch, steps=steps)


def test_rejects_missing_and_misaligned_observation_tensors(example):
    task, batch = example
    no_events = {key: value for key, value in batch.items() if key != "future_events_target"}
    with pytest.raises(ValueError, match="future_events_target"):
        diagnostic_predictions(task, no_events)
    wrong_history = dict(batch, history_events=batch["history_events"][:, :1])
    with pytest.raises(ValueError, match="history_events"):
        diagnostic_predictions(task, wrong_history)
    wrong_rgb = dict(batch, future_rgb_target=batch["future_rgb_target"][:1])
    with pytest.raises(ValueError, match="future_rgb_target"):
        diagnostic_predictions(task, wrong_rgb)
    short_future = dict(batch, future_events_target=batch["future_events_target"][:, :1])
    with pytest.raises(ValueError, match="at least steps"):
        diagnostic_predictions(task, short_future)


def test_inference_failure_restores_original_module_modes(example, monkeypatch):
    task, batch = example
    before = {name: module.training for name, module in task.named_modules()}

    def fail(_):
        raise RuntimeError("Injected predictor failure")

    monkeypatch.setattr(task.forecaster, "predictor", fail)
    with pytest.raises(RuntimeError, match="Injected"):
        diagnostic_predictions(task, batch)
    assert {name: module.training for name, module in task.named_modules()} == before
