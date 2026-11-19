"""Stage-two/three supervision boundaries and gradient-path regressions."""

from copy import deepcopy

import pytest
import torch

from foresight.research.task import load_task
from test_research_training import TinyForecaster, save_task, tiny_batch, tiny_config


def trained_transition(tmp_path):
    task = load_task(tiny_config("transition"), forecaster=TinyForecaster())
    optimizer = task.configure_optimizers()
    batch = tiny_batch()
    for _ in range(3):
        optimizer.zero_grad()
        task.compute_losses(batch)["loss"].backward()
        optimizer.step()
    checkpoint = tmp_path / "teacher.ckpt"
    save_task(task, checkpoint)
    return task, checkpoint


def test_transition_supervision_is_observed_interval_diagnostic():
    task = load_task(tiny_config("transition"), forecaster=TinyForecaster())
    assert task.protocol["prediction_kind"] == "observed_transition"
    batch = tiny_batch()
    del batch["future_events_target"]
    with pytest.raises(KeyError):
        task.predict_batch(batch)


def test_predicted_motion_receives_scene_gradient_through_frozen_teacher(tmp_path):
    torch.set_num_threads(1)
    _, checkpoint = trained_transition(tmp_path)
    config = tiny_config("motion", teacher_checkpoint=str(checkpoint))
    config["loss"]["motion_weight"] = 0.0
    task = load_task(config, forecaster=TinyForecaster())
    batch = tiny_batch()
    batch["future_rgb_target"].requires_grad_()
    batch["future_events_target"].requires_grad_()
    causal = {key: value for key, value in batch.items() if not key.startswith("future_")}
    assert torch.equal(task.predict_batch(batch)["scene"], task.predict_batch(causal)["scene"])
    task.compute_losses(batch)["loss"].backward()
    assert all(parameter.grad is None for parameter in task.model.teacher.parameters())
    assert task.model.motion_head.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in task.forecaster.parameters())
    assert batch["future_rgb_target"].grad is None
    assert batch["future_events_target"].grad is None


def test_motion_checkpoint_embeds_teacher_and_rejects_changed_recipe(tmp_path):
    _, source = trained_transition(tmp_path)
    config = tiny_config("motion", teacher_checkpoint=str(source))
    config["data"]["rollout_steps"] = 2
    task = load_task(config, forecaster=TinyForecaster()).eval()
    batch = tiny_batch(2)
    checkpoint = tmp_path / "motion.ckpt"
    save_task(task, checkpoint)
    source.unlink()
    restored = load_task(config, forecaster=TinyForecaster(), checkpoint=checkpoint).eval()
    assert restored.teacher_provenance["sha256"]
    assert torch.equal(task.predict_batch(batch)["scene"], restored.predict_batch(batch)["scene"])
    changed = deepcopy(config)
    changed["data"]["rollout_steps"] = 1
    with pytest.raises(ValueError, match="recipe differs"):
        load_task(changed, forecaster=TinyForecaster(), checkpoint=checkpoint)


def test_direct_control_does_not_require_future_event_targets(tmp_path):
    _, source = trained_transition(tmp_path)
    config = tiny_config("motion", teacher_checkpoint=str(source))
    config["model"]["mode"] = "direct"
    config["loss"]["motion_weight"] = 0.0
    task = load_task(config, forecaster=TinyForecaster())
    batch = tiny_batch()
    del batch["future_events_target"]
    task.compute_losses(batch)["loss"].backward()
    assert task.model.motion_head.weight.grad is None


def test_joint_transition_adaptation_preserves_fixed_target_teacher(tmp_path):
    _, source = trained_transition(tmp_path)
    config = tiny_config("motion", teacher_checkpoint=str(source), train_transition=True)
    task = load_task(config, forecaster=TinyForecaster()).train()
    batch = tiny_batch()
    target_before = task.model.encode_target_motion(batch["future_events_target"]).clone()
    optimizer = task.configure_optimizers()
    task.compute_losses(batch)["loss"].backward()
    assert any(parameter.grad is not None for parameter in task.model.adapted_transition.parameters())
    optimizer.step()
    assert torch.equal(target_before, task.model.encode_target_motion(batch["future_events_target"]))


def test_refinement_initializes_motion_weights_but_allows_new_rollout(tmp_path):
    _, teacher = trained_transition(tmp_path)
    first = load_task(tiny_config("motion", teacher_checkpoint=str(teacher)), forecaster=TinyForecaster())
    with torch.no_grad():
        first.model.motion_head.weight.fill_(0.25)
    source = tmp_path / "first_motion.ckpt"
    save_task(first, source)
    config = tiny_config("motion", teacher_checkpoint=str(teacher), init_checkpoint=str(source), train_transition=True)
    config["data"]["rollout_steps"] = 2
    config["loss"]["motion_weight"] = 0.25
    refined = load_task(config, forecaster=TinyForecaster())
    assert torch.equal(first.model.motion_head.weight, refined.model.motion_head.weight)
    assert refined.initialization_provenance["optimizer"] == "reset"
    assert refined.config["data"]["rollout_steps"] == 2
    assert refined.config["model"]["train_transition"] is True
    assert refined.config["loss"]["motion_weight"] == 0.25
    assert first.config["data"]["rollout_steps"] == 1
    assert not refined.configure_optimizers().state
    future_events = tiny_batch(2)["future_events_target"]
    assert torch.equal(first.model.encode_target_motion(future_events), refined.model.encode_target_motion(future_events))
    for name, value in refined.model.teacher.scene_transition.state_dict().items():
        assert torch.equal(value, refined.model.adapted_transition.state_dict()[name])
    refined.compute_losses(tiny_batch(2))["loss"].backward()
    checkpoint = tmp_path / "refined.ckpt"
    save_task(refined, checkpoint)
    source.unlink()
    teacher.unlink()
    restored = load_task(config, forecaster=TinyForecaster(), checkpoint=checkpoint)
    assert torch.equal(restored.model.motion_head.weight, refined.model.motion_head.weight)
