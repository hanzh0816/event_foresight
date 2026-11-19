import inspect

import pytest
import torch

from foresight.research.motion import MotionWorldModel
from foresight.research.common import parameter_counts
from foresight.research.transition import TransitionWorldModel


def setup_model(**kwargs):
    torch.manual_seed(31)
    teacher = TransitionWorldModel((2, 3), 12, 8, 8, 3)
    torch.nn.init.normal_(teacher.scene_transition.output.weight, std=0.1)
    model = MotionWorldModel(teacher, history_frames=3, **kwargs)
    z = torch.randn(2, 3, 6, 12)
    events = torch.randn(2, 2, 3, 2, 16, 24)
    return model, z, events


def test_explicit_multistep_scene_loss_trains_motion_through_frozen_transition():
    model, z, events = setup_model()
    model.train()
    assert not model.teacher.training
    output = model(z, events, steps=3)
    assert output["scene"].shape == (2, 3, 6, 12)
    assert output["motion"].shape == (2, 3, 6, 8)
    output["scene"].square().mean().backward()
    assert model.motion_head.weight.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in model.teacher.parameters())
    target = model.encode_target_motion(events.requires_grad_(True))
    assert not target.requires_grad
    assert "future" not in " ".join(inspect.signature(model.forward).parameters)


@pytest.mark.parametrize("mode", ["direct", "auxiliary"])
def test_direct_and_auxiliary_never_use_predicted_motion_for_scene(mode):
    model, z, events = setup_model(mode=mode)
    # A learned nonidentity scene code also must ignore the motion output head.
    torch.nn.init.normal_(model.scene_condition_head.weight, std=0.1)
    reference = model(z, events, steps=3)
    with torch.no_grad():
        model.motion_head.bias.add_(100)
    changed = model(z, events, steps=3)
    torch.testing.assert_close(changed["scene"], reference["scene"])
    assert not torch.equal(changed["motion"], reference["motion"])
    changed["scene"].square().mean().backward()
    assert model.motion_head.weight.grad is None
    assert model.scene_condition_head.weight.grad.abs().sum() > 0


def test_adaptation_keeps_target_teacher_stable_and_roundtrips_spec():
    model, z, events = setup_model(train_transition=True)
    before = model.encode_target_motion(events).clone()
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
    model(z, events, steps=2)["scene"].square().mean().backward()
    assert model.adapted_transition.output.weight.grad.abs().sum() > 0
    optimizer.step()
    torch.testing.assert_close(model.encode_target_motion(events), before)
    spec = model.model_spec()
    restored = MotionWorldModel(TransitionWorldModel(**spec["teacher"]["kwargs"]), **spec["kwargs"])
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(z, events, steps=2)["scene"], model(z, events, steps=2)["scene"])


def test_rgb_history_path_and_last_motion_control():
    rgb, z, events = setup_model(motion_input="rgb")
    assert rgb(z, None, steps=2)["scene"].shape == (2, 2, 6, 12)
    held, z, events = setup_model(copy_last_motion=True)
    output = held(z, events, steps=3)
    last_motion = held.encode_target_motion(events)[:, -1]
    torch.testing.assert_close(output["motion"], last_motion[:, None].expand(-1, 3, -1, -1))
    assert not any(p.requires_grad for p in held.parameters())


def test_matched_modes_share_transition_and_have_no_rgb_predictor_bypass():
    model, z, events = setup_model(mode="direct")

    def base(history):
        raise AssertionError("motion-stage modes must not call the RGB predictor")

    result = model(z, events, steps=2, base_predictor=base)
    explicit, _, _ = setup_model(mode="explicit")
    auxiliary, _, _ = setup_model(mode="auxiliary")
    assert parameter_counts(model) == parameter_counts(explicit) == parameter_counts(auxiliary)
    # All heads start at motion persistence; only where losses are attached
    # differs. Shared teacher initialization and complete parameter shapes match.
    torch.testing.assert_close(result["scene"], explicit(z, events, steps=2)["scene"])
    torch.testing.assert_close(result["scene"], auxiliary(z, events, steps=2)["scene"])
