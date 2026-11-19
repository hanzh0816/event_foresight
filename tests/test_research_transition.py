import torch

from foresight.research.transition import TransitionWorldModel


def make_teacher(transition_mode="motion"):
    torch.manual_seed(23)
    teacher = TransitionWorldModel((2, 3), 12, 8, 8, 3,
                                   transition_mode=transition_mode)
    # Stand in for an already learned transition so motion-path derivatives
    # exist; residual adapters intentionally start at identity before learning.
    torch.nn.init.normal_(teacher.scene_transition.output.weight, std=0.1)
    return teacher


def test_observed_transition_rollout_and_encoder_gradients():
    model = make_teacher()
    start = torch.randn(2, 6, 12)
    events = torch.randn(2, 2, 3, 2, 16, 24)
    result = model(start, events)
    assert result["scene"].shape == (2, 2, 6, 12)
    assert result["motion"].shape == (2, 2, 6, 8)
    result["scene"].square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.event_encoder.parameters())
    reconstructed = TransitionWorldModel(**model.model_spec()["kwargs"])
    reconstructed.load_state_dict(model.state_dict())
    torch.testing.assert_close(reconstructed(start, events)["scene"], result["scene"])


def test_retrained_no_motion_architecture_has_no_event_dependency():
    model = make_teacher("no_motion")
    start = torch.randn(2, 6, 12)
    a, b = torch.randn(2, 6, 8), torch.randn(2, 6, 8)
    torch.testing.assert_close(model.transition(start, a), model.transition(start, b))
