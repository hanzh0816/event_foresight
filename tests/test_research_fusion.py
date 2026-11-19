import torch
import pytest

from foresight.research.common import EventMotionEncoder, parameter_counts
from foresight.research.fusion import FusionWorldModel


def tiny_inputs():
    torch.manual_seed(17)
    return torch.randn(2, 3, 6, 12), torch.randn(2, 2, 3, 2, 16, 24)


def test_pooled_is_permutation_invariant_but_ordered_retains_time():
    _, events = tiny_inputs()
    pooled = EventMotionEncoder((2, 3), 8, 8, 3, "pooled").eval()
    ordered = EventMotionEncoder((2, 3), 8, 8, 3, "ordered").eval()
    ordered.load_state_dict(pooled.state_dict())
    # A larger deterministic positional scale makes sensitivity unambiguous.
    with torch.no_grad():
        ordered.temporal.position.mul_(10)
    reverse = events.flip(2)
    torch.testing.assert_close(pooled(events), pooled(reverse), atol=1e-6, rtol=1e-6)
    assert (ordered(events) - ordered(reverse)).abs().max().item() > 1e-6
    assert parameter_counts(pooled) == parameter_counts(ordered)


@pytest.mark.parametrize("mode", sorted(FusionWorldModel.MODES))
def test_fusion_starts_at_rgb_prediction_and_can_train(mode):
    history, events = tiny_inputs()
    baseline = torch.randn_like(history[:, -1])
    model = FusionWorldModel((2, 3), 12, 8, 8, 3, 3, mode)
    output = model(history, events, baseline)["scene"]
    torch.testing.assert_close(output[:, 0], baseline)
    if mode == "rgb":
        assert list(model.parameters()) == []
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        output.square().mean().backward()
        optimizer.step()
        assert not torch.equal(model(history, events, baseline)["scene"], output)
    restored = FusionWorldModel(**model.model_spec()["kwargs"])
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(history, events, baseline)["scene"],
                               model(history, events, baseline)["scene"])


def test_input_shape_contracts():
    history, events = tiny_inputs()
    model = FusionWorldModel((2, 3), 12, 8, 8, 3, 3, "event_ordered")
    with pytest.raises(ValueError, match="connect consecutive"):
        model(history, events[:, :1], history[:, -1])
    with pytest.raises(ValueError, match="floating point"):
        model(history, events.long(), history[:, -1])
