"""Contracts for Event conditioning of the complete pretrained predictor."""

from copy import deepcopy
import inspect

import pytest
import torch
from torch import nn

from foresight.models.forecaster import DINOFeaturePredictor
from foresight.research.conditioning import ConditioningWorldModel


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def components(mode="event_ordered", frozen=True):
    # Retain twelve actual separable blocks, but use small spatial/channel
    # dimensions so all structural and gradient contracts run on CPU.
    predictor = DINOFeaturePredictor(grid_size=(2, 3), history_length=4,
        pca_dim=6, hidden_dim=12, depth=12, heads=3, dropout=0).eval()
    nn.init.normal_(predictor.maskvit.fc_out.weight, std=0.1)
    predictor.requires_grad_(not frozen)
    model = ConditioningWorldModel((2, 3), 6, 12, motion_dim=4, hidden_dim=8,
        bins=5, history_frames=4, injection_layer=6, attention_heads=2, mode=mode)
    return predictor, model


def histories():
    return torch.randn(2, 4, 6, 6), torch.randn(2, 3, 5, 2, 8, 12)


def activate(model):
    nn.init.normal_(model.output.weight, std=0.15)


@pytest.mark.parametrize("autocast", [False, True])
def test_zero_initialization_preserves_complete_rgb_prediction_and_state_keys(autocast):
    torch.manual_seed(81)
    predictor, model = components()
    z, events = histories()
    before = {name: tensor.clone() for name, tensor in predictor.state_dict().items()}
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        expected = predictor(z)
        result = model(z, events, predictor)
        fallback = predictor.forward_conditioned(z, None, after_layer=6)
    torch.testing.assert_close(result["scene"][:, 0], expected, rtol=0, atol=0)
    torch.testing.assert_close(fallback, expected, rtol=0, atol=0)
    assert result["motion"].shape == (2, 1, 6, 4)
    assert len(predictor.maskvit.transformer.layers) == 12
    assert predictor.state_dict().keys() == before.keys()
    for name, tensor in predictor.state_dict().items():
        assert torch.equal(tensor, before[name])
    # The conditioner owns no second registration of pretrained parameters.
    assert not any("maskvit" in name or "predictor." in name for name in model.state_dict())


def test_rgb_mode_has_no_parameters_and_uses_exact_original_predictor():
    predictor, model = components(mode="rgb")
    z, events = histories()
    assert list(model.parameters()) == []
    assert model.state_dict() == {}
    expected = predictor(z)
    assert torch.equal(model(z, None, predictor)["scene"][:, 0], expected)
    assert torch.equal(model(z, events, predictor)["scene"][:, 0], expected)


def test_injection_is_after_six_complete_blocks_and_changes_only_future_slot():
    torch.manual_seed(17)
    predictor, _ = components()
    z, _ = histories()
    batch = z.shape[0]
    observed = z.reshape(batch, 4, 2, 3, 6)
    tokens = torch.cat((predictor.embed(observed), predictor.replace_vector.expand(batch, 1, 2, 3, -1)), dim=1)
    positioned = predictor.maskvit.pos_embd(tokens)
    shape = tuple(positioned.shape)
    prefix = positioned.flatten(1, 3)
    for layer in predictor.maskvit.transformer.layers[:6]:
        prefix = predictor._block_forward(prefix, layer, shape)
    prefix = prefix.reshape(batch, 5, 6, 12)
    received, entered_seventh, calls = [], [], []

    def condition(future_hidden):
        received.append(future_hidden.detach().clone())
        return torch.full_like(future_hidden, 0.25)

    # The next block's temporal attention sees [B*N, T, D].
    handle = predictor.maskvit.transformer.layers[6][0].register_forward_pre_hook(
        lambda module, args: entered_seventh.append(args[0].detach().clone()))
    counters = [layer[2].register_forward_pre_hook(lambda module, args, i=i: calls.append(i))
                for i, layer in enumerate(predictor.maskvit.transformer.layers)]
    try:
        predictor.forward_conditioned(z, condition, after_layer=6)
    finally:
        handle.remove()
        for counter in counters:
            counter.remove()
    assert calls == list(range(12)) and len(received) == len(entered_seventh) == 1
    torch.testing.assert_close(received[0], prefix[:, -1], rtol=0, atol=0)
    actual = entered_seventh[0].reshape(batch, 6, 5, 12).permute(0, 2, 1, 3)
    torch.testing.assert_close(actual[:, :4], prefix[:, :4], rtol=0, atol=0)
    torch.testing.assert_close(actual[:, -1], prefix[:, -1] + 0.25, rtol=0, atol=0)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_zero_projection_learns_then_transmits_scene_gradient_through_frozen_suffix(checkpointed):
    torch.manual_seed(19)
    predictor, model = components()
    predictor.activation_checkpointing = checkpointed
    z, events = histories()
    events.requires_grad_(True)
    target = torch.randn(2, 1, 6, 6)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ((model(z, events, predictor)["scene"] - target) ** 2).mean().backward()
    assert model.output.weight.grad.abs().sum() > 0
    assert events.grad is not None and events.grad.count_nonzero() == 0
    assert all(parameter.grad is None for parameter in predictor.parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    events.grad = None
    ((model(z, events, predictor)["scene"] - target) ** 2).mean().backward()
    for parameter in (model.event_stem[0].weight, model.interval_position,
                      model.bin_position, model.query[1].weight,
                      model.cross_attention.in_proj_weight, model.motion_head.weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    assert events.grad is not None and events.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in predictor.parameters())


@pytest.mark.parametrize("frozen", [True, False])
def test_checkpointed_blocks_preserve_condition_calls_outputs_and_gradients(frozen):
    torch.manual_seed(29)
    predictor, model = components(frozen=frozen)
    activate(model)
    checkpointed, other_model = deepcopy(predictor), deepcopy(model)
    checkpointed.activation_checkpointing = True
    z, events = histories()
    z.requires_grad_(True)
    events.requires_grad_(True)
    other_z, other_events = z.detach().clone().requires_grad_(True), events.detach().clone().requires_grad_(True)
    calls, other_calls = [], []
    first = model.output.register_forward_hook(lambda *args: calls.append(1))
    second = other_model.output.register_forward_hook(lambda *args: other_calls.append(1))
    try:
        plain = model(z, events, predictor)
        actual = other_model(other_z, other_events, checkpointed)
        torch.testing.assert_close(actual["scene"], plain["scene"], rtol=0, atol=0)
        torch.testing.assert_close(actual["motion"], plain["motion"], rtol=0, atol=0)
        plain["scene"].square().mean().backward()
        actual["scene"].square().mean().backward()
    finally:
        first.remove()
        second.remove()
    assert calls == other_calls == [1]
    for left, right in ((z.grad, other_z.grad), (events.grad, other_events.grad)):
        assert left is not None and left.abs().sum() > 0
        torch.testing.assert_close(right, left, rtol=2e-5, atol=1e-7)
    for first_module, second_module in ((model, other_model), (predictor, checkpointed)):
        for (left_name, left), (right_name, right) in zip(first_module.named_parameters(), second_module.named_parameters(), strict=True):
            assert left_name == right_name
            if not left.requires_grad:
                assert left.grad is None and right.grad is None
            else:
                assert left.grad is not None and right.grad is not None
                torch.testing.assert_close(right.grad, left.grad, rtol=2e-5, atol=1e-7)


def test_all_event_bins_remain_tokens_and_pooled_removes_only_bin_order():
    torch.manual_seed(38)
    predictor, ordered = components()
    activate(ordered)
    _, pooled = components(mode="event_pooled")
    pooled.load_state_dict(ordered.state_dict(), strict=True)
    z, events = histories()
    assert sum(p.numel() for p in ordered.parameters()) == sum(p.numel() for p in pooled.parameters())
    assert ordered.state_dict().keys() == pooled.state_dict().keys()
    assert ordered.encode_events(events).shape == pooled.encode_events(events).shape == (2, 6, 15, 8)
    shuffled = events[:, :, [4, 2, 0, 3, 1]]
    ordered_first = ordered(z, events, predictor)["scene"]
    ordered_other = ordered(z, shuffled, predictor)["scene"]
    assert (ordered_first - ordered_other).abs().max() > 1e-7
    # Reduction roundoff can differ with summation order after a permutation.
    torch.testing.assert_close(pooled.encode_events(events), pooled.encode_events(shuffled), rtol=1e-6, atol=3e-7)
    torch.testing.assert_close(pooled(z, events, predictor)["scene"], pooled(z, shuffled, predictor)["scene"], rtol=1e-6, atol=3e-7)
    # The pooled control retains chronology between the three RGB intervals.
    reversed_intervals = events.flip(1)
    assert not torch.equal(pooled(z, events, predictor)["scene"], pooled(z, reversed_intervals, predictor)["scene"])
    assert not torch.equal(ordered_first, ordered(z, torch.zeros_like(events), predictor)["scene"])


def test_bounded_spatial_attention_matches_unchunked_outputs_and_gradients():
    torch.manual_seed(43)
    predictor, model = components()
    activate(model)
    chunked = deepcopy(model)
    chunked.MAX_BATCH_SITES = 5
    z, events = histories()
    first = model(z, events, predictor)["scene"]
    second = chunked(z, events, predictor)["scene"]
    torch.testing.assert_close(second, first, rtol=1e-6, atol=1e-7)
    first.square().mean().backward()
    second.square().mean().backward()
    for left, right in zip(model.parameters(), chunked.parameters(), strict=True):
        torch.testing.assert_close(right.grad, left.grad, rtol=2e-5, atol=1e-7)


def test_model_spec_and_state_roundtrip_do_not_store_predictor_or_future_inputs(tmp_path):
    torch.manual_seed(53)
    predictor, model = components()
    activate(model)
    z, events = histories()
    spec = model.model_spec()
    assert spec["kind"] == "conditioning"
    path = tmp_path / "conditioning.pt"
    torch.save({"model_spec": spec, "state_dict": model.state_dict()}, path)
    saved = torch.load(path, weights_only=True)
    restored = ConditioningWorldModel(**saved["model_spec"]["kwargs"])
    restored.load_state_dict(saved["state_dict"], strict=True)
    assert torch.equal(model(z, events, predictor)["scene"], restored(z, events, predictor)["scene"])
    assert set(inspect.signature(model.forward).parameters) == {"z_history", "events_history", "base_predictor"}
    with pytest.raises(TypeError):
        model(z, events, predictor, future_events_target=torch.randn_like(events))
    # An accidentally appended future interval is rejected, not read.
    with pytest.raises(ValueError, match="historical intervals"):
        model(z, torch.cat((events, events[:, :1]), dim=1), predictor)


@pytest.mark.parametrize("options", [
    {"history_frames": 1}, {"bins": 0}, {"motion_dim": 0},
    {"predictor_dim": 0}, {"injection_layer": 0}, {"injection_layer": True},
    {"attention_heads": 3}, {"hidden_dim": 0}, {"mode": "unknown"},
])
def test_rejects_invalid_dimensions_and_position_configuration(options):
    kwargs = dict(grid_size=(2, 3), scene_dim=6, predictor_dim=12,
                  motion_dim=4, hidden_dim=8, attention_heads=2)
    kwargs.update(options)
    with pytest.raises(ValueError):
        ConditioningWorldModel(**kwargs)


def test_predictor_rejects_invalid_injection_and_condition_shapes():
    predictor, _ = components()
    z, _ = histories()
    for layer in (0, 12, 13, 1.5, True):
        with pytest.raises(ValueError, match="after_layer"):
            predictor.forward_conditioned(z, torch.zeros_like, layer)
    with pytest.raises(TypeError, match="callable"):
        predictor.forward_conditioned(z, 0, 6)
    with pytest.raises(ValueError, match="shape"):
        predictor.forward_conditioned(z, lambda hidden: hidden[:, :1], 6)
    with pytest.raises(ValueError, match="floating point"):
        predictor.forward_conditioned(z, lambda hidden: torch.ones_like(hidden, dtype=torch.long), 6)
