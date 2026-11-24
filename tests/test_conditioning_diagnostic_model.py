"""Diagnostic API contracts; execute on the experiment server, not during authoring."""

from copy import deepcopy

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


def components(mode="event_ordered", activated=True):
    # Use the actual twelve-block predictor so this exercises injection and
    # backpropagation through its suffix, without external weights or data.
    predictor = DINOFeaturePredictor(grid_size=(2, 3), history_length=4,
        pca_dim=6, hidden_dim=12, depth=12, heads=3, dropout=0).eval()
    nn.init.normal_(predictor.maskvit.fc_out.weight, std=0.1)
    predictor.requires_grad_(False)
    model = ConditioningWorldModel((2, 3), 6, 12, motion_dim=4, hidden_dim=8,
        bins=5, history_frames=4, injection_layer=6, attention_heads=2, mode=mode).eval()
    if activated:
        nn.init.normal_(model.output.weight, std=0.15)
        nn.init.normal_(model.output.bias, std=0.05)
    return predictor, model


def histories():
    return torch.randn(2, 4, 6, 6), torch.randn(2, 3, 5, 2, 8, 12)


def legacy_forecast(model, predictor, z, events):
    """The pre-diagnostics condition arithmetic, as a regression oracle."""
    memory = model.encode_events(events)
    b, spatial, length, _ = memory.shape
    saved_motion = []

    def condition(hidden):
        query = model.query(hidden).reshape(b * spatial, 1, model.hidden_dim)
        normalized = model.memory_norm(memory).reshape(b * spatial, length, model.hidden_dim)
        parts = []
        for q, kv in zip(query.split(model.MAX_BATCH_SITES), normalized.split(model.MAX_BATCH_SITES), strict=True):
            value = q + model.cross_attention(q, kv, kv, need_weights=False)[0]
            parts.append(value + model.ffn(value))
        motion = model.motion_head(torch.cat(parts).reshape(b, spatial, model.hidden_dim))
        saved_motion.append(motion)
        return model.output(motion)

    scene = predictor.forward_conditioned(z, condition, model.injection_layer)
    return scene.unsqueeze(1), saved_motion[0].unsqueeze(1)


@pytest.mark.parametrize("mode", ["event_ordered", "event_pooled"])
@pytest.mark.parametrize("activated", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
def test_full_diagnostic_matches_existing_forward_and_legacy_arithmetic(mode, activated, autocast):
    torch.manual_seed(101)
    predictor, model = components(mode, activated)
    # Force more than one attention chunk to exercise concatenation as well.
    model.MAX_BATCH_SITES = 5
    z, events = histories()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        legacy_scene, legacy_motion = legacy_forecast(model, predictor, z, events)
        normal = model(z, events, predictor)
        actual = model.forward_diagnostic(z, events, predictor)
        expected_memory = model.encode_events(events)
    assert set(normal) == {"scene", "motion"}
    assert set(actual) == {"scene", "motion", "diagnostics"}
    for key, expected in (("scene", legacy_scene), ("motion", legacy_motion)):
        torch.testing.assert_close(normal[key], expected, rtol=0, atol=0)
        torch.testing.assert_close(actual[key], expected, rtol=0, atol=0)
    diagnostics = actual["diagnostics"]
    shapes = {"encoded_events": (2, 6, 15, 8), "memory": (2, 6, 15, 8),
              "query": (2, 6, 8), "attention": (2, 6, 8), "hidden": (2, 6, 12),
              "update": (2, 6, 12), "motion": (2, 6, 4)}
    assert set(diagnostics) == set(shapes)
    for name, shape in shapes.items():
        assert diagnostics[name].shape == shape
    torch.testing.assert_close(diagnostics["motion"], actual["motion"][:, 0], rtol=0, atol=0)
    torch.testing.assert_close(diagnostics["memory"], expected_memory, rtol=0, atol=0)


def test_encoded_events_excludes_positions_and_memory_adds_them_once():
    torch.manual_seed(102)
    predictor, model = components()
    z, events = histories()
    with torch.no_grad():
        diagnostics = model.forward_diagnostic(z, events, predictor)["diagnostics"]
    positions = (model.interval_position + model.bin_position).reshape(1, 1, 15, 8)
    torch.testing.assert_close(diagnostics["memory"], diagnostics["encoded_events"] + positions, rtol=0, atol=0)


def test_query_only_removes_event_dependence_but_keeps_rgb_adapter():
    torch.manual_seed(103)
    predictor, model = components()
    z, events = histories()
    other_events = torch.randn_like(events)
    with torch.no_grad():
        query_only = model.forward_diagnostic(z, events, predictor, condition_mode="query_only")
        other_query_only = model.forward_diagnostic(z, other_events, predictor, condition_mode="query_only")
        full = model.forward_diagnostic(z, events, predictor)
        other_full = model.forward_diagnostic(z, other_events, predictor)
        baseline = predictor(z)
    for key in ("scene", "motion"):
        torch.testing.assert_close(query_only[key], other_query_only[key], rtol=0, atol=0)
    assert query_only["diagnostics"]["attention"].count_nonzero() == 0
    assert not torch.equal(query_only["scene"][:, 0], baseline)
    assert not torch.equal(full["scene"], other_full["scene"])
    assert not torch.equal(full["diagnostics"]["attention"], other_full["diagnostics"]["attention"])


@pytest.mark.parametrize("condition_mode", ["full", "query_only"])
def test_zero_scale_matches_rgb_and_scale_multiplies_complete_update(condition_mode):
    torch.manual_seed(104)
    predictor, model = components()
    z, events = histories()
    with torch.no_grad():
        baseline = predictor(z)
        zero = model.forward_diagnostic(z, events, predictor, condition_mode=condition_mode, condition_scale=0.0)
        full = model.forward_diagnostic(z, events, predictor, condition_mode=condition_mode)
        half = model.forward_diagnostic(z, events, predictor, condition_mode=condition_mode, condition_scale=0.5)
    torch.testing.assert_close(zero["scene"][:, 0], baseline, rtol=0, atol=0)
    assert zero["diagnostics"]["update"].count_nonzero() == 0
    torch.testing.assert_close(half["diagnostics"]["update"], full["diagnostics"]["update"] * 0.5, rtol=0, atol=0)
    torch.testing.assert_close(half["motion"], full["motion"], rtol=0, atol=0)


def test_checkpoint_keys_and_spec_remain_legacy_compatible_and_no_activations_persist():
    torch.manual_seed(105)
    predictor, model = components()
    parameter_modules = ["event_stem.0", "event_stem.1", "event_stem.3", "event_stem.4",
                         "event_stem.7", "event_stem.9", "query.0", "query.1", "memory_norm",
                         "cross_attention.out_proj", "ffn.0", "ffn.1", "ffn.3", "motion_head", "output"]
    expected = {f"{prefix}.{suffix}" for prefix in parameter_modules for suffix in ("weight", "bias")}
    expected.update({"interval_position", "bin_position", "cross_attention.in_proj_weight",
                     "cross_attention.in_proj_bias"})
    checkpoint = deepcopy(model.state_dict())
    assert set(checkpoint) == expected
    spec = deepcopy(model.model_spec())
    attributes = set(vars(model))
    z, events = histories()
    with torch.no_grad():
        model.forward_diagnostic(z, events, predictor)
    assert set(vars(model)) == attributes
    assert model.model_spec() == spec
    assert set(model.state_dict()) == expected
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, checkpoint[name], rtol=0, atol=0)
    restored = ConditioningWorldModel(**spec["kwargs"])
    restored.load_state_dict(checkpoint, strict=True)
    assert set(restored.state_dict()) == expected


def test_full_diagnostics_preserve_parameter_gradients_and_expose_connected_motion():
    torch.manual_seed(106)
    predictor, model = components()
    plain = deepcopy(model)
    z, events = histories()
    events.requires_grad_(True)
    other_events = events.detach().clone().requires_grad_(True)
    diagnostic = model.forward_diagnostic(z, events, predictor)
    diagnostic["diagnostics"]["motion"].retain_grad()
    normal = plain(z, other_events, predictor)
    diagnostic["scene"].square().mean().backward()
    normal["scene"].square().mean().backward()
    for (name, parameter), (other_name, other) in zip(model.named_parameters(), plain.named_parameters(), strict=True):
        assert name == other_name
        assert parameter.grad is not None and other.grad is not None
        torch.testing.assert_close(parameter.grad, other.grad, rtol=2e-5, atol=1e-7)
    assert events.grad is not None and events.grad.abs().sum() > 0
    torch.testing.assert_close(events.grad, other_events.grad, rtol=2e-5, atol=1e-7)
    assert diagnostic["diagnostics"]["motion"].grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in predictor.parameters())


def test_query_only_blocks_scene_gradient_to_event_encoder_and_attention():
    torch.manual_seed(107)
    predictor, model = components()
    z, events = histories()
    events.requires_grad_(True)
    result = model.forward_diagnostic(z, events, predictor, condition_mode="query_only")
    result["scene"].square().mean().backward()
    for parameter in (model.event_stem[0].weight, model.interval_position, model.bin_position,
                      model.cross_attention.in_proj_weight):
        assert parameter.grad is None
    assert events.grad is None
    assert model.query[1].weight.grad.abs().sum() > 0
    assert model.output.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("kwargs", [{"condition_mode": "off"}, {"condition_mode": "unknown"},
    {"condition_scale": float("nan")}, {"condition_scale": float("inf")},
    {"condition_scale": True}, {"condition_scale": "1"}])
def test_rejects_ambiguous_or_invalid_interventions(kwargs):
    predictor, model = components()
    z, events = histories()
    with pytest.raises(ValueError):
        model.forward_diagnostic(z, events, predictor, **kwargs)
