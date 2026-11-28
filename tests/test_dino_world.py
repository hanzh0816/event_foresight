"""CPU contracts for the v3.1 RGB/Event predictor and frozen DINO encoder."""

from copy import deepcopy
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from foresight.models import encoder, forecaster
from foresight.models.forecaster import DINOWorldForecaster
from foresight.models.predictor import DINOWorldPredictor, PREDICTOR_DEFAULTS
from foresight.models.attention import CausalCrossAttention, apply_rope_3d, rope_3d_tables
from foresight.timing import ForecastTiming


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def small_predictor(**options):
    settings = dict(grid_size=(2, 3), history_length=4, feature_dim=12,
                    hidden_dim=64, depth=2, heads=1, mlp_ratio=2., dropout=0.,
                    query_chunk_size=0)
    settings.update(options)
    return DINOWorldPredictor(**settings)


def regular_timing(frames=4):
    return ForecastTiming.from_config({'data': {'context_frames': frames, 'frame_interval_s': .05}})


def assert_parameter_gradients_match(left, right):
    for (left_name, left_value), (right_name, right_value) in zip(
            left.named_parameters(), right.named_parameters(), strict=True):
        assert left_name == right_name
        assert left_value.grad is not None and right_value.grad is not None, left_name
        assert torch.isfinite(left_value.grad).all(), left_name
        torch.testing.assert_close(right_value.grad, left_value.grad, rtol=2e-5, atol=2e-7)


def test_rope_rotates_time_16_y_24_x_24_and_spatial_attention_leaves_time_channels():
    torch.manual_seed(7)
    value = torch.randn(2, 3, 6, 64, requires_grad=True)
    coordinates = torch.tensor([[0., 0., 0.], [.05, 0., 1.], [.1, 0., 2.],
                                [.2, 1., 0.], [.4, 1., 1.], [.8, 1., 2.]])
    actual = apply_rope_3d(value, rope_3d_tables(coordinates))
    expected = value.clone()
    start = 0
    for axis, width in enumerate((16, 24, 24)):
        frequencies = 2 * math.pi / 10 ** torch.linspace(-2, 2, width // 2)
        angle = coordinates[:, axis, None] * frequencies[None, :]
        even, odd = value[..., start:start + width:2], value[..., start + 1:start + width:2]
        expected[..., start:start + width:2] = even * angle.cos() - odd * angle.sin()
        expected[..., start + 1:start + width:2] = even * angle.sin() + odd * angle.cos()
        start += width
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual.square().sum(-1), value.square().sum(-1))
    spatial_coordinates = coordinates.clone()
    spatial_coordinates[:, 0] = 0
    spatial = apply_rope_3d(value, rope_3d_tables(spatial_coordinates))
    torch.testing.assert_close(spatial[..., :16], value[..., :16], rtol=0, atol=0)
    actual.square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()


def test_predictor_has_gradients_with_explicit_next_frame_times():
    torch.manual_seed(13)
    predictor = small_predictor().train()
    history = torch.randn(2, 4, 6, 12, requires_grad=True)
    actual = predictor(history, **regular_timing().forecast_kwargs())
    assert actual.shape == (2, 6, 12)
    explicit = predictor.predict_at(history, **regular_timing().forecast_kwargs())
    torch.testing.assert_close(actual, explicit[:, 0])
    actual.square().mean().backward()
    assert history.grad is not None and history.grad.abs().sum() > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in predictor.parameters())


@pytest.mark.parametrize('target_index', [0, 1, 2])
def test_sequence_prediction_cannot_read_same_time_or_future_context(target_index):
    torch.manual_seed(19)
    predictor = small_predictor().eval()
    history = torch.randn(2, 4, 6, 12)
    altered = history.clone()
    unavailable = target_index + 1
    altered[:, unavailable:] = 100 * torch.randn_like(altered[:, unavailable:])
    original = predictor.predict_sequence(history, **regular_timing().sequence_kwargs())
    changed = predictor.predict_sequence(altered, **regular_timing().sequence_kwargs())
    assert original.shape == (2, 4, 6, 12)
    # Each training target excludes the RGB at its own time and all later RGB.
    torch.testing.assert_close(changed[:, target_index], original[:, target_index], rtol=0, atol=0)
    assert not torch.allclose(changed[:, -1], original[:, -1])
    differentiable_history = history.clone().requires_grad_(True)
    differentiable_prediction = predictor.predict_sequence(
        differentiable_history, **regular_timing().sequence_kwargs())
    differentiable_prediction[:, target_index].square().mean().backward()
    assert differentiable_history.grad[:, :unavailable].abs().sum() > 0
    assert torch.count_nonzero(differentiable_history.grad[:, unavailable:]) == 0


def test_query_time_changes_prediction_even_with_identical_visible_context():
    torch.manual_seed(23)
    predictor = small_predictor().eval()
    history = torch.randn(1, 4, 6, 12)
    times = torch.tensor([0., .05, .10, .15])
    predictions = predictor.predict_at(history, times, torch.tensor([.17, .19]))
    assert predictions.shape == (1, 2, 6, 12)
    assert not torch.allclose(predictions[:, 0], predictions[:, 1])
    torch.testing.assert_close(predictor(history, context_times=times, query_times=torch.tensor([.17])),
                               predictions[:, 0])


def test_rollout_feeds_predictions_back_and_keeps_gradient_path():
    torch.manual_seed(31)
    predictor = small_predictor().eval()
    history = torch.randn(2, 4, 6, 12, requires_grad=True)
    timing = regular_timing().forecast_kwargs(horizon_steps=3)
    actual = predictor.rollout(history, **timing)
    assert actual.shape == (2, 3, 6, 12)
    context, times = history, timing['context_times']
    expected = []
    for target in timing['query_times'].split(1):
        prediction = predictor(context, context_times=times, query_times=target)
        expected.append(prediction)
        context = torch.cat((context[:, 1:], prediction[:, None]), dim=1)
        times = torch.cat((times[1:], target))
    torch.testing.assert_close(actual, torch.stack(expected, dim=1))
    actual[:, -1].square().mean().backward()
    assert history.grad is not None and torch.isfinite(history.grad).all()
    assert history.grad.abs().sum() > 0


@pytest.mark.parametrize('autocast_dtype', [None, torch.bfloat16])
def test_activation_checkpointing_matches_outputs_and_all_gradients(autocast_dtype):
    torch.manual_seed(37)
    plain = small_predictor().train()
    recompute = deepcopy(plain)
    recompute.activation_checkpointing = True
    history = torch.randn(2, 4, 6, 12, requires_grad=True)
    other_history = history.detach().clone().requires_grad_(True)
    with torch.autocast('cpu', dtype=autocast_dtype or torch.bfloat16,
                        enabled=autocast_dtype is not None):
        expected = plain.predict_sequence(history, **regular_timing().sequence_kwargs())
        actual = recompute.predict_sequence(other_history, **regular_timing().sequence_kwargs())
    if autocast_dtype is not None:
        assert actual.dtype == expected.dtype == autocast_dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(other_history.grad, history.grad, rtol=2e-5, atol=2e-7)
    assert_parameter_gradients_match(plain, recompute)


@pytest.mark.parametrize('activation_checkpointing', [False, True])
def test_query_chunking_matches_outputs_and_all_gradients(activation_checkpointing):
    torch.manual_seed(41)
    whole = small_predictor().train()
    whole.activation_checkpointing = activation_checkpointing
    chunked = deepcopy(whole)
    chunked.query_chunk_size = 5
    history = torch.randn(2, 4, 6, 12, requires_grad=True)
    other_history = history.detach().clone().requires_grad_(True)
    times = torch.tensor([0., .05, .10, .15])
    queries = torch.tensor([.025, .075, .125, .175])
    expected = whole.predict_at(history, times, queries)
    actual = chunked.predict_at(other_history, times, queries)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(other_history.grad, history.grad, rtol=2e-5, atol=2e-7)
    assert_parameter_gradients_match(whole, chunked)


@pytest.mark.parametrize('query_chunk_size', [0, 1, 5])
def test_attention_projects_context_once_per_layer_independent_of_query_chunks(query_chunk_size):
    predictor = small_predictor(query_chunk_size=query_chunk_size).eval()
    counts = {block.rgb_attention.kv: 0 for block in predictor.blocks}

    def record_projection(module, inputs, output):
        counts[module] += 1

    handles = [module.register_forward_hook(record_projection) for module in counts]
    try:
        with torch.no_grad():
            predictor.predict_sequence(torch.randn(2, 4, 6, 12), **regular_timing().sequence_kwargs())
    finally:
        for handle in handles:
            handle.remove()
    assert list(counts.values()) == [1] * len(predictor.blocks)


@pytest.mark.parametrize('options', [dict(hidden_dim=32), dict(hidden_dim=64, heads=2)])
def test_predictor_requires_64_dimensional_attention_heads(options):
    with pytest.raises(ValueError):
        small_predictor(**options)


def test_model_source_does_not_download_weights_or_use_hub():
    source = Path(forecaster.__file__).read_text() + Path(encoder.__file__).read_text()
    assert 'torch.hub.' not in source
    assert 'load_state_dict_from_url' not in source


class TinyBackbone(nn.Module):
    """Keep real encoder dimensions without allocating a real ViT."""

    def __init__(self, **options):
        super().__init__()
        self.scale = nn.Parameter(torch.linspace(.5, 2., 768))
        self.seen = []

    def get_intermediate_layers(self, images, n, reshape):
        self.seen.append(dict(batch=len(images), layers=n, reshape=reshape,
                              training=self.training, grad_enabled=torch.is_grad_enabled()))
        offsets = torch.arange(6, device=images.device, dtype=images.dtype)
        features = (images.mean((1, 2, 3))[:, None, None]
                    + offsets[None, :, None] + self.scale[None, None, :])
        return (features,)


def tiny_forecaster(checkpoint):
    return DINOWorldForecaster(
        checkpoint, image_size=(28, 42), history_length=4, encoder_chunk_size=1,
        predictor_options=dict(hidden_dim=64, heads=1, depth=2, mlp_ratio=2., query_chunk_size=0))


def test_forecaster_strictly_loads_frozen_last_layer_tokens_without_pca(tmp_path, monkeypatch):
    monkeypatch.setattr(encoder, 'vit_base', TinyBackbone)
    path = tmp_path / 'encoder.pth'
    state = TinyBackbone().state_dict()
    state['scale'].add_(.125)
    torch.save(state, path)
    model = tiny_forecaster(path).train()
    assert model.predictor.training and not model.encoder.training
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.predictor.parameters())
    assert model.feature_dim == 768
    images = torch.randn(3, 3, 28, 42, requires_grad=True)
    tokens = model.encode_latents(images)
    expected = (images.mean((1, 2, 3))[:, None, None] + torch.arange(6)[None, :, None]
                + state['scale'][None, None, :])
    torch.testing.assert_close(tokens, expected)
    assert not tokens.requires_grad
    assert model.encoder.backbone.seen == [dict(batch=1, layers=[11], reshape=False,
                                      training=False, grad_enabled=False)] * 3
    history = torch.randn(2, 4, 3, 28, 42, requires_grad=True)
    predictions = model(history, **regular_timing().forecast_kwargs())
    assert predictions.shape == (2, 6, 768)
    encoded = model.encode_latents(history.flatten(0, 1)).reshape(2, 4, 6, 768)
    torch.testing.assert_close(predictions, model.predictor(encoded, **regular_timing().forecast_kwargs()))
    predictions.square().mean().backward()
    assert history.grad is None
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.predictor.parameters())


@pytest.mark.parametrize('corruption', ['missing', 'unexpected', 'wrong_shape', 'nonfinite', 'wrapped'])
def test_encoder_checkpoint_rejects_incompatible_or_corrupt_weights(tmp_path, monkeypatch, corruption):
    monkeypatch.setattr(encoder, 'vit_base', TinyBackbone)
    state = TinyBackbone().state_dict()
    if corruption == 'missing':
        del state['scale']
    elif corruption == 'unexpected':
        state['unused'] = torch.ones(1)
    elif corruption == 'wrong_shape':
        state['scale'] = torch.ones(767)
    elif corruption == 'nonfinite':
        state['scale'][0] = float('nan')
    else:
        state = {'state_dict': state}
    path = tmp_path / 'invalid.pth'
    torch.save(state, path)
    with pytest.raises((ValueError, RuntimeError)):
        tiny_forecaster(path)


def test_v31_defaults_and_optional_event_layer_selection():
    assert PREDICTOR_DEFAULTS['depth'] == 8
    assert PREDICTOR_DEFAULTS['hidden_dim'] == 768
    assert PREDICTOR_DEFAULTS['heads'] == 12
    predictor = small_predictor(event_enabled=True, depth=4, event_layers=[2, 4])
    assert [b.event_attention is not None for b in predictor.blocks] == [False, True, False, True]
    assert predictor.architecture['event_layers'] == [2, 4]
    assert isinstance(predictor.context_proj[-1], nn.LayerNorm)
    assert all(block.rgb_attention.query_norm is not block.rgb_attention.context_norm
               for block in predictor.blocks)
    with pytest.raises(ValueError, match='event_layers'):
        small_predictor(event_enabled=True, event_layers=[0])


@pytest.mark.parametrize('target_index', [0, 1, 2])
def test_parallel_predictions_exclude_future_rgb_and_event_values_and_gradients(target_index):
    torch.manual_seed(61)
    predictor = small_predictor(event_enabled=True).eval()
    history = torch.randn(2, 4, 6, 12, requires_grad=True)
    events = torch.randn(2, 4, 6, 12, requires_grad=True)
    original = predictor.predict_sequence(history, event_context=events, **regular_timing().sequence_kwargs())
    for modality in ('rgb', 'event'):
        altered_history, altered_events = history.detach().clone(), events.detach().clone()
        altered = altered_history if modality == 'rgb' else altered_events
        altered[:, target_index + 1:] = 30 * torch.randn_like(altered[:, target_index + 1:])
        changed = predictor.predict_sequence(altered_history, event_context=altered_events,
                                             **regular_timing().sequence_kwargs())
        torch.testing.assert_close(changed[:, target_index], original[:, target_index], rtol=0, atol=0)
        assert not torch.allclose(changed[:, -1], original[:, -1])
    original[:, target_index].square().mean().backward()
    for value in (history, events):
        assert value.grad[:, :target_index + 1].abs().sum() > 0
        assert torch.count_nonzero(value.grad[:, target_index + 1:]) == 0
    # A nonzero gate also reaches the event branch on the very first backward.
    for block in predictor.blocks:
        assert block.gamma.grad.abs() > 0
        assert block.event_attention.kv.weight.grad.abs().sum() > 0


def test_event_mask_uses_observation_cutoff_even_when_event_precedes_target_time():
    torch.manual_seed(67)
    predictor = small_predictor(event_enabled=True).eval()
    history = torch.randn(1, 2, 6, 12)
    events = torch.randn(1, 2, 6, 12, requires_grad=True)
    kwargs = dict(context_times=torch.tensor([0., .05], dtype=torch.float64),
                  query_times=torch.tensor([.05, .1], dtype=torch.float64),
                  event_times=torch.tensor([0., .025], dtype=torch.float64))
    predicted = predictor.predict_at(history, event_context=events, **kwargs)
    changed = events.detach().clone()
    changed[:, 1] = 30 * torch.randn_like(changed[:, 1])
    altered = predictor.predict_at(history, event_context=changed, **kwargs)
    torch.testing.assert_close(predicted[:, 0], altered[:, 0], rtol=0, atol=0)
    assert not torch.allclose(predicted[:, 1], altered[:, 1])
    predicted[:, 0].square().mean().backward()
    assert events.grad[:, 0].abs().sum() > 0
    assert torch.count_nonzero(events.grad[:, 1]) == 0


def test_explicit_cutoff_masks_rgb_and_event_independently_of_future_horizon():
    torch.manual_seed(71)
    predictor = small_predictor(event_enabled=True).eval()
    rgb, events = torch.randn(1, 4, 6, 12), torch.randn(1, 4, 6, 12)
    times = torch.tensor([0., .05, .10, .15], dtype=torch.float64)
    kwargs = dict(context_times=times, query_times=times.new_tensor([.2, .3]),
                  observation_times=times.new_tensor([.05, .05]), event_times=times)
    before = predictor.predict_at(rgb, event_context=events, **kwargs)
    rgb[:, 2:] += 50 * torch.randn_like(rgb[:, 2:])
    events[:, 2:] += 50 * torch.randn_like(events[:, 2:])
    after = predictor.predict_at(rgb, event_context=events, **kwargs)
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert not torch.allclose(before[:, 0], before[:, 1])


def test_gamma_zero_and_off_exactly_reproduce_shared_rgb_baseline():
    torch.manual_seed(73)
    rgb = small_predictor().eval()
    event = small_predictor(event_enabled=True, gate_init=0).eval()
    missing, unexpected = event.load_state_dict(rgb.state_dict(), strict=False)
    assert not unexpected
    assert all('event_' in name or 'gamma' in name for name in missing)
    history, events = torch.randn(2, 4, 6, 12), torch.randn(2, 4, 6, 12)
    expected = rgb.predict_sequence(history, **regular_timing().sequence_kwargs())
    actual = event.predict_sequence(history, event_context=events, **regular_timing().sequence_kwargs())
    off = event.predict_sequence(history, event_off=True, **regular_timing().sequence_kwargs())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(off, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    assert any(block.gamma.grad.abs() > 0 for block in event.blocks)
    assert all(torch.count_nonzero(block.event_attention.kv.weight.grad) == 0
               for block in event.blocks)


@pytest.mark.parametrize('activation_checkpointing', [False, True])
def test_event_chunking_and_checkpointing_preserve_outputs_and_parameter_gradients(activation_checkpointing):
    torch.manual_seed(79)
    whole = small_predictor(event_enabled=True).train()
    chunked = deepcopy(whole)
    chunked.activation_checkpointing = activation_checkpointing
    chunked.query_chunk_size = 5
    history = torch.randn(2, 4, 6, 12, requires_grad=True)
    events = torch.randn(2, 4, 6, 12, requires_grad=True)
    other_history = history.detach().clone().requires_grad_(True)
    other_events = events.detach().clone().requires_grad_(True)
    expected = whole.predict_sequence(history, event_context=events, **regular_timing().sequence_kwargs())
    actual = chunked.predict_sequence(other_history, event_context=other_events,
                                     **regular_timing().sequence_kwargs())
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(other_history.grad, history.grad, rtol=2e-5, atol=2e-7)
    torch.testing.assert_close(other_events.grad, events.grad, rtol=2e-5, atol=2e-7)
    assert_parameter_gradients_match(whole, chunked)


def test_spatial_attention_mixes_patches_but_isolates_each_target_frame():
    torch.manual_seed(83)
    predictor = small_predictor().eval()
    attention = predictor.blocks[0].spatial_attention
    coordinates = predictor._coordinates(torch.zeros(1))
    queries = torch.randn(2, 12, 64, requires_grad=True)
    spatial_rope = rope_3d_tables(coordinates)
    original = attention(queries, spatial_rope, 5)
    changed = queries.detach().clone()
    changed[:, 7] += 20 * torch.randn_like(changed[:, 7])
    altered = attention(changed, spatial_rope, 5)
    torch.testing.assert_close(original[:, :6], altered[:, :6], rtol=0, atol=0)
    assert not torch.allclose(original[:, 6], altered[:, 6])
    original[:, 0].square().sum().backward()
    assert queries.grad[:, 1:6].abs().sum() > 0
    assert torch.count_nonzero(queries.grad[:, 6:]) == 0


def test_parallel_targets_equal_individually_predicted_targets():
    torch.manual_seed(89)
    predictor = small_predictor(event_enabled=True).eval()
    history, events = torch.randn(2, 4, 6, 12), torch.randn(2, 4, 6, 12)
    timing = regular_timing().sequence_kwargs()
    original = predictor.predict_sequence(history, event_context=events, **timing)
    times = timing['context_times']
    for index in range(4):
        query = timing['query_times'][index:index + 1]
        individual = predictor(history[:, :index + 1], context_times=times[:index + 1],
                               query_times=query, event_context=events[:, :index + 1],
                               event_times=times[:index + 1])
        torch.testing.assert_close(individual, original[:, index], rtol=1e-5, atol=2e-6)


def test_event_rollout_keeps_initial_events_and_absolute_times_fixed():
    torch.manual_seed(97)
    predictor = small_predictor(event_enabled=True).eval()
    history = torch.randn(2, 4, 6, 12, requires_grad=True)
    events = torch.randn(2, 4, 6, 12, requires_grad=True)
    timing = regular_timing().forecast_kwargs(horizon_steps=3)
    original_times = timing['context_times']
    actual = predictor.rollout(history, event_context=events, **timing)
    context, times, expected = history, original_times, []
    for target in timing['query_times'].split(1):
        prediction = predictor(context, context_times=times, query_times=target,
                               event_context=events, event_times=original_times)
        expected.append(prediction)
        context = torch.cat((context[:, 1:], prediction[:, None]), dim=1)
        times = torch.cat((times[1:], target))
    torch.testing.assert_close(actual, torch.stack(expected, dim=1), rtol=0, atol=0)
    actual[:, -1].square().mean().backward()
    assert history.grad.abs().sum() > 0 and events.grad.abs().sum() > 0


def test_compact_event_grid_uses_rgb_coordinate_system_and_receives_gradients():
    predictor = small_predictor(grid_size=(4, 6), event_enabled=True).eval()
    rgb = torch.randn(1, 4, 24, 12)
    events = torch.randn(1, 4, 6, 12, requires_grad=True)
    coordinates = predictor._coordinates(torch.zeros(1), grid_size=(2, 3))
    torch.testing.assert_close(coordinates[:, 1], torch.tensor([.5, .5, .5, 2.5, 2.5, 2.5]))
    torch.testing.assert_close(coordinates[:, 2], torch.tensor([.5, 2.5, 4.5, .5, 2.5, 4.5]))
    result = predictor.predict_sequence(rgb, event_context=events, event_grid_size=(2, 3),
                                        **regular_timing().sequence_kwargs())
    assert result.shape == (1, 4, 24, 12)
    result.square().mean().backward()
    assert events.grad.abs().sum() > 0


def test_no_available_events_has_zero_contribution_including_projection_bias():
    predictor = small_predictor(event_enabled=True).eval()
    rgb, events = torch.randn(1, 1, 6, 12), torch.randn(1, 1, 6, 12)
    timing = regular_timing(frames=1).forecast_kwargs()
    original = predictor(rgb, event_off=True, **timing)
    # Event ends after the RGB observation cutoff but before target time.
    timing['event_times'] = torch.tensor([.025], dtype=torch.float64)
    actual = predictor(rgb, event_context=events, **timing)
    torch.testing.assert_close(actual, original, rtol=0, atol=0)


@pytest.mark.parametrize('chunk_size', [0, 5])
def test_visible_prefix_attention_matches_full_mask_with_nonmonotone_cutoffs(chunk_size):
    torch.manual_seed(103)
    prefix_attention = CausalCrossAttention(64, 1, 0.).train()
    masked_attention = deepcopy(prefix_attention)
    query = torch.randn(2, 12, 64, requires_grad=True)
    context = torch.randn(2, 10, 64, requires_grad=True)
    query_coordinates = torch.randn(12, 3, requires_grad=True)
    context_coordinates = torch.randn(10, 3, requires_grad=True)
    copies = [value.detach().clone().requires_grad_(True)
              for value in (query, context, query_coordinates, context_coordinates)]
    other_query, other_context, other_q_coordinates, other_k_coordinates = copies
    times = torch.tensor([0., .025, .07, .11, .2], dtype=torch.float64)
    cutoffs = torch.tensor([.12, -.01, .025, .2], dtype=torch.float64)
    prefixes = DINOWorldPredictor._prefixes(times, cutoffs, 3, 2)
    assert prefixes == ((0, 3, 8), (3, 6, 0), (6, 9, 4), (9, 12, 10))
    actual = prefix_attention(query, context, rope_3d_tables(query_coordinates),
                              rope_3d_tables(context_coordinates), prefixes, chunk_size)
    q = masked_attention.q(masked_attention.query_norm(other_query)).reshape(2, 12, 1, 64).transpose(1, 2)
    k, v = masked_attention.kv(masked_attention.context_norm(other_context)).reshape(2, 10, 2, 1, 64).unbind(2)
    q = apply_rope_3d(q, rope_3d_tables(other_q_coordinates))
    k = apply_rope_3d(k.transpose(1, 2), rope_3d_tables(other_k_coordinates))
    allowed = times.repeat_interleave(2)[None] <= cutoffs.repeat_interleave(3)[:, None]
    value = torch.nn.functional.scaled_dot_product_attention(q, k, v.transpose(1, 2), attn_mask=allowed)
    expected = masked_attention.proj(value.transpose(1, 2).reshape(2, 12, 64)) * allowed.any(-1)[None, :, None]
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
    actual.square().mean().backward()
    expected.square().mean().backward()
    assert_parameter_gradients_match(prefix_attention, masked_attention)
    for left, right in zip((query, context, query_coordinates, context_coordinates), copies, strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=3e-5, atol=2e-6)


def test_empty_event_prefix_keeps_zero_gradients_connected_for_ddp():
    predictor = small_predictor(event_enabled=True).train()
    history = torch.randn(1, 1, 6, 12, requires_grad=True)
    events = torch.randn(1, 1, 6, 12, requires_grad=True)
    timing = regular_timing(frames=1).forecast_kwargs()
    timing['event_times'] = torch.tensor([.025], dtype=torch.float64)
    predictor(history, event_context=events, **timing).square().mean().backward()
    assert events.grad is not None and torch.count_nonzero(events.grad) == 0
    for name, parameter in predictor.named_parameters():
        if 'event_' in name or 'gamma' in name:
            assert parameter.grad is not None, name
            assert torch.count_nonzero(parameter.grad) == 0, name


def test_dynamic_timestamp_gradients_survive_shared_rope_and_checkpointing():
    torch.manual_seed(107)
    plain = small_predictor(event_enabled=True).train()
    recompute = deepcopy(plain)
    recompute.activation_checkpointing = True
    history, events = torch.randn(1, 4, 6, 12), torch.randn(1, 4, 6, 12)
    times = [torch.tensor(values, dtype=torch.float64, requires_grad=True)
             for values in ([0., .05, .10, .15], [.2, .3], [0., .04, .08, .14])]
    other_times = [value.detach().clone().requires_grad_(True) for value in times]
    cutoff = torch.tensor([.05, .12], dtype=torch.float64, requires_grad=True)
    expected = plain.predict_at(history, times[0], times[1], observation_times=cutoff,
                                event_context=events, event_times=times[2])
    actual = recompute.predict_at(history, other_times[0], other_times[1], observation_times=cutoff,
                                  event_context=events, event_times=other_times[2])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for left, right in zip(times, other_times, strict=True):
        assert left.grad is not None and torch.isfinite(left.grad).all()
        assert left.grad.abs().sum() > 0
        torch.testing.assert_close(left.grad, right.grad, rtol=2e-5, atol=2e-6)
    # Visibility is a discrete decision, as with the original Boolean mask.
    assert cutoff.grad is None


def test_event_checkpointing_preserves_bfloat16_outputs_and_gradients():
    torch.manual_seed(101)
    plain = small_predictor(event_enabled=True).train()
    recompute = deepcopy(plain)
    recompute.activation_checkpointing = True
    history, events = torch.randn(1, 4, 6, 12), torch.randn(1, 4, 6, 12)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        expected = plain.predict_sequence(history, event_context=events, **regular_timing().sequence_kwargs())
        actual = recompute.predict_sequence(history, event_context=events, **regular_timing().sequence_kwargs())
    assert expected.dtype == actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().mean().backward()
    actual.square().mean().backward()
    assert_parameter_gradients_match(plain, recompute)


class TinyGridBackbone(TinyBackbone):
    def get_intermediate_layers(self, images, n, reshape):
        spatial = (images.shape[-2] // 14) * (images.shape[-1] // 14)
        offsets = torch.arange(spatial, device=images.device, dtype=images.dtype)
        return ((images.mean((1, 2, 3))[:, None, None]
                 + offsets[None, :, None] + self.scale[None, None, :]),)


def test_forecaster_integrates_rvt_and_only_freezes_rgb_encoder(tmp_path, monkeypatch):
    monkeypatch.setattr(encoder, 'vit_base', TinyGridBackbone)
    path = tmp_path / 'encoder.pth'
    torch.save(TinyGridBackbone().state_dict(), path)
    model = DINOWorldForecaster(
        path, image_size=(28, 56), history_length=2,
        predictor_options=dict(hidden_dim=64, heads=1, depth=2, mlp_ratio=2., query_chunk_size=0),
        event_options=dict(enabled=True, in_channels=2, input_size=(32, 64),
                           channels=(4, 8, 16, 32), heads=(1, 1, 1, 1),
                           num_blocks=(1, 1, 1, 1), fpn_channels=8, partition_size=(2, 2))).train()
    rgb = torch.randn(2, 2, 3, 28, 56, requires_grad=True)
    events = torch.rand(2, 2, 2, 32, 64, requires_grad=True)
    output = model(rgb, events, **regular_timing(frames=2).forecast_kwargs())
    assert output.shape == (2, 8, 768)
    assert model.architecture['model'] == 'dino-world-v31'
    assert model.architecture['event_encoder']['channels'] == (4, 8, 16, 32)
    assert model.event_encoder.training and not model.encoder.training
    output.square().mean().backward()
    assert rgb.grad is None
    assert events.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.event_encoder.parameters())
    with torch.no_grad():
        off = model(rgb, event_off=True, **regular_timing(frames=2).forecast_kwargs())
        encoded = model.encode_latents(rgb.flatten(0, 1)).reshape(2, 2, 8, 768)
        torch.testing.assert_close(off, model.predictor(encoded, event_off=True,
                                                        **regular_timing(frames=2).forecast_kwargs()))
