"""Small CPU contracts for the independently implemented DINO-world model."""

from copy import deepcopy
import math
from pathlib import Path

import pytest
import torch
from torch import nn

import foresight.models.dino_world as dino_world
from foresight.models.dino_world import DINOWorldForecaster, DINOWorldPredictor, apply_rope_3d


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def small_predictor(**options):
    settings = dict(grid_size=(2, 3), history_length=4, feature_dim=12,
                    hidden_dim=64, depth=2, heads=1, mlp_ratio=2., dropout=0.,
                    frame_interval_s=.05, query_chunk_size=0)
    settings.update(options)
    return DINOWorldPredictor(**settings)


def assert_parameter_gradients_match(left, right):
    for (left_name, left_value), (right_name, right_value) in zip(
            left.named_parameters(), right.named_parameters(), strict=True):
        assert left_name == right_name
        assert left_value.grad is not None and right_value.grad is not None, left_name
        assert torch.isfinite(left_value.grad).all(), left_name
        torch.testing.assert_close(right_value.grad, left_value.grad, rtol=2e-5, atol=2e-7)


def test_rope_matches_paper_three_axis_rotation_and_preserves_last_four_dimensions():
    torch.manual_seed(7)
    value = torch.randn(2, 3, 6, 64, requires_grad=True)
    coordinates = torch.tensor([[0., 0., 0.], [.05, 0., 1.], [.1, 0., 2.],
                                [.2, 1., 0.], [.4, 1., 1.], [.8, 1., 2.]])
    actual = apply_rope_3d(value, coordinates)
    frequencies = 2 * math.pi / 10 ** torch.linspace(-2, 2, 10)
    expected = value.clone()
    for axis in range(3):
        angle = coordinates[:, axis, None] * frequencies[None, :]
        start = axis * 20
        even, odd = value[..., start:start + 20:2], value[..., start + 1:start + 20:2]
        expected[..., start:start + 20:2] = even * angle.cos() - odd * angle.sin()
        expected[..., start + 1:start + 20:2] = even * angle.sin() + odd * angle.cos()
    # FP32 logspace and exponentiating linspace round differently before the
    # highest-frequency rotations; the independent formulas agree to 1e-4.
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(actual[..., 60:], value[..., 60:], rtol=0, atol=0)
    torch.testing.assert_close(actual.square().sum(-1), value.square().sum(-1))
    actual.square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()


def test_predictor_has_gradients_and_default_query_is_next_frame():
    torch.manual_seed(13)
    predictor = small_predictor().train()
    history = torch.randn(2, 4, 6, 12, requires_grad=True)
    actual = predictor(history)
    assert actual.shape == (2, 6, 12)
    explicit = predictor.predict_at(history, torch.tensor([0., .05, .10, .15]), torch.tensor([.20]))
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
    original = predictor.predict_sequence(history)
    changed = predictor.predict_sequence(altered)
    assert original.shape == (2, 4, 6, 12)
    # Each training target excludes the RGB at its own time and all later RGB.
    torch.testing.assert_close(changed[:, target_index], original[:, target_index], rtol=0, atol=0)
    assert not torch.allclose(changed[:, -1], original[:, -1])
    differentiable_history = history.clone().requires_grad_(True)
    predictor.predict_sequence(differentiable_history)[:, target_index].square().mean().backward()
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
    actual = predictor.rollout(history, steps=3)
    assert actual.shape == (2, 3, 6, 12)
    context = history
    expected = []
    for _ in range(3):
        prediction = predictor(context)
        expected.append(prediction)
        context = torch.cat((context[:, 1:], prediction[:, None]), dim=1)
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
        expected = plain.predict_sequence(history)
        actual = recompute.predict_sequence(other_history)
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
    counts = {block.kv: 0 for block in predictor.blocks}

    def record_projection(module, inputs, output):
        counts[module] += 1

    handles = [module.register_forward_hook(record_projection) for module in counts]
    try:
        with torch.no_grad():
            predictor.predict_sequence(torch.randn(2, 4, 6, 12))
    finally:
        for handle in handles:
            handle.remove()
    assert list(counts.values()) == [1] * len(predictor.blocks)


@pytest.mark.parametrize('options', [dict(hidden_dim=32), dict(hidden_dim=64, heads=2)])
def test_predictor_requires_paper_64_dimensional_attention_heads(options):
    with pytest.raises(ValueError):
        small_predictor(**options)


@pytest.mark.parametrize('context_times,query_times', [
    ([0., .05, .10, .15], [0.]),
    ([0., .05, .10, .15], [-.01]),
    ([0., .05, .05, .15], [.2]),
    ([0., .10, .05, .15], [.2]),
    ([0., .05, .10], [.2]),
    ([0., .05, .10, float('nan')], [.2]),
    ([0., .05, .10, .15], [float('inf')]),
])
def test_invalid_or_causally_empty_timestamps_are_rejected(context_times, query_times):
    predictor = small_predictor()
    with pytest.raises(ValueError):
        predictor.predict_at(torch.randn(1, 4, 6, 12), torch.tensor(context_times), torch.tensor(query_times))


def test_model_source_does_not_download_weights_or_use_hub():
    source = Path(dino_world.__file__).read_text()
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
    monkeypatch.setattr(dino_world, 'vit_base', TinyBackbone)
    path = tmp_path / 'encoder.pth'
    state = TinyBackbone().state_dict()
    state['scale'].add_(.125)
    torch.save(state, path)
    model = tiny_forecaster(path).train()
    assert model.predictor.training and not model.encoder.training
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.predictor.parameters())
    assert model.feature_dim == model.latent_dim == 768
    images = torch.randn(3, 3, 28, 42, requires_grad=True)
    tokens = model.encode_dense(images)
    expected = (images.mean((1, 2, 3))[:, None, None] + torch.arange(6)[None, :, None]
                + state['scale'][None, None, :])
    torch.testing.assert_close(tokens, expected)
    assert not tokens.requires_grad
    assert model.encoder.seen == [dict(batch=1, layers=[11], reshape=False,
                                      training=False, grad_enabled=False)] * 3
    history = torch.randn(2, 4, 3, 28, 42, requires_grad=True)
    predictions = model(history)
    assert predictions.shape == (2, 6, 768)
    encoded = model.encode_latents(history.flatten(0, 1)).reshape(2, 4, 6, 768)
    torch.testing.assert_close(predictions, model.predictor(encoded))
    predictions.square().mean().backward()
    assert history.grad is None
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.predictor.parameters())


@pytest.mark.parametrize('corruption', ['missing', 'unexpected', 'wrong_shape', 'nonfinite', 'wrapped'])
def test_encoder_checkpoint_rejects_incompatible_or_corrupt_weights(tmp_path, monkeypatch, corruption):
    monkeypatch.setattr(dino_world, 'vit_base', TinyBackbone)
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
