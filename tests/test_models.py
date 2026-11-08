"""CPU contracts for the isolated pretrained model components."""

import importlib
from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from foresight.models.criterion import DetectionCriterion
from foresight.models.detector import FutureDetector, load_detector, postprocess
from foresight.models.forecaster import (
    DINOForecaster, DINOFeaturePredictor, PCAProjection, internal_state_dict, read_weights,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def pca_state():
    torch.manual_seed(17)
    basis, _ = torch.linalg.qr(torch.randn(6, 6))
    return dict(mean=torch.randn(6), std=torch.rand(6) + 0.3,
                pca_mean=torch.randn(6), pca_components=basis[:3])


def test_pca_keeps_official_affine_coordinates_and_gradients():
    state = pca_state()
    pca = PCAProjection(state, feature_dim=6, pca_dim=3)
    features = torch.randn(2, 4, 6, requires_grad=True)
    projected = pca.project(features)
    expected = (((features - state['mean']) / state['std']) - state['pca_mean']) @ state['pca_components'].T
    torch.testing.assert_close(projected, expected, rtol=0, atol=0)
    torch.testing.assert_close(pca.project(pca.inverse(projected)), projected, rtol=1e-5, atol=1e-6)
    pca.inverse(projected).square().mean().backward()
    assert features.grad is not None and features.grad.abs().sum() > 0
    assert list(pca.parameters()) == []


def test_pca_rejects_incompatible_or_corrupt_buffers():
    for replacement in (torch.zeros(6), torch.full((6,), float('nan'))):
        with pytest.raises(ValueError):
            PCAProjection(dict(pca_state(), std=replacement), 6, 3)
    with pytest.raises(ValueError):
        PCAProjection(dict(pca_state(), pca_components=torch.zeros(2, 6)), 6, 3)


def small_predictor():
    predictor = DINOFeaturePredictor(grid_size=(2, 3), pca_dim=3, hidden_dim=12,
                                     depth=2, heads=3, dropout=0)
    nn.init.normal_(predictor.maskvit.fc_out.weight, std=.1)
    return predictor


def test_predictor_is_trainable_and_uses_only_four_observed_maps():
    torch.manual_seed(9)
    predictor = small_predictor().train()
    history = torch.randn(2, 4, 6, 3, requires_grad=True)
    result = predictor(history)
    assert result.shape == (2, 6, 3) and predictor.training
    result.square().mean().backward()
    assert history.grad.abs().sum() > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in predictor.parameters())
    with pytest.raises(ValueError):
        predictor(torch.randn(2, 5, 6, 3))


def test_bf16_predictor_keeps_fp32_future_token_promotion():
    predictor = small_predictor().eval()
    history = torch.randn(1, 4, 6, 3)
    captured = []
    handle = predictor.maskvit.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = predictor(history)
        embedded = predictor.embed(history.reshape(1, 4, 2, 3, 3))
        expected_tokens = torch.cat((embedded, predictor.replace_vector.expand(1, 1, 2, 3, -1)), dim=1)
    handle.remove()
    assert embedded.dtype == torch.bfloat16 and captured[0].dtype == torch.float32
    torch.testing.assert_close(captured[0], expected_tokens, rtol=0, atol=0)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        expected, = predictor.maskvit(expected_tokens)
    torch.testing.assert_close(actual, expected.reshape(1, 5, 6, 3)[:, -1], rtol=0, atol=0)


def test_activation_checkpointing_preserves_outputs_and_all_gradients():
    torch.manual_seed(29)
    plain = small_predictor().eval()
    recompute = deepcopy(plain)
    recompute.activation_checkpointing = True
    # eval() disables dropout without disabling autograd in Predictor training.
    inputs = torch.randn(2, 4, 6, 3, requires_grad=True)
    other_inputs = inputs.detach().clone().requires_grad_(True)
    expected, actual = plain(inputs), recompute(other_inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(other_inputs.grad, inputs.grad, rtol=1e-6, atol=1e-7)
    for (left_name, left), (right_name, right) in zip(plain.named_parameters(), recompute.named_parameters(), strict=True):
        assert left_name == right_name and left.grad is not None and right.grad is not None
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-6, atol=1e-7)
    with torch.inference_mode():
        torch.testing.assert_close(recompute(inputs), plain(inputs), rtol=0, atol=0)


def test_forecaster_encodes_chunk1_and_returns_latent_or_dense():
    class TinyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = []

        def get_intermediate_layers(self, images, n, reshape):
            self.seen.append(len(images))
            features = images.mean((2, 3))[:, None].expand(-1, 6, -1)
            return (features, features * 2)

    class RecordingPCA(PCAProjection):
        def __init__(self):
            super().__init__(pca_state(), 6, 3)
            self.seen = []

        def project(self, features):
            self.seen.append(len(features))
            return super().project(features)

    model = DINOForecaster.__new__(DINOForecaster)
    nn.Module.__init__(model)
    model.image_size, model.grid_size = (4, 6), (2, 3)
    model.feature_dim, model.pca_dim, model.history_length = 6, 3, 4
    model.d_layers, model.encoder_chunk_size = [0, 1], 1
    model.encoder, model.pca, model.predictor = TinyEncoder(), RecordingPCA(), small_predictor()
    history = torch.randn(2, 4, 3, 4, 6, requires_grad=True)
    latent = model.forecast_latents(history)
    dense = model.forecast(history)
    torch.testing.assert_close(dense, model.pca.inverse(latent), rtol=0, atol=0)
    assert dense.shape == (2, 6, 6) and model.encoder.seen == [1] * 16
    assert model.pca.seen == [1] * 16
    dense.square().mean().backward()
    assert history.grad.abs().sum() > 0


def test_detector_loss_and_postprocess_support_empty_targets():
    torch.manual_seed(11)
    detector = FutureDetector(feature_dim=12, grid_size=(2, 3), hidden_dim=32,
                              num_queries=8, num_levels=2, encoder_layers=1,
                              decoder_layers=2, heads=4, points=2, mlp_dim=64, dropout=0)
    inputs = torch.randn(2, 6, 12, requires_grad=True)
    outputs = detector(inputs)
    targets = [dict(labels=torch.tensor([2]), boxes=torch.tensor([[.5, .5, .3, .2]])),
               dict(labels=torch.empty(0, dtype=torch.long), boxes=torch.empty(0, 4))]
    loss = DetectionCriterion()(outputs, targets)['loss']
    assert torch.isfinite(loss)
    loss.backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    predictions = postprocess(outputs, max_dets=10)
    assert len(predictions) == 2 and all(len(item['scores']) <= 10 for item in predictions)
    assert all((item['boxes'][:, 0::2] <= 639).all() and (item['boxes'][:, 1::2] <= 479).all()
               for item in predictions)


def test_checkpoint_mapping_is_bijective_and_rejects_unknown_modules():
    state = {key: torch.tensor([i]) for i, key in enumerate(
        ['dino_v2.cls_token', 'maskvit.weight', 'embed.weight', 'replace_vector', 'mean', 'std', 'pca_mean', 'pca_components'])}
    mapped = internal_state_dict(state)
    assert len(mapped) == len(state)
    assert mapped['encoder.cls_token'] is state['dino_v2.cls_token']
    assert mapped['pca.mean'] is state['mean']
    assert mapped['predictor.maskvit.weight'] is state['maskvit.weight']
    with pytest.raises(ValueError):
        internal_state_dict({'unexpected.weight': torch.zeros(1)})


def test_wrong_export_format_fails_before_model_construction(tmp_path):
    path = tmp_path / 'wrong.pt'
    torch.save(dict(format='unknown', state_dict={'weight': torch.zeros(1)}), path)
    with pytest.raises(ValueError):
        DINOForecaster(path)
    with pytest.raises(ValueError):
        load_detector(path)


def test_vendor_uses_private_namespace_and_has_no_hub_loader():
    module = importlib.import_module('foresight.vendor.dinov2.models.vision_transformer')
    assert module.PatchEmbed.__module__.startswith('foresight.vendor.')
    root = Path(importlib.import_module('foresight.models.forecaster').__file__).parent
    assert 'torch.hub.' not in (root / 'forecaster.py').read_text()


@pytest.mark.skipif(not os.environ.get('FORESIGHT_ASSET_DIR'), reason='Explicit real-asset CPU check is opt-in')
def test_real_export_shapes_and_trainability():
    directory = Path(os.environ['FORESIGHT_ASSET_DIR'])
    model = DINOForecaster(directory / 'dinof.pt')
    detector = load_detector(directory / 'detector.pt')
    assert len(model.state_dict()) == 404 and len(detector.state_dict()) == 296
    assert model.encoder.pos_embed.shape == (1, 1370, 768)
    assert model.predictor.replace_vector.shape == (1, 1, 1, 1, 1152)
    assert all(parameter.requires_grad for parameter in model.predictor.parameters())
    assert model.training and detector.training
    assert not torch.cuda.is_initialized()
