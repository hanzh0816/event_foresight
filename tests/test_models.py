"""CPU contracts for detection readout and private encoder dependencies."""

import importlib

import pytest
import torch

from foresight.models.detection.criterion import DetectionCriterion
from foresight.models.detection.detector import FutureDetector, postprocess


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


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


def test_encoder_vendor_uses_private_namespace():
    module = importlib.import_module('foresight.vendor.dinov2.models.vision_transformer')
    assert module.PatchEmbed.__module__.startswith('foresight.vendor.dinov2.')
