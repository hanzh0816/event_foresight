"""Shared detector loading keeps predictor weights and feature contracts separate."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from foresight.inference import pipeline
from foresight.models.checkpoints import TRAINING_FORMAT, input_spec
from foresight.models.detection.detector import FutureDetector


class TinyForecaster(nn.Module):
    image_size = (28, 42)
    history_length = 4
    feature_dim = 4
    grid_size = (2, 3)
    feature_space = "tiny-last-layer"
    architecture = {"model": "tiny-world-v31", "history_length": 4,
                    "frame_interval_s": 0.05, "feature_space": feature_space,
                    "feature_dim": feature_dim, "image_size": list(image_size),
                    "predictor": {"depth": 2, "event_enabled": False}}

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.predictor = nn.Linear(2, 2)
        self.event_encoder = None
        self.architecture = deepcopy(type(self).architecture)
        with torch.no_grad():
            self.encoder.weight.fill_(0.25)
            self.encoder.bias.fill_(0.5)
            self.predictor.weight.zero_()
            self.predictor.bias.zero_()


def assert_state_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        assert torch.equal(value, expected[name]), name


@pytest.fixture
def trained_detector(tmp_path, monkeypatch):
    architecture = dict(feature_dim=4, grid_size=[2, 3], hidden_dim=8,
                        num_classes=8, num_queries=3, num_levels=1,
                        encoder_layers=1, decoder_layers=1, heads=2,
                        points=1, mlp_dim=16, dropout=0.0)
    config = dict(model=dict(encoder_checkpoint=str(tmp_path / "unused_dinov2.pth"),
                             predictor_checkpoint=str(tmp_path / "predictor.ckpt")),
                  data=dict(image_size=[28, 42], context_frames=4),
                  runtime=dict(encoder_chunk_size=1))
    source = pipeline.ForecastDetector(TinyForecaster(), FutureDetector(**architecture))
    payload = dict(foresight_format=TRAINING_FORMAT, stage="detector",
                   detector_architecture=architecture,
                   encoder_architecture={"model": "tiny-encoder"},
                   input_spec=input_spec(config, source.forecaster),
                   state_dict={**{f"encoder.{name}": value.clone()
                                   for name, value in source.forecaster.encoder.state_dict().items()},
                               **{f"detector.{name}": value.clone()
                                   for name, value in source.detector.state_dict().items()}})
    checkpoint = tmp_path / "detector.ckpt"
    torch.save(payload, checkpoint)
    predictor = {name: value.clone() for name, value in source.state_dict().items()
                 if name.startswith("forecaster.")}
    predictor["forecaster.predictor.weight"].fill_(7.0)
    predictor["forecaster.predictor.bias"].fill_(-3.0)
    predictor_payload = dict(foresight_format=TRAINING_FORMAT, stage="predictor", state_dict=predictor,
                             forecaster_architecture=deepcopy(source.forecaster.architecture))
    torch.save(predictor_payload, config["model"]["predictor_checkpoint"])
    created = []

    def create_forecaster(checkpoint, *, encoder_chunk_size, **kwargs):
        assert encoder_chunk_size == 1
        forecaster = TinyForecaster()
        created.append(forecaster)
        return forecaster

    monkeypatch.setattr(pipeline, "DINOWorldForecaster", create_forecaster)
    return SimpleNamespace(config=config, payload=payload, checkpoint=checkpoint,
                           predictor_payload=predictor_payload, directory=tmp_path, created=created)


def test_pipeline_assembles_shared_head_and_separate_trained_predictor(trained_detector):
    case = trained_detector
    restored = pipeline.load_pipeline(case.config, case.checkpoint)
    expected = {name: value for name, value in case.payload["state_dict"].items()
                if name.startswith("detector.")}
    expected.update(case.predictor_payload["state_dict"])
    assert_state_equal(restored.state_dict(), expected)
    assert restored.detector.grid_hw == (2, 3)
    assert restored.detector.query_embedding.weight.shape == (3, 16)
    assert all(value.device.type == "cpu" for value in restored.state_dict().values())
    assert all(not module.training for module in restored.modules())
    assert all(not parameter.requires_grad for parameter in restored.parameters())


def test_persistence_loads_only_shared_head_and_encoder(trained_detector):
    case = trained_detector
    case.config["model"]["predictor_checkpoint"] = None
    case.config["model"]["detector_checkpoint"] = str(case.checkpoint)
    restored = pipeline.load_pipeline(case.config, persistence=True)
    expected = {name.removeprefix("detector."): value for name, value in case.payload["state_dict"].items()
                if name.startswith("detector.")}
    assert_state_equal(restored.detector.state_dict(), expected)
    assert all(torch.count_nonzero(value) == 0 for value in restored.forecaster.predictor.parameters())


@pytest.mark.parametrize("corruption", ["missing", "unexpected"])
def test_shared_detector_requires_strict_head_weights(trained_detector, corruption):
    case = trained_detector
    if corruption == "missing":
        del case.payload["state_dict"]["detector.class_heads.0.bias"]
        message = "Missing key"
    else:
        case.payload["state_dict"]["detector.unexpected"] = torch.zeros(1)
        message = "Unexpected key"
    torch.save(case.payload, case.checkpoint)
    with pytest.raises(RuntimeError, match=message):
        pipeline.load_pipeline(case.config, case.checkpoint)


@pytest.mark.parametrize("checkpoint_kind", ["detector", "predictor"])
@pytest.mark.parametrize("key", ["weight", "bias"])
def test_shared_head_and_predictor_reject_changed_frozen_encoder(trained_detector, checkpoint_kind, key):
    case = trained_detector
    payload = case.payload if checkpoint_kind == "detector" else case.predictor_payload
    prefix = "encoder" if checkpoint_kind == "detector" else "forecaster.encoder"
    payload["state_dict"][f"{prefix}.{key}"].add_(1.0)
    path = case.checkpoint if checkpoint_kind == "detector" else case.config["model"]["predictor_checkpoint"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="different frozen encoder"):
        pipeline.load_pipeline(case.config, case.checkpoint)
    assert all(torch.count_nonzero(value) == 0 for value in case.created[-1].predictor.parameters())


def test_inference_requires_head_and_predictor_paths_before_constructing_models(trained_detector):
    case = trained_detector
    with pytest.raises(ValueError, match="shared Detector"):
        pipeline.load_pipeline(case.config)
    case.config["model"]["predictor_checkpoint"] = None
    with pytest.raises(ValueError, match="model.predictor_checkpoint"):
        pipeline.load_pipeline(case.config, case.checkpoint)
    assert not case.created


@pytest.mark.parametrize("field,value", [("feature_dim", 3), ("grid_size", [1, 6]), ("num_classes", 9)])
def test_shared_head_rejects_feature_grid_or_class_mismatch(trained_detector, field, value):
    case = trained_detector
    case.payload["detector_architecture"][field] = value
    torch.save(case.payload, case.checkpoint)
    with pytest.raises(ValueError, match="feature space|grid"):
        pipeline.load_pipeline(case.config, case.checkpoint)


@pytest.mark.parametrize("change", ["architecture", "dtype"])
def test_predictor_rejects_semantic_or_encoder_dtype_mismatch(trained_detector, change):
    case = trained_detector
    payload = case.predictor_payload
    if change == "architecture":
        payload["forecaster_architecture"]["frame_interval_s"] = 0.1
    else:
        payload["state_dict"]["forecaster.encoder.weight"] = payload["state_dict"]["forecaster.encoder.weight"].double()
    torch.save(payload, case.config["model"]["predictor_checkpoint"])
    with pytest.raises(ValueError, match="architecture|frozen encoder"):
        pipeline.load_pipeline(case.config, case.checkpoint)


def test_shared_head_checkpoint_contains_no_predictor_or_forecaster(trained_detector):
    case = trained_detector
    assert "forecaster_architecture" not in case.payload
    assert all(name.startswith(("encoder.", "detector.")) for name in case.payload["state_dict"])
    restored = pipeline.load_pipeline(case.config, case.checkpoint)
    assert torch.equal(restored.forecaster.predictor.weight,
                       case.predictor_payload["state_dict"]["forecaster.predictor.weight"])


def test_detector_checkpoint_cannot_supply_prediction_weights(trained_detector):
    case = trained_detector
    case.config["model"]["predictor_checkpoint"] = str(case.checkpoint)
    with pytest.raises(ValueError, match="[Pp]redictor"):
        pipeline.load_pipeline(case.config, case.checkpoint)


@pytest.mark.parametrize('key,value', [('feature_space', 'different-space'),
                                       ('image_size', [42, 28]),
                                       ('preprocessing', 'different-normalization')])
def test_shared_head_rejects_feature_preprocessing_contract_changes(trained_detector, key, value):
    case = trained_detector
    case.payload['input_spec'][key] = value
    torch.save(case.payload, case.checkpoint)
    with pytest.raises(ValueError, match='protocol|feature|preprocessing'):
        pipeline.load_pipeline(case.config, case.checkpoint)
