"""CPU regression tests for full Detector restoration and Predictor overrides."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from foresight import pipeline
from foresight.checkpoints import TRAINING_FORMAT, input_spec
from foresight.models.detector import FutureDetector


class TinyForecaster(nn.Module):
    """Replace only the large asset-backed model, preserving component namespaces."""

    image_size = (28, 42)
    history_length = 4
    feature_dim = 4
    grid_size = (2, 3)
    feature_space = "tiny-last-layer"
    architecture = {"name": "tiny-world", "history_length": 4, "frame_interval_s": 0.05}

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.predictor = nn.Linear(2, 2)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.zero_()


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
    source = pipeline.ForecastDetector(TinyForecaster(), FutureDetector(**architecture))
    with torch.no_grad():
        # Distinguish every saved component from the fresh forecaster. The
        # override guard must compare against restored encoder, not initial assets.
        source.forecaster.encoder.weight.fill_(0.25)
        source.forecaster.encoder.bias.fill_(0.5)
        source.forecaster.predictor.weight.fill_(1.25)
        source.forecaster.predictor.bias.fill_(1.5)
    config = dict(model=dict(encoder_checkpoint=str(tmp_path / "unused_dinov2.pth"),
                             predictor_checkpoint=None),
                  data=dict(image_size=[28, 42], context_frames=4),
                  runtime=dict(encoder_chunk_size=1))
    payload = dict(foresight_format=TRAINING_FORMAT, stage="detector",
                   detector_architecture=architecture,
                   forecaster_architecture=deepcopy(source.forecaster.architecture),
                   input_spec=input_spec(config, source.forecaster),
                   state_dict=deepcopy(source.state_dict()))
    checkpoint = tmp_path / "detector.ckpt"
    torch.save(payload, checkpoint)
    created = []

    def create_forecaster(checkpoint, *, encoder_chunk_size, **kwargs):
        assert encoder_chunk_size == 1
        forecaster = TinyForecaster()
        created.append(forecaster)
        return forecaster

    monkeypatch.setattr(pipeline, "DINOWorldForecaster", create_forecaster)
    return SimpleNamespace(config=config, payload=payload, checkpoint=checkpoint,
                           directory=tmp_path, created=created)


def predictor_override(case):
    state = {name: value.clone() for name, value in case.payload["state_dict"].items()
             if name.startswith("forecaster.")}
    state["forecaster.predictor.weight"].fill_(7.0)
    state["forecaster.predictor.bias"].fill_(-3.0)
    return dict(foresight_format=TRAINING_FORMAT, stage="predictor", state_dict=state,
                forecaster_architecture=deepcopy(case.payload["forecaster_architecture"]))


def test_trained_detector_restores_complete_pipeline(trained_detector):
    case = trained_detector
    restored = pipeline.load_pipeline(case.config, case.checkpoint)
    assert_state_equal(restored.state_dict(), case.payload["state_dict"])
    assert restored.detector.grid_hw == (2, 3)
    assert restored.detector.query_embedding.weight.shape == (3, 16)
    assert all(value.device.type == "cpu" for value in restored.state_dict().values())
    assert all(not module.training for module in restored.modules())
    assert all(not parameter.requires_grad for parameter in restored.parameters())


@pytest.mark.parametrize("corruption", ["missing", "unexpected"])
def test_trained_detector_requires_a_strict_complete_state(trained_detector, corruption):
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


def test_explicit_predictor_overrides_restored_predictor_last(trained_detector):
    case = trained_detector
    override = predictor_override(case)
    path = case.directory / "predictor.ckpt"
    torch.save(override, path)
    case.config["model"]["predictor_checkpoint"] = str(path)
    restored = pipeline.load_pipeline(case.config, case.checkpoint)
    expected = deepcopy(case.payload["state_dict"])
    for name, value in override["state_dict"].items():
        if name.startswith("forecaster.predictor."):
            assert not torch.equal(value, expected[name])
            expected[name] = value
    # Full-state comparison also proves that Encoder and Detector survive
    # the override unchanged, rather than merely checking the new P weights.
    assert_state_equal(restored.state_dict(), expected)
    assert all(not parameter.requires_grad for parameter in restored.parameters())


@pytest.mark.parametrize("component,key", [("encoder", "weight"), ("encoder", "bias")])
def test_predictor_override_rejects_incompatible_frozen_coordinates(trained_detector, component, key):
    case = trained_detector
    override = predictor_override(case)
    override["state_dict"][f"forecaster.{component}.{key}"].add_(1.0)
    path = case.directory / "incompatible_predictor.ckpt"
    torch.save(override, path)
    case.config["model"]["predictor_checkpoint"] = str(path)
    with pytest.raises(ValueError, match=f"different frozen {component}"):
        pipeline.load_pipeline(case.config, case.checkpoint)
    expected = {name.removeprefix("forecaster.predictor."): value
                for name, value in case.payload["state_dict"].items()
                if name.startswith("forecaster.predictor.")}
    # The failed override must not partially replace the trained Detector's P.
    assert_state_equal(case.created[-1].predictor.state_dict(), expected)


def test_inference_requires_trained_world_detector_before_loading_assets(trained_detector):
    case = trained_detector
    with pytest.raises(ValueError, match="trained DINO World Detector"):
        pipeline.load_pipeline(case.config)
    case.payload["foresight_format"] = "event-foresight-training-v1"
    torch.save(case.payload, case.checkpoint)
    with pytest.raises(ValueError, match="DINO World Detector"):
        pipeline.load_pipeline(case.config, case.checkpoint)
    assert not case.created


@pytest.mark.parametrize("field", ["forecaster_architecture", "input_spec", "detector_architecture"])
def test_inference_rejects_model_or_protocol_mismatch(trained_detector, field):
    case = trained_detector
    case.payload[field] = {}
    torch.save(case.payload, case.checkpoint)
    with pytest.raises(ValueError, match="architecture|protocol|contract"):
        pipeline.load_pipeline(case.config, case.checkpoint)


@pytest.mark.parametrize("change", ["architecture", "dtype"])
def test_predictor_override_rejects_semantic_or_encoder_dtype_mismatch(trained_detector, change):
    case = trained_detector
    override = predictor_override(case)
    if change == "architecture":
        override["forecaster_architecture"]["frame_interval_s"] = 0.1
    else:
        override["state_dict"]["forecaster.encoder.weight"] = override["state_dict"]["forecaster.encoder.weight"].double()
    path = case.directory / "wrong_predictor.ckpt"
    torch.save(override, path)
    case.config["model"]["predictor_checkpoint"] = str(path)
    with pytest.raises(ValueError, match="architecture|frozen encoder"):
        pipeline.load_pipeline(case.config, case.checkpoint)
