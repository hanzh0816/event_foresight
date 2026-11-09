"""CPU regression tests for full Detector restoration and Predictor overrides."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from foresight import pipeline
from foresight.checkpoints import TRAINING_FORMAT
from foresight.models.detector import FutureDetector


class TinyForecaster(nn.Module):
    """Replace only the large asset-backed model, preserving component namespaces."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.pca = nn.Module()
        self.pca.register_buffer("basis", torch.eye(2))
        self.pca.register_buffer("mean", torch.zeros(2))
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
        # override guard must compare against restored E/PCA, not initial assets.
        source.forecaster.encoder.weight.fill_(0.25)
        source.forecaster.encoder.bias.fill_(0.5)
        source.forecaster.pca.basis.add_(0.125)
        source.forecaster.pca.mean.fill_(0.75)
        source.forecaster.predictor.weight.fill_(1.25)
        source.forecaster.predictor.bias.fill_(1.5)
    payload = dict(foresight_format=TRAINING_FORMAT, stage="detector",
                   detector_architecture=architecture,
                   state_dict=deepcopy(source.state_dict()))
    checkpoint = tmp_path / "detector.ckpt"
    torch.save(payload, checkpoint)
    config = dict(model=dict(dinof_checkpoint=str(tmp_path / "unused_dinof.pt"),
                             detector_checkpoint=None, predictor_checkpoint=None),
                  runtime=dict(encoder_chunk_size=1))
    created = []

    def create_forecaster(checkpoint, *, encoder_chunk_size):
        assert encoder_chunk_size == 1
        forecaster = TinyForecaster()
        created.append(forecaster)
        return forecaster

    load_asset = Mock(side_effect=AssertionError("A trained Detector must not read an initial head asset"))
    monkeypatch.setattr(pipeline, "DINOForecaster", create_forecaster)
    monkeypatch.setattr(pipeline, "load_detector", load_asset)
    return SimpleNamespace(config=config, payload=payload, checkpoint=checkpoint,
                           directory=tmp_path, created=created, load_asset=load_asset)


def predictor_override(case):
    state = {name: value.clone() for name, value in case.payload["state_dict"].items()
             if name.startswith("forecaster.")}
    state["forecaster.predictor.weight"].fill_(7.0)
    state["forecaster.predictor.bias"].fill_(-3.0)
    return dict(foresight_format=TRAINING_FORMAT, stage="predictor", state_dict=state)


@pytest.mark.parametrize("configured_asset", [False, True], ids=["no-head-asset", "missing-head-asset"])
def test_trained_detector_restores_without_original_head_asset(trained_detector, configured_asset):
    case = trained_detector
    if configured_asset:
        case.config["model"]["detector_checkpoint"] = str(case.directory / "deleted_detector.pt")
    restored = pipeline.load_pipeline(case.config, case.checkpoint)
    case.load_asset.assert_not_called()
    assert_state_equal(restored.state_dict(), case.payload["state_dict"])
    assert restored.detector.grid_hw == (2, 3)
    assert restored.detector.query_embedding.weight.shape == (3, 16)
    assert all(value.device.type == "cpu" for value in restored.state_dict().values())
    assert all(not module.training for module in restored.modules())
    assert all(not parameter.requires_grad for parameter in restored.parameters())


@pytest.mark.parametrize("spec", [
    {"readout_feature_source": "rgb_current_retained"},
    {"readout_coordinates": "latent @ pca_components"},
    {"prediction_ms": 0},
    {"training_objective": "current_detection"},
    {"readout_feature_source": "rgb_current_retained",
     "readout_coordinates": "latent @ pca_components", "prediction_ms": 0},
])
def test_forecast_pipeline_rejects_current_detector_protocol(trained_detector, monkeypatch, spec):
    case = trained_detector
    case.payload["input_spec"] = spec
    torch.save(case.payload, case.checkpoint)
    construct_detector = Mock(side_effect=AssertionError("Reject the protocol before loading detector tensors"))
    monkeypatch.setattr(pipeline, "FutureDetector", construct_detector)
    with pytest.raises(ValueError, match="requires a forecast detector"):
        pipeline.load_pipeline(case.config, case.checkpoint)
    construct_detector.assert_not_called()
    case.load_asset.assert_not_called()


@pytest.mark.parametrize("spec", [
    {"prediction_ms": 50},
    {"readout_feature_source": "rgb_forecast", "readout_coordinates": "inverse_pca",
     "prediction_ms": 50, "training_objective": "forecast_detection"},
])
def test_forecast_pipeline_accepts_explicit_forecast_protocol(trained_detector, spec):
    case = trained_detector
    case.payload["input_spec"] = spec
    torch.save(case.payload, case.checkpoint)
    restored = pipeline.load_pipeline(case.config, case.checkpoint)
    assert_state_equal(restored.state_dict(), case.payload["state_dict"])


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
    case.load_asset.assert_not_called()


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
    # Full-state comparison also proves that Encoder, PCA and Detector survive
    # the override unchanged, rather than merely checking the new P weights.
    assert_state_equal(restored.state_dict(), expected)
    case.load_asset.assert_not_called()
    assert all(not parameter.requires_grad for parameter in restored.parameters())


@pytest.mark.parametrize("component,key", [("encoder", "weight"), ("pca", "basis")])
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
    case.load_asset.assert_not_called()
