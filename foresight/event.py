"""Event Encoder asset loading and the frozen RGB forecasting/readout path."""

from copy import deepcopy

import torch
from torch import nn

from .checkpoints import read_checkpoint, component_state, TRAINING_FORMAT
from .data.events import encoding_spec
from .models.event_encoder import EventEncoder
from .models.detector import postprocess
from .pipeline import load_pipeline

EVENT_FORMAT = "event-foresight-encoder-v1"


def read_event_checkpoint(filename):
    saved = read_checkpoint(filename)
    if saved.get("format") == EVENT_FORMAT:
        state = saved["state_dict"]
    elif saved.get("foresight_format") == TRAINING_FORMAT and saved.get("stage") in ("event", "event_predictor"):
        state = component_state(saved, "event_encoder.")
    else:
        raise ValueError("Expected an exported Event Encoder or an event/event_predictor training checkpoint")
    if saved.get("image_size") != [448, 896] and saved.get("input_spec", {}).get("image_size") != [448, 896]:
        raise ValueError("Event checkpoint must use the 448x896 input protocol")
    encoding = encoding_spec(saved["event_encoding"])
    return saved, state, encoding


def load_event_encoder(encoder, filename):
    saved, state, encoding = read_event_checkpoint(filename)
    expected = encoder.state_dict()
    if state.keys() != expected.keys() or any(state[k].shape != expected[k].shape for k in expected):
        raise ValueError("Event Encoder checkpoint structure differs")
    for key in ("mean", "std", "pca_mean", "pca_components", "input_mean", "input_std"):
        if not torch.equal(expected[key].detach().cpu(), state[key].detach().cpu()):
            raise ValueError(f"Event checkpoint changes frozen feature coordinates: {key}")
    encoder.load_state_dict(state, strict=True)
    return saved, encoding


class EventForecastDetector(nn.Module):
    def __init__(self, rgb_pipeline, encoder, encoding, chunk_size=1):
        super().__init__()
        self.forecaster = rgb_pipeline.forecaster
        self.detector = rgb_pipeline.detector
        self.event_encoder = encoder
        self.encoding = encoding
        self.chunk_size = chunk_size

    def forward(self, history_events):
        if history_events.ndim != 5 or history_events.shape[1] != 4:
            raise ValueError("Event inference requires four observed event windows")
        latent = torch.cat([self.event_encoder(part) for part in history_events.flatten(0, 1).split(self.chunk_size)])
        latent = latent.reshape(len(history_events), 4, *latent.shape[1:])
        future = self.forecaster.predictor(latent)
        return self.detector(self.forecaster.pca.inverse(future))

    def predict(self, history_events):
        return postprocess(self(history_events))


def load_event_predictor(forecaster, encoder, encoding, filename):
    """Load Event-conditioned P only after validating its entire frozen input path."""
    saved = read_checkpoint(filename)
    if saved.get("foresight_format") != TRAINING_FORMAT or saved.get("stage") != "event_predictor":
        raise ValueError("Expected an event_predictor training checkpoint")
    spec = saved.get("input_spec", {})
    if (spec.get("training_objective") != "event_predictor"
            or spec.get("input_modality") != "events"
            or spec.get("trainable_encoder") is not False
            or spec.get("trainable_predictor") is not True
            or spec.get("image_size") != list(forecaster.image_size)
            or spec.get("context_frames") != 4 or spec.get("prediction_ms") != 50
            or spec.get("feature_space") != "dinov2-multilayer-fixed-pca"
            or spec.get("event_encoding") != encoding
            or encoding_spec(saved.get("event_encoding", {})) != encoding):
        raise ValueError("Event Predictor checkpoint uses a different input/encoding protocol")
    for label, module, prefix in (("Event Encoder", encoder, "event_encoder."),
                                  ("RGB encoder", forecaster.encoder, "forecaster.encoder."),
                                  ("PCA", forecaster.pca, "forecaster.pca.")):
        expected, actual = module.state_dict(), component_state(saved, prefix)
        if expected.keys() != actual.keys() or any(
                not torch.equal(expected[key].detach().cpu(), actual[key].detach().cpu()) for key in expected):
            raise ValueError(f"Event Predictor checkpoint uses a different frozen {label}")
    # Check keys/shapes before mutating any Predictor parameter.
    state = component_state(saved, "forecaster.predictor.")
    expected = forecaster.predictor.state_dict()
    if expected.keys() != state.keys() or any(expected[key].shape != state[key].shape for key in expected):
        raise ValueError("Event Predictor checkpoint structure differs")
    forecaster.predictor.load_state_dict(state, strict=True)
    return saved


def load_event_pipeline(config, encoder_checkpoint, detector_checkpoint=None):
    # The detector checkpoint may contain an older RGB Predictor. Assemble that
    # backend first, then the fixed Event Encoder, then apply the P override last.
    base = deepcopy(config)
    override = base["model"].pop("predictor_checkpoint", None)
    rgb = load_pipeline(base, detector_checkpoint)
    encoder = EventEncoder(rgb.forecaster)
    saved, encoding = load_event_encoder(encoder, encoder_checkpoint)
    predictor_source = override or (encoder_checkpoint if saved.get("stage") == "event_predictor" else None)
    if predictor_source:
        predictor_saved = read_checkpoint(predictor_source)
        if predictor_saved.get("stage") == "event_predictor":
            load_event_predictor(rgb.forecaster, encoder, encoding, predictor_source)
        else:
            from .checkpoints import load_predictor
            load_predictor(rgb.forecaster.predictor, predictor_source, forecaster=rgb.forecaster)
    pipeline = EventForecastDetector(rgb, encoder, encoding,
        config["runtime"].get("event_encoder_chunk_size", config["runtime"]["encoder_chunk_size"]))
    pipeline.predictor_source = str(predictor_source or detector_checkpoint or config["model"]["dinof_checkpoint"])
    return pipeline.eval().requires_grad_(False)
