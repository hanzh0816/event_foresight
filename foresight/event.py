"""Event Encoder asset loading and the frozen RGB forecasting/readout path."""

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
    elif saved.get("foresight_format") == TRAINING_FORMAT and saved.get("stage") == "event":
        state = component_state(saved, "event_encoder.")
    else:
        raise ValueError("Expected an exported Event Encoder or a new-project event training checkpoint")
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


def load_event_pipeline(config, encoder_checkpoint, detector_checkpoint=None):
    rgb = load_pipeline(config, detector_checkpoint)
    encoder = EventEncoder(rgb.forecaster)
    _, encoding = load_event_encoder(encoder, encoder_checkpoint)
    return EventForecastDetector(rgb, encoder, encoding,
                                 config["runtime"]["encoder_chunk_size"]).eval().requires_grad_(False)
