"""Frozen DINOv2 features, optional RVT encoding and the latent predictor."""

from copy import deepcopy
import math

from torch import nn

from .encoder import DINOFeatureEncoder
from .predictor import DINOWorldPredictor, PREDICTOR_DEFAULTS


class DINOWorldForecaster(nn.Module):
    """Frozen last-layer DINOv2-reg features and a randomly initialized predictor."""

    feature_space = DINOFeatureEncoder.feature_space
    feature_dim = DINOFeatureEncoder.feature_dim
    patch_size = DINOFeatureEncoder.patch_size

    def __init__(self, encoder_checkpoint, image_size=(448, 896), history_length=4,
                 frame_interval_s=0.05, encoder_chunk_size=1, predictor_options=None,
                 event_options=None):
        super().__init__()
        self.encoder = DINOFeatureEncoder(encoder_checkpoint, image_size, encoder_chunk_size)
        self.image_size = self.encoder.image_size
        self.grid_size = self.encoder.grid_size
        self.history_length = history_length
        options = dict(PREDICTOR_DEFAULTS)
        supplied = dict(predictor_options or {})
        if set(supplied) - set(options):
            raise ValueError(f"Unknown predictor options: {sorted(set(supplied) - set(options))}")
        options.update(supplied)
        event_options = dict(event_options or {})
        event_enabled = event_options.pop("enabled", False)
        if type(event_enabled) is not bool:
            raise ValueError("event.enabled must be bool")
        if "event_enabled" in supplied and supplied["event_enabled"] != event_enabled:
            raise ValueError("predictor.event_enabled must agree with event.enabled")
        options["event_enabled"] = event_enabled
        self.predictor = DINOWorldPredictor(grid_size=self.grid_size, history_length=history_length,
                                            feature_dim=self.feature_dim, frame_interval_s=frame_interval_s,
                                            **options)
        self.event_encoder = None
        if event_enabled:
            from .rvt import RVTEventEncoder
            self.event_encoder = RVTEventEncoder(feature_dim=self.feature_dim, **event_options)
            grid = self.event_encoder.output_grid
            if tuple(grid) not in (self.grid_size, tuple(max(1, x // 2) for x in self.grid_size)):
                raise ValueError("RVT output grid must equal the RGB grid or its half-resolution compact grid")
        self.architecture = dict(model="dino-world-v31", encoder=deepcopy(self.encoder.architecture),
                                 predictor=deepcopy(self.predictor.architecture),
                                 event_encoder=(deepcopy(self.event_encoder.architecture)
                                                if self.event_encoder is not None else None))

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def encode_latents(self, rgb):
        """Patch tokens after encoder's final LN; no CLS/registers, L2 norm or PCA."""
        return self.encoder(rgb)

    def encode_events(self, events):
        if self.event_encoder is None:
            raise ValueError("Event encoding requires event.enabled=True")
        if events is None:
            raise ValueError("Event-enabled forecasting requires observed event voxels or event_off=True")
        return self.event_encoder(events)

    def forecast_latents(self, history, events=None, *, event_off=False):
        encoded = self.encode_latents(history.flatten(0, 1))
        context = encoded.reshape(history.shape[0], self.history_length,
                                  math.prod(self.grid_size), self.feature_dim)
        if self.event_encoder is None:
            if events is not None:
                raise ValueError("RGB-only forecaster does not accept events")
            return self.predictor(context)
        event_context = None if event_off else self.encode_events(events)
        return self.predictor(context, event_context=event_context,
                              event_grid_size=self.event_encoder.output_grid, event_off=event_off)

    forecast = forecast_latents
    forward = forecast_latents
