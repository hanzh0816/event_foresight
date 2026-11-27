"""Frozen DINOv2 features, optional RVT encoding and the latent predictor."""

from collections.abc import Mapping
from copy import deepcopy
import math
from pathlib import Path

import torch
from torch import nn

from ..vendor.dinov2.models.vision_transformer import vit_base
from .predictor import DINOWorldPredictor, PREDICTOR_DEFAULTS


BACKBONE = dict(img_size=518, patch_size=14, init_values=1.0, ffn_layer="mlp",
                block_chunks=0, num_register_tokens=4,
                interpolate_antialias=True, interpolate_offset=0.0)


class DINOWorldForecaster(nn.Module):
    """Frozen last-layer DINOv2-reg features and a randomly initialized predictor."""

    feature_space = "dinov2-vitb14-reg-last-layer"
    feature_dim = 768
    patch_size = 14

    def __init__(self, encoder_checkpoint, image_size=(448, 896), history_length=4,
                 frame_interval_s=0.05, encoder_chunk_size=1, predictor_options=None,
                 event_options=None):
        super().__init__()
        if len(image_size) != 2 or any(type(x) is not int or x <= 0 or x % 14 for x in image_size):
            raise ValueError("image_size must contain two positive multiples of 14")
        if type(encoder_chunk_size) is not int or encoder_chunk_size < 1:
            raise ValueError("encoder_chunk_size must be a positive integer")
        self.image_size = tuple(image_size)
        self.grid_size = tuple(x // 14 for x in image_size)
        self.history_length, self.encoder_chunk_size = history_length, encoder_chunk_size
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
        path = Path(encoder_checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"DINOv2 encoder weights not found: {path}. See README.md for manual download.")
        try:
            state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except RuntimeError as error:
            if "mmap" not in str(error):
                raise
            state = torch.load(path, map_location="cpu", weights_only=True)
        if (not isinstance(state, Mapping) or not state
                or any(not isinstance(v, torch.Tensor) for v in state.values())):
            raise ValueError("Expected the official raw DINOv2 ViT-B/14-reg4 tensor state_dict")
        with torch.device("meta"):
            self.encoder = vit_base(**BACKBONE)
        self.encoder.load_state_dict(state, strict=True, assign=True)
        if any(v.is_meta for v in self.encoder.state_dict().values()):
            raise RuntimeError("Encoder load did not materialize every tensor")
        if any(v.is_floating_point() and (v.dtype != torch.float32 or not torch.isfinite(v).all())
               for v in self.encoder.state_dict().values()):
            raise ValueError("Encoder weights must be finite FP32 tensors")
        self.encoder.requires_grad_(False).eval()
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
        self.architecture = dict(model="dino-world-v31", encoder="dinov2_vitb14_reg",
                                 backbone=deepcopy(BACKBONE), image_size=list(image_size),
                                 feature_space=self.feature_space, feature_dim=self.feature_dim,
                                 predictor=deepcopy(self.predictor.architecture),
                                 event_encoder=(deepcopy(self.event_encoder.architecture)
                                                if self.event_encoder is not None else None))

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def _validate_rgb(self, rgb, ndim):
        if (rgb.ndim != ndim or rgb.shape[0] < 1 or rgb.shape[-3] != 3
                or tuple(rgb.shape[-2:]) != self.image_size or not rgb.is_floating_point()):
            raise ValueError(f"Expected floating {ndim}-D RGB ending [3,{self.image_size}]")

    @torch.no_grad()
    def encode_latents(self, rgb):
        """Patch tokens after encoder's final LN; no CLS/registers, L2 norm or PCA."""
        self._validate_rgb(rgb, 4)
        pieces = []
        for batch in rgb.split(self.encoder_chunk_size):
            features, = self.encoder.get_intermediate_layers(batch, n=[11], reshape=False)
            if tuple(features.shape) != (len(batch), math.prod(self.grid_size), self.feature_dim):
                raise ValueError("DINOv2 returned an incompatible last-layer patch feature shape")
            pieces.append(features)
        return torch.cat(pieces)

    def encode_events(self, events):
        if self.event_encoder is None:
            raise ValueError("Event encoding requires event.enabled=True")
        if events is None:
            raise ValueError("Event-enabled forecasting requires observed event voxels or event_off=True")
        return self.event_encoder(events)

    def forecast_latents(self, history, events=None, *, event_off=False):
        self._validate_rgb(history, 5)
        if history.shape[1] != self.history_length:
            raise ValueError(f"Expected {self.history_length} historical RGB frames")
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
