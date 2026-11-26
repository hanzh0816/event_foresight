"""Assemble a trained predictor and one shared frozen detection head."""

from contextlib import nullcontext

import torch
from torch import nn

from .checkpoints import load_predictor, load_shared_detector
from .models.dino_world import DINOWorldForecaster
from .models.detector import postprocess


class ForecastDetector(nn.Module):
    def __init__(self, forecaster, detector):
        super().__init__()
        self.forecaster, self.detector = forecaster, detector

    def features(self, history_rgb, history_events=None, *, horizon_steps=1, event_off=False, persistence=False):
        b, t = history_rgb.shape[:2]
        history = self.forecaster.encode_latents(history_rgb.flatten(0, 1)).reshape(
            b, t, -1, self.forecaster.feature_dim)
        if persistence:
            return history[:, -1]
        kwargs = {}
        if getattr(self.forecaster, "event_encoder", None) is not None:
            if history_events is None and not event_off:
                raise ValueError("Event-RGB inference requires history_events")
            kwargs = dict(event_context=None if event_off else self.forecaster.encode_events(history_events),
                          event_grid_size=self.forecaster.event_encoder.output_grid, event_off=event_off)
        elif history_events is not None:
            raise ValueError("RGB baseline cannot consume events")
        if horizon_steps == 1:
            return self.forecaster.predictor(history, **kwargs)
        return self.forecaster.predictor.rollout(history, horizon_steps, **kwargs)[:, -1]

    def forward(self, history_rgb, history_events=None, **kwargs):
        return self.detector(self.features(history_rgb, history_events, **kwargs))

    def predict(self, history_rgb, history_events=None, **kwargs):
        return postprocess(self(history_rgb, history_events, **kwargs))


def load_pipeline(config, checkpoint=None, *, persistence=False):
    """A shared-head checkpoint never supplies random predictor weights for inference."""
    model, data = config["model"], config["data"]
    checkpoint = checkpoint or model.get("detector_checkpoint")
    if not checkpoint:
        raise ValueError("Inference requires a trained shared Detector checkpoint")
    if not persistence and not model.get("predictor_checkpoint"):
        raise ValueError("Forecasting requires a trained model.predictor_checkpoint")
    forecaster = DINOWorldForecaster(
        model["encoder_checkpoint"], image_size=data["image_size"],
        history_length=data.get("context_frames", 4), frame_interval_s=data.get("frame_interval_s", 0.05),
        encoder_chunk_size=config["runtime"]["encoder_chunk_size"],
        predictor_options=model.get("predictor", {}), event_options=model.get("event", {}))
    detector = load_shared_detector(checkpoint, forecaster)
    if not persistence:
        load_predictor(forecaster.predictor, model["predictor_checkpoint"], forecaster=forecaster)
    return ForecastDetector(forecaster, detector).eval().requires_grad_(False)


def setup_cuda(config):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; expose GPUs through CUDA_VISIBLE_DEVICES")
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.set_float32_matmul_precision("high")
    return torch.device("cuda", 0)


def inference_context(config):
    if config["runtime"]["precision"] == "bf16-mixed":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()
