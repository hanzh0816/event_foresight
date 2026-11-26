"""DINO World inference assembly, independent of Lightning and annotations."""

from contextlib import nullcontext

import torch
from torch import nn

from .checkpoints import TRAINING_FORMAT, read_checkpoint, load_predictor, input_spec, validate_forecaster
from .models.dino_world import DINOWorldForecaster
from .models.detector import FutureDetector, postprocess


class ForecastDetector(nn.Module):
    def __init__(self, forecaster, detector):
        super().__init__()
        self.forecaster, self.detector = forecaster, detector

    def forward(self, history_rgb):
        return self.detector(self.forecaster.forecast(history_rgb))

    def predict(self, history_rgb):
        return postprocess(self(history_rgb))


def load_pipeline(config, checkpoint=None):
    """Restore a complete trained World Detector, then apply an explicit P override."""
    if not checkpoint:
        raise ValueError("Inference requires --checkpoint with a trained DINO World Detector checkpoint")
    saved = read_checkpoint(checkpoint)
    if saved.get("foresight_format") != TRAINING_FORMAT or saved.get("stage") != "detector":
        raise ValueError("--checkpoint must be a DINO World Detector training checkpoint")
    architecture = saved.get("detector_architecture")
    if not isinstance(architecture, dict):
        raise ValueError("Trained Detector checkpoint lacks its architecture")
    model, data = config["model"], config["data"]
    forecaster = DINOWorldForecaster(
        model["encoder_checkpoint"], image_size=data["image_size"],
        history_length=data.get("context_frames", 4), frame_interval_s=data.get("frame_interval_s", 0.05),
        encoder_chunk_size=config["runtime"]["encoder_chunk_size"],
        predictor_options=model.get("predictor", {}))
    validate_forecaster(saved, forecaster)
    if saved.get("input_spec") != input_spec(config, forecaster):
        raise ValueError("Detector checkpoint and configured RGB data/feature protocol differ")
    if (architecture.get("feature_dim") != forecaster.feature_dim
            or tuple(architecture.get("grid_size", ())) != tuple(forecaster.grid_size)
            or architecture.get("num_classes") != 8):
        raise ValueError("Detector architecture does not match the DINO World DSEC feature/class contract")
    pipeline = ForecastDetector(forecaster, FutureDetector(**architecture))
    pipeline.load_state_dict(saved["state_dict"], strict=True)
    if model.get("predictor_checkpoint"):
        load_predictor(forecaster.predictor, model["predictor_checkpoint"], forecaster=forecaster)
    return pipeline.eval().requires_grad_(False)


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
