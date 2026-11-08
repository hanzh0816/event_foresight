"""Inference assembly, independent of Lightning and dataset annotations."""

from contextlib import nullcontext

import torch
from torch import nn

from .checkpoints import TRAINING_FORMAT, read_checkpoint, load_predictor
from .models.forecaster import DINOForecaster
from .models.detector import FutureDetector, load_detector, postprocess


class ForecastDetector(nn.Module):
    def __init__(self, forecaster, detector):
        super().__init__()
        self.forecaster, self.detector = forecaster, detector

    def forward(self, history_rgb):
        return self.detector(self.forecaster.forecast(history_rgb))

    def predict(self, history_rgb):
        return postprocess(self(history_rgb))


def load_pipeline(config, checkpoint=None):
    """A trained Detector restores its P; an explicit P override wins last."""
    model = config["model"]
    forecaster = DINOForecaster(model["dinof_checkpoint"],
                               encoder_chunk_size=config["runtime"]["encoder_chunk_size"])
    saved = None
    if checkpoint:
        saved = read_checkpoint(checkpoint)
        if saved.get("foresight_format") != TRAINING_FORMAT or saved.get("stage") != "detector":
            raise ValueError("--checkpoint must be a new-project Detector training checkpoint")
        architecture = saved.get("detector_architecture")
        if not isinstance(architecture, dict):
            raise ValueError("Trained Detector checkpoint lacks its architecture")
        detector = FutureDetector(**architecture)
    else:
        if not model.get("detector_checkpoint"):
            raise ValueError("A detector asset or trained Detector checkpoint is required")
        detector = load_detector(model["detector_checkpoint"])
    pipeline = ForecastDetector(forecaster, detector)
    if saved is not None:
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
