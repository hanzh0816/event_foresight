"""Read complete DINO World checkpoints and verify their feature protocol."""

from collections.abc import Mapping
from pathlib import Path

import torch

TRAINING_FORMAT = "event-foresight-dino-world-training-v1"


def read_checkpoint(filename):
    path = Path(filename).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as error:
        if "mmap" not in str(error):
            raise
        saved = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(saved, Mapping) or not isinstance(saved.get("state_dict"), Mapping):
        raise ValueError("Checkpoint must contain a state_dict mapping")
    return saved


def input_spec(config, forecaster):
    """Keep the DSEC inventory protocol separate from model architecture."""
    data = config["data"]
    if tuple(data["image_size"]) != tuple(forecaster.image_size):
        raise ValueError("Dataset image_size differs from the loaded DINO World model")
    if data.get("context_frames", 4) != forecaster.history_length:
        raise ValueError("Dataset context_frames differs from the DINO World model")
    if data.get("frame_interval_s", 0.05) != 0.05:
        raise ValueError("DSEC uses a fixed 50 ms frame interval")
    return {
        "image_size": list(data["image_size"]),
        "context_frames": data.get("context_frames", 4),
        "sample_history_frames": data.get("sample_history_frames", 40),
        "tolerance_us": data.get("tolerance_us", 1000),
        "prediction_ms": 50,
        "frame_interval_s": data.get("frame_interval_s", 0.05),
        "preprocessing": "pil-bilinear-imagenet-mean-std",
        "feature_space": forecaster.feature_space,
    }


def validate_forecaster(saved, forecaster):
    if saved.get("forecaster_architecture") != forecaster.architecture:
        raise ValueError("Checkpoint DINO World architecture or temporal protocol differs")


def component_state(saved, prefix):
    state = {name.removeprefix(prefix): value for name, value in saved["state_dict"].items()
             if name.startswith(prefix)}
    if not state:
        raise ValueError(f"Checkpoint has no component {prefix}")
    return state


def load_predictor(predictor, filename, *, forecaster=None):
    saved = read_checkpoint(filename)
    if saved.get("foresight_format") != TRAINING_FORMAT or saved.get("stage") not in ("predictor", "detector"):
        raise ValueError("Predictor override must be a DINO World checkpoint trained by this project")
    if forecaster is not None:
        validate_forecaster(saved, forecaster)
        expected = forecaster.encoder.state_dict()
        actual = component_state(saved, "forecaster.encoder.")
        if expected.keys() != actual.keys() or any(
                not isinstance(actual[key], torch.Tensor)
                or expected[key].dtype != actual[key].dtype
                or not torch.equal(expected[key].detach().cpu(), actual[key]) for key in expected):
            raise ValueError("Predictor checkpoint uses a different frozen encoder")
    predictor.load_state_dict(component_state(saved, "forecaster.predictor."), strict=True)
    return saved
