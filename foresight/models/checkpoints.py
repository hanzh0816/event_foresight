"""Read complete DINO World checkpoints and verify their feature protocol."""

from collections.abc import Mapping
from pathlib import Path

import torch

TRAINING_FORMAT = "event-foresight-v31-training-v1"


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
    spec = {
        "image_size": list(data["image_size"]),
        "context_frames": data.get("context_frames", 4),
        "sample_history_frames": data.get("sample_history_frames", 40),
        "tolerance_us": data.get("tolerance_us", 1000),
        "prediction_ms": 50,
        "frame_interval_s": data.get("frame_interval_s", 0.05),
        "preprocessing": "pil-bilinear-imagenet-mean-std",
        "feature_space": forecaster.feature_space,
    }

    if config["model"].get("event", {}).get("enabled", False):
        spec["events"] = dict(bins=data.get("event_bins", 10), window_us=50000,
                              input_size=list(config["model"]["event"].get("input_size", [512, 1024])),
                              representation="polarity-major-count-log1p",
                              coordinates="left_distorted_event",
                              activity_threshold=data.get("event_activity_threshold", 1))
    return spec


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
    if saved.get("foresight_format") != TRAINING_FORMAT or saved.get("stage") != "predictor":
        raise ValueError("Predictor override must be a DINO World checkpoint trained by this project")
    if forecaster is not None:
        validate_forecaster(saved, forecaster)
        validate_encoder(saved, forecaster)
    predictor.load_state_dict(component_state(saved, "forecaster.predictor."), strict=True)
    if forecaster is not None and getattr(forecaster, "event_encoder", None) is not None:
        forecaster.event_encoder.load_state_dict(component_state(saved, "forecaster.event_encoder."), strict=True)
    return saved


def validate_encoder(saved, forecaster):
    expected = forecaster.encoder.state_dict()
    actual = component_state(saved, "forecaster.encoder.")
    if expected.keys() != actual.keys() or any(
            not isinstance(actual[key], torch.Tensor) or expected[key].dtype != actual[key].dtype
            or not torch.equal(expected[key].detach().cpu(), actual[key]) for key in expected):
        raise ValueError("Checkpoint uses a different frozen encoder")


def load_rgb_initializer(forecaster, filename):
    """Load all shared RGB weights, leaving only the new event modules random."""
    saved = read_checkpoint(filename)
    if saved.get("foresight_format") != TRAINING_FORMAT or saved.get("stage") != "predictor":
        raise ValueError("rgb_checkpoint must be a v3.1 RGB predictor checkpoint")
    validate_encoder(saved, forecaster)
    if any(key.startswith("forecaster.event_encoder.") for key in saved["state_dict"]):
        raise ValueError("rgb_checkpoint must contain a pure RGB baseline")
    expected_arch = dict(forecaster.predictor.architecture)
    actual_arch = dict(saved["forecaster_architecture"]["predictor"])
    for arch in (expected_arch, actual_arch):
        for key in ("event_enabled", "event_layers", "gate_init"):
            arch.pop(key, None)
    if actual_arch != expected_arch:
        raise ValueError("RGB initializer and Event-RGB predictor architectures differ")
    rgb = component_state(saved, "forecaster.predictor.")
    current = forecaster.predictor.state_dict()
    unexpected = set(rgb) - set(current)
    missing = set(current) - set(rgb)
    # Only explicitly named event parameters may be absent from the RGB checkpoint.
    if unexpected or any("event_" not in key and "gamma" not in key for key in missing):
        raise ValueError("RGB initializer is missing shared predictor weights")
    forecaster.predictor.load_state_dict({**current, **rgb}, strict=True)
    with torch.no_grad():
        for name, parameter in forecaster.predictor.named_parameters():
            if "gamma" in name:
                parameter.zero_()
    return saved


def load_shared_detector(filename, forecaster):
    from .detection import FutureDetector
    saved = read_checkpoint(filename)
    if saved.get("foresight_format") != TRAINING_FORMAT or saved.get("stage") != "detector":
        raise ValueError("detector_checkpoint must be a v3.1 shared detection head")
    validate_encoder(saved, forecaster)
    spec = saved.get("input_spec", {})
    if (spec.get("feature_space") != forecaster.feature_space
            or tuple(spec.get("image_size", ())) != tuple(forecaster.image_size)
            or spec.get("preprocessing") != "pil-bilinear-imagenet-mean-std"):
        raise ValueError("Shared detector feature or preprocessing protocol differs")
    architecture = saved.get("detector_architecture", {})
    if (architecture.get("feature_dim") != forecaster.feature_dim
            or tuple(architecture.get("grid_size", ())) != tuple(forecaster.grid_size)
            or architecture.get("num_classes") != 8):
        raise ValueError("Shared detector feature space or grid differs")
    detector = FutureDetector(**architecture)
    detector.load_state_dict(component_state(saved, "detector."), strict=True)
    return detector.eval().requires_grad_(False)
