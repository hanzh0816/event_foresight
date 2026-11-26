"""Small, explicit project configuration; paths never depend on shell cwd."""

from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def load_config(filename, _seen=None):
    path = resolve_path(filename)
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError("Configuration inheritance cycle")
    seen.add(path)
    parent = config.pop("extends", None)
    if parent:
        base = load_config(path.parent / parent, seen)
        def merge(left, right):
            result = deepcopy(left)
            for key, value in right.items():
                result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
            return result
        config = merge(base, config)
    config = deepcopy(config)
    for section in ("data", "model", "optimization", "runtime"):
        if section not in config:
            config[section] = {}
        if not isinstance(config[section], dict):
            raise ValueError(f"{section} must be a mapping")
    data, model, runtime = config["data"], config["model"], config["runtime"]
    for key, value in dict(image_size=[448, 896], sample_history_frames=40,
                           context_frames=4, tolerance_us=1000,
                           frame_interval_s=0.05, batch_size=1, num_workers=4).items():
        data.setdefault(key, value)
    for key, value in dict(seed=42, precision="bf16-mixed", encoder_chunk_size=1,
                           cpu_threads=4, log_every=10).items():
        runtime.setdefault(key, value)
    size = data["image_size"]
    if (not isinstance(size, (list, tuple)) or len(size) != 2
            or any(type(x) is not int or x < 14 or x % 14 for x in size)):
        raise ValueError("image_size must contain two positive multiples of 14")
    if type(data["context_frames"]) is not int or not 1 <= data["context_frames"] <= data["sample_history_frames"]:
        raise ValueError("context_frames must be within the inventory history")
    if data["frame_interval_s"] != 0.05:
        raise ValueError("The unchanged DSEC protocol requires frame_interval_s=0.05")
    if not isinstance(model.get("predictor", {}), dict):
        raise ValueError("model.predictor must be a mapping")
    unknown_model_keys = model.keys() - {"encoder_checkpoint", "predictor_checkpoint", "rgb_checkpoint", "detector_checkpoint", "predictor", "event"}
    if unknown_model_keys:
        raise ValueError(f"Unsupported model configuration fields: {sorted(unknown_model_keys)}")
    if runtime["precision"] not in ("bf16-mixed", "32-true"):
        raise ValueError("precision must be bf16-mixed or 32-true")
    for container, keys in ((data, ("root",)),
                            (model, ("encoder_checkpoint", "predictor_checkpoint", "rgb_checkpoint", "detector_checkpoint"))):
        for key in keys:
            if container.get(key):
                container[key] = str(resolve_path(container[key]))
    if model.get("predictor_checkpoint") and model.get("rgb_checkpoint"):
        raise ValueError("Choose predictor_checkpoint or rgb_checkpoint, not both")
    event = model.get("event", {})
    if not isinstance(event, dict):
        raise ValueError("model.event must be a mapping")
    if event.get("enabled", False):
        bins = data.setdefault("event_bins", 10)
        if type(bins) is not int or bins < 1:
            raise ValueError("event_bins must be a positive integer")
        event.setdefault("in_channels", 2 * bins)
        event.setdefault("input_size", [512, 1024])
        if event["in_channels"] != 2 * bins:
            raise ValueError("RVT input channels must equal 2 * event_bins")
        if tuple(event["input_size"]) != tuple((x // 14) * 16 for x in size):
            raise ValueError("Event input must align its stride-16 grid with the RGB patch grid")
        if model.get("rgb_checkpoint"):
            config["optimization"]["rgb_finetune"] = True
            model.setdefault("predictor", {})["gate_init"] = 0.0
    elif model.get("rgb_checkpoint"):
        raise ValueError("rgb_checkpoint initializes an Event-RGB model")
    if not model.get("encoder_checkpoint"):
        raise ValueError("model.encoder_checkpoint is required")
    return config
