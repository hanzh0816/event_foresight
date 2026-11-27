"""YAML configuration; source paths use the repository, runtime paths use $Work."""

from copy import deepcopy
import os
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]
_ENV_VAR = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


def work_root():
    """Read the case-sensitive runtime root without an implicit machine default."""
    value = os.environ.get("Work")
    if value is None or not value.strip():
        raise ValueError("Set Work to an absolute runtime root, for example: export Work=/data3/hzh")
    path = Path(value).expanduser()
    if not path.is_absolute() or _ENV_VAR.search(str(path)):
        raise ValueError("Work must be an absolute path with no unexpanded environment variables")
    return path.resolve()


def _expand_path(value):
    def replace(match):
        name = match.group(1) or match.group(2)
        if name == "Work":
            return str(work_root())
        replacement = os.environ.get(name)
        if replacement is None or not replacement.strip():
            raise ValueError(f"Path environment variable {name!r} is unset or empty")
        return replacement

    expanded = _ENV_VAR.sub(replace, os.fspath(value))
    if _ENV_VAR.search(expanded):
        raise ValueError(f"Path contains unexpanded environment variables: {value}")
    return Path(expanded).expanduser()


def resolve_path(value):
    """Expand an asset/input/output path, resolving relative paths under $Work."""
    path = _expand_path(value)
    return (path if path.is_absolute() else work_root() / path).resolve()


def resolve_config_path(value):
    """Locate a source configuration independently of the runtime/data root."""
    path = _expand_path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def load_config(filename, _seen=None):
    path = resolve_config_path(filename)
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
        base = load_config(path.parent / _expand_path(parent), seen)
        def merge(left, right):
            result = deepcopy(left)
            for key, value in right.items():
                result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
            return result
        config = merge(base, config)
    config = deepcopy(config)
    for section in ("data", "model", "optimization", "runtime", "evaluation", "inference", "resume_overrides"):
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
                           cpu_threads=4, log_every=10,
                           limit_train_batches=1.0, limit_val_batches=1.0).items():
        runtime.setdefault(key, value)
    if "stage" in config and config["stage"] not in ("predictor", "detector"):
        raise ValueError("stage must be predictor or detector")
    for key, minimum in (("batch_size", 1), ("num_workers", 0)):
        if type(data[key]) is not int or data[key] < minimum:
            raise ValueError(f"data.{key} must be an integer >= {minimum}")
    for key in ("limit_train_batches", "limit_val_batches"):
        value = runtime[key]
        if not ((type(value) is int and value >= 0)
                or (type(value) is float and 0 <= value <= 1)):
            raise ValueError(f"runtime.{key} must be a nonnegative batch count or a float in [0,1]")
    overrides = config["resume_overrides"]
    if overrides.keys() - {"epochs", "max_steps"}:
        raise ValueError("resume_overrides only supports epochs and max_steps")
    for section in (config["optimization"], overrides):
        for key in ("epochs", "max_steps", "accumulation_steps"):
            if key not in section:
                continue
            value = section[key]
            if type(value) is not int or not (value > 0 or (key in ("epochs", "max_steps") and value == -1)):
                raise ValueError(f"{key} must be positive" + (" or -1" if key in ("epochs", "max_steps") else ""))
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
                            (model, ("encoder_checkpoint", "predictor_checkpoint", "rgb_checkpoint", "detector_checkpoint")),
                            (config, ("output_dir",)),
                            (config["evaluation"], ("manifest", "output"))):
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
