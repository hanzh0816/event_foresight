"""Small, explicit project configuration; paths never depend on shell cwd."""

from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def load_config(filename):
    path = resolve_path(filename)
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
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
    if data["context_frames"] != 4:
        raise ValueError("The DSEC comparison protocol uses four historical RGB frames")
    if data["frame_interval_s"] != 0.05:
        raise ValueError("The unchanged DSEC protocol requires frame_interval_s=0.05")
    if not isinstance(model.get("predictor", {}), dict):
        raise ValueError("model.predictor must be a mapping")
    if model.get("dinof_checkpoint"):
        raise ValueError("DINO-world requires model.encoder_checkpoint; DINO-Foresight weights are incompatible")
    if runtime["precision"] not in ("bf16-mixed", "32-true"):
        raise ValueError("precision must be bf16-mixed or 32-true")
    for container, keys in ((data, ("root",)),
                            (model, ("encoder_checkpoint", "detector_checkpoint", "predictor_checkpoint"))):
        for key in keys:
            if container.get(key):
                container[key] = str(resolve_path(container[key]))
    if not model.get("encoder_checkpoint"):
        raise ValueError("model.encoder_checkpoint is required")
    return config
