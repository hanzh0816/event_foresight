"""Small, explicit project configuration; paths never depend on shell cwd."""

from copy import deepcopy
import json
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
                           batch_size=1, num_workers=4).items():
        data.setdefault(key, value)
    for key, value in dict(seed=42, precision="bf16-mixed", encoder_chunk_size=1,
                           cpu_threads=4, log_every=10).items():
        runtime.setdefault(key, value)
    if list(data["image_size"]) != [448, 896] or data["context_frames"] != 4:
        raise ValueError("This baseline uses four RGB frames at 448x896")
    if runtime["precision"] not in ("bf16-mixed", "32-true"):
        raise ValueError("precision must be bf16-mixed or 32-true")
    for container, keys in ((data, ("root", "manifest")),
                            (model, ("dinof_checkpoint", "detector_checkpoint", "predictor_checkpoint", "event_encoder_checkpoint"))):
        for key in keys:
            if container.get(key):
                container[key] = str(resolve_path(container[key]))
    if not model.get("dinof_checkpoint"):
        raise ValueError("model.dinof_checkpoint is required")
    if data.get("manifest"):
        selection = json.loads(Path(data["manifest"]).read_text())
        for split in ("train", "val", "test"):
            if split in selection and f"{split}_sample_ids" not in data:
                ids = selection[split]["sample_ids"]
                if not ids or any(not isinstance(i, str) for i in ids) or len(ids) != len(set(ids)):
                    raise ValueError(f"Invalid {split} sample manifest")
                data[f"{split}_sample_ids"] = ids
    return config
