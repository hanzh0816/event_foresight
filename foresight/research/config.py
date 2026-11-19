"""Research recipes with explicit inheritance and causal-stage validation."""

from copy import deepcopy
import math
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

def _merge(parent, child):
    result = deepcopy(parent)
    for key, value in child.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def _read(path, stack=()):
    path = Path(path).expanduser().resolve()
    if path in stack:
        raise ValueError(f"Configuration inheritance cycle: {path}")
    with path.open() as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    parent = values.pop("extends", None)
    result = {}
    if parent:
        if not isinstance(parent, str):
            raise ValueError("extends must name one YAML file")
        result = _read(path.parent / parent, (*stack, path))
    # Asset paths follow the repository contract; extends is YAML-relative.
    for section, keys in (("data", ("root",)), ("model", ("dinof_checkpoint", "teacher_checkpoint", "detector_checkpoint", "init_checkpoint"))):
        if isinstance(values.get(section), dict):
            for key in keys:
                value = values[section].get(key)
                if value:
                    expanded = os.path.expandvars(os.path.expanduser(str(value)))
                    if "$" in expanded:
                        raise ValueError(f"Unresolved environment variable in {section}.{key}")
                    values[section][key] = str((ROOT / expanded).resolve())
    return _merge(result, values)


def validate_config(config):
    """Return a normalized copy. Assets are opened only by model/data creation."""
    cfg = deepcopy(config)
    stage = cfg.get("stage")
    if stage not in ("fusion", "transition", "motion"):
        raise ValueError("stage must be fusion, transition, or motion")
    defaults = {
        "data": dict(image_size=[448, 896], event_image_size=[128, 256], context_frames=4,
                     bins=5, rollout_steps=1, batch_size=1, num_workers=4, tolerance_us=1000,
                     stride=1, train_split="train", val_split="val"),
        "model": dict(motion_dim=128, hidden_dim=128, encoder_chunk_size=1,
                      temporal_mode="ordered", transition_mode="motion", motion_input="event",
                      copy_last_motion=False, train_transition=False),
        "loss": dict(scene_weight=1.0, motion_weight=1.0 if stage == "motion" else 0.0, kind="mse"),
        "optimization": dict(lr=1e-4, weight_decay=0.01, epochs=20, max_steps=-1,
                             accumulation_steps=1, clip_grad=1.0),
        "runtime": dict(accelerator="auto", devices="auto", precision="32-true", seed=42,
                        cpu_threads=4, log_every=10),
    }
    for section, base in defaults.items():
        if not isinstance(cfg.get(section, {}), dict):
            raise ValueError(f"{section} must be a mapping")
        cfg[section] = _merge(base, cfg.get(section, {}))
    data, model, loss, opt, runtime = (cfg[k] for k in defaults)
    model.setdefault("mode", "event_ordered" if stage == "fusion" else "explicit")
    for container, keys in ((data, ("context_frames", "bins", "rollout_steps", "batch_size", "stride")),
                            (model, ("motion_dim", "hidden_dim", "encoder_chunk_size")),
                            (opt, ("epochs", "accumulation_steps")), (runtime, ("cpu_threads", "log_every"))):
        for key in keys:
            if type(container[key]) is not int or container[key] <= 0:
                raise ValueError(f"{key} must be a positive integer")
    for key in ("image_size", "event_image_size"):
        if not isinstance(data[key], (list, tuple)) or len(data[key]) != 2 or any(type(v) is not int or v <= 0 for v in data[key]):
            raise ValueError(f"data.{key} must be two positive integers")
    if data["context_frames"] < 2:
        raise ValueError("At least two context_frames are required")
    if type(data["num_workers"]) is not int or data["num_workers"] < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if type(data["tolerance_us"]) is not int or data["tolerance_us"] < 0:
        raise ValueError("tolerance_us must be a nonnegative integer")
    if type(opt["max_steps"]) is not int or opt["max_steps"] == 0 or opt["max_steps"] < -1:
        raise ValueError("max_steps must be -1 or positive")
    for container, keys in ((loss, ("scene_weight", "motion_weight")), (opt, ("lr", "weight_decay", "clip_grad"))):
        for key in keys:
            value = container[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
    if opt["lr"] == 0 or loss["scene_weight"] == 0:
        raise ValueError("lr and scene_weight must be positive")
    if loss["kind"] not in ("mse", "smooth_l1"):
        raise ValueError("loss.kind must be mse or smooth_l1")
    if runtime["accelerator"] not in ("auto", "cpu", "gpu") or runtime["precision"] not in ("32-true", "bf16-mixed"):
        raise ValueError("Unsupported accelerator or precision")
    if stage == "fusion":
        if model["mode"] not in ("rgb", "rgb_control", "event_pooled", "event_ordered"):
            raise ValueError("Unknown fusion mode")
        if data["rollout_steps"] != 1:
            raise ValueError("Fusion screening uses exactly one rollout step")
        if loss["motion_weight"]:
            raise ValueError("Fusion has no motion supervision")
    if stage == "transition":
        if model["temporal_mode"] not in ("ordered", "pooled") or model["transition_mode"] not in ("motion", "no_motion"):
            raise ValueError("Invalid transition encoder/condition mode")
        if loss["motion_weight"]:
            raise ValueError("Transition stage has no motion-target loss")
    if stage == "motion":
        if not model.get("teacher_checkpoint"):
            raise ValueError("Motion stage requires model.teacher_checkpoint provenance")
        if model["mode"] not in ("explicit", "auxiliary", "direct"):
            raise ValueError("Unknown motion mode")
        if model["mode"] == "direct" and loss["motion_weight"]:
            raise ValueError("Direct fusion control requires motion_weight=0")
        if model["motion_input"] not in ("event", "rgb"):
            raise ValueError("motion_input must be event or rgb")
        if model["train_transition"] and model["mode"] != "explicit":
            raise ValueError("train_transition applies only to explicit motion rollout")
        if model["copy_last_motion"] and (model["mode"] != "explicit" or model["motion_input"] != "event"):
            raise ValueError("copy_last_motion is an explicit Event diagnostic")
    elif model.get("init_checkpoint"):
        raise ValueError("Weight-only init_checkpoint is supported for motion refinement only")
    horizons = data.get("evaluation_horizons")
    if horizons is not None and (not isinstance(horizons, list) or not horizons or any(type(v) is not int or v < 1 or v > data["rollout_steps"] for v in horizons) or len(set(horizons)) != len(horizons)):
        raise ValueError("evaluation_horizons must be unique steps within rollout_steps")
    return cfg


def load_config(path):
    return validate_config(_read(path))
