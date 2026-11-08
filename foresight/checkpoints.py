"""Read exported model assets and checkpoints made by this project."""

from collections.abc import Mapping
from pathlib import Path

import torch

TRAINING_FORMAT = "event-foresight-training-v1"


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


def component_state(saved, prefix):
    state = {name.removeprefix(prefix): value for name, value in saved["state_dict"].items()
             if name.startswith(prefix)}
    if not state:
        raise ValueError(f"Checkpoint has no component {prefix}")
    return state


def load_predictor(predictor, filename, *, forecaster=None):
    saved = read_checkpoint(filename)
    if saved.get("foresight_format") != TRAINING_FORMAT or saved.get("stage") not in ("predictor", "detector"):
        raise ValueError("Predictor override must be a checkpoint trained by this project")
    if forecaster is not None:
        for name in ("encoder", "pca"):
            expected = getattr(forecaster, name).state_dict()
            actual = component_state(saved, f"forecaster.{name}.")
            if expected.keys() != actual.keys() or any(
                    not torch.equal(expected[key].detach().cpu(), actual[key]) for key in expected):
                raise ValueError(f"Predictor checkpoint uses a different frozen {name}")
    predictor.load_state_dict(component_state(saved, "forecaster.predictor."), strict=True)
    return saved
