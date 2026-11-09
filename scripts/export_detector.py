"""Export a trained RGB forecast detector as a portable, model-only asset."""

import argparse
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from foresight.checkpoints import TRAINING_FORMAT, component_state, read_checkpoint
from foresight.config import resolve_path
from foresight.models.detector import DETECTOR_FORMAT, load_detector


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate_forecast_source(saved):
    if saved.get("foresight_format") != TRAINING_FORMAT or saved.get("stage") != "detector":
        raise ValueError("Require a detector training checkpoint from this repository")
    spec = saved.get("input_spec")
    if not isinstance(spec, Mapping):
        raise ValueError("Detector checkpoint has no input_spec")
    # A current-frame retained-coordinate head shares stage='detector' and the
    # architecture, but consumes different features and must never be mislabeled.
    if (spec.get("readout_feature_source") not in (None, "rgb_forecast")
            or spec.get("training_objective") not in (None, "forecast_detection")
            or spec.get("readout_coordinates") not in (None, "inverse_pca")
            or spec.get("prediction_ms") != 50):
        raise ValueError("Only a forecast detector can be exported; current/retained heads are incompatible")
    expected = dict(image_size=[448, 896], context_frames=4,
                    feature_space="dinov2-multilayer-fixed-pca")
    if any(spec.get(key) != value for key, value in expected.items()):
        raise ValueError("Detector checkpoint uses an incompatible RGB feature protocol")
    if not isinstance(saved.get("detector_architecture"), Mapping):
        raise ValueError("Detector checkpoint has no architecture")
    state = component_state(saved, "detector.")
    for name, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                or value.device.type != "cpu" or not torch.isfinite(value).all()):
            raise ValueError(f"Detector tensor {name} must be finite CPU FP32")
    return state


def export_detector(source, output):
    source, output = resolve_path(source), resolve_path(output)
    if source == output:
        raise ValueError("Export destination must differ from its source checkpoint")
    if output.exists():
        raise FileExistsError(output)
    saved = read_checkpoint(source)
    state = validate_forecast_source(saved)
    provenance = deepcopy(saved.get("provenance", {}))
    if not isinstance(provenance, dict):
        raise ValueError("Checkpoint provenance must be a mapping")
    provenance.update(source_filename=source.name, source_checkpoint_sha256=file_sha256(source),
                      source_checkpoint_format=TRAINING_FORMAT,
                      source_global_step=saved.get("global_step"), source_epoch=saved.get("epoch"))
    metadata = dict(architecture=deepcopy(dict(saved["detector_architecture"])),
                    input_spec=deepcopy(dict(saved["input_spec"])), provenance=provenance)
    # Export metadata contains only portable primitives, so weights_only loading
    # does not depend on the original training Python objects or checkout.
    try:
        metadata = json.loads(json.dumps(metadata, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("Detector metadata must contain JSON-compatible finite values") from error
    exported = dict(format=DETECTOR_FORMAT,
                    state_dict={name: value.detach().clone() for name, value in state.items()},
                    **metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(exported, temporary)
        model = load_detector(temporary)  # Real architecture, strict tensor keys/shapes.
        actual = model.state_dict()
        if actual.keys() != state.keys() or any(not torch.equal(actual[key], state[key]) for key in state):
            raise RuntimeError("Export changed detector tensors")
        # Hard-link promotion is atomic and fails if another writer created the
        # requested destination, unlike replace(), which can overwrite a file.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(output=str(output), bytes=output.stat().st_size, sha256=file_sha256(output),
                tensor_count=len(state), all_tensors_equal=True,
                source_checkpoint_sha256=provenance["source_checkpoint_sha256"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source", required=True, help="Trusted best.ckpt/last.ckpt from --stage detector")
    parser.add_argument("--output", required=True, help="New detector.pt asset; existing files are never overwritten")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    if torch.cuda.is_initialized():
        raise RuntimeError("Detector export must run without CUDA initialization")
    print(json.dumps(export_detector(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
