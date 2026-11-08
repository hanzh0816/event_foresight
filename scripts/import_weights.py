"""One-time CPU conversion of trusted source weights, with tensor-exact audit.

Run from the installed project as ``python -m scripts.import_weights``. Source
Lightning files are only accepted here; normal inference loads weights_only
exports and never consults their old repository or training configuration.
"""

import argparse
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from foresight.models.detector import DETECTOR_FORMAT, load_detector
from foresight.models.forecaster import DINO_ARCHITECTURE, DINO_FORMAT, DINOForecaster


CLASSES = ["pedestrian", "rider", "car", "bus", "truck", "bicycle", "motorcycle", "train"]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_tensors(source, exported):
    if set(source) != set(exported):
        raise ValueError("Export changed the tensor key set")
    for key in source:
        left, right = source[key], exported[key]
        if (left.dtype != right.dtype or left.shape != right.shape or left.device.type != "cpu"
                or right.device.type != "cpu" or not torch.equal(left, right)):
            raise ValueError(f"Export changed tensor {key}")
    return len(source)


def validate_sources(dinof, detector, dinof_hash):
    if detector.get("dinof_format") != "dino-foresight-dsec-head-v1":
        raise ValueError("Expected the existing RGB DSEC detector source checkpoint")
    spec = detector.get("dinof_inputs", {})
    expected = dict(image_size=[448, 896], context_frames=4, sample_history_frames=40,
                    prediction_ms=50, tolerance_us=1000, preprocessing="pil-bilinear-imagenet-mean-std",
                    feature_space="dinov2-multilayer-pca-inverse", feature_dim=3072,
                    grid_size=[32, 64], classes=CLASSES, asset_sha256={"checkpoint": dinof_hash})
    if spec != expected:
        raise ValueError("Detector source was not trained with this exact official RGB asset/protocol")
    config = detector["hyper_parameters"]["config"]
    teacher = config["forecaster"]
    for key in ("image_size", "history_length", "d_layers", "hidden_dim", "depth", "heads", "dropout"):
        if teacher.get(key) != DINO_ARCHITECTURE[key]:
            raise ValueError(f"Source model configuration mismatch: {key}")
    hyper = dinof.get("hyper_parameters", {})
    args = hyper.get("args", hyper)
    args = args if isinstance(args, dict) else vars(args)
    settings = {"sequence_length": 5, "dinov2_variant": "vitb14_reg", "masking": "simple_replace",
                "feature_extractor": "dino", "seperable_attention": True, "seperable_window_size": 1,
                "use_first_last": False, "use_fc_bias": True, "output_activation": "none",
                "crop_feats": False, "sliding_window_inference": False}
    for key, value in settings.items():
        if key in args and args[key] != value:
            raise ValueError(f"Unsupported source Predictor setting: {key}")
    if "img_size" in args and list(args["img_size"]) != [448, 896]:
        raise ValueError("Official source resolution mismatch")
    if "d_layers" in args and list(args["d_layers"]) != [2, 5, 8, 11]:
        raise ValueError("Official source intermediate layers mismatch")
    states = (dinof.get("state_dict", {}), detector.get("state_dict", {}))
    if len(states[0]) != 404 or len(states[1]) != 296:
        raise ValueError("Expected all404 official tensors and all296 detector tensors")
    if any(not k.startswith("detector.") for k in states[1]):
        raise ValueError("Detector source contains unexpected frozen or training parameters")
    for state in states:
        for key, value in state.items():
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(f"Source tensor {key} is not finite FP32")
    return config, spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dinof-source", required=True)
    parser.add_argument("--detector-source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    if torch.cuda.is_initialized():
        raise RuntimeError("Weight conversion must run without CUDA initialization")
    sources = {name: Path(getattr(args, f"{name}_source")).expanduser().resolve() for name in ("dinof", "detector")}
    output = Path(args.output_dir).expanduser().resolve()
    destinations = {name: output / f"{name}.pt" for name in sources}
    if any(path in sources.values() for path in destinations.values()):
        raise ValueError("Export destinations must differ from source assets")
    if not args.verify_only and any((output / name).exists() for name in ("dinof.pt", "detector.pt", "assets.json")):
        raise FileExistsError("Export destination already contains assets; use --verify-only or another directory")
    hashes = {name: file_sha256(path) for name, path in sources.items()}
    # These two explicitly supplied files are trusted historical Lightning
    # assets. Their training metadata is discarded after conversion.
    original = {name: torch.load(path, map_location="cpu", weights_only=False, mmap=True)
                for name, path in sources.items()}
    config, spec = validate_sources(original["dinof"], original["detector"], hashes["dinof"])
    detector_architecture = dict(config["model"], feature_dim=3072, grid_size=[32, 64])
    state_dicts = {"dinof": OrderedDict(original["dinof"]["state_dict"]),
                  "detector": OrderedDict((key.removeprefix("detector."), value)
                                          for key, value in original["detector"]["state_dict"].items())}
    provenance = {name: {"source_filename": sources[name].name, "source_checkpoint_sha256": hashes[name],
                         "conversion": "CPU tensor-exact export; optimizer/scheduler/trainer state excluded"}
                  for name in sources}
    bundles = {
        "dinof": {"format": DINO_FORMAT, "architecture": deepcopy(DINO_ARCHITECTURE),
                  "state_dict": state_dicts["dinof"], "provenance": provenance["dinof"]},
        "detector": {"format": DETECTOR_FORMAT, "architecture": detector_architecture,
                     "state_dict": state_dicts["detector"], "input_spec": deepcopy(spec),
                     "provenance": provenance["detector"]},
    }
    if not args.verify_only:
        output.mkdir(parents=True, exist_ok=True)
        for name, bundle in bundles.items():
            temporary = destinations[name].with_suffix(".pt.tmp")
            with temporary.open("xb") as handle:
                torch.save(bundle, handle)
            temporary.replace(destinations[name])
    report = {"format": "event-foresight-assets-v1", "assets": {}}
    for name, path in destinations.items():
        loaded = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if loaded["format"] != bundles[name]["format"] or loaded["architecture"] != bundles[name]["architecture"]:
            raise ValueError("Export metadata mismatch")
        count = verify_tensors(state_dicts[name], loaded["state_dict"])
        report["assets"][name] = {"filename": path.name, "sha256": file_sha256(path),
                                  "source_checkpoint_sha256": hashes[name], "tensor_count": count,
                                  "tensor_elements": sum(v.numel() for v in loaded["state_dict"].values()),
                                  "all_tensors_equal": True}
    # Strict real-model construction also checks all names and tensor shapes.
    forecaster = DINOForecaster(destinations["dinof"])
    detector = load_detector(destinations["detector"])
    assert len(forecaster.state_dict()) == 404 and len(detector.state_dict()) == 296
    assert not any(v.is_meta for model in (forecaster, detector) for v in model.state_dict().values())
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU conversion unexpectedly initialized CUDA")
    report["verification"] = {"tensor_count": 700, "all_tensors_equal": True,
                              "strict_real_architecture_load": True, "cuda_initialized": False,
                              "numerical_forward_validated": False}
    report["input_spec"] = deepcopy(spec)
    if not args.verify_only:
        temporary = output / "assets.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output / "assets.json")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
