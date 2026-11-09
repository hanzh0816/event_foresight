"""One-time CPU conversion of trusted source weights, with tensor-exact audit.

Run from the installed project as ``python -m scripts.import_weights``. Source
Lightning files are only accepted here; normal inference loads weights_only
exports and never consults their old repository or training configuration.
"""

import argparse
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import os
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


def validate_fp32_state(state, count, label):
    if not isinstance(state, Mapping) or len(state) != count:
        raise ValueError(f"Expected all{count} {label} tensors")
    for key, value in state.items():
        if (not isinstance(key, str) or not isinstance(value, torch.Tensor)
                or value.device.type != "cpu" or value.dtype != torch.float32
                or not torch.isfinite(value).all()):
            raise ValueError(f"Source tensor {key} is not finite CPU FP32")


def validate_dinof_source(dinof):
    """Validate the official asset independently of any DSEC-trained task head."""
    if not isinstance(dinof, Mapping):
        raise ValueError("Official source must be a Lightning checkpoint mapping")
    hyper = dinof.get("hyper_parameters", {})
    if not isinstance(hyper, Mapping):
        raise ValueError("Official source hyper_parameters must be a mapping")
    args = hyper.get("args", hyper)
    if not isinstance(args, Mapping):
        if not hasattr(args, "__dict__"):
            raise ValueError("Official source args must be a mapping or namespace")
        args = vars(args)
    settings = {"sequence_length": 5, "dinov2_variant": "vitb14_reg", "masking": "simple_replace",
                "feature_extractor": "dino", "seperable_attention": True, "seperable_window_size": 1,
                "use_first_last": False, "use_fc_bias": True, "output_activation": "none",
                "crop_feats": False, "sliding_window_inference": False,
                "hidden_dim": 1152, "heads": 8, "layers": 12, "dropout": 0.1}
    for key, value in settings.items():
        if key in args and args[key] != value:
            raise ValueError(f"Unsupported source Predictor setting: {key}")
    if "img_size" in args and list(args["img_size"]) != [448, 896]:
        raise ValueError("Official source resolution mismatch")
    if "d_layers" in args and list(args["d_layers"]) != [2, 5, 8, 11]:
        raise ValueError("Official source intermediate layers mismatch")
    validate_fp32_state(dinof.get("state_dict"), 404, "official")


def validate_sources(dinof, detector, dinof_hash):
    validate_dinof_source(dinof)
    if not isinstance(detector, Mapping) or detector.get("dinof_format") != "dino-foresight-dsec-head-v1":
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
    validate_fp32_state(detector.get("state_dict"), 296, "detector")
    if any(not k.startswith("detector.") for k in detector["state_dict"]):
        raise ValueError("Detector source contains unexpected frozen or training parameters")
    return config, spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dinof-source", required=True)
    parser.add_argument("--detector-source", help="Optional compatible DSEC head; omit to export only the official RGB model")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    if torch.cuda.is_initialized():
        raise RuntimeError("Weight conversion must run without CUDA initialization")
    sources = {name: Path(getattr(args, f"{name}_source")).expanduser().resolve()
               for name in ("dinof", "detector") if getattr(args, f"{name}_source")}
    output = Path(args.output_dir).expanduser().resolve()
    destinations = {name: output / f"{name}.pt" for name in sources}
    if any(path in sources.values() for path in destinations.values()):
        raise ValueError("Export destinations must differ from source assets")
    if not args.verify_only and any((output / name).exists() for name in ("dinof.pt", "detector.pt", "assets.json")):
        raise FileExistsError("Export destination already contains assets; use --verify-only or another directory")
    hashes = {name: file_sha256(path) for name, path in sources.items()}
    # Explicitly supplied files are trusted historical Lightning
    # assets. Their training metadata is discarded after conversion.
    original = {name: torch.load(path, map_location="cpu", weights_only=False, mmap=True)
                for name, path in sources.items()}
    spec = None
    if "detector" in original:
        config, spec = validate_sources(original["dinof"], original["detector"], hashes["dinof"])
    else:
        validate_dinof_source(original["dinof"])
    state_dicts = {"dinof": OrderedDict(original["dinof"]["state_dict"])}
    provenance = {name: {"source_filename": sources[name].name, "source_checkpoint_sha256": hashes[name],
                         "conversion": "CPU tensor-exact export; optimizer/scheduler/trainer state excluded"}
                  for name in sources}
    bundles = {
        "dinof": {"format": DINO_FORMAT, "architecture": deepcopy(DINO_ARCHITECTURE),
                  "state_dict": state_dicts["dinof"], "provenance": provenance["dinof"]},
    }
    if "detector" in original:
        state_dicts["detector"] = OrderedDict((key.removeprefix("detector."), value)
                                             for key, value in original["detector"]["state_dict"].items())
        bundles["detector"] = {
            "format": DETECTOR_FORMAT, "architecture": dict(config["model"], feature_dim=3072, grid_size=[32, 64]),
            "state_dict": state_dicts["detector"], "input_spec": deepcopy(spec), "provenance": provenance["detector"]}
    staged = []
    paths = destinations if args.verify_only else {name: path.with_suffix(".pt.tmp") for name, path in destinations.items()}
    try:
        if not args.verify_only:
            output.mkdir(parents=True, exist_ok=True)
            for name, bundle in bundles.items():
                with paths[name].open("xb") as handle:
                    staged.append(paths[name])
                    torch.save(bundle, handle)
        report = {"format": "event-foresight-assets-v1", "assets": {}}
        for name, path in paths.items():
            loaded = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            for key in ("format", "architecture", "provenance", "input_spec"):
                if loaded.get(key) != bundles[name].get(key):
                    raise ValueError(f"Export metadata mismatch: {name}.{key}")
            count = verify_tensors(state_dicts[name], loaded["state_dict"])
            report["assets"][name] = {"filename": destinations[name].name, "sha256": file_sha256(path),
                                      "source_checkpoint_sha256": hashes[name], "tensor_count": count,
                                      "tensor_elements": sum(v.numel() for v in loaded["state_dict"].values()),
                                      "all_tensors_equal": True}
        # Validate every name/shape against the real architecture before publishing.
        models = {"dinof": DINOForecaster(paths["dinof"])}
        if "detector" in paths:
            models["detector"] = load_detector(paths["detector"])
        for name, model in models.items():
            if (len(model.state_dict()) != report["assets"][name]["tensor_count"]
                    or any(v.is_meta for v in model.state_dict().values())):
                raise ValueError(f"Incomplete real architecture load: {name}")
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU conversion unexpectedly initialized CUDA")
        report["verification"] = {"tensor_count": sum(item["tensor_count"] for item in report["assets"].values()),
                                  "all_tensors_equal": True, "strict_real_architecture_load": True,
                                  "cuda_initialized": False, "numerical_forward_validated": False}
        if spec is not None:
            report["input_spec"] = deepcopy(spec)
        if not args.verify_only:
            temporary = output / "assets.json.tmp"
            with temporary.open("x") as handle:
                staged.append(temporary)
                handle.write(json.dumps(report, indent=2) + "\n")
            # Hard-link publication is atomic and fails if another process has
            # created the destination; unlike replace(), it never overwrites.
            for name, path in paths.items():
                os.link(path, destinations[name])
            os.link(temporary, output / "assets.json")
    finally:
        for path in staged:
            path.unlink(missing_ok=True)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
