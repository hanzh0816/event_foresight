"""Official-only export and strict compatibility of the optional detector."""

from collections import OrderedDict
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from scripts import import_weights as converter


def official_source():
    return {"state_dict": OrderedDict((f"weight_{i}", torch.tensor([float(i)])) for i in range(404)),
            "hyper_parameters": {"args": SimpleNamespace(img_size=(448, 896), d_layers=[2, 5, 8, 11],
                                                          heads=8, layers=12, hidden_dim=1152)}}


def detector_source(dinof_hash):
    return {
        "dinof_format": "dino-foresight-dsec-head-v1",
        "dinof_inputs": dict(image_size=[448, 896], context_frames=4, sample_history_frames=40,
                             prediction_ms=50, tolerance_us=1000, preprocessing="pil-bilinear-imagenet-mean-std",
                             feature_space="dinov2-multilayer-pca-inverse", feature_dim=3072,
                             grid_size=[32, 64], classes=converter.CLASSES,
                             asset_sha256={"checkpoint": dinof_hash}),
        "hyper_parameters": {"config": {"forecaster": deepcopy(converter.DINO_ARCHITECTURE), "model": {}}},
        "state_dict": OrderedDict((f"detector.weight_{i}", torch.tensor([float(i)])) for i in range(296)),
    }


def fake_model_loader(path):
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)["state_dict"]
    return SimpleNamespace(state_dict=lambda: state)


@pytest.fixture
def source_path(tmp_path):
    path = tmp_path / "official.ckpt"
    torch.save(official_source(), path)
    return path


def test_official_only_export_exact_safe_and_does_not_load_detector(monkeypatch, source_path, tmp_path):
    monkeypatch.setattr(converter, "DINOForecaster", fake_model_loader)
    monkeypatch.setattr(converter, "load_detector", lambda _: pytest.fail("Official bootstrap must not load a detector"))
    output = tmp_path / "output"
    command = ["--dinof-source", str(source_path), "--output-dir", str(output)]
    report = converter.main(command)
    assert set(report["assets"]) == {"dinof"}
    assert report["verification"]["tensor_count"] == 404
    assert report["verification"]["cuda_initialized"] is False
    saved = torch.load(output / "dinof.pt", map_location="cpu", weights_only=True)
    assert set(saved) == {"format", "architecture", "state_dict", "provenance"}
    assert converter.verify_tensors(official_source()["state_dict"], saved["state_dict"]) == 404
    assert saved["provenance"]["source_checkpoint_sha256"] == converter.file_sha256(source_path)
    assert set(path.name for path in output.iterdir()) == {"dinof.pt", "assets.json"}
    before = {path.name: path.stat().st_mtime_ns for path in output.iterdir()}
    assert converter.main(command + ["--verify-only"]) == report
    assert {path.name: path.stat().st_mtime_ns for path in output.iterdir()} == before
    with pytest.raises(FileExistsError, match="already contains"):
        converter.main(command)
    assert json.loads((output / "assets.json").read_text()) == report


def test_optional_detector_keeps_exact_pair_validation(monkeypatch, source_path, tmp_path):
    monkeypatch.setattr(converter, "DINOForecaster", fake_model_loader)
    monkeypatch.setattr(converter, "load_detector", fake_model_loader)
    detector = detector_source(converter.file_sha256(source_path))
    path = tmp_path / "head.ckpt"
    torch.save(detector, path)
    output = tmp_path / "pair"
    report = converter.main(["--dinof-source", str(source_path), "--detector-source", str(path),
                             "--output-dir", str(output)])
    assert report["verification"]["tensor_count"] == 700
    assert set(report["assets"]) == {"dinof", "detector"}
    assert report["input_spec"] == detector["dinof_inputs"]
    exported = torch.load(output / "detector.pt", map_location="cpu", weights_only=True)
    assert set(exported["state_dict"]) == {key.removeprefix("detector.") for key in detector["state_dict"]}
    detector["dinof_inputs"]["asset_sha256"]["checkpoint"] = "wrong-source"
    torch.save(detector, path)
    with pytest.raises(ValueError, match="exact official RGB"):
        converter.main(["--dinof-source", str(source_path), "--detector-source", str(path),
                        "--output-dir", str(tmp_path / "bad-pair")])


@pytest.mark.parametrize("bad", [torch.tensor([float("nan")]), torch.tensor([1.], dtype=torch.float16), "not-a-tensor"])
def test_rejects_nonfinite_or_wrong_tensor_type(bad):
    source = official_source()
    source["state_dict"]["weight_0"] = bad
    with pytest.raises(ValueError, match="finite CPU FP32"):
        converter.validate_dinof_source(source)


@pytest.mark.parametrize("key,value", [("heads", 16), ("layers", 6), ("img_size", [224, 448]),
                                       ("d_layers", [1, 3, 5, 7]), ("seperable_attention", False)])
def test_rejects_unsupported_official_architecture(key, value):
    source = official_source()
    setattr(source["hyper_parameters"]["args"], key, value)
    with pytest.raises(ValueError, match="setting|mismatch"):
        converter.validate_dinof_source(source)


def test_strict_architecture_failure_does_not_publish_assets(monkeypatch, source_path, tmp_path):
    def reject(path):
        raise ValueError("Wrong real model tensor shape")
    monkeypatch.setattr(converter, "DINOForecaster", reject)
    output = tmp_path / "bad-shape"
    with pytest.raises(ValueError, match="tensor shape"):
        converter.main(["--dinof-source", str(source_path), "--output-dir", str(output)])
    assert list(output.iterdir()) == []


def test_verify_only_detects_modified_tensor(monkeypatch, source_path, tmp_path):
    monkeypatch.setattr(converter, "DINOForecaster", fake_model_loader)
    output = tmp_path / "tampered"
    command = ["--dinof-source", str(source_path), "--output-dir", str(output)]
    converter.main(command)
    path = output / "dinof.pt"
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    bundle["state_dict"]["weight_0"] += 1
    torch.save(bundle, path)
    with pytest.raises(ValueError, match="changed tensor"):
        converter.main(command + ["--verify-only"])


def test_destination_created_during_validation_is_not_overwritten(monkeypatch, source_path, tmp_path):
    output = tmp_path / "race"
    def concurrent_create(path):
        (output / "dinof.pt").write_bytes(b"keep-existing")
        return fake_model_loader(path)
    monkeypatch.setattr(converter, "DINOForecaster", concurrent_create)
    with pytest.raises(FileExistsError):
        converter.main(["--dinof-source", str(source_path), "--output-dir", str(output)])
    assert (output / "dinof.pt").read_bytes() == b"keep-existing"
    assert set(path.name for path in output.iterdir()) == {"dinof.pt"}
