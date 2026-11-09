"""CPU checks for scratch forecast-head export and relocated training inputs."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from foresight.config import load_config
from foresight.models.detector import DETECTOR_FORMAT, load_detector
from foresight.training import DetectorTask
from scripts.export_detector import export_detector, file_sha256


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def forecast_checkpoint(tmp_path):
    # Real task metadata and detector implementation; a small hidden dimension
    # avoids allocating the full baseline while retaining its input contract.
    forecaster = nn.Linear(1, 1)
    forecaster.image_size = (448, 896)
    forecaster.feature_dim = 3072
    forecaster.grid_size = (32, 64)
    config = load_config("configs/detector_from_scratch.yaml")
    config["detector"].update(hidden_dim=8, num_queries=2, num_levels=1,
                              encoder_layers=1, decoder_layers=1, heads=2,
                              points=1, mlp_dim=16, dropout=0)
    task = DetectorTask(config, forecaster=forecaster)
    saved = {"state_dict": task.state_dict(), "global_step": 23, "epoch": 2,
             "optimizer_states": [{"state": "training-only"}],
             "provenance": {"experiment": "scratch-head"}}
    task.on_save_checkpoint(saved)
    source = tmp_path / "best.ckpt"
    torch.save(saved, source)
    return source, saved, task.detector


def test_detector_export_roundtrip_keeps_outputs_and_metadata(forecast_checkpoint, tmp_path):
    source, saved, detector = forecast_checkpoint
    output = tmp_path / "export" / "detector.pt"
    report = export_detector(source, output)
    exported = torch.load(output, map_location="cpu", weights_only=True)
    assert set(exported) == {"format", "architecture", "input_spec", "provenance", "state_dict"}
    assert exported["format"] == DETECTOR_FORMAT
    assert exported["architecture"] == saved["detector_architecture"]
    assert exported["input_spec"] == saved["input_spec"]
    assert exported["provenance"]["experiment"] == "scratch-head"
    assert exported["provenance"]["source_checkpoint_sha256"] == file_sha256(source)
    assert exported["provenance"]["source_global_step"] == 23
    assert exported["provenance"]["source_epoch"] == 2
    assert report["all_tensors_equal"] and report["tensor_count"] == len(detector.state_dict())
    assert report["sha256"] == file_sha256(output)
    loaded = load_detector(output).eval()
    for key, tensor in detector.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], tensor)
    features = torch.randn(1, 32 * 64, 3072)
    with torch.inference_mode():
        expected, actual = detector.eval()(features), loaded(features)
    for key in ("pred_logits", "pred_boxes"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("changes", [
    {"readout_feature_source": "rgb_current_retained"},
    {"training_objective": "current_detection"},
    {"readout_coordinates": "latent @ pca_components"},
    {"prediction_ms": 0},
])
def test_export_rejects_current_head_protocol(forecast_checkpoint, tmp_path, changes):
    source, saved, _ = forecast_checkpoint
    saved["input_spec"].update(changes)
    torch.save(saved, source)
    output = tmp_path / "detector.pt"
    with pytest.raises(ValueError, match="Only a forecast detector"):
        export_detector(source, output)
    assert not output.exists()


@pytest.mark.parametrize("corruption, error, match", [
    ("nonfinite", ValueError, "finite CPU FP32"),
    ("missing", RuntimeError, "Missing key"),
    ("shape", RuntimeError, "size mismatch"),
    ("stage", ValueError, "detector training checkpoint"),
])
def test_export_rejects_corruption_without_partial_output(
        forecast_checkpoint, tmp_path, corruption, error, match):
    source, saved, _ = forecast_checkpoint
    key = "detector.input_proj.0.weight"
    if corruption == "nonfinite":
        saved["state_dict"][key].flatten()[0] = float("nan")
    elif corruption == "missing":
        del saved["state_dict"][key]
    elif corruption == "shape":
        saved["state_dict"][key] = torch.zeros(1)
    else:
        saved["stage"] = "predictor"
    torch.save(saved, source)
    output = tmp_path / "detector.pt"
    with pytest.raises(error, match=match):
        export_detector(source, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".detector.pt.*.tmp"))


def test_export_refuses_overwrite(forecast_checkpoint, tmp_path):
    source, _, _ = forecast_checkpoint
    output = tmp_path / "existing.pt"
    output.write_bytes(b"keep existing model")
    with pytest.raises(FileExistsError):
        export_detector(source, output)
    assert output.read_bytes() == b"keep existing model"
    source_hash = file_sha256(source)
    with pytest.raises(ValueError, match="differ from"):
        export_detector(source, source)
    assert file_sha256(source) == source_hash


def test_training_cli_relocates_data_and_asset_on_resume(monkeypatch, forecast_checkpoint, tmp_path):
    from foresight import train

    source, saved, _ = forecast_checkpoint
    observed = {}

    def capture(config, **kwargs):
        observed.update(kwargs)
        observed["config"] = train.resume_config(config, kwargs["resume"], kwargs["stage"])

    monkeypatch.setattr(train, "run", capture)
    data_root, dinof = tmp_path / "new-data", tmp_path / "new-assets" / "dinof.pt"
    train.main(["--stage", "detector", "--config", "configs/detector_from_scratch.yaml",
                "--resume", str(source), "--data-root", str(data_root),
                "--dinof-checkpoint", str(dinof)])
    assert observed["config"]["data"]["root"] == str(data_root)
    assert observed["config"]["model"]["dinof_checkpoint"] == str(dinof)
    assert observed["config"]["optimization"] == saved["config"]["optimization"]
    assert observed["config"]["detector"] == saved["config"]["detector"]


def test_scratch_config_keeps_baseline_budget_and_architecture():
    baseline = deepcopy(load_config("configs/detector.yaml"))
    scratch = load_config("configs/detector_from_scratch.yaml")
    baseline["data"]["root"] = scratch["data"]["root"]
    baseline["model"]["detector_checkpoint"] = None
    assert baseline == scratch
