"""Real checkpoint + synthetic DSEC + shared COCO readout integration."""

from copy import deepcopy
import json

import pytest
import torch

from test_research_data import research_recording
from test_research_evaluate import PCA, TinyDetector
from test_research_training import TinyForecaster, save_task, tiny_config
from foresight.research import diagnose_transition as diagnostic
from foresight.research.compare import compare
from foresight.research.task import load_task


class DiagnosticForecaster(TinyForecaster):
    image_size = (28, 42)
    history_length = 4

    def __init__(self):
        super().__init__()
        self.pca = PCA()


def checkpoint_on_recording(root, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    asset = tmp_path / "asset.pt"
    asset.write_bytes(b"synthetic-export")
    config = tiny_config("transition")
    config["data"].update(root=str(root), image_size=[28, 42], context_frames=4, bins=5,
                          batch_size=1, event_image_size=[6, 8])
    config["model"].update(dinof_checkpoint=str(asset), detector_checkpoint=str(asset))
    teacher = load_task(config, forecaster=DiagnosticForecaster())
    torch.nn.init.normal_(teacher.model.scene_transition.output.weight, std=0.1)
    teacher_path = tmp_path / "teacher.ckpt"
    save_task(teacher, teacher_path)
    config = deepcopy(config)
    config["stage"] = "motion"
    config["model"].update(mode="explicit", teacher_checkpoint=str(teacher_path))
    config["loss"]["motion_weight"] = 1.0
    task = load_task(config, forecaster=DiagnosticForecaster()).eval()
    torch.nn.init.normal_(task.model.motion_head.weight, std=0.1)
    path = tmp_path / "motion.ckpt"
    save_task(task, path)
    teacher_path.unlink()
    original = diagnostic.load_diagnostic_task
    monkeypatch.setattr(diagnostic, "load_diagnostic_task",
                        lambda *args, **kwargs: original(*args, forecaster=DiagnosticForecaster(), **kwargs))
    monkeypatch.setattr(diagnostic, "load_detector", lambda *_: TinyDetector())
    return path


@pytest.mark.parametrize("steps", [1, 4])
def test_single_pass_suite_and_checkpoint_preservation(research_recording, tmp_path, monkeypatch, steps):
    checkpoint = checkpoint_on_recording(research_recording[0], tmp_path, monkeypatch)
    folder = tmp_path / "diagnosis"
    horizons = [1] if steps == 1 else [1, 2, 4]
    report = diagnostic.diagnose(checkpoint, output_dir=folder, split="train", steps=steps,
        horizons=horizons, batch_size=2, num_workers=0, device="cpu", max_samples=2,
        save_predictions=True)
    assert report["evaluation_kind"] == "transition_bottleneck_diagnostic"
    assert len(report["routes"]) == (6 if steps == 1 else 7)
    assert report["checkpoint_config"]["data"]["rollout_steps"] == 1
    assert report["checkpoint_config"]["data"]["batch_size"] == 1
    assert report["runtime"]["batch_size"] == 2
    assert report["provenance"]["teacher_state_sha256"]
    assert report["provenance"]["active_transition_state_sha256"] == report["provenance"]["frozen_transition_state_sha256"]
    assert report["smoke_test"]
    assert len(report["sample_ids"]) == 2
    assert report["sample_selection"]["evaluated"] == 2
    assert report["sample_selection"]["full_inventory_size"] >= 2
    for name, route in report["routes"].items():
        assert list(route["horizons"]) == list(map(str, horizons))
        for h, row in route["horizons"].items():
            assert row["metrics"]["AP"] > .99
            assert row["metadata"]["num_samples"] == 2
            assert row["metadata"]["cutoff_to_target_ms"]["mean"] == int(h)*50-5
            assert row["features"]["all_l1"] is not None
            assert row["features"]["fg_l1"] is not None
            assert row["features"]["bg_l1"] is not None
    assert report["routes"]["true_future_rgb"]["horizons"]["1"]["features"]["all_l1"] == 0
    assert report["routes"]["true_future_event"]["horizons"]["1"]["features"]["motion_l1"] == 0
    assert report["routes"]["predicted_motion"]["horizons"]["1"]["features"]["motion_l1"] > 0
    assert report["routes"]["true_future_event"]["future_event"] is True
    assert report["routes"]["true_future_event"]["future_rgb"] is False
    assert report["routes"]["predicted_motion"]["future_event"] is False
    saved = json.loads((folder / "report.json").read_text())
    assert saved["sample_timing_sha256"] == report["sample_timing_sha256"]
    assert "Foreground L1" in (folder / "summary.md").read_text()
    assert (folder / "predictions.json").exists()
    with pytest.raises(ValueError, match="Privileged"):
        compare([report, report])
    with pytest.raises(FileExistsError):
        diagnostic.diagnose(checkpoint, output_dir=folder, device="cpu")
    # Rerun with a different execution batch size and exactly the saved sample
    # manifest. Both retain the same scientific inputs and loaded teacher.
    replay = diagnostic.diagnose(checkpoint, output_dir=tmp_path / "replay", split="train",
        steps=steps, horizons=horizons, batch_size=1, num_workers=0, device="cpu",
        manifest=folder / "manifest.json")
    assert replay["sample_digest"] == report["sample_digest"]
    assert replay["sample_timing_sha256"] == report["sample_timing_sha256"]
    assert replay["provenance"]["teacher_state_sha256"] == report["provenance"]["teacher_state_sha256"]
    assert replay["sample_selection"]["selected_before_limit"] == 2
    assert replay["sample_selection"]["covers_full_inventory"] == report["sample_selection"]["covers_full_inventory"]


def test_manifest_time_mismatch_cannot_publish_completed_report(research_recording, tmp_path, monkeypatch):
    checkpoint = checkpoint_on_recording(research_recording[0], tmp_path, monkeypatch)
    first = diagnostic.diagnose(checkpoint, output_dir=tmp_path / "first", split="train",
        batch_size=1, num_workers=0, device="cpu", max_samples=1)
    manifest = {key: first[key] for key in ("split", "sample_ids", "samples")}
    manifest["samples"][0]["input_cutoff_us"] += 1
    path = tmp_path / "bad_manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="time boundary mismatch"):
        diagnostic.diagnose(checkpoint, output_dir=tmp_path / "bad", split="train",
            batch_size=1, num_workers=0, device="cpu", manifest=path)
    assert not (tmp_path / "bad/report.json").exists()


def test_feature_errors_keep_region_and_motion_reductions_distinct():
    accumulator = diagnostic.FeatureErrors()
    target = torch.zeros(1, 3, 2)
    scene = torch.tensor([[[0., 2.], [2., 4.], [4., 6.]]])
    mask = torch.tensor([[True, False, False]])
    accumulator.update(scene, target, mask, torch.ones(1, 3, 4), torch.zeros(1, 3, 4))
    values = accumulator.compute()
    assert values["all_l1"] == 3
    assert values["fg_l1"] == 1
    assert values["bg_l1"] == 4
    assert values["motion_l1"] == 1
    assert values["foreground_patch_fraction"] == 1/3
    empty_foreground = diagnostic.FeatureErrors()
    empty_foreground.update(scene, target, torch.zeros_like(mask))
    assert empty_foreground.compute()["fg_l1"] is None


@pytest.mark.parametrize("kwargs", [dict(steps=0), dict(batch_size=0), dict(num_workers=-1),
    dict(max_samples=0), dict(steps=2, horizons=[3]), dict(steps=2, horizons=[1, 1])])
def test_invalid_execution_options_fail_before_asset_loading(tmp_path, kwargs):
    with pytest.raises(ValueError):
        diagnostic.diagnose("missing.ckpt", output_dir=tmp_path / "result", device="cpu", **kwargs)
