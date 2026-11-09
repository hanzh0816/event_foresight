"""Data readiness checks against a small, complete DSEC-shaped recording."""

from io import BytesIO
import json

import numpy as np
from PIL import Image
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("hdf5plugin")

from foresight.data.definitions import DEFAULT_SPLITS
from scripts import check_dsec


TRACK_DTYPE = np.dtype([
    ("t", "<u8"), ("x", "<f4"), ("y", "<f4"), ("w", "<f4"), ("h", "<f4"),
    ("class_id", "u1"), ("class_confidence", "<f4"), ("track_id", "<u4"),
])


@pytest.fixture
def complete_dsec(tmp_path, monkeypatch):
    """Keep the real 40-frame window rules and train/val directory mapping."""
    root = tmp_path / "DSEC"
    names = {split: f"{split}_sequence" for split in ("train", "val", "test")}
    for split, name in names.items():
        monkeypatch.setitem(DEFAULT_SPLITS, split, (name,))
    times = 50_000_000_000 + np.arange(45, dtype=np.int64) * 50_000
    tracks = np.array([
        (times[i], 10, 20, 30, 40, 2, 1.0, i) for i in range(40, 45)
    ], dtype=TRACK_DTYPE)
    encoded = BytesIO()
    Image.new("RGB", (640, 480), color=(32, 64, 128)).save(encoded, format="PNG")
    paths = {}
    for split, name in names.items():
        path = root / ("test" if split == "test" else "train") / name
        paths[split] = path
        (path / "images/left/distorted").mkdir(parents=True)
        (path / "object_detections/left").mkdir(parents=True)
        (path / "events/left").mkdir(parents=True)
        np.savetxt(path / "images/timestamps.txt", times, fmt="%d")
        np.savetxt(path / "images/left/exposure_timestamps.txt",
                   np.column_stack((times - 1000, times + 5000)), fmt="%d", delimiter=",")
        np.save(path / "object_detections/left/tracks.npy", tracks)
        for frame in range(len(times)):
            (path / f"images/left/distorted/{frame:06d}.png").write_bytes(encoded.getvalue())
        local_times = np.arange(0, 2_500_001, 10_000, dtype=np.int64)
        with h5py.File(path / "events/left/events.h5", "w") as handle:
            handle["events/t"] = local_times
            handle["events/x"] = np.zeros(len(local_times), dtype=np.uint16)
            handle["events/y"] = np.zeros(len(local_times), dtype=np.uint16)
            handle["events/p"] = np.ones(len(local_times), dtype=np.uint8)
            handle["t_offset"] = int(times[0]) - 200_000
            handle["ms_to_idx"] = np.searchsorted(local_times, np.arange(2502) * 1000)
    return root, paths, times, tracks


def errors_for(report, split):
    return "\n".join(error for seq in report["splits"][split]["sequences"]
                     for error in seq["errors"])


@pytest.mark.parametrize("event_window_us,deep", [(50_000, False), (200_000, False), (50_000, True)])
def test_complete_inventory_passes_all_stages(complete_dsec, event_window_us, deep):
    root, paths, _, _ = complete_dsec
    report = check_dsec.check_dsec(root, event_window_us=event_window_us, deep=deep)
    assert report["passed"], report
    assert report["stages"] == {"train": True, "eval": True, "test": True}
    assert not (root / "val").exists()
    assert paths["val"].parent == root / "train"
    for split in ("train", "val", "test"):
        assert report["splits"][split]["passed"]
        assert report["splits"][split]["windows"] == 5
        assert not errors_for(report, split)


def test_missing_future_rgb_is_detected_before_training(complete_dsec):
    root, paths, _, _ = complete_dsec
    (paths["train"] / "images/left/distorted/000044.png").unlink()
    report = check_dsec.check_dsec(root)
    assert not report["passed"]
    assert not report["stages"]["train"]
    assert "000044.png" in errors_for(report, "train")


def test_missing_test_annotations_preserves_train_and_eval_readiness(complete_dsec):
    root, paths, _, _ = complete_dsec
    (paths["test"] / "object_detections/left/tracks.npy").unlink()
    report = check_dsec.check_dsec(root)
    assert not report["passed"]
    assert report["stages"] == {"train": True, "eval": True, "test": False}
    assert "tracks.npy" in errors_for(report, "test")


@pytest.mark.parametrize("corruption", ["class", "timestamps"])
def test_invalid_annotations_or_timestamps_fail(complete_dsec, corruption):
    root, paths, times, tracks = complete_dsec
    if corruption == "class":
        tracks["class_id"][0] = 8
        np.save(paths["val"] / "object_detections/left/tracks.npy", tracks)
    else:
        times[20] = times[19]
        np.savetxt(paths["val"] / "images/timestamps.txt", times, fmt="%d")
    report = check_dsec.check_dsec(root)
    assert not report["passed"]
    assert not report["splits"]["val"]["passed"]
    assert not report["stages"]["eval"]
    assert errors_for(report, "val")


@pytest.mark.parametrize("corruption", ["coverage", "field"])
def test_event_stream_must_cover_windows_and_have_required_fields(complete_dsec, corruption):
    root, paths, times, _ = complete_dsec
    with h5py.File(paths["test"] / "events/left/events.h5", "r+") as handle:
        if corruption == "coverage":
            handle["t_offset"][()] = int(times[40])
        else:
            del handle["events/p"]
    report = check_dsec.check_dsec(root)
    assert not report["passed"]
    assert not report["stages"]["test"]
    assert errors_for(report, "test")


def test_rgb_only_does_not_require_events_or_exposure(complete_dsec):
    root, paths, _, _ = complete_dsec
    for path in paths.values():
        (path / "events/left/events.h5").unlink()
        (path / "images/left/exposure_timestamps.txt").unlink()
    assert check_dsec.check_dsec(root, rgb_only=True)["passed"]
    assert not check_dsec.check_dsec(root)["passed"]


def test_deep_check_decodes_middle_image(complete_dsec):
    root, paths, _, _ = complete_dsec
    (paths["train"] / "images/left/distorted/000010.png").write_bytes(b"not a PNG")
    assert check_dsec.check_dsec(root, rgb_only=True)["passed"]
    report = check_dsec.check_dsec(root, rgb_only=True, deep=True)
    assert not report["passed"]
    assert "000010.png" in errors_for(report, "train")


def test_cli_writes_report_and_returns_failure_code(complete_dsec, tmp_path):
    root, paths, _, _ = complete_dsec
    output = tmp_path / "reports/check.json"
    assert check_dsec.main(["--data-root", str(root), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["passed"]
    (paths["test"] / "object_detections/left/tracks.npy").unlink()
    assert check_dsec.main(["--data-root", str(root), "--output", str(output)]) == 1
    assert not json.loads(output.read_text())["passed"]


def test_cli_rejects_output_inside_dataset(complete_dsec):
    root, _, _, _ = complete_dsec
    with pytest.raises(SystemExit) as exc:
        check_dsec.main(["--data-root", str(root), "--output", str(root / "report.json")])
    assert exc.value.code == 2
    assert not (root / "report.json").exists()
