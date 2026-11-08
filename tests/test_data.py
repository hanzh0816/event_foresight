import importlib
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from foresight.data import RGBDataset, RGBSequenceReader, collate_rgb, preprocess_rgb
from foresight.data.definitions import DEFAULT_SPLITS

TRACK_DTYPE = np.dtype([("t", "<u8"), ("x", "<f4"), ("y", "<f4"),
                        ("w", "<f4"), ("h", "<f4"), ("class_id", "u1"),
                        ("class_confidence", "<f4"), ("track_id", "<u4")])


@pytest.fixture
def rgb_fixture(tmp_path, monkeypatch):
    monkeypatch.setitem(DEFAULT_SPLITS, "train", ("seq",))
    path = tmp_path / "train/seq"
    (path / "images/left/distorted").mkdir(parents=True)
    (path / "object_detections/left").mkdir(parents=True)
    times = 50_000_000_000 + np.arange(45, dtype=np.int64) * 50000
    np.savetxt(path / "images/timestamps.txt", times, fmt="%d")
    # Only future frames have labels: historical-label availability is irrelevant.
    tracks = np.array([(times[i], 10, 20, 30, 40, 2, 1, i) for i in range(40, 45)], dtype=TRACK_DTYPE)
    np.save(path / "object_detections/left/tracks.npy", tracks)
    for i in range(45):
        pixels = np.full((28, 42, 3), i, np.uint8)
        Image.fromarray(pixels).save(path / f"images/left/distorted/{i:06d}.png")
    return tmp_path, path, times, tracks


def test_inventory_first_window_stride_and_rgb_only_files(rgb_fixture):
    root, path, _, _ = rgb_fixture
    with RGBDataset(root, image_size=(28, 42)) as dataset:
        assert dataset.windows == [("seq", i) for i in range(39, 44)]
        assert dataset.history_frames == 40 and dataset.context_frames == 4
        assert dataset.prediction_frames == 1 and dataset.full_inventory_size == 5
        assert not (path / "events").exists()
    with RGBDataset(root, image_size=(28, 42), stride=2) as dataset:
        assert dataset.windows == [("seq", i) for i in (39, 41, 43)]


@pytest.mark.parametrize("gap,expected", [(1000, 5), (1001, 0)])
def test_full_outer_window_checks_every_timestamp_interval(rgb_fixture, gap, expected):
    root, path, times, tracks = rgb_fixture
    times[20:] += gap
    tracks["t"] = times[40:]
    np.savetxt(path / "images/timestamps.txt", times, fmt="%d")
    np.save(path / "object_detections/left/tracks.npy", tracks)
    with RGBDataset(root, image_size=(28, 42)) as dataset:
        assert len(dataset) == expected


def test_empty_and_clipped_degenerate_future_targets_are_excluded(rgb_fixture):
    root, path, _, tracks = rgb_fixture
    tracks = tracks[tracks["track_id"] != 42]
    tracks["x"][tracks["track_id"] == 43] = 700
    np.save(path / "object_detections/left/tracks.npy", tracks)
    with RGBDataset(root, image_size=(28, 42)) as dataset:
        assert dataset.windows == [("seq", i) for i in (39, 40, 43)]


def test_selection_preserves_requested_order_and_rejects_duplicates_or_missing(rgb_fixture):
    root = rgb_fixture[0]
    with RGBDataset(root, image_size=(28, 42), sample_ids=["seq:43", "seq:39"]) as dataset:
        assert dataset.windows == [("seq", 43), ("seq", 39)]
        assert dataset.source_indices == [4, 0]
    for selection in (["seq:39", "seq:39"], ["seq:38"], "seq:39"):
        with pytest.raises(ValueError):
            RGBDataset(root, image_size=(28, 42), sample_ids=selection)


def test_default_has_no_future_rgb_reads_and_explicit_target_is_separate(rgb_fixture, monkeypatch):
    root = rgb_fixture[0]
    calls = []
    original = RGBSequenceReader.read_image
    def record(self, index):
        calls.append(index)
        return original(self, index)
    monkeypatch.setattr(RGBSequenceReader, "read_image", record)
    with RGBDataset(root, image_size=(28, 42)) as dataset:
        sample = dataset[0]
        assert calls == [36, 37, 38, 39]
        assert "future_rgb_target" not in sample
        assert sample["meta"]["target_id"] == "seq/000040"
        assert sample["history_rgb"].shape == (4, 3, 28, 42)
        assert sample["target"]["labels"].tolist() == [2]
        torch.testing.assert_close(sample["target"]["boxes"], torch.tensor([[25/640, 40/480, 30/640, 40/480]]))
        restored = pickle.loads(pickle.dumps(dataset))
        assert not restored._readers
        restored.close()
    calls.clear()
    with RGBDataset(root, image_size=(28, 42), future_rgb=True) as dataset:
        training = dataset[0]
        assert calls == [36, 37, 38, 39, 40]
        torch.testing.assert_close(training["history_rgb"], sample["history_rgb"], rtol=0, atol=0)
        batch = collate_rgb([training, training])
        assert batch["future_rgb_target"].shape == (2, 3, 28, 42)
        assert batch["history_rgb"].shape == (2, 4, 3, 28, 42)
        assert len(batch["targets"]) == 2
    assert "future_rgb_target" not in collate_rgb([sample])
    with pytest.raises(ValueError, match="mix"):
        collate_rgb([sample, training])


def test_rgb_preprocessing_matches_pil_reference_exactly():
    pixels = np.random.default_rng(7).integers(0, 256, (17, 23, 3), dtype=np.uint8)
    expected = np.array(Image.fromarray(pixels).resize((42, 28), Image.Resampling.BILINEAR), copy=True)
    expected = torch.from_numpy(expected).permute(2, 0, 1).float() / 255
    expected = (expected - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
    torch.testing.assert_close(preprocess_rgb(pixels, (28, 42)), expected, rtol=0, atol=0)


def test_reader_clips_original_coordinates_and_drops_zero_area(rgb_fixture):
    root, path, times, _ = rgb_fixture
    tracks = np.array([(times[40], -10, -20, 1000, 1000, 0, .7, 1),
                       (times[40], 700, 1, 2, 2, 2, 1, 2)], dtype=TRACK_DTYPE)
    np.save(path / "object_detections/left/tracks.npy", tracks)
    with RGBSequenceReader(path) as reader:
        target = reader.read_target(40)
        assert target.boxes.tolist() == [[0, 0, 639, 479]]
        assert target.class_ids.tolist() == [0]
        assert target.track_ids.tolist() == [1]


@pytest.mark.parametrize("kwargs", [{"context_frames": 41}, {"stride": 0}, {"tolerance_us": -1},
                                    {"image_size": (29, 42)}, {"split": "unknown"},
                                    {"sequences": ["outside"]}, {"future_rgb": 1}])
def test_invalid_protocol_is_rejected_before_loading(rgb_fixture, kwargs):
    with pytest.raises(ValueError):
        RGBDataset(rgb_fixture[0], **kwargs)


def test_import_from_unrelated_cwd_does_not_load_sensor_or_old_project_modules(tmp_path):
    package_root = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=package_root, PYTHONDONTWRITEBYTECODE="1")
    code = "import sys; from foresight.data import RGBDataset; from foresight.evaluation import FutureDetectionEvaluator; assert not any(n.startswith(('dsec_data', 'h5py', 'hdf5plugin', 'dinof_future_detect')) for n in sys.modules)"
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, check=True)
