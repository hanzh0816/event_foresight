"""Causality, paired inventories and sensor contracts for the new research path."""
from io import BytesIO
import pickle

import numpy as np
from PIL import Image
import pytest
import torch

pytest.importorskip("hdf5plugin")
h5py = pytest.importorskip("h5py")

from foresight.data.definitions import DEFAULT_SPLITS, MissingDataError
from foresight.data.event_reader import EventStream
from foresight.data.reader import RGBSequenceReader
from foresight.evaluation import FutureDetectionEvaluator
from foresight.research.data import (
    HorizonDatasetView, ResearchDSECDataset, collate_research, voxelize_interval,
)


TRACK_DTYPE = np.dtype([
    ("t", "<u8"), ("x", "<f4"), ("y", "<f4"), ("w", "<f4"), ("h", "<f4"),
    ("class_id", "u1"), ("class_confidence", "<f4"), ("track_id", "<u4"),
])


@pytest.fixture
def research_recording(tmp_path, monkeypatch):
    for split in ("train", "val", "test"):
        monkeypatch.setitem(DEFAULT_SPLITS, split, (f"{split}_seq",))
    path = tmp_path / "train/train_seq"
    (path / "images/left/distorted").mkdir(parents=True)
    (path / "object_detections/left").mkdir(parents=True)
    (path / "events/left").mkdir(parents=True)
    times = 50_000_000_000 + np.arange(12, dtype=np.int64) * 50_000
    np.savetxt(path / "images/timestamps.txt", times, fmt="%d")
    np.savetxt(path / "images/left/exposure_timestamps.txt",
               np.column_stack((times - 1000, times + 5000)), fmt="%d", delimiter=",")
    tracks = np.array([(t, 10, 20, 30, 40, 2, 1, i) for i, t in enumerate(times)], dtype=TRACK_DTYPE)
    np.save(path / "object_detections/left/tracks.npy", tracks)
    encoded = BytesIO()
    Image.new("RGB", (640, 480), color=(32, 64, 128)).save(encoded, format="PNG")
    for index in range(len(times)):
        (path / f"images/left/distorted/{index:06d}.png").write_bytes(encoded.getvalue())
    local = np.arange(0, 600_001, 1000, dtype=np.int64)
    with h5py.File(path / "events/left/events.h5", "w") as handle:
        handle["events/t"] = local
        handle["events/x"] = (local // 1000 % 640).astype(np.uint16)
        handle["events/y"] = np.zeros(len(local), dtype=np.uint16)
        handle["events/p"] = (local // 1000 % 2).astype(np.uint8)
        handle["t_offset"] = int(times[0])
        handle["ms_to_idx"] = np.searchsorted(local, np.arange(602) * 1000)
    return tmp_path, path, times, tracks


def make_dataset(root, **kwargs):
    options = dict(image_size=(28, 42), event_image_size=(6, 8), context_frames=4,
                   bins=5, alignment="prealigned_event_view",
                   alignment_evidence="Synthetic RGB, boxes and events share native coordinates")
    options.update(kwargs)
    return ResearchDSECDataset(root, **options)


def test_dataset_exposure_cutoff_shapes_and_disjoint_intervals(research_recording, monkeypatch):
    root, _, times, _ = research_recording
    calls = []
    original = EventStream.read_events
    def record(self, start, end):
        calls.append((start, end))
        return original(self, start, end)
    monkeypatch.setattr(EventStream, "read_events", record)
    with make_dataset(root, rollout_steps=4, evaluation_horizons=[1, 2, 4]) as dataset:
        sample = dataset[0]
        assert dataset.windows == [("train_seq", end) for end in range(3, 8)]
        assert sample["history_rgb"].shape == (4, 3, 28, 42)
        assert sample["history_events"].shape == (3, 5, 2, 6, 8)
        assert sample["future_rgb_target"].shape == (4, 3, 28, 42)
        assert sample["future_events_target"].shape == (4, 5, 2, 6, 8)
        bounds = times[:8] + 5000
        assert calls == list(zip(bounds[:-1], bounds[1:]))
        assert sample["meta"]["input_cutoff_us"] == times[3] + 5000
        assert sample["meta"]["history_start_us"] == times[0] - 1000
        assert sample["meta"]["horizon_us"] == [45000, 95000, 145000, 195000]
        assert sample["meta"]["transition_horizon_us"] == [50000, 100000, 150000, 200000]
        assert sample["meta"]["history_dt_us"] == [50000] * 3
        assert sample["meta"]["target_ids"] == [f"train_seq/{i:06d}" for i in range(4, 8)]
        torch.testing.assert_close(sample["targets"][0]["boxes"], torch.tensor([[25/640, 40/480, 30/640, 40/480]]))
        # Half-open consecutive intervals contain exactly 50 distinct events each.
        torch.testing.assert_close(torch.expm1(sample["history_events"]).sum((1, 2, 3, 4)), torch.full((3,), 50.))


def test_inference_reads_only_history_and_has_identical_inventory(research_recording, monkeypatch):
    root, _, times, _ = research_recording
    image_calls, event_calls = [], []
    original_image, original_event = RGBSequenceReader.read_image, EventStream.read_events
    def image_record(self, index):
        image_calls.append(index)
        return original_image(self, index)
    def event_record(self, start, end):
        event_calls.append((start, end))
        return original_event(self, start, end)
    monkeypatch.setattr(RGBSequenceReader, "read_image", image_record)
    monkeypatch.setattr(EventStream, "read_events", event_record)
    with make_dataset(root, rollout_steps=4, future_targets=False) as inference:
        sample = inference[0]
        assert image_calls == [0, 1, 2, 3]
        assert len(event_calls) == 3
        assert max(end for _, end in event_calls) == times[3] + 5000
        assert not any(key.startswith("future_") for key in sample)
        assert len(sample["targets"]) == 4
        with make_dataset(root, rollout_steps=4, future_targets=True) as teacher:
            assert inference.windows == teacher.windows
            target_sample = teacher[0]
            torch.testing.assert_close(sample["history_events"], target_sample["history_events"], rtol=0, atol=0)
        assert "future_events_target" not in collate_research([sample])


def test_rgb_supervision_never_reads_future_event_payload(research_recording, monkeypatch):
    root, _, times, _ = research_recording
    image_calls, event_calls = [], []
    original_image, original_event = RGBSequenceReader.read_image, EventStream.read_events
    def image_record(self, index):
        image_calls.append(index)
        return original_image(self, index)
    def event_record(self, start, end):
        event_calls.append((start, end))
        return original_event(self, start, end)
    monkeypatch.setattr(RGBSequenceReader, "read_image", image_record)
    monkeypatch.setattr(EventStream, "read_events", event_record)
    with make_dataset(root, rollout_steps=4, future_event_targets=False) as rgb_teacher:
        sample = rgb_teacher[0]
        assert image_calls == list(range(8))
        assert event_calls == list(zip(times[:3] + 5000, times[1:4] + 5000))
        assert sample["future_rgb_target"].shape == (4, 3, 28, 42)
        assert "future_events_target" not in sample
        batch = collate_research([sample, sample])
        assert batch["future_rgb_target"].shape == (2, 4, 3, 28, 42)
        assert "future_events_target" not in batch
        with make_dataset(root, rollout_steps=4) as full_teacher:
            assert rgb_teacher.sample_ids == full_teacher.sample_ids
            assert rgb_teacher.excluded_windows == full_teacher.excluded_windows


def test_common_horizon_inventory_and_evaluator_ids(research_recording):
    root, path, _, tracks = research_recording
    np.save(path / "object_detections/left/tracks.npy", tracks[tracks["track_id"] != 6])
    with make_dataset(root, rollout_steps=4, evaluation_horizons=[1, 2, 4]) as dataset:
        assert dataset.windows == [("train_seq", 3), ("train_seq", 6), ("train_seq", 7)]
        for horizon in (1, 2, 4):
            view = HorizonDatasetView(dataset, horizon)
            evaluator = FutureDetectionEvaluator(view)
            assert set(evaluator._inventory) == {
                (f"train_seq:{end}", f"train_seq/{end + horizon:06d}") for _, end in dataset.windows
            }
            assert view.prediction_frames == horizon
        with pytest.raises(ValueError, match="evaluation_horizons"):
            HorizonDatasetView(dataset, 3)


def test_explicit_ids_stride_and_inventory_history(research_recording):
    root = research_recording[0]
    with make_dataset(root, sample_history_frames=6, stride=2, sample_ids=["train_seq:9", "train_seq:5"]) as dataset:
        assert dataset.windows == [("train_seq", 9), ("train_seq", 5)]
        assert dataset.source_indices == [2, 0]
        assert dataset[1]["meta"]["history_indices"] == [2, 3, 4, 5]
    for ids in (["train_seq:2"], ["train_seq:3", "train_seq:3"], "train_seq:3"):
        with pytest.raises(ValueError):
            make_dataset(root, sample_ids=ids)


@pytest.mark.parametrize("options", [
    {"alignment": None}, {"alignment": "resize"}, {"alignment_evidence": "  "},
    {"context_frames": 1}, {"sample_history_frames": 3}, {"future_targets": 1},
    {"future_event_targets": 1},
    {"sequences": ["val_seq"]}, {"split": "unknown"}, {"bins": 0},
    {"rollout_steps": 1, "evaluation_horizons": [2]}, {"evaluation_horizons": []},
])
def test_invalid_protocol_is_rejected(research_recording, options):
    with pytest.raises(ValueError):
        make_dataset(research_recording[0], **options)


def test_missing_exposure_and_bad_native_geometry_rejected(research_recording):
    root, path, _, _ = research_recording
    Image.new("RGB", (1280, 720)).save(path / "images/left/distorted/000000.png")
    with make_dataset(root) as dataset:
        with pytest.raises(MissingDataError, match="480x640"):
            dataset[0]
    (path / "images/left/exposure_timestamps.txt").unlink()
    with pytest.raises(MissingDataError, match="exposure"):
        make_dataset(root)


def test_missing_event_coverage_removes_same_windows_for_teacher_and_inference(research_recording):
    root, path, times, _ = research_recording
    with h5py.File(path / "events/left/events.h5", "r+") as handle:
        handle["t_offset"][()] = times[2]
    with make_dataset(root, future_targets=False) as inference, make_dataset(root) as teacher, \
            make_dataset(root, future_event_targets=False) as rgb_teacher:
        assert inference.windows == teacher.windows == rgb_teacher.windows
        assert inference.windows[0] == ("train_seq", 5)
        assert inference.excluded_windows["coverage"] == 2


def test_noncausal_exposure_is_excluded(research_recording):
    root, path, times, _ = research_recording
    exposures = np.column_stack((times - 1000, times + 5000))
    exposures[3, 1] = times[4]
    np.savetxt(path / "images/left/exposure_timestamps.txt", exposures, fmt="%d", delimiter=",")
    with make_dataset(root) as dataset:
        assert ("train_seq", 3) not in dataset.windows
        assert dataset.excluded_windows["noncausal"] == 1


def test_collate_and_worker_pickle_drop_handles(research_recording):
    with make_dataset(research_recording[0]) as dataset:
        sample = dataset[0]
        batch = collate_research([sample, sample])
        assert batch["history_events"].shape == (2, 3, 5, 2, 6, 8)
        assert batch["future_events_target"].shape == (2, 1, 5, 2, 6, 8)
        assert len(batch["targets"]) == 2 and len(batch["targets"][0]) == 1
        restored = pickle.loads(pickle.dumps(dataset))
        assert not restored._readers and not restored._streams and not restored._handles._handles
        torch.testing.assert_close(restored[0]["history_events"], sample["history_events"])
        restored.close()
    with make_dataset(research_recording[0], future_targets=False) as inference:
        with pytest.raises(ValueError, match="mix"):
            collate_research([sample, inference[0]])


def test_temporal_bins_are_ordered_fixed_log_counts_and_half_open():
    events = {"x": np.array([0, 0, 0, 0]), "y": np.zeros(4, dtype=np.int64),
              "p": np.array([0, 1, 1, 1]), "t": np.array([100, 109, 110, 119])}
    voxel = voxelize_interval(events, 100, 120, bins=2, image_size=(1, 1))
    torch.testing.assert_close(torch.expm1(voxel[:, :, 0, 0]), torch.tensor([[1., 1.], [0., 2.]]))
    # Counts, not averages of log counts, define a time-collapsed control.
    torch.testing.assert_close(torch.expm1(voxel).sum(0), torch.tensor([[[1.]], [[3.]]]))
    events["t"][-1] = 120
    with pytest.raises(ValueError, match="half-open"):
        voxelize_interval(events, 100, 120)
