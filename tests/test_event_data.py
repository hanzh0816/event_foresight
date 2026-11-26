"""Causal intervals, common image geometry, augmentation and worker-safe reads."""

from io import BytesIO
import pickle

import h5py
import numpy as np
from PIL import Image
import pytest
import torch

from foresight.data import EventRGBDataset, RGBDataset, collate_rgb, dataset_from_config
from foresight.data.definitions import DEFAULT_SPLITS, MissingDataError, WindowUnavailableError
from foresight.data.events import events_to_voxel, reverse_event_bins
from foresight.data.event_reader import EventStream


TRACK_DTYPE = np.dtype([
    ("t", "<u8"), ("x", "<f4"), ("y", "<f4"), ("w", "<f4"), ("h", "<f4"),
    ("class_id", "u1"), ("class_confidence", "<f4"), ("track_id", "<u4"),
])


@pytest.fixture
def event_fixture(tmp_path, monkeypatch):
    monkeypatch.setitem(DEFAULT_SPLITS, "train", ("seq",))
    path = tmp_path / "train/seq"
    for folder in ("images/left/distorted", "object_detections/left", "events/left"):
        (path / folder).mkdir(parents=True)
    times = 50_000_000_000 + np.arange(45, dtype=np.int64) * 50_000
    np.savetxt(path / "images/timestamps.txt", times, fmt="%d")
    # Deliberately shifted exposures: v3.1 windows follow tau_t, not exposure end.
    np.savetxt(path / "images/left/exposure_timestamps.txt",
               np.stack((times - 500, times + 5000), axis=1), delimiter=",", fmt="%d")
    tracks = np.array([(times[i], 10, 20, 30, 40, 2, 1., i) for i in range(40, 45)], dtype=TRACK_DTYPE)
    np.save(path / "object_detections/left/tracks.npy", tracks)
    pixels = np.zeros((480, 640, 3), dtype=np.uint8)
    pixels[:, :320, 0] = 240
    image = BytesIO()
    Image.fromarray(pixels).save(image, format="PNG")
    for index in range(45):
        (path / f"images/left/distorted/{index:06d}.png").write_bytes(image.getvalue())
    offset = times[0] - 50_000
    local = np.arange(0, 2_300_001, 5000, dtype=np.int64)
    with h5py.File(path / "events/left/events.h5", "w") as handle:
        handle["events/t"] = local
        handle["events/x"] = np.full(len(local), 40, np.uint16)
        handle["events/y"] = np.full(len(local), 40, np.uint16)
        handle["events/p"] = (np.arange(len(local)) % 2).astype(np.uint8)
        handle["t_offset"] = offset
        handle["ms_to_idx"] = np.searchsorted(local, np.arange(2302) * 1000)
    return tmp_path, path, times


def small_dataset(root, **kwargs):
    return EventRGBDataset(root, image_size=(28, 56), event_size=(32, 64), **kwargs)


def test_voxel_bins_are_half_open_and_preserve_polarity():
    events = dict(t=np.array([99, 100, 104, 105, 149, 150]), x=np.zeros(6, dtype=int),
                  y=np.zeros(6, dtype=int), p=np.array([1, 0, 0, 1, 1, 1]))
    voxels, mask = events_to_voxel(events, 100, 150, (2, 4), bins=10,
                                   sensor_size=(2, 4), grid_size=(1, 2))
    assert torch.count_nonzero(voxels).item() == 3
    assert voxels[0, 0, 0].item() == pytest.approx(np.log1p(2))
    assert voxels[11, 0, 0].item() == pytest.approx(np.log1p(1))
    assert voxels[19, 0, 0].item() == pytest.approx(np.log1p(1))
    assert mask.tolist() == [True, False]


def test_native_sensor_geometry_is_resized_like_rgb():
    events = dict(t=np.array([0, 1]), x=np.array([0, 3]), y=np.array([0, 1]), p=np.array([0, 1]))
    voxels, mask = events_to_voxel(events, 0, 10, (4, 8), bins=5,
                                   sensor_size=(2, 4), grid_size=(2, 4))
    source = torch.zeros(10, 2, 4)
    source[0, 0, 0] = np.log(2)
    source[5, 1, 3] = np.log(2)
    expected = torch.nn.functional.interpolate(source[None], size=(4, 8), mode="bilinear", align_corners=False)[0]
    torch.testing.assert_close(voxels, expected)
    assert mask.reshape(2, 4).tolist() == [[True, False, False, False], [False, False, False, True]]


def test_activity_counts_raw_events_before_log_or_resize():
    events = dict(t=np.array([0, 1, 2]), x=np.array([0, 0, 3]), y=np.array([0, 0, 1]), p=np.array([0, 1, 1]))
    _, mask = events_to_voxel(events, 0, 10, (4, 8), bins=5, sensor_size=(2, 4),
                             grid_size=(1, 2), activity_threshold=2)
    assert mask.tolist() == [True, False]
    empty = {key: np.zeros(0, dtype=np.int64) for key in ("t", "x", "y", "p")}
    voxel, mask = events_to_voxel(empty, 0, 10, (4, 8), sensor_size=(2, 4), grid_size=(1, 2))
    assert not voxel.any() and not mask.any()


def test_bin_reverse_keeps_polarities_and_handles_batch_four():
    voxels = torch.arange(4 * 3 * 20 * 2 * 4).reshape(4, 3, 20, 2, 4)
    reversed_voxels = reverse_event_bins(voxels)
    torch.testing.assert_close(reversed_voxels[:, :, :10], voxels[:, :, :10].flip(2))
    torch.testing.assert_close(reversed_voxels[:, :, 10:], voxels[:, :, 10:].flip(2))
    torch.testing.assert_close(reverse_event_bins(reversed_voxels), voxels)


def test_event_dataset_matches_rgb_inventory_and_timestamp_cutoff(event_fixture):
    root, _, times = event_fixture
    with small_dataset(root) as dataset, RGBDataset(root, image_size=(28, 56)) as rgb:
        assert dataset.windows == rgb.windows
        sample = dataset[0]
        assert sample["history_events"].shape == (4, 20, 32, 64)
        assert sample["event_activity"].shape == (4, 8)
        assert sample["meta"]["coordinate_frame"] == "left_distorted_event"
        assert sample["meta"]["event_end_us"] == times[36:40].tolist()
        assert sample["meta"]["event_start_us"] == (times[36:40] - 50000).tolist()
        assert sample["meta"]["input_cutoff_us"] == times[39]
        assert sample["meta"]["target_timestamp_us"] == times[40]
        torch.testing.assert_close(sample["history_rgb"], rgb[0]["history_rgb"])


def test_h5_slice_excludes_endpoint_exactly(event_fixture):
    _, path, times = event_fixture
    stream = EventStream(path / "events/left/events.h5")
    events = stream.read_events(int(times[39] - 50000), int(times[39]))
    assert len(events["t"]) == 10
    assert events["t"][0] == times[39] - 50000
    assert events["t"][-1] == times[39] - 5000
    stream.close()


def test_future_event_mutation_does_not_change_history(event_fixture):
    root, path, times = event_fixture
    with small_dataset(root) as dataset:
        before = dataset[0]
    with h5py.File(path / "events/left/events.h5", "r+") as handle:
        start = np.searchsorted(handle["events/t"][:] + handle["t_offset"][()], times[39])
        handle["events/x"][start:] = 600
        handle["events/p"][start:] = 1
    with small_dataset(root) as dataset:
        after = dataset[0]
    torch.testing.assert_close(before["history_events"], after["history_events"], rtol=0, atol=0)
    assert torch.equal(before["event_activity"], after["event_activity"])


def test_flip_is_synchronized_and_never_negates_polarity(event_fixture):
    root = event_fixture[0]
    with small_dataset(root, future_rgb=True) as dataset:
        plain = dataset[0]
    with small_dataset(root, future_rgb=True, horizontal_flip_prob=1.) as dataset:
        flipped = dataset[0]
    for key in ("history_rgb", "history_events", "future_rgb_target"):
        torch.testing.assert_close(flipped[key], plain[key].flip(-1))
    assert torch.equal(flipped["event_activity"], plain["event_activity"].reshape(4, 2, 4).flip(-1).flatten(1))
    torch.testing.assert_close(flipped["target"]["boxes"][:, 0], 1 - plain["target"]["boxes"][:, 0])
    torch.testing.assert_close(flipped["target"]["boxes"][:, 1:], plain["target"]["boxes"][:, 1:])
    assert torch.equal(flipped["target"]["labels"], plain["target"]["labels"])
    assert flipped["meta"]["horizontal_flipped"]
    with RGBDataset(root, image_size=(28, 56), horizontal_flip_prob=1.) as rgb:
        torch.testing.assert_close(rgb[0]["history_rgb"], flipped["history_rgb"])


def test_batch_four_and_pickle_reopen_handles(event_fixture):
    root = event_fixture[0]
    with small_dataset(root, future_rgb=True) as dataset:
        samples = [dataset[index] for index in range(4)]
        batch = collate_rgb(samples)
        assert batch["history_events"].shape == (4, 4, 20, 32, 64)
        assert batch["event_activity"].shape == (4, 4, 8)
        assert len(batch["targets"]) == 4
        restored = pickle.loads(pickle.dumps(dataset))
        assert not restored._readers and not restored._event_streams
        assert not restored._event_handles._handles
        torch.testing.assert_close(restored[0]["history_events"], samples[0]["history_events"])
        restored.close()


def test_missing_event_coverage_fails_without_silently_changing_inventory(event_fixture):
    root, path, times = event_fixture
    with h5py.File(path / "events/left/events.h5", "r+") as handle:
        handle["t_offset"][()] = int(times[39])
    with pytest.raises(WindowUnavailableError, match="RGB inventory is unchanged"):
        small_dataset(root)
    with RGBDataset(root, image_size=(28, 56)) as rgb:
        assert len(rgb) == 5


def test_missing_event_file_and_unremapped_rgb_fail_clearly(event_fixture):
    root, path, _ = event_fixture
    Image.new("RGB", (1440, 1080)).save(path / "images/left/distorted/000036.png")
    with small_dataset(root) as dataset:
        with pytest.raises(MissingDataError, match="calibrated/remapped"):
            dataset[0]
    (path / "events/left/events.h5").unlink()
    with pytest.raises(MissingDataError, match="Missing DSEC events"):
        small_dataset(root)


@pytest.mark.parametrize("horizon", [1, 2, 3, 4])
def test_rollout_dataset_keeps_fixed_observed_history(event_fixture, horizon):
    root, _, times = event_fixture
    with small_dataset(root, context_frames=7, prediction_frames=horizon, future_rgb=True) as dataset:
        sample = dataset[0]
        assert sample["history_rgb"].shape[0] == 7
        assert sample["history_events"].shape[0] == 7
        assert sample["meta"]["event_end_us"][-1] == times[39]
        assert sample["meta"]["target_timestamp_us"] == times[39 + horizon]
        assert sample["future_rgb_target"].shape == (3, 28, 56)


def test_factory_reads_model_event_input_size_and_uses_shared_augmentation(event_fixture):
    config = {"data": {"root": str(event_fixture[0]), "image_size": [28, 56], "horizontal_flip_prob": 1.},
              "model": {"event": {"enabled": True, "input_size": [32, 64]}}}
    with dataset_from_config(config, "train", future_rgb=True, prediction_frames=4) as dataset:
        assert isinstance(dataset, EventRGBDataset)
        assert dataset[0]["history_events"].shape[-2:] == (32, 64)
        assert dataset[0]["meta"]["horizontal_flipped"]
    config["model"]["event"]["enabled"] = False
    with dataset_from_config(config, "train") as dataset:
        assert type(dataset) is RGBDataset
        assert dataset[0]["meta"]["horizontal_flipped"]


def test_factory_respects_split_selection_and_explicit_overrides(event_fixture, monkeypatch):
    monkeypatch.setitem(DEFAULT_SPLITS, "val", ("seq",))
    config = {"data": {"root": str(event_fixture[0]), "image_size": [28, 56],
                        "stride": 3, "val_stride": 2, "val_sequences": ["seq"],
                        "val_sample_ids": ["seq:43", "seq:39"], "horizontal_flip_prob": 1.},
              "model": {"event": {"enabled": False}}}
    with dataset_from_config(config, "val") as dataset:
        assert dataset.windows == [("seq", 43), ("seq", 39)]
        assert dataset.stride == 2
        assert dataset.selected_sequences == ("seq",)
        assert dataset.horizontal_flip_prob == 0
    with dataset_from_config(config, "val", sample_ids=None, stride=1) as dataset:
        assert dataset.windows == [("seq", end) for end in range(39, 44)]


def test_spawn_workers_return_batch_four_after_parent_opened_hdf5(event_fixture):
    root = event_fixture[0]
    with small_dataset(root) as dataset:
        expected = dataset[0]["history_events"]
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=2,
                                             multiprocessing_context="spawn", prefetch_factor=1,
                                             collate_fn=collate_rgb)
        batch = next(iter(loader))
        assert batch["history_events"].shape == (4, 4, 20, 32, 64)
        torch.testing.assert_close(batch["history_events"][0], expected)


@pytest.mark.parametrize("corruption", ["field", "empty", "length", "index"])
def test_malformed_event_stream_fails_with_path(event_fixture, corruption):
    root, path, _ = event_fixture
    with h5py.File(path / "events/left/events.h5", "r+") as handle:
        if corruption == "field":
            del handle["events/p"]
        elif corruption == "empty":
            del handle["events/t"]
            handle["events/t"] = np.array([], dtype=np.int64)
        elif corruption == "length":
            del handle["events/x"]
            handle["events/x"] = np.array([0], dtype=np.uint16)
        else:
            del handle["ms_to_idx"]
            handle["ms_to_idx"] = np.array([], dtype=np.int64)
    with pytest.raises(MissingDataError, match="events.h5"):
        small_dataset(root)
