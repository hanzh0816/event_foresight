"""Intervention and subset contracts; these tests are intended for the GPU host suite."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from foresight.evaluation.diagnostics import mismatched_donors, motion_subset
from foresight.inference.pipeline import ForecastDetector
from foresight.timing import ForecastTiming


class DonorInventory:
    windows = [(name, end) for name in ("a", "b", "c") for end in range(39, 44)]
    sample_ids = [f"{name}:{end}" for name, end in windows]


def test_donors_are_cross_sequence_and_independent_of_recipient_order_or_batch():
    dataset = DonorInventory()
    recipients = ["a:39", "b:42", "a:43", "c:41", "b:40"]
    expected = mismatched_donors(dataset, recipients)
    assert expected == mismatched_donors(dataset, list(reversed(recipients)))
    for batch_size in (1, 2, 4):
        actual = {}
        for start in range(0, len(recipients), batch_size):
            actual.update(mismatched_donors(dataset, recipients[start:start + batch_size]))
        assert actual == expected
    for recipient, index in expected.items():
        assert dataset.windows[index][0] != recipient.rsplit(":", 1)[0]


def test_mismatch_rejects_a_single_sequence_inventory():
    with pytest.raises(ValueError, match="at least two sequences"):
        mismatched_donors(SimpleNamespace(windows=[("a", 39), ("a", 40)]), ["a:39"])


TRACK_DTYPE = np.dtype([("x", "<f8"), ("y", "<f8"), ("w", "<f8"), ("h", "<f8"), ("track_id", "<i8")])


class ObservedTrackReader:
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.calls = []

    def read_tracks(self, index):
        self.calls.append(index)
        if index not in (1, 2, 3):
            raise AssertionError("A diagnostic subset read outside the observed last three frames")
        x, y = self.coordinates[index - 1]
        return np.array([(x, y, 2., 2., 7)], dtype=TRACK_DTYPE)


class TrackInventory:
    def __init__(self):
        self.windows = [(name, 3) for name in ("steady", "turning", "slow")]
        self.sample_ids = [f"{name}:3" for name, _ in self.windows]
        self.timestamps = {name: 50_000_000_000 + np.arange(8, dtype=np.int64) * 50000
                           for name, _ in self.windows}
        self.readers = {
            "steady": ObservedTrackReader([(0, 0), (10, 0), (20, 0)]),
            "turning": ObservedTrackReader([(0, 0), (3, 0), (3, 3)]),
            "slow": ObservedTrackReader([(0, 0), (1, 0), (2, 0)]),
        }

    def _reader(self, name):
        return self.readers[name]


def test_high_motion_and_nonuniform_rank_only_observed_tracks():
    dataset = TrackInventory()
    assert motion_subset(dataset, "high_motion", 1 / 3) == ["steady:3"]
    assert motion_subset(dataset, "nonuniform", 1 / 3) == ["turning:3"]
    for reader in dataset.readers.values():
        assert reader.calls == [1, 2, 3, 1, 2, 3]


def test_motion_subset_uses_timestamps_and_preserves_inventory_order():
    dataset = TrackInventory()
    # A much shorter final interval makes the slow object's observed velocity
    # largest; frame-index displacement alone would rank the steady object first.
    dataset.timestamps["slow"][3] = dataset.timestamps["slow"][2] + 1000
    assert motion_subset(dataset, "high_motion", 1 / 3) == ["slow:3"]
    selected = motion_subset(dataset, "high_motion", 2 / 3)
    assert selected == [sample for sample in dataset.sample_ids if sample in {"steady:3", "slow:3"}]


@pytest.mark.parametrize("kind,fraction", [("future_motion", .2), ("high_motion", 0), ("nonuniform", 1.1)])
def test_invalid_motion_subsets_fail(kind, fraction):
    with pytest.raises(ValueError):
        motion_subset(TrackInventory(), kind, fraction)


class RecordingPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, history, context_times, query_times, **kwargs):
        self.calls.append(dict(kwargs, context_times=context_times, query_times=query_times))
        if kwargs.get("event_off"):
            return history[:, -1]
        return history[:, -1] + kwargs["event_context"][:, -1]

    def rollout(self, history, context_times, query_times, **kwargs):
        self.calls.append(dict(kwargs, context_times=context_times, query_times=query_times))
        value = history[:, -1] + kwargs["event_context"][:, -1]
        return value[:, None].expand(-1, len(query_times), -1, -1)


class RecordingForecaster(nn.Module):
    feature_dim = 2

    def __init__(self):
        super().__init__()
        self.predictor = RecordingPredictor()
        self.event_encoder = nn.Identity()
        self.event_encoder.output_grid = (1, 1)
        self.event_calls = []

    def encode_latents(self, rgb):
        return rgb[:, :2, 0, 0].unsqueeze(1)

    def encode_events(self, events):
        self.event_calls.append(events.clone())
        # A trained event encoder can have nonzero output for all-zero voxels.
        return events[:, :, :2].mean((-2, -1)).unsqueeze(2) + 3


def test_zero_intervention_runs_encoder_and_off_skips_entire_event_path():
    forecaster = RecordingForecaster()
    model = ForecastDetector(forecaster, nn.Identity(), timing=ForecastTiming.from_config({'data': {}}))
    history = torch.arange(4 * 3 * 2 * 2, dtype=torch.float32).reshape(1, 4, 3, 2, 2)
    events = torch.zeros(1, 4, 20, 2, 2)
    zero = model.features(history, events)
    assert len(forecaster.event_calls) == 1
    off = model.features(history, events, event_off=True)
    assert len(forecaster.event_calls) == 1
    torch.testing.assert_close(zero - off, torch.full_like(off, 3))
    assert forecaster.predictor.calls[-1]["event_context"] is None
    assert forecaster.predictor.calls[-1]["event_off"]


def test_rollout_pipeline_encodes_only_supplied_observed_events_once():
    forecaster = RecordingForecaster()
    model = ForecastDetector(forecaster, nn.Identity(), timing=ForecastTiming.from_config({'data': {}}))
    history = torch.zeros(4, 4, 3, 2, 2)
    events = torch.randn(4, 4, 20, 2, 2)
    prediction = model.features(history, events, horizon_steps=4)
    assert prediction.shape == (4, 1, 2)
    assert len(forecaster.event_calls) == 1
    torch.testing.assert_close(forecaster.event_calls[0], events)
    call = forecaster.predictor.calls[-1]
    torch.testing.assert_close(call['context_times'], torch.tensor([0., .05, .10, .15], dtype=torch.float64))
    torch.testing.assert_close(call['query_times'], torch.tensor([.20, .25, .30, .35], dtype=torch.float64))
    torch.testing.assert_close(call['event_times'], call['context_times'])
    assert call["event_context"].shape[1] == events.shape[1]


def test_persistence_uses_last_rgb_without_running_predictor_or_event_encoder():
    forecaster = RecordingForecaster()
    model = ForecastDetector(forecaster, nn.Identity(), timing=ForecastTiming.from_config({'data': {}}))
    history = torch.randn(4, 4, 3, 2, 2)
    actual = model.features(history, torch.zeros(4, 4, 20, 2, 2), horizon_steps=4, persistence=True)
    torch.testing.assert_close(actual, history[:, -1, :2, 0, 0].unsqueeze(1))
    assert not forecaster.event_calls and not forecaster.predictor.calls
