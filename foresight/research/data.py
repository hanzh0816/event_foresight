"""Causal, exposure-bounded DSEC protocol for motion-representation experiments.

RGB and detection boxes must already be in the raw Event camera's 640 x 480
view. An explicit alignment declaration records that preprocessing assumption;
resizing here never performs camera calibration or geometric registration.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from foresight.data.definitions import (
    CLASSES, DEFAULT_SPLITS, HEIGHT, WIDTH, MissingDataError, require_int,
)
from foresight.data.event_reader import EventStream, H5HandleCache
from foresight.data.reader import RGBSequenceReader
from foresight.data.rgb import _size, preprocess_rgb


def voxelize_interval(events, start_us, end_us, *, bins=5, image_size=(128, 256)):
    """Return [K, polarity, H, W] log1p counts on the half-open interval.

    Counts use a fixed elementwise transform, without per-window or per-bin
    standardization. Collapsing the temporal axis can therefore be implemented
    by summing counts (``expm1``), then reapplying ``log1p``. Polarity 0 is
    negative; polarity 1 is positive. No event is duplicated across intervals.
    """
    require_int(start_us, "start_us")
    require_int(end_us, "end_us")
    require_int(bins, "bins", 1)
    if end_us <= start_us:
        raise ValueError("Event intervals require start_us < end_us")
    height, width = _size(image_size)
    arrays = {key: np.asarray(events[key]) for key in ("x", "y", "p", "t")}
    count = len(arrays["t"])
    if any(array.shape != (count,) for array in arrays.values()):
        raise ValueError("Event fields must be equal-length vectors")
    output = np.zeros((bins, 2, height, width), dtype=np.float32)
    if not count:
        return torch.from_numpy(output)
    x, y, polarity, times = (arrays[key].astype(np.int64) for key in ("x", "y", "p", "t"))
    if np.any((times < start_us) | (times >= end_us)):
        raise ValueError("Event timestamps fall outside the half-open interval")
    if np.any((x < 0) | (x >= WIDTH) | (y < 0) | (y >= HEIGHT)):
        raise ValueError("Event coordinates must lie in the native 640x480 view")
    if np.any((polarity != 0) & (polarity != 1)):
        raise ValueError("DSEC event polarity must be 0 or 1")
    temporal = (times - start_us) * bins // (end_us - start_us)
    np.add.at(output, (temporal, polarity, y * height // HEIGHT, x * width // WIDTH), 1)
    np.log1p(output, out=output)
    return torch.from_numpy(output)


def _normalized_target(target):
    xyxy = torch.from_numpy(target.boxes.copy()) / torch.tensor([WIDTH, HEIGHT, WIDTH, HEIGHT])
    x0, y0, x1, y1 = xyxy.unbind(-1)
    boxes = torch.stack(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0), dim=-1)
    return {"boxes": boxes, "labels": torch.from_numpy(target.class_ids.copy())}


class ResearchDSECDataset(Dataset):
    """A common inventory of causal histories and consecutive future targets.

    ``sample_history_frames`` controls inventory eligibility; ``context_frames``
    controls actual model input. They default to the same value, independently
    of the legacy RGB baseline's 40-frame inventory convention. All compared
    variants must use the same inventory configuration or explicit sample IDs.

    ``future_targets=False`` reads future annotations and timestamp metadata,
    but never future RGB pixels or future Event payloads. Coverage eligibility
    remains identical to the teacher-target dataset so comparisons are paired.
    """

    def __init__(self, root, split="train", image_size=(448, 896),
                 event_image_size=(128, 256), context_frames=4, bins=5,
                 rollout_steps=1, alignment=None, alignment_evidence=None,
                 future_targets=True, stride=1, sample_ids=None,
                 sample_history_frames=None, sequences=None, tolerance_us=1000,
                 evaluation_horizons=None):
        if alignment != "prealigned_event_view":
            raise ValueError("alignment must explicitly be 'prealigned_event_view'; raw RGB/Event views are not interchangeable")
        if not isinstance(alignment_evidence, str) or not alignment_evidence.strip():
            raise ValueError("alignment_evidence must describe the verified RGB/box-to-Event preprocessing")
        require_int(context_frames, "context_frames", 2)
        require_int(bins, "bins", 1)
        require_int(rollout_steps, "rollout_steps", 1)
        require_int(stride, "stride", 1)
        require_int(tolerance_us, "tolerance_us", 0)
        history = context_frames if sample_history_frames is None else sample_history_frames
        require_int(history, "sample_history_frames", context_frames)
        if split not in DEFAULT_SPLITS:
            raise ValueError(f"Unknown DSEC split: {split!r}")
        if type(future_targets) is not bool:
            raise ValueError("future_targets must be a boolean")
        self.image_size, self.event_image_size = _size(image_size), _size(event_image_size)
        if any(dimension % 14 for dimension in self.image_size):
            raise ValueError("image_size must be divisible by patch_size 14")
        self.grid_size = tuple(dimension // 14 for dimension in self.image_size)
        self.root = Path(root).expanduser().resolve()
        self.split, self.alignment, self.alignment_evidence = split, alignment, alignment_evidence.strip()
        self.context_frames, self.bins, self.rollout_steps = int(context_frames), int(bins), int(rollout_steps)
        self.history_frames = self.sample_history_frames = self.window_history_frames = int(history)
        self.history_ms = (self.history_frames - 1) * 50
        self.prediction_frames, self.prediction_ms = self.rollout_steps, self.rollout_steps * 50
        self.future_targets, self.stride, self.tolerance_us = future_targets, int(stride), int(tolerance_us)
        if evaluation_horizons is None:
            evaluation_horizons = list(range(1, rollout_steps + 1))
        if isinstance(evaluation_horizons, (str, bytes)):
            raise ValueError("evaluation_horizons must be a sequence of frame offsets")
        self.evaluation_horizons = tuple(evaluation_horizons)
        if not self.evaluation_horizons or len(set(self.evaluation_horizons)) != len(self.evaluation_horizons):
            raise ValueError("evaluation_horizons must be nonempty and unique")
        for horizon in self.evaluation_horizons:
            require_int(horizon, "evaluation horizon", 1)
            if horizon > self.rollout_steps:
                raise ValueError("evaluation horizons cannot exceed rollout_steps")
        if isinstance(sequences, (str, bytes)):
            raise ValueError("sequences must be a sequence of names")
        self.sequences = tuple(DEFAULT_SPLITS[split] if sequences is None else sequences)
        if len(set(self.sequences)) != len(self.sequences):
            raise ValueError("sequences must not contain duplicates")
        invalid = set(self.sequences) - set(DEFAULT_SPLITS[split])
        if invalid:
            raise ValueError(f"Sequences outside the fixed {split} split: {sorted(invalid)}")
        for other, names in DEFAULT_SPLITS.items():
            if other != split and set(self.sequences).intersection(names):
                raise ValueError(f"Selected sequences overlap the {other} split")
        self.timestamps, self.exposures, self._readers, self._streams = {}, {}, {}, {}
        self._handles = H5HandleCache(max_open=8)
        try:
            inventory = self._build_windows()
        finally:
            self._handles.close()
        self.full_inventory_size = len(inventory)
        self.source_indices = list(range(len(inventory)))
        if sample_ids is not None:
            if isinstance(sample_ids, (str, bytes)):
                raise ValueError("sample_ids must be a sequence of unique strings")
            ids = list(sample_ids)
            if any(not isinstance(item, str) for item in ids) or len(set(ids)) != len(ids):
                raise ValueError("sample_ids must be unique strings")
            lookup = {f"{name}:{end}": index for index, (name, end) in enumerate(inventory)}
            missing = [item for item in ids if item not in lookup]
            if missing:
                raise ValueError(f"Sample IDs not in the selected inventory: {missing[:5]}")
            self.source_indices = [lookup[item] for item in ids]
        self.windows = [inventory[index] for index in self.source_indices]
        self.sample_ids = [f"{name}:{end}" for name, end in self.windows]

    def _path(self, sequence):
        if sequence not in self.sequences:
            raise ValueError(f"Sequence is outside the selected inventory: {sequence}")
        return self.root / ("test" if self.split == "test" else "train") / sequence

    def reader(self, sequence):
        return RGBSequenceReader(self._path(sequence))

    def _stream(self, sequence):
        if sequence not in self._streams:
            self._streams[sequence] = EventStream(self._path(sequence) / "events/left/events.h5", self._handles)
        return self._streams[sequence]

    def _build_windows(self):
        windows = []
        self.excluded_windows = {"unlabeled": 0, "cadence": 0, "coverage": 0, "noncausal": 0}
        for name in self.sequences:
            with self.reader(name) as reader:
                times = reader.timestamps.copy()
                exposures = reader.exposures
                if not len(times) or np.any(np.diff(times) <= 0):
                    raise MissingDataError(f"{name}: RGB timestamps must be nonempty and strictly increasing")
                if exposures is None or exposures.shape != (len(times), 2):
                    raise MissingDataError(f"{name}: exposure_timestamps.txt requires one start,end row per frame")
                if np.any(exposures[:, 1] < exposures[:, 0]) or np.any(np.diff(exposures[:, 1]) <= 0):
                    raise MissingDataError(f"{name}: invalid or non-increasing exposure intervals")
                if np.any(np.diff(reader.tracks["t"].astype(np.int64)) < 0):
                    raise MissingDataError(f"{name}: detection annotations must be sorted by timestamp")
                self.timestamps[name], self.exposures[name] = times, exposures.copy()
                stream = self._stream(name)
                handle = self._handles.get(stream.event_path)
                keys = [f"events/{key}" for key in ("x", "y", "p", "t")] + ["t_offset", "ms_to_idx"]
                if any(key not in handle for key in keys) or not len(handle["events/t"]):
                    raise MissingDataError(f"{name}: missing event fields or empty event stream")
                count = len(handle["events/t"])
                if any(handle[f"events/{key}"].shape != (count,) or handle[f"events/{key}"].dtype.kind not in "uib"
                       for key in ("x", "y", "p", "t")):
                    raise MissingDataError(f"{name}: event fields must be equal-length integer vectors")
                offset = handle["t_offset"]
                lookup = np.asarray(handle["ms_to_idx"])
                if offset.shape != () or offset.dtype.kind not in "ui":
                    raise MissingDataError(f"{name}: t_offset must be an integer scalar")
                if (lookup.ndim != 1 or not len(lookup) or lookup.dtype.kind not in "ui"
                        or lookup[0] != 0 or np.any(lookup < 0) or np.any(lookup > count)
                        or np.any(lookup[1:] < lookup[:-1])):
                    raise MissingDataError(f"{name}: invalid ms_to_idx lookup")
                available_start, available_end = stream.available_time_range
                if not 0 <= stream.event_metadata["first_local_us"] <= stream.event_metadata["last_local_us"]:
                    raise MissingDataError(f"{name}: invalid event timestamp bounds")
                invalid = np.abs(np.diff(times) - 50_000) > self.tolerance_us
                prefix = np.concatenate(([0], np.cumsum(invalid)))
                labeled = set()
                for index in np.flatnonzero(reader.label_ends > reader.label_starts):
                    target = reader.read_target(int(index))
                    if np.any((target.class_ids < 0) | (target.class_ids >= len(CLASSES))):
                        raise MissingDataError(f"{name}: detection class outside the fixed class set")
                    if len(target.boxes):
                        labeled.add(int(index))
                for end in range(self.history_frames - 1, len(times) - self.rollout_steps, self.stride):
                    outer_start = end - self.history_frames + 1
                    context_start = end - self.context_frames + 1
                    future_end = end + self.rollout_steps
                    if any(end + horizon not in labeled for horizon in self.evaluation_horizons):
                        self.excluded_windows["unlabeled"] += 1
                    elif prefix[future_end] != prefix[outer_start]:
                        self.excluded_windows["cadence"] += 1
                    elif times[end + 1] <= exposures[end, 1]:
                        self.excluded_windows["noncausal"] += 1
                    elif exposures[context_start, 1] < available_start or exposures[future_end, 1] > available_end:
                        self.excluded_windows["coverage"] += 1
                    else:
                        windows.append((name, end))
        return windows

    def _reader(self, sequence):
        if sequence not in self._readers:
            self._readers[sequence] = self.reader(sequence)
        return self._readers[sequence]

    def _rgb(self, reader, index):
        pixels = reader.read_image(index).image
        if pixels.shape[:2] != (HEIGHT, WIDTH):
            raise MissingDataError(f"{reader.name}: prealigned RGB must be 480x640 before resize; got {pixels.shape[:2]}")
        return preprocess_rgb(pixels, self.image_size)

    def _event_interval(self, sequence, start_us, end_us):
        events = self._stream(sequence).read_events(int(start_us), int(end_us))
        return voxelize_interval(events, int(start_us), int(end_us), bins=self.bins,
                                 image_size=self.event_image_size)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        name, end = self.windows[index]
        reader = self._reader(name)
        start = end - self.context_frames + 1
        future_indices = list(range(end + 1, end + self.rollout_steps + 1))
        stops = self.exposures[name][:, 1]
        historical_intervals = [(int(stops[i]), int(stops[i + 1])) for i in range(start, end)]
        future_intervals = [(int(stops[i - 1]), int(stops[i])) for i in future_indices]
        targets = [reader.read_target(i) for i in future_indices]
        cutoff = int(stops[end])
        sample = {
            "history_rgb": torch.stack([self._rgb(reader, i) for i in range(start, end + 1)]),
            "history_events": torch.stack([self._event_interval(name, a, b) for a, b in historical_intervals]),
            "targets": [_normalized_target(target) for target in targets],
            "meta": {
                "sample_id": f"{name}:{end}", "sequence": name, "end": end,
                "history_indices": list(range(start, end + 1)), "future_indices": future_indices,
                "target_ids": [target.target_id for target in targets],
                "target_has_labels": [bool(reader.label_ends[i] > reader.label_starts[i]) for i in future_indices],
                "target_timestamps_us": [target.timestamp_us for target in targets],
                "input_cutoff_us": cutoff,
                "history_start_us": int(self.exposures[name][start, 0]),
                "event_history_start_us": int(stops[start]),
                "history_intervals_us": historical_intervals,
                "future_intervals_us": future_intervals,
                "history_dt_us": [b - a for a, b in historical_intervals],
                "future_dt_us": [b - a for a, b in future_intervals],
                "horizon_us": [target.timestamp_us - cutoff for target in targets],
                "transition_horizon_us": [int(stops[i]) - cutoff for i in future_indices],
                "alignment": self.alignment, "alignment_evidence": self.alignment_evidence,
            },
        }
        if self.future_targets:
            sample["future_rgb_target"] = torch.stack([self._rgb(reader, i) for i in future_indices])
            sample["future_events_target"] = torch.stack([self._event_interval(name, a, b) for a, b in future_intervals])
        return sample

    def close(self):
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()
        self._handles.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_readers"], state["_streams"] = {}, {}
        state["_handles"] = H5HandleCache(max_open=self._handles.max_open)
        return state

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def collate_research(samples):
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    batch = {key: torch.stack([sample[key] for sample in samples])
             for key in ("history_rgb", "history_events")}
    batch["targets"] = [sample["targets"] for sample in samples]
    batch["meta"] = [sample["meta"] for sample in samples]
    for key in ("future_rgb_target", "future_events_target"):
        present = [key in sample for sample in samples]
        if any(present) and not all(present):
            raise ValueError("Cannot mix inference and teacher-target samples in one batch")
        if all(present):
            batch[key] = torch.stack([sample[key] for sample in samples])
    return batch


class HorizonDatasetView:
    """Read-only label/inventory adapter for ``FutureDetectionEvaluator``."""

    def __init__(self, dataset, horizon):
        require_int(horizon, "horizon", 1)
        if horizon not in dataset.evaluation_horizons:
            raise ValueError("horizon must be one of the common inventory's evaluation_horizons")
        self.dataset = dataset
        for attribute in ("windows", "timestamps", "history_frames", "context_frames", "tolerance_us",
                          "split", "sequences", "sample_ids", "root", "image_size"):
            setattr(self, attribute, getattr(dataset, attribute))
        self.prediction_frames, self.prediction_ms = int(horizon), int(horizon) * 50

    def reader(self, sequence):
        return self.dataset.reader(sequence)

    def __len__(self):
        return len(self.dataset)
