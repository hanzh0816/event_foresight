"""Causal event windows paired with the existing RGB sample inventory."""

import math
import numpy as np
import torch
from torch.nn import functional as F
from .definitions import HEIGHT, WIDTH
from .reader import RGBSequenceReader
from .rgb import RGBDataset, preprocess_rgb
from .event_reader import EventStream, H5HandleCache


def encoding_spec(data):
    """Closed input protocol for the uniform E2 branch."""
    window = data.get("event_window_us", 50000)
    bins = data.get("event_bins", 5)
    widths = data.get("event_bin_widths_us")
    if type(window) is not int or window != 50000 or type(bins) is not int or bins != 5:
        raise ValueError("E2 uses a 50000-us interval and five bins per polarity")
    if widths is not None and widths != [10000] * 5:
        raise ValueError("This branch supports uniform event bins only")
    if data.get("event_representation", "histogram") != "histogram":
        raise ValueError("Only histogram event inputs are supported")
    return dict(event_window_us=window, event_bins=bins, event_bin_widths_us=[10000] * 5,
                event_representation="histogram")


def exposure_end_us(reader, frame_index):
    """The time when this RGB observation is fully available."""
    if reader.exposures is None:
        return int(reader.timestamps[frame_index])
    start, end = map(int, reader.exposures[frame_index])
    if end < start:
        raise ValueError(f"Invalid exposure interval at frame {frame_index}")
    return end


def event_interval(reader, frame_index, *, window_us=50000, cutoff_us=None):
    """Return a causal half-open interval; reject a frame past the input cutoff."""
    if window_us <= 0:
        raise ValueError("event_window_us must be positive")
    end = exposure_end_us(reader, frame_index)
    if cutoff_us is not None and end > cutoff_us:
        raise ValueError(f"Event endpoint {end} exceeds input cutoff {cutoff_us}")
    return end - int(window_us), end


def events_to_voxel(events, start_us, end_us, image_size=(448, 896), *, bins=5,
                    sensor_size=(HEIGHT, WIDTH), count_scale=5.0):
    """Polarity-major counts on fixed half-open temporal bins, then log/resize."""
    if end_us <= start_us or bins <= 0 or count_scale <= 0:
        raise ValueError("Require end_us > start_us, positive bins and count_scale")
    height, width = map(int, sensor_size)
    out_height, out_width = map(int, image_size)
    if min(height, width, out_height, out_width) <= 0:
        raise ValueError("Image dimensions must be positive")
    arrays = {key: np.asarray(events[key]) for key in ("t", "x", "y", "p")}
    if any(value.ndim != 1 for value in arrays.values()):
        raise ValueError("Event arrays must be one-dimensional")
    if len({len(value) for value in arrays.values()}) != 1:
        raise ValueError("Event arrays must have equal lengths")
    t = arrays["t"].astype(np.int64, copy=False)
    x = arrays["x"].astype(np.int64, copy=False)
    y = arrays["y"].astype(np.int64, copy=False)
    p = arrays["p"].astype(np.int64, copy=False)
    if np.any((p != 0) & (p != 1)):
        raise ValueError("DSEC event polarity must be 0 (negative) or 1 (positive)")
    keep = ((t >= start_us) & (t < end_us) &
            (x >= 0) & (x < width) & (y >= 0) & (y < height))
    temporal_bin = ((t[keep] - int(start_us)) * bins) // (int(end_us) - int(start_us))
    channels = p[keep] * bins + temporal_bin
    flat = (channels * height + y[keep]) * width + x[keep]
    counts = np.bincount(flat, minlength=2 * bins * height * width)
    counts = counts.reshape(2 * bins, height, width).astype(np.float32)
    np.log1p(counts, out=counts)
    counts /= math.log1p(count_scale)
    np.clip(counts, 0.0, 1.0, out=counts)
    voxel = torch.from_numpy(counts)
    if (out_height, out_width) != (height, width):
        voxel = F.interpolate(voxel[None], size=(out_height, out_width),
                              mode="bilinear", align_corners=False)[0]
    return voxel


class EventSequenceReader(RGBSequenceReader):
    def __init__(self, path, cache=None):
        super().__init__(path)
        self.event_stream = EventStream(self.path / "events/left/events.h5", cache)

    def read_events(self, start, end):
        return self.event_stream.read_events(start, end)

    def close(self):
        super().close()
        self.event_stream.close()


class EventDataset(RGBDataset):
    """Last-frame RGB supervision, or four event-only inputs for forecasting."""
    def __init__(self, *args, alignment=False, event_window_us=50000, event_bins=5,
                 event_bin_widths_us=None, event_representation="histogram", **kwargs):
        self.encoding = encoding_spec(dict(event_window_us=event_window_us, event_bins=event_bins,
                                          event_bin_widths_us=event_bin_widths_us,
                                          event_representation=event_representation))
        self.alignment = alignment
        self._event_handles = H5HandleCache(max_open=8)
        if kwargs.get("future_rgb"):
            raise ValueError("Event alignment never reads future RGB")
        super().__init__(*args, **kwargs)

    def reader(self, sequence):
        if sequence not in self.sequences:
            raise ValueError("Sequence is outside the selected inventory")
        directory = "test" if self.split == "test" else "train"
        return EventSequenceReader(self.root / directory / sequence, self._event_handles)

    def voxel(self, reader, index, cutoff):
        start, stop = event_interval(reader, index, window_us=self.encoding["event_window_us"], cutoff_us=cutoff)
        return events_to_voxel(reader.read_events(start, stop), start, stop, self.image_size), start, stop

    def __getitem__(self, index):
        name, end = self.windows[index]
        reader = self._reader(name)
        cutoff = exposure_end_us(reader, end)
        target_index = end + self.prediction_frames
        timestamp = int(reader.timestamps[target_index])
        if timestamp <= cutoff:
            raise ValueError("Future target must follow the input exposure cutoff")
        indices = [end] if self.alignment else range(end - self.context_frames + 1, end + 1)
        frames = [self.voxel(reader, i, cutoff) for i in indices]
        meta = dict(sample_id=f"{name}:{end}", target_id=f"{name}/{target_index:06d}",
                    target_timestamp_us=timestamp, event_start_us=[v[1] for v in frames],
                    event_end_us=[v[2] for v in frames], input_cutoff_us=cutoff,
                    effective_prediction_us=timestamp-cutoff)
        if self.alignment:
            return dict(event_voxel=frames[0][0], rgb=preprocess_rgb(reader.read_image(end).image, self.image_size), meta=meta)
        return dict(history_events=torch.stack([v[0] for v in frames]), meta=meta)

    def close(self):
        super().close()
        self._event_handles.close()


def collate_event(samples):
    keys = ("rgb", "event_voxel") if "rgb" in samples[0] else ("history_events",)
    return {**{key: torch.stack([sample[key] for sample in samples]) for key in keys},
            "meta": [sample["meta"] for sample in samples]}


