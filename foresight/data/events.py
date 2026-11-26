"""Historical event histograms in the DSEC detection image coordinate system."""

import numpy as np
import torch
from torch.nn import functional as F

from .definitions import HEIGHT, WIDTH, MissingDataError, WindowUnavailableError, require_int
from .event_reader import EventStream, H5HandleCache
from .rgb import RGBDataset, _size, preprocess_rgb


COORDINATE_FRAME = "left_distorted_event"


def events_to_voxel(events, start_us, end_us, event_size=(512, 1024), *, bins=10,
                    sensor_size=(HEIGHT, WIDTH), grid_size=(32, 64), activity_threshold=1):
    """Return log-count voxels [2*bins,H,W] and a raw-count patch activity mask.

    Channels are negative bins followed by positive bins, oldest to newest.
    The source plane is cam0: dsec-det's remapped/distorted RGB and tracks
    already share this plane with raw events. Applying rectify_map here would
    move only the events to camRect0 and break their image correspondence.
    Resize uses the same full sensor field of view as RGB preprocessing.
    """
    require_int(start_us, "start_us")
    require_int(end_us, "end_us")
    require_int(bins, "bins", 1)
    require_int(activity_threshold, "activity_threshold", 1)
    if end_us <= start_us:
        raise ValueError("Event windows require start_us < end_us")
    height, width = _size(sensor_size)
    out_height, out_width = _size(event_size)
    grid_height, grid_width = _size(grid_size)
    arrays = {key: np.asarray(events[key]) for key in ("t", "x", "y", "p")}
    if any(value.ndim != 1 for value in arrays.values()):
        raise ValueError("Event arrays must be one-dimensional")
    if len({len(value) for value in arrays.values()}) != 1:
        raise ValueError("Event arrays must have equal lengths")
    if any(not np.issubdtype(value.dtype, np.integer) for value in arrays.values()):
        raise ValueError("DSEC timestamps, coordinates and polarities must be integers")
    keep = (arrays["t"] >= start_us) & (arrays["t"] < end_us)
    t, x, y, p = (arrays[key][keep].astype(np.int64, copy=False) for key in ("t", "x", "y", "p"))
    if np.any((p != 0) & (p != 1)):
        raise ValueError("DSEC polarity must be 0 (negative) or 1 (positive)")
    if np.any((x < 0) | (x >= width) | (y < 0) | (y >= height)):
        raise ValueError(f"Event coordinates must lie in the {width}x{height} raw sensor plane")
    temporal_bin = ((t - start_us) * bins) // (end_us - start_us)
    channels = p * bins + temporal_bin
    flat = (channels * height + y) * width + x
    counts = np.bincount(flat, minlength=2 * bins * height * width)
    counts = counts.reshape(2 * bins, height, width).astype(np.float32)
    np.log1p(counts, out=counts)
    voxel = torch.from_numpy(counts)
    if (out_height, out_width) != (height, width):
        voxel = F.interpolate(voxel.unsqueeze(0), size=(out_height, out_width),
                              mode="bilinear", align_corners=False).squeeze(0)
    # Pixel centers are mapped to the shared normalized image plane. Count
    # original events, before log compression and image interpolation.
    patch_x = ((2 * x + 1) * grid_width) // (2 * width)
    patch_y = ((2 * y + 1) * grid_height) // (2 * height)
    patch_counts = np.bincount(patch_y * grid_width + patch_x,
                               minlength=grid_height * grid_width)
    activity = torch.from_numpy(patch_counts >= activity_threshold)
    return voxel, activity


def reverse_event_bins(voxels, bins=None):
    """Reverse temporal bins within each polarity, preserving all other axes."""
    if voxels.ndim < 3:
        raise ValueError("Event voxels must end in [2*bins,height,width]")
    channels, height, width = voxels.shape[-3:]
    if bins is None:
        bins = channels // 2
    require_int(bins, "bins", 1)
    if channels != 2 * bins:
        raise ValueError("Event channel count must equal 2*bins")
    return voxels.reshape(*voxels.shape[:-3], 2, bins, height, width).flip(-3).reshape_as(voxels)


class EventRGBDataset(RGBDataset):
    """Use exactly the RGB inventory, with one preceding 50 ms window per frame.

    Event coverage failures raise an error rather than change the sample set.
    All event endpoints are RGB timestamps; exposure timestamps do not shift
    the event window or the fixed 50 ms forecast grid.
    """

    def __init__(self, *args, event_bins=10, event_size=(512, 1024),
                 event_window_us=50000, event_activity_threshold=1, **kwargs):
        require_int(event_bins, "event_bins", 1)
        if event_bins not in (5, 10):
            raise ValueError("v3.1 supports 10 event bins or the 5-bin ablation")
        if type(event_window_us) is not int or event_window_us != 50000:
            raise ValueError("v3.1 requires event_window_us=50000")
        require_int(event_activity_threshold, "event_activity_threshold", 1)
        self.event_bins = int(event_bins)
        self.event_size = _size(event_size)
        self.event_window_us = event_window_us
        self.event_activity_threshold = int(event_activity_threshold)
        self._event_handles = H5HandleCache(max_open=8)
        self._event_streams = {}
        super().__init__(*args, **kwargs)
        self._validate_event_coverage()

    def _stream(self, sequence):
        if sequence not in self.sequences:
            raise ValueError(f"Sequence is outside the selected inventory: {sequence}")
        if sequence not in self._event_streams:
            directory = "test" if self.split == "test" else "train"
            path = self.root / directory / sequence / "events/left/events.h5"
            if not path.is_file():
                raise MissingDataError(f"Missing DSEC events: {path}")
            self._event_streams[sequence] = EventStream(path, self._event_handles)
        return self._event_streams[sequence]

    def _validate_event_coverage(self):
        for name in self.sequences:
            ends = [end for sequence, end in self.windows if sequence == name]
            if not ends:
                continue
            stream = self._stream(name)
            available_start, available_end = stream.available_time_range
            times = self.timestamps[name]
            for end in ends:
                first = end - self.context_frames + 1
                start_us, end_us = int(times[first]) - self.event_window_us, int(times[end])
                if start_us < available_start or end_us > available_end:
                    raise WindowUnavailableError(
                        f"{name}:{end}: historical event windows [{start_us}, {end_us}) "
                        f"exceed stream coverage [{available_start}, {available_end}); "
                        "RGB inventory is unchanged. Provide the complete recording."
                    )
        self._event_handles.close()

    def _image(self, reader, index):
        pixels = reader.read_image(index).image
        if pixels.shape[:2] != (HEIGHT, WIDTH):
            raise MissingDataError(
                f"{reader.name}/{index:06d}: Event-RGB requires dsec-det remapped RGB "
                f"in images/left/distorted at {HEIGHT}x{WIDTH}; got {pixels.shape[:2]}. "
                "Native camRect1 RGB must be calibrated/remapped before use."
            )
        return preprocess_rgb(pixels, self.image_size)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        name, end = self.windows[index]
        times = self.timestamps[name][end - self.context_frames + 1:end + 1]
        stream = self._stream(name)
        voxels, activities = [], []
        for timestamp in times:
            stop = int(timestamp)
            start = stop - self.event_window_us
            voxel, activity = events_to_voxel(
                stream.read_events(start, stop), start, stop, self.event_size,
                bins=self.event_bins, grid_size=self.grid_size,
                activity_threshold=self.event_activity_threshold,
            )
            voxels.append(voxel)
            activities.append(activity)
        sample["history_events"] = torch.stack(voxels)
        sample["event_activity"] = torch.stack(activities)
        if sample["meta"]["horizontal_flipped"]:
            sample["history_events"] = sample["history_events"].flip(-1)
            sample["event_activity"] = sample["event_activity"].reshape(
                self.context_frames, *self.grid_size).flip(-1).flatten(1)
        sample["meta"].update(
            event_start_us=(times - self.event_window_us).tolist(),
            event_end_us=times.tolist(), coordinate_frame=COORDINATE_FRAME,
        )
        return sample

    def close(self):
        super().close()
        self._event_handles.close()
        self._event_streams.clear()

    def __getstate__(self):
        state = super().__getstate__()
        state["_event_streams"] = {}
        return state
