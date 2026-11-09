"""Process-local DSEC HDF5 slices in absolute microseconds."""

import os
from collections import OrderedDict
from pathlib import Path
import h5py
import hdf5plugin  # Registers DSEC compression filters.
import numpy as np
from .definitions import WindowUnavailableError, require_int


class H5HandleCache:
    """Bounded, process-local handles. No open handle is pickled into a worker."""

    def __init__(self, max_open=8):
        require_int(max_open, "max_open", 1)
        self.max_open = max_open
        self._pid = os.getpid()
        self._handles = OrderedDict()

    def get(self, path):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        key = str(Path(path).resolve())
        handle = self._handles.pop(key, None)
        if handle is None or not handle.id.valid:
            handle = h5py.File(key, "r")
        self._handles[key] = handle
        while len(self._handles) > self.max_open:
            _, old = self._handles.popitem(last=False)
            old.close()
        return handle

    def close_path(self, path):
        handle = self._handles.pop(str(Path(path).resolve()), None)
        if handle is not None:
            handle.close()

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self):
        return {
            "max_open": self.max_open,
            "_pid": os.getpid(),
            "_handles": OrderedDict(),
        }

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class EventStream:
    def __init__(self, filename, cache=None):
        self.event_path = Path(filename)
        self.name = self.event_path.parent.parent.parent.name
        self._handles = cache if cache is not None else H5HandleCache()
        self._event_meta = None

    @property
    def event_metadata(self):
        if self._event_meta is None:
            handle = self._handles.get(self.event_path)
            self._event_meta = {
                "count": len(handle["events/t"]),
                "offset_us": int(handle["t_offset"][()]),
                "first_local_us": int(handle["events/t"][0]),
                "last_local_us": int(handle["events/t"][-1]),
                "ms_to_idx": np.asarray(handle["ms_to_idx"], dtype=np.int64),
            }
        return self._event_meta

    @property
    def available_time_range(self):
        meta = self.event_metadata
        return meta["offset_us"], meta["offset_us"] + meta["last_local_us"] + 1

    def _validate_time(self, time_us):
        require_int(time_us, "event time")
        start, end = self.available_time_range
        if time_us < start or time_us > end:
            raise WindowUnavailableError(
                f"{self.name}: event time {time_us} is outside [{start}, {end}]"
            )

    def _lower_bound(self, absolute_us):
        """Exact first event >= a timestamp, using at most one millisecond bucket."""
        meta = self.event_metadata
        local = int(absolute_us) - meta["offset_us"]
        if local <= meta["first_local_us"]:
            return 0
        if local > meta["last_local_us"]:
            return meta["count"]
        bucket = local // 1000
        ms = meta["ms_to_idx"]
        if bucket >= len(ms):
            # Some valid recordings omit a trailing sentinel bucket. Use a
            # binary search over HDF5 instead of loading the complete time array.
            lo, hi = int(ms[-1]), meta["count"]
            handle = self._handles.get(self.event_path)
            while lo < hi:
                mid = (lo + hi) // 2
                if int(handle["events/t"][mid]) < local:
                    lo = mid + 1
                else:
                    hi = mid
            return lo
        lo = int(ms[bucket])
        hi = int(ms[bucket + 1]) if bucket + 1 < len(ms) else meta["count"]
        times = self._handles.get(self.event_path)["events/t"][lo:hi]
        return lo + int(np.searchsorted(times, local, side="left"))

    def event_indices(self, start_us, end_us):
        self._validate_time(start_us)
        self._validate_time(end_us)
        if start_us >= end_us:
            raise ValueError("Event windows require start_us < end_us")
        return self._lower_bound(start_us), self._lower_bound(end_us)

    def read_events_by_index(self, start, end):
        require_int(start, "start index", 0)
        require_int(end, "end index", 0)
        n = self.event_metadata["count"]
        if start > end or end > n:
            raise WindowUnavailableError(
                f"{self.name}: invalid event indices [{start}, {end}) for {n} events"
            )
        try:
            handle = self._handles.get(self.event_path)
            events = {
                key: handle[f"events/{key}"][start:end] for key in ("x", "y", "p", "t")
            }
            events["t"] = (
                events["t"].astype(np.int64) + self.event_metadata["offset_us"]
            )
        except OSError as exc:
            raise OSError(
                f"Cannot decode {self.event_path}: {exc}; check hdf5plugin installation"
            ) from exc
        return events

    def read_events(self, start_us, end_us):
        return self.read_events_by_index(*self.event_indices(start_us, end_us))

    def close(self):
        self._handles.close_path(self.event_path)
