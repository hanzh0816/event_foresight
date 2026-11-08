"""Read DSEC RGB images and detection annotations; no sensor-stream dependency."""
from pathlib import Path
import numpy as np
from PIL import Image
from .definitions import HEIGHT, WIDTH, Frame, DetectionTarget, WindowUnavailableError, require_int


class RGBSequenceReader:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.name = self.path.name
        self.image_dir = self.path / "images/left/distorted"
        self.timestamps = np.loadtxt(self.path / "images/timestamps.txt", dtype=np.int64, ndmin=1)
        self.tracks = np.load(self.path / "object_detections/left/tracks.npy", mmap_mode="r", allow_pickle=False)
        track_times = self.tracks["t"].astype(np.int64)
        self.label_starts = np.searchsorted(track_times, self.timestamps, side="left")
        self.label_ends = np.searchsorted(track_times, self.timestamps, side="right")
        self.image_files = [self.image_dir / f"{i:06d}.png" for i in range(len(self.timestamps))]
        exposure = self.path / "images/left/exposure_timestamps.txt"
        self.exposures = np.loadtxt(exposure, dtype=np.int64, delimiter=",", ndmin=2) if exposure.exists() else None

    def __len__(self):
        return len(self.timestamps)

    def _frame_index(self, index):
        require_int(index, "frame index")
        if not 0 <= index < len(self):
            raise WindowUnavailableError(f"{self.name}: frame {index} is outside [0, {len(self)})")
        return int(index)

    def read_image(self, index):
        index = self._frame_index(index)
        with Image.open(self.image_files[index]) as image:
            pixels = np.array(image.convert("RGB"))
        exposure = tuple(map(int, self.exposures[index])) if self.exposures is not None else None
        return Frame(index, int(self.timestamps[index]), pixels, exposure)

    def read_tracks(self, index):
        index = self._frame_index(index)
        return self.tracks[self.label_starts[index]:self.label_ends[index]].copy()

    def read_target(self, index):
        index = self._frame_index(index)
        tracks = self.read_tracks(index)
        boxes = np.column_stack((tracks["x"], tracks["y"], tracks["x"] + tracks["w"],
                                 tracks["y"] + tracks["h"])).astype(np.float32)
        boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, WIDTH - 1)
        boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, HEIGHT - 1)
        keep = (boxes[:, 2:] > boxes[:, :2]).all(axis=1)
        classes = tracks["class_id"].astype(np.int64)
        return DetectionTarget(f"{self.name}/{index:06d}", index, int(self.timestamps[index]),
                               boxes[keep], classes[keep], tracks["track_id"][keep].copy(),
                               classes[keep].copy(), tracks["class_confidence"][keep].astype(np.float32))

    def close(self):
        mapping = getattr(self.tracks, "_mmap", None)
        if mapping is not None:
            mapping.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
