"""Fixed RGB detection classes, splits and typed reader records."""
from dataclasses import dataclass
from numbers import Integral

import numpy as np


CLASSES = (
    "pedestrian",
    "rider",
    "car",
    "bus",
    "truck",
    "bicycle",
    "motorcycle",
    "train",
)
HEIGHT, WIDTH = 480, 640

# Fixed project split: 41 train, 6 val, 12 test sequences.
# Based on dsec-det; the test list does not contain thun_02_a.
DEFAULT_SPLITS = {
    "train": tuple(
        """
        thun_00_a interlaken_00_c interlaken_00_d interlaken_00_e interlaken_00_f interlaken_00_g
        zurich_city_00_a zurich_city_00_b zurich_city_01_a zurich_city_01_b zurich_city_01_c
        zurich_city_01_d zurich_city_01_e zurich_city_01_f zurich_city_02_a zurich_city_02_b
        zurich_city_02_c zurich_city_02_d zurich_city_02_e zurich_city_03_a zurich_city_04_a
        zurich_city_04_b zurich_city_04_c zurich_city_04_d zurich_city_04_e zurich_city_04_f
        zurich_city_05_a zurich_city_05_b zurich_city_06_a zurich_city_07_a zurich_city_08_a
        zurich_city_09_a zurich_city_09_b zurich_city_09_c zurich_city_09_d zurich_city_09_e
        zurich_city_10_a zurich_city_10_b zurich_city_11_a zurich_city_11_b zurich_city_11_c
    """.split()
    ),
    "val": tuple(f"zurich_city_{i}_a" for i in range(16, 22)),
    "test": tuple(
        """
        thun_01_a thun_01_b interlaken_00_a interlaken_00_b interlaken_01_a
        zurich_city_12_a zurich_city_13_a zurich_city_13_b zurich_city_14_a zurich_city_14_b
        zurich_city_14_c zurich_city_15_a
    """.split()
    ),
}


class DataError(Exception):
    """Base exception for data and configuration errors."""


class MissingDataError(DataError):
    """A selected sequence is absent, incomplete, or invalid."""


class WindowUnavailableError(DataError):
    """A requested frame or event range falls outside a sequence."""


def require_int(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


@dataclass
class Frame:
    frame_index: int
    timestamp_us: int
    image: np.ndarray
    exposure_us: tuple[int, int] | None = None


@dataclass
class DetectionTarget:
    target_id: str
    frame_index: int
    timestamp_us: int
    boxes: np.ndarray
    class_ids: np.ndarray
    track_ids: np.ndarray
    source_class_ids: np.ndarray
    class_confidence: np.ndarray
