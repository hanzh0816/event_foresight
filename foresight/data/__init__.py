"""DSEC RGB and causal Event-RGB inputs for feature forecasting."""
from .definitions import CLASSES, DEFAULT_SPLITS, HEIGHT, WIDTH
from .reader import RGBSequenceReader
from .rgb import RGBDataset, collate_rgb, preprocess_rgb
from .factory import dataset_from_config

__all__ = ["CLASSES", "DEFAULT_SPLITS", "HEIGHT", "WIDTH", "RGBSequenceReader", "RGBDataset",
           "EventRGBDataset", "collate_rgb", "preprocess_rgb", "dataset_from_config"]


def __getattr__(name):
    if name == "EventRGBDataset":
        from .events import EventRGBDataset
        return EventRGBDataset
    raise AttributeError(name)
