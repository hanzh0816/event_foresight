"""Standalone RGB inputs and labels for feature forecasting."""
from .definitions import CLASSES, DEFAULT_SPLITS, HEIGHT, WIDTH
from .reader import RGBSequenceReader
from .rgb import RGBDataset, collate_rgb, preprocess_rgb

__all__ = ["CLASSES", "DEFAULT_SPLITS", "HEIGHT", "WIDTH", "RGBSequenceReader", "RGBDataset", "collate_rgb", "preprocess_rgb"]
