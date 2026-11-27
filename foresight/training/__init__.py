"""Lightning tasks for predictor and shared detection-head training."""

from .base import CHECKPOINT_FORMAT
from .detector import DetectorTask
from .predictor import PredictorTask

__all__ = ["CHECKPOINT_FORMAT", "DetectorTask", "PredictorTask"]
