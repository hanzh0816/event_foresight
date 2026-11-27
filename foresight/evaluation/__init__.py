"""Detection metrics; importing this package does not load a model or Trainer."""

from .detection import FutureDetectionEvaluator, MetricReport, Prediction

__all__ = ["FutureDetectionEvaluator", "MetricReport", "Prediction"]
