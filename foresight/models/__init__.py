"""DINO-world RGB encoder, causal predictor and detection readout."""

from .dino_world import DINOWorldForecaster, DINOWorldPredictor
# Legacy components remain importable for old asset conversion and regression tests.
from .forecaster import DINOForecaster, DINOFeaturePredictor, PCAProjection
from .detector import FutureDetector, load_detector, postprocess
