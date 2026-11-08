"""RGB Encoder, fixed PCA, temporal Predictor and detection readout."""

from .forecaster import DINOForecaster, DINOFeaturePredictor, PCAProjection
from .detector import FutureDetector, load_detector, postprocess
