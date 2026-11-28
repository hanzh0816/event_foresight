"""Train the shared detection head on real frozen DINO features."""

from copy import deepcopy
import inspect

import torch

from .base import _ForecastTask, _detection_report
from .monitoring import log_training_losses


class DetectorTask(_ForecastTask):
    """Train the shared detection head on real frozen DINO features."""

    stage = "detector"

    def __init__(self, config, *, forecaster=None, detector=None, criterion=None):
        if config["model"].get("event", {}).get("enabled", False):
            raise ValueError("Train the shared head with the RGB detector configuration")
        if config["model"].get("predictor_checkpoint") or config["model"].get("rgb_checkpoint"):
            raise ValueError("The shared head trains on real DINO features without a predictor initializer")
        super().__init__(config, forecaster=forecaster)
        self.forecaster.requires_grad_(False).eval()
        architecture = deepcopy(getattr(detector, "architecture", None))
        if detector is None:
            from ..models.detection.detector import FutureDetector
            architecture = {name: parameter.default for name, parameter in
                            inspect.signature(FutureDetector.__init__).parameters.items()
                            if name != "self" and parameter.default is not inspect.Parameter.empty}
            architecture.update(config.get("detector", {}))
            architecture.update(feature_dim=self.forecaster.feature_dim,
                                grid_size=list(self.forecaster.grid_size))
            detector = FutureDetector(**architecture)
        self.detector_architecture = architecture
        self.detector = detector.requires_grad_(True)
        if criterion is None:
            from ..models.detection.criterion import DetectionCriterion
            criterion = DetectionCriterion(**config.get("loss", {}))
        self.criterion = criterion
        self.validation_predictions = []
        self.train()

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["detector_architecture"] = deepcopy(self.detector_architecture)

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        if checkpoint.get("detector_architecture") != self.detector_architecture:
            raise ValueError("Detector architecture differs from the resume checkpoint")

    def train(self, mode=True):
        super().train(mode)
        self.forecaster.eval()
        return self

    def forward(self, rgb):
        if rgb.ndim != 4:
            raise ValueError("Shared detector training requires a real target RGB frame [B,3,H,W]")
        with torch.no_grad():
            features = self.forecaster.encode_latents(rgb)
        return self.detector(features)

    @torch.no_grad()
    def predict(self, rgb):
        from ..models.detection.detector import postprocess
        return postprocess(self(rgb))

    def training_step(self, batch, batch_idx):
        losses = self.criterion(self(batch["future_rgb_target"]), batch["targets"], sync_num_boxes=True)
        batch_size = len(batch["history_rgb"])
        log_training_losses(self, {f"train_{key}": value for key, value in losses.items()},
                            batch_size=batch_size)
        return self._training_loss(losses["loss"], batch_size)

    def on_validation_epoch_start(self):
        self.validation_predictions = []

    def validation_step(self, batch, batch_idx):
        if "val" in self.datasets and self.global_rank >= len(self.datasets["val"]):
            return
        predictions = self.predict(batch["future_rgb_target"])
        for prediction, meta in zip(predictions, batch["meta"]):
            self.validation_predictions.append({
                "sample_id": meta["sample_id"], "target_id": meta["target_id"],
                "boxes": prediction["boxes"].cpu().tolist(),
                "class_ids": prediction["labels"].cpu().tolist(),
                "scores": prediction["scores"].cpu().tolist(),
            })

    def on_validation_epoch_end(self):
        report = _detection_report(self, self.validation_predictions)
        report["metadata"].update(stage=self.stage, feature_source="real_frozen_dinov2",
                                  input_spec=self.input_spec)
        self.last_report = report
        self._log_validation_report()
        self.validation_predictions = []
        self.last_validation_step = int(self.global_step)
