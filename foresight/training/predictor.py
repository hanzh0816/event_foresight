"""Predictor training with causal next-frame and event alignment supervision."""

from copy import deepcopy
import math

import torch
from torch import distributed as dist
from torch.nn import functional as F

from .base import _ForecastTask, _detection_report, _distributed
from .monitoring import log_training_losses


class PredictorTask(_ForecastTask):
    """Frozen encoder and causal next-frame teacher forcing across every history step."""

    stage = "predictor"

    def __init__(self, config, *, forecaster=None):
        super().__init__(config, forecaster=forecaster)
        self.forecaster.requires_grad_(False)
        self.forecaster.predictor.requires_grad_(True)
        if getattr(self.forecaster, "event_encoder", None) is not None:
            self.forecaster.event_encoder.requires_grad_(True)
        self.detector = None
        if config.get("resume_detector_checkpoint"):
            from ..models.checkpoints import component_state, read_checkpoint
            from ..models.detection.detector import FutureDetector
            saved = read_checkpoint(config["resume_detector_checkpoint"])
            self.detector = FutureDetector(**saved["detector_architecture"])
            self.detector.load_state_dict(component_state(saved, "detector."), strict=True)
            self.detector.eval().requires_grad_(False)
        elif config["model"].get("detector_checkpoint"):
            from ..models.checkpoints import load_shared_detector
            self.detector = load_shared_detector(config["model"]["detector_checkpoint"], self.forecaster)
        options = config["optimization"]
        self.predictor_dropout = options.get("predictor_dropout", True)
        self.smooth_l1_beta = options.get("smooth_l1_beta", 1.0)
        if not math.isfinite(self.smooth_l1_beta) or self.smooth_l1_beta <= 0:
            raise ValueError("smooth_l1_beta must be finite and positive")
        if hasattr(self.forecaster.predictor, "activation_checkpointing"):
            self.forecaster.predictor.activation_checkpointing = options.get("activation_checkpointing", True)
        self.align_weight = options.get("align_weight", 0.1)
        self.align_cosine_weight = options.get("align_cosine_weight", 0.2)
        if any(not math.isfinite(x) or x < 0 for x in (self.align_weight, self.align_cosine_weight)):
            raise ValueError("Alignment weights must be finite and nonnegative")
        self.train()
        self.validation_totals = None

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        if self.detector is not None:
            checkpoint["detector_architecture"] = deepcopy(self.detector.architecture)

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        expected = self.detector.architecture if self.detector is not None else None
        if checkpoint.get("detector_architecture") != expected:
            raise ValueError("Frozen validation head differs from the resume checkpoint")

    def train(self, mode=True):
        super().train(mode)
        self.forecaster.encoder.eval()
        self.forecaster.predictor.train(mode and self.predictor_dropout)
        if getattr(self.forecaster, "event_encoder", None) is not None:
            self.forecaster.event_encoder.train(mode)
        if getattr(self, "detector", None) is not None:
            self.detector.eval()
        return self

    def _encode_history(self, history_rgb):
        batch, frames = history_rgb.shape[:2]
        with torch.no_grad():
            history = self.forecaster.encode_latents(history_rgb.flatten(0, 1))
        return history.reshape(batch, frames, *history.shape[1:])

    def _predict(self, history, events=None):
        if getattr(self.forecaster, "event_encoder", None) is None:
            if events is not None:
                raise ValueError("RGB baseline cannot consume events")
            return self.forecaster.predictor.predict_sequence(history), None
        if events is None:
            raise ValueError("Event-RGB prediction requires history_events")
        encoded = self.forecaster.encode_events(events)
        predicted = self.forecaster.predictor.predict_sequence(
            history, event_context=encoded, event_grid_size=self.forecaster.event_encoder.output_grid)
        return predicted, encoded

    def forward(self, history_rgb, history_events=None):
        return self._predict(self._encode_history(history_rgb), history_events)[0]

    def feature_outputs(self, batch):
        from .losses import event_alignment_loss
        if "future_rgb_target" not in batch:
            raise ValueError("Predictor training requires a separate future_rgb_target")
        history = self._encode_history(batch["history_rgb"])
        with torch.no_grad():
            future = self.forecaster.encode_latents(batch["future_rgb_target"])
        prediction, event_features = self._predict(history, batch.get("history_events"))
        with torch.no_grad():
            target = torch.cat((history[:, 1:], future[:, None]), dim=1)
        if prediction.shape != target.shape:
            raise ValueError("Predicted and teacher next-frame latent shapes differ")
        pred_loss = F.smooth_l1_loss(prediction.float(), target.float(), beta=self.smooth_l1_beta)
        align = pred_loss.new_zeros(())
        if event_features is not None and self.align_weight:
            if "event_activity" not in batch:
                raise ValueError("Event alignment requires event_activity from raw counts")
            event_grid = self.forecaster.event_encoder.output_grid
            aligned = event_features
            if tuple(event_grid) != tuple(self.forecaster.grid_size):
                b, t, _, d = aligned.shape
                aligned = F.interpolate(aligned.reshape(b*t, *event_grid, d).permute(0,3,1,2),
                                        size=self.forecaster.grid_size, mode="bilinear", align_corners=False)
                aligned = aligned.permute(0,2,3,1).reshape(b,t,-1,d)
            align = event_alignment_loss(aligned, history, batch["event_activity"],
                                         beta=self.smooth_l1_beta, cosine_weight=self.align_cosine_weight)
        losses = dict(prediction=pred_loss, alignment=align, total=pred_loss+self.align_weight*align)
        return losses, prediction, target

    def feature_loss(self, batch):
        return self.feature_outputs(batch)[0]["total"]

    def training_step(self, batch, batch_idx):
        losses, _, _ = self.feature_outputs(batch)
        batch_size = len(batch["history_rgb"])
        log_training_losses(self, {f"train_{key}_loss": value for key, value in losses.items()},
                            batch_size=batch_size, progress_metric="train_total_loss")
        return self._training_loss(losses["total"], batch_size)

    def on_validation_epoch_start(self):
        self.validation_totals = torch.zeros(4, device=self.device, dtype=torch.float64)
        self.region_totals = torch.zeros(3, 3, device=self.device, dtype=torch.float64)
        self.validation_predictions = []

    def validation_step(self, batch, batch_idx):
        from ..evaluation.latent import latent_region_totals
        from ..models.detection.detector import postprocess
        if "val" in self.datasets and self.global_rank >= len(self.datasets["val"]):
            return
        losses, prediction, target = self.feature_outputs(batch)
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError("Non-finite validation feature loss")
        batch_size = len(batch["history_rgb"])
        self.validation_totals[:3] += torch.stack([losses[k].detach().double() for k in ("prediction", "alignment", "total")]) * batch_size
        self.validation_totals[3] += batch_size
        if "targets" in batch:
            totals = latent_region_totals(prediction[:,-1], target[:,-1], batch["targets"], self.forecaster.grid_size)
            self.region_totals += torch.stack(list(totals.values()))
        if self.detector is not None:
            for pred, meta in zip(postprocess(self.detector(prediction[:,-1])), batch["meta"], strict=True):
                self.validation_predictions.append(dict(sample_id=meta["sample_id"], target_id=meta["target_id"],
                    boxes=pred["boxes"].float().cpu().tolist(), class_ids=pred["labels"].cpu().tolist(),
                    scores=pred["scores"].float().cpu().tolist()))

    def on_validation_epoch_end(self):
        if _distributed():
            dist.all_reduce(self.validation_totals)
            dist.all_reduce(self.region_totals)
        count = self.validation_totals[3].item()
        if count <= 0:
            raise RuntimeError("No validation samples were evaluated")
        values = self.validation_totals[:3].tolist()
        metrics = {f"{key}_loss": value/count for key,value in zip(("prediction", "alignment", "total"),values)}
        for name, totals in zip(("all", "foreground", "background"), self.region_totals.tolist()):
            l1, cosine, patches = totals
            if patches:
                metrics[f"{name}_l1"] = l1/patches
                metrics[f"{name}_cosine"] = cosine/patches
        self.last_report = {"metrics": metrics, "metadata": {"num_samples": int(count), "stage": self.stage,
            "supervision": "all_shifted_next_frames", "smooth_l1_beta": self.smooth_l1_beta,
            "align_weight": self.align_weight, "input_spec": self.input_spec}}
        if self.detector is not None:
            detection = _detection_report(self, self.validation_predictions)
            self.last_report["metrics"].update(detection["metrics"])
            self.last_report["per_class"] = detection.get("per_class", {})
            self.last_report["metadata"]["detection"] = detection["metadata"]
        self._log_validation_report()
        self.last_validation_step = int(self.global_step)
