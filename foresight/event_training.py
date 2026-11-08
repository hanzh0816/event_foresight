"""E2: current RGB-to-event alignment with a frozen RGB teacher and PCA."""

import numpy as np
import torch
from torch.nn import functional as F

from .training import _RGBTask, PredictorTask
from .data.events import EventDataset, collate_event, encoding_spec
from .models.event_encoder import EventEncoder
from .event import load_event_encoder


class EventTask(_RGBTask):
    stage = "event"

    def __init__(self, config, *, forecaster=None, event_encoder=None):
        super().__init__(config, forecaster=forecaster)
        self.forecaster.requires_grad_(False).eval()
        self.event_encoding = encoding_spec(config["data"])
        self.input_spec["event_encoding"] = self.event_encoding
        self.event_encoder = event_encoder if event_encoder is not None else EventEncoder(
            self.forecaster, gradient_checkpointing=config["optimization"].get("gradient_checkpointing", False))
        if config["model"].get("event_encoder_checkpoint"):
            # Explicit weight-only initialization permits another binning protocol;
            # full resume separately requires identical input_spec and optimizer state.
            load_event_encoder(self.event_encoder, config["model"]["event_encoder_checkpoint"])
        self.validation_totals = None
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.forecaster.eval()
        return self

    def _make_dataset(self, split):
        data = self.config["data"]
        ids = data.get(f"{split}_sample_ids")
        if split == "val" and ids and data.get("alignment_val_samples"):
            count = min(len(ids), data["alignment_val_samples"])
            ids = [ids[i] for i in np.linspace(0, len(ids)-1, count, dtype=int)]
        return EventDataset(**{key: data[key] for key in ("root", "image_size", "context_frames",
                                                        "sample_history_frames", "tolerance_us")},
                            split=split, sample_ids=ids, sequences=data.get(f"{split}_sequences"),
                            stride=data.get(f"{split}_stride", 1), alignment=True, **self.event_encoding)

    def _collate_fn(self):
        return collate_event

    def forward(self, events):
        return self.event_encoder(events)

    def feature_loss(self, batch):
        with torch.no_grad():
            target = self.forecaster.encode_latents(batch["rgb"])
        prediction = self(batch["event_voxel"])
        return F.smooth_l1_loss(prediction.float(), target.float())

    def training_step(self, batch, batch_idx):
        loss = self.feature_loss(batch)
        count = len(batch["rgb"])
        self.log("train_alignment_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 sync_dist=True, batch_size=count)
        return self._training_loss(loss, count)

    on_validation_epoch_start = PredictorTask.on_validation_epoch_start
    _write_report = PredictorTask._write_report

    def validation_step(self, batch, batch_idx):
        if "val" in self.datasets and self.global_rank >= len(self.datasets["val"]):
            return
        loss = self.feature_loss(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite alignment validation loss")
        count = len(batch["rgb"])
        self.validation_totals[0] += loss.detach().double() * count
        self.validation_totals[1] += count

    def on_validation_epoch_end(self):
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.validation_totals)
        total, count = self.validation_totals.tolist()
        if count <= 0:
            raise RuntimeError("No alignment validation samples")
        loss = total / count
        self.log("val_alignment_loss", loss, prog_bar=True)
        self.log("val_samples", count)
        self.last_report = dict(metrics={"current_smooth_l1": loss},
                                metadata={"num_samples": int(count), "stage": self.stage})
        self._write_report()
        self.last_validation_step = int(self.global_step)

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["event_encoding"] = self.event_encoding
