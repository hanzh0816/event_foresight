"""Adapt only the Predictor to a fixed Event Encoder and true future RGB targets."""

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .data.events import EventDataset, encoding_spec
from .data.rgb import preprocess_rgb
from .event import load_event_encoder
from .models.event_encoder import EventEncoder
from .training import _RGBTask, PredictorTask


class EventPredictorDataset(EventDataset):
    """Four causal Event windows; only the later RGB frame is read as a target."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, alignment=False, **kwargs)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        name, end = self.windows[index]
        sample["future_rgb_target"] = preprocess_rgb(
            self._reader(name).read_image(end + self.prediction_frames).image, self.image_size)
        return sample


def collate_event_predictor(samples):
    keys = ("history_events", "future_rgb_target")
    return {**{key: torch.stack([sample[key] for sample in samples]) for key in keys},
            "meta": [sample["meta"] for sample in samples]}


class EventPredictorTask(_RGBTask):
    """Checkpoint stage distinct from RGB Predictor and Event Encoder training."""

    stage = "event_predictor"

    def __init__(self, config, *, forecaster=None, event_encoder=None, resume=False):
        config = deepcopy(config)
        options = config["optimization"]
        if options.get("trainable_encoder", False):
            raise ValueError("Event Predictor adaptation requires a frozen Event Encoder")
        if options.get("current_weight", 0) != 0 or options.get("future_weight", 1) != 1:
            raise ValueError("Event Predictor adaptation uses only unweighted future SmoothL1")
        source = config["model"].get("event_encoder_checkpoint")
        if not source and not resume:
            raise ValueError("Provide the trained full_joint --encoder-checkpoint, or --resume")
        self.predictor_dropout = options.get("predictor_dropout", False)
        self.event_chunk_size = int(config.get("runtime", {}).get("event_encoder_chunk_size", 1))
        if self.event_chunk_size < 1:
            raise ValueError("event_encoder_chunk_size must be positive")
        super().__init__(config, forecaster=forecaster)
        self.event_encoding = encoding_spec(config["data"])
        self.event_encoder = event_encoder if event_encoder is not None else EventEncoder(self.forecaster)
        if source and not resume:
            _, encoding = load_event_encoder(self.event_encoder, source)
            if encoding != self.event_encoding:
                raise ValueError("Frozen Event Encoder checkpoint and configured event encoding differ")
        self.forecaster.requires_grad_(False)
        self.event_encoder.requires_grad_(False)
        self.forecaster.predictor.requires_grad_(True)
        if hasattr(self.forecaster.predictor, "activation_checkpointing"):
            self.forecaster.predictor.activation_checkpointing = options.get("activation_checkpointing", True)
        self.input_spec.update(event_encoding=self.event_encoding, training_objective="event_predictor",
                               input_modality="events", trainable_encoder=False, trainable_predictor=True)
        self.initialization = {"event_encoder_checkpoint": source,
                               "rgb_checkpoint": config["model"]["dinof_checkpoint"],
                               "predictor_checkpoint": config["model"].get("predictor_checkpoint")}
        self._awaiting_resume = resume
        self._gradient_audited = False
        self.microbatches_seen = self.examples_seen = 0
        self.validation_totals = None
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.forecaster.eval()
        if hasattr(self, "event_encoder"):
            self.event_encoder.eval()
            self.forecaster.predictor.train(mode and self.predictor_dropout)
        return self

    def on_fit_start(self):
        if self._awaiting_resume:
            raise ValueError("Resume construction requires Trainer.fit(..., ckpt_path=...)")
        super().on_fit_start()

    def _make_dataset(self, split):
        data = self.config["data"]
        return EventPredictorDataset(
            **{key: data[key] for key in ("root", "image_size", "context_frames",
                                         "sample_history_frames", "tolerance_us")},
            split=split, sample_ids=data.get(f"{split}_sample_ids"),
            sequences=data.get(f"{split}_sequences"), stride=data.get(f"{split}_stride", 1),
            **self.event_encoding)

    def _collate_fn(self):
        return collate_event_predictor

    def forward(self, history_events):
        if history_events.ndim != 5 or history_events.shape[1] != self.config["data"]["context_frames"]:
            raise ValueError("Expected Event histories [B,T,10,H,W]")
        with torch.no_grad():
            history = torch.cat([self.event_encoder(part)
                                 for part in history_events.flatten(0, 1).split(self.event_chunk_size)])
        history = history.reshape(*history_events.shape[:2], *history.shape[1:])
        return self.forecaster.predictor(history)

    def feature_loss(self, batch):
        if "future_rgb_target" not in batch:
            raise ValueError("Event Predictor training requires a separate future_rgb_target")
        prediction = self(batch["history_events"])
        with torch.no_grad():
            target = self.forecaster.encode_latents(batch["future_rgb_target"])
        if prediction.shape != target.shape:
            raise ValueError("Predicted and teacher future latent shapes differ")
        return F.smooth_l1_loss(prediction.float(), target.float())

    def training_step(self, batch, batch_idx):
        count = len(batch["history_events"])
        self.microbatches_seen += 1
        self.examples_seen += count
        if self.config.get("output_dir") and "meta" in batch:
            filename = "sample_order.jsonl" if self.global_rank == 0 else f"sample_order_rank{self.global_rank}.jsonl"
            path = Path(self.config["output_dir"]) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as handle:
                handle.write(json.dumps({"global_step": int(self.global_step), "epoch": int(self.current_epoch),
                    "batch_idx": batch_idx, "rank": self.global_rank,
                    "sample_ids": [item["sample_id"] for item in batch["meta"]]}) + "\n")
        loss = self.feature_loss(batch)
        self.log("train_feature_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 sync_dist=True, batch_size=count)
        return self._training_loss(loss, count)

    on_validation_epoch_start = PredictorTask.on_validation_epoch_start
    on_validation_epoch_end = PredictorTask.on_validation_epoch_end
    _write_report = PredictorTask._write_report

    def validation_step(self, batch, batch_idx):
        if "val" in self.datasets and self.global_rank >= len(self.datasets["val"]):
            return
        loss = self.feature_loss(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite Event Predictor validation loss")
        count = len(batch["history_events"])
        self.validation_totals[0] += loss.detach().double() * count
        self.validation_totals[1] += count

    def on_before_optimizer_step(self, optimizer):
        super().on_before_optimizer_step(optimizer)
        if self._gradient_audited:
            return
        gradients = [p.grad.detach() for p in self.forecaster.predictor.parameters() if p.grad is not None]
        norm = torch.stack([g.float().norm() for g in gradients]).norm().item() if gradients else 0.0
        frozen = [name for name, p in self.named_parameters() if not p.requires_grad and p.grad is not None]
        audit = {"global_step": int(self.global_step), "predictor_gradient_norm": norm,
                 "predictor_parameters_with_grad": len(gradients), "frozen_parameters_with_grad": frozen,
                 "event_encoder_trainable": False, "event_encoder_eval": not self.event_encoder.training,
                 "predictor_dropout": self.predictor_dropout}
        if self.global_rank == 0 and self.config.get("output_dir"):
            path = Path(self.config["output_dir"]) / "gradient_audit.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(audit, indent=2) + "\n")
        if frozen or not math.isfinite(norm) or norm <= 0:
            raise RuntimeError(f"Event Predictor gradient audit failed: {audit}")
        self._gradient_audited = True

    def on_train_start(self):
        if self.global_rank != 0 or not self.config.get("output_dir"):
            return
        splits = {split: {"num_samples": len(dataset), "full_inventory_size": dataset.full_inventory_size,
                  "sequence_counts": dict(Counter(name for name, _ in dataset.windows)),
                  "ordered_sample_ids_sha256": hashlib.sha256("\n".join(dataset.sample_ids).encode()).hexdigest()}
                  for split, dataset in self.datasets.items()}
        report = {"stage": self.stage, "initialization": self.initialization, "splits": splits,
                  "train_sampler": "torch.utils.data.DistributedSampler", "sampler_drop_last": False,
                  "loader_drop_last": False, "world_size": self.trainer.world_size,
                  "train_padding_per_epoch": ((-len(self.datasets["train"])) % self.trainer.world_size
                                               if "train" in self.datasets else None)}
        folder = Path(self.config["output_dir"])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "data_inventory.json").write_text(json.dumps(report, indent=2) + "\n")

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["event_encoding"] = deepcopy(self.event_encoding)
        checkpoint["event_predictor_initialization"] = deepcopy(self.initialization)
        checkpoint["event_predictor_progress"] = {"microbatches_seen_per_rank": self.microbatches_seen,
                                                  "examples_seen_per_rank": self.examples_seen}

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        self.initialization = deepcopy(checkpoint["event_predictor_initialization"])
        progress = checkpoint["event_predictor_progress"]
        self.microbatches_seen = progress["microbatches_seen_per_rank"]
        self.examples_seen = progress["examples_seen_per_rank"]
        self._awaiting_resume = False

    def on_train_end(self):
        if self.global_rank == 0 and self.config.get("output_dir"):
            report = {"optimizer_steps": int(self.global_step),
                      "microbatches_seen_per_rank": self.microbatches_seen,
                      "examples_seen_per_rank": self.examples_seen, "world_size": self.trainer.world_size,
                      "examples_seen_global_including_sampler_padding": self.examples_seen * self.trainer.world_size,
                      "initialization": self.initialization}
            (Path(self.config["output_dir"]) / "training_progress.json").write_text(
                json.dumps(report, indent=2) + "\n")
