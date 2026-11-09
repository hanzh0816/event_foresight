"""Make event representations predictable through the frozen RGB predictor.

The predictor is frozen by default, but its forward is deliberately outside
no_grad: true future RGB supervision must reach every event history encoder.
"""

from copy import deepcopy
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data.events import EventDataset
from .data.rgb import preprocess_rgb
from .event_training import EventTask


class FutureEventDataset(EventDataset):
    """Causal event histories plus separately named RGB training targets."""

    def __init__(self, *args, current_scope="all", **kwargs):
        if current_scope not in ("all", "last"):
            raise ValueError("current_scope must be all or last")
        self.current_scope = current_scope
        super().__init__(*args, alignment=False, **kwargs)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        name, end = self.windows[index]
        reader = self._reader(name)
        indices = range(end - self.context_frames + 1, end + 1) if self.current_scope == "all" else [end]
        sample["history_rgb_target"] = torch.stack([
            preprocess_rgb(reader.read_image(i).image, self.image_size) for i in indices])
        sample["future_rgb_target"] = preprocess_rgb(
            reader.read_image(end + self.prediction_frames).image, self.image_size)
        return sample


def collate_future_event(samples):
    keys = ("history_events", "history_rgb_target", "future_rgb_target")
    return {**{key: torch.stack([sample[key] for sample in samples]) for key in keys},
            "meta": [sample["meta"] for sample in samples]}


class FutureEventTask(EventTask):
    """RGB initialization (or explicit Event warm start) with a frozen RGB backend.

    Keep stage='event' so existing Event Encoder loaders accept the checkpoint.
    input_spec records the different objective and prevents cross-task resumes.
    """

    def __init__(self, config, *, forecaster=None, event_encoder=None):
        config = deepcopy(config)
        options = config["optimization"]
        options.setdefault("gradient_checkpointing", True)
        options.setdefault("activation_checkpointing", True)
        self.current_weight = float(options.get("current_weight", 1.0))
        self.future_weight = float(options.get("future_weight", 1.0))
        if (not all(math.isfinite(v) and v >= 0 for v in (self.current_weight, self.future_weight))
                or self.current_weight + self.future_weight <= 0):
            raise ValueError("Current/future weights must be finite, nonnegative and not both zero")
        self.current_scope = options.get("current_scope", "all")
        if self.current_scope not in ("all", "last"):
            raise ValueError("current_scope must be all or last")
        if options.get("trainable_predictor", False) is not False:
            raise ValueError("This task supports only a frozen Predictor; event inference restores Encoder weights only")
        self.trainable_predictor = False
        self.event_chunk_size = int(config.get("runtime", {}).get("event_encoder_chunk_size", 1))
        if self.event_chunk_size <= 0:
            raise ValueError("event_encoder_chunk_size must be positive")
        super().__init__(config, forecaster=forecaster, event_encoder=event_encoder)
        self.forecaster.predictor.requires_grad_(self.trainable_predictor)
        if hasattr(self.forecaster.predictor, "activation_checkpointing"):
            self.forecaster.predictor.activation_checkpointing = options["activation_checkpointing"]
        self.input_spec.update(training_objective="future_event", current_scope=self.current_scope,
                               current_weight=self.current_weight, future_weight=self.future_weight,
                               trainable_predictor=self.trainable_predictor)
        self.train()
        self._gradient_audited = False
        self.microbatches_seen = 0
        self.examples_seen = 0
        event_checkpoint = config["model"].get("event_encoder_checkpoint")
        self.initialization = {
            "kind": "event_checkpoint" if event_checkpoint else "rgb_pretrained",
            "checkpoint": event_checkpoint or config["model"]["dinof_checkpoint"],
            "rgb_checkpoint": config["model"]["dinof_checkpoint"],
        }

    def on_train_start(self):
        if self.global_rank != 0 or not self.config.get("output_dir"):
            return
        inventories = {}
        for split, dataset in self.datasets.items():
            ids = dataset.sample_ids
            inventories[split] = {
                "num_samples": len(dataset), "full_inventory_size": dataset.full_inventory_size,
                "sequence_counts": dict(Counter(name for name, _ in dataset.windows)),
                "ordered_sample_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            }
        report = {
            "initialization": self.initialization, "splits": inventories,
            "train_sampler": "torch.utils.data.DistributedSampler",
            "sampler_drop_last": False, "loader_drop_last": False,
            "world_size": self.trainer.world_size,
            "train_padding_per_epoch": ((-len(self.datasets["train"])) % self.trainer.world_size
                                        if "train" in self.datasets else None),
        }
        folder = Path(self.config["output_dir"])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "data_inventory.json").write_text(json.dumps(report, indent=2) + "\n")

    def _make_dataset(self, split):
        data = self.config["data"]
        ids = data.get(f"{split}_sample_ids")
        if split == "val" and ids and data.get("alignment_val_samples"):
            count = min(len(ids), data["alignment_val_samples"])
            ids = [ids[i] for i in np.linspace(0, len(ids) - 1, count, dtype=int)]
        return FutureEventDataset(
            **{key: data[key] for key in ("root", "image_size", "context_frames",
                                         "sample_history_frames", "tolerance_us")},
            split=split, sample_ids=ids, sequences=data.get(f"{split}_sequences"),
            stride=data.get(f"{split}_stride", 1), current_scope=self.current_scope,
            **self.event_encoding)

    def _collate_fn(self):
        return collate_future_event

    def encode_history(self, events):
        if events.ndim != 5 or events.shape[1] != self.config["data"].get("context_frames", 4):
            raise ValueError("Expected event histories [B,T,10,H,W]")
        latents = torch.cat([self.event_encoder(part)
                             for part in events.flatten(0, 1).split(self.event_chunk_size)])
        return latents.reshape(*events.shape[:2], *latents.shape[1:])

    def forward(self, history_events):
        # requires_grad_(False) freezes parameters without severing input gradients.
        return self.forecaster.predictor(self.encode_history(history_events))

    def feature_losses(self, batch):
        if "future_rgb_target" not in batch or "history_rgb_target" not in batch:
            raise ValueError("Future Event training requires separate current and true future RGB targets")
        history = self.encode_history(batch["history_events"])
        observed = history if self.current_scope == "all" else history[:, -1:]
        current_rgb = batch["history_rgb_target"]
        if tuple(current_rgb.shape[:2]) != tuple(observed.shape[:2]):
            raise ValueError("Current RGB targets do not match the selected event history states")
        with torch.no_grad():
            target_current = self.forecaster.encode_latents(current_rgb.flatten(0, 1))
            target_future = self.forecaster.encode_latents(batch["future_rgb_target"])
        current = F.smooth_l1_loss(observed.flatten(0, 1).float(), target_current.float())
        # A current-only control still logs future quality without building an
        # unnecessary predictor backward graph. Encoder history remains attached.
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.future_weight > 0):
            prediction = self.forecaster.predictor(history)
            if prediction.shape != target_future.shape:
                raise ValueError("Predicted and true future latent shapes differ")
            future = F.smooth_l1_loss(prediction.float(), target_future.float())
        total = self.current_weight * current + self.future_weight * future
        return {"loss": total, "current_smooth_l1": current, "future_smooth_l1": future}

    def feature_loss(self, batch):
        return self.feature_losses(batch)["loss"]

    def training_step(self, batch, batch_idx):
        self.microbatches_seen += 1
        self.examples_seen += len(batch["history_events"])
        if self.config.get("output_dir") and "meta" in batch:
            filename = "sample_order.jsonl" if self.global_rank == 0 else f"sample_order_rank{self.global_rank}.jsonl"
            audit_path = Path(self.config["output_dir"]) / filename
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open("a") as handle:
                handle.write(json.dumps({"global_step": int(self.global_step), "batch_idx": batch_idx,
                    "epoch": int(self.current_epoch), "rank": self.global_rank,
                    "sample_ids": [item["sample_id"] for item in batch["meta"]]}) + "\n")
        losses = self.feature_losses(batch)
        count = len(batch["history_events"])
        for name, loss in losses.items():
            self.log("train_" + name, loss, on_step=True, on_epoch=True,
                     prog_bar=name == "loss", sync_dist=True, batch_size=count)
        return self._training_loss(losses["loss"], count)

    def on_before_optimizer_step(self, optimizer):
        super().on_before_optimizer_step(optimizer)
        if self._gradient_audited:
            return
        self._gradient_audited = True
        gradients = [parameter.grad.detach() for parameter in self.event_encoder.parameters()
                     if parameter.grad is not None]
        norm = torch.stack([gradient.float().norm() for gradient in gradients]).norm().item() if gradients else 0.0
        frozen_gradients = [name for name, parameter in self.forecaster.named_parameters()
                            if not parameter.requires_grad and parameter.grad is not None]
        audit = {"global_step": int(self.global_step), "encoder_gradient_norm": norm,
                 "encoder_parameters_with_grad": len(gradients),
                 "frozen_forecaster_parameters_with_grad": frozen_gradients,
                 "predictor_trainable": self.trainable_predictor,
                 "predictor_eval": not self.forecaster.predictor.training,
                 "current_weight": self.current_weight, "future_weight": self.future_weight}
        if self.global_rank == 0 and self.config.get("output_dir"):
            path = Path(self.config["output_dir"]) / "gradient_audit.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(audit, indent=2) + "\n")
        if frozen_gradients or not math.isfinite(norm) or norm <= 0:
            raise RuntimeError(f"Future Event gradient audit failed: {audit}")

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["event_initialization"] = deepcopy(self.initialization)
        checkpoint["future_event_progress"] = {
            "microbatches_seen_per_rank": self.microbatches_seen,
            "examples_seen_per_rank": self.examples_seen,
        }

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        if "event_initialization" in checkpoint:
            self.initialization = deepcopy(checkpoint["event_initialization"])
        else:
            # Older pilots predate this metadata; recover their original source
            # before resume_config clears model initialization paths.
            model = checkpoint.get("config", {}).get("model", {})
            event_checkpoint = model.get("event_encoder_checkpoint")
            self.initialization = {
                "kind": "event_checkpoint" if event_checkpoint else "rgb_pretrained",
                "checkpoint": event_checkpoint or model.get("dinof_checkpoint"),
                "rgb_checkpoint": model.get("dinof_checkpoint"),
            }
        progress = checkpoint.get("future_event_progress", {})
        self.microbatches_seen = progress.get("microbatches_seen_per_rank", 0)
        self.examples_seen = progress.get("examples_seen_per_rank", 0)

    def on_train_end(self):
        if self.global_rank == 0 and self.config.get("output_dir"):
            report = {"optimizer_steps": int(self.global_step),
                      "microbatches_seen_per_rank": self.microbatches_seen,
                      "examples_seen_per_rank": self.examples_seen,
                      "world_size": self.trainer.world_size,
                      "examples_seen_global_including_sampler_padding": self.examples_seen * self.trainer.world_size,
                      "initialization": self.initialization}
            (Path(self.config["output_dir"]) / "training_progress.json").write_text(
                json.dumps(report, indent=2) + "\n")

    def on_validation_epoch_start(self):
        self.validation_totals = torch.zeros(4, device=self.device, dtype=torch.float64)

    def validation_step(self, batch, batch_idx):
        if "val" in self.datasets and self.global_rank >= len(self.datasets["val"]):
            return
        losses = self.feature_losses(batch)
        count = len(batch["history_events"])
        for index, name in enumerate(("loss", "current_smooth_l1", "future_smooth_l1")):
            if not torch.isfinite(losses[name]):
                raise FloatingPointError("Non-finite future Event validation loss")
            self.validation_totals[index] += losses[name].detach().double() * count
        self.validation_totals[3] += count

    def on_validation_epoch_end(self):
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.validation_totals)
        *totals, count = self.validation_totals.tolist()
        if count <= 0:
            raise RuntimeError("No future Event validation samples")
        metrics = {name: total / count for name, total in
                   zip(("loss", "current_smooth_l1", "future_smooth_l1"), totals)}
        for name, value in metrics.items():
            self.log("val_" + name, value, prog_bar=True)
        self.last_report = dict(metrics=metrics, metadata={"num_samples": int(count),
            "stage": "future_event", "current_scope": self.current_scope,
            "current_weight": self.current_weight, "future_weight": self.future_weight,
            "trainable_predictor": self.trainable_predictor})
        self._write_report()
        self.last_validation_step = int(self.global_step)
