"""Two small Lightning tasks: adapt the RGB predictor or train its detection head."""

from copy import deepcopy
import inspect
import json
import math
from pathlib import Path

from lightning.pytorch import LightningModule
import torch
from torch import distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler

from .samplers import ValidationSampler
from .checkpoints import TRAINING_FORMAT, input_spec, validate_forecaster

CHECKPOINT_FORMAT = TRAINING_FORMAT


def _distributed():
    return dist.is_available() and dist.is_initialized()


class _RGBTask(LightningModule):
    """Shared data loading, example-count normalization and checkpoint metadata."""

    stage = None

    def __init__(self, config, *, forecaster=None):
        super().__init__()
        self.config = deepcopy(config)
        self.save_hyperparameters({"config": self.config})
        if forecaster is None:
            from .models.dino_world import DINOWorldForecaster
            data = config["data"]
            forecaster = DINOWorldForecaster(
                config["model"]["encoder_checkpoint"], image_size=data["image_size"],
                history_length=data.get("context_frames", 4),
                frame_interval_s=data.get("frame_interval_s", 0.05),
                encoder_chunk_size=config.get("runtime", {}).get("encoder_chunk_size", 1),
                predictor_options=config["model"].get("predictor", {}))
        self.forecaster = forecaster
        predictor_checkpoint = config["model"].get("predictor_checkpoint")
        if predictor_checkpoint:
            from .checkpoints import load_predictor
            load_predictor(self.forecaster.predictor, predictor_checkpoint, forecaster=self.forecaster)
        self.datasets = {}
        self.accumulated_examples = 0
        self.last_report = None
        self.last_validation_step = None
        self.input_spec = input_spec(config, forecaster)

    def _loader(self, split):
        from .data import RGBDataset, collate_rgb
        data = self.config["data"]
        if split not in self.datasets:
            self.datasets[split] = RGBDataset(
                data["root"], split=split, image_size=data["image_size"],
                sample_history_frames=data.get("sample_history_frames", 40),
                context_frames=data.get("context_frames", 4),
                stride=data.get(f"{split}_stride", 1),
                tolerance_us=data.get("tolerance_us", 1000),
                sequences=data.get(f"{split}_sequences"),
                sample_ids=data.get(f"{split}_sample_ids"),
                future_rgb=self.stage == "predictor")
        dataset = self.datasets[split]
        if not len(dataset):
            raise ValueError(f"No valid DSEC {split} windows")
        sampler_class = DistributedSampler if split == "train" else ValidationSampler
        sampler = sampler_class(dataset, num_replicas=self.trainer.world_size, rank=self.global_rank,
                                shuffle=split == "train", seed=self.config["runtime"].get("seed", 42))
        workers = data.get("num_workers", 8)
        return DataLoader(
            dataset, batch_size=data["batch_size"], sampler=sampler, num_workers=workers,
            collate_fn=collate_rgb, pin_memory=True,
            **({"multiprocessing_context": "spawn", "prefetch_factor": data.get("prefetch_factor", 1),
                "persistent_workers": data.get("persistent_workers", True)} if workers else {}))

    def train_dataloader(self):
        return self._loader("train")

    def val_dataloader(self):
        return self._loader("val")

    def teardown(self, stage):
        for dataset in self.datasets.values():
            dataset.close()

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def on_fit_start(self):
        local_batch = self.config["data"]["batch_size"]
        accumulation = self.trainer.accumulate_grad_batches
        world_size = self.trainer.world_size
        self.print(f"stage={self.stage} per_gpu_batch={local_batch} visible_gpus_used={world_size} "
                   f"accumulation={accumulation} nominal_global_batch={local_batch * accumulation * world_size} "
                   f"trainable_parameters={sum(p.numel() for p in self.trainable_parameters())}")

    def _training_loss(self, loss, batch_size):
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {self.stage} loss")
        self.accumulated_examples += batch_size
        # Lightning divides by accumulation_steps; the optimizer hook accounts
        # for short batches and a partial final accumulation group.
        return loss * batch_size

    def on_before_optimizer_step(self, optimizer):
        if self.accumulated_examples <= 0:
            raise RuntimeError("Optimizer step without accumulated examples")
        count = torch.tensor(float(self.accumulated_examples), device=self.device)
        if _distributed():
            dist.all_reduce(count)
        # DDP averages gradients over ranks; undo that average and Lightning's
        # accumulation divisor before dividing by the actual global examples.
        scale = self.trainer.accumulate_grad_batches * self.trainer.world_size / count.item()
        for parameter in self.trainable_parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
        self.log("train_global_batch", count, on_step=True, on_epoch=False, sync_dist=False)
        self.accumulated_examples = 0

    def configure_optimizers(self):
        options = self.config["optimization"]
        betas = tuple(options.get("betas", (0.9, 0.999) if self.stage == "predictor" else (0.9, 0.95)))
        optimizer = torch.optim.AdamW(self.trainable_parameters(), lr=options["lr"], betas=betas,
                                      weight_decay=options["weight_decay"])
        schedule = options.get("scheduler", "constant" if self.stage == "predictor" else "cosine")
        if schedule == "constant":
            return optimizer
        if schedule == "warmup_constant":
            warmup = options.get("warmup_steps", 5000)
            if type(warmup) is not int or warmup < 0:
                raise ValueError("warmup_steps must be a nonnegative integer")
            def warmup_factor(step):
                return min(1.0, (step + 1) / max(1, warmup))
            return {"optimizer": optimizer, "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_factor), "interval": "step"}}
        if schedule != "cosine":
            raise ValueError(f"Unknown learning-rate scheduler: {schedule}")
        total = self.trainer.estimated_stepping_batches
        if not math.isfinite(total) or total < 1:
            raise ValueError("Cosine schedule requires a finite positive number of optimizer steps")
        warmup = math.ceil(total * options.get("warmup_fraction", 0.05))

        def factor(step):
            if step < warmup:
                return (step + 1) / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1 + math.cos(math.pi * min(1, progress)))

        return {"optimizer": optimizer, "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, factor), "interval": "step"}}

    def _training_spec(self):
        data = self.config["data"]
        return {
            "batch_size_per_gpu": data["batch_size"],
            "accumulation_steps": self.config["optimization"].get("accumulation_steps", 1),
            "world_size": self.trainer.world_size if self._trainer is not None else None,
            "train_stride": data.get("train_stride", 1),
            "train_sequences": data.get("train_sequences"),
            "train_sample_ids": data.get("train_sample_ids"),
        }

    def on_save_checkpoint(self, checkpoint):
        # The complete model state is intentional: resume never silently
        # substitutes a different frozen encoder or predictor.
        checkpoint["foresight_format"] = CHECKPOINT_FORMAT
        checkpoint["stage"] = self.stage
        checkpoint["config"] = deepcopy(self.config)
        checkpoint["input_spec"] = deepcopy(self.input_spec)
        checkpoint["training_spec"] = self._training_spec()
        checkpoint["forecaster_architecture"] = deepcopy(self.forecaster.architecture)

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("foresight_format") != CHECKPOINT_FORMAT:
            raise ValueError("--resume requires a full event-foresight training checkpoint")
        if checkpoint.get("stage") != self.stage:
            raise ValueError("Cannot resume a checkpoint from a different training stage")
        validate_forecaster(checkpoint, self.forecaster)
        if checkpoint.get("input_spec") != self.input_spec:
            raise ValueError("Training checkpoint and configured RGB data/feature protocol differ")
        if checkpoint.get("training_spec") != self._training_spec():
            raise ValueError("Resume requires the saved batch, accumulation, GPU count and training sample selection")


class PredictorTask(_RGBTask):
    """Frozen encoder and causal next-frame teacher forcing across every history step."""

    stage = "predictor"

    def __init__(self, config, *, forecaster=None):
        super().__init__(config, forecaster=forecaster)
        self.forecaster.requires_grad_(False)
        self.forecaster.predictor.requires_grad_(True)
        options = config["optimization"]
        self.predictor_dropout = options.get("predictor_dropout", True)
        self.smooth_l1_beta = options.get("smooth_l1_beta", 0.1)
        if not math.isfinite(self.smooth_l1_beta) or self.smooth_l1_beta <= 0:
            raise ValueError("smooth_l1_beta must be finite and positive")
        if hasattr(self.forecaster.predictor, "activation_checkpointing"):
            self.forecaster.predictor.activation_checkpointing = options.get("activation_checkpointing", True)
        self.train()
        self.validation_totals = None

    def train(self, mode=True):
        super().train(mode)
        self.forecaster.encoder.eval()
        self.forecaster.predictor.train(mode and self.predictor_dropout)
        return self

    def _encode_history(self, history_rgb):
        batch, frames = history_rgb.shape[:2]
        with torch.no_grad():
            history = self.forecaster.encode_latents(history_rgb.flatten(0, 1))
        return history.reshape(batch, frames, *history.shape[1:])

    def forward(self, history_rgb):
        return self.forecaster.predictor.predict_sequence(self._encode_history(history_rgb))

    def feature_loss(self, batch):
        if "future_rgb_target" not in batch:
            raise ValueError("Predictor training requires a separate future_rgb_target")
        history = self._encode_history(batch["history_rgb"])
        prediction = self.forecaster.predictor.predict_sequence(history)
        with torch.no_grad():
            future = self.forecaster.encode_latents(batch["future_rgb_target"])
            target = torch.cat((history[:, 1:], future[:, None]), dim=1)
        if prediction.shape != target.shape:
            raise ValueError("Predicted and teacher next-frame latent shapes differ")
        return F.smooth_l1_loss(prediction.float(), target.float(), beta=self.smooth_l1_beta)

    def training_step(self, batch, batch_idx):
        loss = self.feature_loss(batch)
        batch_size = len(batch["history_rgb"])
        self.log("train_feature_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 sync_dist=True, batch_size=batch_size)
        return self._training_loss(loss, batch_size)

    def on_validation_epoch_start(self):
        self.validation_totals = torch.zeros(2, device=self.device, dtype=torch.float64)

    def validation_step(self, batch, batch_idx):
        if "val" in self.datasets and self.global_rank >= len(self.datasets["val"]):
            return  # Ignore ValidationSampler's placeholder on an empty rank.
        loss = self.feature_loss(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite validation feature loss")
        batch_size = len(batch["history_rgb"])
        self.validation_totals[0] += loss.detach().double() * batch_size
        self.validation_totals[1] += batch_size

    def on_validation_epoch_end(self):
        if _distributed():
            dist.all_reduce(self.validation_totals)
        total, count = self.validation_totals.tolist()
        if count <= 0:
            raise RuntimeError("No RGB validation samples were evaluated")
        loss = total / count
        self.log("val_feature_loss", loss, prog_bar=True, sync_dist=False)
        self.log("val_samples", count, sync_dist=False)
        self.last_report = {"metrics": {"next_frame_smooth_l1": loss},
                            "metadata": {"num_samples": int(count), "stage": self.stage,
                                         "supervision": "all_shifted_next_frames",
                                         "smooth_l1_beta": self.smooth_l1_beta}}
        self._write_report()
        self.last_validation_step = int(self.global_step)

    def _write_report(self):
        if self.global_rank == 0 and self.config.get("output_dir"):
            path = Path(self.config["output_dir"]) / "metrics_val.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.last_report, indent=2) + "\n")


class DetectorTask(_RGBTask):
    """Train a fresh detection head on a trained, frozen DINO World predictor."""

    stage = "detector"

    def __init__(self, config, *, forecaster=None, detector=None, criterion=None,
                 allow_uninitialized_predictor=False):
        if not config["model"].get("predictor_checkpoint") and not allow_uninitialized_predictor:
            raise ValueError("Detector training requires model.predictor_checkpoint from trained DINO World, or --resume")
        super().__init__(config, forecaster=forecaster)
        self.forecaster.requires_grad_(False).eval()
        architecture = deepcopy(getattr(detector, "architecture", None))
        if detector is None:
            from .models.detector import FutureDetector
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
            from .models.criterion import DetectionCriterion
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

    def forward(self, history_rgb):
        with torch.no_grad():
            features = self.forecaster.forecast(history_rgb)
        return self.detector(features)

    @torch.no_grad()
    def predict(self, history_rgb):
        from .models.detector import postprocess
        return postprocess(self(history_rgb))

    def training_step(self, batch, batch_idx):
        losses = self.criterion(self(batch["history_rgb"]), batch["targets"], sync_num_boxes=True)
        batch_size = len(batch["history_rgb"])
        self.log_dict({f"train_{key}": value for key, value in losses.items()}, on_step=True,
                      on_epoch=True, sync_dist=True, batch_size=batch_size)
        return self._training_loss(losses["loss"], batch_size)

    def on_validation_epoch_start(self):
        self.validation_predictions = []

    def validation_step(self, batch, batch_idx):
        if "val" in self.datasets and self.global_rank >= len(self.datasets["val"]):
            return
        predictions = self.predict(batch["history_rgb"])
        for prediction, meta in zip(predictions, batch["meta"]):
            self.validation_predictions.append({
                "sample_id": meta["sample_id"], "target_id": meta["target_id"],
                "boxes": prediction["boxes"].cpu().tolist(),
                "class_ids": prediction["labels"].cpu().tolist(),
                "scores": prediction["scores"].cpu().tolist(),
            })

    def on_validation_epoch_end(self):
        from .evaluation import FutureDetectionEvaluator
        gathered = [None] * self.trainer.world_size if self.global_rank == 0 else None
        if _distributed():
            dist.gather_object(self.validation_predictions, gathered, dst=0)
        else:
            gathered = [self.validation_predictions]
        report = None
        if self.global_rank == 0:
            predictions = [item for rank_items in gathered for item in rank_items]
            if not predictions:
                raise RuntimeError("No RGB detection validation samples were evaluated")
            evaluator = FutureDetectionEvaluator(self.datasets["val"], sample_ids=[p["sample_id"] for p in predictions])
            evaluator.update(predictions)
            report = evaluator.compute().to_dict()
            report["metadata"].update(stage=self.stage, feature_source="frozen_dino_world_prediction",
                                      input_spec=self.input_spec)
            if self.config.get("output_dir"):
                path = Path(self.config["output_dir"]) / "metrics_val.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(report, indent=2) + "\n")
        if _distributed():
            reports = [report]
            dist.broadcast_object_list(reports, src=0)
            report = reports[0]
        self.last_report = report
        self.log_dict({f"val_{key}": float(value) for key, value in report["metrics"].items()
                       if value is not None}, prog_bar=True, sync_dist=False)
        self.log("val_samples", float(report["metadata"]["num_samples"]), sync_dist=False)
        self.validation_predictions = []
        self.last_validation_step = int(self.global_step)
