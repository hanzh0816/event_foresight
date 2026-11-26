"""Train a shared DINO detection head or the v3.1 RGB/Event-RGB predictor."""

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


class _ForecastTask(LightningModule):
    """Shared data loading, example-count normalization and checkpoint metadata."""

    stage = None

    def __init__(self, config, *, forecaster=None):
        super().__init__()
        self.config = deepcopy(config)
        if self.config["model"].get("rgb_checkpoint"):
            self.config["optimization"]["rgb_finetune"] = True
        self.save_hyperparameters({"config": self.config})
        if forecaster is None:
            from .models.dino_world import DINOWorldForecaster
            data = config["data"]
            forecaster = DINOWorldForecaster(
                config["model"]["encoder_checkpoint"], image_size=data["image_size"],
                history_length=data.get("context_frames", 4),
                frame_interval_s=data.get("frame_interval_s", 0.05),
                encoder_chunk_size=config.get("runtime", {}).get("encoder_chunk_size", 1),
                predictor_options=config["model"].get("predictor", {}),
                event_options=config["model"].get("event", {}))
        self.forecaster = forecaster
        predictor_checkpoint = config["model"].get("predictor_checkpoint")
        if predictor_checkpoint:
            from .checkpoints import load_predictor
            load_predictor(self.forecaster.predictor, predictor_checkpoint, forecaster=self.forecaster)
        if config["model"].get("rgb_checkpoint"):
            from .checkpoints import load_rgb_initializer
            load_rgb_initializer(self.forecaster, config["model"]["rgb_checkpoint"])
        self.datasets = {}
        self.accumulated_examples = 0
        self.last_report = None
        self.last_validation_step = None
        self.input_spec = input_spec(config, forecaster)

    def _loader(self, split):
        from .data import dataset_from_config, collate_rgb
        data = self.config["data"]
        if split not in self.datasets:
            self.datasets[split] = dataset_from_config(self.config, split, future_rgb=True, prediction_frames=1)
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
        betas = tuple(options.get("betas", (0.9, 0.95)))
        event_encoder = getattr(self.forecaster, "event_encoder", None)
        if self.stage == "predictor" and event_encoder is not None:
            event_params = list(event_encoder.parameters())
            event_ids = {id(p) for p in event_params}
            main_params = [p for p in self.trainable_parameters() if id(p) not in event_ids]
            if options.get("rgb_finetune", bool(self.config["model"].get("rgb_checkpoint"))):
                shared, fusion = [], []
                for name, parameter in self.forecaster.predictor.named_parameters():
                    (fusion if "event_" in name or "gamma" in name else shared).append(parameter)
                groups = [{"params": shared, "lr": options.get("finetune_lr", 2e-5)},
                          {"params": fusion, "lr": options.get("fusion_lr", 1e-4)},
                          {"params": event_params, "lr": options.get("finetune_event_lr", 1e-4)}]
            else:
                groups = [{"params": main_params, "lr": options["lr"]},
                          {"params": event_params, "lr": options.get("event_lr", 2e-4)}]
        else:
            groups = self.trainable_parameters()
        optimizer = torch.optim.AdamW(groups, lr=options["lr"], betas=betas,
                                      weight_decay=options["weight_decay"])
        schedule = options.get("scheduler", "cosine")
        if schedule == "constant":
            return optimizer
        if schedule != "cosine":
            raise ValueError(f"Unknown learning-rate scheduler: {schedule}")
        total = self.trainer.estimated_stepping_batches
        if not math.isfinite(total) or total < 1:
            raise ValueError("Cosine schedule requires a finite positive number of optimizer steps")
        warmup = math.ceil(total * options.get("warmup_fraction", 0.05))

        minimum = options.get("min_lr", 1e-6)
        def schedule_for(base_lr):
            if not 0 <= minimum <= base_lr:
                raise ValueError("min_lr must lie between zero and each group learning rate")
            def factor(step):
                if step < warmup:
                    return (step + 1) / max(1, warmup)
                progress = (step - warmup) / max(1, total - warmup - 1)
                return minimum / base_lr + (1 - minimum / base_lr) * 0.5 * (1 + math.cos(math.pi * min(1, progress)))
            return factor
        return {"optimizer": optimizer, "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, [schedule_for(g["lr"]) for g in optimizer.param_groups]),
            "interval": "step"}}

    def _training_spec(self):
        data = self.config["data"]
        return {
            "batch_size_per_gpu": data["batch_size"],
            "accumulation_steps": self.config["optimization"].get("accumulation_steps", 1),
            "world_size": self.trainer.world_size if self._trainer is not None else None,
            "horizontal_flip_prob": data.get("horizontal_flip_prob", 0.5),
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
            from .checkpoints import component_state, read_checkpoint
            from .models.detector import FutureDetector
            saved = read_checkpoint(config["resume_detector_checkpoint"])
            self.detector = FutureDetector(**saved["detector_architecture"])
            self.detector.load_state_dict(component_state(saved, "detector."), strict=True)
            self.detector.eval().requires_grad_(False)
        elif config["model"].get("detector_checkpoint"):
            from .checkpoints import load_shared_detector
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
        prediction, event_features = self._predict(history, batch.get("history_events"))
        with torch.no_grad():
            future = self.forecaster.encode_latents(batch["future_rgb_target"])
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
        self.log_dict({f"train_{key}_loss": value for key, value in losses.items()},
                      on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return self._training_loss(losses["total"], batch_size)

    def on_validation_epoch_start(self):
        self.validation_totals = torch.zeros(4, device=self.device, dtype=torch.float64)
        self.region_totals = torch.zeros(3, 3, device=self.device, dtype=torch.float64)
        self.validation_predictions = []

    def validation_step(self, batch, batch_idx):
        from .losses import latent_region_totals
        from .models.detector import postprocess
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
        metrics["next_frame_smooth_l1"] = metrics["prediction_loss"]
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
            self.log("val_AP", detection["metrics"]["AP"], prog_bar=True, sync_dist=False)
        self.log("val_feature_loss", metrics["prediction_loss"], prog_bar=True, sync_dist=False)
        if "foreground_l1" in metrics:
            self.log("val_foreground_l1", metrics["foreground_l1"], sync_dist=False)
        self.log("val_samples", count, sync_dist=False)
        self._write_report()
        self.last_validation_step = int(self.global_step)

    def _write_report(self):
        if self.global_rank == 0 and self.config.get("output_dir"):
            path = Path(self.config["output_dir"]) / "metrics_val.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.last_report, indent=2) + "\n")


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

    def forward(self, rgb):
        if rgb.ndim != 4:
            raise ValueError("Shared detector training requires a real target RGB frame [B,3,H,W]")
        with torch.no_grad():
            features = self.forecaster.encode_latents(rgb)
        return self.detector(features)

    @torch.no_grad()
    def predict(self, rgb):
        from .models.detector import postprocess
        return postprocess(self(rgb))

    def training_step(self, batch, batch_idx):
        losses = self.criterion(self(batch["future_rgb_target"]), batch["targets"], sync_num_boxes=True)
        batch_size = len(batch["history_rgb"])
        self.log_dict({f"train_{key}": value for key, value in losses.items()}, on_step=True,
                      on_epoch=True, sync_dist=True, batch_size=batch_size)
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
        if self.global_rank == 0:
            report["metadata"].update(stage=self.stage, feature_source="real_frozen_dinov2",
                                      input_spec=self.input_spec)
            if self.config.get("output_dir"):
                path = Path(self.config["output_dir"]) / "metrics_val.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(report, indent=2) + "\n")
        self.last_report = report
        self.log_dict({f"val_{key}": float(value) for key, value in report["metrics"].items()
                       if value is not None}, prog_bar=True, sync_dist=False)
        self.log("val_samples", float(report["metadata"]["num_samples"]), sync_dist=False)
        self.validation_predictions = []
        self.last_validation_step = int(self.global_step)


def _detection_report(task, predictions):
    from .evaluation import FutureDetectionEvaluator
    gathered = [None] * task.trainer.world_size if task.global_rank == 0 else None
    if _distributed():
        dist.gather_object(predictions, gathered, dst=0)
    else:
        gathered = [predictions]
    report = None
    if task.global_rank == 0:
        combined = [item for items in gathered for item in items]
        if not combined:
            raise RuntimeError("No detection validation samples were evaluated")
        evaluator = FutureDetectionEvaluator(task.datasets["val"], sample_ids=[p["sample_id"] for p in combined])
        evaluator.update(combined)
        report = evaluator.compute().to_dict()
    if _distributed():
        reports = [report]
        dist.broadcast_object_list(reports, src=0)
        report = reports[0]
    return report
