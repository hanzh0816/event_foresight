"""Shared Lightning task, training state and distributed validation helpers."""

from copy import deepcopy
import math

from lightning.pytorch import LightningModule
import torch
from torch import distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from ..data.samplers import ValidationSampler
from ..models.checkpoints import TRAINING_FORMAT, input_spec, validate_forecaster
from .monitoring import validation_scalars, write_validation_report

CHECKPOINT_FORMAT = TRAINING_FORMAT


def _distributed():
    return dist.is_available() and dist.is_initialized()


class _ForecastTask(LightningModule):
    """Shared data loading, example-count normalization and checkpoint metadata."""

    stage = None

    def __init__(self, config, *, forecaster=None, encoder=None):
        super().__init__()
        self.config = deepcopy(config)
        if self.config["model"].get("rgb_checkpoint"):
            self.config["optimization"]["rgb_finetune"] = True
        self.save_hyperparameters({"config": self.config})
        data = config["data"]
        if self.stage == "detector":
            if forecaster is not None:
                raise ValueError("Detector training accepts an encoder, not a forecaster")
            if encoder is None:
                from ..models.encoder import DINOFeatureEncoder
                encoder = DINOFeatureEncoder(
                    config["model"]["encoder_checkpoint"], image_size=data["image_size"],
                    chunk_size=config.get("runtime", {}).get("encoder_chunk_size", 1))
            self.encoder = encoder
            model = self.encoder
        else:
            if encoder is not None:
                raise ValueError("Predictor training owns its encoder through the forecaster")
            if forecaster is None:
                from ..models.forecaster import DINOWorldForecaster
                forecaster = DINOWorldForecaster(
                    config["model"]["encoder_checkpoint"], image_size=data["image_size"],
                    history_length=data.get("context_frames", 4),
                    encoder_chunk_size=config.get("runtime", {}).get("encoder_chunk_size", 1),
                    predictor_options=config["model"].get("predictor", {}),
                    event_options=config["model"].get("event", {}))
            self.forecaster = forecaster
            predictor_checkpoint = config["model"].get("predictor_checkpoint")
            if predictor_checkpoint:
                from ..models.checkpoints import load_predictor
                load_predictor(self.forecaster.predictor, predictor_checkpoint, forecaster=self.forecaster)
            if config["model"].get("rgb_checkpoint"):
                from ..models.checkpoints import load_rgb_initializer
                load_rgb_initializer(self.forecaster, config["model"]["rgb_checkpoint"])
            model = self.forecaster
        self.datasets = {}
        self.accumulated_examples = 0
        self.last_report = None
        self.last_validation_step = None
        self.input_spec = input_spec(config, model)

    def _loader(self, split):
        from ..data import dataset_from_config, collate_rgb
        data = self.config["data"]
        if split not in self.datasets:
            self.datasets[split] = dataset_from_config(
                self.config, split, future_rgb=True, prediction_frames=1,
                target_only=self.stage == "detector")
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
        if optimizer is not None:
            groups = optimizer.param_groups
            if self.stage == "predictor" and getattr(self.forecaster, "event_encoder", None) is not None:
                names = ("predictor", "event_fusion", "event_encoder") if len(groups) == 3 else ("predictor", "event_encoder")
            else:
                names = (self.stage,)
            for index, group in enumerate(groups):
                name = group.get("name", names[index] if index < len(names) else f"group_{index}")
                self.log(f"lr_{name}", group["lr"], on_step=True, on_epoch=False, sync_dist=False)
        self.accumulated_examples = 0

    def _log_validation_report(self):
        for name, value in validation_scalars(self.last_report).items():
            self.log(name, value, prog_bar=name in ("val_AP", "val_feature_loss"), sync_dist=False)
        if self.global_rank == 0 and self.config.get("output_dir"):
            write_validation_report(self.config["output_dir"], self.last_report,
                                    epoch=self.current_epoch, global_step=self.global_step,
                                    log_dir=getattr(self.logger, "log_dir", None))

    def configure_optimizers(self):
        options = self.config["optimization"]
        betas = tuple(options.get("betas", (0.9, 0.95)))
        event_encoder = (getattr(self.forecaster, "event_encoder", None)
                         if self.stage == "predictor" else None)
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
        # Keep the complete stage model, including its frozen feature encoder.
        checkpoint["foresight_format"] = CHECKPOINT_FORMAT
        checkpoint["stage"] = self.stage
        checkpoint["config"] = deepcopy(self.config)
        checkpoint["input_spec"] = deepcopy(self.input_spec)
        checkpoint["training_spec"] = self._training_spec()
        if self.stage == "predictor":
            checkpoint["forecaster_architecture"] = deepcopy(self.forecaster.architecture)
        else:
            checkpoint["encoder_architecture"] = deepcopy(self.encoder.architecture)

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("foresight_format") != CHECKPOINT_FORMAT:
            raise ValueError("--resume requires a full event-foresight training checkpoint")
        if checkpoint.get("stage") != self.stage:
            raise ValueError("Cannot resume a checkpoint from a different training stage")
        if self.stage == "predictor":
            validate_forecaster(checkpoint, self.forecaster)
        elif checkpoint.get("encoder_architecture") != self.encoder.architecture:
            raise ValueError("Checkpoint DINO encoder architecture differs")
        if checkpoint.get("input_spec") != self.input_spec:
            raise ValueError("Training checkpoint and configured RGB data/feature protocol differ")
        if checkpoint.get("training_spec") != self._training_spec():
            raise ValueError("Resume requires the saved batch, accumulation, GPU count and training sample selection")


def _detection_report(task, predictions):
    from ..evaluation import FutureDetectionEvaluator
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
