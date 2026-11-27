"""Training execution, checkpoint recovery and Lightning configuration."""

from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
import torch
import yaml

from ..config import ROOT, resolve_path
from .base import CHECKPOINT_FORMAT
from .detector import DetectorTask
from .monitoring import TrainingTensorBoardLogger
from .predictor import PredictorTask


class ValidateAtStepBudget(Callback):
    """Request the normal fit-loop validation when a step budget ends mid-epoch."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.max_steps > 0 and trainer.global_step >= trainer.max_steps:
            # Lightning validates on an explicit stop request before leaving fit,
            # so ModelCheckpoint sees the final metric and saves best/last there.
            trainer.should_stop = True


def make_trainer(config, stage, *, limit_train_batches=None, limit_val_batches=None):
    options, runtime = config["optimization"], config["runtime"]
    if limit_train_batches is None:
        limit_train_batches = runtime.get("limit_train_batches", 1.0)
    if limit_val_batches is None:
        limit_val_batches = runtime.get("limit_val_batches", 1.0)
    folder = Path(config["output_dir"])
    checkpoint_options = dict(dirpath=folder / "checkpoints", save_last=True,
                              enable_version_counter=False, auto_insert_metric_name=False)
    if limit_val_batches:
        monitor, mode = ("val_AP", "max") if stage == "detector" or config["model"].get("detector_checkpoint") else ("val_feature_loss", "min")
        checkpoint = ModelCheckpoint(filename="best", monitor=monitor, mode=mode,
                                     save_top_k=1, save_on_train_epoch_end=False, **checkpoint_options)
    else:
        checkpoint = ModelCheckpoint(save_top_k=0, save_on_train_epoch_end=True, **checkpoint_options)
    callbacks = [ValidateAtStepBudget(), checkpoint]
    if limit_val_batches and stage == "predictor":
        callbacks.append(ModelCheckpoint(filename="best_foreground", monitor="val_foreground_l1", mode="min",
                                         save_top_k=1, save_last=False, save_on_train_epoch_end=False,
                                         dirpath=folder / "checkpoints", auto_insert_metric_name=False,
                                         enable_version_counter=False))
    return Trainer(
        accelerator="gpu", devices="auto",
        strategy=DDPStrategy(broadcast_buffers=False) if torch.cuda.device_count() > 1 else "auto",
        precision=runtime.get("precision", "bf16-mixed"),
        max_epochs=options.get("epochs", 40), max_steps=options.get("max_steps", -1),
        accumulate_grad_batches=options.get("accumulation_steps", 1),
        gradient_clip_val=options.get("clip_grad", 1.0),
        default_root_dir=folder, callbacks=callbacks,
        logger=[CSVLogger(folder, name="logs", version=None),
                TrainingTensorBoardLogger(folder, name="tensorboard", version=None,
                                          default_hp_metric=False, stage=stage)],
        limit_train_batches=limit_train_batches, limit_val_batches=limit_val_batches,
        log_every_n_steps=runtime.get("log_every", 10), num_sanity_val_steps=0,
        use_distributed_sampler=False,
    )


def resume_config(config, checkpoint, stage):
    """Restore the training recipe, allowing relocated assets/data and runtime settings."""
    from ..models.checkpoints import read_checkpoint
    saved = read_checkpoint(checkpoint)
    if saved.get("foresight_format") != CHECKPOINT_FORMAT or saved.get("stage") != stage:
        raise ValueError("--resume must be a full training checkpoint from the same stage")
    restored = deepcopy(saved["config"])
    restored["data"]["root"] = config["data"]["root"]
    for key in ("num_workers", "prefetch_factor", "persistent_workers"):
        if key in config["data"]:
            restored["data"][key] = config["data"][key]
    # Seed, precision and encoder chunking affect the data/numerical protocol.
    # Keep them from the checkpoint; only execution/logging knobs may relocate.
    for key in ("cpu_threads", "log_every", "limit_train_batches", "limit_val_batches"):
        if key in config["runtime"]:
            restored["runtime"][key] = config["runtime"][key]
    # Full resume will restore all parameter values. Only the initial asset
    # locations are taken from today's config; no old checkout is required.
    restored["model"]["encoder_checkpoint"] = config["model"]["encoder_checkpoint"]
    restored["model"]["predictor_checkpoint"] = None
    restored["model"]["rgb_checkpoint"] = None
    if config["model"].get("detector_checkpoint"):
        restored["model"]["detector_checkpoint"] = config["model"]["detector_checkpoint"]
    restored.pop("resume_detector_checkpoint", None)
    if stage == "predictor" and saved.get("detector_architecture") is not None:
        restored["resume_detector_checkpoint"] = str(checkpoint)
    if stage == "detector":
        architecture = saved.get("detector_architecture")
        if not isinstance(architecture, dict):
            raise ValueError("Detector resume checkpoint is missing its architecture")
        restored["detector"] = {key: value for key, value in architecture.items()
                                if key not in ("feature_dim", "grid_size")}
    restored["stage"] = stage
    # Explicit resume edits stay separate from defaults in a fresh recipe.
    restored["resume_overrides"] = deepcopy(config.get("resume_overrides", {}))
    for key, value in restored["resume_overrides"].items():
        restored["optimization"][key] = value
    return restored


def run(config, *, stage=None, resume=None, output_dir=None, max_steps=None, epochs=None,
        batch_size=None, num_workers=None, limit_train_batches=None, limit_val_batches=None):
    stage = stage or config.get("stage")
    if stage not in ("predictor", "detector"):
        raise ValueError("stage must be predictor or detector")
    if not resume and config.get("resume_overrides"):
        raise ValueError("resume_overrides requires --resume")
    if stage == "detector" and any(config["model"].get(key) for key in ("predictor_checkpoint", "rgb_checkpoint")):
        raise ValueError("The shared detector trains on real DINO features and does not take a predictor initializer")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; expose GPUs through CUDA_VISIBLE_DEVICES")
    config = deepcopy(config)
    configured_output = config.get("output_dir")
    resume = str(resolve_path(resume)) if resume else None
    if resume:
        config = resume_config(config, resume, stage)
    for key, value in (("max_steps", max_steps), ("epochs", epochs)):
        if value is not None:
            config["optimization"][key] = value
    if batch_size is not None:
        config["data"]["batch_size"] = batch_size
    if num_workers is not None:
        config["data"]["num_workers"] = num_workers
    config["stage"] = stage
    runtime = config["runtime"]
    if limit_train_batches is None:
        limit_train_batches = runtime.get("limit_train_batches", 1.0)
    if limit_val_batches is None:
        limit_val_batches = runtime.get("limit_val_batches", 1.0)
    runtime.update(limit_train_batches=limit_train_batches, limit_val_batches=limit_val_batches)
    options = config["optimization"]
    if options.get("max_steps", -1) == 0 or options.get("max_steps", -1) < -1:
        raise ValueError("max_steps must be -1 (unlimited) or positive")
    if options.get("accumulation_steps", 1) < 1 or config["data"]["batch_size"] < 1:
        raise ValueError("Batch size and accumulation_steps must be positive")
    if config["data"]["num_workers"] < 0:
        raise ValueError("num_workers cannot be negative")
    inherited = os.environ.get("FORESIGHT_TRAIN_OUTPUT")
    if output_dir:
        folder = resolve_path(output_dir)
    elif resume:
        folder = Path(resume).parent.parent
    elif inherited and "LOCAL_RANK" in os.environ:
        folder = Path(inherited)
    elif configured_output:
        folder = resolve_path(configured_output)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        folder = ROOT / "outputs" / f"{stage}_{stamp}"
    # Lightning's subprocess ranks inherit this run's directory after rank zero
    # has written config_train.yaml. Only a fresh invocation checks for reuse.
    child_rank = inherited == str(folder) and "LOCAL_RANK" in os.environ
    if not resume and not child_rank and any((folder / name).exists() for name in
            ("config_train.yaml", "checkpoints/last.ckpt", "checkpoints/best.ckpt")):
        raise FileExistsError(f"Training output already exists: {folder}; use --resume or a new --output-dir")
    config["output_dir"] = str(folder)
    os.environ["FORESIGHT_TRAIN_OUTPUT"] = str(folder)
    try:
        runtime = config["runtime"]
        torch.set_num_threads(runtime.get("cpu_threads", 4))
        torch.set_float32_matmul_precision("high")
        seed_everything(runtime.get("seed", 42), workers=True)
        task = PredictorTask(config) if stage == "predictor" else DetectorTask(config)
        trainer = make_trainer(config, stage, limit_train_batches=limit_train_batches,
                               limit_val_batches=limit_val_batches)
        if trainer.is_global_zero:
            folder.mkdir(parents=True, exist_ok=True)
            label = "resume" if resume else "train"
            (folder / f"config_{label}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
            (folder / f"run_{label}.json").write_text(json.dumps({
                "stage": stage, "resume": resume, "max_steps": options.get("max_steps", -1),
                "epochs": options.get("epochs", 40), "limit_train_batches": limit_train_batches,
                "limit_val_batches": limit_val_batches,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
                "checkpoint_format": CHECKPOINT_FORMAT,
            }, indent=2) + "\n")
        trainer.fit(task, ckpt_path=resume)
        # An explicit final save also covers a max_steps stop in mid-epoch.
        trainer.save_checkpoint(folder / "checkpoints" / "last.ckpt")
        if limit_val_batches and task.last_validation_step != trainer.global_step:
            # A step budget may end before the first epoch boundary. Evaluate
            # that final state once, using the same configured validation limit.
            trainer.validate(task)
        return trainer
    finally:
        if inherited is None:
            os.environ.pop("FORESIGHT_TRAIN_OUTPUT", None)
        else:
            os.environ["FORESIGHT_TRAIN_OUTPUT"] = inherited
