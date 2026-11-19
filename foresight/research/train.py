"""Train a configured research stage; dataset targets never enter forecast inputs."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
import torch
import yaml

from ..train import batch_limit
from .config import load_config, validate_config
from .task import CHECKPOINT_FORMAT, git_revision, load_task


def make_trainer(config, *, limit_train_batches=1.0, limit_val_batches=1.0):
    runtime, options = config["runtime"], config["optimization"]
    folder = Path(config["output_dir"])
    accelerator, devices = runtime["accelerator"], runtime["devices"]
    multi_gpu = accelerator != "cpu" and torch.cuda.is_available() and (
        (devices == "auto" and torch.cuda.device_count() > 1) or isinstance(devices, list) and len(devices) > 1 or isinstance(devices, int) and devices > 1)
    callback = ModelCheckpoint(dirpath=folder / "checkpoints", save_last=True, save_top_k=0,
                               enable_version_counter=False, save_on_train_epoch_end=True)
    return Trainer(accelerator=accelerator, devices=devices,
                   strategy=DDPStrategy(broadcast_buffers=False, find_unused_parameters=True) if multi_gpu else "auto",
                   precision=runtime["precision"], max_epochs=options["epochs"], max_steps=options["max_steps"],
                   accumulate_grad_batches=options["accumulation_steps"], gradient_clip_val=options["clip_grad"],
                   default_root_dir=folder, callbacks=[callback], logger=CSVLogger(folder, name="logs"),
                   log_every_n_steps=runtime["log_every"], num_sanity_val_steps=0,
                   limit_train_batches=limit_train_batches, limit_val_batches=limit_val_batches,
                   use_distributed_sampler=True)


def run(config, *, output_dir, resume=None, limit_train_batches=1.0, limit_val_batches=1.0,
        forecaster=None, datasets=None):
    config = validate_config(config)
    config["output_dir"] = str(Path(output_dir).expanduser().resolve())
    folder = Path(config["output_dir"])
    primary_process = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0
    if primary_process and not resume and folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Fresh training requires an empty output directory: {folder}")
    runtime = config["runtime"]
    torch.set_num_threads(runtime["cpu_threads"])
    torch.set_float32_matmul_precision("high")
    seed_everything(runtime["seed"], workers=True)
    task = load_task(config, forecaster=forecaster, checkpoint=resume)
    if config["model"].get("mode") == "rgb" or config["model"].get("copy_last_motion"):
        raise ValueError("RGB baseline and copy-last-motion are evaluation-only controls")
    if datasets is not None:
        task.datasets = dict(datasets)
    trainer = make_trainer(config, limit_train_batches=limit_train_batches, limit_val_batches=limit_val_batches)
    if trainer.is_global_zero:
        folder.mkdir(parents=True, exist_ok=True)
        suffix = "resume_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") if resume else "train"
        (folder / f"config_{suffix}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        (folder / f"run_{suffix}.json").write_text(json.dumps(dict(
            format=CHECKPOINT_FORMAT, stage=task.stage, git_commit=git_revision(), resume=str(resume) if resume else None,
            protocol=task.protocol, model_spec=task.model.model_spec(), teacher_provenance=task.teacher_provenance,
            initialization_provenance=task.initialization_provenance,
            trainable_parameters=sum(p.numel() for p in task.parameters() if p.requires_grad),
            frozen_parameters=sum(p.numel() for p in task.parameters() if not p.requires_grad),
            seed=runtime["seed"], precision=runtime["precision"], accelerator=runtime["accelerator"], devices=runtime["devices"],
            world_size=trainer.world_size, per_device_batch_size=config["data"]["batch_size"],
            accumulation_steps=config["optimization"]["accumulation_steps"],
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
            limit_train_batches=limit_train_batches, limit_val_batches=limit_val_batches,
            validation_sampler="Lightning distributed sampler may pad; formal AP evaluation uses the separate evaluation command",
        ), indent=2) + "\n")
    trainer.fit(task, ckpt_path=resume)
    trainer.save_checkpoint(folder / "checkpoints" / "last.ckpt")
    if limit_val_batches and task.last_validation_step != trainer.global_step:
        trainer.validate(task)
    return trainer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", help="Complete same-stage checkpoint; recipe/optimizer settings must match")
    parser.add_argument("--limit-train-batches", type=batch_limit, default=1.0)
    parser.add_argument("--limit-val-batches", type=batch_limit, default=1.0)
    args = parser.parse_args(argv)
    return run(load_config(args.config), output_dir=args.output_dir, resume=args.resume,
               limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches)


if __name__ == "__main__":
    main()
