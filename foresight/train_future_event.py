"""Train Event representations from RGB weights with true future RGB targets."""

import argparse
from copy import deepcopy
import json
import os

from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import yaml

from .config import load_config, resolve_path
from .future_event_training import FutureEventTask
from .train import batch_limit, make_trainer, resume_config


def configure_epoch_checkpoints(trainer, config):
    """Retain full-training candidates without changing other launchers/pilots."""
    if not config["optimization"].get("keep_epoch_checkpoints", False):
        return
    callback = ModelCheckpoint(
        dirpath=resolve_path(config["output_dir"]) / "checkpoints",
        filename="epoch_{epoch:04d}", auto_insert_metric_name=False,
        save_top_k=-1, save_last=True, every_n_epochs=1,
        save_on_train_epoch_end=True, enable_version_counter=False)
    # Two unmonitored ModelCheckpoints have the same state_key; replace the
    # generic last-only callback so their resume states cannot overwrite each other.
    trainer.callbacks = [item for item in trainer.callbacks if not isinstance(item, ModelCheckpoint)] + [callback]


def run(config, *, output_dir, max_steps=None, epochs=None, limit_train_batches=1.0,
        limit_val_batches=1.0, resume=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config = deepcopy(config)
    resume = str(resolve_path(resume)) if resume else None
    if resume:
        config = resume_config(config, resume, "event")
    options = config["optimization"]
    for key, value in (("max_steps", max_steps), ("epochs", epochs)):
        if value is not None:
            options[key] = value
    if options.get("max_steps", -1) == 0 or options.get("max_steps", -1) < -1:
        raise ValueError("max_steps must be -1 (unlimited) or positive")
    if options.get("epochs", 40) < 1:
        raise ValueError("epochs must be positive")
    if options.get("accumulation_steps", 1) < 1 or config["data"]["batch_size"] < 1:
        raise ValueError("Batch size and accumulation_steps must be positive")
    if config["data"]["num_workers"] < 0:
        raise ValueError("num_workers cannot be negative")
    options.setdefault("gradient_checkpointing", True)
    options.setdefault("activation_checkpointing", True)
    folder = resolve_path(output_dir)
    config["output_dir"] = str(folder)
    runtime = config["runtime"]
    torch.set_num_threads(runtime.get("cpu_threads", 4))
    torch.set_float32_matmul_precision("high")
    seed_everything(runtime.get("seed", 42), workers=True)
    task = FutureEventTask(config)
    trainer = make_trainer(config, "event", limit_train_batches=limit_train_batches,
                           limit_val_batches=limit_val_batches)
    configure_epoch_checkpoints(trainer, config)
    if trainer.is_global_zero:
        folder.mkdir(parents=True, exist_ok=True)
        label = "resume" if resume else "train"
        (folder / f"config_{label}.yaml").write_text(yaml.safe_dump(task.config, sort_keys=False))
        (folder / f"run_{label}.json").write_text(json.dumps({
            "stage": "future_event", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
            "resume": resume, "limit_train_batches": limit_train_batches,
            "limit_val_batches": limit_val_batches,
            "epochs": options.get("epochs", 40), "max_steps": options.get("max_steps", -1),
            "keep_epoch_checkpoints": options.get("keep_epoch_checkpoints", False),
            "resume_data_policy": "standard Lightning; use epoch-boundary checkpoints for exact epoch coverage",
        }, indent=2) + "\n")
    trainer.fit(task, ckpt_path=resume)
    trainer.save_checkpoint(folder / "checkpoints" / "last.ckpt")
    if limit_val_batches and task.last_validation_step != trainer.global_step:
        trainer.validate(task)
    return trainer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", default="configs/event_full_joint.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init-from", help="Optional Event weights; otherwise initialize from the configured RGB model")
    parser.add_argument("--resume", help="Full checkpoint; restores optimizer, objective, seed and batch protocol")
    parser.add_argument("--max-steps", type=int, help="Total optimizer steps including resumed steps; -1 uses epochs")
    parser.add_argument("--epochs", type=int, help="Total epochs including resumed epochs")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-root", help="Relocate DSEC without editing the recipe")
    parser.add_argument("--dinof-checkpoint", help="Relocate the exported RGB encoder/PCA/Predictor asset")
    parser.add_argument("--current-weight", type=float)
    parser.add_argument("--future-weight", type=float)
    parser.add_argument("--current-scope", choices=("all", "last"))
    parser.add_argument("--limit-train-batches", type=batch_limit, default=1.0)
    parser.add_argument("--limit-val-batches", type=batch_limit, default=1.0)
    args = parser.parse_args(argv)
    if args.resume:
        conflicting = [name for name in ("init_from", "batch_size", "seed", "current_weight", "future_weight", "current_scope")
                       if getattr(args, name) is not None]
        if conflicting:
            parser.error("--resume restores the saved protocol; cannot override " +
                         ", ".join("--" + name.replace("_", "-") for name in conflicting))
    cfg = load_config(args.config)
    if args.init_from:
        cfg["model"]["event_encoder_checkpoint"] = str(resolve_path(args.init_from))
    if args.data_root:
        cfg["data"]["root"] = str(resolve_path(args.data_root))
    if args.dinof_checkpoint:
        cfg["model"]["dinof_checkpoint"] = str(resolve_path(args.dinof_checkpoint))
    if args.seed is not None:
        cfg["runtime"]["seed"] = args.seed
    for key in ("batch_size", "num_workers"):
        value = getattr(args, key)
        if value is not None:
            cfg["data"][key] = value
    for key in ("current_weight", "future_weight", "current_scope"):
        value = getattr(args, key)
        if value is not None:
            cfg["optimization"][key] = value
    return run(cfg, output_dir=args.output_dir, max_steps=args.max_steps, epochs=args.epochs, resume=args.resume,
               limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches)


if __name__ == "__main__":
    main()
