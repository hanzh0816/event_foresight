"""Fine-tune only Predictor using a frozen full_joint Event Encoder."""

import argparse
from copy import deepcopy
import json
import os

from lightning.pytorch import seed_everything
import torch
import yaml

from .config import load_config, resolve_path
from .event_predictor_training import EventPredictorTask
from .train import batch_limit, make_trainer, resume_config
from .train_future_event import configure_epoch_checkpoints


def run(config, *, output_dir, resume=None, max_steps=None, epochs=None,
        limit_train_batches=1.0, limit_val_batches=1.0):
    config = deepcopy(config)
    resume = str(resolve_path(resume)) if resume else None
    if resume:
        config = resume_config(config, resume, "event_predictor")
    options = config["optimization"]
    for key, value in (("max_steps", max_steps), ("epochs", epochs)):
        if value is not None:
            options[key] = value
    if options.get("max_steps", -1) == 0 or options.get("max_steps", -1) < -1:
        raise ValueError("max_steps must be -1 (unlimited) or positive")
    if options.get("epochs", 10) < 1:
        raise ValueError("epochs must be positive")
    if options.get("accumulation_steps", 1) < 1 or config["data"]["batch_size"] < 1:
        raise ValueError("Batch size and accumulation_steps must be positive")
    if config["data"]["num_workers"] < 0:
        raise ValueError("num_workers cannot be negative")
    if not resume and not config["model"].get("event_encoder_checkpoint"):
        raise ValueError("Provide the trained full_joint --encoder-checkpoint, or --resume")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    folder = resolve_path(output_dir)
    config["output_dir"] = str(folder)
    runtime = config["runtime"]
    torch.set_num_threads(runtime.get("cpu_threads", 4))
    torch.set_float32_matmul_precision("high")
    seed_everything(runtime.get("seed", 42), workers=True)
    task = EventPredictorTask(config, resume=bool(resume))
    trainer = make_trainer(config, "event_predictor", limit_train_batches=limit_train_batches,
                           limit_val_batches=limit_val_batches)
    configure_epoch_checkpoints(trainer, config)
    if trainer.is_global_zero:
        folder.mkdir(parents=True, exist_ok=True)
        label = "resume" if resume else "train"
        (folder / f"config_{label}.yaml").write_text(yaml.safe_dump(task.config, sort_keys=False))
        (folder / f"run_{label}.json").write_text(json.dumps({
            "stage": "event_predictor", "resume": resume,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
            "limit_train_batches": limit_train_batches, "limit_val_batches": limit_val_batches,
            "epochs": options.get("epochs", 10), "max_steps": options.get("max_steps", -1),
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
    parser.add_argument("--config", default="configs/event_predictor_full.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--encoder-checkpoint", help="Chosen full_joint Encoder; required for a fresh run")
    parser.add_argument("--resume", help="Full event_predictor checkpoint including optimizer and frozen Encoder")
    parser.add_argument("--max-steps", type=int, help="Total optimizer steps including resumed steps; -1 uses epochs")
    parser.add_argument("--epochs", type=int, help="Total epochs including resumed epochs")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-root")
    parser.add_argument("--dinof-checkpoint")
    parser.add_argument("--limit-train-batches", type=batch_limit, default=1.0)
    parser.add_argument("--limit-val-batches", type=batch_limit, default=1.0)
    args = parser.parse_args(argv)
    if args.resume:
        conflicting = [key for key in ("encoder_checkpoint", "batch_size", "seed") if getattr(args, key) is not None]
        if conflicting:
            parser.error("--resume restores the saved protocol; cannot override " +
                         ", ".join("--" + key.replace("_", "-") for key in conflicting))
    config = load_config(args.config)
    for key in ("encoder_checkpoint", "dinof_checkpoint"):
        value = getattr(args, key)
        if value:
            config["model"]["event_encoder_checkpoint" if key == "encoder_checkpoint" else key] = str(resolve_path(value))
    if args.data_root:
        config["data"]["root"] = str(resolve_path(args.data_root))
    if args.seed is not None:
        config["runtime"]["seed"] = args.seed
    for key in ("batch_size", "num_workers"):
        if getattr(args, key) is not None:
            config["data"][key] = getattr(args, key)
    return run(config, output_dir=args.output_dir, resume=args.resume, max_steps=args.max_steps,
               epochs=args.epochs, limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches)


if __name__ == "__main__":
    main()
