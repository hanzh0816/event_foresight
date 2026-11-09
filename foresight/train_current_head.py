"""Train the current retained-feature RGB head from random initialization."""

import argparse
from copy import deepcopy
import json

from lightning.pytorch import seed_everything
import torch
import yaml

from .config import load_config, resolve_path
from .current_head_training import CurrentHeadTask
from .train import batch_limit, make_trainer, resume_config


def run(config, *, output_dir, resume=None, epochs=None, max_steps=None,
        limit_train_batches=1.0, limit_val_batches=1.0):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config = deepcopy(config)
    resume = str(resolve_path(resume)) if resume else None
    if resume:
        config = resume_config(config, resume, "detector")
    for key, value in (("epochs", epochs), ("max_steps", max_steps)):
        if value is not None:
            config["optimization"][key] = value
    options, runtime = config["optimization"], config["runtime"]
    if options.get("epochs", 10) < 1 or options.get("max_steps", -1) == 0 or options.get("max_steps", -1) < -1:
        raise ValueError("epochs must be positive; max_steps must be -1 or positive")
    if config["data"]["batch_size"] < 1 or options.get("accumulation_steps", 1) < 1:
        raise ValueError("batch_size and accumulation_steps must be positive")
    if config["data"]["num_workers"] < 0:
        raise ValueError("num_workers cannot be negative")
    folder = resolve_path(output_dir)
    config["output_dir"] = str(folder)
    torch.set_num_threads(runtime.get("cpu_threads", 4))
    torch.set_float32_matmul_precision("high")
    seed_everything(runtime.get("seed", 42), workers=True)
    task = CurrentHeadTask(config)
    trainer = make_trainer(config, "detector", limit_train_batches=limit_train_batches,
                           limit_val_batches=limit_val_batches)
    if trainer.is_global_zero:
        folder.mkdir(parents=True, exist_ok=True)
        label = "resume" if resume else "train"
        (folder / f"config_{label}.yaml").write_text(yaml.safe_dump(task.config, sort_keys=False))
        (folder / f"run_{label}.json").write_text(json.dumps({
            "stage": "current_retained_head", "resume": resume,
            "head_initialization": "checkpoint" if resume else "random",
            "input_spec": task.input_spec,
            "limit_train_batches": limit_train_batches, "limit_val_batches": limit_val_batches,
        }, indent=2) + "\n")
    trainer.fit(task, ckpt_path=resume)
    trainer.save_checkpoint(folder / "checkpoints" / "last.ckpt")
    if limit_val_batches and task.last_validation_step != trainer.global_step:
        trainer.validate(task)
    return trainer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", default="configs/current_head.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--data-root")
    parser.add_argument("--dinof-checkpoint")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit-train-batches", type=batch_limit, default=1.0)
    parser.add_argument("--limit-val-batches", type=batch_limit, default=1.0)
    args = parser.parse_args(argv)
    if args.resume and (args.batch_size is not None or args.seed is not None):
        parser.error("--resume restores the saved batch size and seed")
    config = load_config(args.config)
    for section, key, value in (("data", "root", args.data_root),
                                ("model", "dinof_checkpoint", args.dinof_checkpoint)):
        if value:
            config[section][key] = str(resolve_path(value))
    for key in ("batch_size", "num_workers"):
        if getattr(args, key) is not None:
            config["data"][key] = getattr(args, key)
    if args.seed is not None:
        config["runtime"]["seed"] = args.seed
    return run(config, output_dir=args.output_dir, resume=args.resume, epochs=args.epochs,
               max_steps=args.max_steps, limit_train_batches=args.limit_train_batches,
               limit_val_batches=args.limit_val_batches)


if __name__ == "__main__":
    main()
