"""Train an Event Encoder, RGB Predictor or Detector on all visible GPUs."""

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
import torch
import yaml

from .config import ROOT, load_config, resolve_path
from .training import CHECKPOINT_FORMAT, DetectorTask, PredictorTask


def batch_limit(value):
    """Integer means batches; a float in [0, 1] means a fraction of the loader."""
    text = str(value)
    try:
        if text.isdigit():
            return int(text)
        fraction = float(text)
        if 0 <= fraction <= 1:
            return fraction
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("Use an integer batch count or a fraction between 0 and 1")


def make_trainer(config, stage, *, limit_train_batches=1.0, limit_val_batches=1.0):
    options, runtime = config["optimization"], config["runtime"]
    folder = Path(config["output_dir"])
    checkpoint_options = dict(dirpath=folder / "checkpoints", save_last=True,
                              enable_version_counter=False, auto_insert_metric_name=False)
    if stage == "detector" and limit_val_batches:
        checkpoint = ModelCheckpoint(filename="best", monitor="val_AP", mode="max",
                                     save_top_k=1, save_on_train_epoch_end=False, **checkpoint_options)
    else:
        # Predictor adaptation is defined by its configured budget; preserve the
        # final model without selecting a checkpoint by validation feature loss.
        checkpoint = ModelCheckpoint(save_top_k=0, save_on_train_epoch_end=True, **checkpoint_options)
    return Trainer(
        accelerator="gpu", devices="auto",
        strategy=DDPStrategy(broadcast_buffers=False) if torch.cuda.device_count() > 1 else "auto",
        precision=runtime.get("precision", "bf16-mixed"),
        max_epochs=options.get("epochs", 40), max_steps=options.get("max_steps", -1),
        accumulate_grad_batches=options.get("accumulation_steps", 1),
        gradient_clip_val=options.get("clip_grad", 1.0),
        default_root_dir=folder, callbacks=[checkpoint],
        logger=[CSVLogger(folder, name="logs", version=None),
                TensorBoardLogger(folder, name="tensorboard", version=None, default_hp_metric=False)],
        limit_train_batches=limit_train_batches, limit_val_batches=limit_val_batches,
        log_every_n_steps=runtime.get("log_every", 10), num_sanity_val_steps=0,
        use_distributed_sampler=False,
    )


def resume_config(config, checkpoint, stage):
    """Restore the training recipe, allowing relocated assets/data and runtime settings."""
    from .checkpoints import read_checkpoint
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
    for key in ("cpu_threads", "log_every"):
        if key in config["runtime"]:
            restored["runtime"][key] = config["runtime"][key]
    # Full resume will restore all parameter values. Only the initial asset
    # locations are taken from today's config; no old checkout is required.
    restored["model"]["dinof_checkpoint"] = config["model"]["dinof_checkpoint"]
    restored["model"]["detector_checkpoint"] = None
    restored["model"]["predictor_checkpoint"] = None
    restored["model"]["event_encoder_checkpoint"] = None
    if stage == "detector":
        architecture = saved.get("detector_architecture")
        if not isinstance(architecture, dict):
            raise ValueError("Detector resume checkpoint is missing its architecture")
        restored["detector"] = {key: value for key, value in architecture.items()
                                if key not in ("feature_dim", "grid_size")}
    return restored


def run(config, *, stage="detector", resume=None, output_dir=None, max_steps=None, epochs=None,
        batch_size=None, num_workers=None, limit_train_batches=1.0, limit_val_batches=1.0):
    if stage not in ("predictor", "detector", "event"):
        raise ValueError("stage must be predictor, detector or event")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; expose GPUs through CUDA_VISIBLE_DEVICES")
    config = deepcopy(config)
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
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        folder = ROOT / "outputs" / f"{stage}_{stamp}"
    config["output_dir"] = str(folder)
    os.environ["FORESIGHT_TRAIN_OUTPUT"] = str(folder)
    try:
        runtime = config["runtime"]
        torch.set_num_threads(runtime.get("cpu_threads", 4))
        torch.set_float32_matmul_precision("high")
        seed_everything(runtime.get("seed", 42), workers=True)
        if stage == "event":
            from .event_training import EventTask
            task = EventTask(config)
        else:
            task = (PredictorTask if stage == "predictor" else DetectorTask)(config)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--stage", choices=("predictor", "detector", "event"), default="detector")
    parser.add_argument("--config", help="Defaults to configs/<stage>.yaml")
    parser.add_argument("--data-root", help="Override the DSEC root, including when resuming")
    parser.add_argument("--dinof-checkpoint", help="Override the exported RGB model asset, including when resuming")
    parser.add_argument("--resume", help="Full same-stage checkpoint; restores optimizer and training recipe")
    parser.add_argument("--init-from", help="Event Encoder weights only; starts a fresh optimizer and step count")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int, help="Optimizer steps, including steps already completed on resume")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, help="Examples per GPU")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--limit-train-batches", type=batch_limit, default=1.0)
    parser.add_argument("--limit-val-batches", type=batch_limit, default=1.0)
    args = parser.parse_args(argv)
    config = load_config(args.config or f"configs/{args.stage}.yaml")
    if args.data_root:
        config["data"]["root"] = str(resolve_path(args.data_root))
    if args.dinof_checkpoint:
        config["model"]["dinof_checkpoint"] = str(resolve_path(args.dinof_checkpoint))
    if args.init_from:
        if args.stage != "event" or args.resume:
            parser.error("--init-from requires --stage event and cannot be combined with --resume")
        config["model"]["event_encoder_checkpoint"] = str(resolve_path(args.init_from))
    return run(config, stage=args.stage, resume=args.resume, output_dir=args.output_dir,
               max_steps=args.max_steps, epochs=args.epochs, batch_size=args.batch_size,
               num_workers=args.num_workers, limit_train_batches=args.limit_train_batches,
               limit_val_batches=args.limit_val_batches)


if __name__ == "__main__":
    main()
