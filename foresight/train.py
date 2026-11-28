"""Train a v3.1 Predictor or shared Detector; all visible GPUs are used automatically."""

import argparse

from .config import load_config
from .training.runner import run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", required=True, help="YAML training recipe, including stage and output_dir")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--resume", help="Full same-stage checkpoint; restores optimizer, batch size and training recipe")
    start.add_argument("--batch-size", type=int, help="Per-GPU batch size for a new run; overrides data.batch_size")
    parser.add_argument("--output-dir", help="Override the configured directory; resume otherwise reuses its run directory")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if config.get("stage") not in ("predictor", "detector"):
        parser.error("Set stage: predictor or stage: detector in the training config")
    return run(config, stage=config["stage"], resume=args.resume, output_dir=args.output_dir,
               batch_size=args.batch_size)


if __name__ == "__main__":
    main()
