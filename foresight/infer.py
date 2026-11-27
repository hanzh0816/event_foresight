"""Predict future detections from historical RGB and optional event voxels."""

import argparse

from .config import load_config
from .inference.images import infer_images


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--events", help="Preprocessed history .npy [T,2*bins,H,W], using the training coordinate/bin contract")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    return infer_images(config, args.images, events_path=args.events, output=args.output)


if __name__ == "__main__":
    main()
