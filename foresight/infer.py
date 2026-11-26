"""Predict future detections from historical RGB and optional event voxels."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .config import load_config, resolve_path
from .data import CLASSES, HEIGHT, WIDTH, preprocess_rgb
from .pipeline import load_pipeline, setup_cuda, inference_context


def read_images(paths, expected_count=4):
    if len(paths) != expected_count:
        raise ValueError(f"Exactly {expected_count} images, oldest to newest, are required")
    images = []
    for filename in paths:
        with Image.open(filename) as image:
            if image.size != (WIDTH, HEIGHT):
                raise ValueError(f"Expected native {WIDTH}x{HEIGHT} RGB: {filename}")
            images.append(np.array(image.convert("RGB"), copy=True))
    return images


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", default="configs/detector.yaml")
    parser.add_argument("--checkpoint", help="Optional new-project trained Detector checkpoint")
    parser.add_argument("--predictor-checkpoint", help="Optional new-project trained Predictor")
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--events", help="Preprocessed history .npy [T,2*bins,H,W], using the training coordinate/bin contract")
    parser.add_argument("--horizon-steps", type=int, choices=(1,2,3,4), default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.)
    args = parser.parse_args(argv)
    if not math.isfinite(args.score_threshold) or not 0 <= args.score_threshold <= 1:
        parser.error("score-threshold must lie in [0,1]")
    config = load_config(args.config)
    if args.predictor_checkpoint:
        config["model"]["predictor_checkpoint"] = str(resolve_path(args.predictor_checkpoint))
    paths = [Path(value).expanduser().resolve() for value in args.images]
    output = Path(args.output).expanduser().resolve()
    protected = paths + [Path(v).resolve() for v in config["model"].values() if isinstance(v, str)]
    if args.checkpoint:
        protected.append(resolve_path(args.checkpoint))
    if args.events:
        protected.append(resolve_path(args.events))
    if output in protected:
        parser.error("Output must differ from input images and model assets")
    images = read_images(paths, config["data"]["context_frames"])
    device = setup_cuda(config)
    pipeline = load_pipeline(config, resolve_path(args.checkpoint) if args.checkpoint else None).to(device)
    history = torch.stack([preprocess_rgb(image, config["data"]["image_size"]) for image in images])[None].to(device)
    events = None
    if args.events:
        raw = np.load(resolve_path(args.events), allow_pickle=False)
        if raw.ndim != 4 or raw.shape[0] != len(images) or not np.isfinite(raw).all() or (raw < 0).any():
            raise ValueError("Events must be finite, nonnegative [T,2*bins,H,W] log-count features")
        events = torch.from_numpy(raw).float()[None].to(device)
    with torch.inference_mode(), inference_context(config):
        prediction = pipeline.predict(history, events, horizon_steps=args.horizon_steps)[0]
    detections = []
    for box, label, score in zip(prediction["boxes"].float().cpu().tolist(),
                                 prediction["labels"].cpu().tolist(),
                                 prediction["scores"].float().cpu().tolist()):
        if score >= args.score_threshold:
            detections.append(dict(box_xyxy=box, label=label, class_name=CLASSES[label], score=score))
    result = dict(history_images=list(map(str, paths)), prediction_ms=50*args.horizon_steps,
                  assumed_history_interval_ms=50, original_image_size=[HEIGHT, WIDTH],
                  model_image_size=config["data"]["image_size"],
                  coordinate_format="xyxy in original image pixels", detections=detections)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(detections)} detections to {output}")
    return result


if __name__ == "__main__":
    main()
