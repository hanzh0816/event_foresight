"""Run forecasting on explicit historical image files and optional event voxels."""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ..data import CLASSES, HEIGHT, WIDTH, preprocess_rgb
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


def infer_images(config, image_paths, *, events_path=None, output):
    options = config.get("inference", {})
    horizon_steps = options.get("horizon_steps", 1)
    score_threshold = options.get("score_threshold", 0.0)
    if type(horizon_steps) is not int or horizon_steps not in (1, 2, 3, 4):
        raise ValueError("inference.horizon_steps must be 1, 2, 3 or 4")
    if (type(score_threshold) not in (int, float) or not math.isfinite(score_threshold)
            or not 0 <= score_threshold <= 1):
        raise ValueError("inference.score_threshold must lie in [0,1]")
    # Explicit input/output files follow the shell cwd.
    paths = [Path(value).expanduser().resolve() for value in image_paths]
    output = Path(output).expanduser().resolve()
    events_path = Path(events_path).expanduser().resolve() if events_path else None
    protected = paths + [Path(v).resolve() for v in config["model"].values() if isinstance(v, str)]
    if events_path is not None:
        protected.append(events_path)
    if output in protected:
        raise ValueError("Output must differ from input images and model assets")
    images = read_images(paths, config["data"]["context_frames"])
    device = setup_cuda(config)
    pipeline = load_pipeline(config).to(device)
    history = torch.stack([preprocess_rgb(image, config["data"]["image_size"]) for image in images])[None].to(device)
    events = None
    if events_path is not None:
        raw = np.load(events_path, allow_pickle=False)
        if raw.ndim != 4 or raw.shape[0] != len(images) or not np.isfinite(raw).all() or (raw < 0).any():
            raise ValueError("Events must be finite, nonnegative [T,2*bins,H,W] log-count features")
        events = torch.from_numpy(raw).float()[None].to(device)
    with torch.inference_mode(), inference_context(config):
        prediction = pipeline.predict(history, events, horizon_steps=horizon_steps)[0]
    detections = []
    for box, label, score in zip(prediction["boxes"].float().cpu().tolist(),
                                 prediction["labels"].cpu().tolist(),
                                 prediction["scores"].float().cpu().tolist()):
        if score >= score_threshold:
            detections.append(dict(box_xyxy=box, label=label, class_name=CLASSES[label], score=score))
    result = dict(history_images=list(map(str, paths)), prediction_ms=50*horizon_steps,
                  assumed_history_interval_ms=50, original_image_size=[HEIGHT, WIDTH],
                  model_image_size=config["data"]["image_size"],
                  coordinate_format="xyxy in original image pixels", detections=detections)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(detections)} detections to {output}")
    return result
