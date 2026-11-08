"""Forecast from a historical event NPZ (t/x/y/p), without RGB or labels."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from .config import load_config, resolve_path
from .data import CLASSES
from .data.events import events_to_voxel
from .event import load_event_pipeline
from .pipeline import setup_cuda, inference_context


def make_history(events, frame_ends, encoding, image_size=(448, 896)):
    if len(frame_ends) != 4 or any(b <= a for a, b in zip(frame_ends, frame_ends[1:])):
        raise ValueError("Four strictly increasing observed endpoints are required")
    if any(abs(b-a-50000) > 1000 for a, b in zip(frame_ends, frame_ends[1:])):
        raise ValueError("Observed endpoints must be spaced by approximately 50ms")
    if len(events["t"]) and np.max(events["t"]) >= frame_ends[-1]:
        raise ValueError("Input contains events at or beyond the historical cutoff")
    window = encoding["event_window_us"]
    return torch.stack([events_to_voxel(events, end-window, end, image_size,
                                       event_bin_widths_us=encoding["event_bin_widths_us"])
                        for end in frame_ends])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, help="NPZ with absolute microsecond t and distorted 640x480 x/y/p")
    parser.add_argument("--frame-ends-us", type=int, nargs=4, required=True, help="Historical exposure endpoints, oldest first")
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--config", default="configs/detector.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    destination = Path(args.output).expanduser().resolve()
    protected = [Path(args.events).expanduser().resolve(), resolve_path(args.encoder_checkpoint)]
    protected += [Path(v) for v in config["model"].values() if isinstance(v, str)]
    if destination in protected:
        parser.error("Output must differ from input events and weights")
    device = setup_cuda(config)
    model = load_event_pipeline(config, resolve_path(args.encoder_checkpoint)).to(device)
    with np.load(args.events, allow_pickle=False) as archive:
        history = make_history(archive, args.frame_ends_us, model.encoding)[None].to(device)
    with torch.inference_mode(), inference_context(config):
        prediction = model.predict(history)[0]
    detections = [dict(box_xyxy=box, label=label, class_name=CLASSES[label], score=score)
                  for box, label, score in zip(prediction["boxes"].float().cpu().tolist(),
                                              prediction["labels"].cpu().tolist(),
                                              prediction["scores"].float().cpu().tolist())]
    report = dict(frame_ends_us=args.frame_ends_us, input_cutoff_us=args.frame_ends_us[-1],
                  nominal_frame_interval_ms=50, prediction_reference="next RGB frame",
                  coordinate_format="xyxy in original 640x480 pixels",
                  event_encoding=model.encoding, detections=detections)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(f"Saved {len(detections)} detections to {destination}")


if __name__ == "__main__":
    main()
