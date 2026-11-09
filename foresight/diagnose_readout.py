"""Paired current/future readouts and true-RGB feature diagnostics.

Both detector heads use FP32/highest/TF32-off, matching the earlier readout
diagnostic. Encoder and Predictor retain the canonical BF16/high/chunk1 path.
Current labels are from the last observed frame, never the future target.
"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import load_config, resolve_path
from .data.events import EventDataset
from .data.definitions import HEIGHT, WIDTH
from .data.rgb import RGBDataset, preprocess_rgb
from .evaluate import read_sample_ids
from .evaluation import FutureDetectionEvaluator
from .event import load_event_pipeline
from .models.detector import postprocess
from .pipeline import inference_context, load_pipeline, setup_cuda


class PairedEventDataset(EventDataset):
    """Future RGB is returned in a separate teacher-only target field."""

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        name, end = self.windows[index]
        reader = self._reader(name)
        sample["current_rgb_target"] = preprocess_rgb(reader.read_image(end).image, self.image_size)
        sample["future_rgb_target"] = preprocess_rgb(
            reader.read_image(end + self.prediction_frames).image, self.image_size)
        return sample


class CurrentTimeline:
    """Read-only evaluator view preserving inventory but selecting end labels."""

    prediction_frames = 0
    prediction_ms = 0

    def __init__(self, dataset):
        self.dataset = dataset

    def __getattr__(self, name):
        return getattr(self.dataset, name)


class CurrentDetectionEvaluator(FutureDetectionEvaluator):
    """Include empty current targets, as in the legacy current probe."""

    def to_coco(self):
        dataset, results = super().to_coco()
        existing = {(item["sample_id"], item["target_id"]) for item in dataset["images"]}
        for key in sorted(set(self._inventory) - existing):
            sequence, frame, _, _ = self._inventory[key]
            reader = self.dataset.reader(sequence)
            try:
                target = reader.read_target(frame)
                if len(target.boxes):
                    raise RuntimeError("Nonempty target was unexpectedly omitted")
                image_id = len(dataset["images"]) + 1
                dataset["images"].append(dict(id=image_id, width=WIDTH, height=HEIGHT,
                    file_name=f"{sequence}/{frame:06d}.png", sample_id=key[0],
                    target_id=key[1], timestamp_us=int(target.timestamp_us)))
                prediction = self._predictions.get(key)
                if prediction is not None:
                    for box, category, score in zip(prediction.boxes, prediction.class_ids, prediction.scores):
                        x, y, x2, y2 = map(float, box)
                        results.append(dict(image_id=image_id, category_id=int(category) + 1,
                                            bbox=[x, y, x2 - x, y2 - y], score=float(score)))
            finally:
                reader.close()
        return dataset, results


def collate_pairs(samples):
    keys = ("history_events", "current_rgb_target", "future_rgb_target") if "history_events" in samples[0] else (
        "history_rgb", "future_rgb_target")
    return {**{key: torch.stack([s[key] for s in samples]) for key in keys},
            "meta": [s["meta"] for s in samples]}


def feature_errors(prediction, target):
    """Per-image PCA errors, with identical definitions to the prior probes."""
    if prediction.ndim != 3 or prediction.shape != target.shape:
        raise ValueError("Require matching [B,N,D] features")
    prediction, target = prediction.float(), target.float()
    difference = prediction - target
    return {
        "l1": difference.abs().mean((1, 2)),
        "mse": difference.square().mean((1, 2)),
        "smooth_l1": F.smooth_l1_loss(prediction, target, reduction="none").mean((1, 2)),
        "cosine": F.cosine_similarity(prediction, target, dim=-1).mean(1),
        "zero_smooth_l1": F.smooth_l1_loss(torch.zeros_like(target), target, reduction="none").mean((1, 2)),
        "token_norm_ratio": prediction.norm(dim=-1).mean(1) / target.norm(dim=-1).mean(1).clamp_min(1e-8),
        "spatial_std_ratio": prediction.std(dim=1, correction=0).mean(1) / target.std(dim=1, correction=0).mean(1).clamp_min(1e-8),
    }


def retained_coordinates(latent, pca):
    """True-feature probe expects standardized retained xQ = z C, not raw RGB."""
    return latent.float() @ pca.pca_components.float()


@contextmanager
def fp32_readout():
    previous = torch.get_float32_matmul_precision()
    previous_tf32 = torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.autocast("cuda", enabled=False):
            yield
    finally:
        torch.set_float32_matmul_precision(previous)
        torch.backends.cudnn.allow_tf32 = previous_tf32


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_real_head(path, detector):
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if (saved.get("format") != "current-pca-head-probe-v1" or saved.get("branch") != "roundtrip"
            or saved.get("modality") != "rgb" or saved.get("steps") != 500):
        raise ValueError("Require the fixed RGB retained-feature head after 500 updates")
    plan_path = Path(path).parent / "plan.json"
    plan = json.loads(plan_path.read_text())
    if saved.get("plan_sha256") != sha256(plan_path):
        raise ValueError("RGB head plan hash differs from checkpoint")
    if plan.get("input_coordinates") != "shared x=(f-mean)/std-pca_mean; raw=x, roundtrip=x C.T C":
        raise ValueError("RGB head uses unexpected feature coordinates")
    head = deepcopy(detector)
    head.load_state_dict(saved["state_dict"], strict=True)
    return head.eval().requires_grad_(False)


def prediction_records(predictions, metadata, current=False):
    result = []
    for prediction, meta in zip(predictions, metadata, strict=True):
        name, end = meta["sample_id"].rsplit(":", 1)
        target_id = f"{name}/{int(end):06d}" if current else meta["target_id"]
        result.append(dict(sample_id=meta["sample_id"], target_id=target_id,
                           boxes=prediction["boxes"].float().cpu().tolist(),
                           class_ids=prediction["labels"].cpu().tolist(),
                           scores=prediction["scores"].float().cpu().tolist()))
    return result


def write_json(path, value):
    # Full-split prediction files are large; do not build a second giant string.
    with Path(path).open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def make_diagnostic_dataset(dataset_class, config, args, *, default_manifest=None, **extra):
    """Select either an explicit inventory or the complete standard DSEC split.

    Full-split diagnostics deliberately ignore training-manifest IDs and any
    subset/stride settings in the training config. The selected inventory is
    recorded in run.json by both diagnostic entry points.
    """
    full_split = bool(getattr(args, "full_split", False))
    manifest = None if full_split else (args.manifest or default_manifest)
    if manifest is None and not full_split:
        raise ValueError("Specify --manifest or --full-split")
    manifest = resolve_path(manifest) if manifest else None
    selected = read_sample_ids(manifest, args.split) if manifest else None
    limit = args.max_samples
    if limit is not None:
        if limit <= 0:
            raise ValueError("max-samples must be positive")
        if selected is not None:
            selected = selected[:limit]
    data = config["data"]
    kwargs = {key: data[key] for key in ("root", "image_size", "sample_history_frames", "context_frames", "tolerance_us")}
    kwargs.update(split=args.split, sample_ids=selected, **extra)
    dataset = dataset_class(**kwargs)
    if selected is None and limit is not None and len(dataset) > limit:
        selected = [f"{name}:{end}" for name, end in dataset.windows[:limit]]
        dataset.close()
        kwargs["sample_ids"] = selected
        dataset = dataset_class(**kwargs)
    ids = [f"{name}:{end}" for name, end in dataset.windows]
    if not ids:
        dataset.close()
        raise ValueError("Diagnostic evaluation selection is empty")
    if selected is not None and ids != selected:
        dataset.close()
        raise RuntimeError("Diagnostic inventory differs from manifest order")
    return dataset, ids, manifest


def run(args):
    config = load_config(args.config)
    if getattr(args, "data_root", None):
        config["data"]["root"] = str(resolve_path(args.data_root))
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    if args.predictor_checkpoint:
        config["model"]["predictor_checkpoint"] = str(resolve_path(args.predictor_checkpoint))
    data = config["data"]
    output = resolve_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    device = setup_cuda(config)
    torch.manual_seed(config["runtime"]["seed"])
    event = bool(args.encoder_checkpoint)
    pipeline = (load_event_pipeline(config, resolve_path(args.encoder_checkpoint)) if event
                else load_pipeline(config))
    real_head = load_real_head(resolve_path(args.readout_checkpoint), pipeline.detector) if args.readout_checkpoint else None
    pipeline = pipeline.to(device).eval().requires_grad_(False)
    if real_head is not None:
        real_head = real_head.to(device)
    fc = pipeline.forecaster
    dataset, ids, manifest = make_diagnostic_dataset(
        PairedEventDataset if event else RGBDataset, config, args,
        default_manifest="configs/val600.json",
        **(pipeline.encoding if event else {"future_rgb": True}))
    names = ["current_old_head_fp32", "future_old_head_fp32"]
    if not event:
        names += ["current_raw_rgb_old_head_fp32"]
    if real_head is not None:
        names += ["current_real_feature_head_fp32", "future_real_feature_head_fp32"]
    evaluators = {name: (CurrentDetectionEvaluator(CurrentTimeline(dataset)) if name.startswith("current_")
                         else FutureDetectionEvaluator(dataset))
                  for name in names}
    predictions = {name: [] for name in names}
    error_records = []
    provenance = dict(arguments=vars(args), input_modality="event" if event else "rgb",
                      encoder_predictor_precision=config["runtime"]["precision"],
                      encoder_predictor_matmul_precision="high", encoder_chunk_size=config["runtime"]["encoder_chunk_size"],
                      readout_precision="FP32/highest/TF32-off", current_labels="last history frame end",
                      future_labels="end+1", current_empty_gt_policy="include, matching legacy current RGB/Event head probe",
                      future_rgb_teacher_only=True, prediction_inputs="history only",
                      model=config["model"], data_root=data["root"], sample_ids=ids,
                      inventory_source="manifest" if manifest else "full_split",
                      manifest=str(manifest) if manifest else None,
                      source_sha256=sha256(__file__), manifest_sha256=sha256(manifest) if manifest else None,
                      encoder_sha256=sha256(resolve_path(args.encoder_checkpoint)) if event else None,
                      readout_sha256=sha256(resolve_path(args.readout_checkpoint)) if real_head is not None else None)
    write_json(output / "run.json", provenance)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=data["num_workers"], pin_memory=True, collate_fn=collate_pairs)
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for batch in loader:
                with inference_context(config):
                    if event:
                        events = batch["history_events"].to(device, non_blocking=True)
                        latent = torch.cat([pipeline.event_encoder(part) for part in events.flatten(0, 1).split(pipeline.chunk_size)])
                        history = latent.reshape(len(events), 4, *latent.shape[1:])
                        current_dense = fc.encode_dense(batch["current_rgb_target"].to(device, non_blocking=True))
                        current_target = fc.pca.project(current_dense)
                    else:
                        rgb = batch["history_rgb"].to(device, non_blocking=True)
                        # Encode/project each frame separately to preserve chunk1 numerics.
                        dense_frames = [fc.encode_dense(part) for part in rgb.flatten(0, 1).split(1)]
                        latent = torch.cat([fc.pca.project(part) for part in dense_frames])
                        history = latent.reshape(len(rgb), 4, *latent.shape[1:])
                        current_dense = torch.cat([dense_frames[i * 4 + 3] for i in range(len(rgb))])
                        current_target = history[:, -1]
                    predicted = fc.predictor(history)
                    future_target = fc.encode_latents(batch["future_rgb_target"].to(device, non_blocking=True))
                errors = {"current": feature_errors(history[:, -1], current_target),
                          "future": feature_errors(predicted, future_target)}
                for i, meta in enumerate(batch["meta"]):
                    error_records.append(dict(sample_id=meta["sample_id"],
                        **{phase: {key: values[i].item() for key, values in values_by_key.items()}
                           for phase, values_by_key in errors.items()}))
                with fp32_readout():
                    head_outputs = {
                        "current_old_head_fp32": pipeline.detector(fc.pca.inverse(history[:, -1].float())),
                        "future_old_head_fp32": pipeline.detector(fc.pca.inverse(predicted.float())),
                    }
                    if not event:
                        head_outputs["current_raw_rgb_old_head_fp32"] = pipeline.detector(current_dense.float())
                    if real_head is not None:
                        head_outputs["current_real_feature_head_fp32"] = real_head(retained_coordinates(history[:, -1], fc.pca))
                        head_outputs["future_real_feature_head_fp32"] = real_head(retained_coordinates(predicted, fc.pca))
                for name, values in head_outputs.items():
                    records = prediction_records(postprocess(values), batch["meta"], name.startswith("current_"))
                    predictions[name].extend(records)
                    evaluators[name].update(records)
                count = len(error_records)
                if count % 25 < args.batch_size or count == len(dataset):
                    print(json.dumps(dict(samples=count, total=len(dataset), elapsed_seconds=time.monotonic()-started)), flush=True)
        errors = {phase: {key: sum(record[phase][key] for record in error_records) / len(error_records)
                          for key in error_records[0][phase]} for phase in ("current", "future")}
        reports = {}
        for name in names:
            if [p["sample_id"] for p in predictions[name]] != ids:
                raise RuntimeError("Readout sample order differs")
            report = evaluators[name].compute().to_dict()
            report["feature_errors"] = errors
            report["readout"] = name
            reports[name] = report
            write_json(output / f"metrics_{name}.json", report)
            write_json(output / f"predictions_{name}.json", dict(predictions=predictions[name]))
        for phase in ("current", "future"):
            coco, _ = evaluators[f"{phase}_old_head_fp32"].to_coco()
            write_json(output / f"ground_truth_{phase}.json", coco)
        write_json(output / "feature_errors_per_sample.json", error_records)
        completed = dict(status="completed", samples=len(dataset), elapsed_seconds=time.monotonic()-started,
                         feature_errors=errors, metrics={name: r["metrics"] for name, r in reports.items()})
        write_json(output / "completed.json", completed)
        print(json.dumps(completed, indent=2), flush=True)
        return reports
    finally:
        dataset.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", default="configs/event.yaml")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--manifest", help="JSON sample inventory; defaults to configs/val600.json")
    selection.add_argument("--full-split", action="store_true",
                           help="Evaluate every valid window in --split, ignoring config subsets")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--data-root", help="Override the DSEC root in the config")
    parser.add_argument("--encoder-checkpoint")
    parser.add_argument("--predictor-checkpoint")
    parser.add_argument("--readout-checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    return run(args)


if __name__ == "__main__":
    main()
