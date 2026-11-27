"""Evaluate future detection on a DSEC split or an explicit sample inventory."""

import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from ..config import resolve_path
from ..data import dataset_from_config, collate_rgb
from .detection import FutureDetectionEvaluator
from ..inference.pipeline import load_pipeline, setup_cuda, inference_context

_UNSET = object()


def read_sample_ids(filename, split):
    if filename is None:
        return None
    selection = json.loads(Path(filename).read_text())
    if isinstance(selection, dict) and split in selection:
        selection = selection[split]
    if isinstance(selection, dict):
        selection = selection.get("sample_ids")
    if (not isinstance(selection, list) or not selection
            or not all(isinstance(value, str) for value in selection)
            or len(set(selection)) != len(selection)):
        raise ValueError("Manifest must provide a nonempty unique sample_ids list for the split")
    return selection


def evaluate(config, *, split=None, manifest=_UNSET, checkpoint=None,
             output=None, batch_size=None, max_samples=_UNSET,
             horizon_steps=None, event_mode=None, subset=None, top_fraction=None, persistence=None):
    """Use evaluation settings unless overridden; None clears manifest/max_samples."""
    options = config.get("evaluation", {})
    if not isinstance(options, dict):
        raise ValueError("evaluation must be a mapping")
    split = options.get("split", "val") if split is None else split
    manifest = options.get("manifest") if manifest is _UNSET else manifest
    output = options.get("output", "outputs/metrics.json") if output is None else output
    batch_size = options.get("batch_size", 1) if batch_size is None else batch_size
    max_samples = options.get("max_samples") if max_samples is _UNSET else max_samples
    horizon_steps = options.get("horizon_steps", 1) if horizon_steps is None else horizon_steps
    event_mode = options.get("event_mode", "correct") if event_mode is None else event_mode
    subset = options.get("subset", "all") if subset is None else subset
    top_fraction = options.get("top_fraction", 0.2) if top_fraction is None else top_fraction
    persistence = options.get("persistence", False) if persistence is None else persistence
    if split not in ("train", "val", "test"):
        raise ValueError("evaluation.split must be train, val or test")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("evaluation.batch_size must be a positive integer")
    if max_samples is not None and (type(max_samples) is not int or max_samples < 1):
        raise ValueError("evaluation.max_samples must be null or a positive integer")
    if type(horizon_steps) is not int or horizon_steps not in (1, 2, 3, 4):
        raise ValueError("evaluation.horizon_steps must be 1, 2, 3 or 4")
    if event_mode not in ("correct", "mismatched", "bin_reverse", "zero", "off"):
        raise ValueError("Unknown evaluation.event_mode")
    if subset not in ("all", "high_motion", "nonuniform"):
        raise ValueError("evaluation.subset must be all, high_motion or nonuniform")
    if (type(top_fraction) not in (int, float) or not math.isfinite(top_fraction)
            or not 0 < top_fraction <= 1):
        raise ValueError("evaluation.top_fraction must lie in (0,1]")
    if type(persistence) is not bool:
        raise ValueError("evaluation.persistence must be a boolean")
    manifest = resolve_path(manifest) if manifest is not None else None
    checkpoint = checkpoint or config["model"].get("detector_checkpoint")
    checkpoint = resolve_path(checkpoint) if checkpoint is not None else None
    data = config["data"]
    selection = read_sample_ids(manifest, split) if manifest is not None else data.get(f"{split}_sample_ids")
    if event_mode != "correct" and not config["model"].get("event", {}).get("enabled", False):
        raise ValueError("Event interventions require an Event-RGB model")
    kwargs = dict(sample_ids=selection, prediction_frames=horizon_steps, horizontal_flip_prob=0.0)
    dataset = dataset_from_config(config, split, future_rgb=True, **kwargs)
    if subset != "all":
        from .diagnostics import motion_subset
        kwargs["sample_ids"] = motion_subset(dataset, subset, top_fraction)
        dataset.close()
        dataset = dataset_from_config(config, split, future_rgb=True, **kwargs)
    if max_samples is not None:
        kwargs["sample_ids"] = dataset.sample_ids[:max_samples]
        dataset.close()
        dataset = dataset_from_config(config, split, future_rgb=True, **kwargs)
    if not len(dataset):
        raise ValueError("Evaluation selection is empty")
    expected_ids = [f"{name}:{end}" for name, end in dataset.windows]
    output = resolve_path(output)
    predictions_path = output.with_name(output.stem + "_predictions.json")
    protected = {Path(v).resolve() for v in config["model"].values() if isinstance(v, str)}
    if checkpoint:
        protected.add(Path(checkpoint).resolve())
    if manifest:
        protected.add(Path(manifest).resolve())
    if output in protected or predictions_path in protected:
        raise ValueError("Evaluation output must not overwrite input assets or manifest")
    device = setup_cuda(config)
    model = load_pipeline(config, checkpoint, persistence=persistence).to(device)
    donor_dataset, donors = None, {}
    if event_mode == "mismatched":
        from .diagnostics import mismatched_donors
        donor_dataset = dataset_from_config(config, split, future_rgb=False, sample_ids=None,
                                            horizontal_flip_prob=0.0, prediction_frames=1)
        donors = mismatched_donors(donor_dataset, expected_ids)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=data["num_workers"], pin_memory=True,
                        collate_fn=collate_rgb)
    evaluator = FutureDetectionEvaluator(dataset, sample_ids=expected_ids)
    records = []
    region_totals = {}
    started = time.monotonic()
    try:
        with torch.inference_mode(), inference_context(config):
            for batch in loader:
                from .latent import latent_region_totals
                from ..models.detection.detector import postprocess
                history = batch["history_rgb"].to(device, non_blocking=True)
                events = batch.get("history_events")
                if event_mode == "mismatched":
                    events = torch.stack([donor_dataset[donors[m["sample_id"]]]["history_events"] for m in batch["meta"]])
                elif event_mode == "bin_reverse":
                    from ..data.events import reverse_event_bins
                    events = reverse_event_bins(events)
                elif event_mode == "zero":
                    events = torch.zeros_like(events)
                if events is not None:
                    events = events.to(device, non_blocking=True)
                features = model.features(history, events, horizon_steps=horizon_steps,
                                          event_off=event_mode == "off", persistence=persistence)
                predictions = postprocess(model.detector(features))
                targets = model.forecaster.encode_latents(batch["future_rgb_target"].to(device, non_blocking=True))
                totals = latent_region_totals(features, targets, batch["targets"], model.forecaster.grid_size)
                for name, value in totals.items():
                    region_totals[name] = region_totals.get(name, torch.zeros_like(value)) + value
                current = [dict(sample_id=meta["sample_id"], target_id=meta["target_id"],
                                boxes=pred["boxes"].float().cpu().tolist(),
                                class_ids=pred["labels"].cpu().tolist(),
                                scores=pred["scores"].float().cpu().tolist())
                           for pred, meta in zip(predictions, batch["meta"], strict=True)]
                records.extend(current)
                evaluator.update(current)
                if len(records) % 50 < batch_size or len(records) == len(dataset):
                    print(json.dumps(dict(samples=len(records), total=len(dataset),
                                          seconds=round(time.monotonic()-started, 2))), flush=True)
        if [record["sample_id"] for record in records] != expected_ids:
            raise RuntimeError("Evaluation sample order changed")
        report = evaluator.compute().to_dict()
        for name, value in region_totals.items():
            l1, cosine, count = value.tolist()
            report["metrics"][f"{name}_l1"] = l1/count if count else None
            report["metrics"][f"{name}_cosine"] = cosine/count if count else None
        report["metadata"].update(event_mode=event_mode, horizon_steps=horizon_steps, subset=subset,
                                  subset_definition="observed image-plane track velocity or velocity change",
                                  top_fraction=top_fraction, persistence=persistence,
                                  sample_ids=expected_ids,
                                  donor_sample_ids={key: donor_dataset.sample_ids[value] for key,value in donors.items()})
        report["runtime"] = dict(precision=config["runtime"]["precision"],
                                 float32_matmul_precision="high", batch_size=batch_size,
                                 encoder_chunk_size=config["runtime"]["encoder_chunk_size"],
                                 model=config["model"], checkpoint=str(checkpoint) if checkpoint else None,
                                 elapsed_seconds=time.monotonic()-started)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        predictions_path.write_text(json.dumps(dict(predictions=records), allow_nan=False) + "\n")
        print(json.dumps(report["metrics"], indent=2))
        return report
    finally:
        dataset.close()
        if donor_dataset is not None:
            donor_dataset.close()
