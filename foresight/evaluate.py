"""Evaluate future detection on a DSEC split or an explicit sample inventory."""

import argparse
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from .config import load_config, resolve_path
from .data import dataset_from_config, collate_rgb
from .evaluation import FutureDetectionEvaluator
from .pipeline import load_pipeline, setup_cuda, inference_context


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


def evaluate(config, *, split="val", manifest=None, checkpoint=None,
             output="outputs/metrics.json", batch_size=1, max_samples=None,
             horizon_steps=1, event_mode="correct", subset="all", top_fraction=0.2, persistence=False):
    data = config["data"]
    selection = read_sample_ids(manifest, split) if manifest else data.get(f"{split}_sample_ids")
    if horizon_steps not in (1, 2, 3, 4):
        raise ValueError("horizon_steps must be 1, 2, 3 or 4")
    if event_mode not in ("correct", "mismatched", "bin_reverse", "zero", "off"):
        raise ValueError("Unknown event intervention")
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
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        kwargs["sample_ids"] = dataset.sample_ids[:max_samples]
        dataset.close()
        dataset = dataset_from_config(config, split, future_rgb=True, **kwargs)
    if not len(dataset):
        raise ValueError("Evaluation selection is empty")
    expected_ids = [f"{name}:{end}" for name, end in dataset.windows]
    output = Path(output).expanduser().resolve()
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
                from .losses import latent_region_totals
                from .models.detector import postprocess
                history = batch["history_rgb"].to(device, non_blocking=True)
                events = batch.get("history_events")
                if event_mode == "mismatched":
                    events = torch.stack([donor_dataset[donors[m["sample_id"]]]["history_events"] for m in batch["meta"]])
                elif event_mode == "bin_reverse":
                    from .data.events import reverse_event_bins
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", default="configs/detector.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--manifest", help="JSON sample_ids or existing split/sample_ids manifest")
    parser.add_argument("--checkpoint", help="Optional new-project trained Detector checkpoint")
    parser.add_argument("--predictor-checkpoint", help="Optional new-project trained Predictor")
    parser.add_argument("--data-root")
    parser.add_argument("--horizon-steps", type=int, choices=(1,2,3,4), default=1)
    parser.add_argument("--event-mode", choices=("correct", "mismatched", "bin_reverse", "zero", "off"), default="correct")
    parser.add_argument("--subset", choices=("all", "high_motion", "nonuniform"), default="all")
    parser.add_argument("--top-fraction", type=float, default=0.2)
    parser.add_argument("--persistence", action="store_true")
    parser.add_argument("--output", default="outputs/metrics.json")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    config = load_config(args.config)
    if args.data_root:
        config["data"]["root"] = str(resolve_path(args.data_root))
    if args.predictor_checkpoint:
        config["model"]["predictor_checkpoint"] = str(resolve_path(args.predictor_checkpoint))
    return evaluate(config, split=args.split, manifest=resolve_path(args.manifest) if args.manifest else None,
                    checkpoint=resolve_path(args.checkpoint) if args.checkpoint else None,
                    output=args.output, batch_size=args.batch_size, max_samples=args.max_samples,
                    horizon_steps=args.horizon_steps, event_mode=args.event_mode, subset=args.subset,
                    top_fraction=args.top_fraction, persistence=args.persistence)


if __name__ == "__main__":
    main()
