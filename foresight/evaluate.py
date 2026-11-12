"""Evaluate future detection on a DSEC split or an explicit sample inventory."""

import argparse
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from .config import load_config, resolve_path
from .data import RGBDataset, collate_rgb
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
             output="outputs/metrics.json", batch_size=1, max_samples=None, encoder_checkpoint=None):
    device = setup_cuda(config)
    if encoder_checkpoint:
        from .event import load_event_pipeline
        from .data.events import EventDataset, collate_event
        model = load_event_pipeline(config, encoder_checkpoint, checkpoint).to(device)
        dataset_class, collate, input_key = EventDataset, collate_event, "history_events"
    else:
        model = load_pipeline(config, checkpoint).to(device)
        dataset_class, collate, input_key = RGBDataset, collate_rgb, "history_rgb"
    data = config["data"]
    selection = read_sample_ids(manifest, split) if manifest else data.get(f"{split}_sample_ids")
    kwargs = {key: data[key] for key in ("root", "image_size", "sample_history_frames",
                                       "context_frames", "tolerance_us")}
    kwargs.update(split=split, future_rgb=False, sample_ids=selection,
                  stride=data.get(f"{split}_stride", 1), sequences=data.get(f"{split}_sequences"))
    if encoder_checkpoint:
        kwargs.update(model.encoding)
    dataset = dataset_class(**kwargs)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        ids = [f"{name}:{end}" for name, end in dataset.windows[:max_samples]]
        dataset.close()
        kwargs["sample_ids"] = ids
        dataset = dataset_class(**kwargs)
    if not len(dataset):
        raise ValueError("Evaluation selection is empty")
    expected_ids = [f"{name}:{end}" for name, end in dataset.windows]
    output = Path(output).expanduser().resolve()
    predictions_path = output.with_name(output.stem + "_predictions.json")
    protected = {Path(v).resolve() for v in config["model"].values() if isinstance(v, str)}
    if checkpoint:
        protected.add(Path(checkpoint).resolve())
    if encoder_checkpoint:
        protected.add(Path(encoder_checkpoint).resolve())
    if manifest:
        protected.add(Path(manifest).resolve())
    if output in protected or predictions_path in protected:
        raise ValueError("Evaluation output must not overwrite input assets or manifest")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=data["num_workers"], pin_memory=True,
                        collate_fn=collate)
    evaluator = FutureDetectionEvaluator(dataset, sample_ids=expected_ids)
    records = []
    cutoff_horizons = []
    started = time.monotonic()
    try:
        with torch.inference_mode(), inference_context(config):
            for batch in loader:
                predictions = model.predict(batch[input_key].to(device, non_blocking=True))
                if encoder_checkpoint:
                    cutoff_horizons.extend(meta["effective_prediction_us"]/1000 for meta in batch["meta"])
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
        if encoder_checkpoint:
            report["metadata"].update(input_modality="event", event_encoding=model.encoding,
                                       predictor_checkpoint=model.predictor_source,
                                       input_cutoff="last RGB exposure end; event intervals are half-open",
                                       cutoff_to_target_ms={"min": min(cutoff_horizons),
                                                            "max": max(cutoff_horizons),
                                                            "mean": sum(cutoff_horizons)/len(cutoff_horizons)})
        report["runtime"] = dict(precision=config["runtime"]["precision"],
                                 float32_matmul_precision="high", batch_size=batch_size,
                                 encoder_chunk_size=config["runtime"]["encoder_chunk_size"],
                                 model=config["model"], checkpoint=str(checkpoint) if checkpoint else None,
                                 encoder_checkpoint=str(encoder_checkpoint) if encoder_checkpoint else None,
                                 elapsed_seconds=time.monotonic()-started)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        predictions_path.write_text(json.dumps(dict(predictions=records), allow_nan=False) + "\n")
        print(json.dumps(report["metrics"], indent=2))
        return report
    finally:
        dataset.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", default="configs/detector.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--manifest", help="JSON sample_ids or existing split/sample_ids manifest")
    parser.add_argument("--checkpoint", help="Optional new-project trained Detector checkpoint")
    parser.add_argument("--predictor-checkpoint", help="Optional new-project trained Predictor")
    parser.add_argument("--encoder-checkpoint", help="Evaluate Event-only forecasting using this Event Encoder")
    parser.add_argument("--data-root")
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
                    encoder_checkpoint=resolve_path(args.encoder_checkpoint) if args.encoder_checkpoint else None)


if __name__ == "__main__":
    main()
