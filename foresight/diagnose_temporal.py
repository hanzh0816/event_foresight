"""Inference perturbations of event timing on a fixed evaluation inventory.

These are sensitivity/OOD diagnostics, not proof that useful dynamics were
learned. No optimization occurs. Both heads retain the FP32 readout protocol.
"""

import argparse
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from .config import load_config, resolve_path
from .data.events import EventDataset
from .data.rgb import preprocess_rgb
from .diagnose_readout import (feature_errors, fp32_readout, load_real_head,
    make_diagnostic_dataset, prediction_records, retained_coordinates, sha256, write_json)
from .evaluation import FutureDetectionEvaluator
from .event import load_event_pipeline
from .models.detector import postprocess
from .pipeline import inference_context, setup_cuda


CONTROL_DESCRIPTIONS = {
    "identity": "Unperturbed event input and four-state history.",
    "reverse_history": "Reverse the four observed latent states before the Predictor; original newest state is unchanged.",
    "repeat_last": "Repeat the original newest latent state four times as Predictor input.",
    "persist_latent": "Use the original newest observed latent directly as the future prediction; bypass Predictor.",
    "reverse_bins": "Reverse the five event bins independently within each polarity, preserving each input value.",
}


def perturb_events(events, control):
    if control not in CONTROL_DESCRIPTIONS:
        raise ValueError(f"Unknown temporal control: {control}")
    if events.ndim != 5 or events.shape[1:3] != (4, 10):
        raise ValueError("Require event inputs [B,4,10,H,W]")
    if control == "reverse_bins":
        channels = torch.tensor([4, 3, 2, 1, 0, 9, 8, 7, 6, 5], device=events.device)
        return events.index_select(2, channels)
    return events


def perturb_history(history, control):
    if control not in CONTROL_DESCRIPTIONS:
        raise ValueError(f"Unknown temporal control: {control}")
    if history.ndim != 4 or history.shape[1] != 4:
        raise ValueError("Require latent history [B,4,N,D]")
    if control == "reverse_history":
        return history.flip(1)
    if control == "repeat_last":
        return history[:, -1:].expand_as(history)
    return history


def predict_future(history, predictor, control):
    """Separate state persistence from repeating inputs to a learned Predictor."""
    predictor_history = perturb_history(history, control)
    if control == "persist_latent":
        return history[:, -1]
    return predictor(predictor_history)


class FutureTeacherEventDataset(EventDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        name, end = self.windows[index]
        sample["future_rgb_target"] = preprocess_rgb(
            self._reader(name).read_image(end + self.prediction_frames).image, self.image_size)
        return sample


def collate_temporal(samples):
    return {key: torch.stack([sample[key] for sample in samples])
            for key in ("history_events", "future_rgb_target")} | {
        "meta": [sample["meta"] for sample in samples]}


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
    model = load_event_pipeline(config, resolve_path(args.encoder_checkpoint))
    real_head = load_real_head(resolve_path(args.readout_checkpoint), model.detector,
                               pca=model.forecaster.pca) if args.readout_checkpoint else None
    model = model.to(device).eval().requires_grad_(False)
    if real_head is not None:
        real_head = real_head.to(device)
    fc = model.forecaster
    dataset, ids, manifest = make_diagnostic_dataset(
        FutureTeacherEventDataset, config, args, **model.encoding)
    names = ["future_old_head_fp32"]
    if real_head is not None:
        names.append("future_real_feature_head_fp32")
    evaluators = {name: FutureDetectionEvaluator(dataset) for name in names}
    predictions = {name: [] for name in names}
    error_records = []
    provenance = dict(arguments=vars(args), input_modality="event", control=args.control,
        control_description=CONTROL_DESCRIPTIONS[args.control],
        interpretation="Inference perturbation/OOD sensitivity; not proof of learned useful dynamics.",
        encoder_predictor_precision=config["runtime"]["precision"], encoder_predictor_matmul_precision="high",
        encoder_chunk_size=config["runtime"]["encoder_chunk_size"], readout_precision="FP32/highest/TF32-off",
        model=config["model"], predictor_checkpoint=model.predictor_source, event_encoding=model.encoding, data_root=data["root"], sample_ids=ids,
        inventory_source="manifest" if manifest else "full_split", manifest=str(manifest) if manifest else None,
        future_rgb_teacher_only=True, no_parameter_updates=True,
        source_sha256={__file__: sha256(__file__),
            str(Path(__file__).with_name("diagnose_readout.py")): sha256(Path(__file__).with_name("diagnose_readout.py"))},
        manifest_sha256=sha256(manifest) if manifest else None,
        encoder_sha256=sha256(resolve_path(args.encoder_checkpoint)),
        readout_sha256=sha256(resolve_path(args.readout_checkpoint)) if real_head is not None else None)
    write_json(output / "run.json", provenance)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=data["num_workers"], pin_memory=True, collate_fn=collate_temporal)
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for batch in loader:
                with inference_context(config):
                    events = perturb_events(batch["history_events"].to(device, non_blocking=True), args.control)
                    latent = torch.cat([model.event_encoder(part) for part in events.flatten(0, 1).split(model.chunk_size)])
                    history = latent.reshape(len(events), 4, *latent.shape[1:])
                    # Keep original history distinct: a current readout must use
                    # history[:, -1], never the last slot of reversed P input.
                    predicted = predict_future(history, fc.predictor, args.control)
                    target = fc.encode_latents(batch["future_rgb_target"].to(device, non_blocking=True))
                errors = feature_errors(predicted, target)
                for i, meta in enumerate(batch["meta"]):
                    error_records.append(dict(sample_id=meta["sample_id"], future={key: value[i].item() for key, value in errors.items()}))
                with fp32_readout():
                    head_outputs = {"future_old_head_fp32": model.detector(fc.pca.inverse(predicted.float()))}
                    if real_head is not None:
                        head_outputs["future_real_feature_head_fp32"] = real_head(retained_coordinates(predicted, fc.pca))
                for name, values in head_outputs.items():
                    records = prediction_records(postprocess(values), batch["meta"])
                    predictions[name].extend(records)
                    evaluators[name].update(records)
                count = len(error_records)
                if count % 25 < args.batch_size or count == len(dataset):
                    print(json.dumps(dict(control=args.control, samples=count, total=len(dataset),
                        elapsed_seconds=time.monotonic()-started)), flush=True)
        errors = {key: sum(record["future"][key] for record in error_records) / len(error_records)
                  for key in error_records[0]["future"]}
        reports = {}
        for name in names:
            if [record["sample_id"] for record in predictions[name]] != ids:
                raise RuntimeError("Temporal-control predictions differ from inventory")
            report = evaluators[name].compute().to_dict()
            report.update(feature_errors={"future": errors}, readout=name, control=args.control)
            reports[name] = report
            write_json(output / f"metrics_{name}.json", report)
            write_json(output / f"predictions_{name}.json", dict(predictions=predictions[name]))
        coco, _ = evaluators[names[0]].to_coco()
        write_json(output / "ground_truth_future.json", coco)
        write_json(output / "feature_errors_per_sample.json", error_records)
        completed = dict(status="completed", samples=len(dataset), control=args.control,
            elapsed_seconds=time.monotonic()-started, feature_errors={"future": errors},
            metrics={name: report["metrics"] for name, report in reports.items()})
        write_json(output / "completed.json", completed)
        print(json.dumps(completed, indent=2), flush=True)
        return reports
    finally:
        dataset.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", default="configs/event.yaml")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--manifest", help="JSON sample inventory")
    selection.add_argument("--full-split", action="store_true",
                           help="Evaluate every valid window in --split, ignoring config subsets")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--data-root", help="Override the DSEC root in the config")
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--predictor-checkpoint")
    parser.add_argument("--readout-checkpoint")
    parser.add_argument("--control", choices=tuple(CONTROL_DESCRIPTIONS), required=True)
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
