"""Paired, evaluation-only diagnosis of the Event motion/transition bottleneck."""

import argparse
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from .bottleneck import diagnostic_predictions
from .data import HorizonDatasetView, ResearchDSECDataset, collate_research
from .diagnostic_checkpoint import load_diagnostic_task
from .evaluate import dataset_kwargs, file_sha256, foreground_mask, sample_digest
from .task import git_revision
from ..evaluation import FutureDetectionEvaluator
from ..models.detector import load_detector, postprocess


FORMAT = "event-wm-transition-diagnostic-v1"
ROUTES = {
    "rgb_predictor": dict(kind="history_only_forecast", future_rgb=False, future_event=False,
                          description="Frozen RGB predictor, recursively applied to its own states"),
    "scene_persistence": dict(kind="history_only_forecast", future_rgb=False, future_event=False,
                              description="Repeat the last observed RGB scene latent"),
    "true_future_rgb": dict(kind="future_rgb_readout_diagnostic", future_rgb=True, future_event=False,
                            description="Read out real future RGB latents; not a forecasting result or a guaranteed upper bound"),
    "true_future_event": dict(kind="observed_transition_diagnostic", future_rgb=False, future_event=True,
                              description="Real future Event motion through the active transition, recursively using predicted states"),
    "last_motion": dict(kind="history_only_forecast", future_rgb=False, future_event=False,
                        description="Repeat last historical Event motion through the same active transition"),
    "predicted_motion": dict(kind="history_only_forecast", future_rgb=False, future_event=False,
                             description="Checkpoint motion predictor with historical observations only"),
    "true_state_true_event": dict(kind="true_state_transition_diagnostic", future_rgb=True, future_event=True,
                                  description="Each interval starts from its true RGB state and uses real interval Event motion"),
}


def worker_init(_worker_id):
    # Prevent per-worker CPU oversubscription on large GPU servers.
    torch.set_num_threads(1)


class FeatureErrors:
    def __init__(self):
        self.sums = {key: 0.0 for key in ("all", "fg", "bg", "motion")}
        self.counts = {key: 0 for key in self.sums}

    def update(self, scene, target, mask, motion=None, motion_target=None):
        if scene.shape != target.shape or mask.shape != scene.shape[:2]:
            raise ValueError("Scene/target/foreground shapes differ")
        error = (scene.float() - target.float()).abs().mean(-1)
        if not torch.isfinite(error).all():
            raise FloatingPointError("Non-finite scene error")
        for key, selected in (("all", torch.ones_like(mask)), ("fg", mask), ("bg", ~mask)):
            self.sums[key] += error[selected].double().sum().item()
            self.counts[key] += selected.sum().item()
        if motion is not None:
            if motion_target is None or motion.shape != motion_target.shape:
                raise ValueError("Motion and target shapes differ")
            difference = (motion.float() - motion_target.float()).abs()
            if not torch.isfinite(difference).all():
                raise FloatingPointError("Non-finite motion error")
            self.sums["motion"] += difference.double().sum().item()
            self.counts["motion"] += difference.numel()

    def compute(self):
        return {
            **{key + "_l1": self.sums[key] / self.counts[key] if self.counts[key] else None
               for key in self.sums},
            "foreground_patch_fraction": self.counts["fg"] / self.counts["all"] if self.counts["all"] else None,
            "counts": deepcopy(self.counts),
            "scene_reduction": "mean absolute channel error, then mean over selected patches in all samples",
            "motion_reduction": "mean absolute error over all motion channels and spatial tokens",
        }


def paired_deltas(routes, horizons):
    pairs = [
        ("predicted_motion", "last_motion", "Benefit over maintaining historical motion"),
        ("predicted_motion", "true_future_event", "Replacing observed future motion with predicted motion"),
        ("true_future_event", "scene_persistence", "Observed-event transition versus a static scene"),
        ("true_future_event", "rgb_predictor", "Privileged transition reference versus RGB forecast, not a forecast ranking"),
        ("predicted_motion", "rgb_predictor", "Complete motion route versus RGB forecast"),
        ("true_state_true_event", "true_future_event", "True starting states versus recursive predicted states"),
    ]
    output = []
    for left, right, meaning in pairs:
        if left not in routes or right not in routes:
            continue
        delta = {}
        for horizon in horizons:
            a, b = routes[left]["horizons"][str(horizon)], routes[right]["horizons"][str(horizon)]
            values = {}
            for key in ("AP", "AP50", "AP75"):
                x, y = a["metrics"].get(key), b["metrics"].get(key)
                values[key + "_delta_points"] = 100 * (x-y) if x is not None and y is not None else None
            for key in ("all_l1", "fg_l1", "bg_l1", "motion_l1"):
                x, y = a["features"].get(key), b["features"].get(key)
                values[key + "_delta"] = x-y if x is not None and y is not None else None
            delta[str(horizon)] = values
        output.append(dict(left=left, right=right, interpretation=meaning, horizons=delta))
    return output


def markdown_summary(report):
    lines = ["# Event motion/transition diagnosis", "",
             "All rows share samples, RGB features and detector. Future-observation rows are privileged diagnostics, not forecast competitors or guaranteed upper bounds.", "",
             f"Samples: {len(report['sample_ids'])}/{report['sample_selection']['full_inventory_size']} eligible windows; split: {report['split']}; checkpoint: `{report['checkpoint']}`", "",
             "| Route | Kind | Step | AP | AP50 | AP75 | All L1 | Foreground L1 | Background L1 | Motion L1 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def number(value, scale=1):
        return "—" if value is None else f"{value * scale:.4f}"

    for name, route in report["routes"].items():
        for step, result in route["horizons"].items():
            values = [number(result["metrics"].get(k), 100) for k in ("AP", "AP50", "AP75")]
            values += [number(result["features"].get(k)) for k in ("all_l1", "fg_l1", "bg_l1", "motion_l1")]
            lines.append("| " + " | ".join([name, route["kind"], step, *values]) + " |")
    lines += ["", "AP columns use the 0–100 scale; JSON metrics use 0–1. L1 uses the same target features and foreground masks in every row.", "",
              "## Paired differences", "", "| Left minus right | Step | AP points | Foreground L1 delta |", "|---|---:|---:|---:|"]
    for comparison in report["comparisons"]:
        for step, delta in comparison["horizons"].items():
            lines.append(f"| {comparison['left']} − {comparison['right']} | {step} | {number(delta['AP_delta_points'])} | {number(delta['fg_l1_delta'])} |")
    lines += ["", "Differences are descriptive. Positive AP differences and negative L1 differences favor the left row. Do not interpret a single run as statistical significance.", "",
              "Interpretation: first inspect true_future_rgb and true_future_event. If observed-event transitions are already poor, investigate G/T and readout before enlarging the motion predictor. If observed-event transitions are strong but predicted_motion is weak, investigate motion predictability. Multi-step true_state_true_event separates state-feedback effects from real-motion transition quality.", ""]
    return "\n".join(lines)


def _load_manifest(path, split):
    if path is None:
        return None
    value = json.loads(Path(path).read_text())
    if isinstance(value, dict):
        if value.get("split") != split:
            raise ValueError("Manifest split differs from evaluation split")
        return value
    if not isinstance(value, list):
        raise ValueError("Manifest must contain sample_ids or be a list of sample IDs")
    return {"sample_ids": value}


def _check_manifest_times(manifest, samples):
    if not manifest or not manifest.get("samples"):
        return
    previous = {item["sample_id"]: item for item in manifest["samples"]}
    keys = ("input_cutoff_us", "history_start_us", "event_history_start_us",
            "target_timestamps_us", "horizon_us", "transition_horizon_us")
    for item in samples:
        if item["sample_id"] not in previous:
            raise ValueError("Sample missing from manifest timing records")
        for key in keys:
            if key in previous[item["sample_id"]] and previous[item["sample_id"]][key] != item[key]:
                raise ValueError(f"Manifest time boundary mismatch: {item['sample_id']} {key}")


def diagnose(checkpoint, *, output_dir, data_root=None, dinof_checkpoint=None,
             detector_checkpoint=None, split="val", steps=1, horizons=None,
             batch_size=8, num_workers=8, device="auto", precision=None,
             cpu_threads=4, max_samples=None, manifest=None, save_predictions=False):
    """Run every reference on one common loader; never modify a training recipe."""
    if type(steps) is not int or steps < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("steps and batch_size must be positive integers")
    if type(num_workers) is not int or num_workers < 0 or type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("num_workers must be nonnegative and cpu_threads positive")
    if max_samples is not None and (type(max_samples) is not int or max_samples < 1):
        raise ValueError("max_samples must be a positive integer")
    horizons = list(range(1, steps+1)) if horizons is None else list(horizons)
    if not horizons or len(set(horizons)) != len(horizons) or any(type(h) is not int or h < 1 or h > steps for h in horizons):
        raise ValueError("horizons must be unique steps within the evaluation rollout")
    if split not in ("train", "val", "test") or device not in ("auto", "cpu", "cuda"):
        raise ValueError("Invalid split or device")
    selected_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    precision = precision or ("bf16-mixed" if selected_device.type == "cuda" else "32-true")
    if precision not in ("32-true", "bf16-mixed") or precision == "bf16-mixed" and selected_device.type != "cuda":
        raise ValueError("Use 32-true on CPU; CUDA supports 32-true or bf16-mixed")
    folder = Path(output_dir).expanduser().resolve()
    if folder.exists():
        raise FileExistsError(f"Use a new diagnostic output directory: {folder}")
    torch.set_num_threads(cpu_threads)
    torch.set_float32_matmul_precision("high")
    task, provenance = load_diagnostic_task(checkpoint, data_root=data_root,
        dinof_checkpoint=dinof_checkpoint, detector_checkpoint=detector_checkpoint)
    task = task.eval().requires_grad_(False).to(selected_device)
    detector_file = task.config["model"].get("detector_checkpoint")
    if not detector_file:
        raise ValueError("Provide --detector-checkpoint or a checkpoint config with its export path")
    detector = load_detector(detector_file).eval().requires_grad_(False).to(selected_device)
    # Evaluation windows are independent from checkpoint training rollout length.
    evaluation_config = deepcopy(task.config)
    evaluation_config["data"].update(rollout_steps=steps, evaluation_horizons=horizons)
    options = dataset_kwargs(evaluation_config, split, future_targets=True)
    saved_manifest = _load_manifest(manifest, split)
    if saved_manifest is not None:
        options["sample_ids"] = saved_manifest["sample_ids"]
    names = list(ROUTES) if steps > 1 else [name for name in ROUTES if name != "true_state_true_event"]
    started = time.monotonic()
    with ResearchDSECDataset(**options) as inventory:
        selected_before_limit = len(inventory)
        options["sample_ids"] = inventory.sample_ids[:max_samples] if max_samples else inventory.sample_ids
    with ResearchDSECDataset(**options) as dataset:
        if not len(dataset):
            raise ValueError("No common labeled evaluation windows")
        views = {h: HorizonDatasetView(dataset, h) for h in horizons}
        evaluators = {name: {h: FutureDetectionEvaluator(views[h]) for h in horizons} for name in names}
        errors = {name: {h: FeatureErrors() for h in horizons} for name in names}
        predictions = {name: {h: [] for h in horizons} for name in names} if save_predictions else None
        samples = []
        folder.mkdir(parents=True, exist_ok=False)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            collate_fn=collate_research, pin_memory=selected_device.type == "cuda", worker_init_fn=worker_init,
            **({"multiprocessing_context": "spawn", "prefetch_factor": 1} if num_workers else {}))
        amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16-mixed" else nullcontext()
        with torch.inference_mode():
            for batch in loader:
                batch = {key: value.to(selected_device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                with amp():
                    output = diagnostic_predictions(task, batch, steps=steps)
                    if set(output["routes"]) != set(names):
                        raise RuntimeError("Unexpected diagnostic route set")
                    for h in horizons:
                        mask = foreground_mask([target[h-1] for target in batch["targets"]], task.forecaster.grid_size, selected_device)
                        for name in names:
                            route = output["routes"][name]
                            scene = route["scene"][:, h-1]
                            errors[name][h].update(scene, output["target_scene"][:, h-1], mask,
                                route["motion"][:, h-1] if "motion" in route else None, output["target_motion"][:, h-1])
                            detected = postprocess(detector(task.forecaster.pca.inverse(scene)))
                            records = [dict(sample_id=meta["sample_id"], target_id=meta["target_ids"][h-1],
                                boxes=item["boxes"].float().cpu().tolist(), class_ids=item["labels"].cpu().tolist(),
                                scores=item["scores"].float().cpu().tolist())
                                for item, meta in zip(detected, batch["meta"], strict=True)]
                            evaluators[name][h].update(records)
                            if predictions is not None:
                                predictions[name][h].extend(records)
                for meta in batch["meta"]:
                    samples.append({key: deepcopy(meta[key]) for key in ("sample_id", "sequence", "input_cutoff_us",
                        "history_start_us", "event_history_start_us", "target_ids", "target_timestamps_us", "horizon_us",
                        "transition_horizon_us", "history_intervals_us", "future_intervals_us")})
        if [item["sample_id"] for item in samples] != dataset.sample_ids:
            raise RuntimeError("Diagnostic sample inventory/order mismatch")
        _check_manifest_times(saved_manifest, samples)
        report = dict(format=FORMAT, evaluation_kind="transition_bottleneck_diagnostic", git_commit=git_revision(),
            checkpoint=str(Path(checkpoint).resolve()), provenance=provenance,
            detector_sha256=file_sha256(detector_file), split=split, steps=steps, horizons=horizons,
            sample_ids=dataset.sample_ids, sample_digest=sample_digest(dataset.sample_ids), samples=samples,
            sample_timing_sha256=hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest(),
            checkpoint_config=deepcopy(task.config), evaluation_data=options,
            runtime=dict(device=str(selected_device), precision=precision, batch_size=batch_size,
                         num_workers=num_workers, worker_torch_threads=1, cpu_threads=cpu_threads),
            max_samples=max_samples, smoke_test=max_samples is not None, manifest_source=str(manifest) if manifest else None,
            sample_selection=dict(full_inventory_size=dataset.full_inventory_size,
                selected_before_limit=selected_before_limit, evaluated=len(dataset),
                covers_full_inventory=len(dataset) == dataset.full_inventory_size),
            routes={}, interpretation="Paired diagnostics; privileged future-observation rows are not forecasting results or guaranteed upper bounds. No automatic significance or success claim.")
        for name in names:
            route = {**ROUTES[name], "horizons": {}}
            for h in horizons:
                value = evaluators[name][h].compute().to_dict()
                milliseconds = [meta["horizon_us"][h-1]/1000 for meta in samples]
                value["metadata"]["cutoff_to_target_ms"] = dict(min=min(milliseconds), max=max(milliseconds), mean=sum(milliseconds)/len(milliseconds))
                value["features"] = errors[name][h].compute()
                route["horizons"][str(h)] = value
            report["routes"][name] = route
        report["comparisons"] = paired_deltas(report["routes"], horizons)
        report["elapsed_seconds"] = time.monotonic()-started
        artifacts = {"manifest.json": {key: report[key] for key in ("split", "steps", "horizons", "sample_ids", "sample_digest", "samples")}}
        if predictions is not None:
            artifacts["predictions.json"] = predictions
        for name, value in artifacts.items():
            with (folder/name).open("x") as stream:
                json.dump(value, stream, indent=2, allow_nan=False)
                stream.write("\n")
        with (folder/"summary.md").open("x") as stream:
            stream.write(markdown_summary(report))
        # Written last: presence of report.json indicates complete evaluation.
        with (folder/"report.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--dinof-checkpoint")
    parser.add_argument("--detector-checkpoint")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--horizons", type=int, nargs="+")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("32-true", "bf16-mixed"))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--manifest")
    parser.add_argument("--save-predictions", action="store_true")
    args = vars(parser.parse_args(argv))
    report = diagnose(**args)
    print(json.dumps({name: {h: row["metrics"] for h, row in route["horizons"].items()}
                      for name, route in report["routes"].items()}, indent=2))


if __name__ == "__main__":
    main()
