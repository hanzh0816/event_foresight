"""Fixed-checkpoint Event interventions and optional gradients; never optimizes weights.

Run --help without model dependencies. All intervention reports are diagnostics,
not independent forecasting baselines. See docs/research/CONDITIONING_DIAGNOSTICS.md.
"""
from contextlib import ExitStack, nullcontext
from copy import deepcopy
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

from .diagnostic_protocol import (
    ALLOWED_VARIANTS, deterministic_bin_permutation, make_event_donor_map,
    validate_variants,
)


class PairedDiagnosticDataset:
    """Keep receiver RGB/targets fixed and load only the donor's historical Events."""

    def __init__(self, dataset, sample_ids, donor_map):
        self.dataset = dataset
        self.sample_ids = list(sample_ids)
        self.lookup = {sid: i for i, sid in enumerate(dataset.sample_ids)}
        self.donor_map = donor_map

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sid = self.sample_ids[index]
        sample = dict(self.dataset[self.lookup[sid]])
        donor = self.donor_map.get(sid)
        sample["donor_sample_id"] = donor
        if donor is not None:
            donor_index = self.lookup[donor]
            sample["donor_events"] = self.dataset.read_history_events(donor_index)
        return sample


def collate_diagnostic(samples):
    import torch
    from .data import collate_research
    batch = collate_research(samples)
    batch["donor_sample_ids"] = [sample["donor_sample_id"] for sample in samples]
    if any("donor_events" in sample for sample in samples):
        if not all("donor_events" in sample for sample in samples):
            raise ValueError("Mixed donor availability in diagnostic batch")
        batch["donor_events"] = torch.stack([sample["donor_events"] for sample in samples])
    return batch


def intervene_events(events, variant, sample_ids, seed, donor_events=None):
    """Change raw bins before CNN/position encoding; never shuffle batch rows in-place."""
    import torch
    if variant == "mismatched":
        if donor_events is None or donor_events.shape != events.shape:
            raise ValueError("mismatched requires a same-shaped fixed donor tensor")
        return donor_events
    if variant == "bin_reverse":
        return events.flip(2)
    if variant == "bin_shuffle":
        permutations = [deterministic_bin_permutation(events.shape[2], seed, sid) for sid in sample_ids]
        return torch.stack([value[:, permutation] for value, permutation in zip(events, permutations, strict=True)])
    if variant == "zero":
        return torch.zeros_like(events)
    if variant not in ("correct", "query_only", "off"):
        raise ValueError(f"Unknown Event intervention: {variant}")
    return events


def _number(value):
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError("Non-finite diagnostic value")
    return result


def _rms(value):
    return _number(value.detach().float().square().mean().sqrt().item())


def _ratio(left, right):
    denominator = _rms(right)
    return _rms(left) / denominator if denominator else None


def _regional(values, foreground):
    """Per-patch values -> regional means and counts (empty regions stay null)."""
    import torch
    result, counts = {}, {}
    for region, mask in (("all", torch.ones_like(foreground)), ("fg", foreground), ("bg", ~foreground)):
        count = int(mask.sum().item())
        counts[region] = count
        result[region] = _number(values[mask].mean().item()) if count else None
    return result, counts


def sample_metrics(prediction, target, reference, foreground, diagnostics=None, baseline=None):
    """One sample, scene tensors [N,D]. Values are detached JSON scalars."""
    from torch.nn import functional as F
    prediction, target, reference = prediction.float(), target.float(), reference.float()
    l1, counts = _regional((prediction - target).abs().mean(-1), foreground)
    smooth, _ = _regional(F.smooth_l1_loss(prediction, target, reduction="none").mean(-1), foreground)
    change, _ = _regional((prediction - reference).abs().mean(-1), foreground)
    features = {f"{region}_{metric}": source[region]
                for metric, source in (("l1", l1), ("smooth_l1", smooth)) for region in counts}
    details = {}
    if diagnostics is not None:
        for key, tensor in diagnostics.items():
            details[key + "_rms"] = _rms(tensor)
            if baseline is not None and key in baseline:
                details[key + "_change_l1"] = _number(
                    (tensor.detach().float() - baseline[key].detach().float()).abs().mean().item())
        details["update_to_hidden_l2_ratio"] = _ratio(diagnostics["update"], diagnostics["hidden"])
        details["attention_to_query_l2_ratio"] = _ratio(diagnostics["attention"], diagnostics["query"])
        encoded = diagnostics["encoded_events"]
        # Distinguish content variation from variation introduced by position embeddings.
        details["encoded_temporal_variation_rms"] = _rms(encoded - encoded.mean(dim=1, keepdim=True))
    return dict(features=features, patch_counts=counts,
                prediction_change_vs_correct={f"{region}_l1": value for region, value in change.items()},
                diagnostics=details)


class MetricAccumulator:
    """Weight feature metrics by patch count; diagnostic scalars by sample count."""

    def __init__(self):
        self.num_samples = 0
        self.counts = dict(all=0, fg=0, bg=0)
        self.sums = {}
        self.diag_sums, self.diag_counts = {}, {}

    def update(self, row):
        self.num_samples += 1
        for region, count in row["patch_counts"].items():
            self.counts[region] += count
        for field in ("features", "prediction_change_vs_correct"):
            for key, value in row[field].items():
                region = key.split("_", 1)[0]
                if value is not None:
                    total_key = (field, key)
                    self.sums[total_key] = self.sums.get(total_key, 0.) + value * row["patch_counts"][region]
        for key, value in row["diagnostics"].items():
            if value is not None:
                self.diag_sums[key] = self.diag_sums.get(key, 0.) + value
                self.diag_counts[key] = self.diag_counts.get(key, 0) + 1

    def compute(self):
        result = dict(num_samples=self.num_samples)
        for field, metrics in (("features", ("l1", "smooth_l1")), ("prediction_change_vs_correct", ("l1",))):
            result[field] = {f"{region}_{metric}": (
                self.sums.get((field, f"{region}_{metric}"), 0.) / count if count else None)
                for region, count in self.counts.items() for metric in metrics}
        result["features"]["foreground_patch_fraction"] = (
            self.counts["fg"] / self.counts["all"] if self.counts["all"] else None)
        result["diagnostic_means"] = {key: value / self.diag_counts[key] for key, value in self.diag_sums.items()}
        return result


def _gradient_statistics(parameters, gradients):
    """Separate packed Q/K/V gradients and retain disconnected versus zero states."""
    groups = {}
    for (name, parameter), gradient in zip(parameters, gradients, strict=True):
        if name in ("cross_attention.in_proj_weight", "cross_attention.in_proj_bias"):
            chunks = parameter.chunk(3, dim=0)
            grad_chunks = gradient.chunk(3, dim=0) if gradient is not None else (None,) * 3
            entries = [(f"cross_attention_{tag}", p, g) for tag, p, g in zip(("q", "k", "v"), chunks, grad_chunks, strict=True)]
        else:
            key = "cross_attention_out" if name.startswith("cross_attention.out_proj.") else name.split(".", 1)[0]
            entries = [(key, parameter, gradient)]
        for key, value, grad in entries:
            item = groups.setdefault(key, dict(numel=0, grad_numel=0, nonzero_grad_numel=0,
                                               parameter_square_sum=0., gradient_square_sum=0., disconnected_tensors=0))
            item["numel"] += value.numel()
            item["parameter_square_sum"] += _number(value.detach().float().square().sum().item())
            if grad is None:
                item["disconnected_tensors"] += 1
            else:
                item["grad_numel"] += grad.numel()
                item["nonzero_grad_numel"] += int(grad.count_nonzero().item())
                item["gradient_square_sum"] += _number(grad.detach().float().square().sum().item())
    for item in groups.values():
        item["parameter_rms"] = math.sqrt(item.pop("parameter_square_sum") / item["numel"])
        square_sum = item.pop("gradient_square_sum")
        item["gradient_rms"] = math.sqrt(square_sum / item["numel"]) if item["grad_numel"] else None
        item["nonzero_gradient_fraction"] = item["nonzero_grad_numel"] / item["numel"]
    return groups


def probe_gradients(task, history, events, targets, sample_ids, amp):
    """Local scene-loss gradients, one sample at a time; no backward/optimizer step.

    The predictor is held constant to avoid allocating its parameter gradients.
    autograd.grad differentiates through its suffix into the condition branch.
    """
    import torch
    flags = [(parameter, parameter.requires_grad) for parameter in task.parameters()]
    rows = []
    try:
        task.forecaster.requires_grad_(False)
        task.model.requires_grad_(True)
        parameters = list(task.model.named_parameters())
        for index, sid in enumerate(sample_ids):
            with torch.enable_grad():
                # Inputs encoded under inference_mode must become normal tensors.
                z = history[index:index + 1].detach().clone()
                e = events[index:index + 1].detach().clone().requires_grad_(True)
                target = targets[index:index + 1].detach().clone()
                with amp():
                    result = task.model(z, e, task.forecaster.predictor)
                    loss = task._distance(result["scene"], target) * task.config["loss"]["scene_weight"]
                gradients = torch.autograd.grad(loss, [p for _, p in parameters] + [e], allow_unused=True)
                input_grad = gradients[-1]
                rows.append(dict(sample_id=sid, loss=_number(loss.detach().item()),
                                 modules=_gradient_statistics(parameters, gradients[:-1]),
                                 event_input_gradient_rms=None if input_grad is None else _rms(input_grad),
                                 optimizer_steps=0))
    finally:
        for parameter, requires_grad in flags:
            parameter.requires_grad_(requires_grad)
    return rows


def _write_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _json_line(stream, value):
    stream.write(json.dumps(value, allow_nan=False) + "\n")


def _git_status():
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=root, text=True)
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _time_summary(values):
    return dict(min=min(values), max=max(values), mean=sum(values) / len(values)) if values else None


def run(config, *, checkpoint, output_dir, split="val", manifest=None, max_samples=None,
        variants=ALLOWED_VARIANTS, seed=42, gradient_samples=0, loader_batch_size=None,
        num_workers=0, skip_detection=False):
    import torch
    from torch.utils.data import DataLoader
    from .config import validate_config
    from .data import ResearchDSECDataset, HorizonDatasetView
    from .evaluate import (dataset_kwargs, feature_space_digest, file_sha256, foreground_mask,
                           module_digest, parameter_counts, sample_digest)
    from .task import git_revision, load_task
    from ..evaluation import FutureDetectionEvaluator
    from ..models.detector import load_detector, postprocess

    config = validate_config(config)
    variants = validate_variants(variants)
    if config["stage"] != "conditioning" or config["model"]["mode"] not in ("event_ordered", "event_pooled"):
        raise ValueError("Diagnostics require a trained Event conditioning model")
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError("A trained conditioning checkpoint is required")
    if split not in ("train", "val", "test"):
        raise ValueError("Unknown split")
    for name, value, minimum in (("seed", seed, 0), ("gradient_samples", gradient_samples, 0),
                                  ("num_workers", num_workers, 0)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name, value in (("max_samples", max_samples), ("loader_batch_size", loader_batch_size)):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{name} must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Diagnostic runner is single-process; do not launch with torchrun/DDP")
    requested = config["runtime"]["accelerator"]
    device = torch.device("cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu")
    if requested == "gpu" and device.type != "cuda":
        raise RuntimeError("GPU diagnostics requested, but CUDA is unavailable")
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)
    kwargs = dataset_kwargs(config, split, future_targets=True)
    if manifest is not None:
        saved_manifest = json.loads(Path(manifest).read_text())
        if isinstance(saved_manifest, dict):
            if saved_manifest.get("split") != split:
                raise ValueError("Manifest split mismatch")
            kwargs["sample_ids"] = saved_manifest["sample_ids"]
        else:
            kwargs["sample_ids"] = saved_manifest
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a new diagnostic output directory: {output}")
    dataset = ResearchDSECDataset(**kwargs)
    try:
        full_ids = list(dataset.sample_ids)
        if not full_ids:
            raise ValueError("Empty diagnostic inventory")
        donor_map = make_event_donor_map(full_ids, seed) if "mismatched" in variants else {}
        selected_ids = full_ids if max_samples is None else full_ids[:max_samples]
        paired = PairedDiagnosticDataset(dataset, selected_ids, donor_map)
        loader = DataLoader(paired, batch_size=loader_batch_size or config["data"]["batch_size"],
                            shuffle=False, num_workers=num_workers, collate_fn=collate_diagnostic,
                            **({"multiprocessing_context": "spawn"} if num_workers else {}))
        task = load_task(config, checkpoint=checkpoint).eval().to(device)
        detector = None if skip_detection else load_detector(config["model"]["detector_checkpoint"]).eval().requires_grad_(False).to(device)
        evaluators = {} if detector is None else {
            name: FutureDetectionEvaluator(HorizonDatasetView(dataset, 1), sample_ids=selected_ids) for name in variants}
        accumulators = {name: MetricAccumulator() for name in variants}
        counts = parameter_counts(task)
        task_digest_before = module_digest(task)
        precision = config["runtime"]["precision"]
        amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and precision == "bf16-mixed" else nullcontext()
        output.mkdir(parents=True, exist_ok=False)
        provenance = dict(format="event-conditioning-diagnostics-v1", evaluation_kind="conditioning_intervention_diagnostic",
            config=deepcopy(config), git_commit=git_revision(), git_status=_git_status(), split=split,
            checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=file_sha256(checkpoint),
            dinof_sha256=file_sha256(config["model"]["dinof_checkpoint"]),
            detector_sha256=None if detector is None else file_sha256(config["model"]["detector_checkpoint"]),
            feature_space_state_sha256=feature_space_digest(task.forecaster),
            predictor_state_sha256=module_digest(task.forecaster.predictor),
            conditioner_state_sha256=module_digest(task.model), parameter_counts=counts,
            task_state_sha256=task_digest_before,
            execution=dict(device=str(device), precision=precision, loader_batch_size=loader.batch_size,
                           num_workers=num_workers, seed=seed, gradient_samples_requested=gradient_samples,
                           data_reader="online_hdf5", variants=list(variants), optimizer_steps=0),
            sample_ids=selected_ids, sample_digest=sample_digest(selected_ids),
            scope=dict(full_inventory_size=len(full_ids), evaluated_samples=len(selected_ids),
                       available_inventory_size_before_manifest=dataset.full_inventory_size,
                       excluded_windows=dict(dataset.excluded_windows),
                       is_full_inventory=selected_ids == full_ids, donor_pool_sample_digest=sample_digest(full_ids),
                       manifest=str(Path(manifest).resolve()) if manifest else None,
                       manifest_sha256=file_sha256(manifest) if manifest else None))
        _write_json(output / "run.json", provenance)
        _write_json(output / "manifest.json", dict(split=split, sample_ids=selected_ids))
        _write_json(output / "donor_map.json", dict(seed=seed, split=split, policy="different_sequence_with_replacement",
                                                    donor_pool_sample_digest=sample_digest(full_ids), mapping=donor_map))
        started, seen, gradient_count, identity_max_error = time.monotonic(), [], 0, None
        timing = {name: [] for name in ("actual_event_history_ms", "rgb_observation_support_ms", "cutoff_to_target_ms")}
        with ExitStack() as stack:
            rows_files = {name: stack.enter_context((output / f"{name}.samples.jsonl").open("x")) for name in variants}
            pred_files = {} if detector is None else {
                name: stack.enter_context((output / f"{name}.predictions.jsonl").open("x")) for name in variants}
            gradient_file = stack.enter_context((output / "gradients.jsonl").open("x"))
            for batch in loader:
                batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                ids = [meta["sample_id"] for meta in batch["meta"]]
                for meta in batch["meta"]:
                    timing["actual_event_history_ms"].append((meta["input_cutoff_us"] - meta["event_history_start_us"]) / 1000)
                    timing["rgb_observation_support_ms"].append((meta["input_cutoff_us"] - meta["history_start_us"]) / 1000)
                    timing["cutoff_to_target_ms"].append(meta["horizon_us"][0] / 1000)
                with torch.inference_mode(), amp():
                    history, targets = task.encode_history(batch), task.encode_targets(batch)
                    events = batch["history_events"]
                    correct = task.model.forward_diagnostic(history, events, task.forecaster.predictor)
                    if identity_max_error is None:
                        original = task.model(history, events, task.forecaster.predictor)["scene"]
                        identity_max_error = _number((original.float() - correct["scene"].float()).abs().max().item())
                        torch.testing.assert_close(original, correct["scene"], rtol=0, atol=0)
                    foreground = foreground_mask([target[0] for target in batch["targets"]], task.forecaster.grid_size, device)
                    for variant in variants:
                        changed_events = intervene_events(events, variant, ids, seed, batch.get("donor_events"))
                        if variant == "correct":
                            result = correct
                        elif variant == "off":
                            result = {"scene": task.forecaster.predictor(history).unsqueeze(1)}
                        else:
                            result = task.model.forward_diagnostic(history, changed_events, task.forecaster.predictor,
                                condition_mode="query_only" if variant == "query_only" else "full")
                        for index, sid in enumerate(ids):
                            diagnostics = ({key: tensor[index] for key, tensor in result["diagnostics"].items()}
                                           if "diagnostics" in result else None)
                            baseline = {key: tensor[index] for key, tensor in correct["diagnostics"].items()}
                            row = sample_metrics(result["scene"][index, 0], targets[index, 0],
                                                 correct["scene"][index, 0], foreground[index], diagnostics, baseline)
                            row.update(sample_id=sid, variant=variant,
                                donor_sample_id=batch["donor_sample_ids"][index] if variant == "mismatched" else None,
                                event_input_change_l1=_number((changed_events[index].float() - events[index].float()).abs().mean().item()),
                                metadata=batch["meta"][index])
                            accumulators[variant].update(row)
                            _json_line(rows_files[variant], row)
                        if detector is not None:
                            decoded = postprocess(detector(task.forecaster.pca.inverse(result["scene"][:, 0])))
                            predictions = []
                            for detected, meta in zip(decoded, batch["meta"], strict=True):
                                item = dict(sample_id=meta["sample_id"], target_id=meta["target_ids"][0],
                                    boxes=detected["boxes"].float().cpu().tolist(),
                                    class_ids=detected["labels"].cpu().tolist(), scores=detected["scores"].float().cpu().tolist())
                                predictions.append(item)
                                _json_line(pred_files[variant], item)
                            evaluators[variant].update(predictions)
                remaining = min(len(ids), gradient_samples - gradient_count)
                if remaining > 0:
                    for row in probe_gradients(task, history[:remaining], events[:remaining], targets[:remaining], ids[:remaining], amp):
                        _json_line(gradient_file, row)
                        gradient_count += 1
                seen.extend(ids)
                print(f"diagnosed {len(seen)}/{len(selected_ids)} samples; gradient probes {gradient_count}", flush=True)
        if seen != selected_ids:
            raise RuntimeError("Diagnostic sample order differs from paired inventory")
        if module_digest(task) != task_digest_before:
            raise RuntimeError("Diagnostics unexpectedly changed model weights or buffers")
        report = dict(provenance, elapsed_seconds=time.monotonic() - started, variants={},
            protocol_timing={key: _time_summary(values) for key, values in timing.items()},
            diagnostic_forward_max_abs_error=identity_max_error, weights_unchanged=True,
            gradient_probe=dict(num_samples=gradient_count, file="gradients.jsonl", optimizer_steps=0,
                description="Local scene-loss derivatives at the loaded checkpoint, eval mode, one sample per probe; no optimizer or historical training-gradient claim."),
            artifacts=dict(run="run.json", manifest="manifest.json", donor_map="donor_map.json", gradients="gradients.jsonl",
                           samples={name: f"{name}.samples.jsonl" for name in variants},
                           predictions={name: f"{name}.predictions.jsonl" for name in pred_files}))
        for name in variants:
            report["variants"][name] = accumulators[name].compute()
            report["variants"][name]["detection"] = evaluators[name].compute().to_dict() if detector is not None else None
            if detector is not None:
                metadata = report["variants"][name]["detection"]["metadata"]
                # Legacy evaluator history refers to eligibility, not the four
                # frames actually passed to DinoF. Preserve it under explicit names.
                for old, new in (("history_ms", "inventory_history_ms"),
                                 ("actual_history_ms", "actual_inventory_history_ms")):
                    if old in metadata:
                        metadata[new] = metadata.pop(old)
                metadata.update(deepcopy(report["protocol_timing"]))
        from .conditioning_report import render_summary
        with (output / "summary.md").open("x") as stream:
            stream.write(render_summary(report))
        # Completion marker: consumers only treat report.json as a complete run.
        _write_json(output / "report.json", report)
        return report
    finally:
        dataset.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", required=True, help="Original archived config_train.yaml; scientific recipe stays unchanged")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True, help="New directory; never overwrite prior results")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--manifest", help="Original evaluation JSON or {split,sample_ids}; defines the complete donor pool")
    parser.add_argument("--max-samples", type=int, help="Receiver prefix only; donor pool still uses the complete manifest")
    parser.add_argument("--variants", nargs="+", choices=ALLOWED_VARIANTS, default=list(ALLOWED_VARIANTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-samples", type=int, default=0, help="Local gradient probes without optimizer steps; default disabled")
    parser.add_argument("--loader-batch-size", type=int, help="Execution override without changing checkpoint recipe")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-detection", action="store_true", help="Only feature/dependency diagnostics; AP stays null")
    args = parser.parse_args(argv)
    from .config import load_config
    config = load_config(args.config)
    result = run(config, checkpoint=args.checkpoint, output_dir=args.output_dir, split=args.split,
                 manifest=args.manifest, max_samples=args.max_samples, variants=args.variants,
                 seed=args.seed, gradient_samples=args.gradient_samples, loader_batch_size=args.loader_batch_size,
                 num_workers=args.num_workers, skip_detection=args.skip_detection)
    print(json.dumps({name: item["features"] for name, item in result["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
