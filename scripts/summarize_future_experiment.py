"""Summarize the bounded future-supervision experiment without loading models on GPU.

All inputs are existing local artifacts. Outputs are JSON and plain text only.
Checkpoint comparisons use CPU mmap and release each trained checkpoint in turn.
"""

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

GROUPS = ("rgb", "e2_baseline", "current_only", "future_only", "current_future")
TRAINED = GROUPS[2:]
READOUTS = tuple(f"{phase}_{head}" for phase in ("current", "future")
                 for head in ("old_head_fp32", "real_feature_head_fp32"))
FIXED_EVENT = ("mean", "std", "pca_mean", "pca_components", "input_mean", "input_std")


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def resolve(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT / path).resolve()


def discover_evaluations(root, overrides=(), *, enabled=True):
    aliases = {"e2": "e2_baseline", "baseline": "e2_baseline"}
    found = {}
    for folder in sorted(root.glob("eval*")) if enabled else ():
        if not folder.is_dir() or not (folder / "run.json").exists():
            continue
        label = folder.name.removeprefix("eval_")
        label = aliases.get(label, label)
        # Extra seeds/subsets must be explicitly mapped; their checkpoint may
        # belong to a main arm, but their evaluation inventory is different.
        if label not in GROUPS:
            continue
        if label in GROUPS:
            if label in found:
                raise ValueError(f"Multiple evaluation folders for {label}; supply --eval {label}=PATH")
            found[label] = folder
    for value in overrides:
        label, path = value.split("=", 1)
        if label not in GROUPS:
            raise ValueError(f"Unknown evaluation group {label}")
        found[label] = resolve(path)
    return found


def metric_delta(left, right):
    return {key: None if left.get(key) is None or right.get(key) is None else
            100.0 * (left[key] - right[key]) for key in left.keys() | right.keys()}


def comparisons(rows):
    result = {}
    pairs = [(group, "e2_baseline") for group in TRAINED]
    pairs += [("current_future", "current_only"), ("future_only", "current_only"),
              ("current_future", "future_only")]
    for left, right in pairs:
        if left not in rows or right not in rows:
            continue
        item = {}
        for name in READOUTS:
            a, b = rows[left]["readouts"].get(name), rows[right]["readouts"].get(name)
            if a is None or b is None:
                continue
            classes = {cls: None if a["per_class"][cls]["AP"] is None or
                       b["per_class"][cls]["AP"] is None else
                       100 * (a["per_class"][cls]["AP"] - b["per_class"][cls]["AP"])
                       for cls in a["per_class"]}
            main = [classes.get(cls) for cls in ("car", "pedestrian")]
            item[name] = {"metric_delta_points": metric_delta(a["metrics"], b["metrics"]),
                          "per_class_AP_delta_points": classes,
                          "car_pedestrian_mean_AP_delta_points":
                          sum(main) / 2 if all(v is not None for v in main) else None}
        item["feature_error_delta"] = {
            phase: {key: rows[left]["feature_errors"][phase][key] - value
                    for key, value in rows[right]["feature_errors"][phase].items()}
            for phase in ("current", "future")}
        result[f"{left}_minus_{right}"] = item
    return result


def load_evaluations(found, expected_samples):
    rows, audits, missing = {}, {}, []
    reference_ids, reference_gt = None, {}
    for group in GROUPS:
        folder = found.get(group)
        if folder is None or not (folder / "completed.json").exists():
            missing.append(f"evaluation:{group}")
            continue
        done, run = read(folder / "completed.json"), read(folder / "run.json")
        if done.get("status") != "completed":
            missing.append(f"evaluation:{group}:not_completed")
            continue
        ids = run["sample_ids"]
        if reference_ids is None:
            reference_ids = ids
        audit = {"samples": len(ids), "expected_samples": expected_samples,
                 "sample_inventory_matches": ids == reference_ids,
                 "sample_count_matches": len(ids) == expected_samples,
                 "sample_inventory_sha256": digest(ids), "readouts": {}, "ground_truth": {}}
        rows[group] = {"path": str(folder), "feature_errors": done["feature_errors"],
                       "elapsed_seconds": done.get("elapsed_seconds"), "readouts": {}}
        for name in READOUTS:
            metric_path, prediction_path = folder / f"metrics_{name}.json", folder / f"predictions_{name}.json"
            if not metric_path.exists() or not prediction_path.exists():
                missing.append(f"evaluation:{group}:{name}")
                continue
            metrics = read(metric_path)
            rows[group]["readouts"][name] = {key: metrics[key] for key in ("metrics", "per_class", "metadata")}
            predictions = read(prediction_path)["predictions"]
            current = name.startswith("current_")
            expected_targets = [f"{sample.rsplit(':', 1)[0]}/{int(sample.rsplit(':', 1)[1]) + (0 if current else 1):06d}"
                                for sample in ids]
            audit["readouts"][name] = {
                "sample_order_matches": [p["sample_id"] for p in predictions] == ids,
                "target_ids_match_timeline": [p["target_id"] for p in predictions] == expected_targets,
                "prediction_records": len(predictions),
                "missing_prediction_records": metrics["metadata"].get("num_missing_prediction_records"),
            }
        for phase in ("current", "future"):
            # diagnose_readout.py writes ground_truth_*.json, not coco_*.json.
            path = folder / f"ground_truth_{phase}.json"
            if not path.exists():
                missing.append(f"evaluation:{group}:ground_truth_{phase}")
                continue
            coco = read(path)
            value = digest(coco)
            reference_gt.setdefault(phase, value)
            populated = {a["image_id"] for a in coco["annotations"]}
            audit["ground_truth"][phase] = {
                "path": str(path), "sha256_canonical": value,
                "matches_other_groups": value == reference_gt[phase],
                "images": len(coco["images"]), "annotations": len(coco["annotations"]),
                "empty_sample_ids": [i["sample_id"] for i in coco["images"] if i["id"] not in populated],
            }
        checks = [audit["sample_inventory_matches"], audit["sample_count_matches"]]
        checks += [a["sample_order_matches"] and a["target_ids_match_timeline"] and
                   a["missing_prediction_records"] == 0 for a in audit["readouts"].values()]
        checks += [a["matches_other_groups"] and a["images"] == expected_samples
                   for a in audit["ground_truth"].values()]
        audit["passed"] = all(checks) and len(audit["readouts"]) == 4 and len(audit["ground_truth"]) == 2
        audits[group] = audit
    return rows, audits, missing


def audit_training(root, planned_updates):
    import yaml
    result, missing, traces = {}, [], {}
    reference_config = None
    for group in TRAINED:
        folder = root / group
        paths = {name: folder / name for name in
                 ("sample_order.jsonl", "training_progress.json", "gradient_audit.json", "config_train.yaml")}
        absent = [name for name, path in paths.items() if not path.exists()]
        if absent:
            missing.extend(f"training:{group}:{name}" for name in absent)
            result[group] = {"passed": False, "missing": absent}
            continue
        records = [json.loads(line) for line in paths["sample_order.jsonl"].read_text().splitlines() if line.strip()]
        traces[group] = records
        progress, gradient = read(paths["training_progress.json"]), read(paths["gradient_audit.json"])
        config = yaml.safe_load(paths["config_train.yaml"].read_text())
        comparable = {key: config[key] for key in ("data", "model", "runtime", "optimization")}
        comparable = json.loads(json.dumps(comparable))
        for key in ("current_weight", "future_weight"):
            comparable["optimization"].pop(key, None)
        if reference_config is None:
            reference_config = comparable
        counts = Counter(row["global_step"] for row in records)
        examples = sum(len(row["sample_ids"]) for row in records)
        accumulation = config["optimization"].get("accumulation_steps", 1)
        world = progress["world_size"]
        expected = planned_updates * config["data"]["batch_size"] * accumulation
        checks = {
            "same_recipe_except_loss_weights": comparable == reference_config,
            "optimizer_steps_match_plan": progress["optimizer_steps"] == planned_updates,
            "configured_steps_match_plan": config["optimization"]["max_steps"] == planned_updates,
            "trace_microbatches_match": len(records) == progress["microbatches_seen_per_rank"],
            "trace_examples_match": examples == progress["examples_seen_per_rank"] == expected,
            "complete_accumulation_groups": counts == dict.fromkeys(range(planned_updates), accumulation),
            "single_rank_trace_covers_run": world == 1,
            "encoder_gradient_nonzero": gradient["encoder_gradient_norm"] > 0,
            "frozen_parameters_have_no_grad": gradient["frozen_forecaster_parameters_with_grad"] == [],
            "predictor_frozen_eval": not gradient["predictor_trainable"] and gradient["predictor_eval"],
        }
        result[group] = {"checks": checks, "passed": all(checks.values()), "progress": progress,
                         "gradient_audit": gradient, "sample_trace_sha256": file_digest(paths["sample_order.jsonl"]),
                         "sample_trace_canonical_sha256": digest(records), "examples_per_rank": examples,
                         "global_examples": examples * world}
    complete = len(traces) == len(TRAINED)
    trace_equal = complete and all(traces[group] == traces[TRAINED[0]] for group in TRAINED)
    return {"groups": result, "all_traces_available": complete, "exact_sample_traces_equal": trace_equal,
            "passed": complete and trace_equal and all(item["passed"] for item in result.values())}, missing


def compare_tensors(actual, expected):
    import torch
    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(actual.keys() - expected.keys())
    changed = [key for key in expected.keys() & actual.keys()
               if actual[key].dtype != expected[key].dtype or actual[key].shape != expected[key].shape or
               not torch.equal(actual[key].detach().contiguous().reshape(-1).view(torch.uint8),
                               expected[key].detach().contiguous().reshape(-1).view(torch.uint8))]
    return {"tensor_count": len(actual), "missing": missing, "extra": extra,
            "changed": sorted(changed), "identical": not (missing or extra or changed)}


def audit_checkpoints(root, plan, planned_updates):
    import torch
    import yaml
    from foresight.models.forecaster import internal_state_dict
    torch.set_num_threads(2)
    config = yaml.safe_load((root / "train.yaml").read_text())
    initial_path = resolve(config["model"]["dinof_checkpoint"])
    initial = torch.load(initial_path, map_location="cpu", weights_only=True, mmap=True)
    initial_forecaster = internal_state_dict(initial["state_dict"])
    event_path = resolve(plan["baseline_encoder"])
    event = torch.load(event_path, map_location="cpu", weights_only=True, mmap=True)["state_dict"]
    result = {"initial_forecaster": str(initial_path), "initial_encoder": str(event_path), "groups": {}}
    missing = []
    smoke = root / "smoke/checkpoints/last.ckpt"
    if smoke.exists():
        saved = torch.load(smoke, map_location="cpu", weights_only=False, mmap=True)
        state = {k.removeprefix("forecaster."): v for k, v in saved["state_dict"].items() if k.startswith("forecaster.")}
        result["smoke_forecaster_vs_official"] = compare_tensors(state, initial_forecaster)
        del saved, state
        gc.collect()
    for group in TRAINED:
        path = root / group / "checkpoints/last.ckpt"
        if not path.exists():
            missing.append(f"checkpoint:{group}")
            continue
        saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        state = saved["state_dict"]
        forecaster = {k.removeprefix("forecaster."): v for k, v in state.items() if k.startswith("forecaster.")}
        encoder = {k.removeprefix("event_encoder."): v for k, v in state.items() if k.startswith("event_encoder.")}
        frozen = compare_tensors(forecaster, initial_forecaster)
        full_encoder = compare_tensors(encoder, event)
        fixed = compare_tensors({k: encoder[k] for k in FIXED_EVENT}, {k: event[k] for k in FIXED_EVENT})
        updated = [k for k in full_encoder["changed"] if k not in FIXED_EVENT]
        progress_path = root / group / "training_progress.json"
        progress = read(progress_path) if progress_path.exists() else {}
        checkpoint_progress = saved.get("future_event_progress", {})
        progress_matches = all(checkpoint_progress.get(key) == progress.get(key) for key in
                               ("microbatches_seen_per_rank", "examples_seen_per_rank")) and bool(checkpoint_progress)
        audit = {"path": str(path), "bytes": path.stat().st_size, "global_step": saved.get("global_step"),
                 "forecaster_vs_official": frozen, "fixed_event_buffers_vs_e2": fixed,
                 "event_tensors_changed": len(updated), "changed_event_tensors": updated,
                 "missing_event_tensors": full_encoder["missing"], "extra_event_tensors": full_encoder["extra"],
                 "checkpoint_progress": checkpoint_progress, "checkpoint_progress_matches": progress_matches,
                 "predictor_trainable": saved.get("input_spec", {}).get("trainable_predictor"),
                 "format": saved.get("foresight_format"), "objective": saved.get("input_spec", {})}
        audit["passed"] = (frozen["identical"] and fixed["identical"] and bool(updated) and
                           not full_encoder["missing"] and not full_encoder["extra"] and progress_matches and
                           saved.get("global_step") == planned_updates and audit["predictor_trainable"] is False)
        result["groups"][group] = audit
        del saved, state, forecaster, encoder
        gc.collect()
    result["passed"] = len(result["groups"]) == 3 and all(a["passed"] for a in result["groups"].values())
    if "smoke_forecaster_vs_official" in result:
        result["passed"] &= result["smoke_forecaster_vs_official"]["identical"]
    return result, missing


def score_saved(coco, predictions, omitted_sequence=None):
    """Recompute exact saved predictions, optionally excluding one sequence."""
    from foresight.evaluation import FutureDetectionEvaluator
    images = [image for image in coco["images"] if
              omitted_sequence is None or image["sample_id"].rsplit(":", 1)[0] != omitted_sequence]
    ids = {image["id"] for image in images}
    key_to_id = {(image["sample_id"], image["target_id"]): image["id"] for image in images}
    subset = {**coco, "images": images, "annotations": [a for a in coco["annotations"] if a["image_id"] in ids]}
    results = []
    for record in predictions:
        image_id = key_to_id.get((record["sample_id"], record["target_id"]))
        if image_id is None:
            continue
        for box, label, score in zip(record["boxes"], record["class_ids"], record["scores"], strict=True):
            x, y, x2, y2 = box
            results.append({"image_id": image_id, "category_id": int(label) + 1,
                            "bbox": [x, y, x2-x, y2-y], "score": score})
    evaluator = FutureDetectionEvaluator.__new__(FutureDetectionEvaluator)
    evaluator.max_dets = 100
    metrics, per_class = evaluator._evaluate_coco(subset, results)
    return {"metrics": metrics, "per_class": per_class, "samples": len(images),
            "effective_classes": sum(row["num_gt"] > 0 for row in per_class.values())}


def sequence_sensitivity(found, rows, left="current_future", right="current_only"):
    if left not in rows or right not in rows:
        return {"status": "missing_comparison_groups", "left": left, "right": right}
    result = {"left": left, "right": right, "readouts": {},
              "limitation": "Overlapping leave-one-sequence-out sensitivity views; not independent trials or confidence intervals."}
    for name in READOUTS:
        phase = name.split("_", 1)[0]
        coco = read(found[left] / f"ground_truth_{phase}.json")
        if digest(coco) != digest(read(found[right] / f"ground_truth_{phase}.json")):
            raise ValueError(f"Ground truth differs for {left}/{right}/{phase}")
        sequences = sorted({image["sample_id"].rsplit(":", 1)[0] for image in coco["images"]})
        predictions = {group: read(found[group] / f"predictions_{name}.json")["predictions"] for group in (left, right)}
        values = {}
        for sequence in (None, *sequences):
            scores = {group: score_saved(coco, predictions[group], sequence) for group in (left, right)}
            a, b = scores[left], scores[right]
            classes = {cls: None if a["per_class"][cls]["AP"] is None or b["per_class"][cls]["AP"] is None else
                       100 * (a["per_class"][cls]["AP"] - b["per_class"][cls]["AP"]) for cls in a["per_class"]}
            main = [classes.get(cls) for cls in ("car", "pedestrian")]
            values[sequence or "all_sequences"] = {
                "samples": a["samples"], "effective_classes": a["effective_classes"],
                "metric_delta_points": metric_delta(a["metrics"], b["metrics"]),
                "per_class_AP_delta_points": classes,
                "car_pedestrian_mean_AP_delta_points": sum(main)/2 if all(v is not None for v in main) else None,
            }
            if sequence is None:
                values["all_sequences"]["rescore_matches_saved"] = all(
                    all(value is None and rows[group]["readouts"][name]["metrics"][key] is None or
                        value is not None and abs(value - rows[group]["readouts"][name]["metrics"][key]) < 1e-10
                        for key, value in scores[group]["metrics"].items()) for group in (left, right))
        result["readouts"][name] = values
    result["status"] = "completed"
    return result


def plain_text(report):
    lines = ["Event future-supervision experiment", "", f"Status: {report['status']}",
             "Detection columns are percentages; differences are percentage points.",
             "Current and future tasks use different target frames and must not be directly subtracted.", ""]
    def fmt(value):
        return "NA" if value is None else f"{100*value:.3f}"
    for head in ("old_head_fp32", "real_feature_head_fp32"):
        lines += [head, "Group | current AP | current APs | future AP | future AP50 | future APs | future car | future pedestrian | current SmoothL1 | future SmoothL1"]
        for group in GROUPS:
            if group not in report["groups"]:
                continue
            row = report["groups"][group]
            current, future = row["readouts"].get(f"current_{head}"), row["readouts"].get(f"future_{head}")
            if current is None or future is None:
                continue
            values = [current["metrics"].get("AP"), current["metrics"].get("APs"), future["metrics"].get("AP"),
                      future["metrics"].get("AP50"), future["metrics"].get("APs"),
                      future["per_class"]["car"]["AP"], future["per_class"]["pedestrian"]["AP"]]
            errors = row["feature_errors"]
            lines.append(" | ".join([group, *(fmt(v) for v in values),
                f"{errors['current']['smooth_l1']:.6f}", f"{errors['future']['smooth_l1']:.6f}"]))
        lines.append("")
    for comparison, changes in report["comparisons"].items():
        lines.append(comparison)
        for name in READOUTS:
            if name in changes:
                values = changes[name]["metric_delta_points"]
                lines.append(f"  {name}: AP {values['AP']:+.3f}, APs {values['APs']:+.3f}")
        lines.append("")
    lines += [f"Evaluation inventory audit passed: {report['evaluation_audit_passed']}",
              f"Training trace/budget audit passed: {report['training_audit']['passed']}",
              f"Checkpoint audit passed: {report['checkpoint_audit'].get('passed', 'skipped')}", ""]
    if report["missing"]:
        lines.append("Missing artifacts: " + ", ".join(report["missing"]))
    lines += ["Limitations:", *("- " + value for value in report["limitations"])]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", default="outputs/future_supervision_20261108")
    parser.add_argument("--eval", action="append", default=[], metavar="GROUP=PATH")
    parser.add_argument("--no-discover", action="store_true", help="Use only explicit --eval mappings for auxiliary comparisons")
    parser.add_argument("--expected-samples", type=int, default=600)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--skip-checkpoint-audit", action="store_true")
    parser.add_argument("--leave-one-out", action="store_true", help="Rescore primary combined-minus-current-only comparison")
    parser.add_argument("--output-stem", default="summary")
    args = parser.parse_args(argv)
    root = resolve(args.root)
    plan = read(root / "plan.json")
    planned_updates = plan["planned_training"]["updates"]
    if not isinstance(planned_updates, int):
        raise ValueError("Register a concrete shared update budget in plan.json first")
    found = discover_evaluations(root, args.eval, enabled=not args.no_discover)
    rows, eval_audits, missing = load_evaluations(found, args.expected_samples)
    training_audit, absent = audit_training(root, planned_updates)
    missing += absent
    checkpoint_audit = {"status": "skipped"}
    if not args.skip_checkpoint_audit:
        checkpoint_audit, absent = audit_checkpoints(root, plan, planned_updates)
        missing += absent
    eval_pass = len(eval_audits) == len(GROUPS) and all(item["passed"] for item in eval_audits.values())
    passed = not missing and eval_pass and training_audit["passed"] and checkpoint_audit.get("passed", False)
    report = {"status": "complete_audited" if passed else "incomplete_or_failed_audit", "root": str(root),
              "metric_units": "0-1 for absolute detection metrics; percentage points for comparisons",
              "planned_updates": planned_updates, "groups": rows, "comparisons": comparisons(rows),
              "evaluation_audit": eval_audits, "evaluation_audit_passed": eval_pass,
              "training_audit": training_audit, "checkpoint_audit": checkpoint_audit, "missing": missing,
              "limitations": ["Exploratory fixed val600, single seed and short adaptation budget; no full-test claim.",
                  "Detection readouts do not establish dense or full-scene understanding.",
                  "Sparse rider/bus classes can dominate macro AP; inspect car, pedestrian and APs.",
                  "Historical matched-capacity current-head training used FP32 PCA projection; its results are a separate protocol.",
                  "No comparison here measures temporal-order reliance or transfer to another horizon."]}
    if args.leave_one_out:
        report["sequence_sensitivity"] = sequence_sensitivity(found, rows)
    root.mkdir(parents=True, exist_ok=True)
    for suffix, payload in ((".json", json.dumps(report, indent=2, allow_nan=False) + "\n"),
                            (".txt", plain_text(report))):
        destination = root / (args.output_stem + suffix)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(payload)
        temporary.replace(destination)
    print(json.dumps({"status": report["status"], "groups": list(rows), "missing": missing,
                      "output": str(root / (args.output_stem + '.json'))}, indent=2))
    return 0 if passed or args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
