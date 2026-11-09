"""CPU-only audits of extra seed, extra validation targets, and timing controls.

Missing or unfinished runs are recorded without inventing results. AP is never
averaged across subsets; all deltas compare the same task and target inventory.
"""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_future_experiment import READOUTS, digest, metric_delta, resolve

HEADS = ("old_head_fp32", "real_feature_head_fp32")
CONTROLS = ("identity", "persist_latent", "repeat_last", "reverse_history", "reverse_bins")
PHASES = ("current", "future")


def safe_json(path, missing):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        missing.append(f"{path}: {type(exc).__name__}")
        return None


def expected_targets(sample_ids, phase):
    offset = 0 if phase == "current" else 1
    return [f"{name}/{int(end) + offset:06d}" for name, end in
            (sample.rsplit(":", 1) for sample in sample_ids)]


def load_evaluation(folder, samples, missing, phases=PHASES):
    """Keep compact audits plus paths to predictions for optional exact parity."""
    folder = Path(folder)
    done = safe_json(folder / "completed.json", missing)
    if done is None or done.get("status") != "completed":
        if done is not None:
            missing.append(f"{folder}: not_completed")
        return {"status": "missing_or_incomplete", "path": str(folder), "audit": {"passed": False}}
    run = safe_json(folder / "run.json", missing)
    if run is None:
        return {"status": "missing_or_incomplete", "path": str(folder), "audit": {"passed": False}}
    ids = run.get("sample_ids", [])
    row = {"status": "loaded", "path": str(folder), "sample_ids": ids,
           "feature_errors": done.get("feature_errors", {}), "readouts": {}, "ground_truth": {},
           "protocol": {key: run.get(key) for key in ("readout_precision", "encoder_predictor_precision",
               "encoder_predictor_matmul_precision", "encoder_chunk_size", "encoder_sha256", "readout_sha256")}}
    checks = {"sample_count": len(ids) == samples == done.get("samples"),
              "unique_sample_ids": len(set(ids)) == len(ids),
              "head_precision": run.get("readout_precision") == "FP32/highest/TF32-off"}
    for phase in phases:
        path = folder / f"ground_truth_{phase}.json"
        coco = safe_json(path, missing)
        if coco is not None:
            images, annotations = coco.get("images", []), coco.get("annotations", [])
            keys = [(image["sample_id"], image["target_id"]) for image in images]
            expected = list(zip(ids, expected_targets(ids, phase), strict=True))
            checks[f"{phase}_gt_inventory"] = len(keys) == len(expected) and set(keys) == set(expected)
            nonempty = {ann["image_id"] for ann in annotations}
            row["ground_truth"][phase] = {
                "path": str(path), "sha256_canonical": digest(coco), "images": len(images),
                "annotations": len(annotations), "target_ids": [image["target_id"] for image in images],
                "sequences": sorted({image["sample_id"].rsplit(":", 1)[0] for image in images}),
                "empty_sample_ids": [image["sample_id"] for image in images if image["id"] not in nonempty]}
        else:
            checks[f"{phase}_gt_inventory"] = False
        for head in HEADS:
            name = f"{phase}_{head}"
            metric_path, prediction_path = folder / f"metrics_{name}.json", folder / f"predictions_{name}.json"
            metric, saved = safe_json(metric_path, missing), safe_json(prediction_path, missing)
            if metric is None or saved is None:
                checks[name] = False
                continue
            predictions = saved.get("predictions", [])
            row["readouts"][name] = {key: metric.get(key, {}) for key in ("metrics", "per_class", "metadata")}
            row["readouts"][name]["prediction_path"] = str(prediction_path)
            row["readouts"][name]["predictions_sha256_canonical"] = digest(predictions)
            checks[name] = ([item["sample_id"] for item in predictions] == ids and
                [item["target_id"] for item in predictions] == expected_targets(ids, phase) and
                metric.get("metadata", {}).get("num_missing_prediction_records") == 0)
    row["audit"] = {"checks": checks, "passed": all(checks.values())}
    return row


def compare_rows(left, right, phases=PHASES, same_encoder=False):
    if left.get("status") != "loaded" or right.get("status") != "loaded":
        return {"status": "missing_comparison_rows", "audit_passed": False}
    checks = {"same_ordered_sample_ids": left["sample_ids"] == right["sample_ids"],
              "both_row_audits_pass": left["audit"]["passed"] and right["audit"]["passed"]}
    for phase in phases:
        a, b = left["ground_truth"].get(phase), right["ground_truth"].get(phase)
        checks[f"identical_{phase}_ground_truth"] = bool(a and b and a["sha256_canonical"] == b["sha256_canonical"])
    for key in ("readout_precision", "encoder_predictor_precision", "encoder_predictor_matmul_precision",
                "encoder_chunk_size", "readout_sha256"):
        checks[f"same_{key}"] = left["protocol"].get(key) == right["protocol"].get(key)
    if same_encoder:
        checks["same_encoder_checkpoint"] = left["protocol"].get("encoder_sha256") == right["protocol"].get("encoder_sha256")
    changes = {}
    # Suppress deltas on mismatched inventories/protocols instead of silently
    # comparing AP values from different target sets.
    if all(checks.values()):
        for phase in phases:
            for head in HEADS:
                name = f"{phase}_{head}"
                a, b = left["readouts"].get(name), right["readouts"].get(name)
                if a is None or b is None:
                    continue
                classes = {}
                for label in a["per_class"].keys() | b["per_class"].keys():
                    ap_a, ap_b = a["per_class"].get(label, {}).get("AP"), b["per_class"].get(label, {}).get("AP")
                    classes[label] = None if ap_a is None or ap_b is None else 100 * (ap_a - ap_b)
                changes[name] = {"metric_delta_points": metric_delta(a["metrics"], b["metrics"]),
                                 "per_class_AP_delta_points": classes}
    errors = {phase: {key: left["feature_errors"].get(phase, {}).get(key) - value
                     for key, value in right["feature_errors"].get(phase, {}).items()
                     if isinstance(value, (int, float)) and
                     isinstance(left["feature_errors"].get(phase, {}).get(key), (int, float))}
              for phase in phases} if all(checks.values()) else {}
    return {"status": "compared" if all(checks.values()) else "failed_pair_audit",
            "checks": checks, "audit_passed": all(checks.values()),
            "readouts": changes, "feature_error_delta": errors}


def audit_seed7_training(root, missing):
    import yaml
    groups, traces, configs = {}, {}, {}
    for group, weights in (("current_future_seed7", (1.0, 1.0)), ("current_only_seed7", (1.0, 0.0))):
        folder = root / group
        progress = safe_json(folder / "training_progress.json", missing)
        try:
            trace = [json.loads(line) for line in (folder / "sample_order.jsonl").read_text().splitlines() if line.strip()]
            config = yaml.safe_load((folder / "config_train.yaml").read_text())
        except (OSError, ValueError, yaml.YAMLError) as exc:
            missing.append(f"{folder}: training_trace_or_config: {type(exc).__name__}")
            groups[group] = {"passed": False}
            continue
        if progress is None:
            groups[group] = {"passed": False}
            continue
        traces[group] = trace
        comparable = deepcopy({key: config[key] for key in ("data", "model", "runtime", "optimization")})
        actual_weights = tuple(comparable["optimization"].pop(key, None) for key in ("current_weight", "future_weight"))
        configs[group] = comparable
        counts = Counter(record["global_step"] for record in trace)
        examples = sum(len(record["sample_ids"]) for record in trace)
        checks = {"seed7": config["runtime"]["seed"] == 7,
            "expected_loss_weights": actual_weights == weights,
            "planned_and_actual_300_optimizer_steps": progress.get("optimizer_steps") == config["optimization"].get("max_steps") == 300,
            "single_rank": progress.get("world_size") == 1,
            "exactly_600_training_examples": examples == progress.get("examples_seen_per_rank") == 600,
            "trace_microbatch_count": len(trace) == progress.get("microbatches_seen_per_rank") == 600,
            "exact_accumulation_budget": config["data"].get("batch_size") == 1 and config["optimization"].get("accumulation_steps") == 2,
            "complete_300_accumulation_groups": counts == dict.fromkeys(range(300), 2)}
        groups[group] = {"passed": all(checks.values()), "checks": checks, "progress": progress,
                         "sample_trace_sha256_canonical": digest(trace)}
    names = ("current_future_seed7", "current_only_seed7")
    complete = all(name in traces and name in configs for name in names)
    same_trace = complete and traces[names[0]] == traces[names[1]]
    same_recipe = complete and configs[names[0]] == configs[names[1]]
    return {"groups": groups, "exact_sample_traces_equal": same_trace,
            "same_recipe_except_loss_weights": same_recipe,
            "passed": complete and same_trace and same_recipe and all(group["passed"] for group in groups.values())}


def audit_extra_targets(extra, original):
    if extra.get("status") != "loaded" or original.get("status") != "loaded":
        return {"status": "missing_reference", "passed": False}
    result = {"status": "checked", "by_phase": {}, "cross_phase_frame_reuse": {},
              "scope": "Additional targets from the same validation sequences; overlapping histories and adjacent frames remain possible. This is not an independent test set."}
    for phase in PHASES:
        a, b = extra["ground_truth"].get(phase), original["ground_truth"].get(phase)
        if not a or not b:
            result["by_phase"][phase] = {"passed": False}
            continue
        overlap = sorted(set(a["target_ids"]) & set(b["target_ids"]))
        same_sequences = a["sequences"] == b["sequences"]
        result["by_phase"][phase] = {"target_overlap_count": len(overlap), "target_overlap_first10": overlap[:10],
            "same_sequence_inventory": same_sequences, "extra_targets": len(a["target_ids"]),
            "original_targets": len(b["target_ids"]), "passed": not overlap and same_sequences}
    for left_phase, right_phase in (("current", "future"), ("future", "current")):
        a, b = extra["ground_truth"].get(left_phase), original["ground_truth"].get(right_phase)
        if a and b:
            result["cross_phase_frame_reuse"][f"extra_{left_phase}_vs_original_{right_phase}"] = len(set(a["target_ids"]) & set(b["target_ids"]))
    result["passed"] = all(value["passed"] for value in result["by_phase"].values())
    return result


def identity_parity(identity, full, missing):
    if identity.get("status") != "loaded" or full.get("status") != "loaded":
        return {"status": "missing_reference", "passed": False}
    checks = {"subset_of_main_inventory": set(identity["sample_ids"]) <= set(full["sample_ids"]),
              "same_encoder_checkpoint": identity["protocol"].get("encoder_sha256") == full["protocol"].get("encoder_sha256")}
    details = {}
    for head in HEADS:
        name = f"future_{head}"
        a, b = identity["readouts"].get(name), full["readouts"].get(name)
        if not a or not b:
            checks[name] = False
            continue
        a_records, b_records = safe_json(a["prediction_path"], missing), safe_json(b["prediction_path"], missing)
        if a_records is None or b_records is None:
            checks[name] = False
            continue
        by_id = {record["sample_id"]: record for record in b_records["predictions"]}
        mismatches = [record["sample_id"] for record in a_records["predictions"] if by_id.get(record["sample_id"]) != record]
        checks[name] = not mismatches
        details[name] = {"exact_predictions_equal": not mismatches, "mismatch_count": len(mismatches), "mismatch_first10": mismatches[:10]}
    return {"status": "checked", "checks": checks, "heads": details, "passed": all(checks.values())}


def build_report(root, check_identity=True):
    missing = []
    specifications = {
        "current_future": ("eval_current_future", 600, PHASES),
        "current_only": ("eval_current_only", 600, PHASES),
        "e2": ("eval_e2", 600, PHASES),
        "current_future_seed7": ("eval_current_future_seed7", 600, PHASES),
        "current_only_seed7": ("eval_current_only_seed7", 600, PHASES),
        "current_future_extra1800": ("eval_current_future_extra1800", 1800, PHASES),
        "current_only_extra1800": ("eval_current_only_extra1800", 1800, PHASES),
    }
    specifications.update({f"temporal_{model}_{control}": (f"temporal_{model}_{control}", 120, ("future",))
                           for model in ("e2", "current_future") for control in CONTROLS})
    rows = {name: load_evaluation(root / folder, samples, missing, phases)
            for name, (folder, samples, phases) in specifications.items()}
    seeds = {str(seed): compare_rows(rows[f"current_future{suffix}"], rows[f"current_only{suffix}"])
             for seed, suffix in ((42, ""), (7, "_seed7"))}
    training = audit_seed7_training(root, missing)
    extra = compare_rows(rows["current_future_extra1800"], rows["current_only_extra1800"])
    extra["target_disjointness_from_val600"] = audit_extra_targets(rows["current_future_extra1800"], rows["current_future"])
    temporal = {}
    for model in ("e2", "current_future"):
        identity = rows[f"temporal_{model}_identity"]
        temporal[model] = {control: compare_rows(rows[f"temporal_{model}_{control}"], identity,
            phases=("future",), same_encoder=True) for control in CONTROLS if control != "identity"}
    temporal_inventory = compare_rows(rows["temporal_current_future_identity"], rows["temporal_e2_identity"], phases=("future",))
    temporal_inventory = {key: temporal_inventory[key] for key in ("status", "audit_passed", "checks") if key in temporal_inventory}
    parity = {model: identity_parity(rows[f"temporal_{model}_identity"], rows[model], missing)
              for model in ("e2", "current_future")} if check_identity else {"status": "skipped"}
    passed = (not missing and training["passed"] and all(row["audit"]["passed"] for row in rows.values())
              and all(row["audit_passed"] for row in seeds.values()) and extra["audit_passed"]
              and extra["target_disjointness_from_val600"]["passed"]
              and temporal_inventory["audit_passed"]
              and all(row["audit_passed"] for model in temporal.values() for row in model.values())
              and (not check_identity or all(row["passed"] for row in parity.values())))
    return {"status": "complete_audited" if passed else "incomplete_or_failed_audit", "root": str(root),
        "metric_units": "Absolute AP values 0-1; deltas are percentage points.",
        "rows": rows, "per_seed_joint_minus_current": seeds, "seed7_training_audit": training,
        "extra1800_joint_minus_current": extra, "temporal_minus_same_model_identity": temporal,
        "temporal_cross_model_inventory_audit": temporal_inventory,
        "temporal_identity_vs_full_validation": parity, "missing": sorted(set(missing)),
        "limitations": ["Exploratory validation analyses; no significance or independent-test claim.",
            "Seed42 and seed7 deltas are reported separately; no averaging AP across subsets.",
            "Extra1800 has task-specific disjoint target IDs but uses the same sequences; histories and cross-task RGB frames can overlap.",
            "Temporal perturbations measure inference/OOD sensitivity, not proof of learned useful dynamics.",
            "persist_latent bypasses Predictor; repeat_last repeats the newest state through Predictor.",
            "Detection readouts alone do not establish dense or complete scene representation."]}


def text_report(report):
    lines = ["Event foresight robustness diagnostics", f"Status: {report['status']}",
             "All deltas are percentage points on matching targets. No AP is averaged across subsets.", ""]
    def append_pair(label, pair):
        lines.append(f"{label}: {pair['status']}")
        for name, row in pair.get("readouts", {}).items():
            values = row["metric_delta_points"]
            rendered = [f"{key} {'NA' if values.get(key) is None else format(values[key], '+.3f')}" for key in ("AP", "AP50", "APs")]
            lines.append(f"  {name}: " + ", ".join(rendered))
        lines.append("")
    for seed, comparison in report["per_seed_joint_minus_current"].items():
        append_pair(f"Seed{seed}: joint minus current-only on val600", comparison)
    lines.append(f"Seed7 identical trace, 600 examples/300 updates audit: {report['seed7_training_audit']['passed']}")
    append_pair("Extra1800: joint minus current-only", report["extra1800_joint_minus_current"])
    disjoint = report["extra1800_joint_minus_current"]["target_disjointness_from_val600"]
    lines.append("Extra targets audit: " + json.dumps(disjoint, sort_keys=True))
    for model, controls in report["temporal_minus_same_model_identity"].items():
        for control, comparison in controls.items():
            append_pair(f"{model}: {control} minus identity on val120", comparison)
    lines.append("Identity vs full validation prediction parity: " + json.dumps(report["temporal_identity_vs_full_validation"], sort_keys=True))
    lines.append("Temporal cross-model inventory audit: " + json.dumps(report["temporal_cross_model_inventory_audit"], sort_keys=True))
    lines += ["", "Limitations:", *("- " + text for text in report["limitations"])]
    if report["missing"]:
        lines += ["", "Missing/unfinished artifacts:", *("- " + path for path in report["missing"])]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", default="outputs/future_supervision_20261108")
    parser.add_argument("--output-stem", default="robustness_summary")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--skip-identity-parity", action="store_true")
    args = parser.parse_args(argv)
    root = resolve(args.root)
    report = build_report(root, check_identity=not args.skip_identity_parity)
    root.mkdir(parents=True, exist_ok=True)
    for suffix, payload in ((".json", json.dumps(report, indent=2, allow_nan=False) + "\n"), (".txt", text_report(report))):
        destination = root / (args.output_stem + suffix)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(payload)
        temporary.replace(destination)
    print(json.dumps({"status": report["status"], "missing_count": len(report["missing"]),
        "output": str(root / (args.output_stem + ".json"))}, indent=2))
    return 0 if report["status"] == "complete_audited" or args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
