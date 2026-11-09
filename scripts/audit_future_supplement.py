"""Independent CPU checkpoint and run-provenance audit for supplementary runs."""

import argparse
import gc
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_future_experiment import compare_tensors, digest, file_digest, read, resolve, FIXED_EVENT


def checkpoint_audit(root):
    import torch
    from foresight.models.forecaster import internal_state_dict

    torch.set_num_threads(2)
    config = yaml.safe_load((root / "train.yaml").read_text())
    dinof = resolve(config["model"]["dinof_checkpoint"])
    e2_path = resolve(config["model"]["event_encoder_checkpoint"])
    initial = torch.load(dinof, map_location="cpu", weights_only=True, mmap=True)
    forecaster = internal_state_dict(initial["state_dict"])
    event = torch.load(e2_path, map_location="cpu", weights_only=True, mmap=True)["state_dict"]
    result = {"reference_forecaster": str(dinof), "reference_event": str(e2_path), "groups": {}}
    for group, weights in (("current_only_seed7", (1., 0.)), ("current_future_seed7", (1., 1.))):
        path = root / group / "checkpoints/last.ckpt"
        if not path.exists():
            result["groups"][group] = {"status": "checkpoint_not_present", "passed": False}
            continue
        saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        state = saved["state_dict"]
        fc = {k.removeprefix("forecaster."): v for k, v in state.items() if k.startswith("forecaster.")}
        enc = {k.removeprefix("event_encoder."): v for k, v in state.items() if k.startswith("event_encoder.")}
        fc_audit = compare_tensors(fc, forecaster)
        enc_audit = compare_tensors(enc, event)
        fixed_audit = compare_tensors({k: enc[k] for k in FIXED_EVENT}, {k: event[k] for k in FIXED_EVENT})
        config_saved = saved["config"]
        seed42_config = yaml.safe_load((root / group.removesuffix("_seed7") / "config_train.yaml").read_text())
        comparable = json.loads(json.dumps(config_saved))
        comparable.pop("output_dir", None)
        comparable["runtime"].pop("seed", None)
        seed42_config.pop("output_dir", None)
        seed42_config["runtime"].pop("seed", None)
        progress_path = root / group / "training_progress.json"
        progress = read(progress_path) if progress_path.exists() else {}
        grad_path = root / group / "gradient_audit.json"
        gradient = read(grad_path) if grad_path.exists() else {}
        checks = {
            "frozen_forecaster_bitwise_equal": fc_audit["identical"],
            "fixed_event_buffers_bitwise_equal": fixed_audit["identical"],
            "event_structure_matches_e2": not enc_audit["missing"] and not enc_audit["extra"],
            "event_parameters_changed": bool(enc_audit["changed"]),
            "step300": saved.get("global_step") == progress.get("optimizer_steps") == 300,
            "seed7": config_saved["runtime"]["seed"] == 7,
            "same_seed42_recipe_except_seed_output": comparable == seed42_config,
            "loss_weights_match_claim": tuple(config_saved["optimization"].get(k) for k in ("current_weight", "future_weight")) == weights,
            "predictor_frozen": saved.get("input_spec", {}).get("trainable_predictor") is False,
            "checkpoint_example_count": saved.get("future_event_progress", {}).get("examples_seen_per_rank") == progress.get("examples_seen_per_rank") == 600,
            "checkpoint_microbatch_count": saved.get("future_event_progress", {}).get("microbatches_seen_per_rank") == progress.get("microbatches_seen_per_rank") == 600,
            "gradient_boundary_audit": gradient.get("encoder_gradient_norm", 0) > 0 and
                gradient.get("frozen_forecaster_parameters_with_grad") == [] and gradient.get("predictor_eval") is True,
        }
        result["groups"][group] = {"status": "audited", "path": str(path), "checks": checks,
            "passed": all(checks.values()), "frozen_forecaster": fc_audit, "fixed_event_buffers": fixed_audit,
            "event_changed_count": len(enc_audit["changed"]), "event_changed_names": enc_audit["changed"],
            "checkpoint_progress": saved.get("future_event_progress"), "gradient_audit": gradient}
        del saved, state, fc, enc
        gc.collect()
    result["passed"] = all(group["passed"] for group in result["groups"].values())
    result["status"] = "complete_audited" if result["passed"] else "incomplete_or_failed_audit"
    return result


def protocol_audit(root):
    reference = yaml.safe_load((root / "train.yaml").read_text())["model"]
    expected_models = {key: str(resolve(reference[key])) if reference.get(key) else None
                       for key in ("dinof_checkpoint", "detector_checkpoint", "predictor_checkpoint")}
    e2 = resolve(reference["event_encoder_checkpoint"])
    hashes = {}
    def hash_file(path):
        path = resolve(path)
        if path not in hashes:
            hashes[path] = file_digest(path)
        return hashes[path]
    rows = {}
    for folder in sorted(root.iterdir()):
        name = folder.name
        if not folder.is_dir() or not name.startswith(("eval_", "temporal_")):
            continue
        if not (folder / "completed.json").exists() or not (folder / "run.json").exists():
            continue
        done, run = read(folder / "completed.json"), read(folder / "run.json")
        if done.get("status") != "completed":
            continue
        args = run["arguments"]
        if name.startswith("temporal_current_future_"):
            claim = "current_future"
        elif name.startswith("temporal_e2_"):
            claim = "e2"
        else:
            claim = name.removeprefix("eval_").removesuffix("_extra1800")
        expected_encoder = None if claim == "rgb" else (e2 if claim == "e2" else root / claim / "checkpoints/last.ckpt")
        expected_manifest = (root / "control_val120.json" if name.startswith("temporal_") else
                             root / "val1800_extra.json" if name.endswith("_extra1800") else resolve("configs/val600.json"))
        manifest_path = resolve(args["manifest"])
        selection = read(manifest_path)
        split = args.get("split", "val")
        if isinstance(selection, dict) and split in selection:
            selection = selection[split]
        if isinstance(selection, dict):
            selection = selection["sample_ids"]
        if args.get("max_samples") is not None:
            selection = selection[:args["max_samples"]]
        models = {key: str(resolve(run["model"][key])) if run["model"].get(key) else None for key in expected_models}
        encoder_path = resolve(args["encoder_checkpoint"]) if args.get("encoder_checkpoint") else None
        encoder_hash = hash_file(encoder_path) if encoder_path else None
        readout_path = resolve(args["readout_checkpoint"]) if args.get("readout_checkpoint") else None
        checks = {
            "batch_size_one": args.get("batch_size") == 1,
            "model_three_paths_match_reference": models == expected_models,
            "manifest_path_matches_claimed_cohort": manifest_path == expected_manifest.resolve(),
            "manifest_hash_matches_file": run.get("manifest_sha256") == hash_file(manifest_path),
            "manifest_order_matches_run": selection == run["sample_ids"],
            "complete_sample_count": len(selection) == done.get("samples"),
            "encoder_path_matches_claimed_model": encoder_path == (expected_encoder.resolve() if expected_encoder else None),
            "encoder_hash_matches_file": encoder_hash == run.get("encoder_sha256"),
            "modality_matches_claim": run.get("input_modality") == ("rgb" if claim == "rgb" else "event"),
            "real_head_hash_matches_file": bool(readout_path) and hash_file(readout_path) == run.get("readout_sha256"),
            "canonical_precision": run.get("encoder_predictor_precision") == "bf16-mixed" and
                run.get("encoder_predictor_matmul_precision") == "high" and run.get("encoder_chunk_size") == 1 and
                run.get("readout_precision") == "FP32/highest/TF32-off",
        }
        if name.startswith("temporal_"):
            expected_control = name.removeprefix(f"temporal_{claim}_")
            checks["control_matches_folder"] = run.get("control") == args.get("control") == done.get("control") == expected_control
        rows[name] = {"passed": all(checks.values()), "checks": checks, "claimed_encoder": claim,
            "batch_size": args.get("batch_size"), "models": models,
            "manifest": {"path": str(manifest_path), "sha256": hash_file(manifest_path), "samples": len(selection),
                         "sample_ids_sha256_canonical": digest(selection)},
            "encoder": {"path": str(encoder_path) if encoder_path else None, "sha256": encoder_hash},
            "real_head": {"path": str(readout_path) if readout_path else None,
                          "sha256": hash_file(readout_path) if readout_path else None}}
    expected_names = {f"eval_{name}" for name in ("rgb", "e2", "current_only", "future_only", "current_future",
        "current_only_seed7", "current_future_seed7", "current_only_extra1800", "current_future_extra1800")}
    expected_names |= {f"temporal_{model}_{control}" for model in ("e2", "current_future") for control in
                       ("identity", "persist_latent", "repeat_last", "reverse_history", "reverse_bins")}
    missing = sorted(expected_names - rows.keys())
    return {"status": "complete_audited" if not missing and all(row["passed"] for row in rows.values()) else "incomplete_or_failed_audit",
        "coverage": "All completed eval_*/temporal_* directories at audit time; unfinished runs are not treated as failures of their model.",
        "required_incomplete_runs": missing, "completed_run_count": len(rows),
        "all_completed_run_protocols_pass": all(row["passed"] for row in rows.values()),
        "model_asset_hashes": {key: hash_file(path) if path else None for key, path in expected_models.items()},
        "runs": rows,
        "limitations": ["Different trained encoders need different hashes; each hash is checked against its claimed file, not equated across model arms.",
            "Extra1800 is another target subset from the same validation sequences, not an independent dataset.",
            "Asset hashes here describe the files present at audit time; original run metadata did not hash every frozen RGB/head asset."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/future_supervision_20261108")
    parser.add_argument("--skip-checkpoints", action="store_true")
    args = parser.parse_args(argv)
    root = resolve(args.root)
    reports = {"supplemental_protocol_audit": protocol_audit(root)}
    if not args.skip_checkpoints:
        reports["seed7_checkpoint_audit"] = checkpoint_audit(root)
    for name, report in reports.items():
        path = root / f"{name}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
        print(json.dumps({"file": str(path), "status": report["status"]}), flush=True)


if __name__ == "__main__":
    main()
