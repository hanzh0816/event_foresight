from copy import deepcopy
import json

import pytest
import yaml

from scripts.summarize_future_robustness import (audit_extra_targets, audit_seed7_training,
    build_report, compare_rows, text_report)


def row(ap=.2):
    return {"status": "loaded", "sample_ids": ["seq:39"], "audit": {"passed": True},
        "protocol": {"readout_precision": "FP32/highest/TF32-off", "encoder_sha256": "encoder"},
        "ground_truth": {phase: {"sha256_canonical": phase, "target_ids": [f"seq/{frame:06d}"], "sequences": ["seq"]}
                         for phase, frame in (("current", 39), ("future", 40))},
        "feature_errors": {phase: {"smooth_l1": 1.0} for phase in ("current", "future")},
        "readouts": {f"{phase}_{head}": {"metrics": {"AP": ap, "APs": ap/2, "AP50": ap+.1},
            "per_class": {"car": {"AP": ap}, "empty_class": {"AP": None}}}
            for phase in ("current", "future") for head in ("old_head_fp32", "real_feature_head_fp32")}}


def test_deltas_are_percentage_points_and_bad_inventory_suppresses_results():
    joint, current = row(.23), row(.2)
    result = compare_rows(joint, current)
    assert result["audit_passed"]
    assert result["readouts"]["future_old_head_fp32"]["metric_delta_points"]["AP"] == pytest.approx(3)
    joint["ground_truth"]["future"]["sha256_canonical"] = "different"
    rejected = compare_rows(joint, current)
    assert not rejected["audit_passed"] and not rejected["readouts"]


def test_disjointness_distinguishes_same_task_targets_from_cross_task_frame_reuse():
    original, extra = row(), row()
    extra["ground_truth"]["current"]["target_ids"] = ["seq/000040"]
    extra["ground_truth"]["future"]["target_ids"] = ["seq/000041"]
    result = audit_extra_targets(extra, original)
    assert result["passed"]
    assert result["cross_phase_frame_reuse"]["extra_current_vs_original_future"] == 1
    extra["ground_truth"]["future"]["target_ids"] = ["seq/000040"]
    assert not audit_extra_targets(extra, original)["passed"]


def test_seed7_trace_audit_detects_same_budget_with_different_sample_order(tmp_path):
    progress = dict(optimizer_steps=300, microbatches_seen_per_rank=600, examples_seen_per_rank=600, world_size=1)
    trace = [dict(global_step=index//2, batch_idx=index, sample_ids=[f"seq:{index}"]) for index in range(600)]
    for group, future_weight in (("current_future_seed7", 1.), ("current_only_seed7", 0.)):
        directory = tmp_path / group
        directory.mkdir()
        (directory / "training_progress.json").write_text(json.dumps(progress))
        (directory / "sample_order.jsonl").write_text("\n".join(json.dumps(item) for item in trace))
        config = dict(data=dict(batch_size=1), model={}, runtime=dict(seed=7),
            optimization=dict(max_steps=300, accumulation_steps=2, current_weight=1., future_weight=future_weight))
        (directory / "config_train.yaml").write_text(yaml.safe_dump(config))
    missing = []
    assert audit_seed7_training(tmp_path, missing)["passed"] and not missing
    changed = deepcopy(trace)
    changed[1]["sample_ids"] = ["different:1"]
    (tmp_path / "current_only_seed7" / "sample_order.jsonl").write_text("\n".join(json.dumps(item) for item in changed))
    rejected = audit_seed7_training(tmp_path, [])
    assert all(value["passed"] for value in rejected["groups"].values())
    assert not rejected["exact_sample_traces_equal"] and not rejected["passed"]


def test_missing_artifacts_produce_serializable_report_instead_of_claimed_results(tmp_path):
    report = build_report(tmp_path)
    assert report["status"] == "incomplete_or_failed_audit"
    assert report["missing"]
    assert report["per_seed_joint_minus_current"]["7"]["status"] == "missing_comparison_rows"
    assert "No AP is averaged across subsets" in text_report(report)
    json.dumps(report, allow_nan=False)
