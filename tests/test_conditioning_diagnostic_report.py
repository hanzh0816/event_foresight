"""Report checks use only stdlib; they do not run model forward/backward."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from foresight.research.conditioning_report import render_summary


def sample_report():
    correct = {
        "num_samples": 2,
        "detection": {"metrics": {"AP": 0.3727, "AP50": 0.6135, "AP75": 0.4078},
                      "metadata": {}, "per_class": {}},
        "features": {"all_l1": 0.4405, "fg_l1": 0.4809, "bg_l1": 0.4390,
                     "all_smooth_l1": 0.15, "fg_smooth_l1": 0.18, "bg_smooth_l1": 0.14,
                     "foreground_patch_fraction": 0.04},
        "prediction_change_vs_correct": {"all_l1": 0.0, "fg_l1": 0.0, "bg_l1": 0.0},
        "diagnostic_means": {"update_to_hidden_l2_ratio": 0.013, "memory_change_l1": 0.0},
    }
    off = deepcopy(correct)
    off["detection"]["metrics"]["AP"] = 0.3748
    off["prediction_change_vs_correct"]["all_l1"] = 0.012345
    off["diagnostic_means"] = {"update_to_hidden_l2_ratio": 0.0, "memory_change_l1": None}
    return {
        "format": "event-conditioning-diagnostics-v1",
        "evaluation_kind": "conditioning_intervention_diagnostic",
        "split": "val", "sample_ids": ["a:1", "b:2"], "sample_digest": "inventory-digest",
        "checkpoint": "/runs/ordered-joint/last.ckpt", "checkpoint_sha256": "checkpoint-digest",
        "git_commit": "diagnostic-revision", "config": {},
        # Deliberately reverse insertion order to check canonical presentation.
        "variants": {"off": off, "correct": correct},
        "gradient_probe": {"num_samples": 0, "file": "gradients.jsonl", "optimizer_steps": 0},
        "scope": {"full_inventory_size": 128, "evaluated_samples": 2, "is_full_inventory": False},
        "artifacts": {},
    }


class ReportRenderingTests(unittest.TestCase):
    def test_ap_conversion_and_difference_direction(self):
        result = render_summary(sample_report())
        self.assertIn("| correct | 37.270 | 61.350 | 40.780 | +0.000 |", result)
        self.assertIn("| off | 37.480 | 61.350 | 40.780 | +0.210 |", result)
        self.assertLess(result.index("| correct |"), result.index("| off |"))
        self.assertIn("干预减 correct", result)

    def test_negative_ap_difference_is_not_reversed(self):
        report = sample_report()
        report["variants"]["off"]["detection"]["metrics"]["AP"] = 0.3700
        self.assertIn("| off | 37.000 | 61.350 | 40.780 | -0.270 |", render_summary(report))

    def test_direct_prediction_difference_and_gt_loss_are_distinct(self):
        result = render_summary(sample_report())
        self.assertIn("| off | 0.012345 | 0.000000 | 0.000000 |", result)
        self.assertIn("all_smooth_l1", result)
        self.assertIn("0.440500", result)
        self.assertIn("4.000%", result)

    def test_skipped_detection_and_empty_regions_remain_missing(self):
        report = sample_report()
        for variant in report["variants"].values():
            variant["detection"] = None
            variant["features"]["fg_l1"] = None
            variant["prediction_change_vs_correct"]["fg_l1"] = None
        result = render_summary(report)
        self.assertIn("| correct | — | — | — | — |", result)
        self.assertIn("| off | 0.012345 | — | 0.000000 |", result)
        self.assertIn("| memory_change_l1 | 0.000000 | — |", result)

    def test_unknown_future_diagnostic_stat_is_retained(self):
        report = sample_report()
        report["variants"]["correct"]["diagnostic_means"]["future_extra_stat"] = 0.125
        self.assertIn("| future_extra_stat | 0.125000 | — |", render_summary(report))

    def test_small_nonzero_sensitivity_is_not_rounded_to_zero(self):
        report = sample_report()
        report["variants"]["off"]["prediction_change_vs_correct"]["all_l1"] = 1.25e-9
        self.assertIn("| off | 1.250e-09 |", render_summary(report))

    def test_actual_timing_is_distinct_from_inventory_history(self):
        report = sample_report()
        report["protocol_timing"] = {"actual_event_history_ms": {"min": 148, "max": 152, "mean": 150},
                                     "cutoff_to_target_ms": {"min": 42, "max": 50, "mean": 48.5}}
        result = render_summary(report)
        self.assertIn("| actual_event_history_ms | 148.000 | 150.000 | 152.000 |", result)
        self.assertIn("| cutoff_to_target_ms | 42.000 | 48.500 | 50.000 |", result)

    def test_training_loss_is_read_from_archived_config(self):
        report = sample_report()
        report["config"] = {"loss": {"kind": "mse", "scene_weight": 0.5}}
        result = render_summary(report)
        self.assertIn("归档训练 loss：mse", result)
        self.assertIn("scene_weight：0.500000", result)
        self.assertIn("均未乘 scene_weight", result)
        self.assertNotIn("训练目标为 SmoothL1", result)

    def test_scope_and_gradient_evidence_boundaries_are_explicit(self):
        report = sample_report()
        report["gradient_probe"]["num_samples"] = 2
        result = render_summary(report)
        self.assertIn("范围：子集", result)
        self.assertIn("局部梯度 probe：2 个样本", result)
        self.assertIn("不是历史训练梯度", result)
        self.assertIn("非零梯度不证明有效运动编码", result)
        report["scope"]["is_full_inventory"] = True
        self.assertIn("完整选定 manifest", render_summary(report))

    def test_report_is_not_mutated(self):
        report = sample_report()
        original = deepcopy(report)
        render_summary(report)
        self.assertEqual(report, original)

    def test_invalid_reports_are_rejected(self):
        variants = []
        report = sample_report()
        report["format"] = "ordinary-evaluation"
        variants.append(report)
        report = sample_report()
        del report["variants"]["correct"]
        variants.append(report)
        report = sample_report()
        report["variants"]["off"]["num_samples"] = 3
        variants.append(report)
        report = sample_report()
        report["gradient_probe"]["optimizer_steps"] = 1
        variants.append(report)
        report = sample_report()
        report["variants"]["correct"]["detection"]["metrics"]["AP"] = float("nan")
        variants.append(report)
        report = sample_report()
        report["variants"]["correct"]["features"]["all_l1"] = float("inf")
        variants.append(report)
        for report in variants:
            with self.subTest(report=report), self.assertRaises(ValueError):
                render_summary(report)


class ReportCommandTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.environment = {**os.environ, "PYTHONPATH": str(self.root)}

    def run_cli(self, report_path, output_path):
        return subprocess.run(
            [sys.executable, "-m", "foresight.research.conditioning_report", str(report_path),
             "--output", str(output_path)], cwd=self.root, env=self.environment,
            text=True, capture_output=True,
        )

    def test_cli_writes_new_file_and_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "report.json"
            output_path = Path(temporary) / "summary.md"
            report_path.write_text(json.dumps(sample_report()), encoding="utf-8")
            first = self.run_cli(report_path, output_path)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertIn("+0.210", output_path.read_text(encoding="utf-8"))
            output_path.write_text("existing human analysis\n", encoding="utf-8")
            second = self.run_cli(report_path, output_path)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "existing human analysis\n")

    def test_invalid_input_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "report.json"
            output_path = Path(temporary) / "new" / "summary.md"
            report_path.write_text('{"format": "wrong"}', encoding="utf-8")
            result = self.run_cli(report_path, output_path)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output_path.exists())

    def test_import_does_not_load_model_dependencies(self):
        program = (
            "import sys; from foresight.research.conditioning_report import render_summary; "
            "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules; "
            "assert 'lightning' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", program], cwd=self.root,
                                env=self.environment, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
