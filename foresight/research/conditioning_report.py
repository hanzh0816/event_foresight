"""Render fixed-checkpoint Event diagnostics without importing torch or numpy.

This module displays observations and paired differences.  It deliberately does
not classify the cause of a failed intervention from arbitrary norm thresholds.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from foresight.research.diagnostic_protocol import validate_variants


FORMAT = "event-conditioning-diagnostics-v1"
EVALUATION_KIND = "conditioning_intervention_diagnostic"


def _mapping(value: Any, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _number(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number or null")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number or null")
    return float(value)


def _format(value: float | None, *, signed: bool = False, scale: float = 1.0,
            places: int = 6) -> str:
    if value is None:
        return "—"
    scaled = value * scale
    # A small but nonzero sensitivity must not appear as an exact zero.
    if scaled != 0 and abs(scaled) < 10 ** -places:
        return format(scaled, f"{'+' if signed else ''}.3e")
    return format(scaled, f"{'+' if signed else ''}.{places}f")


def _cell(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(_cell(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows),
        "",
    ]


def render_summary(report: Mapping[str, Any]) -> str:
    """Return a Chinese Markdown summary of one v1 diagnostic report.

    AP is supplied on the 0--1 scale; displayed AP uses percent and its paired
    difference uses percentage points.  All deltas are variant minus correct.
    Missing optional metrics remain missing, never converted to zeros.
    """
    report = _mapping(report, "report")
    if report.get("format") != FORMAT or report.get("evaluation_kind") != EVALUATION_KIND:
        raise ValueError("Expected an event-conditioning-diagnostics-v1 intervention report")
    variants = _mapping(report.get("variants"), "variants")
    names = validate_variants(list(variants))
    metrics: dict[str, dict[str, float | None]] = {}
    features, changes, diagnostics = {}, {}, {}
    counts = {}
    for name in names:
        variant = _mapping(variants[name], f"variants.{name}")
        count = variant.get("num_samples")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"variants.{name}.num_samples must be a positive integer")
        counts[name] = count
        detection = variant.get("detection")
        detection_metrics = {} if detection is None else _mapping(
            _mapping(detection, f"variants.{name}.detection").get("metrics"),
            f"variants.{name}.detection.metrics",
        )
        metrics[name] = {key: _number(detection_metrics.get(key), f"{name}.{key}")
                         for key in ("AP", "AP50", "AP75")}
        for target, field in ((features, "features"),
                              (changes, "prediction_change_vs_correct"),
                              (diagnostics, "diagnostic_means")):
            values = _mapping(variant.get(field, {}), f"variants.{name}.{field}")
            target[name] = {key: _number(value, f"{name}.{field}.{key}")
                            for key, value in values.items()}
    if len(set(counts.values())) != 1:
        raise ValueError("All variants must have the same paired sample count")

    scope = _mapping(report.get("scope", {}), "scope")
    is_full = scope.get("is_full_inventory")
    full_label = "完整选定 manifest" if is_full is True else "子集" if is_full is False else "未记录范围"
    lines = [
        "# Event 条件分支诊断摘要", "",
        f"数据划分：{_cell(report.get('split', '未记录'))}；范围：{full_label}；"
        f"评价样本：{counts['correct']}；完整 inventory：{_cell(scope.get('full_inventory_size', '未记录'))}。",
        "“完整”仅相对选定 manifest；子集 AP 不代表完整 val/test。所有干预固定 checkpoint、RGB 和目标。", "",
        f"- Checkpoint：{_cell(report.get('checkpoint', '未记录'))}",
        f"- Checkpoint SHA-256：{_cell(report.get('checkpoint_sha256', '未记录'))}",
        f"- 诊断代码 revision：{_cell(report.get('git_commit', '未记录'))}",
        f"- 样本 digest：{_cell(report.get('sample_digest', '未记录'))}", "",
    ]
    if "protocol_timing" in report:
        timing = _mapping(report["protocol_timing"], "protocol_timing")
        timing_rows = []
        for key in ("actual_event_history_ms", "rgb_observation_support_ms", "cutoff_to_target_ms"):
            values = _mapping(timing.get(key, {}), f"protocol_timing.{key}")
            timing_rows.append([key, *(_format(_number(values.get(stat), f"{key}.{stat}"), places=3)
                                      for stat in ("min", "mean", "max"))])
        lines += ["## 实际输入与目标时间", "",
                  "单位为 ms；实际 Event/RGB 输入支持范围与 inventory 候选窗口跨度分开记录。", ""]
        lines += _table(["范围", "min", "mean", "max"], timing_rows)
    lines += ["## 检测与特征误差", "",
              "AP 输入范围为 0–1，下表换算为百分数；ΔAP 为干预减 correct，单位为百分点。缺失值显示为 —。", ""]
    baseline_ap = metrics["correct"]["AP"]
    rows = []
    for name in names:
        ap = metrics[name]["AP"]
        delta = None if ap is None or baseline_ap is None else ap - baseline_ap
        rows.append([name, *(_format(metrics[name][key], scale=100, places=3)
                              for key in ("AP", "AP50", "AP75")),
                     _format(delta, signed=True, scale=100, places=3)])
    lines += _table(["干预", "AP (%)", "AP50 (%)", "AP75 (%)", "ΔAP (百分点)"], rows)
    config = _mapping(report.get("config", {}), "config")
    loss = _mapping(config.get("loss", {}), "config.loss")
    scene_weight = _number(loss.get("scene_weight"), "config.loss.scene_weight")
    lines += [f"归档训练 loss：{_cell(loss.get('kind', '未记录'))}；scene_weight：{_format(scene_weight)}。"
              "下表同时列出 L1 与 SmoothL1（beta=1）评价值，均未乘 scene_weight。"
              "前景/背景值缺失时不补零。", ""]
    feature_keys = ("all_l1", "fg_l1", "bg_l1", "all_smooth_l1", "fg_smooth_l1", "bg_smooth_l1")
    lines += _table(["干预", *feature_keys], [
        [name, *(_format(features[name].get(key)) for key in feature_keys)] for name in names
    ])
    fraction = features["correct"].get("foreground_patch_fraction")
    lines += [f"correct 的前景 patch 占比：{_format(fraction, scale=100, places=3)}{'%' if fraction is not None else ''}。", ""]

    lines += ["## 相对 correct 的直接预测差异", "",
              "下表比较两次预测本身，不是两次 GT 误差之差。平均 GT L1 接近并不意味着预测相同。", ""]
    lines += _table(["干预", "全图预测 L1 差", "前景预测 L1 差", "背景预测 L1 差"], [
        [name, *(_format(changes[name].get(key)) for key in ("all_l1", "fg_l1", "bg_l1"))]
        for name in names
    ])

    lines += ["## 分支统计", "",
              "各字段为报告中的样本统计均值；change_l1 相对 correct。比例和范数只描述当前路径，不自动判定运动信息是否有用。", ""]
    diagnostic_keys = sorted({key for values in diagnostics.values() for key in values})
    if diagnostic_keys:
        lines += _table(["指标", *names], [
            [key, *(_format(diagnostics[name].get(key)) for name in names)] for key in diagnostic_keys
        ])
    else:
        lines += ["报告未包含分支统计。", ""]

    lines += ["## 梯度与解释边界", ""]
    probe = _mapping(report.get("gradient_probe", {}), "gradient_probe")
    probe_count = probe.get("num_samples", 0)
    if isinstance(probe_count, bool) or not isinstance(probe_count, int) or probe_count < 0:
        raise ValueError("gradient_probe.num_samples must be a nonnegative integer")
    if probe_count:
        lines += [f"局部梯度 probe：{probe_count} 个样本；记录：{_cell(probe.get('file', '未记录'))}；"
                  f"optimizer steps：{_cell(probe.get('optimizer_steps', '未记录'))}。", ""]
    else:
        lines += ["未执行梯度 probe。", ""]
    if probe.get("optimizer_steps", 0) != 0:
        raise ValueError("A fixed-checkpoint diagnostic must not perform optimizer steps")
    lines += [
        "- probe 是当前 checkpoint、当前样本、eval 模式下原 scene loss 的局部梯度，不是历史训练梯度或参数更新记录；非零梯度不证明有效运动编码。",
        "- mismatched 的变化说明内容敏感性；正确 Event 优于错配 Event 才支持当前协议下的收益。零输入可能偏离训练分布。",
        "- query_only 保留直接 RGB query 路径；off 关闭整条分支。二者都是干预，不是独立训练的 RGB baseline。",
        "- 输出变化小不能单独区分 encoder 未提取信息与融合未使用信息；输出变化大也不等于改善任务。",
        "- 使用 val 选择后续方案，最终 test 留到方案确定后；本报告不以固定数值阈值自动判定根因或统计显著性。", "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Diagnostic report.json")
    parser.add_argument("--output", required=True, type=Path, help="New Markdown path; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        with args.report.open(encoding="utf-8") as stream:
            report = json.load(stream)
        summary = render_summary(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(summary)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"conditioning_report: {error}\n")


if __name__ == "__main__":
    main()
