"""Decompose saved COCO small-object AP into supported class contributions.

Uses only saved ground truth/predictions and CPU pycocotools. COCO APs is a
macro average over classes with valid small GT, not an average over objects.
"""

import argparse
from collections import Counter
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def resolve(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT / path).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def valid_mean(values):
    values = values[values >= 0]
    return float(values.mean()) if values.size else None


def evaluate_small(coco, predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    lookup = {(image["sample_id"], image["target_id"]): image["id"] for image in coco["images"]}
    results, seen = [], set()
    for record in predictions:
        key = (record["sample_id"], record["target_id"])
        if key not in lookup or key in seen:
            raise ValueError(f"Unexpected or duplicate prediction record: {key}")
        seen.add(key)
        for box, category, score in zip(record["boxes"], record["class_ids"], record["scores"], strict=True):
            x, y, x2, y2 = map(float, box)
            results.append({"image_id": lookup[key], "category_id": int(category) + 1,
                            "bbox": [x, y, x2-x, y2-y], "score": float(score)})
    if set(lookup) != seen:
        raise ValueError("Prediction records do not cover the complete saved GT image inventory")
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = coco
        gt.createIndex()
        if results:
            detected = gt.loadRes(results)
        else:
            detected = COCO()
            detected.dataset = {"images": coco["images"], "categories": coco["categories"], "annotations": []}
            detected.createIndex()
        evaluator = COCOeval(gt, detected, "bbox")
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
    small_index = evaluator.params.areaRngLbl.index("small")
    low, high = evaluator.params.areaRng[small_index]
    precision, recall = evaluator.eval["precision"], evaluator.eval["recall"]
    at50 = np.flatnonzero(np.isclose(evaluator.params.iouThrs, .5))
    at75 = np.flatnonzero(np.isclose(evaluator.params.iouThrs, .75))
    category_names = {category["id"]: category["name"] for category in coco["categories"]}
    image_sequences = {image["id"]: image["sample_id"].rsplit(":", 1)[0] for image in coco["images"]}
    per_class = {}
    for index, category_id in enumerate(evaluator.params.catIds):
        name = category_names[category_id]
        all_gt = [annotation for annotation in coco["annotations"] if annotation["category_id"] == category_id]
        # COCO's area ignore rule uses area < low OR area > high; endpoints count.
        small_gt = [a for a in all_gt if low <= a["area"] <= high and not a.get("iscrowd", 0)]
        values = precision[:, :, index, small_index, -1]
        per_class[name] = {
            "APs": valid_mean(values),
            "APs50": valid_mean(precision[at50, :, index, small_index, -1]),
            "APs75": valid_mean(precision[at75, :, index, small_index, -1]),
            "ARs100": valid_mean(recall[:, index, small_index, -1]),
            "all_gt_count": len(all_gt), "small_gt_count": len(small_gt),
            "images_with_small_gt": len({a["image_id"] for a in small_gt}),
            "small_gt_by_sequence": dict(sorted(Counter(image_sequences[a["image_id"]] for a in small_gt).items())),
            "included_in_APs_macro": bool((values >= 0).any()),
        }
    supported = [name for name, row in per_class.items() if row["included_in_APs_macro"]]
    aps = valid_mean(precision[:, :, :, small_index, -1])
    reconstructed = sum(per_class[name]["APs"] for name in supported) / len(supported) if supported else None
    if aps is not None and abs(aps - reconstructed) > 1e-12:
        raise RuntimeError("Per-class APs does not reconstruct the reported COCO macro APs")
    return {"APs": aps, "APs50": valid_mean(precision[at50, :, :, small_index, -1]),
            "APs75": valid_mean(precision[at75, :, :, small_index, -1]),
            "small_area_inclusive": [float(low), float(high)], "max_dets": 100,
            "supported_classes": supported, "macro_class_count": len(supported),
            "num_images": len(coco["images"]), "small_gt_count": sum(r["small_gt_count"] for r in per_class.values()),
            "per_class": per_class, "per_class_macro_reconstruction_error": None if aps is None else aps-reconstructed}


def compare(candidate, baseline):
    supported = baseline["supported_classes"]
    if candidate["supported_classes"] != supported:
        raise ValueError("Class support changed; an equal-support macro decomposition is invalid")
    count = len(supported)
    classes = {}
    for name in baseline["per_class"]:
        old, new = baseline["per_class"][name], candidate["per_class"][name]
        if old["small_gt_count"] != new["small_gt_count"]:
            raise ValueError("Small GT counts differ across runs")
        delta = None if name not in supported else 100 * (new["APs"] - old["APs"])
        classes[name] = {"small_gt_count": old["small_gt_count"], "all_gt_count": old["all_gt_count"],
                         "baseline_APs_percent": None if old["APs"] is None else 100*old["APs"],
                         "candidate_APs_percent": None if new["APs"] is None else 100*new["APs"],
                         "APs_delta_points": delta, "macro_APs_contribution_points": None if delta is None else delta/count}
    total = 100 * (candidate["APs"] - baseline["APs"]) if count else None
    recovered = sum(classes[name]["macro_APs_contribution_points"] for name in supported)
    if total is not None and abs(total-recovered) > 1e-10:
        raise RuntimeError("Class contributions do not sum to the global APs change")
    main = [name for name in ("car", "pedestrian") if name in supported]
    other = [name for name in supported if name not in main]
    rare = [name for name in supported if baseline["per_class"][name]["small_gt_count"] < 40]
    def group(names):
        return {"classes": names, "small_gt_count": sum(classes[n]["small_gt_count"] for n in names),
                "mean_class_APs_delta_points": sum(classes[n]["APs_delta_points"] for n in names)/len(names) if names else None,
                "macro_APs_contribution_points": sum(classes[n]["macro_APs_contribution_points"] for n in names)}
    return {"APs_delta_points": total, "macro_class_count": count, "per_class": classes,
            "car_pedestrian": group(main), "other_supported_classes": group(other),
            "small_gt_lt40_diagnostic": group(rare), "contribution_reconstruction_error": None if total is None else total-recovered,
            "grouping_note": "car_pedestrian and other_supported_classes partition the supported classes. small_gt_lt40 is a separate overlapping descriptive grouping, not a causal classification."}


def format_text(report):
    def fmt(value, signed=False):
        return "NA" if value is None else format(value, "+.3f" if signed else ".3f")
    lines = ["COCO small-object AP decomposition", "", f"Phase: {report['phase']}; baseline: {report['baseline']}",
             "APs is an equal-class macro average over supported classes. GT counts do not weight this average.",
             "Small GT follows COCO's inclusive area interval [0, 1024] pixels^2; maxDets=100.", ""]
    for key, result in report["comparisons"].items():
        lines += [key, "Class | small GT | baseline APs | candidate APs | change | contribution to macro change"]
        for name, row in result["per_class"].items():
            lines.append(" | ".join([name, str(row["small_gt_count"]), fmt(row["baseline_APs_percent"]),
                fmt(row["candidate_APs_percent"]), fmt(row["APs_delta_points"], True),
                fmt(row["macro_APs_contribution_points"], True)]))
        lines += [f"Total APs change: {fmt(result['APs_delta_points'], True)} points across {result['macro_class_count']} supported classes.",
                  f"Car+pedestrian contribution: {fmt(result['car_pedestrian']['macro_APs_contribution_points'], True)} points; other classes: {fmt(result['other_supported_classes']['macro_APs_contribution_points'], True)} points.",
                  f"Car+pedestrian mean per-class APs change: {fmt(result['car_pedestrian']['mean_class_APs_delta_points'], True)} points.", ""]
    lines += ["Interpretation limits:",
              "- This is an exact arithmetic decomposition of APs, not a causal explanation of model behavior.",
              "- Very small class supports make the macro metric sensitive to individual objects.",
              "- Improvements in car/pedestrian APs do not imply improvement for every small object or another validation cohort.",
              "- Repeated val usage and a short single-seed run limit generalization claims."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", default="outputs/future_supervision_20261108")
    parser.add_argument("--run", action="append", default=[], metavar="LABEL=DIRECTORY")
    parser.add_argument("--baseline", default="e2_baseline")
    parser.add_argument("--phase", choices=("current", "future"), default="future")
    parser.add_argument("--head", action="append", choices=("old_head_fp32", "real_feature_head_fp32"))
    parser.add_argument("--output-stem", default="small_object_analysis")
    args = parser.parse_args(argv)
    root = resolve(args.root)
    heads = args.head or ["old_head_fp32", "real_feature_head_fp32"]
    if args.run:
        runs = {name: resolve(path) for name, path in (value.split("=", 1) for value in args.run)}
    else:
        runs = {name: root / directory for name, directory in
                (("e2_baseline", "eval_e2"), ("current_only", "eval_current_only"),
                 ("future_only", "eval_future_only"), ("current_future", "eval_current_future"))
                if (root / directory / "completed.json").exists()}
    if args.baseline not in runs:
        raise ValueError("Baseline run is absent; use --run LABEL=DIRECTORY and --baseline LABEL")
    reference = read(runs[args.baseline] / f"ground_truth_{args.phase}.json")
    reference_hash = canonical_hash(reference)
    report = {"phase": args.phase, "baseline": args.baseline, "ground_truth_sha256": reference_hash,
              "units": "0-1 in run metrics, percentages and percentage points in comparisons", "runs": {}, "comparisons": {}}
    for label, folder in runs.items():
        coco_path = folder / f"ground_truth_{args.phase}.json"
        coco = read(coco_path)
        if canonical_hash(coco) != reference_hash:
            raise ValueError(f"GT inventory/content differs for {label}; separate cohorts must be analyzed separately")
        report["runs"][label] = {"path": str(folder), "heads": {}}
        for head in heads:
            name = f"{args.phase}_{head}"
            predictions = read(folder / f"predictions_{name}.json")["predictions"]
            metrics = evaluate_small(coco, predictions)
            saved = read(folder / f"metrics_{name}.json")["metrics"]["APs"]
            if metrics["APs"] is not None and abs(metrics["APs"]-saved) > 1e-12:
                raise RuntimeError(f"APs rescore differs from saved report for {label}/{head}")
            metrics["saved_APs_match"] = metrics["APs"] == saved or abs(metrics["APs"]-saved) < 1e-12
            report["runs"][label]["heads"][head] = metrics
            print(json.dumps({"run": label, "head": head, "APs": metrics["APs"], "supported": metrics["supported_classes"]}), flush=True)
    for label in runs:
        if label == args.baseline:
            continue
        for head in heads:
            report["comparisons"][f"{label}_minus_{args.baseline}/{head}"] = compare(
                report["runs"][label]["heads"][head], report["runs"][args.baseline]["heads"][head])
    root.mkdir(parents=True, exist_ok=True)
    for suffix, value in ((".json", json.dumps(report, indent=2, allow_nan=False)+"\n"),
                          (".txt", format_text(report))):
        path = root / (args.output_stem + suffix)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(value)
        temporary.replace(path)
    return report


if __name__ == "__main__":
    main()
