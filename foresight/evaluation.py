"""COCO evaluation of the single future target in each dataset window."""

import contextlib
import io
from dataclasses import dataclass

import numpy as np

from .data.definitions import CLASSES, HEIGHT, WIDTH, require_int


@dataclass
class Prediction:
    sample_id: str
    target_id: str
    boxes: np.ndarray
    class_ids: np.ndarray
    scores: np.ndarray

    def __post_init__(self):
        if (
            not isinstance(self.sample_id, str)
            or not self.sample_id
            or not isinstance(self.target_id, str)
            or not self.target_id
        ):
            raise ValueError("Prediction IDs must be nonempty strings")
        boxes = np.asarray(self.boxes, dtype=np.float64)
        if boxes.shape == (0,):
            boxes = boxes.reshape(0, 4)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("Prediction boxes must have shape [N,4]")
        self.boxes = boxes.copy()
        raw_classes = np.asarray(self.class_ids)
        scores = np.asarray(self.scores, dtype=np.float64)
        if raw_classes.ndim != 1 or scores.ndim != 1:
            raise ValueError("Prediction class_ids and scores must have shape [N]")
        if not np.issubdtype(raw_classes.dtype, np.integer) and raw_classes.size:
            raise ValueError("Prediction class_ids must be integers")
        self.class_ids = raw_classes.astype(np.int64).copy()
        self.scores = scores.copy()
        if not len(self.boxes) == len(self.class_ids) == len(self.scores):
            raise ValueError(
                "Prediction boxes, class_ids and scores have different lengths"
            )
        if not np.isfinite(self.boxes).all() or not np.isfinite(self.scores).all():
            raise ValueError("Prediction boxes and scores must be finite")
        if np.any(self.boxes[:, 2:] <= self.boxes[:, :2]):
            raise ValueError("Prediction boxes must have positive width and height")
        if np.any(self.scores < 0) or np.any(self.scores > 1):
            raise ValueError("Prediction scores must be in [0, 1]")

    def to_dict(self):
        return {
            "sample_id": self.sample_id,
            "target_id": self.target_id,
            "boxes": self.boxes.tolist(),
            "class_ids": self.class_ids.tolist(),
            "scores": self.scores.tolist(),
        }


@dataclass
class MetricReport:
    metadata: dict
    metrics: dict
    per_class: dict

    def to_dict(self):
        return {
            "metadata": self.metadata,
            "metrics": self.metrics,
            "per_class": self.per_class,
        }


class FutureDetectionEvaluator:
    """Exact frame matching, original-coordinate xyxy predictions, scores in [0,1].

    Empty GT targets and their predictions are ignored, as in DAGr. Nonempty GT
    targets are evaluated even when predictions are missing. Each window has one
    future target. No NMS, confidence threshold or temporal tolerance is applied.
    """

    def __init__(self, dataset, sample_ids=None, max_dets=100):
        self.dataset = dataset
        require_int(max_dets, "max_dets", 10)
        self.max_dets = max_dets
        self._inventory = {}
        selected = None
        if sample_ids is not None:
            ids = list(sample_ids)
            if any(not isinstance(item, str) for item in ids) or len(set(ids)) != len(ids):
                raise ValueError("sample_ids must be unique strings")
            selected = set(ids)
            available = {f"{sequence}:{end}" for sequence, end in dataset.windows}
            if not selected <= available:
                raise ValueError("Selected sample_ids are not in the dataset inventory")
        for sequence, end in dataset.windows:
            sample_id = f"{sequence}:{end}"
            if selected is not None and sample_id not in selected:
                continue
            times = dataset.timestamps[sequence]
            start = end - dataset.history_frames + 1
            frame = end + dataset.prediction_frames
            key = sample_id, f"{sequence}/{frame:06d}"
            if key in self._inventory:
                raise ValueError(f"Duplicate evaluation target: {key}")
            self._inventory[key] = (sequence, frame, float(times[end] - times[start]) / 1000,
                                    float(times[frame] - times[end]) / 1000)
        self.reset()

    def reset(self):
        self._predictions = {}

    def update(self, predictions):
        if isinstance(predictions, (Prediction, dict)):
            predictions = [predictions]
        for value in predictions:
            if not isinstance(value, (Prediction, dict)):
                raise TypeError(
                    "Predictions must be Prediction instances or dictionaries"
                )
            prediction = (
                Prediction(**value)
                if isinstance(value, dict)
                else Prediction(
                    value.sample_id,
                    value.target_id,
                    value.boxes,
                    value.class_ids,
                    value.scores,
                )
            )
            key = prediction.sample_id, prediction.target_id
            if key not in self._inventory:
                raise ValueError(
                    f"Prediction is not in the dataset target inventory: {key}"
                )
            if np.any(prediction.class_ids < 0) or np.any(
                prediction.class_ids >= len(CLASSES)
            ):
                raise ValueError(
                    "Prediction class IDs do not match the configured output classes"
                )
            if key in self._predictions:
                if self._predictions[key].to_dict() != prediction.to_dict():
                    raise ValueError(f"Conflicting duplicate prediction: {key}")
                continue
            self._predictions[key] = prediction

    def state_dict(self):
        return {
            "predictions": [
                self._predictions[key].to_dict() for key in sorted(self._predictions)
            ],
        }

    def merge_state(self, state):
        """Merge workers evaluating the same split and timing configuration."""
        self.update(state["predictions"])

    def to_coco(self):
        """Return COCO GT and predictions only for targets with valid GT boxes."""
        classes = CLASSES
        dataset = {
            "info": {},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {"id": i + 1, "name": name} for i, name in enumerate(classes)
            ],
        }
        results, readers = [], {}
        try:
            for key in sorted(self._inventory):
                sequence, frame, _, _ = self._inventory[key]
                if sequence not in readers:
                    readers[sequence] = self.dataset.reader(sequence)
                target = readers[sequence].read_target(frame)
                if not len(target.boxes):
                    continue
                image_id = len(dataset["images"]) + 1
                dataset["images"].append(
                    {
                        "id": image_id,
                        "width": WIDTH,
                        "height": HEIGHT,
                        "file_name": f"{sequence}/{frame:06d}.png",
                        "sample_id": key[0],
                        "target_id": key[1],
                        "timestamp_us": target.timestamp_us,
                    }
                )
                for box, category in zip(target.boxes, target.class_ids):
                    x, y, x2, y2 = map(float, box)
                    dataset["annotations"].append(
                        {
                            "id": len(dataset["annotations"]) + 1,
                            "image_id": image_id,
                            "category_id": int(category) + 1,
                            "bbox": [x, y, x2 - x, y2 - y],
                            "area": (x2 - x) * (y2 - y),
                            "iscrowd": 0,
                        }
                    )
                prediction = self._predictions.get(key)
                if prediction is not None:
                    for box, category, score in zip(
                        prediction.boxes, prediction.class_ids, prediction.scores
                    ):
                        x, y, x2, y2 = map(float, box)
                        results.append(
                            {
                                "image_id": image_id,
                                "category_id": int(category) + 1,
                                "bbox": [x, y, x2 - x, y2 - y],
                                "score": float(score),
                            }
                        )
        finally:
            for reader in readers.values():
                reader.close()
        return dataset, results

    @staticmethod
    def _mean(values):
        values = values[values >= 0]
        return float(values.mean()) if values.size else None

    def _evaluate_coco(self, dataset, results):
        if not dataset["annotations"]:
            return (
                {
                    name: None
                    for name in (
                        "AP",
                        "AP50",
                        "AP75",
                        "APs",
                        "APm",
                        "APl",
                        "AR1",
                        "AR10",
                        f"AR{self.max_dets}",
                    )
                },
                {name: {"AP": None, "AR": None, "num_gt": 0} for name in CLASSES},
            )
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except ImportError as exc:
            raise ImportError("COCO metrics require pip install pycocotools") from exc
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO()
            gt.dataset = dataset
            gt.createIndex()
            if results:
                detections = gt.loadRes(results)
            else:
                detections = COCO()
                detections.dataset = {
                    "images": dataset["images"],
                    "categories": dataset["categories"],
                    "annotations": [],
                }
                detections.createIndex()
            evaluator = COCOeval(gt, detections, "bbox")
            evaluator.params.maxDets = [1, 10, self.max_dets]
            evaluator.evaluate()
            evaluator.accumulate()
        precision = evaluator.eval["precision"]
        recall = evaluator.eval["recall"]
        ious = evaluator.params.iouThrs
        at_50 = np.flatnonzero(np.isclose(ious, 0.5))
        at_75 = np.flatnonzero(np.isclose(ious, 0.75))
        metrics = {
            "AP": self._mean(precision[:, :, :, 0, -1]),
            "AP50": self._mean(precision[at_50, :, :, 0, -1]),
            "AP75": self._mean(precision[at_75, :, :, 0, -1]),
            "APs": self._mean(precision[:, :, :, 1, -1]),
            "APm": self._mean(precision[:, :, :, 2, -1]),
            "APl": self._mean(precision[:, :, :, 3, -1]),
            "AR1": self._mean(recall[:, :, 0, 0]),
            "AR10": self._mean(recall[:, :, 0, 1]),
            f"AR{self.max_dets}": self._mean(recall[:, :, 0, -1]),
        }
        per_class = {
            name: {
                "AP": self._mean(precision[:, :, i, 0, -1]),
                "AR": self._mean(recall[:, i, 0, -1]),
                "num_gt": sum(
                    a["category_id"] == i + 1 for a in dataset["annotations"]
                ),
            }
            for i, name in enumerate(CLASSES)
        }
        return metrics, per_class

    def compute(self):
        dataset, results = self.to_coco()
        metrics, per_class = self._evaluate_coco(dataset, results)
        keys = [(image["sample_id"], image["target_id"]) for image in dataset["images"]]

        def time_stats(column):
            values = [self._inventory[key][column] for key in keys]
            return {
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "mean": float(np.mean(values)) if values else None,
            }

        metadata = {
            "split": self.dataset.split,
            "sequences": self.dataset.sequences,
            "history_ms": (self.dataset.history_frames - 1) * 50,
            "history_frames": self.dataset.history_frames,
            "window_history_frames": self.dataset.history_frames,
            "model_context_frames": self.dataset.context_frames,
            "model_context_ms": (self.dataset.context_frames - 1) * 50,
            "tolerance_us": self.dataset.tolerance_us,
            "prediction_ms": self.dataset.prediction_ms,
            "max_dets": self.max_dets,
            "coordinate_system": "left_distorted_event",
            "metrics_unit": "0-1",
            "num_samples": len(self._inventory),
            "num_targets": len(keys),
            "num_gt": len(dataset["annotations"]),
            "num_predictions": len(results),
            "num_ignored_empty_gt_targets": len(self._inventory) - len(keys),
            "num_missing_prediction_records": sum(
                key not in self._predictions for key in keys
            ),
            "actual_history_ms": time_stats(2),
            "actual_prediction_ms": time_stats(3),
        }
        return MetricReport(metadata, metrics, per_class)
