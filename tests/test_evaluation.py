from types import SimpleNamespace
import numpy as np
import pytest

from foresight.evaluation import FutureDetectionEvaluator, Prediction


class LabelsOnlyReader:
    def __init__(self, targets):
        self.targets = targets
    def read_target(self, index):
        return self.targets[index]
    def close(self):
        pass
    def read_image(self, index):
        raise AssertionError("Scoring must not decode RGB")


@pytest.fixture
def labels_only():
    targets = {40: SimpleNamespace(boxes=np.array([[10, 20, 40, 60]], dtype=np.float32),
                                  class_ids=np.array([2]), timestamp_us=2_000_000),
               41: SimpleNamespace(boxes=np.empty((0, 4), dtype=np.float32),
                                  class_ids=np.empty(0, dtype=np.int64), timestamp_us=2_050_000)}
    return SimpleNamespace(windows=[("seq", 39)], timestamps={"seq": np.arange(45) * 50000},
                           history_frames=40, context_frames=4, prediction_frames=1, tolerance_us=1000,
                           split="val", sequences=("seq",), prediction_ms=50,
                           reader=lambda sequence: LabelsOnlyReader(targets))


def perfect():
    return {"sample_id": "seq:39", "target_id": "seq/000040",
            "boxes": [[10, 20, 40, 60]], "class_ids": [2], "scores": [.9]}


def test_perfect_scores_and_metadata_separate_outer_and_model_history(labels_only):
    evaluator = FutureDetectionEvaluator(labels_only)
    evaluator.update(perfect())
    report = evaluator.compute().to_dict()
    assert report["metrics"]["AP"] == pytest.approx(1)
    assert report["metrics"]["AP50"] == pytest.approx(1)
    assert report["per_class"]["car"]["num_gt"] == 1
    assert report["per_class"]["train"]["AP"] is None
    metadata = report["metadata"]
    assert metadata["window_history_frames"] == metadata["history_frames"] == 40
    assert metadata["model_context_frames"] == 4 and metadata["model_context_ms"] == 150
    assert metadata["history_ms"] == metadata["actual_history_ms"]["mean"] == 1950
    assert metadata["actual_prediction_ms"]["mean"] == 50
    assert metadata["num_missing_prediction_records"] == 0


def test_missing_prediction_is_scored_and_empty_gt_is_ignored(labels_only):
    assert FutureDetectionEvaluator(labels_only).compute().metrics["AP"] == 0
    labels_only.windows.append(("seq", 40))
    evaluator = FutureDetectionEvaluator(labels_only)
    evaluator.update(perfect())
    evaluator.update(dict(perfect(), sample_id="seq:40", target_id="seq/000041", scores=[.99]))
    report = evaluator.compute()
    assert report.metrics["AP"] == pytest.approx(1)
    assert report.metadata["num_samples"] == 2 and report.metadata["num_targets"] == 1
    assert report.metadata["num_ignored_empty_gt_targets"] == 1
    assert report.metadata["num_predictions"] == 1
    labels_only.windows = []
    assert FutureDetectionEvaluator(labels_only).compute().metrics["AP"] is None


def test_high_scoring_false_positive_lowers_ap(labels_only):
    evaluator = FutureDetectionEvaluator(labels_only)
    evaluator.update(dict(perfect(), boxes=[[100, 100, 150, 150], [10, 20, 40, 60]],
                          class_ids=[2, 2], scores=[.99, .9]))
    assert 0 < evaluator.compute().metrics["AP"] < 1


def test_state_merge_duplicates_copy_and_target_identity(labels_only):
    evaluator = FutureDetectionEvaluator(labels_only)
    prediction = perfect()
    evaluator.update(prediction)
    prediction["boxes"][0][0] = 0
    state = evaluator.state_dict()
    other = FutureDetectionEvaluator(labels_only)
    other.merge_state(state)
    other.merge_state(state)
    assert len(other.state_dict()["predictions"]) == 1
    assert other.compute().metrics["AP"] == pytest.approx(1)
    with pytest.raises(ValueError, match="Conflicting"):
        other.update(dict(perfect(), scores=[.5]))
    with pytest.raises(ValueError, match="inventory"):
        other.update(dict(perfect(), target_id="seq/000041"))
    with pytest.raises(ValueError, match="class IDs"):
        FutureDetectionEvaluator(labels_only).update(dict(perfect(), class_ids=[8]))


def test_selection_and_max_dets_validation(labels_only):
    for selected in (["missing"], ["seq:39", "seq:39"]):
        with pytest.raises(ValueError):
            FutureDetectionEvaluator(labels_only, selected)
    for invalid in (0, 9, True):
        with pytest.raises(ValueError):
            FutureDetectionEvaluator(labels_only, max_dets=invalid)
    result = FutureDetectionEvaluator(labels_only, sample_ids=[], max_dets=20).compute()
    assert result.metrics["AR20"] is None and result.metadata["num_samples"] == 0


@pytest.mark.parametrize("change", [{"boxes": [[1, 2, 0, 3]]}, {"class_ids": [.2]},
                                    {"scores": [float('nan')]}, {"scores": [2]},
                                    {"boxes": [1, 2, 3, 4]}, {"class_ids": [[0]]}])
def test_invalid_predictions_are_rejected(change):
    with pytest.raises(ValueError):
        Prediction(**dict(perfect(), **change))
