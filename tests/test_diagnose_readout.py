from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from foresight.diagnose_readout import (CurrentTimeline, CurrentDetectionEvaluator,
    feature_errors, make_diagnostic_dataset, prediction_records, retained_coordinates)
from foresight.evaluation import FutureDetectionEvaluator


def test_current_and_future_labels_are_independent_and_empty_current_is_kept():
    targets = {
        39: SimpleNamespace(boxes=np.array([[1., 2., 11., 12.]]), class_ids=np.array([2]), timestamp_us=1950000),
        40: SimpleNamespace(boxes=np.empty((0, 4)), class_ids=np.empty(0, dtype=np.int64), timestamp_us=2000000),
        41: SimpleNamespace(boxes=np.array([[101., 102., 111., 112.]]), class_ids=np.array([2]), timestamp_us=2050000),
    }
    class Reader:
        def read_target(self, index):
            return targets[index]
        def close(self):
            pass
    dataset = SimpleNamespace(windows=[("seq", 39), ("seq", 40)],
        timestamps={"seq": np.arange(45) * 50000}, history_frames=40,
        context_frames=4, prediction_frames=1, prediction_ms=50, tolerance_us=1000,
        split="val", sequences=("seq",), reader=lambda sequence: Reader())
    current = CurrentDetectionEvaluator(CurrentTimeline(dataset))
    future = FutureDetectionEvaluator(dataset)
    current.update(dict(sample_id="seq:39", target_id="seq/000039",
                        boxes=[[1, 2, 11, 12]], class_ids=[2], scores=[.9]))
    current.update(dict(sample_id="seq:40", target_id="seq/000040",
                        boxes=[[1, 2, 11, 12]], class_ids=[2], scores=[.99]))
    coco, result = current.to_coco()
    assert [im["target_id"] for im in coco["images"]] == ["seq/000039", "seq/000040"]
    assert len(coco["annotations"]) == 1 and len(result) == 2
    report = current.compute()
    assert 0 < report.metrics["AP"] < 1  # False positive on empty current frame counts.
    assert report.metadata["num_targets"] == 2
    assert report.metadata["actual_prediction_ms"]["mean"] == 0
    assert dataset.prediction_frames == 1  # Read-only view did not alter forecast inventory.
    assert [im["target_id"] for im in future.to_coco()[0]["images"]] == ["seq/000041"]
    with pytest.raises(ValueError, match="inventory"):
        current.update(dict(sample_id="seq:39", target_id="seq/000040", boxes=[], class_ids=[], scores=[]))


def test_feature_errors_are_per_sample_and_rgb_identity_is_zero():
    target = torch.tensor([[[1., 2.], [3., 4.]], [[2., 4.], [6., 8.]]])
    identity = feature_errors(target, target)
    assert torch.equal(identity["smooth_l1"], torch.zeros(2))
    assert torch.allclose(identity["cosine"], torch.ones(2))
    perturbed = target.clone()
    perturbed[1] += 1
    errors = feature_errors(perturbed, target)
    assert torch.equal(errors["l1"], torch.tensor([0., 1.]))
    assert torch.equal(errors["smooth_l1"], torch.tensor([0., .5]))
    with pytest.raises(ValueError):
        feature_errors(target[0], target[0])


def test_retained_head_coordinates_do_not_add_rgb_mean_or_scale():
    latent = torch.tensor([[[1., 2.]]])
    pca = SimpleNamespace(pca_components=torch.tensor([[1., 2., 3.], [4., 5., 6.]]))
    assert torch.equal(retained_coordinates(latent, pca), torch.tensor([[[9., 12., 15.]]]))


def test_prediction_records_use_end_only_for_current():
    predictions = [dict(boxes=torch.tensor([[1., 2., 4., 6.]]), labels=torch.tensor([2]), scores=torch.tensor([.9]))]
    meta = [dict(sample_id="sequence:39", target_id="sequence/000040")]
    assert prediction_records(predictions, meta, current=True)[0]["target_id"] == "sequence/000039"
    assert prediction_records(predictions, meta)[0]["target_id"] == "sequence/000040"


class InventoryDataset:
    """A small inventory exercising selection without opening DSEC images."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        available = [("sequence", frame) for frame in (39, 40, 41)]
        lookup = {f"{name}:{end}": (name, end) for name, end in available}
        ids = kwargs.get("sample_ids")
        self.windows = available if ids is None else [lookup[item] for item in ids]

    def __len__(self):
        return len(self.windows)

    def close(self):
        self.closed = True


def diagnostic_config():
    return {"data": dict(root="/portable/DSEC", image_size=[448, 896], sample_history_frames=40,
                         context_frames=4, tolerance_us=1000,
                         val_sample_ids=["sequence:40"], val_sequences=["other"], val_stride=20)}


def test_full_split_ignores_training_subsets_and_does_not_read_manifest():
    args = SimpleNamespace(full_split=True, manifest="does-not-exist.json", split="val", max_samples=None)
    dataset, ids, manifest = make_diagnostic_dataset(InventoryDataset, diagnostic_config(), args)
    assert ids == ["sequence:39", "sequence:40", "sequence:41"]
    assert dataset.kwargs["sample_ids"] is None
    assert "stride" not in dataset.kwargs and "sequences" not in dataset.kwargs
    assert dataset.kwargs["root"] == "/portable/DSEC"
    assert manifest is None


def test_manifest_order_default_and_limit_remain_compatible(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"val": {"sample_ids": ["sequence:41", "sequence:39"]}}))
    args = SimpleNamespace(full_split=False, manifest=None, split="val", max_samples=1)
    dataset, ids, manifest = make_diagnostic_dataset(
        InventoryDataset, diagnostic_config(), args, default_manifest=path, future_rgb=True)
    assert ids == ["sequence:41"]
    assert dataset.kwargs["future_rgb"] is True
    assert manifest == path


def test_full_split_limit_uses_selected_inventory_and_closes_original():
    instances = []
    def build(**kwargs):
        dataset = InventoryDataset(**kwargs)
        instances.append(dataset)
        return dataset
    args = SimpleNamespace(full_split=True, manifest=None, split="test", max_samples=2)
    dataset, ids, _ = make_diagnostic_dataset(build, diagnostic_config(), args)
    assert ids == ["sequence:39", "sequence:40"]
    assert len(instances) == 2 and instances[0].closed
    assert not dataset.closed
    assert dataset.kwargs["split"] == "test"


def test_empty_diagnostic_selection_fails_and_closes_dataset():
    dataset = InventoryDataset(sample_ids=[])
    args = SimpleNamespace(full_split=True, manifest=None, split="val", max_samples=None)
    with pytest.raises(ValueError, match="empty"):
        make_diagnostic_dataset(lambda **_: dataset, diagnostic_config(), args)
    assert dataset.closed


def test_readout_cli_accepts_portable_full_split_and_rejects_manifest_conflict(monkeypatch):
    from foresight import diagnose_readout
    monkeypatch.setattr(diagnose_readout, "run", lambda args: args)
    args = diagnose_readout.main(["--full-split", "--split", "test", "--data-root", "/portable/DSEC",
                                  "--output", "outputs/full_test"])
    assert args.full_split and args.split == "test" and args.data_root == "/portable/DSEC"
    assert args.manifest is None
    with pytest.raises(SystemExit):
        diagnose_readout.main(["--full-split", "--manifest", "subset.json", "--output", "outputs/conflict"])
