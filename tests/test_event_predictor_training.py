"""Real CPU optimization/resume and Event P inference protocol boundaries."""

from copy import deepcopy
import json

from lightning.pytorch import Trainer
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from foresight import event, pipeline
from foresight.checkpoints import TRAINING_FORMAT, load_predictor
from foresight.data.events import EventSequenceReader, encoding_spec
from foresight.event_predictor_training import EventPredictorDataset, EventPredictorTask, collate_event_predictor
from foresight.models.event_encoder import EventEncoder
from foresight.train import resume_config
from test_events import SmallTeacher, task_config, event_fixture, rgb_fixture, threads


class SmallPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.proj = nn.Linear(8, 8)

    def forward(self, history):
        return self.proj(self.dropout(history.mean(1)))


def make_source(tmp_path):
    torch.manual_seed(51)
    teacher = SmallTeacher()
    teacher.predictor = SmallPredictor()
    encoder = EventEncoder(teacher)
    with torch.no_grad():
        encoder.backbone.patch_embed.proj.weight.add_(0.1)
    cfg = task_config(tmp_path)
    cfg["optimization"].update(predictor_dropout=False)
    cfg["data"]["batch_size"] = 1
    cfg["runtime"].update(encoder_chunk_size=1)
    source = tmp_path / "full_joint.ckpt"
    # Small tensor architecture, same source container format as full_joint.
    saved = dict(foresight_format=TRAINING_FORMAT, stage="event", image_size=[448, 896],
                 event_encoding=encoding_spec({}), state_dict={
                     **{"event_encoder." + k: v for k, v in encoder.state_dict().items()},
                     **{"forecaster." + k: v for k, v in teacher.state_dict().items()}})
    torch.save(saved, source)
    cfg["model"]["event_encoder_checkpoint"] = str(source)
    return cfg, teacher, encoder


def make_batch():
    generator = torch.Generator().manual_seed(53)
    return {"history_events": torch.rand(2, 4, 10, 28, 28, generator=generator),
            "future_rgb_target": torch.rand(2, 3, 28, 28, generator=generator)}


def make_task(cfg, teacher, **kwargs):
    return EventPredictorTask(cfg, forecaster=deepcopy(teacher), **kwargs)


def attach_data(task):
    batch = make_batch()
    samples = [{**{key: value[i] for key, value in batch.items()}, "meta": {"sample_id": f"s{i}"}}
               for i in range(2)]
    task.train_dataloader = lambda: DataLoader(samples, batch_size=1, collate_fn=collate_event_predictor)


def trainer(tmp_path, steps):
    return Trainer(accelerator="cpu", devices=1, max_epochs=3, max_steps=steps,
                   logger=False, enable_checkpointing=False, enable_progress_bar=False,
                   enable_model_summary=False, num_sanity_val_steps=0, limit_val_batches=0,
                   default_root_dir=tmp_path)


def assert_equal(actual, expected):
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[k], expected[k]) for k in expected)


def test_future_loss_updates_only_predictor_and_cannot_read_future_inputs(tmp_path):
    cfg, teacher, encoder = make_source(tmp_path)
    task = make_task(cfg, teacher)
    batch = make_batch()
    batch["history_events"].requires_grad_()
    task.train()
    before = deepcopy(task.state_dict())
    prediction = task(batch["history_events"])
    loss = task.feature_loss(batch)
    loss.backward()
    assert loss.dtype == torch.float32
    assert batch["history_events"].grad is None
    assert all(p.grad is None and not p.requires_grad for p in task.event_encoder.parameters())
    assert all(p.grad is None and not p.requires_grad for p in task.forecaster.encoder.parameters())
    assert all(p.grad is None and not p.requires_grad for p in task.forecaster.pca.parameters())
    assert all(p.requires_grad and p.grad is not None for p in task.forecaster.predictor.parameters())
    assert not any(m.training for m in task.modules() if isinstance(m, nn.Dropout))
    modified = dict(batch, future_rgb_target=batch["future_rgb_target"] + 100)
    torch.testing.assert_close(prediction, task(modified["history_events"]))
    assert not torch.equal(loss, task.feature_loss(modified))
    task.configure_optimizers().step()
    changed = [k for k, value in task.state_dict().items() if not torch.equal(before[k], value)]
    assert changed and all(k.startswith("forecaster.predictor.") for k in changed)
    assert_equal(task.event_encoder.state_dict(), encoder.state_dict())


def test_dataset_reads_only_future_rgb_and_causal_event_windows(event_fixture, monkeypatch):
    root, _, _, _ = event_fixture
    reads = []
    original = EventSequenceReader.read_image

    def read(self, index):
        reads.append(index)
        return original(self, index)

    monkeypatch.setattr(EventSequenceReader, "read_image", read)
    with EventPredictorDataset(root, image_size=(28, 28)) as dataset:
        sample = dataset[0]
        assert reads == [40]
        assert "history_rgb" not in sample and "history_rgb_target" not in sample
        assert sample["history_events"].shape == (4, 10, 28, 28)
        assert max(sample["meta"]["event_end_us"]) == sample["meta"]["input_cutoff_us"]
        assert sample["meta"]["target_timestamp_us"] > sample["meta"]["input_cutoff_us"]


def test_checkpoint_resume_restores_frozen_encoder_optimizer_and_provenance(tmp_path):
    cfg, teacher, source_encoder = make_source(tmp_path)
    task = make_task(cfg, teacher)
    attach_data(task)
    first = trainer(tmp_path, 2)
    first.fit(task)
    path = tmp_path / "last.ckpt"
    first.save_checkpoint(path)
    audit = json.loads((tmp_path / "gradient_audit.json").read_text())
    assert audit["predictor_gradient_norm"] > 0 and audit["frozen_parameters_with_grad"] == []
    before = task(make_batch()["history_events"]).detach()
    restored_cfg = resume_config(cfg, path, "event_predictor")
    assert restored_cfg["model"]["event_encoder_checkpoint"] is None
    # Resume must not depend on the original Encoder file or teacher initialization.
    source_path = cfg["model"]["event_encoder_checkpoint"]
    from pathlib import Path
    Path(source_path).unlink()
    new_teacher = deepcopy(teacher)
    with torch.no_grad():
        new_teacher.encoder.cls_token.add_(1)
    restored = make_task(restored_cfg, new_teacher, resume=True)
    attach_data(restored)
    second = trainer(tmp_path, 3)
    outputs_at_resume = []
    original_start = restored.on_train_start

    def on_start():
        original_start()
        outputs_at_resume.append(restored(make_batch()["history_events"]).detach())
        assert restored.initialization["event_encoder_checkpoint"] == source_path

    restored.on_train_start = on_start
    second.fit(restored, ckpt_path=path)
    assert len(outputs_at_resume) == 1
    torch.testing.assert_close(before, outputs_at_resume[0])
    assert second.global_step == 3
    assert all(int(state["step"]) == 3 for state in second.optimizers[0].state.values())
    assert_equal(restored.event_encoder.state_dict(), source_encoder.state_dict())
    assert_equal(restored.forecaster.encoder.state_dict(), teacher.encoder.state_dict())
    assert restored.examples_seen == 3 and restored.microbatches_seen == 3
    assert not torch.equal(before, restored(make_batch()["history_events"]))


def predictor_payload(task):
    saved = {"state_dict": deepcopy(task.state_dict()), "image_size": [448, 896]}
    task.on_save_checkpoint(saved)
    return saved


def test_inference_loads_paired_predictor_last_and_supports_bundle(tmp_path, monkeypatch):
    cfg, teacher, encoder = make_source(tmp_path)
    task = make_task(cfg, teacher)
    with torch.no_grad():
        task.forecaster.predictor.proj.weight.add_(3)
    path = tmp_path / "event_predictor.ckpt"
    torch.save(predictor_payload(task), path)
    monkeypatch.setattr(pipeline, "DINOForecaster", lambda *a, **kw: deepcopy(teacher))
    monkeypatch.setattr(pipeline, "load_detector", lambda *a: nn.Identity())
    cfg["model"]["detector_checkpoint"] = "unused-head.pt"
    cfg["model"]["predictor_checkpoint"] = str(path)
    with torch.no_grad():
        expected = task.forecaster.pca.inverse(task(make_batch()["history_events"]))
    assembled = event.load_event_pipeline(cfg, cfg["model"]["event_encoder_checkpoint"])
    torch.testing.assert_close(assembled(make_batch()["history_events"]), expected)
    assert_equal(assembled.event_encoder.state_dict(), encoder.state_dict())
    cfg["model"]["predictor_checkpoint"] = None
    bundled = event.load_event_pipeline(cfg, path)
    torch.testing.assert_close(bundled(make_batch()["history_events"]), expected)
    assert bundled.predictor_source == str(path)
    assert all(not p.requires_grad for p in bundled.parameters())
    with pytest.raises(ValueError, match="Predictor override"):
        load_predictor(deepcopy(teacher.predictor), path, forecaster=teacher)


@pytest.mark.parametrize("corruption,match", [
    ("encoder", "frozen Event Encoder"), ("teacher", "frozen RGB encoder"),
    ("pca", "frozen PCA"), ("encoding", "input/encoding protocol"),
    ("objective", "input/encoding protocol"), ("shape", "structure differs"),
])
def test_rejects_mismatched_predictor_before_mutation(tmp_path, corruption, match):
    cfg, teacher, encoder = make_source(tmp_path)
    task = make_task(cfg, teacher)
    saved = predictor_payload(task)
    if corruption == "encoder":
        saved["state_dict"]["event_encoder.backbone.cls_token"].add_(1)
    elif corruption == "teacher":
        saved["state_dict"]["forecaster.encoder.cls_token"].add_(1)
    elif corruption == "pca":
        saved["state_dict"]["forecaster.pca.mean"].add_(1)
    elif corruption == "encoding":
        saved["input_spec"]["event_encoding"]["event_bin_widths_us"] = [20000, 12000, 8000, 6000, 4000]
    elif corruption == "objective":
        saved["input_spec"]["input_modality"] = "rgb"
    else:
        saved["state_dict"]["forecaster.predictor.proj.weight"] = torch.zeros(3, 3)
    path = tmp_path / "wrong.ckpt"
    torch.save(saved, path)
    before = deepcopy(teacher.predictor.state_dict())
    with pytest.raises(ValueError, match=match):
        event.load_event_predictor(teacher, encoder, encoding_spec({}), path)
    assert_equal(teacher.predictor.state_dict(), before)


def test_fresh_training_rejects_missing_encoder_and_changed_encoding(tmp_path):
    cfg, teacher, _ = make_source(tmp_path)
    cfg["data"].update(event_bin_widths_us=[20000, 12000, 8000, 6000, 4000])
    with pytest.raises(ValueError, match="configured event encoding differ"):
        make_task(cfg, teacher)
    cfg["model"]["event_encoder_checkpoint"] = None
    with pytest.raises(ValueError, match="trained full_joint"):
        make_task(cfg, teacher)


def test_legacy_encoder_and_explicit_rgb_predictor_override_still_work(tmp_path, monkeypatch):
    cfg, teacher, encoder = make_source(tmp_path)
    monkeypatch.setattr(pipeline, "DINOForecaster", lambda *a, **kw: deepcopy(teacher))
    monkeypatch.setattr(pipeline, "load_detector", lambda *a: nn.Identity())
    cfg["model"]["detector_checkpoint"] = "unused-head.pt"
    old = event.load_event_pipeline(cfg, cfg["model"]["event_encoder_checkpoint"])
    assert_equal(old.forecaster.predictor.state_dict(), teacher.predictor.state_dict())
    rgb = deepcopy(teacher)
    with torch.no_grad():
        rgb.predictor.proj.bias.add_(2)
    path = tmp_path / "rgb_predictor.ckpt"
    torch.save(dict(foresight_format=TRAINING_FORMAT, stage="predictor",
                    state_dict={"forecaster." + k: v for k, v in rgb.state_dict().items()}), path)
    cfg["model"]["predictor_checkpoint"] = str(path)
    assembled = event.load_event_pipeline(cfg, cfg["model"]["event_encoder_checkpoint"])
    assert_equal(assembled.forecaster.predictor.state_dict(), rgb.predictor.state_dict())
    assert_equal(assembled.event_encoder.state_dict(), encoder.state_dict())
    # An explicit RGB override also intentionally takes priority over bundle P.
    bundle = tmp_path / "bundle.ckpt"
    torch.save(predictor_payload(make_task({**cfg, "model": {**cfg["model"], "predictor_checkpoint": None}}, teacher)), bundle)
    overridden = event.load_event_pipeline(cfg, bundle)
    assert_equal(overridden.forecaster.predictor.state_dict(), rgb.predictor.state_dict())
    assert overridden.predictor_source == str(path)
