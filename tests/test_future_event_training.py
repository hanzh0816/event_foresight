"""Causality and gradient boundaries of predictor-shaped event training."""

from copy import deepcopy
import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from foresight.future_event_training import FutureEventTask, FutureEventDataset, collate_future_event
from foresight.models.forecaster import DINOFeaturePredictor
from foresight.data.events import EventSequenceReader
from test_training import TinyForecaster, config, cpu_trainer
from test_events import event_fixture, rgb_fixture


class TinyEventEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(10, 2)

    def forward(self, events):
        return self.proj(events.mean((-1, -2)))[:, None]


def make_batch():
    generator = torch.Generator().manual_seed(80)
    return dict(history_events=torch.randn(2, 4, 10, 2, 2, generator=generator),
                history_rgb_target=torch.randn(2, 4, 3, 2, 2, generator=generator),
                future_rgb_target=torch.randn(2, 3, 2, 2, generator=generator))


def test_rejects_predictor_updates_that_event_inference_cannot_restore(tmp_path):
    cfg = config(tmp_path)
    cfg["optimization"]["trainable_predictor"] = True
    with pytest.raises(ValueError, match="only a frozen Predictor"):
        FutureEventTask(cfg, forecaster=TinyForecaster(), event_encoder=TinyEventEncoder())


def test_future_only_gradient_crosses_frozen_predictor_and_uses_real_target(tmp_path):
    cfg = config(tmp_path)
    cfg["optimization"].update(current_weight=0, future_weight=1)
    task = FutureEventTask(cfg, forecaster=TinyForecaster(), event_encoder=TinyEventEncoder())
    task.train()
    batch = make_batch()
    batch["history_events"].requires_grad_()
    state = deepcopy(task.forecaster.state_dict())
    prediction = task(batch["history_events"])
    losses = task.feature_losses(batch)
    losses["loss"].backward()
    assert losses["loss"].dtype == torch.float32
    assert task.event_encoder.proj.weight.grad.abs().sum() > 0
    assert all(batch["history_events"].grad[:, frame].abs().sum() > 0 for frame in range(4))
    assert all(not p.requires_grad and p.grad is None for p in task.forecaster.parameters())
    assert all(torch.equal(task.forecaster.state_dict()[k], v) for k, v in state.items())
    assert not any(module.training for module in task.forecaster.modules())
    assert torch.equal(task.forecaster.seen[-1], batch["future_rgb_target"])
    changed = dict(batch, future_rgb_target=batch["future_rgb_target"] + 100)
    assert torch.equal(task(changed["history_events"]), prediction)
    assert not torch.equal(task.feature_losses(changed)["future_smooth_l1"], losses["future_smooth_l1"])


def test_current_control_and_combined_weighting(tmp_path):
    cfg = config(tmp_path)
    cfg["optimization"].update(current_weight=2, future_weight=0)
    task = FutureEventTask(cfg, forecaster=TinyForecaster(), event_encoder=TinyEventEncoder())
    batch = make_batch()
    losses = task.feature_losses(batch)
    assert not losses["future_smooth_l1"].requires_grad
    torch.testing.assert_close(losses["loss"], 2 * losses["current_smooth_l1"])
    losses["loss"].backward()
    assert task.event_encoder.proj.weight.grad.abs().sum() > 0
    cfg["optimization"].update(current_weight=2, future_weight=3)
    combined = FutureEventTask(cfg, forecaster=TinyForecaster(), event_encoder=TinyEventEncoder())
    losses = combined.feature_losses(batch)
    torch.testing.assert_close(losses["loss"], 2 * losses["current_smooth_l1"] + 3 * losses["future_smooth_l1"])
    cfg["optimization"].update(current_weight=0, future_weight=0)
    with pytest.raises(ValueError, match="not both zero"):
        FutureEventTask(cfg, forecaster=TinyForecaster(), event_encoder=TinyEventEncoder())


def test_current_scope_and_checkpoint_objective_guard(tmp_path):
    cfg = config(tmp_path)
    cfg["optimization"].update(current_scope="last")
    task = FutureEventTask(cfg, forecaster=TinyForecaster(), event_encoder=TinyEventEncoder())
    batch = make_batch()
    with pytest.raises(ValueError, match="do not match"):
        task.feature_loss(batch)
    batch["history_rgb_target"] = batch["history_rgb_target"][:, -1:]
    assert torch.isfinite(task.feature_loss(batch))
    payload = {}
    task.on_save_checkpoint(payload)
    assert payload["stage"] == "event"
    assert payload["input_spec"]["training_objective"] == "future_event"
    assert payload["input_spec"]["current_scope"] == "last"


@pytest.mark.parametrize("encoding", [
    {},
    dict(event_window_us=200000, event_bin_widths_us=[100000, 50000, 25000, 15000, 10000]),
])
def test_dataset_separates_true_future_from_four_causal_inputs(event_fixture, monkeypatch, encoding):
    root, path, times, offset = event_fixture
    reads = []
    original = EventSequenceReader.read_image

    def read(self, index):
        reads.append(index)
        return original(self, index)

    monkeypatch.setattr(EventSequenceReader, "read_image", read)
    with FutureEventDataset(root, image_size=(28, 42), **encoding) as dataset:
        sample = dataset[0]
        assert reads == [36, 37, 38, 39, 40]
        assert sample["history_events"].shape == (4, 10, 28, 42)
        assert sample["history_rgb_target"].shape == (4, 3, 28, 42)
        assert sample["future_rgb_target"].shape == (3, 28, 42)
        assert max(sample["meta"]["event_end_us"]) == sample["meta"]["input_cutoff_us"]
        assert sample["meta"]["target_timestamp_us"] > sample["meta"]["input_cutoff_us"]
        assert all(stop - start == encoding.get("event_window_us", 50000)
                   for start, stop in zip(sample["meta"]["event_start_us"], sample["meta"]["event_end_us"]))
        batch = collate_future_event([sample, sample])
        assert batch["history_events"].shape[0] == batch["future_rgb_target"].shape[0] == 2


def test_published_predictor_checkpointing_preserves_input_gradients():
    torch.manual_seed(4)
    model = DINOFeaturePredictor(grid_size=(1, 2), history_length=4, pca_dim=12,
                                hidden_dim=12, depth=2, heads=2, dropout=0.1)
    # The vendor constructor starts fc_out at zero; pretrained weights do not.
    nn.init.normal_(model.maskvit.fc_out.weight, std=0.1)
    model.requires_grad_(False).eval()
    history = torch.randn(1, 4, 2, 12, requires_grad=True)
    plain = model(history)
    plain.square().mean().backward()
    plain_grad = history.grad.clone()
    history.grad = None
    model.activation_checkpointing = True
    checked = model(history)
    checked.square().mean().backward()
    torch.testing.assert_close(plain, checked)
    torch.testing.assert_close(plain_grad, history.grad)
    assert history.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())


def test_real_training_emits_gradient_and_budget_audits(tmp_path):
    cfg = config(tmp_path)
    cfg["optimization"].update(current_weight=0, future_weight=1)
    task = FutureEventTask(cfg, forecaster=TinyForecaster(), event_encoder=TinyEventEncoder())
    batch = make_batch()
    samples = [{**{key: value[i % 2] for key, value in batch.items()},
                "meta": {"sample_id": f"s{i}"}} for i in range(4)]
    task.train_dataloader = lambda: DataLoader(samples, batch_size=2, collate_fn=collate_future_event)
    trainer = cpu_trainer(tmp_path)
    trainer.fit(task)
    audit = json.loads((tmp_path / "gradient_audit.json").read_text())
    assert audit["encoder_gradient_norm"] > 0
    assert audit["frozen_forecaster_parameters_with_grad"] == []
    assert audit["predictor_eval"] and not audit["predictor_trainable"]
    progress = json.loads((tmp_path / "training_progress.json").read_text())
    assert progress["optimizer_steps"] == progress["microbatches_seen_per_rank"] == 2
    assert progress["examples_seen_per_rank"] == 4
    trace = [json.loads(line) for line in (tmp_path / "sample_order.jsonl").read_text().splitlines()]
    assert [sample for row in trace for sample in row["sample_ids"]] == ["s0", "s1", "s2", "s3"]
