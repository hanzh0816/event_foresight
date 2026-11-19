"""CPU tests for causal stage-one training and complete recipe restoration."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from foresight.research.config import load_config, validate_config
from foresight.research.task import load_task


class TinyPredictor(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.projection = nn.Linear(dim, dim)

    def forward(self, history):
        return self.projection(history[:, -1])


class TinyForecaster(nn.Module):
    image_size = (8, 12)
    grid_size = (2, 3)
    history_length = 3
    pca_dim = 8
    architecture = {"kind": "test_only"}

    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 8, 1)
        self.predictor = TinyPredictor()

    def encode_latents(self, image):
        value = torch.nn.functional.adaptive_avg_pool2d(self.encoder(image), self.grid_size)
        return value.flatten(2).transpose(1, 2)


def tiny_config(stage="fusion", **model):
    return validate_config(dict(stage=stage,
        data=dict(image_size=[8, 12], event_image_size=[8, 12], context_frames=3, bins=3,
                  batch_size=2, num_workers=0, alignment="prealigned_event_view", alignment_evidence="synthetic_common_grid"),
        model=dict(motion_dim=4, hidden_dim=8, **model),
        runtime=dict(accelerator="cpu", devices=1, cpu_threads=1, log_every=1),
        optimization=dict(epochs=1, max_steps=2)))


def tiny_batch(steps=1):
    return dict(history_rgb=torch.randn(2, 3, 3, 8, 12),
                history_events=torch.randn(2, 2, 3, 2, 8, 12),
                future_rgb_target=torch.randn(2, steps, 3, 8, 12),
                future_events_target=torch.randn(2, steps, 3, 2, 8, 12))


def save_task(task, path):
    saved = {"state_dict": task.state_dict()}
    task.on_save_checkpoint(saved)
    torch.save(saved, path)


def test_fusion_targets_do_not_enter_forecast_and_frozen_encoder():
    torch.set_num_threads(1)
    task = load_task(tiny_config(), forecaster=TinyForecaster()).train()
    batch = tiny_batch()
    expected = task.predict_batch(batch)["scene"]
    causal = {key: value for key, value in batch.items() if not key.startswith("future_")}
    assert torch.equal(expected, task.predict_batch(causal)["scene"])
    batch["future_rgb_target"].fill_(1000)
    batch["future_events_target"].fill_(-999)
    assert torch.equal(expected, task.predict_batch(batch)["scene"])
    task.compute_losses(batch)["loss"].backward()
    assert all(parameter.grad is None for parameter in task.forecaster.parameters())
    assert not task.forecaster.training
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in task.model.parameters())


def test_fusion_roundtrip_and_resume_recipe_check(tmp_path):
    task = load_task(tiny_config(), forecaster=TinyForecaster()).eval()
    batch = tiny_batch()
    checkpoint = tmp_path / "fusion.ckpt"
    save_task(task, checkpoint)
    restored = load_task(task.config, forecaster=TinyForecaster(), checkpoint=checkpoint).eval()
    assert torch.equal(task.predict_batch(batch)["scene"], restored.predict_batch(batch)["scene"])
    changed = deepcopy(task.config)
    changed["model"]["mode"] = "event_pooled"
    with pytest.raises(ValueError, match="recipe differs"):
        load_task(changed, forecaster=TinyForecaster(), checkpoint=checkpoint)


def test_recipe_inheritance_and_policy_validation(tmp_path):
    (tmp_path / "base.yaml").write_text("stage: fusion\nmodel:\n  hidden_dim: 16\ndata:\n  bins: 3\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\nmodel:\n  mode: event_pooled\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg["model"]["hidden_dim"] == 16 and cfg["data"]["bins"] == 3
    assert cfg["model"]["mode"] == "event_pooled"
    cfg["data"]["rollout_steps"] = 2
    with pytest.raises(ValueError, match="one rollout"):
        validate_config(cfg)
    (tmp_path / "cycle.yaml").write_text("extends: cycle.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        load_config(tmp_path / "cycle.yaml")


def test_real_lightning_fit_and_full_checkpoint(tmp_path):
    from lightning.pytorch import Callback, Trainer
    from torch.utils.data import DataLoader

    config = tiny_config()
    config["optimization"]["max_steps"] = 4
    config["optimization"]["epochs"] = 3
    task = load_task(config, forecaster=TinyForecaster())
    batch = tiny_batch()
    # A prebatched loader isolates real optimizer/checkpoint integration from I/O.
    loader = DataLoader([batch, batch, batch, batch], batch_size=None)
    checkpoint = tmp_path / "complete.ckpt"

    class InterruptAfterTwo(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if trainer.global_step == 2:
                trainer.save_checkpoint(checkpoint)
                raise RuntimeError("synthetic interruption")

    trainer = Trainer(accelerator="cpu", devices=1, max_steps=4, max_epochs=3,
                      logger=False, enable_checkpointing=False, enable_progress_bar=False,
                      num_sanity_val_steps=0, limit_val_batches=0, callbacks=[InterruptAfterTwo()])
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        trainer.fit(task, train_dataloaders=loader)
    assert trainer.global_step == 2
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["optimizer_states"] and saved["protocol"]["prediction_kind"] == "history_only_forecast"
    restored = load_task(config, forecaster=TinyForecaster(), checkpoint=checkpoint)
    assert torch.equal(task.eval().predict_batch(batch)["scene"], restored.eval().predict_batch(batch)["scene"])
    restored.train()
    resumed = Trainer(accelerator="cpu", devices=1, max_steps=4, max_epochs=3,
                      logger=False, enable_checkpointing=False, enable_progress_bar=False,
                      num_sanity_val_steps=0, limit_val_batches=0)
    resumed.fit(restored, train_dataloaders=loader, ckpt_path=checkpoint)
    assert resumed.global_step == 4


def test_train_run_rejects_existing_output_before_loading_weights(tmp_path):
    from foresight.research.train import run

    (tmp_path / "precious.txt").write_text("preserve")
    with pytest.raises(FileExistsError, match="empty output"):
        run(tiny_config(), output_dir=tmp_path)
