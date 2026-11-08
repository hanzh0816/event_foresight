"""CPU tests of stage boundaries and real Lightning resume; no model assets needed."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer

from foresight.training import CHECKPOINT_FORMAT, DetectorTask, PredictorTask
from foresight.samplers import ValidationSampler
from foresight.train import batch_limit, make_trainer, resume_config


class TinyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.activation_checkpointing = False

    def forward(self, history):
        return self.proj(self.dropout(history.mean(1)))


class TinyForecaster(nn.Module):
    image_size = (2, 2)
    feature_dim = 4
    grid_size = (1, 1)

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3, 4), nn.Dropout(0.5))
        self.pca = nn.Linear(4, 2, bias=False)
        self.predictor = TinyPredictor()
        self.seen = []

    def encode_latents(self, rgb):
        self.seen.append(rgb.detach().clone())
        return self.pca(self.encoder(rgb.mean((-1, -2))))[:, None]

    def forecast(self, history):
        b, t = history.shape[:2]
        encoded = self.encode_latents(history.flatten(0, 1)).reshape(b, t, 1, 2)
        return self.predictor(encoded) @ self.pca.weight


class TinyCriterion(nn.Module):
    def forward(self, outputs, targets, sync_num_boxes=False):
        assert sync_num_boxes
        target = torch.stack([item["value"] for item in targets])
        return {"loss": (outputs[:, 0, 0] - target).square().mean()}


def config(tmp_path):
    return {
        "data": {"root": str(tmp_path), "image_size": [2, 2], "context_frames": 4,
                 "sample_history_frames": 40, "batch_size": 2, "num_workers": 0},
        "model": {"dinof_checkpoint": "unused.pt", "predictor_checkpoint": None,
                  "detector_checkpoint": None},
        "optimization": {"epochs": 3, "max_steps": 2, "accumulation_steps": 1,
                         "lr": 1e-3, "weight_decay": 0.0, "scheduler": "constant",
                         "clip_grad": 1.0, "predictor_dropout": False},
        "runtime": {"seed": 42, "precision": "32-true", "log_every": 1},
        "output_dir": str(tmp_path),
    }


def sample(index):
    g = torch.Generator().manual_seed(index)
    return {"history_rgb": torch.randn(4, 3, 2, 2, generator=g),
            "future_rgb_target": torch.randn(3, 2, 2, generator=g)}


def cpu_trainer(tmp_path, **kwargs):
    return Trainer(accelerator="cpu", devices=1, max_epochs=3, max_steps=2,
                   logger=False, enable_checkpointing=False, enable_progress_bar=False,
                   enable_model_summary=False, num_sanity_val_steps=0,
                   limit_val_batches=0, default_root_dir=tmp_path, **kwargs)


class TinyPredictorTask(PredictorTask):
    def train_dataloader(self):
        return DataLoader([sample(i) for i in range(6)], batch_size=2, shuffle=False)


def test_predictor_uses_history_only_and_freezes_teacher(tmp_path):
    forecaster = TinyForecaster()
    task = PredictorTask(config(tmp_path), forecaster=forecaster)
    task.train()
    batch = next(iter(DataLoader([sample(1), sample(2)], batch_size=2)))
    prediction = task(batch["history_rgb"])
    loss = task.feature_loss(batch)
    loss.backward()
    assert loss.dtype == torch.float32
    assert all(p.grad is None and not p.requires_grad for p in forecaster.encoder.parameters())
    assert all(p.grad is None and not p.requires_grad for p in forecaster.pca.parameters())
    assert all(p.grad is not None and p.requires_grad for p in forecaster.predictor.parameters())
    assert not forecaster.encoder.training and not forecaster.pca.training
    assert not forecaster.predictor.training
    assert torch.equal(forecaster.seen[-1], batch["future_rgb_target"])
    modified = dict(batch, future_rgb_target=batch["future_rgb_target"] + 20)
    assert torch.equal(prediction, task(modified["history_rgb"]))
    assert not torch.equal(loss, task.feature_loss(modified))
    with pytest.raises(ValueError, match="separate"):
        task.feature_loss({"history_rgb": batch["history_rgb"]})


def test_detector_updates_only_head(tmp_path):
    task = DetectorTask(config(tmp_path), forecaster=TinyForecaster(),
                        detector=nn.Linear(4, 1), criterion=TinyCriterion())
    task.train()
    batch = next(iter(DataLoader([sample(1), sample(2)], batch_size=2)))
    outputs = task(batch["history_rgb"])
    outputs.square().mean().backward()
    assert all(not p.requires_grad and p.grad is None for p in task.forecaster.parameters())
    assert all(p.requires_grad and p.grad is not None for p in task.detector.parameters())
    assert not task.forecaster.training and task.detector.training
    assert all(key.startswith(("forecaster.", "detector.")) for key in task.state_dict())


def test_full_lightning_checkpoint_resumes_optimizer_and_all_weights(tmp_path):
    torch.manual_seed(2)
    original = TinyPredictorTask(config(tmp_path), forecaster=TinyForecaster())
    frozen = {k: v.clone() for k, v in original.forecaster.state_dict().items()
              if not k.startswith("predictor.")}
    trainer = cpu_trainer(tmp_path)
    trainer.fit(original)
    checkpoint = tmp_path / "last.ckpt"
    trainer.save_checkpoint(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["foresight_format"] == CHECKPOINT_FORMAT and saved["stage"] == "predictor"
    assert saved["global_step"] == 2
    assert all(key.startswith("forecaster.") for key in saved["state_dict"])
    assert all(torch.equal(original.forecaster.state_dict()[k], v) for k, v in frozen.items())
    assert all(int(state["step"]) == 2 for state in saved["optimizer_states"][0]["state"].values())

    restored = TinyPredictorTask(config(tmp_path), forecaster=TinyForecaster())
    resumed = Trainer(accelerator="cpu", devices=1, max_steps=3, max_epochs=3,
                      logger=False, enable_checkpointing=False, enable_progress_bar=False,
                      enable_model_summary=False, num_sanity_val_steps=0, limit_val_batches=0)
    resumed.fit(restored, ckpt_path=checkpoint)
    assert resumed.global_step == 3
    assert all(torch.equal(restored.forecaster.state_dict()[k], v) for k, v in frozen.items())
    assert all(int(state["step"]) == 3 for state in resumed.optimizers[0].state.values())
    assert any(not torch.equal(restored.state_dict()[key], value)
               for key, value in saved["state_dict"].items() if "predictor." in key)


def test_checkpoint_rejects_stage_protocol_and_batch_changes(tmp_path):
    task = PredictorTask(config(tmp_path), forecaster=TinyForecaster())
    payload = {"state_dict": task.state_dict()}
    task.on_save_checkpoint(payload)
    task.on_load_checkpoint(payload)
    for field, bad_value, match in (
        ("stage", "detector", "different training stage"),
        ("input_spec", {}, "protocol"),
        ("training_spec", {}, "batch"),
    ):
        changed = deepcopy(payload)
        changed[field] = bad_value
        with pytest.raises(ValueError, match=match):
            task.on_load_checkpoint(changed)
    cfg = config(tmp_path)
    cfg["data"]["batch_size"] = 1
    changed_batch = PredictorTask(cfg, forecaster=TinyForecaster())
    with pytest.raises(ValueError, match="batch"):
        changed_batch.on_load_checkpoint(payload)


def test_accumulation_normalizes_actual_global_examples(tmp_path):
    task = PredictorTask(config(tmp_path), forecaster=TinyForecaster())
    task._trainer = SimpleNamespace(accumulate_grad_batches=4, world_size=1)
    logged = {}
    task.log = lambda key, value, **kwargs: logged.update({key: value.item()})
    task.accumulated_examples = 5
    for parameter in task.trainable_parameters():
        parameter.grad = torch.ones_like(parameter) * (5 / 4)
    task.on_before_optimizer_step(None)
    assert all(torch.allclose(p.grad, torch.ones_like(p)) for p in task.trainable_parameters())
    assert logged["train_global_batch"] == 5
    assert task.accumulated_examples == 0


def test_validation_sampler_no_duplicates_and_empty_rank_placeholder():
    dataset = list(range(5))
    parts = [list(ValidationSampler(dataset, num_replicas=3, rank=rank)) for rank in range(3)]
    assert sorted(sum(parts, [])) == dataset
    empty = ValidationSampler([0], num_replicas=3, rank=2)
    assert list(empty) == [0] and len(empty) == 1


def test_trainer_uses_all_visible_gpus_and_cli_batch_limit(monkeypatch, tmp_path):
    monkeypatch.setattr("foresight.train.torch.cuda.device_count", lambda: 2)
    monkeypatch.setattr("foresight.train.Trainer", lambda **kwargs: kwargs)
    kwargs = make_trainer(config(tmp_path), "predictor", limit_train_batches=2, limit_val_batches=0)
    assert kwargs["devices"] == "auto" and kwargs["accelerator"] == "gpu"
    assert kwargs["use_distributed_sampler"] is False
    assert kwargs["strategy"]._ddp_kwargs["broadcast_buffers"] is False
    first_logger = kwargs["logger"][0]
    first_logger.log_metrics({"loss": 1.0}, step=0)
    first_logger.save()
    second = make_trainer(config(tmp_path), "predictor", limit_train_batches=2, limit_val_batches=0)
    assert second["logger"][0].log_dir != first_logger.log_dir
    assert (tmp_path / "logs" / "version_0" / "metrics.csv").is_file()
    assert batch_limit("2") == 2 and type(batch_limit("2")) is int
    assert batch_limit("1.0") == 1.0 and type(batch_limit("1.0")) is float
    assert batch_limit("0.25") == 0.25
    with pytest.raises(Exception):
        batch_limit("2.5")


def test_resume_recipe_preserves_budget_and_selection(monkeypatch, tmp_path):
    saved_config = config(tmp_path)
    saved_config["data"]["train_sample_ids"] = ["a", "b"]
    saved_config["optimization"]["lr"] = 2e-5
    saved_config["runtime"].update(seed=123, precision="bf16-mixed", encoder_chunk_size=2)
    saved_config["model"]["predictor_checkpoint"] = "/old/missing/init.ckpt"
    monkeypatch.setattr("foresight.checkpoints.read_checkpoint", lambda path: {
        "foresight_format": CHECKPOINT_FORMAT, "stage": "predictor", "config": saved_config})
    current = config(tmp_path)
    current["data"]["root"] = "/new/data"
    current["model"]["dinof_checkpoint"] = "/new/dinof.pt"
    result = resume_config(current, "unused.ckpt", "predictor")
    assert result["optimization"]["lr"] == 2e-5
    assert result["data"]["train_sample_ids"] == ["a", "b"]
    assert result["data"]["root"] == "/new/data"
    assert result["model"]["predictor_checkpoint"] is None
    assert result["runtime"]["seed"] == 123
    assert result["runtime"]["precision"] == "bf16-mixed"
    assert result["runtime"]["encoder_chunk_size"] == 2


def test_scratch_detector_architecture_is_saved_and_restored(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg["detector"] = {"hidden_dim": 8, "num_classes": 8, "num_queries": 2,
                       "num_levels": 1, "encoder_layers": 1, "decoder_layers": 1,
                       "heads": 2, "points": 1, "mlp_dim": 16, "dropout": 0.1}
    task = DetectorTask(cfg, forecaster=TinyForecaster(), criterion=TinyCriterion())
    payload = {"state_dict": task.state_dict()}
    task.on_save_checkpoint(payload)
    architecture = payload["detector_architecture"]
    assert architecture["feature_dim"] == 4 and architecture["grid_size"] == [1, 1]
    assert architecture["hidden_dim"] == 8
    monkeypatch.setattr("foresight.checkpoints.read_checkpoint", lambda path: payload)
    restored_config = resume_config(config(tmp_path), "unused.ckpt", "detector")
    assert restored_config["model"]["detector_checkpoint"] is None
    assert restored_config["detector"]["hidden_dim"] == 8
    restored = DetectorTask(restored_config, forecaster=TinyForecaster(), criterion=TinyCriterion())
    restored.on_load_checkpoint(payload)
    restored.load_state_dict(payload["state_dict"], strict=True)
    assert all(torch.equal(task.state_dict()[key], value) for key, value in restored.state_dict().items())
