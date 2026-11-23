"""Actual DinoF blocks at CPU-test scale: gradients, continuation and resume."""

from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from foresight.models.forecaster import DINOFeaturePredictor, PCAProjection
from foresight.research.config import load_config, validate_config
from foresight.research.task import load_task


class TinyConditionForecaster(nn.Module):
    image_size = (8, 12)
    grid_size = (2, 3)
    history_length = 3
    pca_dim = 8
    architecture = {"kind": "conditioning_test_only"}

    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 12, 1)
        self.pca = PCAProjection(dict(mean=torch.zeros(12), std=torch.ones(12),
                                      pca_mean=torch.zeros(12), pca_components=torch.eye(12)[:8]), 12, 8)
        self.predictor = DINOFeaturePredictor(self.grid_size, self.history_length, 8, 12, 4, 3)
        # Vendor construction zeros fc_out; a trained RGB checkpoint has a
        # nonzero readout. Model that here or no upstream gradient can exist.
        nn.init.normal_(self.predictor.maskvit.fc_out.weight, std=0.1)

    def encode_latents(self, image):
        dense = F.adaptive_avg_pool2d(self.encoder(image), self.grid_size).flatten(2).transpose(1, 2)
        return self.pca.project(dense)


def config(**model):
    return validate_config(dict(stage="conditioning",
        data=dict(image_size=[8, 12], event_image_size=[8, 12], context_frames=3, bins=3,
                  batch_size=2, num_workers=0, sample_history_frames=3,
                  alignment="prealigned_event_view", alignment_evidence="synthetic_common_grid"),
        model=dict(motion_dim=4, hidden_dim=8, injection_layer=2, **model),
        runtime=dict(accelerator="cpu", devices=1, cpu_threads=1, log_every=1),
        optimization=dict(epochs=3, max_steps=4), loss=dict(kind="smooth_l1")))


def batch():
    return dict(history_rgb=torch.randn(2, 3, 3, 8, 12),
                history_events=torch.rand(2, 2, 3, 2, 8, 12),
                future_rgb_target=torch.randn(2, 1, 3, 8, 12))


def save(task, path):
    saved = {"state_dict": task.state_dict()}
    task.on_save_checkpoint(saved)
    torch.save(saved, path)


def test_warmup_preserves_baseline_and_backpropagates_through_frozen_tail():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    task = load_task(config(), forecaster=TinyConditionForecaster()).train()
    data = batch()
    data["history_events"].requires_grad_()
    data["future_rgb_target"].requires_grad_()
    with torch.no_grad():
        baseline = task.forecaster.predictor(task.encode_history(data))
    torch.testing.assert_close(task.predict_batch(data)["scene"][:, 0], baseline, rtol=1e-6, atol=1e-6)
    optimizer = task.configure_optimizers()
    assert [group["name"] for group in optimizer.param_groups] == ["conditioner"]
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        task.compute_losses(data)["loss"].backward()
        optimizer.step()
    assert data["history_events"].grad.abs().sum() > 0
    assert data["future_rgb_target"].grad is None
    assert all(p.grad is None for p in task.forecaster.parameters())
    assert not task.forecaster.training and not task.forecaster.predictor.training
    expected = task.predict_batch(data)["scene"]
    causal = {k: v for k, v in data.items() if not k.startswith("future_")}
    assert torch.equal(expected, task.predict_batch(causal)["scene"])
    data["future_rgb_target"].data.fill_(999)
    data["future_events_target"] = torch.full((2, 1, 3, 2, 8, 12), float("nan"))
    assert torch.equal(expected, task.predict_batch(data)["scene"])


@pytest.mark.parametrize("mode", ["event_ordered", "event_pooled", "rgb"])
def test_joint_uses_full_predictor_and_separate_learning_rates(mode):
    torch.set_num_threads(1)
    task = load_task(config(mode=mode, train_predictor=True), forecaster=TinyConditionForecaster()).train()
    optimizer = task.configure_optimizers()
    groups = {group["name"]: group for group in optimizer.param_groups}
    assert groups["predictor"]["lr"] == 1e-5
    if mode != "rgb":
        assert groups["conditioner"]["lr"] == 1e-4
    task.compute_losses(batch())["loss"].backward()
    for layer in task.forecaster.predictor.maskvit.transformer.layers:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in layer.parameters())
    assert all(not p.requires_grad and p.grad is None for p in task.forecaster.encoder.parameters())
    assert all(not v.requires_grad for v in task.forecaster.pca.buffers())
    assert not task.forecaster.predictor.training  # eval disables dropout, not gradients


def test_weight_initialization_and_complete_roundtrip(tmp_path):
    task = load_task(config(), forecaster=TinyConditionForecaster()).train()
    data = batch()
    optimizer = task.configure_optimizers()
    task.compute_losses(data)["loss"].backward()
    optimizer.step()
    source = tmp_path / "warmup.ckpt"
    save(task, source)
    joint_cfg = config(train_predictor=True, init_checkpoint=str(source))
    joint = load_task(joint_cfg, forecaster=TinyConditionForecaster()).eval()
    assert joint.initialization_provenance["optimizer"] == "reset"
    assert not joint.configure_optimizers().state
    torch.testing.assert_close(joint.predict_batch(data)["scene"], task.eval().predict_batch(data)["scene"], rtol=0, atol=0)
    destination = tmp_path / "joint.ckpt"
    save(joint, destination)
    source.unlink()  # complete resume must not reopen the warmup source
    restored = load_task(joint_cfg, forecaster=TinyConditionForecaster(), checkpoint=destination).eval()
    assert torch.equal(joint.predict_batch(data)["scene"], restored.predict_batch(data)["scene"])
    assert sum(p.numel() for p in restored.forecaster.predictor.parameters() if p.requires_grad) > 0
    wrong = deepcopy(joint_cfg)
    wrong["optimization"]["predictor_lr"] *= 2
    with pytest.raises(ValueError, match="recipe differs"):
        load_task(wrong, forecaster=TinyConditionForecaster(), checkpoint=destination)


def test_initialization_rejects_changed_conditioner_or_protocol(tmp_path):
    task = load_task(config(), forecaster=TinyConditionForecaster())
    source = tmp_path / "warmup.ckpt"
    save(task, source)
    with pytest.raises(ValueError, match="architecture and mode"):
        load_task(config(mode="event_pooled", init_checkpoint=str(source)), forecaster=TinyConditionForecaster())
    changed = config(init_checkpoint=str(source))
    changed["data"]["sample_history_frames"] = 40
    with pytest.raises(ValueError, match="observation protocol"):
        load_task(changed, forecaster=TinyConditionForecaster())


def test_conditioning_config_contracts():
    for mutation, message in [
        (("data", "rollout_steps", 2), "one rollout"),
        (("loss", "motion_weight", 1), "no motion-target"),
        (("model", "train_predictor", "false"), "boolean"),
        (("model", "attention_heads", 3), "divisible"),
        (("optimization", "predictor_lr", 0), "predictor_lr"),
    ]:
        cfg = config()
        section, key, value = mutation
        cfg[section][key] = value
        with pytest.raises(ValueError, match=message):
            validate_config(cfg)
    invalid_depth = config()
    invalid_depth["model"]["injection_layer"] = 4
    with pytest.raises(ValueError, match="leave at least one"):
        load_task(invalid_depth, forecaster=TinyConditionForecaster())


def test_shipped_conditioning_recipes():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    recipes = list((root / "configs/research").glob("conditioning_*.yaml"))
    assert len(recipes) == 6
    for path in recipes:
        cfg = load_config(path)
        assert cfg["stage"] == "conditioning"
        assert cfg["data"]["sample_history_frames"] == 40
        assert cfg["model"]["injection_layer"] == 6
        assert cfg["loss"]["motion_weight"] == 0


def test_real_lightning_joint_resume_restores_optimizer(tmp_path):
    from lightning.pytorch import Callback, Trainer
    from torch.utils.data import DataLoader

    task = load_task(config(train_predictor=True), forecaster=TinyConditionForecaster())
    data = batch()
    loader = DataLoader([data] * 4, batch_size=None)
    checkpoint = tmp_path / "joint.ckpt"

    class StopAfterTwo(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if trainer.global_step == 2:
                trainer.save_checkpoint(checkpoint)
                raise RuntimeError("synthetic interruption")

    kwargs = dict(accelerator="cpu", devices=1, max_steps=4, max_epochs=3,
                  logger=False, enable_checkpointing=False, enable_progress_bar=False,
                  num_sanity_val_steps=0, limit_val_batches=0)
    trainer = Trainer(**kwargs, callbacks=[StopAfterTwo()])
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        trainer.fit(task, train_dataloaders=loader)
    saved = torch.load(checkpoint, weights_only=False)
    assert len(saved["optimizer_states"][0]["param_groups"]) == 2
    restored = load_task(task.config, forecaster=TinyConditionForecaster(), checkpoint=checkpoint)
    resumed = Trainer(**kwargs)
    resumed.fit(restored, train_dataloaders=loader, ckpt_path=checkpoint)
    assert resumed.global_step == 4
    assert max(int(value["step"]) for value in resumed.optimizers[0].state.values()) == 4


def test_rgb_predictor_override_and_embedded_resume(tmp_path):
    from foresight.checkpoints import TRAINING_FORMAT

    initial = TinyConditionForecaster()
    source = tmp_path / "rgb_predictor.ckpt"
    torch.save(dict(foresight_format=TRAINING_FORMAT, stage="predictor",
                    state_dict={"forecaster." + k: v for k, v in initial.state_dict().items()}), source)
    cfg = config(predictor_checkpoint=str(source))
    # Fixed encoder/PCA must match the RGB override; predictor may differ.
    forecaster = deepcopy(initial)
    nn.init.zeros_(forecaster.predictor.maskvit.fc_out.weight)
    task = load_task(cfg, forecaster=forecaster)
    assert torch.equal(task.forecaster.predictor.maskvit.fc_out.weight, initial.predictor.maskvit.fc_out.weight)
    assert task.predictor_initialization["sha256"]
    destination = tmp_path / "research.ckpt"
    save(task, destination)
    source.unlink()
    restored = load_task(cfg, forecaster=TinyConditionForecaster(), checkpoint=destination)
    assert restored.predictor_initialization == task.predictor_initialization
    assert torch.equal(restored.forecaster.encoder.weight, initial.encoder.weight)


@pytest.mark.parametrize("mode,train_predictor", [("event_ordered", False), ("rgb", True)])
def test_training_entrypoint_writes_usable_checkpoint(tmp_path, mode, train_predictor):
    from foresight.research.train import run

    cfg = config(mode=mode, train_predictor=train_predictor)
    cfg["optimization"].update(max_steps=1, epochs=1)
    data = batch()
    samples = [{**{key: value[index] for key, value in data.items()}, "targets": [], "meta": {}}
               for index in range(2)]
    trainer = run(cfg, output_dir=tmp_path / mode, forecaster=TinyConditionForecaster(),
                  datasets={"train": samples, "val": samples}, limit_train_batches=1, limit_val_batches=1)
    assert trainer.global_step == 1
    saved = torch.load(tmp_path / mode / "checkpoints/last.ckpt", weights_only=False)
    assert saved["research_format"] and saved["optimizer_states"]
    assert saved["protocol"]["target_event_usage"] == "unused"
    assert (tmp_path / mode / "run_train.json").is_file()
