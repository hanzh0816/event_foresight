"""Portable full-data recipes and exact epoch-boundary training resume."""

from copy import deepcopy
import json
from types import SimpleNamespace

from lightning.pytorch import Trainer
import pytest
import torch
from torch.utils.data import DataLoader

from foresight.config import ROOT, load_config
from foresight.future_event_training import FutureEventTask, collate_future_event
from foresight import train_future_event as launcher
from test_events import SmallTeacher, task_config
from test_future_event_training import TinyEventEncoder, make_batch
from test_training import TinyForecaster, config


def test_full_recipes_match_budgets_use_rgb_and_resolve_outside_repo(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    current = load_config("configs/event_full_current.yaml")
    joint = load_config("configs/event_full_joint.yaml")
    assert current["optimization"]["future_weight"] == 0
    assert joint["optimization"]["future_weight"] == 1
    current["optimization"]["future_weight"] = 1
    assert current == joint
    assert joint["data"]["root"] == str(ROOT / "data/DSEC")
    assert joint["model"]["dinof_checkpoint"] == str((ROOT / "checkpoints/dinof.pt").resolve())
    assert joint["model"]["event_encoder_checkpoint"] is None
    assert joint["model"]["predictor_checkpoint"] is None
    assert not any(key in joint["data"] for key in
                   ("manifest", "train_sample_ids", "val_sample_ids", "alignment_val_samples"))
    assert joint["optimization"]["epochs"] == 10
    assert joint["optimization"]["max_steps"] == -1
    assert joint["optimization"]["keep_epoch_checkpoints"]

    # The long-window experiment must retain the same full-data objective,
    # initialization and optimization budget as its 50ms control.
    long = load_config("configs/event_full_joint_200ms_recent_fine.yaml")
    assert long["data"]["event_window_us"] == 200000
    assert long["data"].pop("event_bin_widths_us") == [100000, 50000, 25000, 15000, 10000]
    long["data"]["event_window_us"] = 50000
    assert long == joint


def test_null_event_checkpoint_initializes_actual_rgb_backbone(tmp_path):
    cfg = task_config(tmp_path)
    cfg["model"]["event_encoder_checkpoint"] = None
    teacher = SmallTeacher()
    task = FutureEventTask(cfg, forecaster=teacher)
    encoder = task.event_encoder.backbone
    for name, tensor in teacher.encoder.state_dict().items():
        expected = tensor.mean(1, keepdim=True).repeat(1, 10, 1, 1) * .3 if name == "patch_embed.proj.weight" else tensor
        assert torch.equal(encoder.state_dict()[name], expected), name
    assert all(not parameter.requires_grad for parameter in task.forecaster.parameters())
    assert task.initialization["kind"] == "rgb_pretrained"
    payload = {}
    task.on_save_checkpoint(payload)
    assert payload["event_initialization"] == task.initialization


def test_full_dataset_has_no_implicit_pilot_selection(monkeypatch, tmp_path):
    cfg = config(tmp_path)
    cfg["data"]["tolerance_us"] = 1000
    captured = []
    monkeypatch.setattr("foresight.future_event_training.FutureEventDataset",
                        lambda **kwargs: captured.append(kwargs) or kwargs)
    task = FutureEventTask(cfg, forecaster=TinyForecaster(), event_encoder=TinyEventEncoder())
    task._make_dataset("train")
    task._make_dataset("val")
    assert all(item["sample_ids"] is None and item["sequences"] is None and item["stride"] == 1
               for item in captured)


def test_cli_default_is_full_joint_and_supports_portable_overrides(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "run", lambda cfg, **kwargs: (cfg, kwargs))
    cfg, kwargs = launcher.main(["--output-dir", str(tmp_path), "--data-root", str(tmp_path / "DSEC"),
                                 "--dinof-checkpoint", str(tmp_path / "dinof.pt"), "--seed", "7",
                                 "--epochs", "2", "--max-steps", "-1"])
    assert cfg["model"]["event_encoder_checkpoint"] is None
    assert cfg["data"]["root"] == str(tmp_path / "DSEC")
    assert cfg["model"]["dinof_checkpoint"] == str(tmp_path / "dinof.pt")
    assert cfg["runtime"]["seed"] == 7
    assert kwargs["epochs"] == 2 and kwargs["max_steps"] == -1
    with pytest.raises(SystemExit):
        launcher.main(["--output-dir", str(tmp_path), "--resume", "some.ckpt", "--future-weight", "0"])


class TinyFutureTask(FutureEventTask):
    def __init__(self, cfg):
        super().__init__(cfg, forecaster=TinyForecaster(), event_encoder=TinyEventEncoder())

    def train_dataloader(self):
        batch = make_batch()
        samples = [{**{key: value[i % 2] for key, value in batch.items()},
                    "meta": {"sample_id": f"s{i}"}} for i in range(4)]
        return DataLoader(samples, batch_size=2, collate_fn=collate_future_event)


def cpu_future_trainer(cfg, stage, **limits):
    return Trainer(accelerator="cpu", devices=1, max_epochs=cfg["optimization"]["epochs"],
                   max_steps=cfg["optimization"]["max_steps"],
                   accumulate_grad_batches=cfg["optimization"]["accumulation_steps"],
                   logger=False, enable_progress_bar=False, enable_model_summary=False,
                   num_sanity_val_steps=0, default_root_dir=cfg["output_dir"], **limits)


def test_epoch_checkpoints_and_resume_preserve_optimizer_weights_and_progress(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True), set_num_threads=torch.set_num_threads,
        set_float32_matmul_precision=torch.set_float32_matmul_precision))
    monkeypatch.setattr(launcher, "FutureEventTask", TinyFutureTask)
    monkeypatch.setattr(launcher, "make_trainer", cpu_future_trainer)
    cfg = config(tmp_path)
    cfg["optimization"].update(epochs=2, max_steps=-1, keep_epoch_checkpoints=True)
    continuous = launcher.run(cfg, output_dir=tmp_path / "continuous", limit_val_batches=0)
    split_dir = tmp_path / "split"
    first = launcher.run(cfg, output_dir=split_dir, epochs=1, limit_val_batches=0)
    assert first.global_step == 2
    assert (split_dir / "checkpoints/epoch_0000.ckpt").is_file()
    assert (split_dir / "checkpoints/last.ckpt").is_file()
    original_recipe = (split_dir / "config_train.yaml").read_text()
    # CLI/config changes cannot silently change the checkpoint's objective/seed.
    changed_cfg = deepcopy(cfg)
    changed_cfg["runtime"]["seed"] = 17
    changed_cfg["optimization"]["future_weight"] = 0
    resumed = launcher.run(changed_cfg, output_dir=split_dir, epochs=2, limit_val_batches=0,
                           resume=split_dir / "checkpoints/epoch_0000.ckpt")
    assert resumed.global_step == continuous.global_step == 4
    for name, tensor in continuous.lightning_module.state_dict().items():
        torch.testing.assert_close(resumed.lightning_module.state_dict()[name], tensor, rtol=0, atol=0)
    assert all(int(state["step"]) == 4 for state in resumed.optimizers[0].state.values())
    assert resumed.lightning_module.config["runtime"]["seed"] == 42
    assert resumed.lightning_module.future_weight == 1
    assert (split_dir / "checkpoints/epoch_0001.ckpt").is_file()
    assert (split_dir / "config_train.yaml").read_text() == original_recipe
    assert (split_dir / "config_resume.yaml").is_file()
    progress = json.loads((split_dir / "training_progress.json").read_text())
    assert progress["examples_seen_per_rank"] == 8
    assert progress["microbatches_seen_per_rank"] == 4
    assert progress["initialization"]["kind"] == "rgb_pretrained"
    trace = [json.loads(row) for row in (split_dir / "sample_order.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in trace] == [0, 0, 1, 1]
    assert len(resumed.checkpoint_callbacks) == 1


def test_pilot_does_not_enable_epoch_snapshots(tmp_path):
    cfg = config(tmp_path)
    trainer = cpu_future_trainer(cfg, "event", limit_train_batches=1.0, limit_val_batches=0)
    original = list(trainer.callbacks)
    launcher.configure_epoch_checkpoints(trainer, cfg)
    assert trainer.callbacks == original
