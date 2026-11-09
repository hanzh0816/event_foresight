"""CPU checks of current targets, retained coordinates and resumable head training."""

from copy import deepcopy

from lightning.pytorch import Trainer
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from foresight.current_head_training import CurrentHeadTask, CurrentRGBDataset
from foresight.data.rgb import collate_rgb, preprocess_rgb
from foresight.diagnose_readout import CurrentDetectionEvaluator, load_real_head, retained_coordinates
from foresight.train import resume_config
from test_data import rgb_fixture
from test_training import TinyCriterion, TinyForecaster, config, cpu_trainer, sample


def components():
    forecaster = TinyForecaster()
    forecaster.pca.register_buffer("pca_components", forecaster.pca.weight.detach().clone())
    detector = nn.Linear(4, 1)
    detector.architecture = {"feature_dim": 4, "grid_size": [1, 1]}
    return forecaster, detector


def task_for(tmp_path, cls=CurrentHeadTask):
    forecaster, detector = components()
    return cls(config(tmp_path), forecaster=forecaster, detector=detector, criterion=TinyCriterion())


def test_current_dataset_preserves_inventory_and_empty_current_labels(rgb_fixture):
    root, _, _, _ = rgb_fixture
    with CurrentRGBDataset(root, image_size=(28, 42)) as dataset:
        assert dataset.windows == [("seq", frame) for frame in range(39, 44)]
        first, second = dataset[0], dataset[1]
        assert first["meta"]["target_id"] == "seq/000039"
        assert first["target"]["boxes"].shape == (0, 4)
        assert second["meta"]["target_id"] == "seq/000040"
        assert second["target"]["labels"].tolist() == [2]
        assert first["history_rgb"].shape == (1, 3, 28, 42)
        assert torch.equal(first["history_rgb"][0], preprocess_rgb(dataset._reader("seq").read_image(39).image, (28, 42)))
        coco, _ = CurrentDetectionEvaluator(dataset).to_coco()
        assert len(coco["images"]) == 5 and len(coco["annotations"]) == 4
        assert {image["target_id"] for image in coco["images"]} == {f"seq/{i:06d}" for i in range(39, 44)}


def test_current_head_uses_only_latest_rgb_and_updates_only_fp32_head(tmp_path):
    task = task_for(tmp_path)
    history = torch.stack([sample(1)["history_rgb"], sample(2)["history_rgb"]])
    captured = []
    hook = task.detector.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    output = task(history)
    assert len(task.forecaster.seen) == 2  # One-image encoder chunks.
    assert torch.equal(torch.cat(task.forecaster.seen), history[:, -1])
    with torch.no_grad():
        expected = retained_coordinates(task.forecaster.encode_latents(history[:, -1]), task.forecaster.pca)
    torch.testing.assert_close(captured[0], expected)
    changed = history.clone()
    changed[:, :-1] += 100
    assert torch.equal(task(changed), output)
    output.square().mean().backward()
    assert output.dtype == torch.float32
    assert all(p.grad is None and not p.requires_grad for p in task.forecaster.parameters())
    assert all(p.grad is not None and p.grad.dtype == torch.float32 for p in task.detector.parameters())
    assert not task.forecaster.training and task.detector.training
    hook.remove()


def test_current_head_save_load_rejects_forecast_coordinates_and_changed_pca(tmp_path):
    task = task_for(tmp_path)
    saved = {"state_dict": task.state_dict()}
    task.on_save_checkpoint(saved)
    path = tmp_path / "current.ckpt"
    torch.save(saved, path)
    head = load_real_head(path, task.detector, pca=task.forecaster.pca)
    assert not head.training and all(not p.requires_grad for p in head.parameters())
    assert all(torch.equal(value, head.state_dict()[key]) for key, value in task.detector.state_dict().items())
    changed_pca = deepcopy(task.forecaster.pca)
    changed_pca.pca_components.add_(1)
    with pytest.raises(ValueError, match="PCA coordinates"):
        load_real_head(path, task.detector, pca=changed_pca)
    wrong = deepcopy(saved)
    wrong["input_spec"].pop("readout_feature_source")
    torch.save(wrong, path)
    with pytest.raises(ValueError, match="retained-coordinate"):
        load_real_head(path, task.detector)
    with pytest.raises(ValueError, match="protocol"):
        task.on_load_checkpoint(wrong)


class TinyCurrentTask(CurrentHeadTask):
    def train_dataloader(self):
        records = [{**sample(index), "target": {"value": torch.tensor(index / 10)},
                    "meta": {"sample_id": str(index)}} for index in range(6)]
        return DataLoader(records, batch_size=2, shuffle=False, collate_fn=collate_rgb)


def test_current_head_full_checkpoint_restores_optimizer_and_loads_as_readout(tmp_path):
    task = task_for(tmp_path, TinyCurrentTask)
    frozen = {key: value.clone() for key, value in task.forecaster.state_dict().items()}
    trainer = cpu_trainer(tmp_path)
    trainer.fit(task)
    path = tmp_path / "last.ckpt"
    trainer.save_checkpoint(path)
    saved = torch.load(path, weights_only=False)
    assert saved["input_spec"]["readout_feature_source"] == "rgb_current_retained"
    assert saved["input_spec"]["prediction_ms"] == 0
    loaded = load_real_head(path, task.detector, pca=task.forecaster.pca)
    assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in task.detector.state_dict().items())
    relocated = config(tmp_path)
    relocated["data"]["root"] = "/new/DSEC"
    restored_config = resume_config(relocated, path, "detector")
    assert restored_config["data"]["root"] == "/new/DSEC"
    assert restored_config["model"]["detector_checkpoint"] is None
    restored = task_for(tmp_path, TinyCurrentTask)
    resumed = Trainer(accelerator="cpu", devices=1, max_steps=3, max_epochs=3,
                      logger=False, enable_checkpointing=False, enable_progress_bar=False,
                      enable_model_summary=False, num_sanity_val_steps=0, limit_val_batches=0)
    resumed.fit(restored, ckpt_path=path)
    assert resumed.global_step == 3
    assert all(int(state["step"]) == 3 for state in resumed.optimizers[0].state.values())
    assert all(torch.equal(value, restored.forecaster.state_dict()[key]) for key, value in frozen.items())


def test_current_head_launcher_portable_paths_and_resume_guard(monkeypatch):
    from foresight import train_current_head
    monkeypatch.setattr(train_current_head, "run", lambda config, **kwargs: (config, kwargs))
    cfg, args = train_current_head.main(["--data-root", "/server/DSEC", "--dinof-checkpoint", "/server/dinof.pt",
        "--epochs", "2", "--max-steps", "3", "--output-dir", "outputs/current", "--seed", "7"])
    assert cfg["data"]["root"] == "/server/DSEC" and cfg["model"]["dinof_checkpoint"] == "/server/dinof.pt"
    assert cfg["model"]["detector_checkpoint"] is None and cfg["runtime"]["seed"] == 7
    assert args["epochs"] == 2 and args["max_steps"] == 3
    with pytest.raises(SystemExit):
        train_current_head.main(["--resume", "current.ckpt", "--seed", "7", "--output-dir", "outputs/current"])
