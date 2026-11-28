"""CPU tests of stage boundaries and real Lightning resume; no model assets needed."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from foresight.training import CHECKPOINT_FORMAT, DetectorTask, PredictorTask
from foresight.data.samplers import ValidationSampler
from foresight.training.runner import make_trainer, resume_config, ValidateAtStepBudget
from foresight.timing import ForecastTiming


class TinyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.activation_checkpointing = False

    def predict_sequence(self, history, context_times, query_times, *, event_times=None):
        return self.proj(self.dropout(history))

    def forward(self, history, context_times, query_times, *, event_times=None):
        return self.predict_sequence(history, context_times, query_times, event_times=event_times)[:, -1]


class TinyForecaster(nn.Module):
    image_size = (2, 2)
    feature_dim = 2
    grid_size = (1, 1)
    history_length = 4
    feature_space = "tiny-last-layer"
    architecture = {"name": "tiny-world", "history_length": 4}

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3, 2), nn.Dropout(0.5))
        self.predictor = TinyPredictor()
        self.seen = []

    def encode_latents(self, rgb):
        self.seen.append(rgb.detach().clone())
        return self.encoder(rgb.mean((-1, -2)))[:, None]

    def forecast(self, history, *, context_times, query_times, event_times=None):
        b, t = history.shape[:2]
        encoded = self.encode_latents(history.flatten(0, 1)).reshape(b, t, 1, 2)
        return self.predictor(encoded, context_times, query_times, event_times=event_times)


class TinyEncoder(nn.Module):
    image_size = (2, 2)
    feature_dim = 2
    grid_size = (1, 1)
    feature_space = "tiny-last-layer"
    architecture = {"name": "tiny-encoder", "image_size": [2, 2], "feature_dim": 2}

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(3, 2), nn.Dropout(0.5))
        self.seen = []

    def forward(self, rgb):
        self.seen.append(rgb.detach().clone())
        return self.backbone(rgb.mean((-1, -2)))[:, None]


class TinyCriterion(nn.Module):
    def forward(self, outputs, targets, sync_num_boxes=False):
        assert sync_num_boxes
        target = torch.stack([item["value"] for item in targets])
        return {"loss": (outputs[:, 0, 0] - target).square().mean()}


def config(tmp_path):
    return {
        "data": {"root": str(tmp_path), "image_size": [2, 2], "context_frames": 4,
                 "sample_history_frames": 40, "batch_size": 2, "num_workers": 0},
        "model": {"encoder_checkpoint": "unused.pt", "predictor_checkpoint": None},
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


def test_predictor_uses_history_only_and_freezes_teacher(tmp_path, monkeypatch):
    forecaster = TinyForecaster()
    task = PredictorTask(config(tmp_path), forecaster=forecaster)
    task.train()
    batch = next(iter(DataLoader([sample(1), sample(2)], batch_size=2)))
    prediction = task(batch["history_rgb"])
    original_predict = task._predict

    def predict_after_teacher(history, events=None):
        assert torch.equal(forecaster.seen[-2], batch["history_rgb"].flatten(0, 1))
        assert torch.equal(forecaster.seen[-1], batch["future_rgb_target"])
        assert not history.requires_grad
        return original_predict(history, events)

    with monkeypatch.context() as patch:
        patch.setattr(task, "_predict", predict_after_teacher)
        loss = task.feature_loss(batch)
    loss.backward()
    assert loss.dtype == torch.float32
    assert all(p.grad is None and not p.requires_grad for p in forecaster.encoder.parameters())
    assert all(p.grad is not None and p.requires_grad for p in forecaster.predictor.parameters())
    assert not forecaster.encoder.training
    assert not forecaster.predictor.training
    assert torch.equal(forecaster.seen[-1], batch["future_rgb_target"])
    modified = dict(batch, future_rgb_target=batch["future_rgb_target"] + 20)
    assert torch.equal(prediction, task(modified["history_rgb"]))
    assert not torch.equal(loss, task.feature_loss(modified))
    with pytest.raises(ValueError, match="separate"):
        task.feature_loss({"history_rgb": batch["history_rgb"]})


def test_detector_updates_only_head(tmp_path):
    task = DetectorTask(config(tmp_path), encoder=TinyEncoder(),
                        detector=nn.Linear(2, 1), criterion=TinyCriterion())
    task.train()
    batch = next(iter(DataLoader([sample(1), sample(2)], batch_size=2)))
    outputs = task(batch["future_rgb_target"])
    outputs.square().mean().backward()
    assert all(not p.requires_grad and p.grad is None for p in task.encoder.parameters())
    assert all(p.requires_grad and p.grad is not None for p in task.detector.parameters())
    assert not task.encoder.training and task.detector.training
    assert not hasattr(task, "forecaster")
    assert all(key.startswith(("encoder.", "detector.")) for key in task.state_dict())
    assert torch.equal(task.encoder.seen[-1], batch["future_rgb_target"])

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
        ("forecaster_architecture", {}, "architecture"),
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


def test_trainer_uses_all_visible_gpus_and_configured_batch_limits(monkeypatch, tmp_path):
    monkeypatch.setattr("foresight.training.runner.torch.cuda.device_count", lambda: 2)
    monkeypatch.setattr("foresight.training.runner.Trainer", lambda **kwargs: kwargs)
    cfg = config(tmp_path)
    cfg["runtime"].update(limit_train_batches=2, limit_val_batches=0)
    kwargs = make_trainer(cfg, "predictor")
    assert kwargs["devices"] == "auto" and kwargs["accelerator"] == "gpu"
    assert kwargs["use_distributed_sampler"] is False
    assert kwargs["limit_train_batches"] == 2 and kwargs["limit_val_batches"] == 0
    assert kwargs["strategy"]._ddp_kwargs["broadcast_buffers"] is False
    first_logger = kwargs["logger"][0]
    first_logger.log_metrics({"loss": 1.0}, step=0)
    first_logger.save()
    second = make_trainer(cfg, "predictor")
    assert second["logger"][0].log_dir != first_logger.log_dir
    assert (tmp_path / "logs" / "version_0" / "metrics.csv").is_file()


def test_resume_recipe_preserves_budget_and_selection(monkeypatch, tmp_path):
    saved_config = config(tmp_path)
    saved_config["data"]["train_sample_ids"] = ["a", "b"]
    saved_config["optimization"]["lr"] = 2e-5
    saved_config["runtime"].update(seed=123, precision="bf16-mixed", encoder_chunk_size=2)
    saved_config["model"]["predictor_checkpoint"] = "/old/missing/init.ckpt"
    monkeypatch.setattr("foresight.models.checkpoints.read_checkpoint", lambda path: {
        "foresight_format": CHECKPOINT_FORMAT, "stage": "predictor", "config": saved_config})
    current = config(tmp_path)
    current["data"]["root"] = "/new/data"
    current["model"]["encoder_checkpoint"] = "/new/dinov2.pth"
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
    task = DetectorTask(cfg, encoder=TinyEncoder(), criterion=TinyCriterion())
    payload = {"state_dict": task.state_dict()}
    task.on_save_checkpoint(payload)
    assert payload["encoder_architecture"] == task.encoder.architecture
    assert "forecaster_architecture" not in payload
    assert not any("predictor" in key for key in payload["state_dict"])
    architecture = payload["detector_architecture"]
    assert architecture["feature_dim"] == 2 and architecture["grid_size"] == [1, 1]
    assert architecture["hidden_dim"] == 8
    monkeypatch.setattr("foresight.models.checkpoints.read_checkpoint", lambda path: payload)
    restored_config = resume_config(config(tmp_path), "unused.ckpt", "detector")
    assert restored_config["detector"]["hidden_dim"] == 8
    restored = DetectorTask(restored_config, encoder=TinyEncoder(), criterion=TinyCriterion())
    restored.on_load_checkpoint(payload)
    restored.load_state_dict(payload["state_dict"], strict=True)
    assert all(torch.equal(task.state_dict()[key], value) for key, value in restored.state_dict().items())


def test_teacher_forcing_supervises_every_shifted_frame(tmp_path):
    task = PredictorTask(config(tmp_path), forecaster=TinyForecaster())
    batch = next(iter(DataLoader([sample(1), sample(2)], batch_size=2)))
    with torch.no_grad():
        history = task._encode_history(batch["history_rgb"])
        future = task.forecaster.encode_latents(batch["future_rgb_target"])
        target = torch.cat((history[:, 1:], future[:, None]), dim=1)
        expected = torch.nn.functional.smooth_l1_loss(
            task.forecaster.predictor.predict_sequence(
                history, **ForecastTiming.from_config(task.config).sequence_kwargs()), target, beta=1.0)
    actual = task.feature_loss(batch)
    torch.testing.assert_close(actual, expected)
    assert task(batch["history_rgb"]).shape == (2, 4, 1, 2)


def test_shared_detector_never_constructs_a_forecaster(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    def forbidden_forecaster(*args, **kwargs):
        raise AssertionError("Shared detector training must not construct a forecaster")
    monkeypatch.setattr("foresight.models.forecaster.DINOWorldForecaster", forbidden_forecaster)
    monkeypatch.setattr("foresight.models.encoder.DINOFeatureEncoder", lambda *args, **kwargs: TinyEncoder())
    task = DetectorTask(cfg, detector=nn.Linear(2, 1), criterion=TinyCriterion())
    result = task(torch.randn(2, 3, 2, 2))
    assert result.shape == (2, 1, 1)


def test_cosine_schedule_uses_five_percent_warmup_and_absolute_minimum(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg["optimization"].update(scheduler="cosine", warmup_fraction=0.05, min_lr=1e-6)
    task = PredictorTask(cfg, forecaster=TinyForecaster())
    task._trainer = SimpleNamespace(estimated_stepping_batches=100)
    configured = task.configure_optimizers()
    optimizer, scheduler = configured["optimizer"], configured["lr_scheduler"]["scheduler"]
    schedule = scheduler.lr_lambdas[0]
    base_lr = optimizer.param_groups[0]['initial_lr']
    assert [base_lr * schedule(step) for step in range(5)] == pytest.approx(
        [2e-4, 4e-4, 6e-4, 8e-4, 1e-3])
    assert base_lr * schedule(99) == pytest.approx(1e-6)
    assert schedule(5) > schedule(50) > schedule(98)
    monkeypatch.setattr("foresight.training.runner.Trainer", lambda **kwargs: kwargs)
    trainer = make_trainer(cfg, "predictor")
    callback = next(callback for callback in trainer["callbacks"] if isinstance(callback, ModelCheckpoint))
    assert callback.monitor == "val_feature_loss" and callback.mode == "min"
    assert callback.save_last and callback.save_top_k == 1


def test_predictor_initializer_is_only_used_by_predictor_stage(tmp_path):
    cfg = config(tmp_path)
    predictor_task = PredictorTask(cfg, forecaster=TinyForecaster())
    payload = {"state_dict": deepcopy(predictor_task.state_dict())}
    predictor_task.on_save_checkpoint(payload)
    checkpoint = tmp_path / "predictor.ckpt"
    torch.save(payload, checkpoint)
    cfg["model"]["predictor_checkpoint"] = str(checkpoint)
    fresh = deepcopy(predictor_task.forecaster)
    with torch.no_grad():
        for parameter in fresh.predictor.parameters():
            parameter.zero_()
    restored = PredictorTask(cfg, forecaster=fresh)
    for name, value in predictor_task.forecaster.state_dict().items():
        assert torch.equal(restored.forecaster.state_dict()[name], value)
    with pytest.raises(ValueError, match="without a predictor initializer"):
        DetectorTask(cfg, encoder=TinyEncoder(), detector=nn.Linear(2, 1), criterion=TinyCriterion())


@pytest.mark.parametrize("max_steps,max_epochs,accumulation", [(1, 3, 1), (1, 3, 2), (1, 1, 2), (-1, 1, 2)])
def test_budget_validates_and_saves_best_and_last(tmp_path, max_steps, max_epochs, accumulation):

    class TinyValidatedTask(TinyPredictorTask):
        def val_dataloader(self):
            return DataLoader([sample(6), sample(7)], batch_size=2)

    cfg = config(tmp_path)
    cfg["optimization"].update(max_steps=max_steps, epochs=max_epochs, accumulation_steps=accumulation)
    task = TinyValidatedTask(cfg, forecaster=TinyForecaster())
    best = ModelCheckpoint(dirpath=tmp_path / "checkpoints", filename="best", monitor="val_feature_loss",
                           mode="min", save_top_k=1, save_last=True, save_on_train_epoch_end=False)
    trainer = Trainer(accelerator="cpu", devices=1, max_steps=max_steps, max_epochs=max_epochs,
                      accumulate_grad_batches=accumulation,
                      callbacks=[ValidateAtStepBudget(), best], logger=False,
                      enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0)
    trainer.fit(task)
    expected_steps = 1 if max_steps == 1 else 2
    assert task.last_validation_step == trainer.global_step == expected_steps
    assert best.best_model_path.endswith("best.ckpt") and torch.isfinite(best.best_model_score)
    for filename in ("best.ckpt", "last.ckpt"):
        saved = torch.load(tmp_path / "checkpoints" / filename, map_location="cpu", weights_only=False)
        assert saved["global_step"] == expected_steps
        assert all(torch.equal(task.state_dict()[key], value) for key, value in saved["state_dict"].items())


def test_train_cli_uses_configured_stage_and_initializer(tmp_path, monkeypatch):
    from foresight import train
    checkpoint = tmp_path / "predictor.ckpt"
    recipe = config(tmp_path)
    recipe["stage"] = "predictor"
    recipe["model"]["predictor_checkpoint"] = str(checkpoint)
    monkeypatch.setattr(train, "load_config", lambda path: deepcopy(recipe))
    monkeypatch.setattr(train, "run", lambda cfg, **kwargs: (cfg, kwargs))
    cfg, options = train.main(["--config", "experiment.yaml"])
    assert cfg["model"]["predictor_checkpoint"] == str(checkpoint)
    assert options["stage"] == "predictor" and options["resume"] is None
    with pytest.raises(SystemExit):
        train.main(["--config", "experiment.yaml", "--stage", "predictor"])
    recipe.pop("stage")
    with pytest.raises(SystemExit):
        train.main(["--config", "experiment.yaml"])


def test_shared_head_enables_ap_selection_and_separate_foreground_checkpoint(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg['model']['detector_checkpoint'] = str(tmp_path / 'shared-head.ckpt')
    monkeypatch.setattr('foresight.training.runner.Trainer', lambda **kwargs: kwargs)
    trainer = make_trainer(cfg, 'predictor')
    checkpoints = [callback for callback in trainer['callbacks'] if isinstance(callback, ModelCheckpoint)]
    ap = next(callback for callback in checkpoints if callback.monitor == 'val_AP')
    foreground = next(callback for callback in checkpoints if callback.monitor == 'val_foreground_l1')
    assert ap.mode == 'max' and ap.save_last and ap.save_top_k == 1
    assert foreground.mode == 'min' and foreground.save_top_k == 1


def test_predictor_resume_restores_embedded_head_without_external_checkpoint(tmp_path, monkeypatch):
    from foresight.models.detection.detector import FutureDetector

    source = PredictorTask(config(tmp_path), forecaster=TinyForecaster())
    source.detector = FutureDetector(feature_dim=2, grid_size=(1, 1), hidden_dim=8,
                                     num_classes=8, num_queries=2, num_levels=1,
                                     encoder_layers=1, decoder_layers=1, heads=2,
                                     points=1, mlp_dim=16, dropout=0.0).eval().requires_grad_(False)
    source.config['model']['detector_checkpoint'] = str(tmp_path / 'missing-original-head.ckpt')
    payload = {'state_dict': deepcopy(source.state_dict())}
    source.on_save_checkpoint(payload)
    assert payload['detector_architecture'] == source.detector.architecture
    checkpoint = tmp_path / 'predictor-full.ckpt'
    torch.save(payload, checkpoint)
    restored_config = resume_config(config(tmp_path), checkpoint, 'predictor')
    assert restored_config['resume_detector_checkpoint'] == str(checkpoint)
    assert restored_config['model']['detector_checkpoint'] is None

    def external_head_forbidden(*args, **kwargs):
        raise AssertionError('Full resume must use its embedded validation head')

    monkeypatch.setattr('foresight.models.checkpoints.load_shared_detector', external_head_forbidden)
    restored = PredictorTask(restored_config, forecaster=TinyForecaster())
    restored.on_load_checkpoint(payload)
    restored.load_state_dict(payload['state_dict'], strict=True)
    restored.train()
    assert not restored.detector.training
    assert all(not parameter.requires_grad for parameter in restored.detector.parameters())
    for name, value in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    incompatible = deepcopy(payload)
    incompatible['detector_architecture']['num_queries'] += 1
    with pytest.raises(ValueError, match='validation head'):
        restored.on_load_checkpoint(incompatible)


@pytest.mark.parametrize('stage', ['predictor', 'detector'])
@pytest.mark.parametrize('split', ['train', 'val'])
def test_training_loaders_always_use_one_step_targets(tmp_path, monkeypatch, stage, split):
    cfg = config(tmp_path)
    # A rollout evaluation setting must not change teacher-forcing labels.
    cfg['data']['prediction_frames'] = 4
    cfg['evaluation'] = {'horizon_steps': 4}
    if stage == 'predictor':
        task = PredictorTask(cfg, forecaster=TinyForecaster())
    else:
        task = DetectorTask(cfg, encoder=TinyEncoder(), detector=nn.Linear(2, 1),
                            criterion=TinyCriterion())
    task._trainer = SimpleNamespace(world_size=1, global_rank=0)
    observed = []
    examples = [sample(1), sample(2)]

    def dataset_factory(configuration, requested_split, **kwargs):
        observed.append((requested_split, kwargs))
        return examples

    monkeypatch.setattr('foresight.data.dataset_from_config', dataset_factory)
    loader = task._loader(split)
    assert loader.dataset is examples
    assert observed == [(split, {'future_rgb': True, 'prediction_frames': 1,
                                 'target_only': stage == 'detector'})]
    assert task.config['data']['prediction_frames'] == 4
