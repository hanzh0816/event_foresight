"""v3.1 gradient, supervision, initialization, and optimizer contracts.

These tests use small CPU tensors and no DINO/RVT assets. Run them on the test
server; local execution was stopped at the user's request.
"""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from foresight.models.checkpoints import TRAINING_FORMAT, load_predictor, load_rgb_initializer
from foresight.training.losses import event_alignment_loss
from foresight.models.predictor import DINOWorldPredictor
from foresight.training import PredictorTask
from foresight.timing import ForecastTiming


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


class TinyEvents(nn.Module):
    output_grid = (1, 2)

    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(4, 8, 1)

    def forward(self, events):
        b, t = events.shape[:2]
        features = F.adaptive_avg_pool2d(self.projection(events.flatten(0, 1)), self.output_grid)
        return features.flatten(2).transpose(1, 2).reshape(b, t, 2, 8)


class TinyEventForecaster(nn.Module):
    image_size = (14, 28)
    history_length = 3
    feature_dim = 8
    grid_size = (1, 2)
    feature_space = "tiny-v31-frozen-features"

    def __init__(self, *, event_enabled=True, gate_init=0.1):
        super().__init__()
        self.encoder = nn.Linear(3, self.feature_dim)
        self.event_encoder = TinyEvents() if event_enabled else None
        self.predictor = DINOWorldPredictor(
            grid_size=self.grid_size, history_length=self.history_length,
            feature_dim=self.feature_dim, hidden_dim=64, depth=2, heads=1,
            mlp_ratio=2., query_chunk_size=0, event_enabled=event_enabled, gate_init=gate_init)
        self.architecture = dict(model="tiny-v31", image_size=list(self.image_size),
                                 feature_space=self.feature_space,
                                 predictor=deepcopy(self.predictor.architecture),
                                 event_encoder="tiny" if event_enabled else None)

    def encode_latents(self, rgb):
        pixels = F.adaptive_avg_pool2d(rgb, self.grid_size).flatten(2).transpose(1, 2)
        return self.encoder(pixels)

    def encode_events(self, events):
        return self.event_encoder(events)


def recipe(tmp_path, *, event_enabled=True, align_weight=0.1):
    return dict(
        model=dict(encoder_checkpoint="unused.pth", event=dict(enabled=event_enabled)),
        data=dict(image_size=[14, 28], context_frames=3, sample_history_frames=40, batch_size=2),
        optimization=dict(lr=1e-4, event_lr=2e-4, weight_decay=0.05, scheduler="constant",
                          align_weight=align_weight, align_cosine_weight=0.2, predictor_dropout=False,
                          activation_checkpointing=False),
        runtime=dict(seed=42, precision="32-true"), output_dir=str(tmp_path))


def batch():
    return dict(history_rgb=torch.randn(2, 3, 3, 14, 28, requires_grad=True),
                future_rgb_target=torch.randn(2, 3, 14, 28, requires_grad=True),
                history_events=torch.randn(2, 3, 4, 2, 4, requires_grad=True),
                event_activity=torch.ones(2, 3, 2, dtype=torch.bool))


def parameter_gradient_norm(module):
    return sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None)


def test_prediction_loss_alone_trains_event_encoder_through_nonzero_gates(tmp_path):
    torch.manual_seed(211)
    task = PredictorTask(recipe(tmp_path, align_weight=0), forecaster=TinyEventForecaster())
    inputs = batch()
    losses, prediction, target = task.feature_outputs(inputs)
    assert prediction.shape == target.shape == (2, 3, 2, 8)
    torch.testing.assert_close(losses['prediction'], F.smooth_l1_loss(prediction, target, beta=1.0))
    torch.testing.assert_close(losses['total'], losses['prediction'])
    losses['total'].backward()
    assert parameter_gradient_norm(task.forecaster.event_encoder) > 0
    assert all(block.event_attention.kv.weight.grad.abs().sum() > 0 for block in task.forecaster.predictor.blocks)
    assert all(block.gamma.grad.abs() > 0 for block in task.forecaster.predictor.blocks)
    assert inputs['history_events'].grad.abs().sum() > 0
    assert inputs['history_rgb'].grad is None and inputs['future_rgb_target'].grad is None
    assert all(not p.requires_grad and p.grad is None for p in task.forecaster.encoder.parameters())


def test_alignment_provides_event_gradients_when_prediction_gate_starts_at_zero(tmp_path):
    torch.manual_seed(223)
    task = PredictorTask(recipe(tmp_path), forecaster=TinyEventForecaster(gate_init=0))
    inputs = batch()
    losses, _, _ = task.feature_outputs(inputs)
    task.zero_grad(set_to_none=True)
    losses['prediction'].backward(retain_graph=True)
    assert parameter_gradient_norm(task.forecaster.event_encoder) == 0
    assert any(block.gamma.grad.abs() > 0 for block in task.forecaster.predictor.blocks)
    task.zero_grad(set_to_none=True)
    losses['alignment'].backward()
    assert parameter_gradient_norm(task.forecaster.event_encoder) > 0
    assert all(p.grad is None for p in task.forecaster.predictor.parameters())
    assert all(p.grad is None for p in task.forecaster.encoder.parameters())


def test_empty_activity_returns_finite_differentiable_zero_and_no_teacher_gradient():
    events = torch.randn(2, 3, 4, 8, requires_grad=True)
    teacher = torch.randn_like(events, requires_grad=True)
    activity = torch.zeros(2, 3, 4, dtype=torch.bool)
    loss = event_alignment_loss(events, teacher, activity)
    assert torch.isfinite(loss) and loss.item() == 0.0
    loss.backward()
    assert events.grad is not None and torch.count_nonzero(events.grad) == 0
    assert teacher.grad is None


def test_alignment_normalizes_each_sample_time_and_masks_inactive_gradients():
    events = torch.zeros(1, 3, 4, 2, requires_grad=True)
    teacher = torch.tensor([[[[1., 1.]] * 4, [[3., 3.]] * 4, [[20., 20.]] * 4]],
                           requires_grad=True)
    activity = torch.tensor([[[True, False, False, False], [True, True, False, False],
                              [False, False, False, False]]])
    loss = event_alignment_loss(events, teacher, activity, cosine_weight=0)
    # SmoothL1=0.5 for the first time, 2.5 for the second; the empty third is zero.
    torch.testing.assert_close(loss, torch.tensor((0.5 + 2.5 + 0.0) / 3))
    loss.backward()
    assert torch.count_nonzero(events.grad[~activity]) == 0
    assert events.grad[activity].abs().sum() > 0
    assert teacher.grad is None


def test_auxiliary_alignment_uses_current_history_not_shifted_future_targets(tmp_path):
    torch.manual_seed(227)
    task = PredictorTask(recipe(tmp_path), forecaster=TinyEventForecaster())
    inputs = batch()
    losses, _, _ = task.feature_outputs(inputs)
    changed = dict(inputs, future_rgb_target=inputs['future_rgb_target'] + 10)
    changed_losses, _, _ = task.feature_outputs(changed)
    torch.testing.assert_close(changed_losses['alignment'], losses['alignment'], rtol=0, atol=0)
    assert not torch.equal(changed_losses['prediction'], losses['prediction'])
    with torch.no_grad():
        events = task.forecaster.encode_events(inputs['history_events'])
        current = task._encode_history(inputs['history_rgb'])
        expected = event_alignment_loss(events, current, inputs['event_activity'])
    torch.testing.assert_close(losses['alignment'], expected)


def test_alignment_requires_count_activity_only_when_enabled(tmp_path):
    inputs = batch()
    del inputs['event_activity']
    disabled = PredictorTask(recipe(tmp_path, align_weight=0), forecaster=TinyEventForecaster())
    assert torch.isfinite(disabled.feature_loss(inputs))
    enabled = PredictorTask(recipe(tmp_path), forecaster=TinyEventForecaster())
    with pytest.raises(ValueError, match='event_activity'):
        enabled.feature_loss(inputs)


def test_joint_optimizer_assigns_every_parameter_once_to_correct_learning_rate(tmp_path):
    task = PredictorTask(recipe(tmp_path), forecaster=TinyEventForecaster())
    optimizer = task.configure_optimizers()
    assert optimizer.defaults['betas'] == (0.9, 0.95)
    assert [group['lr'] for group in optimizer.param_groups] == [1e-4, 2e-4]
    grouped = [p for group in optimizer.param_groups for p in group['params']]
    assert len(grouped) == len({id(p) for p in grouped})
    assert {id(p) for p in grouped} == {id(p) for p in task.trainable_parameters()}
    assert {id(p) for p in optimizer.param_groups[1]['params']} == {
        id(p) for p in task.forecaster.event_encoder.parameters()}
    assert not ({id(p) for p in grouped} & {id(p) for p in task.forecaster.encoder.parameters()})


def test_cosine_scheduler_uses_same_absolute_minimum_for_event_and_predictor(tmp_path):
    cfg = recipe(tmp_path)
    cfg['optimization'].update(scheduler='cosine', min_lr=1e-6, warmup_fraction=0.05)
    task = PredictorTask(cfg, forecaster=TinyEventForecaster())
    task._trainer = SimpleNamespace(estimated_stepping_batches=20)
    configured = task.configure_optimizers()
    scheduler = configured['lr_scheduler']['scheduler']
    for group, schedule in zip(configured['optimizer'].param_groups, scheduler.lr_lambdas, strict=True):
        assert group['initial_lr'] * schedule(19) == pytest.approx(1e-6)


def rgb_checkpoint(tmp_path):
    torch.manual_seed(229)
    rgb = TinyEventForecaster(event_enabled=False).eval()
    payload = dict(foresight_format=TRAINING_FORMAT, stage='predictor',
                   forecaster_architecture=deepcopy(rgb.architecture),
                   state_dict={f'forecaster.{name}': value.clone() for name, value in rgb.state_dict().items()})
    path = tmp_path / 'rgb.ckpt'
    torch.save(payload, path)
    return rgb, path, payload


def test_rgb_transfer_preserves_predictions_and_only_adds_zero_gated_event_modules(tmp_path):
    rgb, path, _ = rgb_checkpoint(tmp_path)
    event = TinyEventForecaster().eval()
    event.encoder.load_state_dict(rgb.encoder.state_dict())
    original_event_weights = deepcopy(event.event_encoder.state_dict())
    load_rgb_initializer(event, path)
    for name, value in rgb.predictor.state_dict().items():
        torch.testing.assert_close(event.predictor.state_dict()[name], value, rtol=0, atol=0)
    assert all(block.gamma.item() == 0 for block in event.predictor.blocks)
    for name, value in event.event_encoder.state_dict().items():
        torch.testing.assert_close(value, original_event_weights[name], rtol=0, atol=0)
    history = torch.randn(2, 3, 2, 8)
    events = torch.randn(2, 3, 2, 8)
    timing = ForecastTiming.from_config(recipe(tmp_path)).sequence_kwargs()
    torch.testing.assert_close(event.predictor.predict_sequence(history, event_context=events, **timing),
                               rgb.predictor.predict_sequence(history, **timing), rtol=0, atol=0)


@pytest.mark.parametrize('corruption', ['shared_weight', 'encoder', 'architecture'])
def test_rgb_transfer_rejects_incompatible_shared_contract_before_loading(tmp_path, corruption):
    rgb, path, payload = rgb_checkpoint(tmp_path)
    event = TinyEventForecaster().eval()
    event.encoder.load_state_dict(rgb.encoder.state_dict())
    before = deepcopy(event.predictor.state_dict())
    if corruption == 'shared_weight':
        del payload['state_dict']['forecaster.predictor.context_proj.0.weight']
    elif corruption == 'encoder':
        payload['state_dict']['forecaster.encoder.weight'].add_(1)
    else:
        payload['forecaster_architecture']['predictor']['depth'] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_rgb_initializer(event, path)
    for name, value in event.predictor.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_event_checkpoint_restores_predictor_and_event_encoder_together(tmp_path):
    source = TinyEventForecaster()
    restored = TinyEventForecaster()
    restored.encoder.load_state_dict(source.encoder.state_dict())
    payload = dict(foresight_format=TRAINING_FORMAT, stage='predictor',
                   forecaster_architecture=deepcopy(source.architecture),
                   state_dict={f'forecaster.{name}': value.clone() for name, value in source.state_dict().items()})
    path = tmp_path / 'event.ckpt'
    torch.save(payload, path)
    load_predictor(restored.predictor, path, forecaster=restored)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)


def test_rgb_finetune_optimizer_separates_backbone_fusion_and_rvt(tmp_path):
    rgb, path, _ = rgb_checkpoint(tmp_path)
    forecaster = TinyEventForecaster()
    forecaster.encoder.load_state_dict(rgb.encoder.state_dict())
    cfg = recipe(tmp_path)
    cfg['model']['rgb_checkpoint'] = str(path)
    task = PredictorTask(cfg, forecaster=forecaster)
    optimizer = task.configure_optimizers()
    assert [group['lr'] for group in optimizer.param_groups] == [2e-5, 1e-4, 1e-4]
    shared, fusion, event = [{id(p) for p in group['params']} for group in optimizer.param_groups]
    assert not (shared & fusion or shared & event or fusion & event)
    assert shared | fusion | event == {id(p) for p in task.trainable_parameters()}
    expected_fusion = {id(p) for name, p in forecaster.predictor.named_parameters()
                       if 'event_' in name or 'gamma' in name}
    assert fusion == expected_fusion
    assert event == {id(p) for p in forecaster.event_encoder.parameters()}
