import json

import pytest
import torch
from torch import nn

from foresight.research.diagnostics import TransitionDiagnostics


class AnalyticTransition(nn.Module):
    def __init__(self, ignores_motion=False):
        super().__init__()
        self.ignores_motion = ignores_motion

    def encode_motion(self, events):
        # B,S,1,2,1,1 -> B,S,1,2: each polarity is an analytic motion channel.
        return events[:, :, 0, :, 0, 0].unsqueeze(2)

    def transition(self, scene, motion):
        return scene if self.ignores_motion else scene + motion


def fixture():
    starts = torch.tensor([[[[10., 20.]], [[100., 200.]]],
                           [[[30., 40.]], [[300., 400.]]]])
    motion = torch.tensor([[[[1., 2.]], [[3., 4.]]],
                           [[[5., 6.]], [[7., 8.]]]])
    events = motion.squeeze(2)[:, :, None, :, None, None]
    return starts, events, starts + motion, motion


def test_true_start_statistics_and_batch_partition_invariance():
    starts, events, targets, motion = fixture()
    model = AnalyticTransition().train()
    whole, split = TransitionDiagnostics(), TransitionDiagnostics()
    whole.update(model, starts, events, targets)
    assert model.training  # A diagnostic must not leave training in eval mode.
    for index in range(2):
        split.update(model, starts[index:index+1], events[index:index+1], targets[index:index+1])
    result = whole.compute()
    assert result == split.compute()
    assert result["sample_count"] == 2 and result["interval_count"] == 4
    assert result["teacher_one_step_l1"] == 0.0
    assert result["persistence_l1"] == result["zero_motion_l1"] == 4.5
    assert result["real_vs_zero_prediction_l1"] == 4.5
    expected = motion.var(dim=(0, 1, 2), correction=0)
    torch.testing.assert_close(torch.tensor(result["motion_token_statistics"]["variance_per_channel"]), expected)
    json.dumps(result, allow_nan=False)
    assert whole.motion_sum.device.type == "cpu" and whole.motion_sum.shape == (2,)


def test_ignoring_motion_is_detected_even_when_motion_variance_is_nonzero():
    starts, events, targets, _ = fixture()
    accumulator = TransitionDiagnostics()
    accumulator.update(AnalyticTransition(ignores_motion=True), starts, events, targets)
    result = accumulator.compute()
    assert result["motion_token_statistics"]["mean_channel_std"] > 0
    assert result["real_vs_zero_prediction_l1"] == 0
    assert result["teacher_one_step_l1"] == result["zero_motion_l1"] == result["persistence_l1"]


def test_empty_and_failed_updates_are_json_safe_and_do_not_change_counts():
    accumulator = TransitionDiagnostics()
    assert accumulator.compute()["teacher_one_step_l1"] is None
    json.dumps(accumulator.compute(), allow_nan=False)
    starts, events, targets, _ = fixture()
    events[0, 0, 0, 0, 0, 0] = float("nan")
    model = AnalyticTransition().train()
    with pytest.raises(FloatingPointError, match="non-finite"):
        accumulator.update(model, starts, events, targets)
    assert model.training and accumulator.compute()["sample_count"] == 0
