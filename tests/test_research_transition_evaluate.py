"""The teacher diagnostic is emitted by the real evaluation command pipeline."""

import json

import torch
from torch import nn

from test_research_data import research_recording
from test_research_evaluate import TinyDetector, TinyForecast, evaluation_config
from foresight.research import evaluate as ev
from foresight.research.transition import TransitionWorldModel


class TinyTransitionTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.forecaster = TinyForecast().requires_grad_(False)
        self.model = TransitionWorldModel((2, 3), scene_dim=8, motion_dim=4,
                                          hidden_dim=8, bins=5)
        torch.nn.init.normal_(self.model.scene_transition.output.weight, std=0.1)

    def encode_history(self, batch):
        rgb = batch["history_rgb"]
        return self.forecaster.encode_latents(rgb.flatten(0, 1)).reshape(len(rgb), 4, 6, 8)

    def predict_batch(self, batch):
        return self.model(self.encode_history(batch)[:, -1], batch["future_events_target"])


def test_transition_diagnostics_are_written_for_true_starts(research_recording, tmp_path, monkeypatch):
    config = evaluation_config(research_recording[0], tmp_path)
    config["stage"] = "transition"
    config["model"]["mode"] = "transition"
    config["data"]["rollout_steps"] = 2
    task = TinyTransitionTask().eval()
    monkeypatch.setattr(ev, "load_task", lambda *args, **kwargs: task)
    monkeypatch.setattr(ev, "load_detector", lambda *args, **kwargs: TinyDetector())
    report = ev.evaluate(config, checkpoint=config["model"]["dinof_checkpoint"],
                         split="train", output=tmp_path / "diagnostics.json",
                         feature_metrics=True, max_samples=2)
    diagnostic = report["transition_diagnostics"]
    assert report["evaluation_kind"] == "observed_transition_diagnostic"
    assert diagnostic["sample_count"] == 2
    assert diagnostic["interval_count"] == 4
    assert diagnostic["motion_token_count"] == 24
    assert diagnostic["real_vs_zero_prediction_l1"] > 0
    assert len(diagnostic["motion_token_statistics"]["std_per_channel"]) == 4
    assert "true interval start" in diagnostic["scene_protocol"]
    saved = json.loads((tmp_path / "diagnostics.json").read_text())
    assert saved["transition_diagnostics"] == diagnostic
