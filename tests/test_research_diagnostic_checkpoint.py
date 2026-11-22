"""Portable diagnostics preserve checkpoint identity and strict architecture checks."""

import pytest
import torch

from foresight.research.diagnostic_checkpoint import load_diagnostic_task
from foresight.research.evaluate import module_digest
from foresight.research.task import load_task, read_checkpoint
from test_research_motion_training import trained_transition
from test_research_training import TinyForecaster, save_task, tiny_batch, tiny_config


def motion_checkpoint(tmp_path, *, adapted=False):
    torch.set_num_threads(1)
    _, teacher = trained_transition(tmp_path)
    config = tiny_config("motion", teacher_checkpoint=str(teacher), train_transition=adapted,
                         dinof_checkpoint="/old/server/dinof.pt", detector_checkpoint="/old/server/detector.pt")
    config["data"]["root"] = "/old/server/DSEC"
    config["data"]["batch_size"] = 1
    # Extra server-side scheduling metadata must survive unchanged.
    config["optimization"]["save_every_n_train_steps"] = 250
    task = load_task(config, forecaster=TinyForecaster())
    optimizer = task.configure_optimizers()
    optimizer.zero_grad()
    task.compute_losses(tiny_batch())["loss"].backward()
    optimizer.step()
    checkpoint = tmp_path / "motion.ckpt"
    save_task(task, checkpoint)
    teacher.unlink()
    return task.eval(), checkpoint


@pytest.mark.parametrize("adapted", [False, True])
def test_embedded_teacher_restores_without_source_file_and_records_identities(tmp_path, adapted):
    original, checkpoint = motion_checkpoint(tmp_path, adapted=adapted)
    restored, provenance = load_diagnostic_task(checkpoint, forecaster=TinyForecaster(),
                                                data_root=tmp_path / "DSEC", dinof_checkpoint=tmp_path / "dinof.pt",
                                                detector_checkpoint=tmp_path / "detector.pt")
    assert all(not p.requires_grad for p in restored.parameters())
    assert not restored.training
    assert restored.config["data"]["batch_size"] == 1
    assert restored.config["optimization"]["save_every_n_train_steps"] == 250
    assert restored.config["data"]["root"] == str(tmp_path / "DSEC")
    assert provenance["training_config"]["data"]["root"] == "/old/server/DSEC"
    assert provenance["teacher_source_sha256_status"] == "recorded_not_reopened"
    assert provenance["teacher_state_sha256"] == module_digest(original.model.teacher)
    assert provenance["forecaster_state_sha256"] == module_digest(original.forecaster)
    transition = original.model.adapted_transition if adapted else original.model.teacher.scene_transition
    assert provenance["active_transition_state_sha256"] == module_digest(transition)
    assert provenance["active_transition_kind"] == ("adapted_transition" if adapted else "frozen_teacher_transition")
    if adapted:
        assert provenance["active_transition_state_sha256"] != provenance["frozen_transition_state_sha256"]
    batch = tiny_batch()
    with torch.no_grad():
        assert torch.equal(original.predict_batch(batch)["scene"], restored.predict_batch(batch)["scene"])


def test_old_checkpoint_without_source_teacher_hash_remains_identifiable(tmp_path):
    _, checkpoint = motion_checkpoint(tmp_path)
    saved = read_checkpoint(checkpoint)
    saved["teacher_provenance"] = None
    torch.save(saved, checkpoint)
    _, provenance = load_diagnostic_task(checkpoint, forecaster=TinyForecaster())
    assert provenance["teacher_source_provenance"] is None
    assert provenance["teacher_source_sha256_status"] == "unavailable_or_invalid_in_checkpoint"
    assert len(provenance["teacher_state_sha256"]) == 64


@pytest.mark.parametrize("tamper", ["spec", "protocol", "weights", "teacher_config"])
def test_checkpoint_structure_or_protocol_tampering_is_rejected(tmp_path, tamper):
    _, checkpoint = motion_checkpoint(tmp_path)
    saved = read_checkpoint(checkpoint)
    if tamper == "spec":
        saved["model_spec"]["kwargs"]["history_frames"] += 1
    elif tamper == "protocol":
        saved["protocol"]["target_event_usage"] = "forecast_input"
    elif tamper == "weights":
        del saved["state_dict"]["model.motion_head.weight"]
    else:
        saved["config"]["model"]["motion_dim"] += 1
    torch.save(saved, checkpoint)
    with pytest.raises((ValueError, RuntimeError), match="architecture|protocol|state_dict"):
        load_diagnostic_task(checkpoint, forecaster=TinyForecaster())


@pytest.mark.parametrize("field,value", [("mode", "direct"), ("motion_input", "rgb"), ("copy_last_motion", True)])
def test_nonmatched_motion_paths_are_rejected(tmp_path, field, value):
    _, checkpoint = motion_checkpoint(tmp_path)
    saved = read_checkpoint(checkpoint)
    saved["config"]["model"][field] = value
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="motion/explicit/Event"):
        load_diagnostic_task(checkpoint, forecaster=TinyForecaster())


def test_missing_initialization_source_is_not_reopened(tmp_path):
    _, checkpoint = motion_checkpoint(tmp_path)
    saved = read_checkpoint(checkpoint)
    saved["config"]["model"]["init_checkpoint"] = "/absent/server/initialization.ckpt"
    saved["initialization_provenance"] = dict(path="/absent/server/initialization.ckpt", sha256="e" * 64)
    torch.save(saved, checkpoint)
    task, provenance = load_diagnostic_task(checkpoint, forecaster=TinyForecaster())
    assert task.initialization_provenance == saved["initialization_provenance"]
    assert provenance["initialization_provenance"] == saved["initialization_provenance"]
