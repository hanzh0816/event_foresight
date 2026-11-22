"""Strict, portable restoration for the paired motion-path diagnostic.

Evaluation execution knobs belong to the caller's DataLoader/autocast context.
The task retains the checkpoint's complete training recipe, so restoration does
not need a relaxed recipe check or the original teacher/initialization files.
"""

from copy import deepcopy
from pathlib import Path
import re

from .evaluate import module_digest
from .task import ResearchTask, file_sha256, read_checkpoint


def load_diagnostic_task(checkpoint, *, data_root=None, dinof_checkpoint=None,
                         detector_checkpoint=None, forecaster=None):
    """Return ``(task, provenance)`` for an explicit Event-motion checkpoint.

    Only relocatable asset paths may be overridden. Batch size, workers,
    precision and device should be passed to the diagnostic evaluator rather
    than written into ``task.config``. Both frozen and adapted transitions are
    supported. Source-file teacher provenance may be absent in older artifacts;
    the embedded teacher and active transition always receive content hashes.

    ``forecaster`` is an injection point for tests. Production callers provide
    the DINO asset path; its state is then strictly restored from the complete
    research checkpoint, preserving the trained latent coordinate system.
    """
    checkpoint = Path(checkpoint).expanduser().resolve()
    saved = read_checkpoint(checkpoint)
    config = deepcopy(saved["config"])
    options = config.get("model", {})
    if (config.get("stage") != "motion" or
            options.get("mode") != "explicit" or
            options.get("motion_input", "event") != "event" or
            options.get("copy_last_motion", False)):
        raise ValueError("Diagnostic requires a motion/explicit/Event checkpoint with copy_last_motion=False")
    original_config = deepcopy(config)
    relocations = {}
    for section, key, value in (("data", "root", data_root),
                                ("model", "dinof_checkpoint", dinof_checkpoint),
                                ("model", "detector_checkpoint", detector_checkpoint)):
        if value is not None:
            replacement = str(Path(value).expanduser().resolve())
            relocations[f"{section}.{key}"] = dict(original=config[section].get(key), current=replacement)
            config[section][key] = replacement

    # This is the same strict restoration sequence as load_task, using the
    # already-read checkpoint to avoid a second large CPU deserialization.
    task = ResearchTask(config, forecaster=forecaster, saved=saved)
    task.on_load_checkpoint(saved)
    task.load_state_dict(saved["state_dict"], strict=True)
    teacher = task.model.teacher
    expected_dimensions = dict(grid_size=list(task.forecaster.grid_size),
                               scene_dim=task.forecaster.pca_dim,
                               motion_dim=config["model"]["motion_dim"],
                               hidden_dim=config["model"]["hidden_dim"],
                               bins=config["data"]["bins"])
    for key, expected in expected_dimensions.items():
        actual = teacher.model_spec()["kwargs"][key]
        if actual != expected:
            raise ValueError(f"Embedded teacher architecture differs from training configuration/forecaster: {key}")
    task.eval().requires_grad_(False)
    active_transition = task.model.adapted_transition if task.model.train_transition else teacher.scene_transition
    source = deepcopy(saved.get("teacher_provenance"))
    source_hash = source.get("sha256") if isinstance(source, dict) else None
    source_hash_valid = isinstance(source_hash, str) and re.fullmatch(r"[0-9a-fA-F]{64}", source_hash) is not None
    provenance = dict(
        checkpoint_path=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
        checkpoint_git_commit=saved.get("git_commit"),
        checkpoint_global_step=saved.get("global_step"), checkpoint_epoch=saved.get("epoch"),
        training_config=original_config, restored_config=deepcopy(task.config),
        model_spec=deepcopy(saved["model_spec"]), training_protocol=deepcopy(saved["protocol"]),
        relocations=relocations,
        teacher_source_provenance=source,
        teacher_source_sha256_status="recorded_not_reopened" if source_hash_valid else "unavailable_or_invalid_in_checkpoint",
        teacher_state_sha256=module_digest(teacher),
        event_encoder_state_sha256=module_digest(teacher.event_encoder),
        frozen_transition_state_sha256=module_digest(teacher.scene_transition),
        active_transition_kind="adapted_transition" if task.model.train_transition else "frozen_teacher_transition",
        active_transition_state_sha256=module_digest(active_transition),
        forecaster_state_sha256=module_digest(task.forecaster),
        initialization_provenance=deepcopy(saved.get("initialization_provenance")),
    )
    return task, provenance
