"""Lightning tasks for causal Event/RGB research and observed-transition diagnostics."""

from copy import deepcopy
import hashlib
import subprocess
from pathlib import Path

from lightning.pytorch import LightningModule
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import validate_config

CHECKPOINT_FORMAT = "event-foresight-research-v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def read_checkpoint(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(saved, dict) or saved.get("research_format") != CHECKPOINT_FORMAT:
        raise ValueError("Expected a complete Event research checkpoint")
    for key in ("state_dict", "config", "model_spec", "protocol"):
        if key not in saved:
            raise ValueError(f"Research checkpoint missing {key}")
    return saved


def _transition_from_spec(spec):
    from .transition import TransitionWorldModel
    if spec.get("kind") != "transition" or not isinstance(spec.get("kwargs"), dict):
        raise ValueError("Expected a transition model specification")
    return TransitionWorldModel(**spec["kwargs"])


def _recipe_signature(config):
    """Relocation and execution knobs may differ; scientific/optimizer knobs may not."""
    cfg = deepcopy(config)
    cfg.pop("output_dir", None)
    for key in ("root", "num_workers", "prefetch_factor", "persistent_workers"):
        cfg["data"].pop(key, None)
    for key in ("dinof_checkpoint", "teacher_checkpoint", "detector_checkpoint", "init_checkpoint", "predictor_checkpoint"):
        cfg["model"].pop(key, None)
    for key in ("accelerator", "devices", "cpu_threads", "log_every"):
        cfg["runtime"].pop(key, None)
    return cfg


class ResearchTask(LightningModule):
    """Frozen RGB encoder/PCA with optional full predictor adaptation.

    In forecast stages predict_batch only accesses history inputs. Target RGB
    and Event are read exclusively in compute_losses. Transition is explicitly
    an observed-interval diagnostic and uses target-interval Event observations.
    """

    def __init__(self, config, *, forecaster=None, model=None, saved=None):
        super().__init__()
        self.config = validate_config(config)
        self.stage = self.config["stage"]
        self.save_hyperparameters({"config": self.config})
        options, data = self.config["model"], self.config["data"]
        if forecaster is None:
            from ..models.forecaster import DINOForecaster
            forecaster = DINOForecaster(options["dinof_checkpoint"], encoder_chunk_size=options["encoder_chunk_size"])
        self.forecaster = forecaster.requires_grad_(False).eval()
        self.predictor_initialization = deepcopy(saved.get("predictor_initialization")) if saved else None
        if self.stage == "conditioning":
            if saved is None and options.get("predictor_checkpoint"):
                from ..checkpoints import load_predictor
                path = options["predictor_checkpoint"]
                load_predictor(forecaster.predictor, path, forecaster=forecaster)
                self.predictor_initialization = dict(path=str(Path(path).resolve()), sha256=file_sha256(path))
            forecaster.predictor.requires_grad_(options["train_predictor"])
            forecaster.predictor.activation_checkpointing = options["activation_checkpointing"]
        if tuple(data["image_size"]) != tuple(forecaster.image_size):
            raise ValueError("RGB image_size differs from the frozen forecaster")
        if data["context_frames"] != forecaster.history_length:
            raise ValueError("context_frames differs from the frozen forecaster")
        self.teacher_provenance = deepcopy(saved.get("teacher_provenance")) if saved else None
        self.initialization_provenance = deepcopy(saved.get("initialization_provenance")) if saved else None
        dimensions = dict(grid_size=forecaster.grid_size, scene_dim=forecaster.pca_dim,
                          motion_dim=options["motion_dim"], hidden_dim=options["hidden_dim"], bins=data["bins"])
        if model is not None:
            self.model = model
        elif self.stage == "conditioning":
            from .conditioning import ConditioningWorldModel
            self.model = ConditioningWorldModel(
                **dimensions, predictor_dim=forecaster.predictor.embed.out_features,
                history_frames=data["context_frames"], injection_layer=options["injection_layer"],
                attention_heads=options["attention_heads"], mode=options["mode"])
            depth = len(forecaster.predictor.maskvit.transformer.layers)
            if not 1 <= options["injection_layer"] < depth:
                raise ValueError("injection_layer must leave at least one predictor block after injection")
        elif self.stage == "fusion":
            from .fusion import FusionWorldModel
            self.model = FusionWorldModel(**dimensions, history_frames=data["context_frames"], mode=options["mode"])
        elif self.stage == "transition":
            from .transition import TransitionWorldModel
            self.model = TransitionWorldModel(**dimensions, temporal_mode=options["temporal_mode"], transition_mode=options["transition_mode"])
        else:
            from .motion import MotionWorldModel
            if saved is not None:
                teacher = _transition_from_spec(saved["model_spec"]["teacher"])
            else:
                teacher_file = options["teacher_checkpoint"]
                prior = read_checkpoint(teacher_file)
                if prior["config"]["stage"] != "transition":
                    raise ValueError("Motion teacher must be a transition-stage checkpoint")
                if prior["model_spec"].get("kwargs", {}).get("transition_mode") != "motion":
                    raise ValueError("Motion teacher must use a learned motion-conditioned transition")
                teacher = _transition_from_spec(prior["model_spec"])
                teacher.load_state_dict({k.removeprefix("model."): v for k, v in prior["state_dict"].items() if k.startswith("model.")}, strict=True)
                # G/T were learned in this exact frozen feature coordinate system.
                self.forecaster.load_state_dict({k.removeprefix("forecaster."): v for k, v in prior["state_dict"].items() if k.startswith("forecaster.")}, strict=True)
                if hasattr(self.forecaster, "provenance"):
                    self.forecaster.provenance = deepcopy(prior.get("forecaster_provenance", {}))
                for key in ("image_size", "event_image_size", "bins", "alignment", "alignment_evidence"):
                    if prior["config"]["data"].get(key) != data.get(key):
                        raise ValueError(f"Teacher/current data protocol differs: {key}")
                for key in ("grid_size", "scene_dim", "motion_dim", "hidden_dim", "bins"):
                    teacher_value, requested_value = prior["model_spec"]["kwargs"][key], dimensions[key]
                    matches = tuple(teacher_value) == tuple(requested_value) if key == "grid_size" else teacher_value == requested_value
                    if not matches:
                        raise ValueError(f"Teacher/current architecture differs: {key}")
                self.teacher_provenance = dict(path=str(Path(teacher_file).resolve()), sha256=file_sha256(teacher_file), git_commit=prior.get("git_commit"))
            self.model = MotionWorldModel(teacher, history_frames=data["context_frames"], mode=options["mode"],
                                          motion_input=options["motion_input"], copy_last_motion=options["copy_last_motion"],
                                          train_transition=options["train_transition"])
        if saved is None and options.get("init_checkpoint"):
            self._initialize_refinement(options["init_checkpoint"])
        self.datasets = {}
        self.last_validation_step = None
        self.protocol = dict(stage=self.stage, prediction_kind="observed_transition" if self.stage == "transition" else "history_only_forecast",
                             context_frames=data["context_frames"], rollout_steps=data["rollout_steps"], bins=data["bins"],
                             image_size=list(data["image_size"]), event_image_size=list(data["event_image_size"]),
                             feature_space="frozen_dinov2_fixed_pca", alignment=data.get("alignment"),
                             alignment_evidence=data.get("alignment_evidence"), target_event_usage="supervision_only" if self.stage == "motion" else "observed_interval_input" if self.stage == "transition" else "unused")

    def _initialize_refinement(self, path):
        """Weight-only continuation with a reset optimizer and explicit provenance."""
        source = read_checkpoint(path)
        if self.stage == "conditioning":
            if source["config"]["stage"] != "conditioning" or source["model_spec"] != self.model.model_spec():
                raise ValueError("Conditioning initialization must preserve architecture and mode")
            for key in ("image_size", "event_image_size", "context_frames", "bins", "rollout_steps",
                        "sample_history_frames", "tolerance_us", "alignment", "alignment_evidence"):
                if source["config"]["data"].get(key) != self.config["data"].get(key):
                    raise ValueError(f"Conditioning initialization changes the observation protocol: {key}")
            self.model.load_state_dict({key.removeprefix("model."): value
                                       for key, value in source["state_dict"].items() if key.startswith("model.")}, strict=True)
            self.forecaster.load_state_dict({key.removeprefix("forecaster."): value
                                            for key, value in source["state_dict"].items() if key.startswith("forecaster.")}, strict=True)
            self.forecaster.provenance = deepcopy(source.get("forecaster_provenance", {}))
            self.predictor_initialization = deepcopy(source.get("predictor_initialization"))
            self.initialization_provenance = dict(path=str(Path(path).resolve()), sha256=file_sha256(path),
                                                 git_commit=source.get("git_commit"), optimizer="reset",
                                                 source_global_step=source.get("global_step"),
                                                 source_optimization=deepcopy(source["config"]["optimization"]),
                                                 source_train_predictor=source["config"]["model"]["train_predictor"])
            return
        if self.stage != "motion" or source["config"]["stage"] != "motion":
            raise ValueError("init_checkpoint requires a motion-stage source and destination")
        previous, current = deepcopy(source["model_spec"]), deepcopy(self.model.model_spec())
        was_adapted = previous["kwargs"].pop("train_transition")
        will_adapt = current["kwargs"].pop("train_transition")
        if previous != current or was_adapted and not will_adapt:
            raise ValueError("Refinement must preserve motion architecture/mode and may only enable transition adaptation")
        if source.get("teacher_provenance", {}).get("sha256") != self.teacher_provenance.get("sha256"):
            raise ValueError("Refinement checkpoint must use the same frozen transition teacher")
        for key in ("image_size", "event_image_size", "context_frames", "bins", "alignment", "alignment_evidence"):
            if source["config"]["data"].get(key) != self.config["data"].get(key):
                raise ValueError(f"Refinement changes the observation protocol: {key}")
        state = {key.removeprefix("model."): value for key, value in source["state_dict"].items() if key.startswith("model.")}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        permitted_missing = set(self.model.state_dict()) - set(state) if will_adapt and not was_adapted else set()
        if unexpected or set(missing) != permitted_missing or any(not key.startswith("adapted_transition.") for key in missing):
            raise ValueError("Refinement checkpoint has incompatible weights")
        # Newly trainable T starts at the exact teacher T retained by the source.
        if will_adapt and not was_adapted:
            self.model.adapted_transition.load_state_dict(self.model.teacher.scene_transition.state_dict(), strict=True)
        self.forecaster.load_state_dict({key.removeprefix("forecaster."): value for key, value in source["state_dict"].items() if key.startswith("forecaster.")}, strict=True)
        self.initialization_provenance = dict(path=str(Path(path).resolve()), sha256=file_sha256(path),
                                             git_commit=source.get("git_commit"), optimizer="reset")

    def train(self, mode=True):
        super().train(mode)
        self.forecaster.eval()
        if self.stage == "conditioning":
            options = self.config["model"]
            self.forecaster.predictor.train(mode and options["train_predictor"] and options["predictor_dropout"])
        return self

    @torch.no_grad()
    def encode_history(self, batch):
        rgb = batch["history_rgb"]
        if rgb.ndim != 5 or rgb.shape[1] != self.config["data"]["context_frames"]:
            raise ValueError("history_rgb must be [B,context_frames,3,H,W]")
        return self.forecaster.encode_latents(rgb.flatten(0, 1)).reshape(rgb.shape[0], rgb.shape[1], -1, self.forecaster.pca_dim)

    @torch.no_grad()
    def encode_targets(self, batch):
        rgb = batch["future_rgb_target"]
        if rgb.ndim != 5 or rgb.shape[1] != self.config["data"]["rollout_steps"]:
            raise ValueError("future_rgb_target must be [B,rollout_steps,3,H,W]")
        return self.forecaster.encode_latents(rgb.flatten(0, 1)).reshape(rgb.shape[0], rgb.shape[1], -1, self.forecaster.pca_dim)

    def predict_batch(self, batch):
        history = self.encode_history(batch)
        if self.stage == "conditioning":
            # Frozen backbone parameters still carry gradients to the injected
            # condition through the second half. Do not wrap this in no_grad.
            return self.model(history, batch.get("history_events"), self.forecaster.predictor)
        if self.stage == "fusion":
            with torch.no_grad():
                base = self.forecaster.predictor(history)
            return self.model(history, batch.get("history_events"), base)
        if self.stage == "transition":
            return self.model(history[:, -1], batch["future_events_target"])
        return self.model(history, batch.get("history_events"), steps=self.config["data"]["rollout_steps"], base_predictor=self.forecaster.predictor)

    def _distance(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction/target shapes differ: {tuple(prediction.shape)} vs {tuple(target.shape)}")
        function = F.mse_loss if self.config["loss"]["kind"] == "mse" else F.smooth_l1_loss
        return function(prediction.float(), target.float())

    def compute_losses(self, batch, outputs=None):
        target = self.encode_targets(batch)
        if outputs is None and self.stage == "transition":
            history = self.encode_history(batch)
            starts = torch.cat((history[:, -1:], target[:, :-1]), dim=1)
            scenes, motions = [], []
            for step in range(target.shape[1]):
                item = self.model(starts[:, step], batch["future_events_target"][:, step:step + 1])
                scenes.append(item["scene"])
                motions.append(item["motion"])
            outputs = dict(scene=torch.cat(scenes, dim=1), motion=torch.cat(motions, dim=1))
        elif outputs is None:
            outputs = self.predict_batch(batch)
        scene = self._distance(outputs["scene"], target)
        motion = scene.new_zeros(())
        if self.stage == "motion" and self.config["loss"]["motion_weight"]:
            motion_target = self.model.encode_target_motion(batch["future_events_target"])
            motion = self._distance(outputs["motion"], motion_target.detach())
        loss = self.config["loss"]["scene_weight"] * scene + self.config["loss"]["motion_weight"] * motion
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite research loss")
        return dict(loss=loss, scene_loss=scene, motion_loss=motion)

    def training_step(self, batch, batch_idx):
        losses = self.compute_losses(batch)
        self.log_dict({f"train_{key}": value for key, value in losses.items()}, on_step=True, on_epoch=True, batch_size=batch["history_rgb"].shape[0], sync_dist=True)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        losses = self.compute_losses(batch)
        self.log_dict({f"val_{key}": value for key, value in losses.items()}, on_step=False, on_epoch=True, batch_size=batch["history_rgb"].shape[0], sync_dist=True)
        return losses

    def on_validation_epoch_end(self):
        self.last_validation_step = self.global_step

    def configure_optimizers(self):
        if self.stage == "conditioning":
            options = self.config["optimization"]
            groups = []
            for name, module, lr in (("conditioner", self.model, options["lr"]),
                                     ("predictor", self.forecaster.predictor, options["predictor_lr"])):
                parameters = [p for p in module.parameters() if p.requires_grad]
                if parameters:
                    groups.append(dict(params=parameters, lr=lr, name=name))
            if not groups:
                raise ValueError("Frozen RGB control has no trainable parameters; use the evaluate command")
            return torch.optim.AdamW(groups, weight_decay=options["weight_decay"])
        parameters = [p for p in self.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("This diagnostic has no trainable parameters; use the evaluate command")
        options = self.config["optimization"]
        return torch.optim.AdamW(parameters, lr=options["lr"], weight_decay=options["weight_decay"])

    def _loader(self, role):
        from .data import ResearchDSECDataset, collate_research
        options = self.config["data"]
        if role not in self.datasets:
            keys = ("image_size", "event_image_size", "context_frames", "bins", "rollout_steps", "alignment", "alignment_evidence", "tolerance_us", "sample_history_frames", "evaluation_horizons")
            kwargs = {key: options[key] for key in keys if key in options}
            for key in ("sample_ids", "sequences"):
                if options.get(f"{role}_{key}") is not None:
                    kwargs[key] = options[f"{role}_{key}"]
            self.datasets[role] = ResearchDSECDataset(options["root"], split=options[f"{role}_split"], stride=options.get(f"{role}_stride", options["stride"]), future_targets=True,
                                                    future_event_targets=self.stage != "conditioning", **kwargs)
        dataset = self.datasets[role]
        if not len(dataset):
            raise ValueError(f"Empty {role} dataset")
        workers = options["num_workers"]
        return DataLoader(dataset, batch_size=options["batch_size"], shuffle=role == "train", num_workers=workers,
                          collate_fn=collate_research, pin_memory=torch.cuda.is_available(),
                          **({"multiprocessing_context": "spawn", "prefetch_factor": options.get("prefetch_factor", 1), "persistent_workers": options.get("persistent_workers", True)} if workers else {}))

    def train_dataloader(self):
        return self._loader("train")

    def val_dataloader(self):
        return self._loader("val")

    def teardown(self, stage):
        for dataset in self.datasets.values():
            if hasattr(dataset, "close"):
                dataset.close()

    def on_save_checkpoint(self, checkpoint):
        checkpoint.update(research_format=CHECKPOINT_FORMAT, stage=self.stage, config=deepcopy(self.config),
                          model_spec=self.model.model_spec(), protocol=deepcopy(self.protocol),
                          teacher_provenance=deepcopy(self.teacher_provenance), git_commit=git_revision(),
                          initialization_provenance=deepcopy(self.initialization_provenance),
                          predictor_initialization=deepcopy(self.predictor_initialization),
                          forecaster_architecture=deepcopy(getattr(self.forecaster, "architecture", {})),
                          forecaster_provenance=deepcopy(getattr(self.forecaster, "provenance", {})),
                          trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
                          frozen_parameters=sum(p.numel() for p in self.parameters() if not p.requires_grad))

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("research_format") != CHECKPOINT_FORMAT or _recipe_signature(self.config) != _recipe_signature(checkpoint["config"]):
            raise ValueError("Checkpoint recipe differs; resume requires the same research stage and scientific/optimizer configuration")
        if checkpoint["model_spec"] != self.model.model_spec() or checkpoint["protocol"] != self.protocol:
            raise ValueError("Checkpoint model architecture or causal protocol differs")
        self.teacher_provenance = deepcopy(checkpoint.get("teacher_provenance"))
        self.initialization_provenance = deepcopy(checkpoint.get("initialization_provenance"))
        self.predictor_initialization = deepcopy(checkpoint.get("predictor_initialization"))


def load_task(config, *, forecaster=None, checkpoint=None, model=None):
    """Restore complete task state; embedded teacher requires no source file."""
    saved = read_checkpoint(checkpoint) if checkpoint else None
    task = ResearchTask(config, forecaster=forecaster, model=model, saved=saved)
    if saved is not None:
        task.on_load_checkpoint(saved)
        task.load_state_dict(saved["state_dict"], strict=True)
    return task
