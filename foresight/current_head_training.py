"""Train a current-RGB detector in standardized retained PCA coordinates."""

from copy import deepcopy

import torch
from torch import distributed as dist

from .data.definitions import HEIGHT, WIDTH
from .data.rgb import RGBDataset, preprocess_rgb
from .diagnose_readout import (
    CURRENT_RETAINED_COORDINATES, CURRENT_RETAINED_SOURCE,
    CurrentDetectionEvaluator, fp32_readout, retained_coordinates, write_json,
)
from .training import DetectorTask, _distributed


class CurrentRGBDataset(RGBDataset):
    """Use the forecast window inventory, but read only its last observed RGB.

    Current targets can be empty. Their predictions still count in validation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prediction_frames = 0
        self.prediction_ms = 0

    def __getitem__(self, index):
        name, end = self.windows[index]
        reader = self._reader(name)
        target = reader.read_target(end)
        xyxy = torch.from_numpy(target.boxes.copy()) / torch.tensor([WIDTH, HEIGHT, WIDTH, HEIGHT])
        x0, y0, x1, y1 = xyxy.unbind(-1)
        boxes = torch.stack(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0), dim=-1)
        return {
            "history_rgb": preprocess_rgb(reader.read_image(end).image, self.image_size)[None],
            "target": {"boxes": boxes, "labels": torch.from_numpy(target.class_ids.copy())},
            "meta": {"sample_id": f"{name}:{end}", "target_id": target.target_id,
                     "target_timestamp_us": target.timestamp_us},
        }


class CurrentHeadTask(DetectorTask):
    """Freeze the RGB model; optimize a random head on current retained features."""

    def __init__(self, config, *, forecaster=None, detector=None, criterion=None):
        if config["model"].get("detector_checkpoint"):
            raise ValueError("Current head starts from random weights; use --resume for its training checkpoint")
        super().__init__(config, forecaster=forecaster, detector=detector, criterion=criterion)
        self.detector.float()
        self.input_spec.update(
            prediction_ms=0, readout_feature_source=CURRENT_RETAINED_SOURCE,
            readout_coordinates=CURRENT_RETAINED_COORDINATES,
            current_labels="last history frame end", current_empty_gt_policy="include",
            readout_precision="FP32/highest/TF32-off", encoder_chunk_size=1,
        )

    def _make_dataset(self, split):
        data = self.config["data"]
        return CurrentRGBDataset(
            **{key: data[key] for key in ("root", "image_size", "sample_history_frames", "context_frames", "tolerance_us")},
            split=split, stride=data.get(f"{split}_stride", 1),
            sequences=data.get(f"{split}_sequences"), sample_ids=data.get(f"{split}_sample_ids"))

    def forward(self, history_rgb):
        if history_rgb.ndim != 5 or history_rgb.shape[1] < 1:
            raise ValueError("Current head requires RGB input [B,T,3,H,W]")
        current = history_rgb[:, -1]
        use_bf16 = current.is_cuda and self.config["runtime"].get("precision") == "bf16-mixed"
        with torch.no_grad(), torch.autocast(current.device.type, dtype=torch.bfloat16, enabled=use_bf16):
            latent = torch.cat([self.forecaster.encode_latents(part) for part in current.split(1)])
        with fp32_readout():
            return self.detector(retained_coordinates(latent, self.forecaster.pca))

    def training_step(self, batch, batch_idx):
        outputs = self(batch["history_rgb"])
        with fp32_readout():
            losses = self.criterion(outputs, batch["targets"], sync_num_boxes=True)
        count = len(batch["history_rgb"])
        self.log_dict({f"train_{key}": value for key, value in losses.items()}, on_step=True,
                      on_epoch=True, sync_dist=True, batch_size=count)
        return self._training_loss(losses["loss"], count)

    def on_validation_epoch_end(self):
        # The detector's data gathering is unchanged; the current-frame evaluator
        # additionally counts predictions on empty current targets.
        gathered = [None] * self.trainer.world_size if self.global_rank == 0 else None
        if _distributed():
            dist.gather_object(self.validation_predictions, gathered, dst=0)
        else:
            gathered = [self.validation_predictions]
        report = None
        if self.global_rank == 0:
            predictions = [item for rank_items in gathered for item in rank_items]
            if not predictions:
                raise RuntimeError("No current RGB validation samples were evaluated")
            evaluator = CurrentDetectionEvaluator(self.datasets["val"],
                                                   sample_ids=[p["sample_id"] for p in predictions])
            evaluator.update(predictions)
            report = evaluator.compute().to_dict()
            report["metadata"].update(stage=self.stage, feature_source=CURRENT_RETAINED_SOURCE,
                                      input_spec=deepcopy(self.input_spec))
            if self.config.get("output_dir"):
                from pathlib import Path
                folder = Path(self.config["output_dir"])
                folder.mkdir(parents=True, exist_ok=True)
                write_json(folder / "metrics_val.json", report)
        if _distributed():
            reports = [report]
            dist.broadcast_object_list(reports, src=0)
            report = reports[0]
        self.last_report = report
        self.log_dict({f"val_{key}": float(value) for key, value in report["metrics"].items()
                       if value is not None}, prog_bar=True, sync_dist=False)
        self.log("val_samples", float(report["metadata"]["num_samples"]), sync_dist=False)
        self.validation_predictions = []
        self.last_validation_step = int(self.global_step)
