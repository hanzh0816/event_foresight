"""TensorBoard presentation and complete validation history."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only


def log_training_losses(task, metrics, *, batch_size, progress_metric=None):
    """Log local step values and sample-weighted epoch means across all ranks.

    Every rank contributes every microbatch to the epoch mean. Only the epoch
    computation synchronizes; the rank-zero logger writes its local step value.
    """
    for name, value in metrics.items():
        value = value.detach()
        task.log(f"{name}_step", value, on_step=True, on_epoch=False,
                 sync_dist=False, batch_size=batch_size, prog_bar=name == progress_metric)
        task.log(f"{name}_epoch", value, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=batch_size)


def tensorboard_tag(name):
    """Group presentation tags without changing Lightning/CSV metric names."""
    core = {
        "val_feature_loss": "monitor/val/prediction_loss",
        "val_total_loss": "monitor/val/total_loss",
        "val_foreground_l1": "monitor/val/foreground_l1",
        "val_AP": "monitor/val/AP",
        "train_global_batch": "details/runtime/global_batch",
        "val_samples": "details/runtime/val_samples",
        "epoch": "details/runtime/epoch",
    }
    if name in core:
        return core[name]
    if name.startswith("lr_"):
        return f"monitor/lr/{name[3:]}"
    if name.startswith("val_class/"):
        return f"details/detection/classes/{name[len('val_class/'):]}"
    if name.startswith("train_"):
        metric = name[len("train_"):]
        suffix = ""
        for ending in ("_step", "_epoch"):
            if metric.endswith(ending):
                metric, suffix = metric[:-len(ending)], ending
                break
        if metric in ("loss", "total_loss", "prediction_loss"):
            label = "total_loss" if metric == "loss" else metric
            return f"monitor/train/{label}{suffix}"
        auxiliary = re.fullmatch(r"loss_(class|bbox|giou)_(\d+)", metric)
        if auxiliary:
            component, layer = auxiliary.groups()
            return f"details/detector_aux/layer_{layer}/{component}{suffix}"
        return f"details/loss/train/{metric}{suffix}"
    if name.startswith("val_"):
        metric = name[len("val_"):]
        if metric.startswith(("AP", "AR")):
            return f"details/detection/{metric}"
        region = re.fullmatch(r"(all|foreground|background)_(l1|cosine)", metric)
        if region:
            return f"details/latent/{region[1]}/{region[2]}"
        return f"details/loss/val/{metric}"
    return f"details/other/{name}"


def overview_layout(stage):
    """Reuse scalar tags in a small Custom Scalars dashboard."""
    charts = {
        "Training total loss (rank 0 microbatch)": ["Multiline", ["^monitor/train/total_loss_step$"]],
        "Total objective (epoch)": ["Multiline", ["^monitor/train/total_loss_epoch$"]],
        "Learning rates": ["Multiline", ["^monitor/lr/.*$"]],
        "Validation AP": ["Multiline", ["^monitor/val/AP$"]],
    }
    if stage == "predictor":
        charts["Total objective (epoch)"][1].append("^monitor/val/total_loss$")
        charts["Prediction loss (epoch)"] = ["Multiline", [
            "^monitor/train/prediction_loss_epoch$", "^monitor/val/prediction_loss$"]]
        charts["Validation foreground L1"] = ["Multiline", ["^monitor/val/foreground_l1$"]]
    return {"Training overview": charts}


class TrainingTensorBoardLogger(TensorBoardLogger):
    """Keep detailed scalars, with a focused view for following training."""

    def __init__(self, *args, stage, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage = stage
        self._overview_written = False

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        if not self._overview_written:
            self.experiment.add_custom_scalars(overview_layout(self.stage))
            self._overview_written = True
        super().log_metrics({tensorboard_tag(name): value for name, value in metrics.items()}, step=step)


def validation_scalars(report):
    """Retain checkpoint monitor aliases; absent class AP stays absent, not zero."""
    scalars = {}
    for name, value in report["metrics"].items():
        if value is None:
            continue
        key = "val_feature_loss" if name == "prediction_loss" else f"val_{name}"
        scalars[key] = float(value)
    for label, metrics in report.get("per_class", {}).items():
        for name, value in metrics.items():
            if value is not None:
                scalars[f"val_class/{label}/{name}"] = float(value)
    scalars["val_samples"] = float(report["metadata"]["num_samples"])
    return scalars


def write_validation_report(folder, report, *, epoch, global_step, log_dir=None):
    """Append every completed validation and keep the latest-report interface."""
    record = deepcopy(report)
    record["metadata"].update(epoch=int(epoch), global_step=int(global_step),
                              recorded_at_utc=datetime.now(timezone.utc).isoformat())
    if log_dir is not None:
        record["metadata"]["log_dir"] = str(log_dir)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "validation_history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    latest = folder / "metrics_val.json.tmp"
    latest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    latest.replace(folder / "metrics_val.json")
