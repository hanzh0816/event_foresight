"""Construct RGB and Event-RGB datasets with the same sample and label rules."""

from .rgb import RGBDataset


def dataset_from_config(config, split, future_rgb=False, *, target_only=False, **overrides):
    data = config["data"]
    event = config.get("model", {}).get("event", {})
    kwargs = dict(
        root=data["root"], split=split, image_size=data.get("image_size", (448, 896)),
        sample_history_frames=data.get("sample_history_frames", 40),
        context_frames=data.get("context_frames", 4),
        stride=data.get(f"{split}_stride", data.get("stride", 1)),
        sequences=data.get(f"{split}_sequences"), sample_ids=data.get(f"{split}_sample_ids"),
        tolerance_us=data.get("tolerance_us", 1000),
        prediction_frames=data.get("prediction_frames", 1), future_rgb=future_rgb,
        target_only=target_only,
        horizontal_flip_prob=data.get("horizontal_flip_prob", 0.5) if split == "train" else 0.0,
    )
    cls = RGBDataset
    if event.get("enabled", False):
        if target_only:
            raise ValueError("Target-only detector data must use the RGB configuration")
        from .events import EventRGBDataset
        cls = EventRGBDataset
        kwargs.update(
            event_bins=data.get("event_bins", 10), event_size=event.get("input_size", (512, 1024)),
            event_window_us=data.get("event_window_us", 50000),
            event_activity_threshold=data.get("event_activity_threshold", 1),
        )
    kwargs.update(overrides)
    return cls(**kwargs)
