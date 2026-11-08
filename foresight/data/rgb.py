"""Four observed RGB frames and next-frame labels on the fixed DSEC inventory."""
from numbers import Integral
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .definitions import CLASSES, DEFAULT_SPLITS, HEIGHT, WIDTH, require_int
from .reader import RGBSequenceReader


def _size(value):
    if isinstance(value, Integral) and not isinstance(value, bool):
        values = (value, value)
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        values = tuple(value)
    else:
        raise ValueError("image_size must be an integer or [height, width]")
    for item in values:
        require_int(item, "image dimension", 1)
    return tuple(map(int, values))


def preprocess_rgb(image, image_size=(448, 896)):
    """PIL bilinear resize, RGB ToTensor and ImageNet normalization."""
    height, width = _size(image_size)
    image = Image.fromarray(image).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255.0
    mean = tensor.new_tensor((0.485, 0.456, 0.406))[:, None, None]
    std = tensor.new_tensor((0.229, 0.224, 0.225))[:, None, None]
    return (tensor - mean) / std


class RGBDataset:
    """Inventory uses 40 frames; the model receives the final four by default.

    Future RGB is an explicit training target opt-in. It is never concatenated
    into history_rgb. The default detector/evaluation path does not decode it.
    Sample selection preserves the supplied ID order and original inventory.
    """
    def __init__(self, root, split="train", image_size=(448, 896), sample_history_frames=40,
                 context_frames=4, stride=1, tolerance_us=1000, sequences=None,
                 future_rgb=False, sample_ids=None):
        require_int(sample_history_frames, "sample_history_frames", 1)
        require_int(context_frames, "context_frames", 1)
        require_int(stride, "stride", 1)
        require_int(tolerance_us, "tolerance_us", 0)
        if context_frames > sample_history_frames:
            raise ValueError("context_frames must not exceed sample_history_frames")
        if split not in DEFAULT_SPLITS:
            raise ValueError(f"Unknown DSEC split: {split!r}")
        if type(future_rgb) is not bool:
            raise ValueError("future_rgb must be a boolean")
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.image_size = _size(image_size)
        if any(dimension % 14 for dimension in self.image_size):
            raise ValueError("image_size must be divisible by patch_size 14")
        self.grid_size = tuple(dimension // 14 for dimension in self.image_size)
        self.history_frames = self.sample_history_frames = self.window_history_frames = int(sample_history_frames)
        self.context_frames = int(context_frames)
        self.history_ms = self.history_frames * 50
        self.prediction_ms, self.prediction_frames = 50, 1
        self.stride, self.tolerance_us = int(stride), int(tolerance_us)
        self.future_rgb = future_rgb
        self.selected_sequences = None if sequences is None else tuple(sequences)
        self.sequences = DEFAULT_SPLITS[split] if sequences is None else self.selected_sequences
        if len(set(self.sequences)) != len(self.sequences):
            raise ValueError("sequences must not contain duplicates")
        invalid = set(self.sequences) - set(DEFAULT_SPLITS[split])
        if invalid:
            raise ValueError(f"Sequences outside the fixed {split} split: {sorted(invalid)}")
        self.timestamps, self._readers = {}, {}
        inventory = self._build_windows()
        self.full_inventory_size = len(inventory)
        self.source_indices = list(range(len(inventory)))
        if sample_ids is not None:
            if isinstance(sample_ids, (str, bytes)):
                raise ValueError("sample_ids must be a sequence of unique IDs")
            ids = list(sample_ids)
            if any(not isinstance(item, str) for item in ids) or len(set(ids)) != len(ids):
                raise ValueError("sample_ids must be unique strings")
            lookup = {f"{name}:{end}": index for index, (name, end) in enumerate(inventory)}
            missing = [item for item in ids if item not in lookup]
            if missing:
                raise ValueError(f"Sample IDs not in the selected inventory: {missing[:5]}")
            self.source_indices = [lookup[item] for item in ids]
        self.windows = [inventory[index] for index in self.source_indices]
        self.sample_ids = [f"{name}:{end}" for name, end in self.windows]

    def reader(self, sequence):
        if sequence not in self.sequences:
            raise ValueError(f"Sequence is outside the selected inventory: {sequence}")
        directory = "test" if self.split == "test" else "train"
        return RGBSequenceReader(self.root / directory / sequence)

    def _build_windows(self):
        windows = []
        for name in self.sequences:
            with self.reader(name) as reader:
                times = reader.timestamps.copy()
                self.timestamps[name] = times
                invalid = np.abs(np.diff(times) - 50000) > self.tolerance_us
                prefix = np.concatenate(([0], np.cumsum(invalid)))
                labeled = {int(i) for i in np.flatnonzero(reader.label_ends > reader.label_starts)
                           if len(reader.read_target(int(i)).boxes)}
                for end in range(self.history_frames - 1, len(times) - self.prediction_frames, self.stride):
                    start = end - self.history_frames + 1
                    target_index = end + self.prediction_frames
                    if target_index in labeled and prefix[target_index] == prefix[start]:
                        windows.append((name, end))
        return windows

    def _reader(self, sequence):
        if sequence not in self._readers:
            self._readers[sequence] = self.reader(sequence)
        return self._readers[sequence]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        name, end = self.windows[index]
        start, target_index = end - self.context_frames + 1, end + self.prediction_frames
        reader = self._reader(name)
        target = reader.read_target(target_index)
        xyxy = torch.from_numpy(target.boxes.copy()) / torch.tensor([WIDTH, HEIGHT, WIDTH, HEIGHT])
        x0, y0, x1, y1 = xyxy.unbind(-1)
        boxes = torch.stack(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0), dim=-1)
        sample = {
            "history_rgb": torch.stack([preprocess_rgb(reader.read_image(i).image, self.image_size)
                                        for i in range(start, end + 1)]),
            "target": {"boxes": boxes, "labels": torch.from_numpy(target.class_ids.copy())},
            "meta": {"sample_id": f"{name}:{end}", "target_id": target.target_id,
                     "target_timestamp_us": target.timestamp_us},
        }
        if self.future_rgb:
            sample["future_rgb_target"] = preprocess_rgb(reader.read_image(target_index).image, self.image_size)
        return sample

    def close(self):
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_readers"] = {}
        return state

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def collate_rgb(samples):
    batch = {"history_rgb": torch.stack([sample["history_rgb"] for sample in samples]),
             "targets": [sample["target"] for sample in samples],
             "meta": [sample["meta"] for sample in samples]}
    present = ["future_rgb_target" in sample for sample in samples]
    if any(present) and not all(present):
        raise ValueError("Cannot mix forecast and future-target samples in one batch")
    if all(present):
        batch["future_rgb_target"] = torch.stack([sample["future_rgb_target"] for sample in samples])
    return batch
