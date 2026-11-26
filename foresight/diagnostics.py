"""Deterministic event interventions and subsets defined from observed tracks."""

import hashlib
import math

import numpy as np


def mismatched_donors(dataset, recipients):
    """Choose another sequence without depending on batching or worker ordering."""
    by_sequence = {}
    for index, (name, _) in enumerate(dataset.windows):
        by_sequence.setdefault(name, []).append(index)
    if len(by_sequence) < 2:
        raise ValueError("mismatched requires at least two sequences in the donor inventory")
    result = {}
    for sample_id in recipients:
        source = sample_id.rsplit(":", 1)[0]
        choices = sorted(name for name in by_sequence if name != source)
        digest = hashlib.sha256(sample_id.encode()).digest()
        sequence = choices[int.from_bytes(digest[:8], "little") % len(choices)]
        indices = by_sequence[sequence]
        result[sample_id] = indices[int.from_bytes(digest[8:16], "little") % len(indices)]
    return result


def motion_subset(dataset, kind, fraction=0.2):
    """Rank image-plane displacement/velocity change using only the last three observed frames."""
    if kind not in ("high_motion", "nonuniform") or not 0 < fraction <= 1:
        raise ValueError("Invalid subset or top fraction")
    scores = []
    for name, end in dataset.windows:
        reader = dataset._reader(name)
        tracks = [reader.read_tracks(i) for i in (end-2, end-1, end)]
        maps = [{int(row["track_id"]): np.array([row["x"]+row["w"]/2, row["y"]+row["h"]/2], dtype=float)
                 for row in frame} for frame in tracks]
        common = maps[0].keys() & maps[1].keys() & maps[2].keys()
        times = dataset.timestamps[name][end-2:end+1].astype(np.float64) / 1e6
        values = []
        for track in common:
            first = (maps[1][track]-maps[0][track])/(times[1]-times[0])
            second = (maps[2][track]-maps[1][track])/(times[2]-times[1])
            values.append(float(np.linalg.norm(second if kind == "high_motion" else second-first)))
        scores.append((max(values) if values else 0.0, f"{name}:{end}"))
    ranked = sorted(scores, key=lambda item: (-item[0], item[1]))
    count = max(1, math.ceil(len(ranked)*fraction))
    selected = {sample_id for _, sample_id in ranked[:count]}
    return [sample_id for sample_id in dataset.sample_ids if sample_id in selected]
