"""Deterministic, tensor-independent controls for Event-conditioning diagnostics.

Build the donor map from the complete evaluation inventory before applying a
sample limit or splitting the inventory into batches.  A donor must belong to a
different sequence: adjacent windows in one sequence can share most of their
events and are therefore unsuitable as mismatched controls.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Sequence


ALLOWED_VARIANTS = (
    "correct", "mismatched", "bin_reverse", "bin_shuffle", "zero", "query_only", "off",
)


def _check_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")


def _sequence_name(sample_id: str) -> str:
    if not isinstance(sample_id, str):
        raise ValueError("sample_ids must contain strings in sequence:window format")
    sequence, separator, window = sample_id.rpartition(":")
    if not separator or not sequence or not window:
        raise ValueError(f"Invalid sample ID {sample_id!r}; expected sequence:window")
    return sequence


def _stable_integer(seed: int, domain: str, sample_id: str) -> int:
    # JSON preserves the field boundaries even if a sequence contains colons.
    # Do not use hash(), whose result depends on PYTHONHASHSEED.
    payload = json.dumps([seed, domain, sample_id], ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest(), "big")


def make_event_donor_map(sample_ids: Sequence[str], seed: int) -> dict[str, str]:
    """Choose a deterministic foreign-sequence Event donor for every sample.

    Mapping is independent of inventory order, batch size, and global RNG state.
    Donors are sampled with replacement from all windows outside the recipient's
    sequence; this is not a one-to-one permutation.  At least two sequences and
    unique IDs are required.  Call once on the full evaluation inventory, then
    index the returned map for the requested recipients.
    """
    _check_seed(seed)
    if isinstance(sample_ids, (str, bytes)):
        raise ValueError("sample_ids must be a sequence of unique sample IDs")
    ids = list(sample_ids)
    if not ids:
        raise ValueError("sample_ids must not be empty")
    grouped: dict[str, list[str]] = defaultdict(list)
    for sample_id in ids:
        grouped[_sequence_name(sample_id)].append(sample_id)
    if len(set(ids)) != len(ids):
        raise ValueError("sample_ids must be unique")
    if len(grouped) < 2:
        raise ValueError("Mismatched Event controls require at least two different sequences")

    # A contiguous interval per sequence allows sampling foreign windows in O(1)
    # per recipient rather than constructing an O(N)-sized candidate list N times.
    inventory: list[str] = []
    ranges: dict[str, tuple[int, int]] = {}
    for sequence in sorted(grouped):
        start = len(inventory)
        inventory.extend(sorted(grouped[sequence]))
        ranges[sequence] = (start, len(inventory) - start)

    result = {}
    for sample_id in inventory:
        start, count = ranges[_sequence_name(sample_id)]
        candidate_count = len(inventory) - count
        candidate = _stable_integer(seed, "event-donor", sample_id) % candidate_count
        donor_index = candidate if candidate < start else candidate + count
        result[sample_id] = inventory[donor_index]
    return result


def deterministic_bin_permutation(bins: int, seed: int, sample_id: str) -> list[int]:
    """Return a stable per-sample bin shuffle, non-identity when bins >= 2.

    Apply these indices to raw bin contents before adding temporal embeddings;
    chronological embedding slots must remain fixed.  Reordering already
    positioned key/value rows would not test temporal-order sensitivity.
    """
    _check_seed(seed)
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
        raise ValueError("bins must be a positive integer")
    _sequence_name(sample_id)
    permutation = list(range(bins))
    random.Random(_stable_integer(seed, "event-bin-shuffle", sample_id)).shuffle(permutation)
    if bins >= 2 and permutation == list(range(bins)):
        # The intervention must change content order, including with two bins.
        permutation = permutation[1:] + permutation[:1]
    return permutation


def validate_variants(variants: Sequence[str]) -> tuple[str, ...]:
    """Validate unique controls and return canonical order with correct first."""
    if isinstance(variants, (str, bytes)):
        raise ValueError("variants must be a sequence of variant names")
    values = list(variants)
    if any(not isinstance(value, str) for value in values):
        raise ValueError("variant names must be strings")
    if len(set(values)) != len(values):
        raise ValueError("variants must be unique")
    unknown = set(values) - set(ALLOWED_VARIANTS)
    if unknown:
        raise ValueError(f"Unknown diagnostic variants: {', '.join(sorted(unknown))}")
    if "correct" not in values:
        raise ValueError("Diagnostic variants must include 'correct' as the reference")
    selected = set(values)
    return tuple(variant for variant in ALLOWED_VARIANTS if variant in selected)
