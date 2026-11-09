"""Check the full DSEC inventory needed by RGB/Event train, eval and test (CPU only)."""

import argparse
import json
from pathlib import Path
import sys

# Also support `python /path/to/scripts/check_dsec.py` without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from foresight.data import CLASSES, DEFAULT_SPLITS, HEIGHT, WIDTH, RGBDataset, RGBSequenceReader


def require(condition, message):
    if not condition:
        raise ValueError(message)


def probe_indices(count):
    return sorted({0, count // 2, count - 1}) if count else []


def check_metadata(reader):
    times, tracks = reader.timestamps, reader.tracks
    require(times.ndim == 1 and len(times) > 0, "images/timestamps.txt must be a nonempty vector")
    require(np.all(times[1:] > times[:-1]), "RGB timestamps must be strictly increasing")
    fields = {"t", "x", "y", "w", "h", "class_id", "track_id", "class_confidence"}
    require(tracks.ndim == 1 and fields <= set(tracks.dtype.names or ()),
            "tracks.npy requires fields: " + ", ".join(sorted(fields)))
    for field in fields:
        require(np.isfinite(tracks[field]).all(), f"tracks.npy contains nonfinite {field}")
    require(np.all(tracks["t"][1:] >= tracks["t"][:-1]), "tracks.npy timestamps must be sorted")
    require(np.isin(tracks["t"], times).all(), "tracks.npy timestamps do not match RGB timestamps")
    require(np.isin(tracks["class_id"], np.arange(len(CLASSES))).all(),
            "tracks.npy class_id must be in [0, 7]")
    if reader.exposures is not None:
        require(reader.exposures.shape == (len(times), 2),
                "exposure_timestamps.txt must contain one start,end row per RGB frame")
        require(np.all(reader.exposures[:, 1] >= reader.exposures[:, 0]),
                "exposure_timestamps.txt contains end < start")


def check_images(reader, windows, deep, errors):
    missing = [str(path) for path in reader.image_files if not path.is_file() or path.stat().st_size == 0]
    if missing:
        errors.append(f"Missing/empty RGB images ({len(missing)}): " + ", ".join(missing[:5]))
    indices = set(probe_indices(len(reader)))
    for i in probe_indices(len(windows)):
        end = windows[i][1]
        indices.update(range(end - 3, end + 2))  # Four historical images plus future RGB.
    if deep:
        indices = range(len(reader))
    for i in sorted(indices):
        path = reader.image_files[i]
        try:
            with Image.open(path) as image:
                require(image.size == (WIDTH, HEIGHT), f"Expected {WIDTH}x{HEIGHT}, got {image.size}")
                image.convert("RGB").load()
        except Exception as exc:
            errors.append(f"Cannot decode {path}: {exc}")


def check_events(path, reader, windows, *, deep, event_window_us):
    from foresight.data.event_reader import EventStream, H5HandleCache

    cache = H5HandleCache(max_open=1)
    try:
        handle = cache.get(path)
        required = [f"events/{key}" for key in ("t", "x", "y", "p")] + ["t_offset", "ms_to_idx"]
        require(all(key in handle for key in required),
                f"{path}: missing HDF5 keys: {[key for key in required if key not in handle]}")
        count = len(handle["events/t"])
        require(count > 0, f"{path}: empty event stream")
        for key in ("t", "x", "y", "p"):
            array = handle[f"events/{key}"]
            require(array.shape == (count,) and array.dtype.kind in "uib",
                    f"{path}: events/{key} must be an integer vector of length {count}")
        require(handle["t_offset"].shape == () and handle["t_offset"].dtype.kind in "ui",
                f"{path}: t_offset must be an integer scalar")
        index = handle["ms_to_idx"]
        require(index.ndim == 1 and len(index) > 0 and index.dtype.kind in "ui",
                f"{path}: ms_to_idx must be a nonempty integer vector")
        ms = np.asarray(index, dtype=np.int64)
        require(ms[0] == 0 and np.all((ms >= 0) & (ms <= count)) and np.all(ms[1:] >= ms[:-1]),
                f"{path}: invalid ms_to_idx bounds/order")
        stream = EventStream(path, cache)
        first, last = stream.available_time_range
        require(0 <= stream.event_metadata["first_local_us"] <= stream.event_metadata["last_local_us"],
                f"{path}: invalid first/last event timestamp")

        # Check coverage and causality for EVERY usable window, without loading event payloads.
        ends = np.array([end for _, end in windows], dtype=np.int64)
        history = ends[:, None] + np.arange(-3, 1)
        stops = reader.exposures[history, 1]
        invalid = np.any((stops - event_window_us < first) | (stops > last), axis=1)
        require(not invalid.any(),
                f"{path}: event coverage insufficient for {int(invalid.sum())} windows "
                f"(window_us={event_window_us}, available=[{first}, {last}])")
        require(np.all(stops <= stops[:, -1:]), f"{path}: historical exposure exceeds input cutoff")
        require(np.all(reader.timestamps[ends + 1] > stops[:, -1]),
                f"{path}: future target must follow the input exposure cutoff")

        # Read compressed payloads; --deep scans all chunks with bounded memory.
        chunk = 1_000_000 if deep else 4096
        starts = range(0, count, chunk) if deep else sorted({0, count // 2, max(0, count - chunk)})
        previous = None
        for start in starts:
            stop = min(count, start + chunk)
            arrays = {key: handle[f"events/{key}"][start:stop] for key in ("t", "x", "y", "p")}
            times = arrays["t"]
            require(np.all(times[1:] >= times[:-1]), f"{path}: unsorted event timestamps near index {start}")
            if deep and previous is not None:
                require(times[0] >= previous, f"{path}: unsorted event timestamps at index {start}")
            require(np.isin(arrays["p"], (0, 1)).all(), f"{path}: polarity must be 0/1 near index {start}")
            require(np.all((arrays["x"] >= 0) & (arrays["x"] < WIDTH)) and
                    np.all((arrays["y"] >= 0) & (arrays["y"] < HEIGHT)),
                    f"{path}: event coordinates outside {WIDTH}x{HEIGHT} near index {start}")
            if deep:
                # Verify every indexed millisecond boundary, including repeated indices.
                lo, hi = np.searchsorted(ms, [start, stop])
                positions = ms[lo:hi] - start
                boundaries = np.arange(lo, hi, dtype=np.int64) * 1000
                require(np.all(times[positions] >= boundaries), f"{path}: incorrect ms_to_idx")
                prev_times = np.concatenate(([previous if previous is not None else -1], times[:-1]))
                require(np.all(prev_times[positions] < boundaries), f"{path}: incorrect ms_to_idx")
            previous = int(times[-1])
        require(np.all(np.flatnonzero(ms == count) * 1000 > stream.event_metadata["last_local_us"]),
                f"{path}: invalid ms_to_idx end indices")

        # Exercise the real indexed reader at the beginning, middle and end of the inventory.
        for i in probe_indices(len(windows)):
            for stop in stops[i]:
                start, stop = int(stop) - event_window_us, int(stop)
                for time, index in zip((start, stop), stream.event_indices(start, stop)):
                    local = time - first
                    require((index == count or int(handle["events/t"][index]) >= local) and
                            (index == 0 or int(handle["events/t"][index - 1]) < local),
                            f"{path}: incorrect ms_to_idx lookup at {time}")
                stream.read_events(start, stop)
        return count
    finally:
        cache.close()


def check_sequence(root, split, name, *, rgb_only, deep, event_window_us):
    path = root / ("test" if split == "test" else "train") / name
    result = dict(name=name, path=str(path), frames=0, windows=0, events=0, errors=[])
    required = ["images/timestamps.txt", "object_detections/left/tracks.npy"]
    if not rgb_only:
        required += ["images/left/exposure_timestamps.txt", "events/left/events.h5"]
    for relative in required:
        file = path / relative
        try:
            if not file.is_file() or file.stat().st_size == 0:
                result["errors"].append(f"Missing/empty file: {file}")
        except OSError as exc:
            result["errors"].append(f"Cannot access {file}: {exc}")
    if result["errors"]:
        return result
    try:
        with RGBSequenceReader(path) as reader:
            check_metadata(reader)
            result["frames"] = len(reader)
            # Reuse the training inventory: 40 history frames, four inputs, next-frame target.
            with RGBDataset(root, split=split, sequences=[name]) as dataset:
                windows = dataset.windows
            result["windows"] = len(windows)
            check_images(reader, windows, deep, result["errors"])
            require(bool(windows), "No usable labeled 40-frame-history/next-frame windows")
            if not rgb_only:
                result["events"] = check_events(path / "events/left/events.h5", reader, windows,
                                                deep=deep, event_window_us=event_window_us)
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    return result


def check_dsec(root, *, rgb_only=False, deep=False, event_window_us=50000):
    require(event_window_us in (50000, 200000), "event_window_us must be 50000 or 200000")
    root = Path(root).expanduser().resolve()
    report = dict(root=str(root), mode="rgb" if rgb_only else "rgb+event", deep=deep,
                  event_window_us=event_window_us, splits={})
    for split, names in DEFAULT_SPLITS.items():
        rows = []
        for name in names:
            row = check_sequence(root, split, name, rgb_only=rgb_only, deep=deep,
                                 event_window_us=event_window_us)
            rows.append(row)
            print(f"[{'FAIL' if row['errors'] else 'OK'}] {split}/{name}: "
                  f"frames={row['frames']} windows={row['windows']}", flush=True)
            for error in row["errors"]:
                print(f"  {error}", flush=True)
        report["splits"][split] = dict(passed=bool(rows) and all(not row["errors"] for row in rows),
                                        windows=sum(row["windows"] for row in rows), sequences=rows)
    ready = {split: row["passed"] for split, row in report["splits"].items()}
    report["stages"] = dict(train=ready["train"] and ready["val"], eval=ready["val"], test=ready["test"])
    report["passed"] = all(report["stages"].values())
    print("\nData checks: " + "  ".join(f"{stage}={'PASS' if ok else 'FAIL'}"
                                          for stage, ok in report["stages"].items()))
    print("Windows: " + "  ".join(f"{split}={row['windows']}" for split, row in report["splits"].items()))
    print("Decode: " + ("all images/events" if deep else "sampled images/events; use --deep for full decoding"))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-root", required=True, help="DSEC root containing train/ and test/")
    parser.add_argument("--rgb-only", action="store_true", help="Do not require event streams or exposure files")
    parser.add_argument("--event-window-us", type=int, choices=(50000, 200000), default=50000)
    parser.add_argument("--deep", action="store_true", help="Decode all PNGs and scan all event chunks (slow)")
    parser.add_argument("--output", help="Optional JSON report path outside DSEC (overwrites an existing report)")
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().resolve() if args.output else None
    root = Path(args.data_root).expanduser().resolve()
    if output and output.is_relative_to(root):
        parser.error("--output must be outside the DSEC root")
    report = check_dsec(root, rgb_only=args.rgb_only, deep=args.deep, event_window_us=args.event_window_us)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(f"Report: {output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
