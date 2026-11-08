"""Export a legacy E2 encoder without old runtime paths or optimizer state."""

import argparse
import hashlib
from pathlib import Path

import torch

from foresight.config import resolve_path
from foresight.data.events import encoding_spec
from foresight.event import EVENT_FORMAT, load_event_encoder
from foresight.models.event_encoder import EventEncoder
from foresight.models.forecaster import DINOForecaster


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dinof", default="checkpoints/dinof.pt")
    args = parser.parse_args(argv)
    source, output = resolve_path(args.source), resolve_path(args.output)
    if output.exists():
        raise FileExistsError(output)
    saved = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    if saved.get("format") != "event-dinof-encoder-v1" or saved.get("mode") != "full":
        raise ValueError("Only legacy full-encoder E2 checkpoints are supported")
    config = saved["config"]
    if (config.get("history_encoder") or config.get("encoder", {}).get("frontend", "none") != "none"
            or config["optimization"].get("loss_kind", "pca_smooth_l1") != "pca_smooth_l1"):
        raise ValueError("This import retains the framewise PCA-SmoothL1 E2 architecture only")
    encoding = encoding_spec(config["data"])
    if config["data"]["image_size"] != [448, 896]:
        raise ValueError("Expected the original 448x896 E2 input size")
    exported = dict(format=EVENT_FORMAT, state_dict=saved["state_dict"], image_size=[448, 896],
                    event_encoding=encoding, source_step=int(saved["step"]),
                    provenance={"source_name": source.name, "source_sha256": sha256(source),
                                "manifest_sha256": saved["manifest_sha256"]})
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        torch.save(exported, temporary)
        encoder = EventEncoder(DINOForecaster(resolve_path(args.dinof)))
        load_event_encoder(encoder, temporary)
        assert all(torch.equal(encoder.state_dict()[key], value) for key, value in saved["state_dict"].items())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Exported {len(exported['state_dict'])} identical tensors: {output}")


if __name__ == "__main__":
    main()
