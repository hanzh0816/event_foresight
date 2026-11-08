"""Connect this checkout to the shared model-only checkpoint directory."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/data3/hzh/models/event_foresight")
    args = parser.parse_args(argv)
    source = Path(args.root).expanduser().resolve()
    for filename in ("dinof.pt", "detector.pt", "assets.json"):
        if not (source / filename).is_file():
            raise FileNotFoundError(f"Missing shared asset: {source / filename}")
    target = Path(__file__).resolve().parents[1] / "checkpoints"
    if target.is_symlink():
        if target.resolve() == source:
            print(f"Ready: {target} -> {source}")
            return
        raise FileExistsError(f"Existing link targets another directory: {target}")
    if target.exists():
        raise FileExistsError(f"Refusing to replace existing path: {target}")
    target.symlink_to(source, target_is_directory=True)
    print(f"Linked {target} -> {source}")


if __name__ == "__main__":
    main()
