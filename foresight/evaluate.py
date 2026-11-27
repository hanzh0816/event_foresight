"""Evaluate future detection on a DSEC split or an explicit sample inventory."""

import argparse

from .config import load_config, resolve_path
from .evaluation.runner import evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="Override evaluation.output for this run")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    output = resolve_path(args.output) if args.output is not None else None
    return evaluate(config, output=output)


if __name__ == "__main__":
    main()
