#!/usr/bin/env python3
"""Train the A1 base model. See ``distance_decayed_memory.train.a1``.

    python scripts/train/train_a1.py --config configs/track_a/a1_M.json \
        --set run_dir=/path/to/run --set max_steps=2000

``--set key=value`` overrides any config field; values are parsed as JSON
when possible (numbers, booleans, objects) and as strings otherwise.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import signal

from distance_decayed_memory.train import a1


def parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def main() -> int:
    # Register before importing CUDA-heavy code paths so an early SIGUSR1 is
    # recorded rather than fatal (Phase 0 requeue finding).
    signal.signal(signal.SIGUSR1, a1.request_stop)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    values = json.loads(args.config.read_text()) if args.config else {}
    for item in args.set:
        key, _, text = item.partition("=")
        values[key] = parse_value(text)
    names = {f.name for f in dataclasses.fields(a1.A1Config)}
    unknown = set(values) - names
    if unknown:
        parser.error(f"unknown config keys: {sorted(unknown)}")
    if "betas" in values:
        values["betas"] = tuple(values["betas"])
    return a1.train(a1.A1Config(**values))


if __name__ == "__main__":
    raise SystemExit(main())
