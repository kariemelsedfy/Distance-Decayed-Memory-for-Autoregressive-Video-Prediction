#!/usr/bin/env python3
"""Run one A2 memory fine-tune. See ``distance_decayed_memory.train.a2``.

    python scripts/train/train_a2.py --config configs/track_a/a2_decay_Bmid.json \
        --set run_dir=/path/to/run --set base_checkpoint=/path/to/a1/latest.pt

``--set key=value`` overrides any config field; values are parsed as JSON
when possible (numbers, booleans, objects) and as strings otherwise.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import signal

from distance_decayed_memory.train import a2


def parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def main() -> int:
    # Register before importing CUDA-heavy code paths so an early SIGUSR1 is
    # recorded rather than fatal (Phase 0 requeue finding).
    signal.signal(signal.SIGUSR1, a2.request_stop)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    values = json.loads(args.config.read_text()) if args.config else {}
    for item in args.set:
        key, _, text = item.partition("=")
        values[key] = parse_value(text)
    names = {f.name for f in dataclasses.fields(a2.A2Config)}
    unknown = set(values) - names
    if unknown:
        parser.error(f"unknown config keys: {sorted(unknown)}")
    if "betas" in values:
        values["betas"] = tuple(values["betas"])
    return a2.train(a2.A2Config(**values))


if __name__ == "__main__":
    raise SystemExit(main())
