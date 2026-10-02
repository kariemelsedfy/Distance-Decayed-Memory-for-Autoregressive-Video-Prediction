#!/usr/bin/env python3
"""Paired per-gap comparison of evaluation runs against one reference run.

    python scripts/eval/compare_gaps.py runs/eval-a/ runs/eval-b/ \
        --reference runs/eval-window/ --output comparison.json

Every run must score the same frames (same episodes, windows, and frames, as
P1 on the same split and episode count produces). For each condition (novel
views, then revisits in ``--edges`` gap ranges) it reports each run's mean
metric and the paired difference ``reference − run`` (for LPIPS, positive
means the run is better than the reference), with a 95% bootstrap interval
that resamples whole episodes.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

import numpy as np

from distance_decayed_memory.data.revisit_detector import NOVEL, REVISIT
from distance_decayed_memory.eval.results import load_run

KEYS = ("episode", "window_start", "frame")


def label(run_dir: pathlib.Path, info: dict) -> str:
    """The run directory without its timestamp (unique, unlike policy@budget)."""
    return re.sub(r"-\d{8}T\d{6}Z$", "", pathlib.Path(run_dir).resolve().name)


def conditions(records: dict, edges: list[int]) -> dict[str, np.ndarray]:
    kind, gap = records["kind"], records["gap"]
    out = {"novel": kind == NOVEL}
    for low, high in zip(edges[:-1], edges[1:]):
        out[f"[{low},{high})"] = (kind == REVISIT) & (gap >= low) & (gap < high)
    return out


def paired(
    difference: np.ndarray, episodes: np.ndarray, resamples: int, rng
) -> tuple[float, float, float]:
    groups = [difference[episodes == e] for e in np.unique(episodes)]
    sums = np.array([g.sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    picks = rng.integers(0, len(groups), (resamples, len(groups)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(difference.mean()), float(low), float(high)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=pathlib.Path)
    parser.add_argument("--reference", required=True, type=pathlib.Path)
    parser.add_argument("--metric", default="lpips")
    parser.add_argument(
        "--edges", type=int, nargs="+", default=[16, 24, 32, 48, 64, 128, 4096]
    )
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    reference = load_run(args.reference)
    ref = reference.records
    key = np.stack([ref[k] for k in KEYS], axis=1)
    masks = conditions(ref, args.edges)
    sign = 1.0 if args.metric == "lpips" else -1.0
    result = {"reference": label(args.reference, reference.info), "rows": {}}
    names = [result["reference"]]
    table = {result["reference"]: {}}
    for name, mask in masks.items():
        table[result["reference"]][name] = (float(np.nanmean(ref[args.metric][mask])),)
    for run_dir in args.runs:
        run = load_run(run_dir)
        records = run.records
        if not np.array_equal(np.stack([records[k] for k in KEYS], axis=1), key):
            raise SystemExit(f"{run_dir} does not score the same frames")
        name = label(run_dir, run.info)
        names.append(name)
        table[name] = {}
        for condition, mask in masks.items():
            difference = sign * (ref[args.metric] - records[args.metric])[mask]
            mean, low, high = paired(
                difference, ref["episode"][mask], args.resamples, rng
            )
            table[name][condition] = (
                float(np.nanmean(records[args.metric][mask])),
                mean,
                low,
                high,
            )
    result["rows"] = table
    result["frames"] = {name: int(mask.sum()) for name, mask in masks.items()}

    header = f"{args.metric:34s}" + "".join(f"{c:>24s}" for c in masks)
    print(header)
    for name in names:
        cells = []
        for condition in masks:
            values = table[name][condition]
            if len(values) == 1:
                cells.append(f"{values[0]:.3f} (reference)")
            else:
                mean, difference, low, high = values
                star = "*" if low > 0 or high < 0 else " "
                cells.append(f"{mean:.3f} {difference:+.3f}{star}")
        print(f"{name[:34]:34s}" + "".join(f"{c:>24s}" for c in cells))
    print("difference = reference − run (positive: run better); * 95% CI excludes 0")
    print("frames:", result["frames"])
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
