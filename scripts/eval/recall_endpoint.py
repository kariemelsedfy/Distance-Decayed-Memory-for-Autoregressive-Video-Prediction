#!/usr/bin/env python3
"""Recall-specific endpoint (D-016): difference-in-differences on revisits.

    python scripts/eval/recall_endpoint.py \
        --jobs docs/a3-h64-jobs.tsv --root outputs/a3-h64-test \
        --jobs docs/a3-h64-matched-jobs.tsv --root outputs/a3-h64-matched \
        --output outputs/recall_endpoint.json

For a policy P compared with ``decay_continuous`` (D) at one budget,

    Δ_recall = [L_P − L_D](revisits, gap 24–63) − [L_P − L_D](control)

with LPIPS paired frame by frame within each seed, pooled over seeds; the
control is revisits with gap ≥ 64 (no policy retains them; primary) or novel
views (secondary). Positive Δ_recall means D recalls retrievable revisits
better beyond any general-context advantage. Intervals resample (seed,
episode) clusters; p-values are two-sided bootstrap, Holm-corrected over the
secondary family. Job maps after the first are labelled "matched" (D-016).
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np

from distance_decayed_memory.data.revisit_detector import NOVEL, REVISIT
from distance_decayed_memory.eval.results import load_run

BUDGETS = ("blow", "bmid", "bvlow")
KEYS = ("episode", "window_start", "frame")
PRIMARY = ("relic_discrete (matched)", "blow")


def load(jobs: pathlib.Path, root: pathlib.Path, suffix: str) -> dict:
    runs = {}
    with jobs.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            name, seed = row["config"][:-3], int(row["config"][-1])
            policy, budget = ("full", None) if name == "full" else name.rsplit("_", 1)
            runs[(policy + suffix, budget, seed)] = load_run(
                root / row["eval_run"]
            ).records
    return runs


def masks(records: dict) -> dict[str, np.ndarray]:
    kind, gap = records["kind"], records["gap"]
    return {
        "target": (kind == REVISIT) & (gap >= 24) & (gap < 64),
        "control_far": (kind == REVISIT) & (gap >= 64),
        "control_novel": kind == NOVEL,
    }


def did(runs, policy, budget, resamples, rng) -> dict:
    parts = {name: ([], []) for name in ("target", "control_far", "control_novel")}
    for seed in range(3):
        run = runs.get((policy, budget, seed))
        ref = runs.get(("decay_continuous", budget, seed))
        if run is None or ref is None:
            continue
        for key in KEYS:
            if not np.array_equal(run[key], ref[key]):
                raise SystemExit(f"{policy}/{budget}/s{seed}: frames differ")
        for name, mask in masks(ref).items():
            parts[name][0].append((run["lpips"] - ref["lpips"])[mask])
            parts[name][1].append(seed * 1_000_000 + ref["episode"][mask])
    if not parts["target"][0]:
        return {}
    data = {k: (np.concatenate(v), np.concatenate(c)) for k, (v, c) in parts.items()}
    clusters = np.unique(np.concatenate([c for _, c in data.values()]))
    index = {k: np.searchsorted(clusters, c) for k, (_, c) in data.items()}
    sums = {
        k: np.bincount(index[k], weights=v, minlength=len(clusters))
        for k, (v, _) in data.items()
    }
    counts = {k: np.bincount(index[k], minlength=len(clusters)) for k in data}
    picks = rng.integers(0, len(clusters), (resamples, len(clusters)))
    boot = {k: sums[k][picks].sum(1) / counts[k][picks].sum(1) for k in data}
    out = {"seeds": len(parts["target"][0])}
    for name in ("target", "control_far", "control_novel"):
        out[name] = float(data[name][0].mean())
    for label, control in (("far", "control_far"), ("novel", "control_novel")):
        estimate = out["target"] - out[control]
        draws = boot["target"] - boot[control]
        low, high = np.percentile(draws, [2.5, 97.5])
        p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
        out[f"did_{label}"] = {
            "estimate": float(estimate),
            "low": float(low),
            "high": float(high),
            "p": float(min(1.0, max(p, 1 / resamples))),
        }
    return out


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    order = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--root", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if len(args.jobs) != len(args.root):
        raise SystemExit("give one --root per --jobs")
    rng = np.random.default_rng(0)
    runs = {}
    for number, (jobs, root) in enumerate(zip(args.jobs, args.root, strict=True)):
        runs.update(load(jobs, root, "" if number == 0 else " (matched)"))
    policies = sorted({p for p, _, _ in runs} - {"decay_continuous", "full"})
    results = {}
    for budget in BUDGETS:
        for policy in policies:
            value = did(runs, policy, budget, args.resamples, rng)
            if value:
                results[f"{policy} @ {budget}"] = value
    primary_key = f"{PRIMARY[0]} @ {PRIMARY[1]}"
    secondary = {k: v["did_far"]["p"] for k, v in results.items() if k != primary_key}
    adjusted = holm(secondary)
    for key, value in results.items():
        value["role"] = "primary" if key == primary_key else "secondary"
        value["did_far"]["p_holm"] = (
            value["did_far"]["p"] if key == primary_key else adjusted[key]
        )

    print(
        "Δ_recall = decay advantage on revisits 24–63 minus on the control "
        "(LPIPS; + favours decay)"
    )
    print(
        f"{'comparison (P vs decay)':42s} {'target':>8s} {'far ctl':>8s} "
        f"{'Δ_recall (far)':>26s} {'p_holm':>7s} {'Δ (novel ctl)':>24s}"
    )
    for key, value in results.items():
        far, novel = value["did_far"], value["did_novel"]
        mark = " (primary)" if value["role"] == "primary" else ""
        print(
            f"{(key + mark)[:42]:42s} {value['target']:+8.3f} "
            f"{value['control_far']:+8.3f} "
            f"{far['estimate']:+.3f} [{far['low']:+.3f},{far['high']:+.3f}]"
            f"{far['p_holm']:8.3f} "
            f"{novel['estimate']:+.3f} [{novel['low']:+.3f},{novel['high']:+.3f}]"
        )
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
