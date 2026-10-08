#!/usr/bin/env python3
"""Pooled A3 results at H 64 (D-015): tables and the headline figure.

    python scripts/eval/a3_summary.py --jobs docs/a3-h64-jobs.tsv \
        --runs-root outputs/a3-h64-test --output-dir outputs/a3-h64-summary

Reads each evaluation listed in the job map (fetched under ``--runs-root``),
pools the three seeds, and reports for every budget, policy, and condition
(novel views; revisits by gap) the mean metric and paired differences against
``window`` and against ``decay_continuous`` at the same budget and seed. The
95% intervals resample (seed, episode) clusters. All runs must score the same
frames per seed (P1 on the same test episodes). Writes ``summary.json``,
``summary.md``, and ``a3_h64.png``.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from distance_decayed_memory.data.revisit_detector import NOVEL, REVISIT  # noqa: E402
from distance_decayed_memory.eval.results import load_run  # noqa: E402

BUDGETS = {
    "bmid": "B-mid (4,096 tokens, 25%)",
    "blow": "B-low (2,048, 12.5%)",
    "bvlow": "B-vlow (1,024, 6.25%)",
}
POLICIES = [
    "window",
    "window_sink",
    "uniform_subsample",
    "relic_discrete",
    "decay_continuous",
]
# Validated categorical order (dataviz reference palette, light mode), as in
# scripts/figures/memory_policy_density.py; `full` is drawn as a dashed ink line.
COLORS = {
    "window": "#2a78d6",
    "window_sink": "#eb6834",
    "uniform_subsample": "#1baf7a",
    "relic_discrete": "#eda100",
    "decay_continuous": "#008300",
}
INK, MUTED, GRID = "#1f2328", "#59636e", "#e6e8eb"
EDGES = [16, 24, 32, 48, 64, 128, 4096]
KEYS = ("episode", "window_start", "frame")


def conditions(records: dict) -> dict[str, np.ndarray]:
    kind, gap = records["kind"], records["gap"]
    out = {"novel": kind == NOVEL}
    for low, high in zip(EDGES[:-1], EDGES[1:]):
        label = f"[{low},{high})" if high < 4096 else f"[{low},+)"
        out[label] = (kind == REVISIT) & (gap >= low) & (gap < high)
    return out


def load_runs(jobs: pathlib.Path, root: pathlib.Path) -> dict:
    runs = {}
    with jobs.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            name = row["config"]
            seed = int(name[-1])
            stem = name[:-3]
            policy, budget = ("full", None) if stem == "full" else stem.rsplit("_", 1)
            runs[(policy, budget, seed)] = load_run(root / row["eval_run"]).records
    return runs


def bootstrap(values: np.ndarray, clusters: np.ndarray, resamples: int, rng):
    unique, index = np.unique(clusters, return_inverse=True)
    sums = np.bincount(index, weights=values, minlength=len(unique))
    counts = np.bincount(index, minlength=len(unique))
    picks = rng.integers(0, len(unique), (resamples, len(unique)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def pooled(runs, policy, budget, reference, metric, resamples, rng):
    """Mean and paired difference ``reference − policy`` per condition."""
    out = {}
    seeds = sorted(s for (p, b, s) in runs if p == "window" and b == budget)
    for condition in conditions(runs[("window", budget, seeds[0])]):
        values, diffs, clusters = [], [], []
        for seed in seeds:
            run = runs[(policy, None if policy == "full" else budget, seed)]
            ref = runs[(reference, None if reference == "full" else budget, seed)]
            for key in KEYS:
                if not np.array_equal(run[key], ref[key]):
                    raise SystemExit(f"{policy}/{budget}/s{seed}: frames differ")
            mask = conditions(ref)[condition]
            values.append(run[metric][mask])
            diffs.append((ref[metric] - run[metric])[mask])
            clusters.append(seed * 1_000_000 + ref["episode"][mask])
        values, diffs = np.concatenate(values), np.concatenate(diffs)
        clusters = np.concatenate(clusters)
        mean, low, high = bootstrap(values, clusters, resamples, rng)
        d, dlow, dhigh = bootstrap(diffs, clusters, resamples, rng)
        out[condition] = {
            "mean": mean,
            "low": low,
            "high": high,
            "diff": d,
            "diff_low": dlow,
            "diff_high": dhigh,
            "frames": int(len(values)),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=pathlib.Path, default="docs/a3-h64-jobs.tsv")
    parser.add_argument("--runs-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--metric", default="lpips")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    runs = load_runs(args.jobs, args.runs_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {"metric": args.metric, "budgets": {}}
    lines = [
        f"# A3 at H 64 — {args.metric.upper()} (lower is better), 3 seeds pooled",
        "",
    ]
    for budget, title in BUDGETS.items():
        block = {}
        for policy in [*POLICIES, "full"]:
            block[policy] = {
                "vs_window": pooled(
                    runs, policy, budget, "window", args.metric, args.resamples, rng
                ),
                "vs_decay": pooled(
                    runs,
                    policy,
                    budget,
                    "decay_continuous",
                    args.metric,
                    args.resamples,
                    rng,
                ),
            }
        summary["budgets"][budget] = block
        names = list(block["window"]["vs_window"])
        lines += [
            f"## {title}",
            "",
            "Mean, with the paired difference to "
            "`decay_continuous` (positive: better than decay; * 95% CI "
            "excludes 0).",
            "",
            "| Policy | " + " | ".join(names) + " |",
            "|---|" + "---|" * len(names),
        ]
        for policy in [*POLICIES, "full"]:
            cells = []
            for name in names:
                row = block[policy]["vs_decay"][name]
                if policy == "decay_continuous":
                    cells.append(f"**{row['mean']:.3f}**")
                    continue
                star = "*" if row["diff_low"] > 0 or row["diff_high"] < 0 else ""
                cells.append(f"{row['mean']:.3f} ({row['diff']:+.3f}{star})")
            lines.append(f"| `{policy}` | " + " | ".join(cells) + " |")
        lines.append("")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")

    revisits = [n for n in names if n != "novel"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4), dpi=150, sharey=True)
    x = np.arange(len(revisits))
    for axis, (budget, title) in zip(axes, BUDGETS.items()):
        block = summary["budgets"][budget]
        for policy in POLICIES:
            rows = [block[policy]["vs_window"][n] for n in revisits]
            axis.plot(
                x,
                [r["mean"] for r in rows],
                color=COLORS[policy],
                linewidth=2,
                marker="o",
                markersize=5,
                label=policy,
            )
            axis.fill_between(
                x,
                [r["low"] for r in rows],
                [r["high"] for r in rows],
                color=COLORS[policy],
                alpha=0.12,
                linewidth=0,
            )
        rows = [block["full"]["vs_window"][n] for n in revisits]
        axis.plot(
            x,
            [r["mean"] for r in rows],
            color=INK,
            linewidth=1.5,
            linestyle="--",
            label="full (oracle)",
        )
        axis.axvline(3.5, color=MUTED, linewidth=1, linestyle=":")
        axis.set_xticks(x, revisits, fontsize=8)
        axis.set_title(title, loc="left", color=INK, fontsize=10)
        axis.set_xlabel("revisit gap (frames since last seen)", color=MUTED, fontsize=9)
        axis.grid(True, color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
        axis.tick_params(colors=MUTED, labelsize=8)
    axes[0].set_ylabel(
        f"{args.metric.upper()} on revisits (lower is better)", color=MUTED, fontsize=9
    )
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    figure.suptitle(
        "Revisit error by gap at H 64, test split, 3 seeds; dotted line: " "horizon",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(args.output_dir / "a3_h64.png")
    print((args.output_dir / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
