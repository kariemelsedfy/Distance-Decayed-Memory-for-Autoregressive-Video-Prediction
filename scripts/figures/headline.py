#!/usr/bin/env python3
"""Headline figure: revisit error vs revisit gap, one curve per policy.

Reads ``summary.json`` from ``scripts/eval/summarize.py``. For one budget it
draws each policy's mean with its 95% bootstrap band, the ``full`` oracle as a
dashed reference, and the ``window`` policy's novel-view error as a flat
reference. Writes a PNG and the plotted numbers as CSV. Budgets are labelled
as policy budget + the shared 8-frame window (D-011).
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical order (dataviz reference palette, light mode).
COLORS = {
    "window": "#2a78d6",
    "window_sink": "#eb6834",
    "uniform_subsample": "#1baf7a",
    "relic_discrete": "#eda100",
    "decay_content": "#e87ba4",
    "decay_continuous": "#008300",
}
ORACLE = "#1a1a19"
LOCAL_WINDOW_TOKENS = 2_048


def bucket_order(condition: str) -> int:
    return int(condition.strip("[)").split(",")[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=pathlib.Path)
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--metric", default=None)
    parser.add_argument(
        "--output", type=pathlib.Path, default=pathlib.Path("outputs/figures/headline")
    )
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    metric = args.metric or data["metric"]
    summary = data["summary"]
    lines = {
        label: values[metric]
        for label, values in summary.items()
        if metric in values
        and (label.endswith(f"@{args.budget}") or label.startswith("full"))
    }
    if not lines:
        raise SystemExit(f"no runs at budget {args.budget} with metric {metric}")
    buckets = sorted(
        {c for v in lines.values() for c in v if c != "novel"}, key=bucket_order
    )
    x = list(range(len(buckets)))

    ink, muted, grid, surface = "#1a1a19", "#6b6a64", "#e6e5df", "#fcfcfb"
    figure, axis = plt.subplots(figsize=(8, 4.6), facecolor=surface)
    axis.set_facecolor(surface)
    rows = []
    for label, values in sorted(lines.items()):
        policy = label.split("@")[0]
        points = [
            (i, values[b]) for i, b in zip(x, buckets, strict=True) if b in values
        ]
        if not points:
            continue
        xs = [p[0] for p in points]
        mean = [p[1]["mean"] for p in points]
        low = [p[1]["low"] for p in points]
        high = [p[1]["high"] for p in points]
        oracle = policy == "full"
        color = ORACLE if oracle else COLORS.get(policy, muted)
        axis.plot(
            xs,
            mean,
            color=color,
            lw=2,
            ls="--" if oracle else "-",
            marker="o",
            ms=4,
            label="full (oracle)" if oracle else policy,
        )
        axis.fill_between(xs, low, high, color=color, alpha=0.12, lw=0)
        axis.annotate(
            label.split("@")[0],
            (xs[-1], mean[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8,
            color=ink,
            va="center",
        )
        for i, m, lo, hi in zip(xs, mean, low, high, strict=True):
            rows.append([label, buckets[i], m, lo, hi])
    window = summary.get(f"window@{args.budget}", {}).get(metric, {})
    if "novel" in window:
        axis.axhline(window["novel"]["mean"], color=muted, lw=1, ls=":")
        axis.annotate(
            "novel views",
            (x[0], window["novel"]["mean"]),
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=8,
            color=muted,
        )
    axis.set_xticks(x, buckets)
    axis.set_xlim(-0.3, len(buckets) - 0.3 + 1.2)
    axis.set_xlabel("revisit gap (frames since the view was last seen)", color=muted)
    axis.set_ylabel(
        f"{metric.upper()} on revisit frames"
        + (" (lower is better)" if metric == "lpips" else ""),
        color=muted,
    )
    axis.set_title(
        f"Revisit error by gap at {args.budget:,} + {LOCAL_WINDOW_TOKENS:,} "
        "shared tokens (95% CI)",
        color=ink,
        loc="left",
        fontsize=11,
    )
    axis.grid(True, color=grid, lw=0.8)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(grid)
    axis.tick_params(colors=muted, labelsize=8)
    axis.legend(frameon=False, fontsize=8, labelcolor=ink, loc="best")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(f"{args.output}_{args.budget}.png", dpi=160)
    with open(f"{args.output}_{args.budget}.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "bucket", "mean", "low", "high"])
        writer.writerows(rows)
    print(f"wrote {args.output}_{args.budget}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
