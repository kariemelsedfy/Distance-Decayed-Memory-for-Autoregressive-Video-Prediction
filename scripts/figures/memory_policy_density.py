#!/usr/bin/env python3
"""Gate 1 figure: cached tokens per frame against distance, every policy, one budget.

Each budgeted policy streams random keys for ``3 × horizon`` frames in 4-frame
chunks, compacting after every chunk; the curve is the density averaged over
the last 64 chunks (steady state). Densities are averaged within log-spaced
distance bins (six per octave) so policies that keep every k-th frame show
their allocation rather than a comb. Writes a PNG and the per-frame numbers
as CSV. The sink frames of ``window_sink`` sit at the start of the episode,
beyond the plotted distances.
"""

from __future__ import annotations

import argparse
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from distance_decayed_memory.memory import Geometry, make_policy  # noqa: E402

# Validated categorical order (dataviz reference palette, light mode, slots 1-6).
SERIES = [
    ("window", "#2a78d6"),
    ("window_sink", "#eb6834"),
    ("uniform_subsample", "#1baf7a"),
    ("relic_discrete", "#eda100"),
    ("decay_content", "#e87ba4"),
    ("decay_continuous", "#008300"),
]
BUDGETS = {"B-low": 2048, "B-mid": 4096, "B-high": 12288}


def steady_density(name: str, budget: int, horizon: int, chunk: int = 4) -> np.ndarray:
    geometry = Geometry()
    policy = make_policy(name, budget_tokens=budget, horizon=horizon, geometry=geometry)
    generator = torch.Generator().manual_seed(0)
    total, average, samples = 3 * horizon, np.zeros(horizon), 0
    for t in range(0, total, chunk):
        k = torch.randn(1, 1, chunk * geometry.tokens_per_frame, 6, generator=generator)
        policy.append(k, k.clone(), None, t)
        policy.compact()
        if t >= total - 64 * chunk:
            average += policy.density(horizon).numpy()
            samples += 1
    return average / samples


def log_bins(values: np.ndarray, per_octave: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Mean of ``values[d]`` over geometric bins of distance ``d + 1``."""
    edges = np.unique(
        np.round(
            2 ** np.arange(0, np.log2(len(values)) + 1 / per_octave, 1 / per_octave)
        ).astype(int)
    )
    edges = np.append(edges[edges < len(values)], len(values))
    means = [values[a - 1 : b - 1].mean() for a, b in zip(edges[:-1], edges[1:])]
    return edges, np.asarray(means + [means[-1]])


def spread_labels(positions: list[float], gap: float) -> list[float]:
    """Nudge sorted label positions apart by at least ``gap`` (log2 units)."""
    order = np.argsort(positions)
    placed = list(positions)
    for previous, current in zip(order[:-1], order[1:]):
        placed[current] = max(placed[current], placed[previous] + gap)
    return placed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", choices=sorted(BUDGETS), default="B-mid")
    parser.add_argument("--horizon", type=int, default=2048)
    parser.add_argument(
        "--output-dir", type=pathlib.Path, default=pathlib.Path("outputs/figures")
    )
    args = parser.parse_args()
    budget = BUDGETS[args.budget]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"memory_policy_density_{args.budget}"

    distance = np.arange(args.horizon)
    curves = {name: steady_density(name, budget, args.horizon) for name, _ in SERIES}
    with open(f"{stem}.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["distance_frames", *curves])
        writer.writerows(zip(distance, *curves.values(), strict=True))

    ink, muted, grid, surface = "#1a1a19", "#6b6a64", "#e6e5df", "#fcfcfb"
    figure, axis = plt.subplots(figsize=(8, 4.6), facecolor=surface)
    axis.set_facecolor(surface)
    ends = []
    for name, color in SERIES:
        edges, means = log_bins(curves[name])
        values = np.where(means > 0, means, np.nan)
        axis.step(edges, values, where="post", color=color, lw=2, label=name)
        last = np.flatnonzero(~np.isnan(values[:-1]))[-1]
        ends.append((name, edges[last + 1], values[last]))
    heights = spread_labels([float(np.log2(v)) for _, _, v in ends], gap=0.45)
    for (name, x, _), height in zip(ends, heights, strict=True):
        axis.annotate(
            name,
            (x, 2**height),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8,
            color=ink,
            va="center",
        )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log", base=2)
    axis.set_xlim(1, args.horizon * 2.5)
    axis.set_xlabel("distance from the current frame (frames, log scale)", color=muted)
    axis.set_ylabel("cached tokens per frame (log scale)", color=muted)
    axis.set_title(
        f"Where each policy spends {budget:,} tokens over a {args.horizon:,}-frame "
        "horizon",
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
    axis.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=ink)
    figure.tight_layout()
    figure.savefig(f"{stem}.png", dpi=160)
    print(f"wrote {stem}.png and {stem}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
