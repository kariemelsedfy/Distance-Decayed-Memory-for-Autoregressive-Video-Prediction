#!/usr/bin/env python3
"""Training curves of one run: losses (log scale) and one-clip sample PSNR.

Reads ``metrics.jsonl`` and ``validation.jsonl`` from a fetched run directory
(``scripts/hpc/fetch.sh``) and writes ``training_curves.png`` next to them. The
training loss is a moving average over ``--smooth`` logged steps; validation
losses are plotted for both the raw and the EMA weights.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Validated categorical order (dataviz reference palette, light mode, slots 1-3).
TRAIN, VAL, VAL_EMA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f2328", "#59636e", "#e6e8eb"


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    picked = [(row["step"], row[key]) for row in rows if key in row]
    steps, values = zip(*picked) if picked else ((), ())
    return np.asarray(steps), np.asarray(values, dtype=float)


def style(axis: plt.Axes) -> None:
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.xaxis.set_major_formatter(lambda x, _: f"{x / 1000:.0f}k")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument(
        "--smooth", type=int, default=20, help="logged steps per average"
    )
    args = parser.parse_args()

    metrics = read_jsonl(args.run_dir / "metrics.jsonl")
    validation = read_jsonl(args.run_dir / "validation.jsonl")
    train_steps, train_loss = series(metrics, "loss")
    window = max(1, min(args.smooth, len(train_loss)))
    smoothed = np.convolve(train_loss, np.ones(window) / window, mode="valid")
    val_steps, val_loss = series(validation, "val_loss")
    ema_steps, ema_loss = series(validation, "val_loss_ema")
    psnr_steps, psnr = series(validation, "sample_psnr_mean")

    figure, (loss_axis, psnr_axis) = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    lines = [
        (train_steps[window - 1 :], smoothed, TRAIN, "train (smoothed)"),
        (val_steps, val_loss, VAL, "val"),
        (ema_steps, ema_loss, VAL_EMA, "val, EMA weights"),
    ]
    for steps, values, color, label in lines:
        loss_axis.plot(steps, values, color=color, linewidth=2, label=label)
    loss_axis.set_yscale("log")
    loss_axis.set_title("Flow-matching loss", loc="left", color=INK, fontsize=11)
    loss_axis.legend(frameon=False, fontsize=9, labelcolor=INK)
    style(loss_axis)

    psnr_axis.plot(psnr_steps, psnr, color=TRAIN, linewidth=2, marker="o", markersize=5)
    psnr_axis.set_title(
        "Sample PSNR (dB), one val clip, EMA weights",
        loc="left",
        color=INK,
        fontsize=11,
    )
    if len(psnr):
        psnr_axis.annotate(
            f"{psnr[-1]:.1f}",
            (psnr_steps[-1], psnr[-1]),
            textcoords="offset points",
            xytext=(6, 0),
            va="center",
            color=INK,
            fontsize=9,
        )
    style(psnr_axis)
    for axis in (loss_axis, psnr_axis):
        axis.set_xlabel("step", color=MUTED, fontsize=9)

    figure.tight_layout()
    output = args.run_dir / "training_curves.png"
    figure.savefig(output)
    print(output)


if __name__ == "__main__":
    main()
