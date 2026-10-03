#!/usr/bin/env python3
"""Figure for ``scripts/eval/rollout_diagnostics.py``: drift and action following.

Left: PSNR of generated frames against the truth by frame since the context
(median and interquartile band), with seed-to-seed and left-vs-right PSNR for
reference. Right: mean horizontal image shift per frame for each action, in
true frames (calibration) and in rollouts driven by that action. Writes
``rollout_diagnostics.png`` next to ``diagnostics.json``.
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
FIRST, SECOND, THIRD = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f2328", "#59636e", "#e6e8eb"
ACTIONS = ("noop", "forward", "left", "right")


def style(axis: plt.Axes) -> None:
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors=MUTED, labelsize=9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path)
    args = parser.parse_args()
    summary = json.loads((args.run_dir / "diagnostics.json").read_text())

    figure, (drift, turning) = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    curve = summary["psnr_by_generated_frame"]
    offsets = np.arange(1, len(curve["median"]) + 1)
    drift.fill_between(
        offsets, curve["q25"], curve["q75"], color=FIRST, alpha=0.18, linewidth=0
    )
    drift.plot(offsets, curve["median"], color=FIRST, linewidth=2, label="vs truth")
    references = (
        ("seed_to_seed_psnr_by_generated_frame", SECOND, "seed 0 vs seed 1"),
        ("left_vs_right_psnr_by_generated_frame", THIRD, "all-left vs all-right"),
    )
    for key, color, label in references:
        if summary.get(key):
            drift.plot(
                offsets, summary[key]["median"], color=color, linewidth=2, label=label
            )
    drift.set_title(
        "PSNR (dB) of generated frames, median and IQR",
        loc="left",
        color=INK,
        fontsize=11,
    )
    drift.set_xlabel("generated frame (after the context)", color=MUTED, fontsize=9)
    drift.legend(frameon=False, fontsize=9, labelcolor=INK)
    style(drift)

    true_shift = summary["shift_true_frames_by_action"]
    generated = summary["shift_generated_by_override"]
    rows = np.arange(len(ACTIONS))
    for offset, values, color, label in (
        (-0.12, [true_shift.get(a, {}).get("mean", np.nan) for a in ACTIONS], FIRST,
         "true frames"),
        (0.12, [generated.get(a, {}).get("mean", np.nan) for a in ACTIONS], SECOND,
         "rollout driven by the action"),
    ):  # fmt: skip
        turning.scatter(values, rows + offset, s=64, color=color, label=label, zorder=3)
    turning.axvline(0, color=MUTED, linewidth=1)
    turning.set_yticks(rows, ACTIONS)
    turning.invert_yaxis()
    turning.set_title(
        "Mean horizontal shift per frame (px)", loc="left", color=INK, fontsize=11
    )
    turning.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    style(turning)

    figure.tight_layout()
    output = args.run_dir / "rollout_diagnostics.png"
    figure.savefig(output)
    print(output)


if __name__ == "__main__":
    main()
