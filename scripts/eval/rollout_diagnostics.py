#!/usr/bin/env python3
"""Drift, collapse, and action-following diagnostics for a base-model checkpoint.

    python scripts/eval/rollout_diagnostics.py --checkpoint runs/a1-.../latest.pt \
        --split-dir data/memmaze9/val --output-dir runs/a1-diagnostics-...

Part A rolls out ``--clips`` validation clips from ``--context`` true frames
with their true actions, once per seed, and scores every generated frame
against the truth. Part B continues the first ``--action-clips`` contexts with
one fixed action (noop, forward, left, right) and measures how the generated
view turns, against the turning of true frames under the same actions. All
rollouts keep the full cache (no memory policy), as in A1's sample GIFs.
Writes ``diagnostics.json``, ``diagnostics.npz``, and contact sheets. The
frozen test split is refused (D-006).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

from distance_decayed_memory.data.navigation import ACTION_NAMES
from distance_decayed_memory.data.shards import SplitReader
from distance_decayed_memory.eval import diagnostics as dx
from distance_decayed_memory.memory import Geometry, make_policy
from distance_decayed_memory.models.dit import DiTConfig, PixelDiT
from distance_decayed_memory.models.flow import rollout, to_model_range, to_uint8

OVERRIDES = (None, 0, 1, 2, 3)  # true actions, noop, forward, left, right


def parse_args() -> argparse.Namespace:
    argv = sys.argv[1:]
    if argv[:1] == ["--args-file"]:
        argv = json.loads(pathlib.Path(argv[1]).read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--split-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--clips", type=int, default=64)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--action-clips", type=int, default=24)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--tail", type=int, default=16, help="frames for 'stuck'")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--sheets", type=int, default=8)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--clip-seed", type=int, default=0)
    return parser.parse_args(argv)


def load_clips(reader: SplitReader, args, no_action: int, chunk: int):
    """Evenly spaced episodes, one chunk-aligned random start each."""
    rng = np.random.default_rng(args.clip_seed)
    count = min(args.clips, len(reader))
    episodes = np.linspace(0, len(reader) - 1, count).round().astype(int)
    frames, prev, where = [], [], []
    for episode in episodes:
        data = reader.episode(int(episode), ("frames", "actions"))
        length = len(data["actions"])
        start = int(rng.integers(0, (length - args.frames) // chunk + 1)) * chunk
        actions = np.asarray(data["actions"], dtype=np.int64)
        lead = actions[start - 1] if start else no_action
        frames.append(np.asarray(data["frames"][start : start + args.frames]))
        prev.append(np.concatenate([[lead], actions[start : start + args.frames - 1]]))
        where.append((int(episode), start))
    return np.stack(frames), np.stack(prev), np.asarray(where)


@torch.no_grad()
def generate(model, frames, prev, args, seed, device, autocast) -> np.ndarray:
    """Rollouts ``[N, T, H, W, 3]`` uint8 from the first ``context`` true frames."""
    geometry = Geometry(model.config.grid)
    out = []
    for first in range(0, len(frames), args.batch):
        block = slice(first, first + args.batch)
        context = to_model_range(
            torch.as_tensor(frames[block, : args.context], device=device)
        )
        actions = torch.as_tensor(prev[block], device=device)
        generator = torch.Generator(device=device).manual_seed(seed * 100_003 + first)
        with autocast:
            video = rollout(
                model,
                lambda: make_policy("full", geometry=geometry),
                context,
                actions,
                args.frames,
                steps=args.steps,
                generator=generator,
            )
        out.append(to_uint8(video.float()).cpu().numpy())
    return np.concatenate(out)


def sheet(rows: list[tuple[str, np.ndarray]], columns: list[int], path) -> None:
    """One labelled row per video, frames at ``columns`` (1-based labels)."""
    size, label = rows[0][1].shape[1], 74
    image = Image.new(
        "RGB", (label + len(columns) * (size + 2), 14 + len(rows) * (size + 2)), "white"
    )
    draw = ImageDraw.Draw(image)
    for j, t in enumerate(columns):
        draw.text((label + j * (size + 2) + 2, 1), str(t + 1), fill="black")
    for i, (name, video) in enumerate(rows):
        y = 14 + i * (size + 2)
        draw.text((2, y + size // 2 - 5), name, fill="black")
        for j, t in enumerate(columns):
            image.paste(Image.fromarray(video[t]), (label + j * (size + 2), y))
    image.save(path)


def main() -> int:
    args = parse_args()
    manifest = json.loads((args.split_dir / "manifest.json").read_text())
    if manifest.get("split") == "test":
        raise SystemExit("The test split is frozen (D-006).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PixelDiT(DiTConfig(**state["model_config"]))
    model.load_state_dict(state[args.weights])
    model.to(device).eval()
    config = model.config
    if args.context % config.chunk_frames or args.frames % config.chunk_frames:
        raise SystemExit("--context and --frames must be multiples of the chunk.")
    autocast = torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=args.precision == "bf16"
    )
    started = time.perf_counter()
    out = args.output_dir
    (out / "sheets").mkdir(parents=True, exist_ok=True)

    truth, prev, where = load_clips(
        SplitReader(args.split_dir), args, config.no_action, config.chunk_frames
    )
    c, tail = args.context, args.tail

    # Part A: drift and collapse with true actions.
    rollouts = np.stack(
        [
            generate(model, truth, prev, args, seed, device, autocast)
            for seed in range(args.seeds)
        ]
    )  # [S, N, T, H, W, 3]
    psnr = dx.psnr_np(rollouts, truth[None])  # [S, N, T]
    seed_psnr = (
        dx.psnr_np(rollouts[0], rollouts[1]) if args.seeds > 1 else None
    )  # [N, T]
    stuck = [
        [dx.stuck(rollouts[s, n, c:], truth[n, c:], tail) for n in range(len(truth))]
        for s in range(args.seeds)
    ]
    flags = np.array([[f for f, _, _ in row] for row in stuck], dtype=object)
    evaluable = flags != None  # noqa: E711 (object array comparison)

    # Part B: action following.
    m = min(args.action_clips, len(truth))
    true_shifts = dx.shifts(truth)  # [N, T]
    calibration = dx.shift_by_action(true_shifts[:, 1:], prev[:, 1:], config.actions)
    action_rollouts, action_shifts = {}, {}
    for action in OVERRIDES:
        name = "true" if action is None else ACTION_NAMES[action]
        conditioned = dx.override_actions(prev[:m], c, action)
        video = (
            rollouts[0, :m]
            if action is None
            else generate(model, truth[:m], conditioned, args, 0, device, autocast)
        )
        action_rollouts[name] = video
        action_shifts[name] = dx.shifts(video)[:, c:]  # [M, T - c]
    generated_shift = {
        name: {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
            "frames": int(np.isfinite(values).sum()),
        }
        for name, values in action_shifts.items()
    }
    left_right_psnr = dx.psnr_np(action_rollouts["left"], action_rollouts["right"])

    # Summaries.
    def by_offset(values: np.ndarray) -> dict[str, list[float]]:
        flat = values[..., c:].reshape(-1, args.frames - c)
        return {
            "median": np.median(flat, axis=0).round(3).tolist(),
            "q25": np.percentile(flat, 25, axis=0).round(3).tolist(),
            "q75": np.percentile(flat, 75, axis=0).round(3).tolist(),
        }

    psnr_curve = by_offset(psnr)
    below = [i + 1 for i, v in enumerate(psnr_curve["median"]) if v < 15.0]
    stuck_rate = float(flags[evaluable].astype(bool).mean()) if evaluable.any() else 0
    summary = {
        "checkpoint": str(args.checkpoint),
        "weights": args.weights,
        "split": manifest.get("split"),
        "clips": [{"episode": int(e), "start": int(s)} for e, s in where],
        "settings": {k: str(v) for k, v in vars(args).items()},
        "psnr_by_generated_frame": psnr_curve,
        "first_generated_frame_median_below_15db": below[0] if below else None,
        "seed_to_seed_psnr_by_generated_frame": (
            by_offset(seed_psnr) if seed_psnr is not None else None
        ),
        "left_vs_right_psnr_by_generated_frame": by_offset(left_right_psnr),
        "stuck": {
            "rule": (
                f"last {tail} frames: generated motion < {dx.STUCK_MOTION} x true and "
                f"detail < {dx.STUCK_DETAIL} x true; true motion >= {dx.MIN_MOTION}"
            ),
            "evaluable_rollouts": int(evaluable.sum()),
            "stuck_rollouts": int(flags[evaluable].astype(bool).sum()),
            "rate": stuck_rate,
            "median_motion_ratio": float(
                np.median([r for row in stuck for _, r, _ in row])
            ),
            "median_detail_ratio": float(
                np.median([d for row in stuck for _, _, d in row])
            ),
        },
        "shift_true_frames_by_action": {
            ACTION_NAMES[a]: {"mean": v[0], "std": v[1], "frames": v[2]}
            for a, v in calibration.items()
        },
        "shift_generated_by_override": generated_shift,
        "seconds": time.perf_counter() - started,
        "git_sha": os.environ.get("DD_MEMORY_GIT_SHA", "unknown"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (out / "diagnostics.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        out / "diagnostics.npz",
        psnr=psnr,
        seed_psnr=seed_psnr if seed_psnr is not None else np.zeros(0),
        left_right_psnr=left_right_psnr,
        motion_ratio=np.array([[r for _, r, _ in row] for row in stuck]),
        detail_ratio=np.array([[d for _, _, d in row] for row in stuck]),
        stuck=np.where(evaluable, flags, -1).astype(np.int8),
        true_shifts=true_shifts,
        prev=prev,
        where=where,
        **{f"shift_{name}": values for name, values in action_shifts.items()},
    )

    # Contact sheets: typical and worst rollouts, and the action sweep.
    columns = list(range(c - 1, args.frames, 4))
    order = np.argsort(psnr[0, :, c:].mean(axis=1))
    picks = {"worst": order[: args.sheets // 2], "median": order[len(order) // 2 :]}
    picks["median"] = picks["median"][: args.sheets - len(picks["worst"])]
    for kind, indices in picks.items():
        for n in indices:
            rows = [("truth", truth[n])] + [
                (f"seed {s}", rollouts[s, n]) for s in range(args.seeds)
            ]
            sheet(rows, columns, out / "sheets" / f"{kind}-clip{n:03d}.png")
    for n in range(min(args.sheets, m)):
        rows = [("truth", truth[n])] + list(
            (name, video[n]) for name, video in action_rollouts.items()
        )
        sheet(rows, columns, out / "sheets" / f"actions-clip{n:03d}.png")
    print(json.dumps({k: summary[k] for k in ("stuck", "seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
